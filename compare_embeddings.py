"""Compare frozen text representations without changing the clustering baseline."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from clustering_experiment import ROOT, load_model, read_json, write_csv, write_json

VARIANTS = ("description", "name_description", "reviewed_readme")
VALID_LABELS = {"yes", "no", "uncertain", ""}
STRONGER_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_VARIANTS = ("minilm", "mpnet")


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def review_excerpts(rows):
    approved = {}
    for row in rows:
        if row["usable"] != "yes" or row["review"] in {"pending", ""}:
            continue
        excerpt = row.get("reviewed_excerpt") or row["excerpt"]
        if not excerpt or excerpt not in row["excerpt"]:
            raise ValueError(f"Reviewed excerpt is not a verbatim source span: {row['id']}")
        approved[row["id"]] = excerpt
    return approved


def summarize(rows, judgments):
    """Keep uncertain and unjudged pairs visible instead of counting them as no."""
    labels = [judgments.get((r["anchor_id"], r["neighbor_id"]), {}).get("related_functionality", "") for r in rows]
    if set(labels) - VALID_LABELS:
        raise ValueError("Judgments must be yes, no, uncertain, or blank")
    counts = Counter(labels)
    total = len(rows)
    yes, no = counts["yes"], counts["no"]
    return {"slots": total, "yes": yes, "no": no, "uncertain": counts["uncertain"],
            "unjudged": counts[""],
            "decisive_relevance": yes / (yes + no) if yes + no else None,
            "precision_lower": yes / total if total else None,
            "precision_upper": (total - no) / total if total else None}


def prepare(output):
    assignments = read_json(output / "assignments.json")
    baseline = read_json(output / "baseline.json")
    reviews = load_csv(output / "readme_review.csv")
    approved = review_excerpts(reviews)
    ids = [p["id"] for p in assignments]
    index = {identity: i for i, identity in enumerate(ids)}
    # The input remains description-only for the baseline; metadata corrections
    # and the 40 missing descriptions are intentionally a separate experiment.
    descriptions = [p["description"] for p in assignments]
    named = [f"{p['name']}: {p['description']}" for p in assignments]
    enriched = [text + "\n" + approved[p["id"]] if p["id"] in approved else text
                for text, p in zip(named, assignments)]
    texts = dict(zip(VARIANTS, [descriptions, named, enriched]))
    signature = hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()
    archive_path = output / "comparison_vectors.npz"
    manifest_path = output / "comparison_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if (manifest.get("text_sha256") == signature
            and manifest.get("model_revision") == baseline["model_revision"]
            and manifest.get("baseline_packages_sha256") == baseline["packages_sha256"]
            and archive_path.exists()):
        archive = np.load(archive_path)
        matrices = {variant: archive[variant] for variant in VARIANTS}
    else:
        model, revision = load_model(baseline["model_revision"])
        if revision != baseline["model_revision"]:
            raise ValueError("Embedding model revision differs from the baseline")
        matrices = {"description": np.load(output / "baseline_vectors.npz")["embeddings"]}
        matrices["name_description"] = model.encode(named, normalize_embeddings=True)
        matrices["reviewed_readme"] = matrices["name_description"].copy()
        changed = [i for i, p in enumerate(assignments) if p["id"] in approved]
        for i in changed:
            token_count = len(model.tokenizer.encode(enriched[i], add_special_tokens=True))
            if token_count > model.max_seq_length:
                raise ValueError(f"Enriched input exceeds model token limit: {ids[i]}")
        if changed:
            matrices["reviewed_readme"][changed] = model.encode(
                [enriched[i] for i in changed], normalize_embeddings=True
            )
        np.savez_compressed(archive_path, **matrices)
        write_json(manifest_path, {"text_sha256": signature, "model_revision": revision,
                   "baseline_packages_sha256": baseline["packages_sha256"],
                   "candidate_count": len(ids), "readme_enriched_count": len(changed),
                   "fallback": "name_description", "evaluation_split": "development",
                   "user_judgments_sha256": hashlib.sha256((output / "neighbor_judgments.csv").read_bytes()).hexdigest()})
    write_json(output / "comparison_inputs.json", [
        {"id": p["id"], **{variant: texts[variant][i] for variant in VARIANTS}}
        for i, p in enumerate(assignments)
    ])
    user_rows = [row for row in load_csv(output / "neighbor_judgments.csv") if row["split"] == "development"]
    user = {(row["anchor_id"], row["neighbor_id"]): row for row in user_rows}
    anchors = list(dict.fromkeys(row["anchor_id"] for row in user_rows))
    neighbors = []
    for variant in VARIANTS:
        distances = cosine_distances(matrices[variant])
        for anchor in anchors:
            i = index[anchor]
            nearest = [int(j) for j in np.argsort(distances[i], kind="stable") if j != i][:5]
            for rank, j in enumerate(nearest, start=1):
                neighbors.append({"variant": variant, "anchor_id": anchor,
                                  "neighbor_id": ids[j], "rank": rank,
                                  "cosine_distance": float(distances[i, j]),
                                  "anchor_enriched": anchor in approved,
                                  "neighbor_enriched": ids[j] in approved})
    write_json(output / "comparison_neighbors.json", neighbors)
    path = output / "comparison_judgments.csv"
    previous = {(r["anchor_id"], r["neighbor_id"]): r for r in load_csv(path)} if path.exists() else {}
    union = sorted({(r["anchor_id"], r["neighbor_id"]) for r in neighbors})
    rows = []
    for anchor, neighbor in union:
        source = user.get((anchor, neighbor))
        old = previous.get((anchor, neighbor), {})
        rows.append({"anchor_id": anchor, "anchor_description": descriptions[index[anchor]],
                     "neighbor_id": neighbor, "neighbor_description": descriptions[index[neighbor]],
                     "related_functionality": source["related_functionality"] if source else old.get("related_functionality", ""),
                     "reviewer": "user" if source else old.get("reviewer", ""),
                     "notes": source["notes"] if source else old.get("notes", "")})
    write_csv(path, rows, list(rows[0]))
    print(f"Compared {len(ids)} candidates at {len(anchors)} development anchors; {len(approved)} enriched packages.")
    print(f"Union review: {len(rows)} pairs; {sum(not r['related_functionality'] for r in rows)} need judgments.")
    evaluate(output)


def evaluate(output):
    neighbors = read_json(output / "comparison_neighbors.json")
    reviews = load_csv(output / "comparison_judgments.csv")
    judgments = {(r["anchor_id"], r["neighbor_id"]): r for r in reviews}
    user = {key: row for key, row in judgments.items() if row["reviewer"] == "user"}
    metrics = {}
    for variant in VARIANTS:
        rows = [r for r in neighbors if r["variant"] == variant]
        metrics[variant] = {
            "all": summarize(rows, judgments), "user_only": summarize(rows, user),
            "enriched_anchors": summarize([r for r in rows if r["anchor_enriched"]], judgments),
            "fallback_anchors": summarize([r for r in rows if not r["anchor_enriched"]], judgments),
        }
    write_json(output / "comparison_metrics.json", metrics)
    write_json(output / "comparison_evaluation.json", {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "user_judgments_sha256": hashlib.sha256((output / "neighbor_judgments.csv").read_bytes()).hexdigest(),
        "comparison_judgments_sha256": hashlib.sha256((output / "comparison_judgments.csv").read_bytes()).hexdigest(),
        "readme_review_sha256": hashlib.sha256((output / "readme_review.csv").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewer_counts": dict(Counter(r["reviewer"] for r in reviews)),
    })
    lines = ["# Text representation comparison", "",
             "Same 460 candidate packages, pinned MiniLM model, and 15 development anchors. "
             "Only reviewed pilot packages receive README text; others use name plus description. "
             "No holdout labels are used. User judgments are preserved, including uncertainty.", "",
             "| Representation | Relevant | Irrelevant | Uncertain | Unjudged | Relevant among decisive | Top-5 relevance bounds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        score = metrics[variant]["all"]
        decisive = f"{score['decisive_relevance']:.1%}" if score["decisive_relevance"] is not None else "n/a"
        lines.append(f"| {variant} | {score['yes']} | {score['no']} | {score['uncertain']} | {score['unjudged']} | {decisive} | {score['precision_lower']:.1%}–{score['precision_upper']:.1%} |")
    lines += ["", "Bounds treat uncertain and unjudged pairs as irrelevant at the lower end and relevant at the upper end. "
              "Decisive rates exclude those pairs and must be read with coverage. Assistant-reviewed new pairs are provisional, "
              "not independent user validation. The user_only metrics in comparison_metrics.json expose that distinction.", "",
              "## By anchor", "", "| Anchor | Description top 5 | Name + description top 5 | Reviewed README top 5 |",
              "| --- | --- | --- | --- |"]
    anchors = list(dict.fromkeys(r["anchor_id"] for r in neighbors))
    for anchor in anchors:
        cells = []
        for variant in VARIANTS:
            rows = sorted([r for r in neighbors if r["anchor_id"] == anchor and r["variant"] == variant], key=lambda r: r["rank"])
            cells.append("; ".join(r["neighbor_id"] + " [" + (judgments[(anchor, r["neighbor_id"])]["related_functionality"] or "unjudged") + "]" for r in rows))
        lines.append(f"| {anchor} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation limits", "",
              "The 15 development anchors were selected for variety, not as a representative random sample. "
              "Forty-six enriched candidates are a small part of the corpus. New candidates introduced by variants "
              "are mostly assistant-reviewed; original pairs were user-reviewed. Compare concrete examples and "
              "enriched-anchor metrics before adopting a change. Wider retrieval or another model needs a separate decision."]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def prepare_model(output):
    inputs = read_json(output / "comparison_inputs.json")
    assignments = read_json(output / "assignments.json")
    if [row["id"] for row in inputs] != [row["id"] for row in assignments]:
        raise ValueError("Text inputs and frozen package identities are misaligned")
    texts = [row["name_description"] for row in inputs]
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    path = output / "model_manifest.json"
    manifest = read_json(path) if path.exists() else {}
    archive_path = output / "model_vectors.npz"
    baseline_manifest = read_json(output / "comparison_manifest.json")
    if (manifest.get("input_sha256") != signature or not archive_path.exists()
            or manifest.get("model") != STRONGER_MODEL
            or manifest.get("minilm_revision") != baseline_manifest["model_revision"]):
        pinned_revision = manifest.get("model_revision") if manifest.get("model") == STRONGER_MODEL else None
        model, revision = load_model(revision=pinned_revision, model_name=STRONGER_MODEL)
        started = time.perf_counter()
        mpnet = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        elapsed = time.perf_counter() - started
        np.savez_compressed(archive_path, mpnet=mpnet)
        write_json(path, {"model": STRONGER_MODEL, "model_revision": revision,
                   "minilm_revision": baseline_manifest["model_revision"],
                   "input_sha256": signature, "package_count": len(texts),
                   "representation": "name_description", "dimensions": mpnet.shape[1],
                   "encode_seconds": elapsed, "device": str(model.device),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "evaluation_split": "development"})
    matrices = {"minilm": np.load(output / "comparison_vectors.npz")["name_description"],
                "mpnet": np.load(archive_path)["mpnet"]}
    user_rows = [r for r in load_csv(output / "neighbor_judgments.csv") if r["split"] == "development"]
    anchors = list(dict.fromkeys(r["anchor_id"] for r in user_rows))
    indices = {row["id"]: i for i, row in enumerate(inputs)}
    neighbors = []
    for variant, embeddings in matrices.items():
        if embeddings.shape[0] != len(inputs):
            raise ValueError("Model vectors do not match the frozen candidate set")
        distances = cosine_distances(embeddings)
        for anchor in anchors:
            i = indices[anchor]
            nearest = [int(j) for j in np.argsort(distances[i], kind="stable") if j != i][:5]
            for rank, j in enumerate(nearest, start=1):
                neighbors.append({"variant": variant, "anchor_id": anchor,
                                  "neighbor_id": inputs[j]["id"], "rank": rank,
                                  "cosine_distance": float(distances[i, j])})
    write_json(output / "model_neighbors.json", neighbors)
    previous_path = output / "model_judgments.csv"
    previous = {(r["anchor_id"], r["neighbor_id"]): r for r in load_csv(previous_path)} if previous_path.exists() else {}
    inherited = {(r["anchor_id"], r["neighbor_id"]): r for r in load_csv(output / "comparison_judgments.csv")}
    # The user's original labels remain authoritative, including uncertain labels.
    inherited.update({(r["anchor_id"], r["neighbor_id"]): {**r, "reviewer": "user"} for r in user_rows})
    union = sorted({(r["anchor_id"], r["neighbor_id"]) for r in neighbors})
    rows = []
    for anchor, neighbor in union:
        source = inherited.get((anchor, neighbor), previous.get((anchor, neighbor), {}))
        rows.append({"anchor_id": anchor, "anchor_description": assignments[indices[anchor]]["description"],
                     "neighbor_id": neighbor, "neighbor_description": assignments[indices[neighbor]]["description"],
                     "related_functionality": source.get("related_functionality", ""),
                     "reviewer": source.get("reviewer", ""), "notes": source.get("notes", "")})
    write_csv(previous_path, rows, list(rows[0]))
    inspect_model_clusters(output, matrices, assignments)
    print(f"Model comparison: {len(texts)} candidates, {len(anchors)} anchors; "
          f"{sum(not r['related_functionality'] for r in rows)} new pairs need review.")
    evaluate_model(output)


def inspect_model_clusters(output, matrices, assignments):
    import hdbscan

    settings = read_json(output / "baseline.json")
    results = {}
    for variant, embeddings in matrices.items():
        distances = cosine_distances(embeddings).astype("float64")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings["min_cluster_size"], min_samples=settings["min_samples"],
            metric="precomputed", cluster_selection_method="eom",
        ).fit(distances)
        clusters = []
        for label in sorted(set(clusterer.labels_) - {-1}):
            indices = np.flatnonzero(clusterer.labels_ == label)
            centroid = embeddings[indices].mean(axis=0, keepdims=True)
            order = indices[np.argsort(cosine_distances(embeddings[indices], centroid).ravel())]
            clusters.append({"cluster": int(label), "size": len(indices),
                             "representatives": [assignments[i]["id"] for i in order[:5]],
                             "members": [assignments[i]["id"] for i in indices]})
        results[variant] = {"cluster_count": len(clusters),
                            "unclustered_count": int(sum(clusterer.labels_ == -1)),
                            "min_cluster_size": settings["min_cluster_size"],
                            "min_samples": settings["min_samples"], "clusters": clusters}
    write_json(output / "model_clustering.json", results)


def evaluate_model(output):
    neighbors = read_json(output / "model_neighbors.json")
    reviews = load_csv(output / "model_judgments.csv")
    judgments = {(r["anchor_id"], r["neighbor_id"]): r for r in reviews}
    user = {key: row for key, row in judgments.items() if row["reviewer"] == "user"}
    metrics = {}
    for variant in MODEL_VARIANTS:
        rows = [r for r in neighbors if r["variant"] == variant]
        metrics[variant] = {"all": summarize(rows, judgments), "user_only": summarize(rows, user)}
    write_json(output / "model_metrics.json", metrics)
    write_json(output / "model_evaluation.json", {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "user_judgments_sha256": hashlib.sha256((output / "neighbor_judgments.csv").read_bytes()).hexdigest(),
        "model_judgments_sha256": hashlib.sha256((output / "model_judgments.csv").read_bytes()).hexdigest(),
        "reviewer_counts": dict(Counter(r["reviewer"] for r in reviews)),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    lines = ["# Embedding model comparison", "",
             "Same 460 packages, exact name-plus-description text, cosine similarity, "
             "and 15 development anchors. MiniLM vectors are reused unchanged. "
             "No README text, metadata corrections, or holdout labels enter this comparison.", "",
             "| Model | Relevant | Irrelevant | Uncertain | Unjudged | Relevant among decisive | Top-5 relevance bounds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in MODEL_VARIANTS:
        score = metrics[variant]["all"]
        decisive = f"{score['decisive_relevance']:.1%}" if score["decisive_relevance"] is not None else "n/a"
        lines.append(f"| {variant} | {score['yes']} | {score['no']} | {score['uncertain']} | {score['unjudged']} | {decisive} | {score['precision_lower']:.1%}–{score['precision_upper']:.1%} |")
    lines += ["", "User and assistant labels are recorded separately in model_judgments.csv. "
              "User-only scores include coverage in model_metrics.json. Bounds count uncertain or missing "
              "judgments as irrelevant at the lower end and relevant at the upper end. "
              "The decisive rate excludes those pairs. These bounds are not statistical confidence intervals.", "",
              "## By anchor", "", "| Anchor | MiniLM top 5 | MPNet top 5 |", "| --- | --- | --- |"]
    for anchor in dict.fromkeys(r["anchor_id"] for r in neighbors):
        cells = []
        for variant in MODEL_VARIANTS:
            rows = [r for r in neighbors if r["anchor_id"] == anchor and r["variant"] == variant]
            cells.append("; ".join(r["neighbor_id"] + " [" + (judgments[(anchor, r["neighbor_id"])]["related_functionality"] or "unjudged") + "]" for r in rows))
        lines.append(f"| {anchor} | " + " | ".join(cells) + " |")
    lines += ["", "## Limits", "", "A small, deliberately varied development set and assistant judgments "
              "support an exploratory choice, not independent validation. Versioned packages can occupy several "
              "neighbor slots. Keep fresh anchors for final validation and compare cluster membership separately.", "",
              "Model reference: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)."]
    if (output / "model_clustering.json").exists():
        clustering = read_json(output / "model_clustering.json")
        lines += ["", "## HDBSCAN at unchanged settings", "",
                  "Both models use name-plus-description, min_cluster_size=5, min_samples=2, and cosine distance. "
                  "Coverage is descriptive and does not measure semantic quality.", "",
                  "| Model | Clusters | Unclustered packages |", "| --- | ---: | ---: |"]
        for variant, result in clustering.items():
            lines.append(f"| {variant} | {result['cluster_count']} | {result['unclustered_count']} |")
        lines += ["", "Refer to model_clustering.json for representatives and all memberships."]
    (output / "MODEL_COMPARISON.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "score", "prepare-model", "score-model"])
    parser.add_argument("--output", type=Path, default=ROOT / "clustering_runs" / "baseline")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "score":
        evaluate(args.output)
    elif args.stage == "prepare-model":
        prepare_model(args.output)
    else:
        evaluate_model(args.output)


if __name__ == "__main__":
    main()
