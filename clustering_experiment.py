"""Freeze a clustering baseline, audit Homebrew metadata, and pilot READMEs.

Run with the development requirements installed; no charts or published data change.
Generated artifacts live in clustering_runs/ and are reused on subsequent runs.
"""

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 2
ANCHOR_COUNT = 20
HOLDOUT_COUNT = 5
PILOT_SIZE = 50
REQUEST_TIMEOUT_SECONDS = 30
MAX_README_BYTES = 1_000_000
PACKAGE_TYPES = {0: "formula", 1: "cask"}
ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def request_json(url, github=False):
    headers = {"User-Agent": "toth-clustering-experiment", "Accept": "application/json"}
    if github:
        headers["Accept"] = "application/vnd.github+json"
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def load_model(revision=None, model_name=MODEL):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sentence_transformers import SentenceTransformer

    try:
        snapshot = Path(snapshot_download(model_name, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        snapshot = Path(snapshot_download(
            model_name, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "1_Pooling/*"],
        ))
    return SentenceTransformer(str(snapshot)), snapshot.name


def diverse_indices(distances, candidates, limit):
    """Farthest-first sampling covers different regions, deterministically."""
    import numpy as np

    remaining = list(candidates)
    selected = []
    while remaining and len(selected) < limit:
        index = remaining[0] if not selected else max(
            remaining, key=lambda i: float(np.min(distances[i, selected]))
        )
        selected.append(index)
        remaining.remove(index)
    return selected


def baseline(output, db):
    import hdbscan
    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances

    snapshot_path = output / "packages.json"
    if not snapshot_path.exists():
        # Existing packages.json snapshots remain frozen; new runs use stored metadata.
        with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as connection:
            date = connection.execute("SELECT MAX(date) FROM counts").fetchone()[0]
            rows = connection.execute(
                """SELECT names.name_id, names.name, names.kind_id, counts.count, names.description
                FROM counts JOIN names ON names.name_id = counts.name_id
                WHERE counts.date = ? ORDER BY counts.count DESC, names.name_id LIMIT 500""",
                (date,),
            ).fetchall()
        packages = [
            {"id": f"{PACKAGE_TYPES[kind]}:{name}", "name": name,
             "package_type": PACKAGE_TYPES[kind], "count": count,
             "description": (description or "").strip()}
            for _, name, kind, count, description in rows
        ]
        write_json(snapshot_path, {"analytics_date": date, "packages": packages})
    snapshot = read_json(snapshot_path)
    packages = snapshot["packages"]
    described = [p for p in packages if p["description"]]
    if len(described) < MIN_CLUSTER_SIZE:
        raise ValueError("Too few described packages for the baseline")
    manifest_path = output / "baseline.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["packages_sha256"] != hashlib.sha256(snapshot_path.read_bytes()).hexdigest():
            raise ValueError("Frozen package snapshot changed; use a new output directory")
        print("Reusing frozen baseline", flush=True)
        return

    model, revision = load_model()
    embeddings = model.encode([p["description"] for p in described], normalize_embeddings=True)
    distances = cosine_distances(embeddings).astype("float64")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
        metric="precomputed", cluster_selection_method="eom",
    ).fit(distances)
    np.savez_compressed(output / "baseline_vectors.npz", embeddings=embeddings, distances=distances)
    assignments = []
    for i, package in enumerate(described):
        assignments.append({**package, "cluster": int(clusterer.labels_[i]),
                            "membership_strength": float(clusterer.probabilities_[i])})
    write_json(output / "assignments.json", assignments)
    clusters = []
    for cid in sorted(set(clusterer.labels_) - {-1}):
        indices = np.flatnonzero(clusterer.labels_ == cid)
        centroid = embeddings[indices].mean(axis=0, keepdims=True)
        representative_order = indices[np.argsort(cosine_distances(embeddings[indices], centroid).ravel())]
        weak_order = indices[np.argsort(clusterer.probabilities_[indices], kind="stable")]
        clusters.append({
            "cluster": int(cid), "size": len(indices),
            "persistence": float(clusterer.cluster_persistence_[cid]),
            "representative_members": [assignments[i] for i in representative_order[:5]],
            "weak_members": [assignments[i] for i in weak_order[:3]],
        })
    write_json(output / "clusters.json", clusters)
    anchors = diverse_indices(distances, range(len(described)), ANCHOR_COUNT)
    judgments = []
    for anchor_number, i in enumerate(anchors):
        neighbors = [j for j in np.argsort(distances[i]) if j != i][:5]
        for rank, j in enumerate(neighbors, start=1):
            judgments.append({
                "split": "holdout" if anchor_number >= ANCHOR_COUNT - HOLDOUT_COUNT else "development",
                "anchor_id": described[i]["id"], "anchor_description": described[i]["description"],
                "neighbor_id": described[j]["id"], "neighbor_description": described[j]["description"],
                "rank": rank, "cosine_distance": float(distances[i, j]),
                "related_functionality": "", "notes": "",
            })
    judgment_path = output / "neighbor_judgments.csv"
    if not judgment_path.exists():
        write_csv(judgment_path, judgments, list(judgments[0]))
    write_json(manifest_path, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analytics_date": snapshot["analytics_date"], "model": MODEL, "model_revision": revision,
        "min_cluster_size": MIN_CLUSTER_SIZE, "min_samples": MIN_SAMPLES,
        "distance": "cosine", "normalized_embeddings": True,
        "packages_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "hdbscan", "scikit-learn", "sentence-transformers", "torch"]},
        "package_count": len(packages), "described_count": len(described),
        "missing_count": len(packages) - len(described), "cluster_count": len(clusters),
        "unclustered_count": int(sum(clusterer.labels_ == -1)),
    })
    print(f"Baseline: {len(clusters)} clusters; {len(judgments)} neighbor judgments prepared", flush=True)


def metadata_index(metadata):
    index = defaultdict(list)
    for kind, items in metadata.items():
        for item in items:
            name = item["token"] if kind == "cask" else item["name"]
            names = {name, item.get("full_name"), item.get("full_token")}
            names.update(item.get("aliases", []))
            names.update(item.get("oldnames", []))
            names.update(item.get("old_tokens", []))
            if item.get("tap"):
                names.add(f"{item['tap']}/{name}")
            for alias in names - {None, ""}:
                index[(kind, alias)].append(item)
    return index


def repository_candidate(item):
    signals = []
    urls = item.get("urls") or {}
    sources = {
        "homepage": item.get("homepage"), "head": (urls.get("head") or {}).get("url"),
        "stable": (urls.get("stable") or {}).get("url"), "download": item.get("url"),
    }
    subdirectory = False
    for source, url in sources.items():
        if not url:
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or not all(re.fullmatch(r"[\w.-]+", p) for p in parts[:2]):
            continue
        owner, repo = parts[0], parts[1].removesuffix(".git")
        if owner.lower() == "homebrew" or repo.lower().startswith("homebrew-"):
            continue
        signals.append({"field": source, "repository": f"{owner}/{repo}", "url": url})
        subdirectory |= len(parts) > 4 and parts[2] in {"tree", "blob"}
    repos = {s["repository"].lower() for s in signals}
    if not repos:
        return {"repository_status": "unresolved", "repository": None, "signals": signals}
    if len(repos) > 1:
        return {"repository_status": "conflicting_urls", "repository": None, "signals": signals}
    return {"repository_status": "subdirectory_review" if subdirectory else "candidate",
            "repository": signals[0]["repository"], "signals": signals}


def audit(output):
    metadata = {}
    for kind in ["formula", "cask"]:
        path = output / f"{kind}_metadata.json"
        if not path.exists():
            print(f"Fetching {kind} metadata", flush=True)
            write_json(path, request_json(f"https://formulae.brew.sh/api/{kind}.json"))
        metadata[kind] = read_json(path)
    index = metadata_index(metadata)
    rows = []
    for package in read_json(output / "packages.json")["packages"]:
        matches = index.get((package["package_type"], package["name"]), [])
        row = {**package, "metadata_status": "matched" if len(matches) == 1 else
               "ambiguous" if matches else "unresolved", "review": "pending"}
        if len(matches) == 1:
            item = matches[0]
            name = item.get("token") or item["name"]
            row.update({"canonical_name": name, "tap": item.get("tap"),
                        "aliases": item.get("aliases", []) + item.get("old_tokens", []),
                        "display_names": item.get("name") if package["package_type"] == "cask" else [name],
                        "metadata_description": item.get("desc") or "",
                        "homepage": item.get("homepage"), **repository_candidate(item)})
        else:
            row.update({"repository_status": "no_metadata", "repository": None})
        rows.append(row)
    repositories = defaultdict(list)
    for row in rows:
        if row["repository"]:
            repositories[row["repository"].lower()].append(row)
    for group in repositories.values():
        families = {re.sub(r"@.*$", "", row["canonical_name"]) for row in group}
        if len(families) > 1:
            for row in group:
                if row["repository_status"] == "candidate":
                    row["repository_status"] = "shared_repository_review"
    write_json(output / "metadata_audit.json", rows)
    counts = {"metadata": dict(Counter(r["metadata_status"] for r in rows)),
              "repositories": dict(Counter(r["repository_status"] for r in rows)),
              "description_differences": sum(r.get("metadata_description", "").strip() != r["description"]
                                             for r in rows if r["metadata_status"] == "matched")}
    write_json(output / "coverage.json", counts)
    write_json(output / "metadata_sources.json", [
        {"url": f"https://formulae.brew.sh/api/{kind}.json",
         "snapshot": f"{kind}_metadata.json",
         "saved_at": datetime.fromtimestamp(
             (output / f"{kind}_metadata.json").stat().st_mtime, timezone.utc
         ).isoformat(),
         "sha256": hashlib.sha256((output / f"{kind}_metadata.json").read_bytes()).hexdigest()}
        for kind in metadata
    ])
    print(json.dumps(counts, indent=2), flush=True)


SKIP_SECTIONS = re.compile(
    r"\b(install(?:ation|ing)?|build(?:ing)?|compil(?:e|ing|ation)|contribut\w*|"
    r"licen[cs]e|sponsor\w*|donat\w*|acknowledg\w*|changelog|development|"
    r"getting started|quick ?start|download\w*|benchmark\w*|table of contents|"
    r"code quality status|contact|community|support|documentation|docs|external links|"
    r"system requirements|release cycle|bug reports|code of conduct)\b", re.I
)


def readme_prose(markdown):
    """Heuristic extraction; exported excerpts still require semantic review."""
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    markdown = re.sub(r"<pre\b.*?</pre>", "", markdown, flags=re.S | re.I)
    markdown = re.sub(r"<script\b.*?</script>", "", markdown, flags=re.S | re.I)
    markdown = re.sub(r"<[^>]+>", "", markdown, flags=re.S)
    # Removing images first also removes the inner image in linked badges.
    markdown = re.sub(r"!\[[^]]*\]\([^)]*\)", "", markdown)
    markdown = re.sub(
        r"\[([^]\n]*)\]\([^)]*\)|\[([^]\n]*)\]\[[^]]*\]",
        lambda match: match[1] if match[1] is not None else match[2], markdown,
    )
    # Normalize Markdown's alternative heading syntax before filtering sections.
    markdown = re.sub(
        r"(?m)^([^\n]+)\n([=-])\2{2,}\s*$",
        lambda match: ("# " if match[2] == "=" else "## ") + match[1],
        markdown,
    )
    lines = []
    fence = None
    skip_level = None
    for line in markdown.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match[1][0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence or line.startswith("    ") or line.startswith("\t"):
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", stripped)
        if heading:
            level, title = len(heading[1]), heading[2]
            if skip_level is not None and level > skip_level:
                continue
            skip_level = level if SKIP_SECTIONS.match(title) else None
        if skip_level is not None:
            continue
        if re.match(r"^\[[^]]+\]:|^\||^\s*[-=^~]{3,}\s*$|^\.\. |^:\w+:|^>", stripped):
            continue
        stripped = re.sub(r"https?://\S+", "", stripped)
        stripped = html.unescape(stripped).strip(" #*`-\t")
        if re.search(r"[a-zA-Z]{3}", stripped):
            lines.append(stripped)
    return "\n".join(lines)


def readme_result(repository, cache):
    path = cache / (repository.lower().replace("/", "--") + ".json")
    if path.exists():
        return read_json(path)
    url = f"https://api.github.com/repos/{repository}/readme"
    try:
        data = request_json(url, github=True)
    except urllib.error.HTTPError as error:
        if error.code in {403, 429}:
            raise RuntimeError(
                f"GitHub access or rate limit at {repository}; rerun later with GH_TOKEN if needed. "
                f"Reset: {error.headers.get('X-RateLimit-Reset', 'unspecified')}"
            ) from error
        if error.code != 404:
            raise
        result = {"status": "not_found", "api_url": url}
    else:
        result = {"api_url": url, "source_url": data.get("html_url"),
                  "blob_sha": data.get("sha"), "path": data.get("path"),
                  "fetched_at": datetime.now(timezone.utc).isoformat()}
        if data.get("encoding") != "base64" or data.get("size", 0) > MAX_README_BYTES:
            result["status"] = "unsupported_content"
        else:
            raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            result.update({"status": "fetched", "markdown": raw})
    write_json(path, result)
    return result


def save_readme_review(path, results):
    previous = {}
    if path.exists():
        with path.open(newline="") as stream:
            previous = {row["id"]: row for row in csv.DictReader(stream)}
    fields = ["id", "repository", "status", "source_url", "blob_sha", "excerpt", "reviewed_excerpt", "review", "usable", "notes"]
    rows = []
    for result in results:
        row = {key: result.get(key, "") for key in fields}
        old = previous.get(result["id"])
        if old:
            if all(old.get(key, "") == row[key] for key in ["excerpt", "repository", "blob_sha"]):
                row.update({key: old.get(key, "") for key in ["reviewed_excerpt", "review", "usable", "notes"]})
            elif old.get("usable") or old.get("notes"):
                row["notes"] = f"Excerpt changed; review again. Previous decision: {old.get('usable', '')}. {old.get('notes', '')}"
        rows.append(row)
    write_csv(path, rows, fields)


def pilot(output):
    import numpy as np

    assignments = read_json(output / "assignments.json")
    audit_rows = {r["id"]: r for r in read_json(output / "metadata_audit.json")}
    distances = np.load(output / "baseline_vectors.npz")["distances"]
    selection_path = output / "pilot_selection.json"
    if not selection_path.exists():
        eligible = [i for i, p in enumerate(assignments)
                    if audit_rows[p["id"]]["repository_status"] == "candidate"]
        indices = diverse_indices(distances, eligible, PILOT_SIZE)
        write_json(selection_path, [assignments[i]["id"] for i in indices])
    selected = read_json(selection_path)
    model, _ = load_model(read_json(output / "baseline.json")["model_revision"])
    cache = output / "readmes"
    cache.mkdir(exist_ok=True)
    results = []
    for number, identity in enumerate(selected, start=1):
        package = audit_rows[identity]
        repository = package["repository"]
        print(f"README {number}/{len(selected)}: {repository}", flush=True)
        result = readme_result(repository, cache)
        # Reserve room for the package name, description, and special tokens.
        prefix = f"{package['canonical_name']}: {package['metadata_description']}\n"
        budget = model.max_seq_length - len(model.tokenizer.encode(prefix, add_special_tokens=False)) - model.tokenizer.num_special_tokens_to_add(pair=False)
        prose = readme_prose(result.get("markdown", ""))
        encoded = model.tokenizer(prose, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        token_count = len(encoded["input_ids"])
        excerpt_end = encoded["offset_mapping"][min(budget, token_count) - 1][1] if budget > 0 and token_count else 0
        excerpt = prose[:excerpt_end]
        results.append({"id": identity, "repository": repository,
                        **{k: v for k, v in result.items() if k != "markdown"},
                        "excerpt": excerpt, "token_budget": max(0, budget),
                        "truncated": token_count > max(0, budget),
                        "review": "pending", "usable": "", "notes": ""})
        # Checkpoint every success so interruption loses no downloads or excerpts.
        write_json(output / "pilot_readmes.json", results)
    save_readme_review(output / "readme_review.csv", results)
    print(f"Pilot: {sum(bool(r['excerpt']) for r in results)} nonempty excerpts; semantic review pending", flush=True)


def report(output):
    baseline_data = read_json(output / "baseline.json")
    text = ["# Clustering experiment", "", "## Frozen baseline", "",
            f"Analytics date: {baseline_data['analytics_date']}. Model: `{MODEL}`.", "",
            f"{baseline_data['package_count']} packages; {baseline_data['described_count']} described; "
            f"{baseline_data['missing_count']} missing descriptions; {baseline_data['cluster_count']} clusters; "
            f"{baseline_data['unclustered_count']} described packages unclustered.", "",
            "`clusters.json` contains representative and weak members, membership strengths, and persistence.", "",
            "## Review", "",
            "Fill `neighbor_judgments.csv`: related_functionality = yes/no/uncertain. "
            "Use related purpose, not shared language or dependencies. Review development anchors first; "
            "leave holdout anchors for the final comparison. Scores are not calibrated semantic probabilities."]
    text += ["", "## Cluster inspection", "",
             "| Cluster | Size | Persistence | Representative packages | Weak members |",
             "| --- | --- | --- | --- | --- |"]
    clusters = read_json(output / "clusters.json")
    for cluster in clusters:
        representatives = ", ".join(p["id"] for p in cluster["representative_members"])
        weak = ", ".join(p["id"] for p in cluster["weak_members"])
        text.append(f"| {cluster['cluster']} | {cluster['size']} | {cluster['persistence']:.3f} | {representatives} | {weak} |")
    if clusters and len({cluster["persistence"] for cluster in clusters}) == 1:
        text += ["", "Persistence is identical for every cluster in this run and cannot rank their quality."]
    described = read_json(output / "assignments.json")
    description_counts = Counter(package["description"] for package in described)
    text += ["", f"{sum(count > 1 for count in description_counts.values())} duplicate-description groups "
             f"cover {sum(count for count in description_counts.values() if count > 1)} packages. "
             "Inspect version families separately when judging density and coverage."]
    if (output / "coverage.json").exists():
        text += ["", "## Metadata coverage", "", "```json", json.dumps(read_json(output / "coverage.json"), indent=2), "```", "",
                 "Repository candidates are automatic matches, not reviewed identities. Shared repositories, "
                 "conflicting URLs, and explicit subdirectories are excluded from automatic enrichment."]
    if (output / "pilot_readmes.json").exists():
        rows = read_json(output / "pilot_readmes.json")
        text += ["", "## README pilot", "", f"{len(rows)} packages attempted; {sum(bool(r['excerpt']) for r in rows)} nonempty excerpts.", "",
                 "Review `readme_review.csv` for correct repository identity, package-specific scope, "
                 "and useful functional prose. Set review and usable; retain reasons for rejected excerpts. "
                 "No semantic improvement has been established."]
    if (output / "COMPARISON.md").exists():
        text += ["", "## Text comparison", "",
                 "Refer to [COMPARISON.md](COMPARISON.md) for the development comparison, "
                 "reviewer provenance, examples, and uncertainty. No representation has been promoted "
                 "to the notebook or published charts."]
    if (output / "MODEL_COMPARISON.md").exists():
        text += ["", "Refer to [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the controlled "
                 "MiniLM/MPNet comparison and HDBSCAN membership inspection."]
    if (output / "FAMILY_COMPARISON.md").exists():
        text += ["", "Latest experiments: [family-aware evaluation](FAMILY_COMPARISON.md), "
                 "[grouping methods](GROUPING_COMPARISON.md), and "
                 "[complete assistant group review](GROUP_REVIEW.md)."]
    (output / "REPORT.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "audit", "pilot", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "clustering_runs" / "baseline")
    parser.add_argument("--db", type=Path, default=ROOT / "brew_stats.db")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in {"baseline", "all"}:
            baseline(args.output, args.db)
        if args.stage in {"audit", "all"}:
            audit(args.output)
        if args.stage in {"pilot", "all"}:
            pilot(args.output)
    finally:
        if (args.output / "baseline.json").exists():
            report(args.output)


if __name__ == "__main__":
    main()
