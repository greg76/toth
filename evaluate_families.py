"""Compare three embedding models by distinct package family and selective matches.

Stages: prepare, score, group. Original user labels and earlier experiments are read-only.
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from clustering_experiment import ROOT, load_model, read_json, write_csv, write_json
from compare_embeddings import load_csv, summarize

QWEN_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
QWEN_TASK = 'Group software packages by their primary functionality, distinguishing application tools from libraries and runtimes.'
QWEN_PROMPT = f'Instruct: {QWEN_TASK}\nQuery:'
MODELS = ('minilm', 'mpnet', 'qwen')
TARGET_PRECISION = 0.8
MIN_CALIBRATION_ACCEPTED = 5
NEIGHBOR_LIMIT = 5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family_id(package, metadata):
    row = metadata.get(package['id'], {})
    name = row.get('canonical_name') or package['name']
    # Preserve tap-qualified unknown identities and formula/cask distinctions.
    name = re.sub(r'@[^/]+$', '', name)
    return f"{package['package_type']}:{name}"


def family_map(packages, metadata):
    families = defaultdict(list)
    for i, package in enumerate(packages):
        families[family_id(package, metadata)].append(i)
    return dict(families)


def family_neighbors(embeddings, anchors, packages, families):
    index = {p['id']: i for i, p in enumerate(packages)}
    membership = {i: family for family, indices in families.items() for i in indices}
    distances = cosine_distances(embeddings)
    rows = []
    for anchor in anchors:
        i = index[anchor]
        candidates = []
        for family, indices in families.items():
            if family == membership[i]:
                continue
            best = min(indices, key=lambda j: (distances[i, j], j))
            candidates.append((float(distances[i, best]), family, best))
        for rank, (distance, family, best) in enumerate(sorted(candidates)[:NEIGHBOR_LIMIT], start=1):
            rows.append({'anchor_id': anchor, 'anchor_family': membership[i],
                         'neighbor_id': packages[best]['id'], 'neighbor_family': family,
                         'rank': rank, 'similarity': 1 - distance})
    return rows


def inherited_judgments(output, packages, metadata):
    package_map = {p['id']: p for p in packages}
    collected = defaultdict(list)
    # Later files carry earlier reviews too; deduplicate their exact directional pairs.
    exact = {}
    for filename in ['comparison_judgments.csv', 'model_judgments.csv']:
        for row in load_csv(output / filename):
            if row['related_functionality']:
                exact[(row['anchor_id'], row['neighbor_id'])] = row
    user_rows = [r for r in load_csv(output / 'neighbor_judgments.csv') if r['split'] == 'development']
    anchors = list(dict.fromkeys(r['anchor_id'] for r in user_rows))
    for row in user_rows:
        exact[(row['anchor_id'], row['neighbor_id'])] = {**row, 'reviewer': 'user'}
    for (anchor, neighbor), row in exact.items():
        if anchor not in anchors:
            continue
        family = family_id(package_map[neighbor], metadata)
        collected[(anchor, family)].append(row)
    combined = {}
    for key, rows in collected.items():
        user = [r for r in rows if r['reviewer'] == 'user']
        preferred = user or rows
        labels = {r['related_functionality'] for r in preferred if r['related_functionality']}
        combined[key] = {'related_functionality': next(iter(labels)) if len(labels) == 1 else 'uncertain',
                         'reviewer': 'user' if user else 'assistant',
                         'notes': 'Inherited across version variants; user labels take precedence. ' +
                                  ('Conflicting labels retained as uncertain.' if len(labels) > 1 else '')}
    return anchors, combined


def prepare(output):
    packages = read_json(output / 'assignments.json')
    metadata = {r['id']: r for r in read_json(output / 'metadata_audit.json')}
    families = family_map(packages, metadata)
    inputs = read_json(output / 'comparison_inputs.json')
    if [p['id'] for p in inputs] != [p['id'] for p in packages]:
        raise ValueError('Frozen inputs and package identities are misaligned')
    texts = [p['name_description'] for p in inputs]
    fingerprint = digest({'ids': [p['id'] for p in packages], 'texts': texts, 'prompt': QWEN_PROMPT})
    manifest_path = output / 'qwen_manifest.json'
    archive_path = output / 'qwen_vectors.npz'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if manifest.get('input_sha256') != fingerprint or not archive_path.exists():
        model, revision = load_model(revision=manifest.get('model_revision'), model_name=QWEN_MODEL)
        started = time.perf_counter()
        # The same instruction is applied to every package, so cosine distances
        # remain symmetric and usable for clustering, unlike query/document prompts.
        vectors = model.encode(texts, prompt=QWEN_PROMPT, normalize_embeddings=True,
                               batch_size=8, show_progress_bar=True)
        # Normalize after conversion as reduced-precision model output can have
        # noticeable norm rounding even when encode(normalize_embeddings=True).
        vectors = vectors.astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.savez_compressed(archive_path, qwen=vectors)
        write_json(manifest_path, {'model': QWEN_MODEL, 'model_revision': revision,
                   'input_sha256': fingerprint, 'prompt': QWEN_PROMPT,
                   'symmetric_instruction': True, 'package_count': len(packages),
                   'dimensions': vectors.shape[1], 'encode_seconds': time.perf_counter() - started,
                   'device': str(model.device), 'stored_vector_normalization': 'float32_l2',
                   'evaluation_split': 'development'})
    cached = np.load(archive_path)['qwen']
    if not np.allclose(np.linalg.norm(cached, axis=1), 1, atol=1e-5):
        cached = cached.astype(np.float32)
        cached /= np.linalg.norm(cached, axis=1, keepdims=True)
        np.savez_compressed(archive_path, qwen=cached)
        manifest = read_json(manifest_path)
        manifest['stored_vector_normalization'] = 'float32_l2'
        write_json(manifest_path, manifest)
    embeddings = load_vectors(output)
    anchors, known = inherited_judgments(output, packages, metadata)
    neighbors = []
    for model, vectors in embeddings.items():
        if vectors.shape[0] != len(packages):
            raise ValueError('Embedding count differs from the frozen corpus')
        neighbors.extend({**row, 'model': model} for row in family_neighbors(vectors, anchors, packages, families))
    write_json(output / 'family_map.json', {family: [packages[i]['id'] for i in indices] for family, indices in families.items()})
    write_json(output / 'family_neighbors.json', neighbors)
    review_path = output / 'family_judgments.csv'
    previous = {(r['anchor_id'], r['neighbor_family']): r for r in load_csv(review_path)} if review_path.exists() else {}
    package_map = {p['id']: p for p in packages}
    union = sorted({(r['anchor_id'], r['neighbor_family']) for r in neighbors})
    rows = []
    for anchor, family in union:
        source = known.get((anchor, family), previous.get((anchor, family), {}))
        representative = packages[families[family][0]]
        rows.append({'anchor_id': anchor, 'anchor_description': package_map[anchor]['description'],
                     'neighbor_family': family, 'neighbor_description': representative['description'],
                     'related_functionality': source.get('related_functionality', ''),
                     'reviewer': source.get('reviewer', ''), 'notes': source.get('notes', '')})
    write_csv(review_path, rows, list(rows[0]))
    write_json(output / 'family_protocol.json', {
        'package_count': len(packages), 'family_count': len(families),
        'family_definition': 'Same package type and canonical name after removing @version suffix. Unknown tap identities retained.',
        'ranking': 'Maximum similarity of members; omit anchor family, return five distinct families.',
        'clustering_vectors': 'Mean normalized package embeddings per family, normalized again.',
        'target_precision': TARGET_PRECISION, 'minimum_calibration_accepted': MIN_CALIBRATION_ACCEPTED,
        'validation': 'Leave one development anchor out of threshold fitting. Not an independent final test.',
        'qwen_instruction': QWEN_PROMPT,
    })
    print(f"{len(packages)} packages -> {len(families)} families. {sum(not r['related_functionality'] for r in rows)} new family pairs need review.")
    score(output)


def load_vectors(output):
    return {'minilm': np.load(output / 'comparison_vectors.npz')['name_description'],
            'mpnet': np.load(output / 'model_vectors.npz')['mpnet'],
            'qwen': np.load(output / 'qwen_vectors.npz')['qwen']}


def choose_threshold(rows, judgments):
    decisive = [(r['similarity'], judgments.get((r['anchor_id'], r['neighbor_family']), {}).get('related_functionality', '')) for r in rows]
    decisive = [(similarity, label) for similarity, label in decisive if label in {'yes', 'no'}]
    options = []
    for threshold in sorted({similarity for similarity, _ in decisive}):
        labels = [label for similarity, label in decisive if similarity >= threshold]
        precision = labels.count('yes') / len(labels)
        if len(labels) >= MIN_CALIBRATION_ACCEPTED and precision >= TARGET_PRECISION:
            options.append((len(labels), precision, threshold))
    return max(options)[2] if options else None


def family_summary(rows, judgments):
    adapted = [{**r, 'neighbor_id': r['neighbor_family']} for r in rows]
    return summarize(adapted, judgments)


def score(output):
    neighbors = read_json(output / 'family_neighbors.json')
    reviews = load_csv(output / 'family_judgments.csv')
    judgments = {(r['anchor_id'], r['neighbor_family']): r for r in reviews}
    metrics, selections = {}, []
    for model in MODELS:
        rows = [r for r in neighbors if r['model'] == model]
        anchors = list(dict.fromkeys(r['anchor_id'] for r in rows))
        accepted, thresholds = [], {}
        for anchor in anchors:
            held_out_family = next(r['anchor_family'] for r in rows if r['anchor_id'] == anchor)
            training = [r for r in rows if r['anchor_family'] != held_out_family]
            threshold = choose_threshold(training, judgments)
            thresholds[anchor] = threshold
            for row in [r for r in rows if r['anchor_id'] == anchor]:
                keep = threshold is not None and row['similarity'] >= threshold
                selections.append({**row, 'threshold': threshold, 'accepted': keep})
                if keep:
                    accepted.append(row)
        accepted_summary = family_summary(accepted, judgments)
        user_judgments = {k: v for k, v in judgments.items() if v['reviewer'] == 'user'}
        accepted_anchors = {r['anchor_id'] for r in accepted}
        metrics[model] = {'top_five': family_summary(rows, judgments),
                          'user_only': family_summary(rows, user_judgments),
                          'selective': accepted_summary,
                          'accepted_slots': len(accepted), 'abstained_slots': len(rows) - len(accepted),
                          'anchors_with_matches': len(accepted_anchors),
                          'anchors_with_no_accepted_match': [a for a in anchors if a not in accepted_anchors],
                          'fold_thresholds': thresholds,
                          'full_development_threshold': choose_threshold(rows, judgments)}
    write_json(output / 'family_metrics.json', metrics)
    write_json(output / 'selective_neighbors.json', selections)
    write_json(output / 'family_evaluation.json', {
        'reviewer_counts': dict(Counter(r['reviewer'] for r in reviews)),
        'family_judgments_sha256': hashlib.sha256((output / 'family_judgments.csv').read_bytes()).hexdigest(),
        'user_judgments_sha256': hashlib.sha256((output / 'neighbor_judgments.csv').read_bytes()).hexdigest(),
        'reference_pairs_sha256': hashlib.sha256((ROOT / 'clustering_reference_pairs.json').read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    lines = ['# Family-aware embedding comparison', '',
             'Same name-plus-description inputs and 15 development anchors. Each candidate family appears once; '
             'the anchor family is omitted. Qwen adds the same fixed task instruction to every package. '
             'These model configurations are compared, rather than claiming an architecture-only effect.', '',
             '| Model | Relevant | Irrelevant | Uncertain | Unjudged | Decisive relevance |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model, data in metrics.items():
        s = data['top_five']
        rate = f"{s['decisive_relevance']:.1%}" if s['decisive_relevance'] is not None else 'n/a'
        lines.append(f"| {model} | {s['yes']} | {s['no']} | {s['uncertain']} | {s['unjudged']} | {rate} |")
    lines += ['', '## Selective matches', '',
              'For each anchor, fit a model-specific similarity threshold on the other 14 anchors. '
              'Require at least five decisively judged calibration pairs and at least 80% relevance there. '
              'If none qualifies, abstain. Score the held-out anchor separately. Uncertain and unjudged '
              'pairs do not become negatives; their accepted counts remain visible. An 80% calibration target '
              'is not a guarantee of evaluation precision or a calibrated probability.', '',
              '| Model | Accepted slots | Yes / no / uncertain / unjudged | Decisive relevance | Anchors with matches |',
              '| --- | ---: | --- | ---: | ---: |']
    for model, data in metrics.items():
        s = data['selective']
        rate = f"{s['decisive_relevance']:.1%}" if s['decisive_relevance'] is not None else 'n/a'
        lines.append(f"| {model} | {data['accepted_slots']} | {s['yes']} / {s['no']} / {s['uncertain']} / {s['unjudged']} | {rate} | {data['anchors_with_matches']} / 15 |")
    lines += ['', '## Neighbors by anchor', '', '| Anchor | MiniLM | MPNet | Qwen |', '| --- | --- | --- | --- |']
    for anchor in dict.fromkeys(r['anchor_id'] for r in neighbors):
        cells = []
        for model in MODELS:
            rows = [r for r in selections if r['anchor_id'] == anchor and r['model'] == model]
            cells.append('; '.join(r['neighbor_family'] + ' [' + (judgments[(anchor, r['neighbor_family'])]['related_functionality'] or 'unjudged') + (', accepted' if r['accepted'] else ', withheld') + ']' for r in rows))
        lines.append(f"| {anchor} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Limits', '',
              'Prior development labels have already informed the project. Leave-one-anchor-out threshold fitting '
              'reduces direct calibration leakage but is not independent final validation. Several new labels are '
              'assistant judgments. Families collapse @versions only and retain formula/cask roles; distributions '
              'with different names can remain separate. Withholding a match means insufficient evidence, not '
              'proof that no related package exists. Distinct-version support no longer counts as independent families.', '',
              'Model source: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).']
    (output / 'FAMILY_COMPARISON.md').write_text('\n'.join(lines) + '\n')


def select_model(metrics):
    eligible = [model for model in MODELS
                if metrics[model]['selective']['decisive_relevance'] is not None
                and metrics[model]['selective']['decisive_relevance'] >= TARGET_PRECISION
                and metrics[model]['selective']['yes'] + metrics[model]['selective']['no'] >= MIN_CALIBRATION_ACCEPTED]
    if eligible:
        chosen = max(eligible, key=lambda model: (metrics[model]['selective']['yes'],
                     -metrics[model]['selective']['uncertain'], metrics[model]['top_five']['yes'],
                     -MODELS.index(model)))
        return chosen, 'Most accepted relevant matches among models meeting the selective precision target; ties favor fewer uncertain matches and the smaller model.'
    return max(MODELS, key=lambda model: (metrics[model]['top_five']['yes'], -MODELS.index(model))), \
        'No model met the selective target with sufficient support; chose the most relevant distinct-family neighbors for exploration only.'


def pair_quality(labels, families, reviews):
    index = {family: i for i, family in enumerate(families)}
    yes = no = uncertain = total_yes = 0
    for row in reviews:
        left, right = row['anchor_family'], row['neighbor_family']
        if left == right:
            continue
        total_yes += row['related_functionality'] == 'yes'
        a, b = labels[index[left]], labels[index[right]]
        together = a != -1 and a == b
        if together:
            yes += row['related_functionality'] == 'yes'
            no += row['related_functionality'] == 'no'
            uncertain += row['related_functionality'] == 'uncertain'
    precision = yes / (yes + no) if yes + no else None
    recall = yes / total_yes if total_yes else 0
    f_half = 1.25 * precision * recall / (.25 * precision + recall) if precision and recall else 0
    return {'coassigned_yes': yes, 'coassigned_no': no, 'coassigned_uncertain': uncertain,
            'decisive_pair_precision': precision, 'known_positive_recall': recall, 'f_half': f_half}


def reference_checks(labels, family_ids, references):
    index = {family: i for i, family in enumerate(family_ids)}
    checks = []
    for reference in references:
        a, b = index.get(reference['left']), index.get(reference['right'])
        if a is None or b is None:
            status = 'missing'
        elif labels[a] == -1 or labels[b] == -1:
            status = 'unassigned'
        else:
            together = bool(labels[a] == labels[b])
            status = 'pass' if together == (reference['related'] == 'yes') else 'fail'
        checks.append({**reference, 'status': status})
    return checks


def cluster_descriptions(labels, family_ids, vectors, texts):
    from sklearn.feature_extraction.text import CountVectorizer

    ids = sorted(set(labels) - {-1})
    if not ids:
        return []
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    word_counts = vectorizer.fit_transform(texts)
    class_counts = np.vstack([np.asarray(word_counts[labels == cid].sum(axis=0)).ravel() for cid in ids])
    lengths = class_counts.sum(axis=1)
    frequencies = class_counts.sum(axis=0)
    idf = np.log1p(lengths.mean() / np.maximum(frequencies, 1))
    ctfidf = class_counts / np.maximum(lengths[:, None], 1) * idf
    terms = vectorizer.get_feature_names_out()
    clusters = []
    for position, cid in enumerate(ids):
        indices = np.flatnonzero(labels == cid)
        centroid = vectors[indices].mean(axis=0, keepdims=True)
        order = indices[np.argsort(cosine_distances(vectors[indices], centroid).ravel())]
        keywords = [terms[j] for j in np.argsort(ctfidf[position])[::-1][:3]]
        clusters.append({'cluster': int(cid), 'label': ', '.join(keywords), 'family_count': len(indices),
                         'representatives': [family_ids[i] for i in order[:5]],
                         'peripheral': [family_ids[i] for i in order[-3:][::-1]],
                         'families': [family_ids[i] for i in indices]})
    return clusters


def group(output):
    import hdbscan
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    reviews = load_csv(output / 'family_judgments.csv')
    if any(not r['related_functionality'] for r in reviews):
        raise ValueError('Finish family judgments before selecting a model for grouping')
    model, reason = select_model(read_json(output / 'family_metrics.json'))
    packages = read_json(output / 'assignments.json')
    metadata = {r['id']: r for r in read_json(output / 'metadata_audit.json')}
    families = family_map(packages, metadata)
    family_ids = list(families)
    package_vectors = load_vectors(output)[model]
    vectors = np.vstack([package_vectors[indices].mean(axis=0) for indices in families.values()])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = cosine_distances(vectors).astype('float64')
    distances = np.maximum(0, (distances + distances.T) / 2)
    np.fill_diagonal(distances, 0)
    condensed = squareform(distances)
    tree = linkage(condensed, method='average', optimal_ordering=True)
    np.savez_compressed(output / 'grouping_vectors.npz', vectors=vectors, linkage=tree)
    package_map = {p['id']: p for p in packages}
    reviewed_pairs = [{**r, 'anchor_family': family_id(package_map[r['anchor_id']], metadata)} for r in reviews]
    references = read_json(ROOT / 'clustering_reference_pairs.json')
    # Predeclared cuts cover the closest 1%, 3%, and 5% of all family distances.
    # Include the retrieval-calibrated cut as another exploratory reference.
    cuts = list(np.quantile(condensed, [.01, .03, .05]))
    calibrated = read_json(output / 'family_metrics.json')[model]['full_development_threshold']
    if calibrated is not None:
        cuts.append(max(0, 1 - calibrated))
    cuts = sorted(set(round(float(cut), 6) for cut in cuts))
    configurations = []
    for minimum in [3, 5]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=minimum, min_samples=2,
                                   metric='precomputed', cluster_selection_method='eom').fit(distances)
        configurations.append((f'hdbscan_min{minimum}', clusterer.labels_, {'method': 'hdbscan', 'min_cluster_size': minimum, 'min_samples': 2}))
        for cut in cuts:
            labels = fcluster(tree, t=cut, criterion='distance').astype(int)
            counts = Counter(labels)
            labels = np.array([label if counts[label] >= minimum else -1 for label in labels])
            configurations.append((f'average_min{minimum}_cut{cut}', labels,
                                   {'method': 'average_linkage', 'minimum_group_size': minimum, 'distance_cut': cut}))
    results = []
    for name, labels, settings in configurations:
        checks = reference_checks(labels, family_ids, references)
        results.append({'configuration': name, 'settings': settings,
                        'family_count': len(family_ids), 'assigned_families': int(sum(labels != -1)),
                        'cluster_count': len(set(labels) - {-1}),
                        'pair_quality': pair_quality(labels, family_ids, reviewed_pairs),
                        'reference_checks': checks,
                        'reference_pass': sum(c['status'] == 'pass' for c in checks),
                        'reference_fail': sum(c['status'] == 'fail' for c in checks),
                        'reference_unassigned': sum(c['status'] == 'unassigned' for c in checks),
                        'assignments': dict(zip(family_ids, labels.tolist()))})
    # This ranks exploratory settings on development labels, not on a final test.
    ranked = sorted(results, key=lambda r: (r['pair_quality']['f_half'], -r['reference_fail'], r['assigned_families']), reverse=True)
    chosen = ranked[0]
    chosen_labels = np.array([chosen['assignments'][identity] for identity in family_ids])
    texts = [packages[indices[0]]['description'] for indices in families.values()]
    chosen['clusters'] = cluster_descriptions(chosen_labels, family_ids, vectors, texts)
    write_json(output / 'grouping_comparison.json', {'model': model, 'model_selection_reason': reason,
               'family_ids': family_ids, 'cuts': cuts, 'configurations': results,
               'exploratory_choice': chosen['configuration'], 'clusters': chosen['clusters'],
               'selection_rule': 'Highest development F0.5 on known family pairs; ties favor fewer failed reference checks, then assigned coverage.',
               'family_judgments_sha256': hashlib.sha256((output / 'family_judgments.csv').read_bytes()).hexdigest(),
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    lines = ['# Grouping comparison', '', f"Selected model configuration: **{model}**. {reason}", '',
             f"Clustering uses {len(family_ids)} distinct families, one normalized mean vector per family. "
             'Version variants no longer supply multiple density samples. Cuts span distance quantiles and the '
             'retrieval-calibrated threshold. Small hierarchy groups are marked unassigned. No dimensionality reduction is used.', '',
             '| Configuration | Clusters | Assigned families | Coassigned yes / no / uncertain | Pair precision | Known-positive recall | Reference pass / fail / unassigned |',
             '| --- | ---: | ---: | --- | ---: | ---: | --- |']
    for result in results:
        q = result['pair_quality']
        precision = f"{q['decisive_pair_precision']:.1%}" if q['decisive_pair_precision'] is not None else 'n/a'
        lines.append(f"| {result['configuration']} | {result['cluster_count']} | {result['assigned_families']} | {q['coassigned_yes']} / {q['coassigned_no']} / {q['coassigned_uncertain']} | {precision} | {q['known_positive_recall']:.1%} | {result['reference_pass']} / {result['reference_fail']} / {result['reference_unassigned']} |")
    lines += ['', '## Exploratory choice', '', f"**{chosen['configuration']}** ranks first by development F0.5, "
              'which puts more weight on precision than recall. This is tuning evidence, not final validation. '
              'Reference checks marked unassigned are not counted as successes. Unknown pairs within a cluster '
              'are not presumed related; known-pair precision is not a full cluster-coherence score.', '',
              '## Groups for inspection', '', '| Keywords | Families | Representative members | Peripheral members |',
              '| --- | ---: | --- | --- |']
    for cluster in chosen['clusters']:
        lines.append(f"| {cluster['label']} | {cluster['family_count']} | {', '.join(cluster['representatives'])} | {', '.join(cluster['peripheral'])} |")
    lines += ['', 'Labels use class-based TF-IDF over group descriptions. They are summaries, not verified category names.', '',
              '## Reference checks for the exploratory choice', '']
    for check in chosen['reference_checks']:
        lines.append(f"- {check['left']} / {check['right']} (expected {check['related']}): **{check['status']}**.")
    lines += ['', '## Limits', '', 'Family-aware evaluation uses the development union of model-generated neighbors and '
              'assistant judgments. Changing the evaluated pair set changes the scores. Report precision alongside coverage '
              'and inspect entire groups before adoption. The averaged family representation also differs from package-level '
              'clustering, so compare methods within this experiment rather than interpreting old coverage differences as model effects. '
              'Keep the hierarchy for exploration across cuts; no configuration or model is promoted to published charts.']
    (output / 'GROUPING_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(f"Grouping: {model}, {len(family_ids)} families; exploratory choice {chosen['configuration']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'score', 'group'])
    parser.add_argument('--output', type=Path, default=ROOT / 'clustering_runs' / 'baseline')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output)
    elif args.stage == 'score':
        score(args.output)
    else:
        group(args.output)


if __name__ == '__main__':
    main()
