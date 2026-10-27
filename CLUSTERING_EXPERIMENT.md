# Clustering experiment

Freeze a description-only baseline, audit upstream repository coverage, and fetch
a 50-package README pilot:

```bash
python clustering_experiment.py all
```

The stages can also run separately: `baseline`, `audit`, and `pilot`. Results and
download caches are stored in the ignored `clustering_runs/baseline/` directory.
Reruns reuse the frozen package selection, baseline vectors, metadata, and READMEs.
Use `--output clustering_runs/another-run` for a fresh experiment.

Start with `clustering_runs/baseline/REPORT.md`. Review functional relationships
in `neighbor_judgments.csv` and repository identity and excerpts in
`readme_review.csv`. The latter preserves decisions for unchanged excerpts and
resets them when the excerpt changes. Keep the five holdout anchors for the final
comparison. Nonempty README text and HDBSCAN scores do not establish semantic
quality.

The first run needs network access. GitHub requests use `GH_TOKEN` or
`GITHUB_TOKEN` when available; public access also works within anonymous rate
limits. A rate-limit error stops the pilot; rerun after the reset or with a token
to resume cached downloads. `HF_HUB_OFFLINE=1` reuses a locally cached embedding
model. Refer to [CLUSTERING_PROGRESS.md](CLUSTERING_PROGRESS.md) for results and
remaining work.

After reviewing the README passages, compare text representations:

```bash
python compare_embeddings.py prepare
python compare_embeddings.py score
```

`prepare` reuses the baseline vectors and pinned model, embeds name-plus-description
and approved README passages, and prepares a union of development neighbor pairs.
User labels are copied from `neighbor_judgments.csv`; add judgments for new pairs
in `comparison_judgments.csv`, with the reviewer recorded. `score` writes
`COMPARISON.md`, showing uncertainty and separate user-only metrics. Approved
passages use `reviewed_excerpt` in `readme_review.csv` and must be verbatim spans
of the extracted source. Original user inputs and holdout rows are preserved.

To compare `all-mpnet-base-v2` against MiniLM using exactly the same
name-plus-description inputs:

```bash
python compare_embeddings.py prepare-model
python compare_embeddings.py score-model
```

The first run downloads MPNet; subsequent runs reuse its pinned revision and
saved vectors. Review new pairs in `model_judgments.csv`, then run `score-model`
to update `MODEL_COMPARISON.md`. `prepare-model` also runs HDBSCAN at the frozen
settings and saves full memberships in `model_clustering.json`. These commands
leave the notebook's model and published charts unchanged.

The next experiment compares Qwen with both controls, collapses versioned
packages into families, fits thresholds for rejecting weak neighbors, and
compares grouping methods:

```bash
python evaluate_families.py prepare
python evaluate_families.py score
python evaluate_families.py group
```

The first command downloads Qwen when needed and prepares `family_judgments.csv`.
Review any new pairs there, recording the reviewer, before running `group`.
Original user labels take precedence over assistant labels across version
variants; conflicting family labels remain uncertain. The reports are
`FAMILY_COMPARISON.md` and `GROUPING_COMPARISON.md`. Threshold fitting excludes
the assessed anchor family and is development evaluation, not final validation.
Refer to [CLUSTERING_RECOMMENDATION.md](CLUSTERING_RECOMMENDATION.md) for the
current recommendation and the separately recorded complete group review.
