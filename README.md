# Top of The Hops - Weekly Homebrew trends 🍺

Weekly snapshots of Homebrew's 30-day package analytics, turned into trend charts.

Homebrew exposes useful install data, but only as a rolling view. **Top of The Hops** takes snapshots so you can see how the popularity of the tools change over time in the different categories.

The report currently covers things like code editors, coding harnesses, AI agents, LLM runners, containers, programming languages, terminals and more.

**See the report:** https://greg76.github.io/toth/

## How it works

- [Fetch](fetch.py) Homebrew's 30-day analytics for [formulas](https://formulae.brew.sh/analytics/install-on-request/30d/) and [casks](https://formulae.brew.sh/analytics/cask-install/30d/).
  - Store compressed snapshots of the original API responses in [dumps](dumps/).
  - Restore historical install counts from those snapshots when needed.
  - Refresh formula and cask display names, descriptions and homepage URLs in SQLite.
- Build a set of curated categories by searching Homebrew package descriptions.
- Generate the charts published in `docs/`.
- [discovery.py](discovery.py) is a notebook script to validate queries, charts before they graduate to [charts.py](charts.py) that produces the output for the [web site folder](docs/)

The data is based on Homebrew's install analytics, so the numbers represent **installs over the preceding 30 days**, sampled at roughly weekly intervals — not installs during that particular week.

## Weekly automation

The [weekly GitHub Actions workflow](.github/workflows/weekly-update.yml) runs every Sunday at 09:17 Europe/Zurich time. From the Actions tab, manual runs can update with fresh analytics, rebuild charts from saved snapshots, or publish the current site without fetching data. It uses Python 3.14 and [requirements-workflow.txt](requirements-workflow.txt), which contains the small dependency set needed to fetch data and generate the site.

Normal updates incrementally update the committed `brew_stats.db`, fetch the latest statistics and package metadata, generate `docs/charts.json`, and commit the database, changed stats snapshots and chart data before deploying `docs/` to GitHub Pages. Rebuild mode restores counts from saved stats snapshots and regenerates charts without fetching metadata. Publish mode deploys the existing site files. The repository's Pages source must be set to **GitHub Actions**. Keep the existing `.zst` files in `dumps/` committed so runs retain the historical trend data.

## Running locally

```bash
pip install -r requirements.txt
python fetch.py
```

Use Python 3.14 or later. The committed `brew_stats.db` contains historical install counts and the latest metadata for each `(kind_id, name)`. Standard SQLite is used without an added compression module; Git handles repository compression. Raw statistics responses remain compressed in `dumps/`. Metadata responses are not archived separately.

Normal fetching adds new names and counts, then refreshes metadata from both Homebrew sources. Both sources must download and validate before metadata is updated. Failed refreshes retain previous metadata and timestamps; unmatched or ambiguous names are left unchanged. Third-party taps may have no metadata in the public Homebrew sources.

To refresh only metadata (a failed request exits with an error):

```bash
python fetch.py --metadata-only
```

Each matched name records `metadata_refreshed_at` in UTC. Close database connections before committing `brew_stats.db`; journal, WAL and shared-memory sidecar files are ignored.

To restore counts from existing snapshots, preserving stored metadata:

```bash
python fetch.py --backfill
```

Generate the [charts.json](docs/charts.json) export used to build the chart cards:

```bash
python charts.py > docs/charts.json
```

Package names in bar labels and line-chart legends show their full name, description and website link on hover, keyboard focus or tap. Clicking a legend name still toggles its line. Missing metadata falls back to the full analytics name. Combined version series list the underlying packages separately.

Chart generation and notebook description searches read SQLite without network access. If the database is lost, stats dumps restore counts; metadata must be recovered from a committed database or fetched again.

## Clustering experiment

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
