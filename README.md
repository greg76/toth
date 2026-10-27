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

The [weekly GitHub Actions workflow](.github/workflows/weekly-update.yml) runs every Sunday at 09:17 Europe/Zurich time. From the Actions tab, manual runs can update with fresh analytics or publish the committed site files without fetching data. It uses Python 3.14 and [requirements-workflow.txt](requirements-workflow.txt), which contains the small dependency set needed to fetch data and generate the site.

Normal updates incrementally update the committed `brew_stats.db`, fetch the latest statistics and package metadata, generate `docs/charts.json`, and commit the database, changed stats snapshots and chart data before deploying `docs/` to GitHub Pages. Publish mode deploys the existing site files. The repository's Pages source must be set to **GitHub Actions**. Keep the existing `.zst` files in `dumps/` committed so runs retain the historical trend data.

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

Package names in bar labels and line-chart legends show a popup on hover, keyboard focus or tap. The popup contains the short package name, linked to its website when available, and a description when present. Clicking a legend name still toggles its line. Combined version series use the base package metadata when available; otherwise they show only metadata shared by all versions.

Chart generation and notebook description searches read SQLite without network access. If the database is lost, stats dumps restore counts; metadata must be recovered from a committed database or fetched again.

## Database

[`brew_stats.db`](brew_stats.db) is a SQLite database. [`init_db()` in fetch.py](fetch.py) creates and updates its schema. Inspect the current schema with:

```bash
sqlite3 brew_stats.db '.schema'
```

`names` table stores package names and their latest metadata; `counts` stores rolling 30-day install counts by snapshot date and package. Join them on `name_id`. In both tables, `kind_id` is `0` for formulae and `1` for casks.

The `totals` table stores the sum of the unfiltered rolling 30-day formula installs on request and cask installs by snapshot date in `formula_total` and `cask_total`; unavailable values are `NULL`.

For example, list the ten most-installed formulae in the latest snapshot:

```sql
SELECT c.date, n.name, c.count
FROM counts AS c
JOIN names AS n ON n.name_id = c.name_id
WHERE c.kind_id = 0
  AND c.date = (SELECT MAX(date) FROM counts WHERE kind_id = 0)
ORDER BY c.count DESC
LIMIT 10;
```
