# Top of The Hops - Weekly Homebrew trends 🍺

Weekly snapshots of Homebrew's 30-day package analytics, turned into trend charts.

Homebrew exposes useful install data, but only as a rolling view. **Top of The Hops** takes snapshots so you can see how the popularity of the tools change over time in the different categories.

The report currently covers things like code editors, coding harnesses, AI agents, LLM runners, containers, programming languages, terminals and more.

**See the report:** https://greg76.github.io/toth/

## How it works

- [Fetch](fetch.py) Homebrew's 30-day analytics for [formulas](https://formulae.brew.sh/analytics/install-on-request/30d/) and [casks](https://formulae.brew.sh/analytics/cask-install/30d/).
  - Store compressed snapshots of the original API responses in [dumps](dumps/).
  - Rebuild the SQLite database from those snapshots when needed.
- Build a set of curated categories by searching Homebrew package descriptions.
- Generate the charts published in `docs/`.
- [discovery.py](discovery.py) is a notebook script to validate queries, charts before they graduate to [charts.py](charts.py) that produces the output for the [web site folder](docs/)

The data is based on Homebrew's install analytics, so the numbers represent **installs over the preceding 30 days**, sampled at roughly weekly intervals — not installs during that particular week.

## Weekly automation

The [weekly GitHub Actions workflow](.github/workflows/weekly-update.yml) runs every Sunday at 09:17 Europe/Zurich time. From the Actions tab, manual runs can update with fresh analytics, rebuild charts from saved snapshots, or publish the current site without fetching data. It uses Python 3.14 and [requirements-workflow.txt](requirements-workflow.txt), which contains the small dependency set needed to fetch data and generate the site.

Each run rebuilds the ignored SQLite database from the committed snapshots, fetches the latest data, updates `docs/charts.json`, commits changed snapshots and chart data, and deploys `docs/` to GitHub Pages. The repository's Pages source must be set to **GitHub Actions**. Keep the existing `.zst` files in `dumps/` committed so runs retain the historical trend data.

## Running locally

```bash
pip install -r requirements.txt
python fetch.py
```

The collected data is stored in `brew_stats.db` and the raw API responses are kept in `dumps/`.

To rebuild the database from existing snapshots:

```bash
python fetch.py --backfill
```

Produce the [charts.json](docs/charts.json) export that is used for dynamically build the list of diagram cards. (in a chart.js friendly format)

```bash
python charts.py
```
