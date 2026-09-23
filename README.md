# Top of The Hops - Weekly Homebrew trends 🍺

Weekly snapshots of Homebrew's 30-day package analytics, turned into trend charts.

Homebrew exposes useful install data, but only as a rolling view. **Top of The Hops** takes snapshots so you can see how the popularity of the tools change over time in the different categories.

The report currently covers things like code editors, coding harnesses, AI agents, LLM runners, containers, programming languages, terminals and more.

**See the report:** https://greg76.github.io/toth/

## How it works

- [Fetch](fetch.py) Homebrew's 30-day analytics for [formulas](https://formulae.brew.sh/analytics/install-on-request/30d/) and [casks](https://formulae.brew.sh/analytics/cask-install/30d/).
  - [dumps](dumps/) the compressed json snapshots of the original api responses
  - Store weekly snapshots locally in SQLite.
- Build a set of curated categories by searching Homebrew package descriptions.
- Generate the charts published in `docs/`.
- [discovery.py](discovery.py) is a notebook script to validate queries, charts before they graduate to [charts.py](charts.py) that produces the output for the [web site folder](docs/)

The data is based on Homebrew's install analytics, so the numbers represent **installs over the preceding 30 days**, sampled at roughly weekly intervals — not installs during that particular week.

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
