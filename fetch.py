#! python3

"""Fetch Homebrew analytics (formula + cask install counts) and store them.

Each run does two things:
  1. Keeps the raw JSON payloads as zstd-compressed dumps on disk (unchanged
     from the original script's behaviour).
  2. Upserts the parsed (date, kind, name, count) rows into a SQLite
     database, with names normalized into their own table so the counts
     table only ever stores integers.

Usage:
    python fetch.py              # fetch today's data (or reuse today's dump
                                  # if it's already on disk) and upsert it
    python fetch.py --backfill   # rebuild the DB from every dump already in
                                  # dumps/, instead of hitting the network
"""

import argparse
import json
import sqlite3
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from compression import zstd

SCRIPT_DIR = Path(__file__).resolve().parent
DUMPS_DIR = SCRIPT_DIR / "dumps"
DB_PATH = SCRIPT_DIR / "brew_stats.db"

COMPRESSION_LEVEL = 6
MIN_INSTALLS = 500


@dataclass(frozen=True)
class KindSpec:
    kind_id: int
    name: str  # "formula" | "cask" — also the key used in each item dict
    stub: str  # URL path segment on formulae.brew.sh


KIND_SPECS: tuple[KindSpec, ...] = (
    KindSpec(0, "formula", "install-on-request"),
    KindSpec(1, "cask", "cask-install"),
)

Record = tuple[int, str, int]  # (kind_id, name, count)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def parse_num(formatted_string: str) -> int:
    return int(formatted_string.replace(",", ""))


def dump_path(dumps_dir: Path, date_str: str, kind_name: str) -> Path:
    return dumps_dir / f"{date_str}{kind_name}.zst"


# --------------------------------------------------------------------------
# Dump (raw JSON) I/O
# --------------------------------------------------------------------------


def read_dump(path: Path) -> dict:
    with zstd.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_dump(path: Path, content: str, level: int = COMPRESSION_LEVEL) -> None:
    with zstd.open(path, "wt", level=level) as f:
        f.write(content)
    ratio = len(content) / path.stat().st_size
    print(f"Dumped {path} (compression ratio {ratio:.1f}x)")


def fetch_json(url: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")


def fetch_and_dump(kind: KindSpec, dumps_dir: Path) -> tuple[str, dict]:
    """Fetch a kind's latest 30d data from the API, dump it, and return it."""

    url = f"https://formulae.brew.sh/api/analytics/{kind.stub}/30d.json"
    print(f"Fetching: {kind.name} data ----------------")
    content = fetch_json(url)
    data = json.loads(content)
    end_date = data.get("end_date", "")
    date_str = end_date.replace("-", "")
    print(f"Downloaded {len(content)} bytes for {end_date}")

    path = dump_path(dumps_dir, date_str, kind.name)
    if path.exists():
        print(f"{path} already exists, not overwriting.")
    else:
        write_dump(path, content)
    return date_str, data


def get_today_data(kind: KindSpec, dumps_dir: Path) -> tuple[str, dict]:
    """Reuse today's dump if we already have it, otherwise fetch it."""
    today_str = date.today().strftime("%Y%m%d")
    path = dump_path(dumps_dir, today_str, kind.name)
    if path.exists():
        print(f"{path} already exists.")
        return today_str, read_dump(path)
    return fetch_and_dump(kind, dumps_dir)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_items(data: dict, kind: KindSpec, min_installs: int) -> Iterator[Record]:
    for item in data["items"]:
        name = item[kind.name]
        count = parse_num(item["count"])
        if count >= min_installs:
            yield (kind.kind_id, name, count)


# --------------------------------------------------------------------------
# SQLite layer
# --------------------------------------------------------------------------


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS names (
            name_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind_id INTEGER NOT NULL,
            name    TEXT    NOT NULL,
            UNIQUE (kind_id, name)
        );

        CREATE TABLE IF NOT EXISTS counts (
            date    TEXT    NOT NULL,
            kind_id INTEGER NOT NULL,
            name_id INTEGER NOT NULL REFERENCES names (name_id),
            count   INTEGER NOT NULL,
            UNIQUE (date, kind_id, name_id)
        );

        CREATE INDEX IF NOT EXISTS idx_counts_name_id ON counts (name_id);
        """
    )
    conn.commit()


def load_name_map(conn: sqlite3.Connection) -> dict[tuple[int, str], int]:
    rows = conn.execute("SELECT name_id, kind_id, name FROM names")
    return {(kind_id, name): name_id for name_id, kind_id, name in rows}


def ensure_names(
    conn: sqlite3.Connection,
    records: Iterable[Record],
    name_map: dict[tuple[int, str], int],
) -> dict[tuple[int, str], int]:
    """Make sure every (kind_id, name) in records has a row in `names`.

    Mutates and returns name_map with any newly created ids added.
    """
    missing = {
        (kind_id, name)
        for kind_id, name, _ in records
        if (kind_id, name) not in name_map
    }
    if not missing:
        return name_map

    conn.executemany(
        "INSERT OR IGNORE INTO names (kind_id, name) VALUES (?, ?)",
        sorted(missing),
    )
    rows = conn.execute(
        "SELECT name_id, kind_id, name FROM names WHERE (kind_id, name) IN "
        f"({', '.join('(?, ?)' for _ in missing)})",
        [value for pair in missing for value in pair],
    )
    for name_id, kind_id, name in rows:
        name_map[(kind_id, name)] = name_id
    return name_map


def upsert_counts(
    conn: sqlite3.Connection,
    date_str: str,
    records: Iterable[Record],
    name_map: dict[tuple[int, str], int],
) -> int:
    rows = [
        (date_str, kind_id, name_map[(kind_id, name)], count)
        for kind_id, name, count in records
    ]
    conn.executemany(
        """
        INSERT INTO counts (date, kind_id, name_id, count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (date, kind_id, name_id) DO UPDATE SET count = excluded.count
        """,
        rows,
    )
    return len(rows)


def store_records(
    conn: sqlite3.Connection,
    date_str: str,
    records: list[Record],
    name_map: dict[tuple[int, str], int],
) -> dict[tuple[int, str], int]:
    """Ensure names exist, upsert counts, and commit. Returns updated name_map."""
    name_map = ensure_names(conn, records, name_map)
    n = upsert_counts(conn, date_str, records, name_map)
    conn.commit()
    return name_map, n


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------


def run_fetch(conn: sqlite3.Connection, dumps_dir: Path, min_installs: int) -> None:
    """Normal run: fetch (or reuse) today's dumps and upsert them."""
    name_map = load_name_map(conn)
    for kind in KIND_SPECS:
        date_str, data = get_today_data(kind, dumps_dir)
        records = list(parse_items(data, kind, min_installs))
        name_map, n = store_records(conn, date_str, records, name_map)
        print(f"{kind.name}: upserted {n} rows for {date_str}")


def run_backfill(conn: sqlite3.Connection, dumps_dir: Path, min_installs: int) -> None:
    """Rebuild counts from every dump already on disk, no network calls."""
    name_map = load_name_map(conn)
    for kind in KIND_SPECS:
        paths = sorted(dumps_dir.glob(f"*{kind.name}.zst"))
        if not paths:
            print(f"No dumps found for {kind.name}.")
            continue
        for path in paths:
            date_str = path.name.removesuffix(f"{kind.name}.zst")
            data = read_dump(path)
            records = list(parse_items(data, kind, min_installs))
            name_map, n = store_records(conn, date_str, records, name_map)
            print(f"[backfill] {path.name}: upserted {n} rows for {date_str}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Homebrew analytics into SQLite."
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Rebuild counts from every existing dump instead of fetching new data.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH.name}).",
    )
    parser.add_argument(
        "--min-installs",
        type=int,
        default=MIN_INSTALLS,
        help=f"Minimum install count to keep a row (default: {MIN_INSTALLS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    try:
        init_db(conn)
        if args.backfill:
            run_backfill(conn, DUMPS_DIR, args.min_installs)
        else:
            run_fetch(conn, DUMPS_DIR, args.min_installs)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
