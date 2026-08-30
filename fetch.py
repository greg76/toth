#! python3

import json
import urllib.request
from datetime import date
from pathlib import Path

from compression import zstd

SCRIPT_DIR = Path(__file__).resolve().parent
DUMPS_DIR = SCRIPT_DIR / "dumps"
DUMPS_DIR.mkdir(parents=True, exist_ok=True)

COMPRESSION_LEVEL = 6

URL_STUBS = [(0, "formula", "install-on-request"), (1, "cask", "cask-install")]

MIN_INSTALLS = 500


def parse_num(formatted_string: str) -> int:
    return int(formatted_string.replace(",", ""))


def main():
    today = date.today().strftime("%Y%m%d")

    merged: list[tuple[int, str, int]] = []

    for kind_id, kind, stub in URL_STUBS:
        file_to_check = DUMPS_DIR / f"{today}{kind}.zst"
        if file_to_check.exists():
            print(f"{file_to_check} already exists.")
            with zstd.open(file_to_check, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            url = f"https://formulae.brew.sh/api/analytics/{stub}/30d.json"
            req = urllib.request.Request(url)
            print(f"Fetching: {kind} data ----------------")
            with urllib.request.urlopen(req) as response:
                content = response.read().decode("utf-8")
                data = json.loads(content)
                end_date = data["end_date"] if "end_date" in data else ""
                print(f"Downloaded {len(content)} bytes for {end_date}")
                dump_name = f"{end_date.replace('-', '')}{kind}.zst"
                dump_path = DUMPS_DIR / dump_name
                print(f"Dumping {dump_path}")
                with zstd.open(dump_path, "wt", level=COMPRESSION_LEVEL) as f:
                    f.write(content)
                    print(
                        f"Compression ratio: {len(content) / dump_path.stat().st_size:.1f}x"
                    )
        for item in data["items"]:
            name, count = item[kind], parse_num(item["count"])
            if count >= MIN_INSTALLS:
                merged.append((kind_id, name, count))

    print(f"{len(merged)} items in total.")

    merged.sort(key=lambda item: item[2], reverse=True)


if __name__ == "__main__":
    main()
