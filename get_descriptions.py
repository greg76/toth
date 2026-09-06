import json
import urllib.request

from compression import zstd

URLS = {
    "formulae": "https://formulae.brew.sh/api/formula.json",
    "casks": "https://formulae.brew.sh/api/cask.json",
}

descriptions = {}

for package_type, url in URLS.items():
    print(f"Downloading {package_type}...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Python Script)"}
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

        for item in data:
            raw_name = item.get("token") or item.get("name")
            desc = item.get("desc")

            if isinstance(raw_name, list) and raw_name:
                name = raw_name[0]
            else:
                name = raw_name

            if name and desc:
                descriptions[name] = desc

# Convert dictionary to JSON byte string
json_bytes = json.dumps(descriptions, indent=2, ensure_ascii=False).encode("utf-8")

# Compress using Zstandard
cctx = zstd.ZstdCompressor(level=6)
compressed_bytes = cctx.compress(json_bytes)

# Write binary compressed data to file
output_filename = "descriptions.json.zst"

with zstd.open(output_filename, "wb", level=6) as f:
    json_bytes = json.dumps(descriptions, indent=2, ensure_ascii=False).encode("utf-8")
    f.write(json_bytes)

print(f"Done! Saved {len(descriptions):,} package descriptions to {output_filename}.")
