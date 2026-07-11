"""
Step 1 - Download and cache HybridQA data from the official GitHub repository.

Official source:
  https://github.com/wenhuchen/HybridQA

Saves:
  data/raw/dev.jsonl      - dev split
  data/raw/train.jsonl    - train split
  data/raw/tables.json    - Wikipedia tables keyed by table_id
  data/raw/passages.json  - Wikipedia passages keyed by passage_id

Notes:
  - The public GitHub repository provides the released QA splits and passage file.
  - Large table or passage files may need to be downloaded manually from the
    official repository or its released/preprocessed data links if a raw URL is
    unavailable.
  - Raw dataset files are intentionally excluded from the public thesis code
    package. Place local raw files under data/raw/ before running the full
    pipeline.
"""

import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

OFFICIAL_REPO = "https://github.com/wenhuchen/HybridQA"
RAW_BASE = "https://raw.githubusercontent.com/wenhuchen/HybridQA/master"
RELEASED_BASE = f"{RAW_BASE}/released_data"


def fetch(url: str, dest: Path, label: str = "") -> bool:
    """Download a file if it does not already exist."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    print(f"  Downloading {label or dest.name} ...")
    try:
        response = requests.get(url, timeout=180, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"  Saved {dest.stat().st_size // 1024:,} KB -> {dest}")
        return True
    except Exception as exc:
        print(f"  ERROR downloading {label or dest.name}: {exc}")
        print(f"  Source attempted: {url}")
        print(f"  If this file is unavailable from the raw URL, download it manually from {OFFICIAL_REPO}.")
        return False


def json_to_jsonl(src: Path, dst: Path) -> None:
    """Convert a JSON array or dict file to one JSON object per line."""
    if dst.exists():
        print(f"  [skip] {dst.name} already exists")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    rows = list(data.values()) if isinstance(data, dict) else data

    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Converted {len(rows)} rows -> {dst}")


def download_split(split: str) -> None:
    """Download a HybridQA split and convert it to JSONL."""
    tmp = RAW_DIR / f"_{split}.json"
    final = RAW_DIR / f"{split}.jsonl"

    if final.exists():
        print(f"  [skip] {final.name} already exists")
        return

    url = f"{RELEASED_BASE}/{split}.json"
    if fetch(url, tmp, split):
        json_to_jsonl(tmp, final)
        tmp.unlink(missing_ok=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Download HybridQA data ===")
    print(f"Official source: {OFFICIAL_REPO}")

    for split in ("dev", "train"):
        download_split(split)

    fetch(f"{RAW_BASE}/data/tables.json", RAW_DIR / "tables.json", "tables")
    fetch(f"{RELEASED_BASE}/all_passages.json", RAW_DIR / "passages.json", "passages")

    print("\nData folder status:", RAW_DIR)
    for path in sorted(RAW_DIR.iterdir()):
        if path.is_file():
            print(f"    {path.name:30s}  {path.stat().st_size // 1024:>8,} KB")


if __name__ == "__main__":
    main()
