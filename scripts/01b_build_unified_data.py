"""
Step 1b – Merge WikiTables + request_tok + QA splits into unified flat files.

Input  (already downloaded):
  data/raw/dev.json               – QA dev split (list, with answer-node)
  data/raw/train.json             – QA train split
  data/raw/wikitables_repo/
      tables_tok/{table_id}.json  – table structure
      request_tok/{table_id}.json – {wiki_url: passage_text} mapping

Output:
  data/raw/tables.json            – {table_id: {header, rows, ...}}
  data/raw/passages.json          – {wiki_url: [sentences]}
  data/raw/dev.jsonl              – dev as JSONL (unified field names)
  data/raw/train.jsonl            – train as JSONL
"""
import json, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

REPO     = RAW_DIR / "wikitables_repo"
TABLES_D = REPO / "tables_tok"
REQ_D    = REPO / "request_tok"


def build_tables_and_passages():
    tables_out   = {}
    passages_out = {}

    # Enumerate all table files we have
    table_files = list(TABLES_D.glob("*.json"))
    print(f"  Found {len(table_files)} table files in tables_tok/")

    for tf in tqdm(table_files, desc="Building tables+passages"):
        tid = tf.stem
        try:
            table = json.loads(tf.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Normalise header: [["colname", ...], ...] → ["colname", ...]
        header = table.get("header", [])
        if header and isinstance(header[0], list):
            header = [h[0] for h in header]

        # Normalise rows: each row is list of [value, [urls]]
        raw_rows = table.get("data", [])
        rows_norm = []
        for row in raw_rows:
            cells = []
            for cell in row:
                if isinstance(cell, list):
                    val   = cell[0] if len(cell) > 0 else ""
                    links = cell[1] if len(cell) > 1 else []
                else:
                    val   = str(cell)
                    links = []
                cells.append({"value": str(val), "links": links})
            rows_norm.append(cells)

        tables_out[tid] = {
            "header": header,
            "rows":   rows_norm,
            "title":  table.get("title", ""),
            "url":    table.get("url",   ""),
        }

        # request_tok: {url: passage_text_or_list}
        req_path = REQ_D / f"{tid}.json"
        if req_path.exists():
            try:
                req = json.loads(req_path.read_text(encoding="utf-8"))
                for url, text in req.items():
                    if url not in passages_out:
                        if isinstance(text, list):
                            passages_out[url] = text
                        elif isinstance(text, str):
                            # split into sentences naively
                            passages_out[url] = [s.strip() for s in text.split(". ") if s.strip()]
                        else:
                            passages_out[url] = [str(text)]
            except Exception:
                pass

    print(f"  tables: {len(tables_out)}   passages: {len(passages_out)}")
    return tables_out, passages_out


def normalise_qa(split: str, data: list) -> list:
    """Unify field names across raw HybridQA JSON."""
    out = []
    for r in data:
        out.append({
            "question_id":   r.get("question_id", ""),
            "question":      r.get("question", ""),
            "table_id":      r.get("table_id", ""),
            "answer_text":   r.get("answer-text", r.get("answer_text", "")),
            "question_postag": r.get("question_postag", ""),
            "answer_node":   r.get("answer-node", []),
        })
    return out


def main():
    print("=== Build unified data files ===")

    # QA splits
    for split in ("dev", "train"):
        src  = RAW_DIR / f"{split}.json"
        dest = RAW_DIR / f"{split}.jsonl"
        if dest.exists():
            print(f"  [skip] {dest.name}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        rows = normalise_qa(split, data)
        with open(dest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {dest.name}: {len(rows)} rows")

    # Tables + Passages
    tables_dest   = RAW_DIR / "tables.json"
    passages_dest = RAW_DIR / "passages.json"

    if tables_dest.exists() and passages_dest.exists():
        print("  [skip] tables.json and passages.json already exist")
    else:
        tables, passages = build_tables_and_passages()
        tables_dest.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
        passages_dest.write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved tables.json ({tables_dest.stat().st_size//1024:,} KB)")
        print(f"  Saved passages.json ({passages_dest.stat().st_size//1024:,} KB)")

    print("\n✓ All data ready:")
    for p in sorted(RAW_DIR.iterdir()):
        if p.is_file():
            print(f"    {p.name:35s}  {p.stat().st_size//1024:>8,} KB")


if __name__ == "__main__":
    main()
