"""
Perturbation dataset construction (Section 3.4 / MiRAGE-style).

Five conditions built from HybridQA dev set:
  golden          – correct table row  + correct passage
  semi_golden     – correct table row  + random passage
  noise           – random table row   + random passage
  counterfactual  – modified table row (key value mutated) + correct passage
  missing         – safe table row     + safe passage (answer entity absent)
"""
from __future__ import annotations
import json, copy, random, re, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DIR, PERT_DIR, RANDOM_SEED
from src.pipeline.rag import table_to_rows, _cell_links, _cell_value

random.seed(RANDOM_SEED)


# ── Loaders ────────────────────────────────────────────────────────────────
def load_dev(path: Path | None = None) -> list[dict]:
    p = path or RAW_DIR / "dev.jsonl"
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_tables(path: Path | None = None) -> dict:
    p = path or RAW_DIR / "tables.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_passages(path: Path | None = None) -> dict:
    p = path or RAW_DIR / "passages.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_gold_table_row(item: dict, tables_db: dict) -> str:
    """Return the single gold table row as a string."""
    table_id = item.get("table_id", "")
    table    = tables_db.get(table_id, {})
    rows     = table_to_rows(table)
    # HybridQA stores answer_node with row index for table answers
    nodes = item.get("answer_node", [])
    for node in nodes:
        if len(node) > 3 and node[3] == "table" and rows:
            row_idx = node[1][0] if isinstance(node[1], list) else 0
            if 0 <= row_idx < len(rows):
                return rows[row_idx]
    # fallback: BM25 best row
    if rows:
        return rows[0]
    return "(no table row found)"


def _get_gold_passage(item: dict, passages_db: dict, tables_db: dict) -> str:
    """Return the gold passage sentence as a string."""
    nodes = item.get("answer_node", [])
    for node in nodes:
        if len(node) > 3 and node[3] == "passage":
            pid   = node[2]   # wiki URL is the passages_db key
            sents = passages_db.get(str(pid), [])
            if isinstance(sents, str):
                sents = [sents]
            if sents:
                # node[1] is the sentence index for passage-type nodes
                sent_idx = node[1] if isinstance(node[1], int) else 0
                if 0 <= sent_idx < len(sents):
                    return sents[sent_idx]
                return sents[0]
    # fallback: first linked passage
    table    = tables_db.get(item.get("table_id", ""), {})
    for row in table.get("rows", []):
        for cell in row:
            for link in _cell_links(cell):
                sents = passages_db.get(str(link), [])
                if isinstance(sents, str):
                    sents = [sents]
                if sents:
                    return sents[0]
    return "(no passage found)"


def _random_row(tables_db: dict, exclude_id: str) -> str:
    tids = [tid for tid in tables_db if tid != exclude_id]
    if not tids:
        return "(random row)"
    tid = random.choice(tids)
    rows = table_to_rows(tables_db[tid])
    return random.choice(rows) if rows else "(random row)"


def _random_passage(passages_db: dict) -> str:
    pid = random.choice(list(passages_db.keys()))
    sents = passages_db[pid]
    if isinstance(sents, str):
        return sents
    return random.choice(sents) if sents else "(random passage)"


def _contains_answer(text: str, answer: str) -> bool:
    return answer.lower() in text.lower()


def _safe_row(tables_db: dict, answer: str, exclude_id: str) -> str:
    if not answer:
        return _random_row(tables_db, exclude_id)
    for _ in range(50):
        row = _random_row(tables_db, exclude_id)
        if not _contains_answer(row, answer):
            return row
    return _random_row(tables_db, exclude_id)


def _safe_passage(passages_db: dict, answer: str) -> str:
    if not answer:
        return _random_passage(passages_db)
    for _ in range(50):
        p = _random_passage(passages_db)
        if not _contains_answer(p, answer):
            return p
    return _random_passage(passages_db)


def _mutate_value(val: str) -> str:
    """Mutate a cell value: increment numbers, prepend 'Modified' for strings."""
    m = re.search(r"\d+\.?\d*", val)
    if m:
        num = float(m.group())
        new_num = int(num + 1) if num == int(num) else round(num * 1.1, 2)
        return val[:m.start()] + str(new_num) + val[m.end():]
    return "Modified_" + val


def _counterfactual_row(item: dict, tables_db: dict) -> str:
    """Swap the answer value in the gold row for a counterfactual."""
    table_id = item.get("table_id", "")
    table    = copy.deepcopy(tables_db.get(table_id, {}))
    answer   = str(item.get("answer_text", ""))
    headers  = table.get("header", [])
    rows_raw = table.get("rows", [])

    nodes = item.get("answer_node", [])
    for node in nodes:
        if len(node) > 3 and node[3] == "table":
            row_idx = node[1][0] if isinstance(node[1], list) else 0
            col_idx = node[1][1] if isinstance(node[1], list) and len(node[1]) > 1 else 0
            if 0 <= row_idx < len(rows_raw) and 0 <= col_idx < len(rows_raw[row_idx]):
                cell = rows_raw[row_idx][col_idx]
                if isinstance(cell, list):
                    rows_raw[row_idx][col_idx][0] = _mutate_value(str(cell[0]))
                elif isinstance(cell, dict):
                    cell["value"] = _mutate_value(str(cell.get("value", "")))
                else:
                    rows_raw[row_idx][col_idx] = _mutate_value(str(cell))
                # rebuild string — normalize headers to str, cells via _cell_value
                new_cells = [
                    f"{_cell_value(h)}: {_cell_value(c)}"
                    for h, c in zip(headers, rows_raw[row_idx])
                ]
                return " | ".join(new_cells)
    # fallback: just mutate something in the gold row
    gold = _get_gold_table_row(item, tables_db)
    return _mutate_value(gold) if gold else "(counterfactual row)"


# ── Main builder ───────────────────────────────────────────────────────────
def build_perturbed(dev: list[dict], tables_db: dict, passages_db: dict) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "golden": [], "semi_golden": [], "noise": [],
        "counterfactual": [], "missing": [],
    }

    for item in tqdm(dev, desc="Building perturbations"):
        answer   = str(item.get("answer_text", ""))
        table_id = item.get("table_id", "")

        gold_row  = _get_gold_table_row(item, tables_db)
        gold_pass = _get_gold_passage(item, passages_db, tables_db)
        rand_row  = _random_row(tables_db, table_id)
        rand_pass = _random_passage(passages_db)
        cf_row    = _counterfactual_row(item, tables_db)
        safe_row  = _safe_row(tables_db, answer, table_id)
        safe_pass = _safe_passage(passages_db, answer)

        base = {k: item[k] for k in item}

        buckets["golden"].append({
            **base,
            "condition": "golden",
            "force_table_ctx": gold_row,
            "force_text_ctx":  gold_pass,
        })
        buckets["semi_golden"].append({
            **base,
            "condition": "semi_golden",
            "force_table_ctx": gold_row,
            "force_text_ctx":  rand_pass,
        })
        buckets["noise"].append({
            **base,
            "condition": "noise",
            "force_table_ctx": rand_row,
            "force_text_ctx":  rand_pass,
        })
        buckets["counterfactual"].append({
            **base,
            "condition": "counterfactual",
            "force_table_ctx": cf_row,
            "force_text_ctx":  gold_pass,
            "expected_behavior": "CONTRADICT",
        })
        buckets["missing"].append({
            **base,
            "condition": "missing",
            "force_table_ctx": safe_row,
            "force_text_ctx":  safe_pass,
            "expected_behavior": "REFUSE",
        })

    return buckets


def save_perturbed(buckets: dict[str, list[dict]]):
    PERT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        dest = PERT_DIR / f"{name}.jsonl"
        with open(dest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(rows)} rows → {dest}")
