"""
RAG Pipeline: Qwen-based retrieval + generation over HybridQA.

For each question the pipeline:
  1. Retrieves the top-K table rows via BM25 (rank_bm25)
  2. Retrieves the top-K passage sentences via BM25
  3. Prompts Qwen to answer using the retrieved context
  4. Returns the answer + retrieved evidence for metric computation
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RETRIEVAL_TOP_K, RAG_MODEL
from src.utils.helpers import qwen_chat, get_logger

log = get_logger("pipeline")


# ── Cell format helpers (handle both list and dict cells) ──────────────────
def _cell_value(c) -> str:
    """Extract text value from a cell (dict, list, or str)."""
    if isinstance(c, dict):
        return str(c.get("value", ""))
    elif isinstance(c, list):
        return str(c[0]) if len(c) > 0 else ""
    return str(c)


def _cell_links(c) -> list:
    """Extract link URLs from a cell (dict, list, or str)."""
    if isinstance(c, dict):
        return c.get("links", [])
    elif isinstance(c, list) and len(c) > 1:
        links = c[1]
        return links if isinstance(links, list) else [links]
    return []

# ── BM25 retrieval ─────────────────────────────────────────────────────────
def _build_bm25(corpus: list[str]):
    from rank_bm25 import BM25Okapi
    tokenised = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenised)


def retrieve_top_k(query: str, corpus: list[str], k: int = RETRIEVAL_TOP_K) -> list[tuple[int, str]]:
    """Return list of (index, text) for top-k BM25 hits."""
    if not corpus:
        return []
    bm25 = _build_bm25(corpus)
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, corpus[i]) for i in top_idx]


# ── Table helpers ──────────────────────────────────────────────────────────
def table_to_rows(table: dict) -> list[str]:
    """Flatten each table row to a readable string."""
    headers = table.get("header", [])
    rows = []
    for row in table.get("rows", []):
        cells = [f"{h}: {_cell_value(c)}" for h, c in zip(headers, row)]
        rows.append(" | ".join(cells))
    return rows


def linearise_table(table: dict) -> str:
    headers = " | ".join(table.get("header", []))
    rows = "\n".join(table_to_rows(table))
    return f"[TABLE]\nHeaders: {headers}\n{rows}"


# ── Answer generation ──────────────────────────────────────────────────────
ANSWER_PROMPT = """\
You are a precise question-answering assistant. Use ONLY the provided context to answer.
If the context does not contain enough information, reply exactly: "I cannot determine this from the provided context."

Question: {question}

Retrieved Table Context:
{table_ctx}

Retrieved Text Context:
{text_ctx}

Instructions:
- Answer as briefly as possible (a word, phrase, or short sentence).
- Do NOT speculate or use outside knowledge.
- If you use information from the table, start your answer with [TABLE].
- If you use information from the text, start your answer with [TEXT].
- If you use both, start with [HYBRID].

Answer:"""


def answer_question(
    question: str,
    table_rows: list[str],   # retrieved rows (strings)
    passages: list[str],      # retrieved passage sentences
    model: str = RAG_MODEL,
) -> dict:
    table_ctx = "\n".join(table_rows) if table_rows else "(no table context retrieved)"
    text_ctx  = "\n".join(passages)  if passages  else "(no text context retrieved)"

    prompt = ANSWER_PROMPT.format(
        question=question,
        table_ctx=table_ctx,
        text_ctx=text_ctx,
    )
    raw = qwen_chat(prompt, model=model, max_tokens=128)

    # parse source tag
    source = "unknown"
    for tag in ("[TABLE]", "[TEXT]", "[HYBRID]"):
        if raw.startswith(tag):
            source = tag[1:-1].lower()
            raw = raw[len(tag):].strip()
            break

    return {
        "answer": raw,
        "source": source,
        "table_ctx": table_ctx,
        "text_ctx": text_ctx,
    }


# ── Full pipeline for one sample ───────────────────────────────────────────
def run_pipeline(
    item: dict,
    tables_db: dict,
    passages_db: dict,
    k: int = RETRIEVAL_TOP_K,
    force_table_ctx: Optional[str] = None,   # override for perturbation
    force_text_ctx: Optional[str]  = None,
) -> dict:
    """
    item  – one row from dev.jsonl
    Returns item enriched with pipeline outputs.
    """
    question = item["question"]
    table_id = item.get("table_id", "")

    # ── Table retrieval ────────────────────────────────────────────────────
    if force_table_ctx is not None:
        table_rows = [force_table_ctx]
        table_hits = []
    else:
        table = tables_db.get(table_id, {})
        all_rows = table_to_rows(table)
        hits = retrieve_top_k(question, all_rows, k)
        table_rows = [h[1] for h in hits]
        table_hits = [h[0] for h in hits]

    # ── Passage retrieval ──────────────────────────────────────────────────
    if force_text_ctx is not None:
        pass_sents = [force_text_ctx]
        pass_hits  = []
    else:
        # Gather all passages linked from this table's cells
        table = tables_db.get(table_id, {})
        linked_ids = []
        for row in table.get("rows", []):
            for cell in row:
                for link in _cell_links(cell):
                    linked_ids.append(link)
        corpus = []
        pid_map = []
        for pid in linked_ids:
            sents = passages_db.get(str(pid), [])
            if isinstance(sents, str):
                sents = [sents]
            for s in sents:
                corpus.append(s)
                pid_map.append(pid)
        hits = retrieve_top_k(question, corpus, k)
        pass_sents = [h[1] for h in hits]
        pass_hits  = [pid_map[h[0]] for h in hits]

    # ── Generate answer ────────────────────────────────────────────────────
    out = answer_question(question, table_rows, pass_sents)

    return {
        **item,
        "pred_answer": out["answer"],
        "pred_source": out["source"],
        "retrieved_table_rows": table_rows,
        "retrieved_table_row_ids": table_hits,
        "retrieved_passages": pass_sents,
        "retrieved_passage_ids": pass_hits,
        "context_table": out["table_ctx"],
        "context_text":  out["text_ctx"],
    }
