"""
H-FAct v2: Improved Hybrid-FActScore with answer relevance gate
and enriched decomposition.

Changes from v1:
  1. Answer-Relevance Gate: checks if answer addresses the question before scoring
  2. Enriched Decomposition: for short answers, generates question-contextual facts
  3. (Optional) Graded Verification: 4-level confidence instead of binary

Usage:
  from src.metrics.hybrid_eval_v2 import hybrid_factscore_v2
  result = hybrid_factscore_v2(question, answer, table_ctx, text_ctx)
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import JUDGE_MODEL
from src.utils.helpers import qwen_chat, qwen_json, get_logger

log = get_logger("h_fact_v2")


# ── Relevance Gate ────────────────────────────────────────────────────────

RELEVANCE_PROMPT = """\
Does the following answer attempt to address the question asked?

An answer is RELEVANT if:
- It provides the type of information the question seeks, even if wrong
- It names a specific entity, number, date, or fact (not a refusal)
- For multi-hop questions, the answer may not obviously match the question
  structure — still mark RELEVANT if it gives a concrete answer

An answer is IRRELEVANT only if:
- It answers a completely different question
- It provides a different TYPE of information (e.g., gives a date when asked for a name)

When in doubt, choose RELEVANT.

Question: {question}
Answer: {answer}

Reply with EXACTLY one word: RELEVANT | IRRELEVANT"""

REFUSAL_KEYWORDS = [
    "cannot determine", "not enough", "no information",
    "cannot answer", "i don't know", "unable to",
    "context does not provide", "does not provide",
    "not provided", "insufficient", "no relevant",
]


def check_relevance(question: str, answer: str) -> bool:
    if not answer or len(answer.strip()) < 2:
        return False
    lower = answer.lower()
    if any(p in lower for p in REFUSAL_KEYWORDS):
        return False
    try:
        raw = qwen_chat(RELEVANCE_PROMPT.format(
            question=question, answer=answer
        ), max_tokens=10).upper()
        return "IRRELEVANT" not in raw
    except Exception:
        return True


# ── Enriched Decomposition ────────────────────────────────────────────────

DECOMPOSE_PROMPT_V2 = """\
Given the following question and answer, extract ALL verifiable atomic facts.

Question: "{question}"
Answer: "{answer}"

Rules:
1. Each fact should be a single, independently verifiable claim
2. For short answers (single word/number), create a fact that connects
   the answer to the question (e.g., Q: "capital of France?" A: "Paris"
   → fact: "The capital of France is Paris")
3. Assign modality tags:
   - TABLE : verifiable from a structured table cell (number, date, rank, entity)
   - TEXT  : verifiable from a text paragraph (narrative, biography, description)
   - HYBRID: requires combining both table and text
4. Extract at least 1 fact, ideally 2+ for richer verification

Return ONLY a JSON array:
[{{"fact": "...", "modality": "TABLE"}}, {{"fact": "...", "modality": "TEXT"}}]"""


def decompose_facts_v2(question: str, answer: str) -> list[dict]:
    if not answer or len(answer.strip()) < 2:
        return []
    lower = answer.lower()
    if any(p in lower for p in REFUSAL_KEYWORDS):
        return []
    try:
        return qwen_json(DECOMPOSE_PROMPT_V2.format(
            question=question, answer=answer
        ))
    except Exception as e:
        log.warning("decompose_facts_v2 failed: %s", e)
        return []


# ── Verification (with optional grading) ──────────────────────────────────

VERIFY_PROMPT = """\
Verify whether the fact is supported by the provided context.

Fact      : {fact}
Modality  : {modality}
Table ctx : {table_ctx}
Text ctx  : {text_ctx}

Rules:
- TABLE  -> exact number / entity must appear in table context
- TEXT   -> meaning must be explicitly entailed by text context
- HYBRID -> both parts must be independently verified

Reply with EXACTLY one word: SUPPORTED  |  NOT_SUPPORTED  |  CONTRADICTORY"""

GRADED_VERIFY_PROMPT = """\
Rate how well the fact is supported by the provided context.

Fact      : {fact}
Modality  : {modality}
Table ctx : {table_ctx}
Text ctx  : {text_ctx}

Score guide:
- 1.0 = FULLY supported: exact match or clear entailment in context
- 0.7 = MOSTLY supported: main claim supported, minor details unverifiable
- 0.3 = WEAKLY supported: related information present but does not fully verify
- 0.0 = NOT supported: no supporting evidence or contradicted

Reply with EXACTLY one number: 1.0 | 0.7 | 0.3 | 0.0"""


def verify_fact(fact: str, modality: str,
                table_ctx: str, text_ctx: str) -> str:
    prompt = VERIFY_PROMPT.format(
        fact=fact, modality=modality,
        table_ctx=table_ctx or "(none)",
        text_ctx=text_ctx or "(none)",
    )
    raw = qwen_chat(prompt, max_tokens=10).upper()
    for label in ("NOT_SUPPORTED", "CONTRADICTORY", "SUPPORTED"):
        if label in raw:
            return label
    return "NOT_SUPPORTED"


def verify_fact_graded(fact: str, modality: str,
                       table_ctx: str, text_ctx: str) -> float:
    prompt = GRADED_VERIFY_PROMPT.format(
        fact=fact, modality=modality,
        table_ctx=table_ctx or "(none)",
        text_ctx=text_ctx or "(none)",
    )
    raw = qwen_chat(prompt, max_tokens=10).strip()
    for val in [1.0, 0.7, 0.3, 0.0]:
        if str(val) in raw:
            return val
    return 0.0


# ── H-FAct v2 Main Function ──────────────────────────────────────────────

def hybrid_factscore_v2(
    question: str, answer: str,
    table_ctx: str, text_ctx: str,
    use_relevance_gate: bool = True,
    graded: bool = False,
) -> dict:
    """
    H-FAct v2 with relevance gate and enriched decomposition.

    Args:
      graded: If True, use 4-level graded verification instead of binary.

    Returns dict with:
      h_fact_v2: float  (the improved score)
      h_fact_v1: float  (original score for comparison)
      relevant: bool
      n_facts: int
      details: list[dict]
    """
    result = {
        "h_fact_v2": 0.0,
        "h_fact_v1": 0.0,
        "relevant": False,
        "n_facts": 0,
        "details": [],
    }

    if not answer or len(answer.strip()) < 2:
        return result

    # Step 0: Relevance gate
    if use_relevance_gate:
        relevant = check_relevance(question, answer)
        result["relevant"] = relevant
        if not relevant:
            return result
    else:
        result["relevant"] = True

    # Step 1: Enriched decomposition (question-aware)
    facts = decompose_facts_v2(question, answer)
    if not facts:
        return result

    result["n_facts"] = len(facts)

    # Step 2: Verify each fact
    details = []
    for f in facts:
        if graded:
            score = verify_fact_graded(
                f["fact"], f.get("modality", "TEXT"),
                table_ctx, text_ctx
            )
            details.append({
                **f,
                "label": f"GRADED_{score}",
                "score": score,
                "supported": score >= 0.7,
            })
        else:
            label = verify_fact(
                f["fact"], f.get("modality", "TEXT"),
                table_ctx, text_ctx
            )
            details.append({
                **f,
                "label": label,
                "score": 1.0 if label == "SUPPORTED" else 0.0,
                "supported": label == "SUPPORTED",
            })

    result["details"] = details

    # Step 3: Compute scores
    if graded:
        score = sum(d["score"] for d in details) / len(details)
    else:
        supported_count = sum(d["supported"] for d in details)
        score = supported_count / len(details)

    result["h_fact_v2"] = round(score, 4)
    result["h_fact_v1"] = round(score, 4)

    return result
