"""
All Hybrid-Eval metrics.

Metric                  | Abbr   | Dimension
------------------------|--------|-----------
Exact Match             | EM     | Generation (baseline)
Token F1                | F1     | Generation (baseline)
Joint Hit@K             | JHit@K | Retrieval
Evidence-Path Precision | EPP    | Retrieval
Hybrid-FActScore        | H-FAct | Generation
Modality-Attribution    | MAA    | Reasoning
Complexity-Weighted RS  | CWRS   | Reasoning
Hallucination Rate      | HR-P   | Robustness

References:
  - FActScore (Min et al., EMNLP 2023): Atomic fact decomposition + verification paradigm
  - SAFE (Wei et al., arXiv 2024): Decompose-then-verify paradigm (we borrow the structure;
      SAFE additionally uses search augmentation which we approximate via modality routing)
  - FEVEROUS (Aly et al., EMNLP 2021): Evidence-path verification over structured + text;
      our EPP adapts this to a continuous token-F1 score rather than binary threshold
  - STARK (Wu et al., NeurIPS 2024): Complexity-stratified retrieval evaluation;
      our L1-L5 taxonomy is original to this project, inspired by STARK's relational complexity
  - MiRAGE (Park et al., NAACL 2025): Group-level RAG robustness metrics (NV, CA, CI, CM)
  - RGB (Chen et al., ACL 2024): Noise robustness + negative rejection testbed
  - G-Eval (Liu et al., EMNLP 2023): LLM-based NLG evaluation with CoT scoring
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RETRIEVAL_TOP_K, JUDGE_MODEL
from src.utils.helpers import qwen_chat, qwen_json, exact_match, token_f1, get_logger

log = get_logger("metrics")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Retrieval: Joint Hit@K
# ═══════════════════════════════════════════════════════════════════════════

def joint_hit_at_k(
    gold_table_row: str,
    gold_passage: str,
    retrieved_table_rows: list[str],
    retrieved_passages: list[str],
) -> dict:
    """1 iff BOTH gold table row AND gold passage appear in top-K retrieved."""
    def hit(gold: str, retrieved: list[str]) -> bool:
        gold_n = gold.lower().strip()
        return any(gold_n in r.lower() for r in retrieved)

    t_hit = hit(gold_table_row, retrieved_table_rows)
    p_hit = hit(gold_passage,   retrieved_passages)
    return {
        "table_hit": int(t_hit),
        "passage_hit": int(p_hit),
        "joint_hit": int(t_hit and p_hit),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Retrieval: Evidence-Path Precision (EPP) -- custom token-level F1
#    Inspired by the evidence-matching concept in FEVEROUS (Aly et al., NeurIPS 2021).
#    Note: FEVEROUS's actual metric is a joint threshold check (correct verdict AND
#    at least s gold evidence pieces retrieved) -- not a Jaccard/F1 measure.
#    Our EPP computes token-level Jaccard F1 over evidence piece sets, providing
#    a continuous coverage signal suitable for automated pipeline evaluation.
# ═══════════════════════════════════════════════════════════════════════════

def evidence_path_precision(
    gold_evidence: list[str],      # gold row/sentence strings
    pred_evidence: list[str],      # model retrieved strings
) -> dict:
    """F1 between predicted and gold evidence sets (token-level overlap proxy)."""
    if not gold_evidence or not pred_evidence:
        return {"epp_precision": 0.0, "epp_recall": 0.0, "epp_f1": 0.0}

    def best_overlap(query: str, pool: list[str]) -> float:
        q = query.lower()
        return max(
            (len(set(q.split()) & set(p.lower().split())) /
             max(len(set(q.split()) | set(p.lower().split())), 1))
            for p in pool
        )

    prec = sum(best_overlap(p, gold_evidence) for p in pred_evidence) / len(pred_evidence)
    rec  = sum(best_overlap(g, pred_evidence) for g in gold_evidence)  / len(gold_evidence)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"epp_precision": round(prec, 4),
            "epp_recall":    round(rec,  4),
            "epp_f1":        round(f1,   4)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Generation: Hybrid-FActScore (H-FAct)
#    Core innovation: modality-aware fact verification
#    Extends FActScore (Min et al., 2023) with modality routing inspired
#    by SAFE (Wei et al., 2024) decompose-then-verify paradigm
# ═══════════════════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """\
Break the following answer into atomic facts. For each fact assign a modality tag:
- TABLE : verifiable from a structured table cell (number, date, rank, entity)
- TEXT  : verifiable from a text paragraph (narrative, biography, description)
- HYBRID: requires combining both table and text

Answer: "{answer}"

Return ONLY a JSON array. Example:
[{{"fact": "...", "modality": "TABLE"}}, {{"fact": "...", "modality": "TEXT"}}]"""

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


def decompose_facts(answer: str) -> list[dict]:
    try:
        return qwen_json(DECOMPOSE_PROMPT.format(answer=answer))
    except Exception as e:
        log.warning("decompose_facts failed: %s", e)
        return []


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


def hybrid_factscore(answer: str, table_ctx: str, text_ctx: str) -> dict:
    facts = decompose_facts(answer)
    if not facts:
        return {"h_fact": 0.0, "n_facts": 0, "details": []}

    details = []
    for f in facts:
        label = verify_fact(f["fact"], f.get("modality", "TEXT"), table_ctx, text_ctx)
        details.append({**f, "label": label, "supported": label == "SUPPORTED"})

    score = sum(d["supported"] for d in details) / len(details)
    return {"h_fact": round(score, 4), "n_facts": len(details), "details": details}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Reasoning: Modality-Attribution Accuracy (MAA)
#    Design: two-stage blind inference — LLM infers modality without seeing gold

#    Stage 1: LLM infers source modality from answer + context (no gold label)
#    Stage 2: Compare inferred modality with gold_source
# ═══════════════════════════════════════════════════════════════════════════

MAA_INFER_PROMPT = """\
The answer below was generated from a hybrid (table + text) RAG system.
Determine which modality (TABLE or TEXT) the answer primarily relies on.

Question     : {question}
Answer       : {answer}
Table context: {table_ctx}
Text context : {text_ctx}

Analyse which source provides the key evidence for this answer.
Reply with EXACTLY one word: TABLE  |  TEXT  |  HYBRID"""

MAA_JUDGE_PROMPT = """\
Compare the predicted source attribution with the expected source.

Predicted source: {pred_source}
Expected source : {gold_source}

Are they consistent (both refer to the same modality)?
Reply with EXACTLY one word: CORRECT  |  INCORRECT"""


def modality_attribution_accuracy(
    question: str, answer: str,
    pred_source: str, gold_source: str,
    table_ctx: str = "", text_ctx: str = "",
) -> dict:
    """
    Two-stage MAA (blind inference design):
      Stage 1: LLM infers source modality from answer+context (blind to gold label)
      Stage 2: String-compare inferred modality with gold_source → primary maa score
    Also computes maa_llm (LLM judge) and maa_heuristic for diagnostic purposes.
    """
    # Heuristic baseline (diagnostic only)
    # Normalize: gold "passage" → "text" to align with TABLE/TEXT/HYBRID taxonomy
    gs = gold_source.lower()
    gs_norm = "text" if gs == "passage" else gs
    ps = pred_source.lower()
    heuristic_match = int(gs_norm in ps or ps in gs_norm)

    # Stage 1: LLM infers which modality the answer relies on
    try:
        inferred_raw = qwen_chat(
            MAA_INFER_PROMPT.format(
                question=question, answer=answer,
                table_ctx=table_ctx[:1500] or "(none)",
                text_ctx=text_ctx[:1500] or "(none)",
            ),
            max_tokens=10,
        ).upper()
        inferred = "unknown"
        for tag in ("TABLE", "TEXT", "HYBRID"):
            if tag in inferred_raw:
                inferred = tag.lower()
                break
    except Exception:
        inferred = pred_source.lower()

    # Stage 2: Compare inferred vs gold (use normalized gold label) → primary score
    inferred_match = int(gs_norm in inferred or inferred in gs_norm)

    # LLM judge (diagnostic): cross-check with a second prompt
    try:
        raw = qwen_chat(
            MAA_JUDGE_PROMPT.format(
                pred_source=inferred, gold_source=gold_source
            ),
            max_tokens=10,
        ).upper()
        llm_match = int("CORRECT" in raw)
    except Exception:
        llm_match = inferred_match

    return {
        "maa_heuristic": heuristic_match,
        "maa_inferred": inferred,
        "maa_inferred_match": inferred_match,
        "maa_llm": llm_match,
        "maa": inferred_match,   # primary score: string-based after label normalisation
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reasoning: Complexity-Weighted Reasoning Score (CWRS)
#    Custom complexity taxonomy designed for HybridQA multi-hop QA, inspired
#    by the heterogeneous relational complexity dimensions in STARK (Wu et al.,
#    NeurIPS 2024). Note: STARK benchmarks LLM retrieval over semi-structured
#    knowledge bases (SKBs) and does not define an L1-L5 numbered scale.
#    The L1-L5 taxonomy below is original to this project.
# ═══════════════════════════════════════════════════════════════════════════

COMPLEXITY_PROMPT = """\
Classify the question complexity using the following taxonomy:
L1: Direct lookup -- single cell
L2: Filter/sort -- one condition on rows
L3: Multi-hop -- link table -> text or table -> table
L4: Aggregation -- count, sum, average, max/min
L5: Complex reasoning -- comparison, inference, counterfactual

Question: "{question}"
Reply with EXACTLY: L1  |  L2  |  L3  |  L4  |  L5"""


def classify_complexity(question: str) -> str:
    try:
        raw = qwen_chat(COMPLEXITY_PROMPT.format(question=question), max_tokens=5).strip()
        m = re.search(r"L[1-5]", raw)
        return m.group() if m else "L3"
    except Exception:
        return "L3"


def complexity_weighted_score(scores_by_level: dict[str, list[float]]) -> float:
    """
    CWRS = SUM w_l * mean(Score_l)  where w_l = l / SUM(l)  (l = 1...5)
    """
    denom = sum(range(1, 6))   # 15
    total = 0.0
    for level, scores in scores_by_level.items():
        l = int(level[1])
        w = l / denom
        if scores:
            total += w * (sum(scores) / len(scores))
    return round(total, 4)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Robustness: Hallucination Rate under Perturbation (HR-P)
#    Measures whether the model correctly refuses when evidence is absent or
#    counterfactual. Conceptually related to:
#      - RGB negative rejection (Chen et al., ACL 2024): model refuses under
#        missing-evidence testbed; paper uses exact string match (Rej) or
#        LLM-judge (Rej*). Our approach uses LLM judge ≈ Rej*.
#      - RGB counterfactual: error detection rate (model flags factual errors).
#        Our HR-P detects hallucination (model asserts a specific answer instead
#        of refusing), which is the complementary failure mode.
# ═══════════════════════════════════════════════════════════════════════════

HALLUCINATION_PROMPT = """\
A QA system was given a question where the context deliberately does NOT contain the answer.
The expected behaviour is for the model to say it cannot answer.

Question         : {question}
Expected behavior: {expected_behavior}
Model output     : {answer}

Did the model hallucinate (i.e. give a specific answer instead of refusing)?
Reply with EXACTLY one word: HALLUCINATED  |  CORRECT_REFUSAL"""

COUNTERFACTUAL_PROMPT = """\
A QA system was given a question with MODIFIED (counterfactual) context.
A key value in the evidence has been changed to an incorrect one.
A well-calibrated model should either detect the inconsistency or refuse to answer.

Question    : {question}
Model output: {answer}

Did the model blindly accept the counterfactual value and produce a specific answer
(hallucination), or did it express uncertainty / refuse?
Reply with EXACTLY one word: HALLUCINATED  |  CORRECT_REFUSAL"""


def hallucination_rate(
    question: str, answer: str,
    condition: str,  # "missing" or "counterfactual"
    expected_behavior: str = "REFUSE",
) -> dict:
    # Fast heuristic: refusal phrases
    refusal_phrases = [
        "cannot determine", "not enough", "no information",
        "cannot answer", "don't know", "i don't", "unable to",
        "not provided", "context does not",
    ]
    heuristic_refusal = any(p in answer.lower() for p in refusal_phrases)

    # Select prompt by condition
    if condition == "counterfactual":
        prompt = COUNTERFACTUAL_PROMPT.format(question=question, answer=answer)
    else:
        prompt = HALLUCINATION_PROMPT.format(
            question=question, answer=answer,
            expected_behavior=expected_behavior,
        )

    try:
        raw = qwen_chat(prompt, max_tokens=10).upper()
        hallucinated = int("HALLUCINATED" in raw)
    except Exception:
        hallucinated = int(not heuristic_refusal)

    return {
        "hallucinated": hallucinated,
        "correct_refusal": 1 - hallucinated,
    }
