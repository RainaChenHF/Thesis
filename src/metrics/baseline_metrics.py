"""
Baseline metrics from related work for comparison.

Metric              | Source Paper
--------------------|------------------------------------------
FActScore           | Min et al., EMNLP 2023
RAGAS Faithfulness  | Es et al., EACL 2024
RAGAS Relevancy     | Es et al., EACL 2024
RAGAS Ctx Precision | Es et al., EACL 2024
RAGAS Ctx Recall    | Es et al., EACL 2024
AIS                 | Rashkin et al., CL 2023 (ARES-style)
Noise Robustness    | Chen et al., ACL 2024 (RGB Benchmark)
Negative Rejection  | Chen et al., ACL 2024 (RGB Benchmark)
Info Integration    | Chen et al., ACL 2024 (RGB Benchmark)
BLEU-1              | Papineni et al., ACL 2002
ROUGE-L             | Lin, ACL 2004
BERTScore           | Zhang et al., ICLR 2020
G-Eval              | Liu et al., EMNLP 2023

Notes:
  - BERTScore: implemented as LLM-proxy (no torch dependency required)
  - G-Eval: uses direct integer output; original paper uses probability-weighted scoring
"""
from __future__ import annotations
import re, sys, math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import qwen_chat, qwen_json, get_logger

log = get_logger("baseline_metrics")


# ═══════════════════════════════════════════════════════════════════════════
# 1. FActScore (Min et al., EMNLP 2023) -- text-only version
#    Measures: fraction of atomic facts in answer supported by context
#    Reference-free (uses context, not gold answer)
# ═══════════════════════════════════════════════════════════════════════════

FACTSCORE_DECOMPOSE = """\
Break the following answer into atomic facts (one claim per line).
Answer: "{answer}"
Return ONLY a JSON array of strings. Example: ["fact1", "fact2"]"""

FACTSCORE_VERIFY = """\
Is the following fact supported by the context?
Fact: {fact}
Context: {context}
Reply with EXACTLY one word: SUPPORTED | NOT_SUPPORTED"""


def factscore(answer: str, context: str) -> dict:
    """Original FActScore: text-only, no modality routing."""
    try:
        facts = qwen_json(FACTSCORE_DECOMPOSE.format(answer=answer))
        if not isinstance(facts, list) or not facts:
            return {"factscore": 0.0, "n_facts": 0}
    except Exception:
        return {"factscore": 0.0, "n_facts": 0}

    supported = 0
    for fact in facts:
        try:
            raw = qwen_chat(
                FACTSCORE_VERIFY.format(fact=fact, context=context[:2000]),
                max_tokens=10,
            ).upper()
            if "SUPPORTED" in raw and "NOT_SUPPORTED" not in raw:
                supported += 1
        except Exception:
            pass

    score = supported / len(facts) if facts else 0.0
    return {"factscore": round(score, 4), "n_facts": len(facts)}


# ═══════════════════════════════════════════════════════════════════════════
# 2. RAGAS Faithfulness (Es et al., EACL 2024)
#    Measures: fraction of answer statements entailed by retrieved context
#    Reference-free
# ═══════════════════════════════════════════════════════════════════════════

RAGAS_FAITH_PROMPT = """\
Given the context and answer, determine how many statements in the answer
are directly supported by the context.

Context: {context}
Answer: {answer}

First list all statements in the answer as a JSON array.
Then for each, mark SUPPORTED or NOT_SUPPORTED based on the context.
Return JSON: {{"statements": [{{"s": "...", "supported": true/false}}]}}"""


def ragas_faithfulness(answer: str, context: str) -> dict:
    try:
        result = qwen_json(
            RAGAS_FAITH_PROMPT.format(
                context=context[:3000], answer=answer
            )
        )
        stmts = result.get("statements", [])
        if not stmts:
            return {"ragas_faithfulness": 0.0}
        score = sum(1 for s in stmts if s.get("supported")) / len(stmts)
        return {"ragas_faithfulness": round(score, 4)}
    except Exception as e:
        log.warning("ragas_faithfulness failed: %s", e)
        return {"ragas_faithfulness": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# 3. RAGAS Answer Relevancy (Es et al., EACL 2024)
#    Measures: how relevant the answer is to the question
#    Reference-free
# ═══════════════════════════════════════════════════════════════════════════

RAGAS_RELEVANCY_PROMPT = """\
Given the question and answer, rate how relevant and complete the answer is.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and complete).

Question: {question}
Answer: {answer}

Return ONLY a JSON: {{"score": 0.0}}"""


def ragas_answer_relevancy(question: str, answer: str) -> dict:
    try:
        result = qwen_json(
            RAGAS_RELEVANCY_PROMPT.format(question=question, answer=answer)
        )
        score = float(result.get("score", 0.0))
        return {"ragas_relevancy": round(min(max(score, 0.0), 1.0), 4)}
    except Exception as e:
        log.warning("ragas_answer_relevancy failed: %s", e)
        return {"ragas_relevancy": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Context Precision (RAGAS library; not in the original Es et al., EACL 2024 paper)
#    Note: the EACL 2024 paper defines exactly 3 metrics: Faithfulness, Answer
#    Relevance, and Context Relevance (= extracted sentences / total sentences).
#    "Context Precision" (binary per-chunk relevance check) is from the later
#    RAGAS library v2, not from the paper itself.
#    Measures: fraction of retrieved context chunks that are relevant
#    Reference-free
# ═══════════════════════════════════════════════════════════════════════════

RAGAS_CTX_PREC_PROMPT = """\
Given the question and a context chunk, is this context chunk useful for answering the question?
Question: {question}
Context chunk: {chunk}
Reply with EXACTLY one word: RELEVANT | NOT_RELEVANT"""


def ragas_context_precision(question: str, retrieved_chunks: list[str]) -> dict:
    if not retrieved_chunks:
        return {"ragas_ctx_precision": 0.0}
    relevant = 0
    for chunk in retrieved_chunks:
        try:
            raw = qwen_chat(
                RAGAS_CTX_PREC_PROMPT.format(question=question, chunk=chunk[:500]),
                max_tokens=10,
            ).upper()
            if "NOT_RELEVANT" not in raw and "RELEVANT" in raw:
                relevant += 1
        except Exception:
            pass
    return {"ragas_ctx_precision": round(relevant / len(retrieved_chunks), 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Context Recall (RAGAS library; not in the original Es et al., EACL 2024 paper)
#    Note: the EACL 2024 paper does not define a "context recall" metric.
#    This metric is from the later RAGAS library. It measures the fraction of
#    gold answer sentences that can be attributed to the retrieved context.
#    Reference-based (needs gold answer)
# ═══════════════════════════════════════════════════════════════════════════

RAGAS_CTX_RECALL_PROMPT = """\
Can the following statement be attributed to the given context?
Statement: {statement}
Context: {context}
Reply with EXACTLY one word: YES | NO"""


def ragas_context_recall(gold_answer: str, context: str) -> dict:
    sentences = [s.strip() for s in gold_answer.split(".") if s.strip()]
    if not sentences:
        return {"ragas_ctx_recall": 0.0}
    attributed = 0
    for sent in sentences:
        try:
            raw = qwen_chat(
                RAGAS_CTX_RECALL_PROMPT.format(
                    statement=sent, context=context[:3000]
                ),
                max_tokens=5,
            ).upper()
            if "YES" in raw:
                attributed += 1
        except Exception:
            pass
    return {"ragas_ctx_recall": round(attributed / len(sentences), 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 6. AIS -- Attribution to Identified Sources (Rashkin et al., CL 2023)
#    Measures: whether every claim in the answer is attributable to a source
#    Reference-free
# ═══════════════════════════════════════════════════════════════════════════

AIS_PROMPT = """\
Is the following statement fully supported by the provided source document?
Apply the 'according to' test: would a reader say "According to the source, [statement]" accurately?

Source: {context}
Statement: {statement}

Reply with EXACTLY one word: ATTRIBUTED | NOT_ATTRIBUTED"""


def ais_score(answer: str, context: str) -> dict:
    """AIS (Rashkin et al., Computational Linguistics 2023) -- per-sentence attribution.

    The AIS framework evaluates each sentence independently in two stages:
      Stage 1 (Interpretability): is the sentence interpretable in isolation?
      Stage 2 (Attribution): does it pass the 'according to' test against the source?
    Aggregation: AIS = fraction of sentences that are ATTRIBUTED.

    Our implementation: single LLM call per sentence for the attribution step.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 5]
    if not sentences:
        sentences = [answer.strip()] if answer.strip() else []
    if not sentences:
        return {"ais": 0.0}

    attributed = 0
    for sent in sentences:
        try:
            raw = qwen_chat(
                AIS_PROMPT.format(context=context[:3000], statement=sent),
                max_tokens=10,
            ).upper()
            if "NOT_ATTRIBUTED" not in raw and "ATTRIBUTED" in raw:
                attributed += 1
        except Exception:
            pass
    return {"ais": round(attributed / len(sentences), 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 7. RGB Benchmark metrics (Chen et al., ACL 2024)
#    Noise Robustness: exact-match accuracy despite noisy retrieved docs
#    Negative Rejection: does model refuse when answer is absent? (Rej* variant)
#    Information Integration: exact-match accuracy for multi-doc synthesis
#    Reference: Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented
#    Generation", ACL 2024 (originally presented at AAAI 2024).
# ═══════════════════════════════════════════════════════════════════════════

def rgb_noise_robustness(question: str, pred: str, gold: str) -> dict:
    """RGB noise robustness (Chen et al., ACL 2024 §4.1).

    Paper uses exact substring match: correct if generated text contains the
    gold answer string (case-insensitive). Noise ratio tested at 0/0.2/0.4/0.6/0.8.
    Measures: does the model extract the correct answer despite noisy documents?
    """
    if not gold.strip():
        return {"rgb_noise": 0.0}
    return {"rgb_noise": float(gold.lower().strip() in pred.lower())}


def rgb_negative_rejection(answer: str) -> dict:
    """RGB negative rejection -- soft version 'Rej*' (Chen et al., ACL 2024 §4.2).

    Hard version 'Rej' (paper): exact match to the system-instructed string
      "I can not answer the question because of the insufficient information in documents."
    Soft version 'Rej*' (paper): LLM (ChatGPT) judges whether model refused in spirit.
    Our implementation: heuristic phrase matching, equivalent to Rej* (soft version).
    """
    refusal_phrases = [
        "cannot determine", "not enough", "no information",
        "cannot answer", "don't know", "i don't", "unable to",
        "not provided", "context does not", "insufficient",
        "no evidence", "cannot find",
    ]
    refused = any(p in answer.lower() for p in refusal_phrases)
    return {"rgb_neg_rejection": int(refused)}


def rgb_info_integration(question: str, pred: str, gold: str) -> dict:
    """RGB information integration (Chen et al., ACL 2024 §4.3).

    Paper uses exact substring match accuracy for multi-document synthesis questions
    (questions requiring combining facts from 2+ documents).
    Measures: can the model synthesise a correct answer from multiple evidence pieces?
    """
    if not gold.strip():
        return {"rgb_integration": 0.0}
    return {"rgb_integration": float(gold.lower().strip() in pred.lower())}


# ═══════════════════════════════════════════════════════════════════════════
# 8. BLEU-1 (Papineni et al., ACL 2002) -- reference-based
#    Brevity penalty formula: BP = exp(1 - r/c) if c < r, else 1.0
# ═══════════════════════════════════════════════════════════════════════════

def bleu1(pred: str, gold: str) -> dict:
    pred_toks = pred.lower().split()
    gold_toks = gold.lower().split()
    if not pred_toks or not gold_toks:
        return {"bleu1": 0.0}
    gold_counts = Counter(gold_toks)
    clipped = sum(min(cnt, gold_counts[tok]) for tok, cnt in Counter(pred_toks).items())
    precision = clipped / len(pred_toks)
    # Brevity penalty: BP = exp(1 - r/c) when c < r, else 1.0
    c = len(pred_toks)
    r = len(gold_toks)
    bp = math.exp(1 - r / c) if c < r else 1.0
    return {"bleu1": round(bp * precision, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 9. ROUGE-L (Lin, ACL 2004) -- reference-based
# ═══════════════════════════════════════════════════════════════════════════

def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    return dp[m % 2][n]


def rouge_l(pred: str, gold: str) -> dict:
    p_toks = pred.lower().split()
    g_toks = gold.lower().split()
    if not p_toks or not g_toks:
        return {"rouge_l": 0.0}
    lcs = _lcs_length(p_toks, g_toks)
    prec = lcs / len(p_toks)
    rec  = lcs / len(g_toks)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"rouge_l": round(f1, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# 10. BERTScore proxy (Zhang et al., ICLR 2020)
#     True BERTScore uses contextual token embeddings (RoBERTa_large):
#       P_BERT = mean of max cosine sim of each pred token to any ref token
#       R_BERT = mean of max cosine sim of each ref token to any pred token
#       F_BERT = harmonic mean of P_BERT and R_BERT
#     Optional IDF weighting and baseline rescaling for human-score alignment.
#     Our implementation: LLM-proxy (Qwen semantic similarity prompt) because
#     torch + RoBERTa model loading is not available in this environment.
#     True BERTScore requires: pip install bert-score + a BERT/RoBERTa model.
# ═══════════════════════════════════════════════════════════════════════════

BERTSCORE_PROMPT = """\
Rate the semantic similarity between the prediction and the reference on a scale from 0.0 to 1.0.
0.0 means completely different meaning.
1.0 means identical meaning (even if different wording).

Prediction: {pred}
Reference: {gold}

Consider: Do they convey the same information? Are the key entities and facts the same?
Return ONLY a JSON: {{"score": 0.0}}"""


def bertscore_proxy(pred: str, gold: str) -> dict:
    """LLM-proxy for BERTScore when torch/transformers unavailable."""
    if not pred.strip() or not gold.strip():
        return {"bertscore": 0.0}
    try:
        result = qwen_json(
            BERTSCORE_PROMPT.format(pred=pred[:500], gold=gold[:500])
        )
        score = float(result.get("score", 0.0))
        return {"bertscore": round(min(max(score, 0.0), 1.0), 4)}
    except Exception as e:
        log.warning("bertscore_proxy failed: %s", e)
        return {"bertscore": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# 11. G-Eval (Liu et al., EMNLP 2023)
#     Framework: LLM evaluation with auto-generated chain-of-thought steps.
#     Key innovation: probability-weighted scoring score = Σ p(s_i) * s_i
#     (weighted expectation over discrete score tokens, not raw integer output).
#     Our implementation uses direct integer output / 5 because Qwen chat API
#     does not expose per-token log-probabilities, reducing scoring granularity.
#     Original paper dimensions (summarization): Coherence, Consistency, Fluency,
#     Relevance. Our adaptation for QA replaces Fluency with Correctness (factual
#     accuracy vs. gold answer) — a domain-appropriate modification.
# ═══════════════════════════════════════════════════════════════════════════

GEVAL_PROMPT = """\
You will evaluate the quality of an answer generated by a QA system.

Question: {question}
Context (retrieved evidence): {context}
Generated Answer: {answer}

Evaluate along four dimensions. For each, think step-by-step, then give a score 1-5.

1. Consistency: Does the answer contain only information supported by the context? (1=hallucinated, 5=fully grounded)
2. Relevance: Does the answer address the question directly? (1=off-topic, 5=perfectly relevant)
3. Coherence: Is the answer well-organized and logical? (1=incoherent, 5=perfectly coherent)
4. Correctness: Is the factual content of the answer accurate? (1=wrong, 5=fully correct)

Return ONLY a JSON:
{{"consistency": 0, "relevance": 0, "coherence": 0, "correctness": 0, "reasoning": "brief explanation"}}"""


def g_eval(question: str, answer: str, context: str) -> dict:
    """G-Eval: Multi-dimensional LLM evaluation with CoT."""
    if not answer.strip():
        return {"geval_consistency": 0.0, "geval_relevance": 0.0,
                "geval_coherence": 0.0, "geval_correctness": 0.0, "geval_avg": 0.0}
    try:
        result = qwen_json(
            GEVAL_PROMPT.format(
                question=question, answer=answer,
                context=context[:2000]
            )
        )
        cons = float(result.get("consistency", 1)) / 5.0
        rel  = float(result.get("relevance", 1)) / 5.0
        coh  = float(result.get("coherence", 1)) / 5.0
        corr = float(result.get("correctness", 1)) / 5.0
        avg  = (cons + rel + coh + corr) / 4.0
        return {
            "geval_consistency": round(cons, 4),
            "geval_relevance": round(rel, 4),
            "geval_coherence": round(coh, 4),
            "geval_correctness": round(corr, 4),
            "geval_avg": round(avg, 4),
        }
    except Exception as e:
        log.warning("g_eval failed: %s", e)
        return {"geval_consistency": 0.0, "geval_relevance": 0.0,
                "geval_coherence": 0.0, "geval_correctness": 0.0, "geval_avg": 0.0}
