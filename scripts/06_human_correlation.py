"""
Script 06 – Human correlation study.

Samples HUMAN_CORR_SUBSET predictions, asks Qwen to act as a human annotator
scoring 1-5, then computes Spearman correlation between each metric and the
LLM-simulated human score.

Output: results/correlation/summary.json
"""
from __future__ import annotations
import json, sys, random, re
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, HUMAN_CORR_SUBSET, RANDOM_SEED
from src.utils.helpers import qwen_chat, get_logger

log = get_logger("06_correlation")
random.seed(RANDOM_SEED)

HUMAN_PROMPT = """\
You are an expert human annotator evaluating a QA system answer.

Question       : {question}
Gold answer    : {gold_answer}
System answer  : {pred_answer}
Context used   : {context}

Score the system answer from 1 to 5:
5 = Perfectly correct, evidence clearly supports it
4 = Mostly correct, minor issue
3 = Partially correct
2 = Mostly wrong but has a relevant element
1 = Completely wrong or hallucinated

Reply with ONLY a single digit (1-5)."""


def get_human_score(item: dict) -> float:
    try:
        raw = qwen_chat(
            HUMAN_PROMPT.format(
                question=item["question"],
                gold_answer=item.get("gold_answer", ""),
                pred_answer=item.get("pred_answer", ""),
                context=(item.get("context_table", "") + " " +
                         item.get("context_text",  ""))[:800],
            ),
            max_tokens=5,
        )
        s = raw.strip()
        m = re.search(r"[1-5]", s)
        return float(m.group()) if m else 3.0
    except Exception:
        return 3.0


def spearman(x: list[float], y: list[float]) -> dict:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": 0.0, "pvalue": 1.0}
    rho, p = spearmanr(x, y)
    return {"rho": round(float(rho), 4), "pvalue": round(float(p), 6)}


def main():
    pred_path = RESULTS_DIR / "baseline" / "predictions.jsonl"
    if not pred_path.exists():
        log.error("Run script 03 first.")
        sys.exit(1)

    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    sample = random.sample(rows, min(HUMAN_CORR_SUBSET, len(rows)))
    log.info("Computing human scores for %d samples …", len(sample))

    for r in tqdm(sample, desc="human scores"):
        r["human_score"] = get_human_score(r)

    human_scores = [r["human_score"] for r in sample]
    metrics = ["em", "f1", "joint_hit", "table_hit", "passage_hit", "epp_f1", "h_fact", "maa"]

    correlations = {}
    for m in metrics:
        vals = [float(r.get(m, 0)) for r in sample]
        correlations[m] = spearman(vals, human_scores)
        log.info("  %-12s  ρ=%.4f  p=%.4f", m,
                 correlations[m]["rho"], correlations[m]["pvalue"])

    # Save annotated sample + correlation table
    out_dir = RESULTS_DIR / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated_sample.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sample), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(correlations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved → %s", out_dir)


if __name__ == "__main__":
    main()
