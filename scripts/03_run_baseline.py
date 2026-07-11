"""
Script 03 – Run the full RAG pipeline on the golden dev set and evaluate
all Hybrid-Eval metrics.

Outputs:
  results/baseline/predictions.jsonl   – per-sample predictions + metrics
  results/baseline/summary.json        – aggregate scores
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, RESULTS_DIR, PERTURBATION_SUBSET, RANDOM_SEED
from src.pipeline.perturbation import load_dev, load_tables, load_passages
from src.pipeline.rag import run_pipeline
from src.pipeline.perturbation import _get_gold_table_row, _get_gold_passage
from src.metrics.hybrid_eval import (
    joint_hit_at_k, evidence_path_precision,
    hybrid_factscore, modality_attribution_accuracy,
    classify_complexity,
)
from src.utils.helpers import exact_match, token_f1, get_logger

log = get_logger("03_baseline")


def evaluate_one(item: dict, result: dict,
                 tables_db: dict, passages_db: dict) -> dict:
    question    = item["question"]
    gold_answer = str(item.get("answer_text", ""))
    pred_answer = result.get("pred_answer", "")

    # Gold evidence
    gold_row  = _get_gold_table_row(item, tables_db)
    gold_pass = _get_gold_passage(item, passages_db, tables_db)

    # EM / F1
    em = exact_match(pred_answer, gold_answer)
    f1 = token_f1(pred_answer, gold_answer)

    # JHit@K
    jh = joint_hit_at_k(
        gold_row, gold_pass,
        result.get("retrieved_table_rows", []),
        result.get("retrieved_passages", []),
    )

    # EPP
    epp = evidence_path_precision(
        [gold_row, gold_pass],
        result.get("retrieved_table_rows", []) + result.get("retrieved_passages", []),
    )

    # H-FAct (only when answer non-empty)
    hf = {"h_fact": 0.0, "n_facts": 0, "details": []}
    if pred_answer and len(pred_answer) > 3:
        hf = hybrid_factscore(
            pred_answer,
            result.get("context_table", ""),
            result.get("context_text",  ""),
        )

    # MAA
    gold_source = "table" if any(
        len(n) > 3 and n[3] == "table" for n in item.get("answer_node", [])
    ) else "passage"
    maa = modality_attribution_accuracy(
        question, pred_answer,
        result.get("pred_source", "unknown"), gold_source,
    )

    # Complexity
    complexity = classify_complexity(question)

    return {
        **result,
        "gold_answer": gold_answer,
        "gold_source": gold_source,
        "em": em, "f1": f1,
        **jh, **epp, **hf, **maa,
        "complexity": complexity,
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key): return round(sum(r.get(key, 0) for r in rows) / max(len(rows), 1), 4)
    return {
        "n_samples": len(rows),
        "em":           mean("em"),
        "f1":           mean("f1"),
        "joint_hit":    mean("joint_hit"),
        "table_hit":    mean("table_hit"),
        "passage_hit":  mean("passage_hit"),
        "epp_f1":       mean("epp_f1"),
        "h_fact":       mean("h_fact"),
        "maa":          mean("maa"),
    }


def main():
    out_dir = RESULTS_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev      = load_dev()
    tables   = load_tables()
    passages = load_passages()

    # Use subset for speed; set PERTURBATION_SUBSET = None to run full dev
    import random; random.seed(RANDOM_SEED)
    sample = random.sample(dev, min(PERTURBATION_SUBSET, len(dev)))
    log.info("Running baseline on %d samples …", len(sample))

    pred_path = out_dir / "predictions.jsonl"
    results   = []
    done_ids: set = set()

    # Resume: load already-completed rows
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r.get("question_id", ""))
        log.info("  Resuming baseline: %d/%d already done", len(results), len(sample))

    remaining = [item for item in sample if item.get("question_id", "") not in done_ids]

    with open(pred_path, "a", encoding="utf-8") as fout:
        for item in tqdm(remaining, desc="baseline", initial=len(results), total=len(sample)):
            result = run_pipeline(item, tables, passages)
            scored = evaluate_one(item, result, tables, passages)
            results.append(scored)
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
            fout.flush()

    summary = aggregate(results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("=== Baseline Summary ===")
    for k, v in summary.items():
        log.info("  %-20s %s", k, v)
    log.info("Results saved → %s", out_dir)


if __name__ == "__main__":
    main()
