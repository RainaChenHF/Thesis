"""
Script 04 – Run the RAG pipeline on all five perturbed conditions and
evaluate Hybrid-Eval metrics for each.

Outputs:
  results/perturbation/{condition}_predictions.jsonl
  results/perturbation/summary.json   – all conditions side-by-side
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PERT_DIR, RESULTS_DIR, PERTURBATION_SUBSET, RANDOM_SEED
from src.pipeline.perturbation import load_tables, load_passages, _get_gold_table_row, _get_gold_passage
from src.pipeline.rag import run_pipeline
from src.metrics.hybrid_eval import (
    joint_hit_at_k, evidence_path_precision,
    hybrid_factscore, hallucination_rate,
    modality_attribution_accuracy, classify_complexity,
)
from src.utils.helpers import exact_match, token_f1, get_logger

log = get_logger("04_perturbation")
random.seed(RANDOM_SEED)

CONDITIONS = ["golden", "semi_golden", "noise", "counterfactual", "missing"]


def load_condition(name: str) -> list[dict]:
    p = PERT_DIR / f"{name}.jsonl"
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_one(item: dict, result: dict, tables_db: dict, passages_db: dict) -> dict:
    question    = item["question"]
    gold_answer = str(item.get("answer_text", ""))
    pred_answer = result.get("pred_answer", "")
    condition   = item.get("condition", "unknown")

    em = exact_match(pred_answer, gold_answer)
    f1 = token_f1(pred_answer, gold_answer)

    gold_row  = _get_gold_table_row(item, tables_db)
    gold_pass = _get_gold_passage(item, passages_db, tables_db)

    jh  = joint_hit_at_k(gold_row, gold_pass,
                          result.get("retrieved_table_rows", []),
                          result.get("retrieved_passages",   []))
    epp = evidence_path_precision(
        [gold_row, gold_pass],
        result.get("retrieved_table_rows", []) + result.get("retrieved_passages", []),
    )

    hf = {"h_fact": 0.0}
    if pred_answer and len(pred_answer) > 3:
        hf = hybrid_factscore(pred_answer,
                               result.get("context_table", ""),
                               result.get("context_text",  ""))

    # HR-P only for adversarial conditions
    hrp = {}
    if condition in ("missing", "counterfactual"):
        hrp = hallucination_rate(
            question, pred_answer, condition,
            item.get("expected_behavior", "REFUSE"),
        )

    gold_src = "table" if any(len(n) > 3 and n[3] == "table" for n in item.get("answer_node", [])) else "passage"
    maa = modality_attribution_accuracy(question, pred_answer,
                                        result.get("pred_source", "unknown"), gold_src)
    complexity = classify_complexity(question)

    return {
        **result,
        "gold_answer": gold_answer,
        "em": em, "f1": f1,
        **jh, **epp, **hf, **maa,
        **hrp,
        "complexity": complexity,
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key): return round(sum(r.get(key, 0) for r in rows) / max(len(rows), 1), 4)
    d = {
        "n": len(rows),
        "em": mean("em"), "f1": mean("f1"),
        "joint_hit": mean("joint_hit"),
        "epp_f1":    mean("epp_f1"),
        "h_fact":    mean("h_fact"),
        "maa":       mean("maa"),
    }
    if any("hallucinated" in r for r in rows):
        d["hr_p"] = mean("hallucinated")
    return d


def compute_mirage_metrics(out_dir: Path) -> dict:
    """MiRAGE-style group metrics: NV, CA, CI, CM.

    noise     → baseline (b): model gets random/noisy context
    golden    → oracle (o): model gets perfect context
    semi_golden → mixed (m): model gets partial context
    """
    cond_rows: dict[str, dict] = {}
    for cond, key in [("noise", "b"), ("golden", "o"), ("semi_golden", "m")]:
        path = out_dir / f"{cond}_predictions.jsonl"
        if not path.exists():
            log.warning("MiRAGE: missing %s", path)
            return {}
        by_id: dict = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_id[r.get("question_id", "")] = int(r.get("em", 0))
        cond_rows[key] = by_id

    shared_ids = set(cond_rows["b"]) & set(cond_rows["o"]) & set(cond_rows["m"])
    if not shared_ids:
        log.warning("MiRAGE: no shared question_ids across conditions")
        return {}

    nv = ca = ci = cm = 0
    n = len(shared_ids)
    for qid in shared_ids:
        b = cond_rows["b"][qid]
        o = cond_rows["o"][qid]
        m = cond_rows["m"][qid]
        if o == 1 and b == 0:
            nv += 1   # noise vulnerability: oracle correct, noise context wrong
        if o == 1 and m == 1:
            ca += 1   # context adherence: oracle and mixed both correct
        if o == 0:
            ci += 1   # context insensitivity: oracle wrong regardless
        if b == 0 and o == 0 and m == 0:
            cm += 1   # consistent miss: always wrong under all conditions

    return {
        "mirage_n":  n,
        "mirage_NV": round(nv / n, 4),
        "mirage_CA": round(ca / n, 4),
        "mirage_CI": round(ci / n, 4),
        "mirage_CM": round(cm / n, 4),
    }


def main():
    out_dir = RESULTS_DIR / "perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables   = load_tables()
    passages = load_passages()

    all_summaries = {}

    for cond in CONDITIONS:
        log.info("── Condition: %s ──", cond)
        data   = load_condition(cond)
        random.seed(RANDOM_SEED)
        sample = random.sample(data, min(PERTURBATION_SUBSET, len(data)))

        pred_path = out_dir / f"{cond}_predictions.jsonl"

        # Resume: load already-completed rows
        rows = []
        done_ids: set = set()
        if pred_path.exists():
            with open(pred_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        rows.append(r)
                        done_ids.add(r.get("question_id", ""))
            log.info("  Resuming %s: %d/%d already done", cond, len(rows), len(sample))

        remaining = [item for item in sample if item.get("question_id", "") not in done_ids]

        with open(pred_path, "a", encoding="utf-8") as fout:
            for item in tqdm(remaining, desc=cond, initial=len(rows), total=len(sample)):
                result = run_pipeline(
                    item, tables, passages,
                    force_table_ctx=item.get("force_table_ctx"),
                    force_text_ctx=item.get("force_text_ctx"),
                )
                scored = evaluate_one(item, result, tables, passages)
                rows.append(scored)
                fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
                fout.flush()

        agg = aggregate(rows)
        all_summaries[cond] = agg
        log.info("  %s", agg)

    (out_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Perturbation results saved → %s", out_dir)

    # MiRAGE group metrics (requires noise/golden/semi_golden all done)
    mirage = compute_mirage_metrics(out_dir)
    if mirage:
        all_summaries["mirage"] = mirage
        (out_dir / "summary.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("MiRAGE metrics: %s", mirage)


if __name__ == "__main__":
    main()
