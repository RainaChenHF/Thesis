"""
Script 08 -- Comparison with baseline metrics from related work.

Runs all related-work metrics on the same 500 baseline samples and
computes Spearman correlation with human scores.

Updated: Added BERTScore (Zhang et al., ICLR 2020) and G-Eval (Liu et al., EMNLP 2023)

Output:
  results/comparison/summary.json      -- per-metric scores
  results/comparison/correlation.json  -- Spearman rho vs human
  results/tables/comparison.md         -- paper-ready table
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, RANDOM_SEED
from src.metrics.baseline_metrics import (
    factscore, ragas_faithfulness, ragas_answer_relevancy,
    ragas_context_precision, ragas_context_recall,
    ais_score, rgb_noise_robustness, rgb_negative_rejection,
    rgb_info_integration, bleu1, rouge_l,
    bertscore_proxy, g_eval,
)
from src.utils.helpers import exact_match, token_f1, get_logger

log = get_logger("08_comparison")
random.seed(RANDOM_SEED)

CMP_DIR = RESULTS_DIR / "comparison"
CMP_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_preds(n: int = 500) -> list[dict]:
    path = RESULTS_DIR / "baseline" / "predictions.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:n]


def load_human_scores() -> dict[str, float]:
    """Load human scores from correlation study."""
    path = RESULTS_DIR / "correlation" / "annotated_sample.jsonl"
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                scores[r["question_id"]] = r.get("human_score", 0)
    return scores


def evaluate_one(row: dict) -> dict:
    pred    = row.get("pred_answer", "")
    gold    = row.get("gold_answer", row.get("answer_text", ""))
    question = row.get("question", "")
    ctx_table = row.get("context_table", "")
    ctx_text  = row.get("context_text", "")
    context   = f"{ctx_table}\n{ctx_text}".strip()
    retrieved = row.get("retrieved_table_rows", []) + row.get("retrieved_passages", [])
    condition = row.get("condition", "golden")

    out = {"question_id": row.get("question_id", "")}

    # Reference-based lexical
    out.update(bleu1(pred, gold))
    out.update(rouge_l(pred, gold))
    out["em"] = exact_match(pred, gold)
    out["f1"] = token_f1(pred, gold)

    # BERTScore (NEW - semantic similarity proxy)
    out.update(bertscore_proxy(pred, gold))

    # LLM-based reference-free
    if pred and len(pred) > 3:
        out.update(factscore(pred, context))
        out.update(ragas_faithfulness(pred, context))
        out.update(ragas_answer_relevancy(question, pred))
        out.update(ragas_context_precision(question, retrieved))
        out.update(ragas_context_recall(gold, context))
        out.update(ais_score(pred, context))
        out.update(rgb_noise_robustness(question, pred, gold))
        out.update(rgb_info_integration(question, pred, gold))
        # G-Eval (NEW - multi-dimensional CoT evaluation)
        out.update(g_eval(question, pred, context))
    else:
        out.update({"factscore": 0.0, "ragas_faithfulness": 0.0,
                    "ragas_relevancy": 0.0, "ragas_ctx_precision": 0.0,
                    "ragas_ctx_recall": 0.0, "ais": 0.0,
                    "rgb_noise": 0.0, "rgb_integration": 0.0,
                    "geval_consistency": 0.0, "geval_relevance": 0.0,
                    "geval_coherence": 0.0, "geval_correctness": 0.0,
                    "geval_avg": 0.0})

    # RGB negative rejection (only meaningful for missing condition;
    # baseline rows are all golden-context so expected value is 0)
    if condition == "missing":
        out.update(rgb_negative_rejection(pred))
    else:
        out["rgb_neg_rejection"] = 0.0

    return out


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    import math
    from scipy.stats import spearmanr
    if len(xs) < 3:
        return 0.0, 1.0
    r, p = spearmanr(xs, ys)
    if math.isnan(r):
        return 0.0, 1.0
    return round(float(r), 4), round(float(p), 4)


def main():
    log.info("Loading baseline predictions ...")
    rows = load_baseline_preds(500)
    human_scores = load_human_scores()

    pred_path = CMP_DIR / "predictions.jsonl"

    # Resume support
    done_ids: set = set()
    results = []
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r.get("question_id", ""))
        log.info("Resuming: %d/%d done", len(results), len(rows))

    remaining = [r for r in rows if r.get("question_id", "") not in done_ids]

    with open(pred_path, "a", encoding="utf-8") as fout:
        for row in tqdm(remaining, desc="comparison", initial=len(results), total=len(rows)):
            scored = evaluate_one(row)
            results.append(scored)
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
            fout.flush()

    # Aggregate means
    metric_keys = [
        "em", "f1", "bleu1", "rouge_l", "bertscore",
        "factscore", "ragas_faithfulness", "ragas_relevancy",
        "ragas_ctx_precision", "ragas_ctx_recall",
        "ais", "rgb_noise", "rgb_integration", "rgb_neg_rejection",
        "geval_consistency", "geval_relevance", "geval_coherence",
        "geval_correctness", "geval_avg",
    ]
    summary = {}
    for k in metric_keys:
        vals = [r.get(k, 0.0) for r in results]
        summary[k] = round(sum(vals) / max(len(vals), 1), 4)

    (CMP_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary: %s", summary)

    # Spearman correlation with human scores
    shared = [(r, human_scores[r["question_id"]])
              for r in results if r.get("question_id") in human_scores]
    log.info("Shared samples for correlation: %d", len(shared))

    corr = {}
    if len(shared) >= 10:
        hs = [h for _, h in shared]
        for k in metric_keys:
            ms = [r.get(k, 0.0) for r, _ in shared]
            rho, pval = spearman(ms, hs)
            corr[k] = {"rho": rho, "pvalue": pval}
            log.info("  %-25s rho=%.4f  p=%.4f", k, rho, pval)

    (CMP_DIR / "correlation.json").write_text(
        json.dumps(corr, indent=2), encoding="utf-8"
    )

    # Generate markdown comparison table
    _write_comparison_table(summary, corr)
    log.info("Done -> %s", CMP_DIR)


def _write_comparison_table(summary: dict, corr: dict):
    # Group metrics by source paper
    groups = [
        ("HybridQA (Chen et al., EMNLP 2020)",    ["em", "f1"]),
        ("Lexical (BLEU/ROUGE)",             ["bleu1", "rouge_l"]),
        ("BERTScore (Zhang et al., ICLR 2020)",   ["bertscore"]),
        ("FActScore (Min et al., EMNLP 2023)",     ["factscore"]),
        ("RAGAS (Es et al., EACL 2024)",          ["ragas_faithfulness", "ragas_relevancy",
                                              "ragas_ctx_precision", "ragas_ctx_recall"]),
        ("G-Eval (Liu et al., EMNLP 2023)",        ["geval_consistency", "geval_relevance",
                                              "geval_coherence", "geval_correctness", "geval_avg"]),
        ("AIS (Rashkin et al., CL 2023)",       ["ais"]),
        ("RGB (Chen et al., ACL 2024)",          ["rgb_noise", "rgb_integration", "rgb_neg_rejection"]),
        ("Hybrid-Eval (Ours)",               ["h_fact", "epp_f1", "joint_hit", "maa"]),
    ]

    # Load our metrics from baseline summary for the "Ours" row
    our = {}
    try:
        our = json.loads(
            (RESULTS_DIR / "baseline" / "summary.json").read_text(encoding="utf-8")
        )
    except Exception:
        pass

    our_corr = {}
    try:
        our_corr = json.loads(
            (RESULTS_DIR / "correlation" / "summary.json").read_text(encoding="utf-8")
        )
    except Exception:
        pass

    md = "# Metric Comparison with Related Work\n\n"
    md += "| Source | Metric | Score | Spearman rho | p-value |\n"
    md += "|:---|:---|---:|---:|---:|\n"

    for source, keys in groups:
        first_source = source
        for k in keys:
            if k in ("h_fact", "epp_f1", "joint_hit", "maa"):
                score = our.get(k, "--")
                rho   = our_corr.get(k, {}).get("rho", "--")
                pval  = our_corr.get(k, {}).get("pvalue", "--")
            else:
                score = summary.get(k, "--")
                rho   = corr.get(k, {}).get("rho", "--")
                pval  = corr.get(k, {}).get("pvalue", "--")

            score_s = f"{score:.4f}" if isinstance(score, float) else str(score)
            rho_s   = f"{rho:.4f}"   if isinstance(rho,   float) else str(rho)
            pval_s  = f"{pval:.4f}"  if isinstance(pval,  float) else str(pval)
            md += f"| {first_source} | {k} | {score_s} | {rho_s} | {pval_s} |\n"
            first_source = ""  # blank for subsequent rows in same group

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)
    (tables_dir / "comparison.md").write_text(md, encoding="utf-8")
    log.info("Comparison table -> %s", tables_dir / "comparison.md")

    metric_label = {
        "em": "EM",
        "f1": "F1",
        "bleu1": "BLEU-1",
        "rouge_l": "ROUGE-L",
        "bertscore": "BERTScore",
        "factscore": "FActScore",
        "ragas_faithfulness": "Faithfulness",
        "ragas_relevancy": "Answer Relevancy",
        "ragas_ctx_precision": "Context Precision",
        "ragas_ctx_recall": "Context Recall",
        "geval_consistency": "Consistency",
        "geval_relevance": "Relevance",
        "geval_coherence": "Coherence",
        "geval_correctness": "Correctness",
        "geval_avg": "Average",
        "ais": "AIS",
        "rgb_noise": "Noise Robustness",
        "rgb_integration": "Info Integration",
        "rgb_neg_rejection": "Negative Rejection",
        "h_fact": "H-FAct",
        "epp_f1": "EPP-F1",
        "joint_hit": "JHit@3",
        "maa": "MAA",
    }

    def _p_tex(v):
        if not isinstance(v, float):
            return "---"
        return "$<$0.001" if v < 0.001 else f"{v:.4f}"

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Metric Comparison with Related Work. Human scores: LLM-simulated annotation (Qwen-max, 1--5 scale, $n{=}100$). $\rho$: Spearman correlation with human scores.}",
        r"\label{tab:comparison}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Metric} & \textbf{Score} & \textbf{$\rho$} & \textbf{$p$-value} \\",
        r"\midrule",
    ]

    for source, keys in groups:
        first_source = source.replace("EMNLP 2020", "2020").replace("ICLR 2020", "2020").replace("EMNLP 2023", "2023").replace("EACL 2024", "2024").replace("ACL 2024", "2024").replace("CL 2023", "2023")
        for k in keys:
            if k in ("h_fact", "epp_f1", "joint_hit", "maa"):
                score = our.get(k, "--")
                rho   = our_corr.get(k, {}).get("rho", "--")
                pval  = our_corr.get(k, {}).get("pvalue", "--")
            else:
                score = summary.get(k, "--")
                rho   = corr.get(k, {}).get("rho", "--")
                pval  = corr.get(k, {}).get("pvalue", "--")

            score_s = f"{score:.3f}" if isinstance(score, float) else "---"
            rho_s   = f"{rho:.4f}" if isinstance(rho, float) else "---"
            p_s     = _p_tex(pval)
            mname   = metric_label.get(k, k)
            if source == "Hybrid-Eval (Ours)" and mname in ("H-FAct", "MAA"):
                if first_source:
                    line = f"\\textbf{{{first_source}}} & \\textbf{{{mname}}} & \\textbf{{{score_s}}} & \\textbf{{{rho_s}}} & {p_s} \\\\"
                else:
                    line = f"                                    & \\textbf{{{mname}}} & \\textbf{{{score_s}}} & \\textbf{{{rho_s}}} & {p_s} \\\\"
            else:
                if first_source:
                    line = f"{first_source} & {mname} & {score_s} & {rho_s} & {p_s} \\\\"
                else:
                    line = f"                                    & {mname} & {score_s} & {rho_s} & {p_s} \\\\"
            tex.append(line)
            first_source = ""
        tex.append(r"\midrule")

    if tex[-1] == r"\midrule":
        tex.pop()
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables_dir / "comparison.tex").write_text("\n".join(tex), encoding="utf-8")
    log.info("Comparison LaTeX -> %s", tables_dir / "comparison.tex")


if __name__ == "__main__":
    main()
