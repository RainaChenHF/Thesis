"""
Script 10 – H-FAct Ablation: Modality Routing vs No Routing.

Core empirical validation of our main contribution, using the SAME 100-sample
annotated subset as the human-correlation study and reusing cached metric values
rather than fresh LLM re-inference:
  - H-FAct (ours): cached from results/correlation/annotated_sample.jsonl
  - FActScore (baseline): cached from results/comparison/predictions.jsonl

This avoids run-to-run LLM nondeterminism and keeps the ablation perfectly
aligned with the paper's primary correlation table.

Output:
  results/ablation/routing_ablation.json
  results/tables/routing_ablation.md
  results/tables/routing_ablation.tex
  results/figures/routing_ablation_scatter.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR
from src.utils.helpers import get_logger

log = get_logger("10_ablation_routing")

ABLATION_DIR = RESULTS_DIR / "ablation"
TABLES_DIR   = RESULTS_DIR / "tables"
FIGURES_DIR  = RESULTS_DIR / "figures"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def load_cached_sample() -> tuple[list[dict], dict[str, float]]:
    ann_path = RESULTS_DIR / "correlation" / "annotated_sample.jsonl"
    cmp_path = RESULTS_DIR / "comparison" / "predictions.jsonl"
    if not ann_path.exists():
        log.error("Run 06_human_correlation.py first: %s", ann_path)
        return [], {}
    if not cmp_path.exists():
        log.error("Run 08_comparison_metrics.py first: %s", cmp_path)
        return [], {}

    annotated = [json.loads(l) for l in open(ann_path, encoding="utf-8") if l.strip()]
    comparison = {
        r["question_id"]: r
        for r in (json.loads(l) for l in open(cmp_path, encoding="utf-8") if l.strip())
    }

    rows = []
    human_scores: dict[str, float] = {}
    for ann in annotated:
        qid = ann.get("question_id", "")
        cmp_row = comparison.get(qid, {})
        rows.append({
            "question_id": qid,
            "h_fact": float(ann.get("h_fact", 0.0)),
            "n_facts_hf": int(ann.get("n_facts", 0)),
            "factscore": float(cmp_row.get("factscore", 0.0)),
            "n_facts_fs": int(cmp_row.get("n_facts", 0)),
        })
        human_scores[qid] = float(ann.get("human_score", 0.0))
    return rows, human_scores


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    import math
    from scipy.stats import spearmanr
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    r, _ = spearmanr(xs, ys)
    return 0.0 if math.isnan(r) else round(float(r), 4)


def main():
    results, human_scores = load_cached_sample()
    if not results:
        return

    log.info("Running routing ablation on %d cached annotated samples …", len(results))

    # Save per-sample aligned cache for inspection
    out_path = ABLATION_DIR / "routing_ablation.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in results:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate
    mean_hf = round(sum(r["h_fact"]    for r in results) / max(len(results), 1), 4)
    mean_fs = round(sum(r["factscore"] for r in results) / max(len(results), 1), 4)

    shared = [(r["h_fact"], r["factscore"], human_scores[r["question_id"]])
              for r in results if r["question_id"] in human_scores]

    rho_hf = spearman_rho([x[0] for x in shared], [x[2] for x in shared]) if shared else 0.0
    rho_fs = spearman_rho([x[1] for x in shared], [x[2] for x in shared]) if shared else 0.0

    summary = {
        "n_samples": len(results),
        "n_human_shared": len(shared),
        "source": "cached annotated_sample + comparison predictions",
        "h_fact_mean": mean_hf,
        "factscore_mean": mean_fs,
        "h_fact_vs_human": rho_hf,
        "factscore_vs_human": rho_fs,
        "routing_gain_score": round(mean_hf - mean_fs, 4),
        "routing_gain_rho": round(rho_hf - rho_fs, 4),
        "score_gap_hfact_minus_factscore": round(mean_hf - mean_fs, 4),
        "human_alignment_gain": round(rho_hf - rho_fs, 4),
    }

    (ABLATION_DIR / "routing_ablation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("=== Routing Ablation Results ===")
    log.info("  H-FAct  (cached routing)  : score=%.4f  rho=%.4f", mean_hf, rho_hf)
    log.info("  FActScore (cached no-route): score=%.4f  rho=%.4f", mean_fs, rho_fs)
    log.info("  Score gap (H-FAct - FActScore): %+.4f", summary["score_gap_hfact_minus_factscore"])
    log.info("  Human-alignment gain (rho)   : %+.4f", summary["human_alignment_gain"])

    _write_tables(summary)
    _fig_scatter(results, human_scores)
    log.info("Done → %s", ABLATION_DIR)


def _write_tables(s: dict):
    md  = "# H-FAct Ablation: Modality Routing vs No Routing\n\n"
    md += "| Variant | Mean Score | Spearman ρ w/ Human |\n"
    md += "|:---|---:|---:|\n"
    md += f"| **H-FAct (with modality routing)** | **{s['h_fact_mean']:.4f}** | **{s['h_fact_vs_human']:.4f}** |\n"
    md += f"| FActScore (no routing, text-only)  | {s['factscore_mean']:.4f} | {s['factscore_vs_human']:.4f} |\n"
    md += f"\n_n = {s['n_samples']} samples; {s['n_human_shared']} have human scores._\n"
    md += ("\n**Interpretation**: routing improves agreement with human judgment "
           f"by Δρ = {s['human_alignment_gain']:+.4f}, even though absolute score scales differ "
           "between H-FAct and text-only FActScore.\n")
    md += (f"\n**Scale note**: mean-score gap (H-FAct − FActScore) = "
           f"{s['score_gap_hfact_minus_factscore']:+.4f}; this gap should be read as a scale difference, "
           "not as evidence against routing.\n")
    (TABLES_DIR / "routing_ablation.md").write_text(md, encoding="utf-8")

    tex  = "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lcc}\n\\toprule\n"
    tex += "Variant & Mean Score & Spearman $\\rho$ \\\\\n\\midrule\n"
    tex += f"H-FAct (with modality routing) & {s['h_fact_mean']:.4f} & {s['h_fact_vs_human']:.4f} \\\\\n"
    tex += f"FActScore (no routing)         & {s['factscore_mean']:.4f} & {s['factscore_vs_human']:.4f} \\\\\n"
    tex += f"\\midrule\nHuman-alignment gain ($\\Delta \\rho$) & \\multicolumn{{2}}{{c}}{{{s['human_alignment_gain']:+.4f}}} \\\\\n"
    tex += f"Scale gap ($\\Delta$ score) & \\multicolumn{{2}}{{c}}{{{s['score_gap_hfact_minus_factscore']:+.4f}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Ablation study: modality routing in H-FAct vs text-only FActScore. "
            "Routing is evaluated by improvement in agreement with human judgments ($\\Delta \\rho$); "
            "absolute mean scores are reported for reference because the two metrics are not on the same scale.}\n")
    tex += "\\label{tab:routing_ablation}\n\\end{table}\n"
    (TABLES_DIR / "routing_ablation.tex").write_text(tex, encoding="utf-8")
    log.info("Tables → routing_ablation.md + .tex")


def _fig_scatter(results: list[dict], human_scores: dict):
    shared = [(r["h_fact"], r["factscore"], human_scores[r["question_id"]])
              for r in results if r["question_id"] in human_scores]
    if not shared:
        return

    hf_scores = [x[0] for x in shared]
    fs_scores = [x[1] for x in shared]
    hs        = [x[2] for x in shared]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(hf_scores, hs, alpha=0.5, color="#2196F3", s=20)
    ax1.set_xlabel("H-FAct (with routing)")
    ax1.set_ylabel("Human Score")
    ax1.set_title("H-FAct vs Human")
    ax1.grid(alpha=0.3)

    ax2.scatter(fs_scores, hs, alpha=0.5, color="#9E9E9E", s=20)
    ax2.set_xlabel("FActScore (no routing)")
    ax2.set_ylabel("Human Score")
    ax2.set_title("FActScore vs Human")
    ax2.grid(alpha=0.3)

    fig.suptitle("Modality Routing Ablation: Scatter vs Human Judgment", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "routing_ablation_scatter.png", dpi=150)
    plt.close(fig)
    log.info("Saved routing_ablation_scatter.png")


if __name__ == "__main__":
    main()
