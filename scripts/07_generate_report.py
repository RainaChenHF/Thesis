"""
Script 07 – Generate all paper-ready tables and figures.

Reads from results/ and outputs:
  results/tables/  – markdown + LaTeX tables
  results/figures/ – PNG charts
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR
from src.utils.helpers import get_logger

log = get_logger("07_report")

TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        log.warning("Missing: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ── Table 1: Baseline summary ──────────────────────────────────────────────
def table_baseline(summary: dict) -> str:
    rows = [
        ("Metric", "Score"),
        ("EM",            f"{summary.get('em',0):.3f}"),
        ("F1",            f"{summary.get('f1',0):.3f}"),
        ("Table Hit@3",   f"{summary.get('table_hit',0):.3f}"),
        ("Passage Hit@3", f"{summary.get('passage_hit',0):.3f}"),
        ("JHit@3",        f"{summary.get('joint_hit',0):.3f}"),
        ("EPP-F1",        f"{summary.get('epp_f1',0):.3f}"),
        ("H-FAct",        f"{summary.get('h_fact',0):.3f}"),
        ("MAA",           f"{summary.get('maa',0):.3f}"),
    ]
    if summary.get("cwrs") is not None:
        rows.append(("CWRS", f"{summary.get('cwrs',0):.3f}"))
    md = "| " + " | ".join(rows[0]) + " |\n"
    md += "|:---|---:|\n"
    for r in rows[1:]:
        md += "| " + " | ".join(r) + " |\n"
    return md


def latex_table_baseline(summary: dict) -> str:
    metrics = [
        ("EM",            summary.get("em", 0)),
        ("F1",            summary.get("f1", 0)),
        ("Table Hit@3",   summary.get("table_hit", 0)),
        ("Passage Hit@3", summary.get("passage_hit", 0)),
        ("JHit@3",        summary.get("joint_hit", 0)),
        ("EPP-F1",        summary.get("epp_f1", 0)),
        ("H-FAct",        summary.get("h_fact", 0)),
        ("MAA",           summary.get("maa", 0)),
    ]
    tex  = "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lc}\n\\toprule\n"
    tex += "Metric & Score \\\\\n\\midrule\n"
    for name, val in metrics:
        tex += f"{name} & {val:.3f} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Baseline performance on HybridQA dev set "
            "(500 samples, BM25 retrieval + Qwen-max generation).}\n")
    tex += "\\label{tab:baseline}\n\\end{table}\n"
    return tex


# ── Table 2: Perturbation ──────────────────────────────────────────────────
def table_perturbation(pert: dict) -> str:
    conds = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
    header = ["Condition", "EM", "F1", "H-FAct", "EPP-F1", "HR-P"]
    md = "| " + " | ".join(header) + " |\n"
    md += "|:---|---:|---:|---:|---:|---:|\n"
    for c in conds:
        d = pert.get(c, {})
        hrp = f"{d.get('hr_p', 0):.3f}" if isinstance(d.get("hr_p"), float) else "—"
        md += (f"| {c} | {d.get('em',0):.3f} | {d.get('f1',0):.3f} |"
               f" {d.get('h_fact',0):.3f} | {d.get('epp_f1',0):.3f} | {hrp} |\n")
    return md


def latex_table_perturbation(pert: dict) -> str:
    conds = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
    clabels = ["Golden", "Semi-Gold.", "Noise", "Counterfact.", "Missing"]
    tex  = "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    tex += "Condition & EM & F1 & H-FAct & EPP-F1 & HR-P \\\\\n\\midrule\n"
    for c, cl in zip(conds, clabels):
        d = pert.get(c, {})
        hrp = f"{d.get('hr_p',0):.3f}" if isinstance(d.get("hr_p"), float) else "---"
        tex += (f"{cl} & {d.get('em',0):.3f} & {d.get('f1',0):.3f} "
                f"& {d.get('h_fact',0):.3f} & {d.get('epp_f1',0):.3f} "
                f"& {hrp} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Metric scores under five perturbation conditions "
            "(500 samples each). HR-P applies only to counterfactual/missing conditions.}\n")
    tex += "\\label{tab:perturbation}\n\\end{table}\n"
    return tex


# ── Table 3: Stratified ────────────────────────────────────────────────────
def table_stratified(strat: dict) -> str:
    header = ["Level", "n", "EM", "H-FAct", "EPP-F1", "JHit@3", "MAA"]
    md = "| " + " | ".join(header) + " |\n"
    md += "|:---|---:|---:|---:|---:|---:|---:|\n"
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        d = strat.get(lvl, {})
        md += (f"| {lvl} | {d.get('n',0)} | {d.get('em',0):.3f} |"
               f" {d.get('h_fact',0):.3f} | {d.get('epp_f1',0):.3f} |"
               f" {d.get('joint_hit',0):.3f} | {d.get('maa',0):.3f} |\n")
    if strat.get("cwrs") is not None:
        md += f"| **CWRS** | — | — | **{strat['cwrs']:.4f}** | — | — | — |\n"
    return md


def latex_table_stratified(strat: dict) -> str:
    tex  = "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    tex += "Level & n & EM & H-FAct & EPP-F1 & JHit@3 & MAA \\\\\n\\midrule\n"
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        d = strat.get(lvl, {})
        tex += (f"{lvl} & {d.get('n',0)} & {d.get('em',0):.3f} "
                f"& {d.get('h_fact',0):.3f} & {d.get('epp_f1',0):.3f} "
                f"& {d.get('joint_hit',0):.3f} & {d.get('maa',0):.3f} \\\\\n")
    if strat.get("cwrs") is not None:
        tex += f"\\midrule\nCWRS & \\multicolumn{{6}}{{r}}{{{strat['cwrs']:.4f}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Performance stratified by question complexity (L1--L5). "
            "CWRS = complexity-weighted reasoning score over H-FAct.}\n")
    tex += "\\label{tab:stratified}\n\\end{table}\n"
    return tex


# ── Table 4: Correlation ───────────────────────────────────────────────────
def table_correlation(corr: dict) -> str:
    header = ["Metric", "Spearman ρ", "p-value"]
    md = "| " + " | ".join(header) + " |\n"
    md += "|:---|---:|---:|\n"
    for m, v in sorted(corr.items(), key=lambda x: -x[1]["rho"]):
        md += f"| {m} | {v['rho']:.4f} | {v['pvalue']:.4f} |\n"
    return md


def table_asset_registry(paths: dict[str, str]) -> str:
    rows = [
        ("Main table", paths.get("comparison_tex", "—"), "Metric comparison with related work"),
        ("Baseline CI", paths.get("baseline_ci_tex", "—"), "Bootstrap 95% CI for baseline metrics"),
        ("Correlation CI", paths.get("corr_ci_tex", "—"), "Bootstrap 95% CI for human correlations"),
        ("Sensitivity ablation", paths.get("ablation_tex", "—"), "Metric sensitivity to perturbation"),
        ("Routing ablation", paths.get("routing_tex", "—"), "H-FAct vs FActScore with aligned cached scores"),
        ("Complementarity", paths.get("complementarity_tex", "—"), "F1 vs H-FAct incremental validity"),
        ("Advisor ranking", paths.get("advisor_ranking_tex", "—"), "Publication-style ranking of all metrics"),
        ("Advisor rescue", paths.get("advisor_rescue_tex", "—"), "Rescue-rate summary"),
        ("Advisor perturbation", paths.get("advisor_perturb_tex", "—"), "Perturbation comparison summary"),
    ]
    md = "| Asset | Path | Purpose |\n|:---|:---|:---|\n"
    for name, path, purpose in rows:
        md += f"| {name} | `{path}` | {purpose} |\n"
    return md


# ── Figures ────────────────────────────────────────────────────────────────
def fig_perturbation(pert: dict):
    conds   = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
    labels  = ["Golden", "Semi-Gold", "Noise", "Counter.", "Missing"]
    metrics = {"EM": "em", "F1": "f1", "H-FAct": "h_fact", "EPP": "epp_f1"}
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    x = np.arange(len(conds))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, key) in enumerate(metrics.items()):
        vals = [pert.get(c, {}).get(key, 0) for c in conds]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Metric Sensitivity Across Perturbation Conditions")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved perturbation_bar.png")


def fig_stratified(strat: dict):
    levels  = ["L1", "L2", "L3", "L4", "L5"]
    metrics = {"EM": "em", "H-FAct": "h_fact", "JHit@3": "joint_hit"}
    colors  = ["#2196F3", "#F44336", "#4CAF50"]

    x = np.arange(len(levels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, key) in enumerate(metrics.items()):
        vals = [strat.get(lvl, {}).get(key, 0) for lvl in levels]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{l}\n(n={strat.get(l,{}).get('n',0)})" for l in levels])
    ax.set_ylabel("Score")
    ax.set_title("Performance by Question Complexity (STARK L1-L5)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stratified_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved stratified_bar.png")


def fig_correlation(corr: dict):
    metrics = sorted(corr.items(), key=lambda x: -x[1]["rho"])
    labels  = [m for m, _ in metrics]
    rhos    = [v["rho"] for _, v in metrics]
    # Blue = Hybrid-Eval, Red = baseline (EM/F1), Orange = sub-metrics
    hybrid_eval = {"h_fact", "epp_f1", "joint_hit", "maa", "table_hit", "passage_hit"}
    baseline_m  = {"em", "f1"}
    colors = []
    for l in labels:
        if l in baseline_m:
            colors.append("#F44336")
        elif l in hybrid_eval:
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    bars = ax.barh(labels, rhos, color=colors, alpha=0.85)
    ax.set_xlabel("Spearman ρ with Human Score")
    ax.set_title("Metric Correlation with Human Judgment")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, rho in zip(bars, rhos):
        xpos = rho + 0.02 if rho >= 0 else rho - 0.02
        ha   = "left" if rho >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{rho:.3f}", va="center", ha=ha, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    legend = [Patch(color="#F44336", alpha=0.85, label="Baseline (EM/F1)"),
              Patch(color="#2196F3", alpha=0.85, label="Hybrid-Eval (Ours)"),
              Patch(color="#FF9800", alpha=0.85, label="Sub-metrics")]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved correlation_bar.png")


def fig_comparison(cmp_corr: dict, our_corr: dict):
    """Bar chart comparing Hybrid-Eval vs related-work metric correlations."""
    if not cmp_corr:
        return
    # Related-work metrics to display (pick best representative per paper)
    related = {
        "EM":               our_corr.get("em",         {}).get("rho", 0),
        "F1":               our_corr.get("f1",         {}).get("rho", 0),
        "BLEU-1":           cmp_corr.get("bleu1",      {}).get("rho", 0),
        "ROUGE-L":          cmp_corr.get("rouge_l",    {}).get("rho", 0),
        "BERTScore":        cmp_corr.get("bertscore",  {}).get("rho", 0),
        "FActScore":        cmp_corr.get("factscore",  {}).get("rho", 0),
        "RAGAS-Faith.":     cmp_corr.get("ragas_faithfulness", {}).get("rho", 0),
        "G-Eval (avg)":     cmp_corr.get("geval_avg",  {}).get("rho", 0),
        "H-FAct (Ours)":    our_corr.get("h_fact",     {}).get("rho", 0),
        "EPP-F1 (Ours)":    our_corr.get("epp_f1",     {}).get("rho", 0),
        "JHit@3 (Ours)":    our_corr.get("joint_hit",  {}).get("rho", 0),
    }
    labels = list(related.keys())
    rhos   = list(related.values())
    colors = ["#2196F3" if "(Ours)" in l else "#9E9E9E" for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), rhos, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ with Human Score")
    ax.set_title("Hybrid-Eval vs Related Metrics: Human Correlation")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(min(min(rhos) - 0.1, -0.2), max(max(rhos) + 0.15, 1.0))
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, rho + 0.01,
                f"{rho:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend = [Patch(color="#2196F3", alpha=0.85, label="Hybrid-Eval (Ours)"),
              Patch(color="#9E9E9E", alpha=0.85, label="Related work")]
    ax.legend(handles=legend)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved comparison_bar.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    baseline = load_json(RESULTS_DIR / "baseline"      / "summary.json")
    pert     = load_json(RESULTS_DIR / "perturbation"  / "summary.json")
    strat    = load_json(RESULTS_DIR / "stratified"    / "summary.json")
    corr     = load_json(RESULTS_DIR / "correlation"   / "summary.json")
    cmp_corr = load_json(RESULTS_DIR / "comparison"    / "correlation.json")
    comparison_tex = (TABLES_DIR / "comparison.tex").read_text(encoding="utf-8") if (TABLES_DIR / "comparison.tex").exists() else ""
    comparison_md  = (TABLES_DIR / "comparison.md").read_text(encoding="utf-8") if (TABLES_DIR / "comparison.md").exists() else ""
    complementarity_md = (TABLES_DIR / "complementarity.md").read_text(encoding="utf-8") if (TABLES_DIR / "complementarity.md").exists() else ""
    complementarity_tex = (TABLES_DIR / "complementarity.tex").read_text(encoding="utf-8") if (TABLES_DIR / "complementarity.tex").exists() else ""
    routing_md = (TABLES_DIR / "routing_ablation.md").read_text(encoding="utf-8") if (TABLES_DIR / "routing_ablation.md").exists() else ""
    routing_tex = (TABLES_DIR / "routing_ablation.tex").read_text(encoding="utf-8") if (TABLES_DIR / "routing_ablation.tex").exists() else ""
    ablation_md = (TABLES_DIR / "ablation_sensitivity.md").read_text(encoding="utf-8") if (TABLES_DIR / "ablation_sensitivity.md").exists() else ""
    ablation_tex = (TABLES_DIR / "ablation_sensitivity.tex").read_text(encoding="utf-8") if (TABLES_DIR / "ablation_sensitivity.tex").exists() else ""
    baseline_ci_md = (TABLES_DIR / "baseline_ci.md").read_text(encoding="utf-8") if (TABLES_DIR / "baseline_ci.md").exists() else ""
    baseline_ci_tex = (TABLES_DIR / "baseline_ci.tex").read_text(encoding="utf-8") if (TABLES_DIR / "baseline_ci.tex").exists() else ""
    corr_ci_md = (TABLES_DIR / "correlation_ci.md").read_text(encoding="utf-8") if (TABLES_DIR / "correlation_ci.md").exists() else ""
    corr_ci_tex = (TABLES_DIR / "correlation_ci.tex").read_text(encoding="utf-8") if (TABLES_DIR / "correlation_ci.tex").exists() else ""
    advisor_dir = RESULTS_DIR / "advisor_demo"
    advisor_ranking_md = (advisor_dir / "ranking_table.md").read_text(encoding="utf-8") if (advisor_dir / "ranking_table.md").exists() else ""
    advisor_ranking_tex = (advisor_dir / "ranking_table.tex").read_text(encoding="utf-8") if (advisor_dir / "ranking_table.tex").exists() else ""
    advisor_rescue_md = (advisor_dir / "rescue_table.md").read_text(encoding="utf-8") if (advisor_dir / "rescue_table.md").exists() else ""
    advisor_rescue_tex = (advisor_dir / "rescue_table.tex").read_text(encoding="utf-8") if (advisor_dir / "rescue_table.tex").exists() else ""
    advisor_perturb_md = (advisor_dir / "perturbation_compare.md").read_text(encoding="utf-8") if (advisor_dir / "perturbation_compare.md").exists() else ""
    advisor_perturb_tex = (advisor_dir / "perturbation_compare.tex").read_text(encoding="utf-8") if (advisor_dir / "perturbation_compare.tex").exists() else ""

    asset_paths = {
        "comparison_tex": "results/tables/comparison.tex",
        "baseline_ci_tex": "results/tables/baseline_ci.tex",
        "corr_ci_tex": "results/tables/correlation_ci.tex",
        "ablation_tex": "results/tables/ablation_sensitivity.tex",
        "routing_tex": "results/tables/routing_ablation.tex",
        "complementarity_tex": "results/tables/complementarity.tex",
        "advisor_ranking_tex": "results/advisor_demo/ranking_table.tex",
        "advisor_rescue_tex": "results/advisor_demo/rescue_table.tex",
        "advisor_perturb_tex": "results/advisor_demo/perturbation_compare.tex",
    }

    # Markdown tables
    md = "# Hybrid-Eval Results\n\n"
    md += "## Paper asset registry\n\n" + table_asset_registry(asset_paths) + "\n"
    md += "## Table 1 – Baseline summary\n\n" + table_baseline(baseline) + "\n"
    md += "## Table 2 – Perturbation experiment\n\n"    + table_perturbation(pert)  + "\n"
    md += "## Table 3 – Complexity stratification\n\n"  + table_stratified(strat)   + "\n"
    md += "## Table 4 – Human correlation\n\n"          + table_correlation(corr)   + "\n"
    if baseline_ci_md:
        md += "## Table 5 – Baseline bootstrap CI\n\n" + baseline_ci_md + "\n"
    if corr_ci_md:
        md += "## Table 6 – Correlation bootstrap CI\n\n" + corr_ci_md + "\n"
    if comparison_md:
        md += "## Table 7 – Comparison with related work\n\n" + comparison_md + "\n"
    if ablation_md:
        md += "## Table 8 – Sensitivity ablation (descriptive)\n\n" + ablation_md + "\n"
    if routing_md:
        md += "## Table 9 – Routing ablation\n\n" + routing_md + "\n"
    if complementarity_md:
        md += "## Table 10 – Complementarity\n\n" + complementarity_md + "\n"
    if advisor_ranking_md:
        md += "## Advisor bundle – ranking table\n\n" + advisor_ranking_md + "\n"
    if advisor_rescue_md:
        md += "## Advisor bundle – rescue table\n\n" + advisor_rescue_md + "\n"
    if advisor_perturb_md:
        md += "## Advisor bundle – perturbation comparison\n\n" + advisor_perturb_md + "\n"

    # MiRAGE group metrics
    mirage = pert.get("mirage", {})
    if mirage:
        md += "## MiRAGE Adaptability Metrics (Park et al., NAACL 2025)\n\n"
        md += "| Metric | Value | Description |\n"
        md += "|:---|---:|:---|\n"
        md += f"| NV (Noise Vulnerability) | {mirage.get('mirage_NV', 0):.3f} | Oracle correct but noise context wrong |\n"
        md += f"| CA (Context Adherence) | {mirage.get('mirage_CA', 0):.3f} | Correct under both oracle and mixed context |\n"
        md += f"| CI (Context Insensitivity) | {mirage.get('mirage_CI', 0):.3f} | Fails even with oracle context |\n"
        md += f"| CM (Consistent Miss) | {mirage.get('mirage_CM', 0):.3f} | Fails under all three context conditions |\n"
        md += f"\n_n = {mirage.get('mirage_n', 0)} shared question IDs across noise/golden/semi-golden._\n\n"

    (TABLES_DIR / "all_results.md").write_text(md, encoding="utf-8")
    log.info("Markdown tables → %s", TABLES_DIR / "all_results.md")

    # LaTeX tables
    tex = "% ── Hybrid-Eval LaTeX Tables ──\n\n"
    if baseline: tex += latex_table_baseline(baseline) + "\n"
    if pert:     tex += latex_table_perturbation(pert)  + "\n"
    if strat:    tex += latex_table_stratified(strat)   + "\n"
    if baseline_ci_tex: tex += baseline_ci_tex + "\n"
    if corr_ci_tex: tex += corr_ci_tex + "\n"
    if comparison_tex: tex += comparison_tex + "\n"
    if ablation_tex: tex += ablation_tex + "\n"
    if routing_tex: tex += routing_tex + "\n"
    if complementarity_tex: tex += complementarity_tex + "\n"
    if advisor_ranking_tex: tex += advisor_ranking_tex + "\n"
    if advisor_rescue_tex: tex += advisor_rescue_tex + "\n"
    if advisor_perturb_tex: tex += advisor_perturb_tex + "\n"
    (TABLES_DIR / "all_results.tex").write_text(tex, encoding="utf-8")
    log.info("LaTeX tables → %s", TABLES_DIR / "all_results.tex")

    # Figures
    if pert:  fig_perturbation(pert)
    if strat: fig_stratified(strat)
    if corr:  fig_correlation(corr)
    if cmp_corr and corr: fig_comparison(cmp_corr, corr)

    log.info("All outputs ready in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
