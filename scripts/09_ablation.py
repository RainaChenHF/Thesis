"""
Script 09 – Metric Sensitivity / Ablation Analysis.

For each Hybrid-Eval metric, computes:
  - Absolute score across all 5 perturbation conditions
  - Sensitivity = score_golden - score_noise  (absolute drop)
  - Relative drop = (score_golden - score_noise) / max(score_golden, 1e-6)

This is the paper's key ablation table: it shows H-FAct is more sensitive
to evidence degradation than EM/F1, which cannot detect factual-grounding
failures when surface strings partially overlap.

Outputs:
  results/ablation/sensitivity.json
  results/tables/ablation_sensitivity.md
  results/tables/ablation_sensitivity.tex
  results/figures/sensitivity_bar.png
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

log = get_logger("09_ablation")

TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ABLATION_DIR = RESULTS_DIR / "ablation"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

METRIC_LABELS = {
    "em":        "EM",
    "f1":        "F1",
    "joint_hit": "JHit@3",
    "epp_f1":    "EPP-F1",
    "h_fact":    "H-FAct (Ours)",
    "maa":       "MAA (Ours)",
}

CONDITIONS = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
COND_LABELS = {
    "golden":       "Golden",
    "semi_golden":  "Semi-Golden",
    "noise":        "Noise",
    "counterfactual": "Counterfactual",
    "missing":      "Missing",
}


def load_pert_summary() -> dict:
    path = RESULTS_DIR / "perturbation" / "summary.json"
    if not path.exists():
        log.error("Perturbation summary not found — run script 04 first.")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_sensitivity(pert: dict) -> list[dict]:
    """
    For each metric, compute scores across all conditions plus
    sensitivity (golden - noise) and relative drop.
    """
    rows = []
    for key, label in METRIC_LABELS.items():
        golden_score = pert.get("golden", {}).get(key, 0.0)
        noise_score  = pert.get("noise",  {}).get(key, 0.0)
        drop     = round(golden_score - noise_score, 4)
        rel_drop = round(drop / max(golden_score, 1e-6), 4)

        row = {
            "metric": key,
            "label":  label,
            "sensitivity_abs":  drop,
            "sensitivity_rel":  rel_drop,
        }
        for cond in CONDITIONS:
            row[cond] = round(pert.get(cond, {}).get(key, 0.0), 4)
        rows.append(row)

    # Sort by absolute sensitivity (most sensitive first)
    rows.sort(key=lambda r: -r["sensitivity_abs"])
    return rows


def write_markdown(rows: list[dict]) -> str:
    header = ["Metric", "Golden", "Semi-Gold", "Noise", "Counter.", "Missing",
              "Δ (Gold−Noise)", "Rel. Drop"]
    md  = "# Metric Sensitivity to Evidence Perturbation\n\n"
    md += "_Δ = score_golden - score_noise  |  Rel. Drop = Δ / score_golden_\n\n"
    md += "| " + " | ".join(header) + " |\n"
    md += "|:---|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        md += (
            f"| {r['label']} "
            f"| {r['golden']:.3f} "
            f"| {r['semi_golden']:.3f} "
            f"| {r['noise']:.3f} "
            f"| {r['counterfactual']:.3f} "
            f"| {r['missing']:.3f} "
            f"| **{r['sensitivity_abs']:.3f}** "
            f"| {r['sensitivity_rel']:.1%} |\n"
        )
    return md


def write_latex(rows: list[dict]) -> str:
    tex  = "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    tex += ("Metric & Golden & Semi-Gold & Noise & Counter. & Missing"
            " & $\\Delta$ & Rel.Drop \\\\\n\\midrule\n")
    for r in rows:
        label = r["label"].replace("(Ours)", "\\textbf{(Ours)}")
        tex += (
            f"{label} & {r['golden']:.3f} & {r['semi_golden']:.3f} "
            f"& {r['noise']:.3f} & {r['counterfactual']:.3f} "
            f"& {r['missing']:.3f} & {r['sensitivity_abs']:.3f} "
            f"& {r['sensitivity_rel']:.1%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Metric sensitivity to evidence perturbation. "
            "$\\Delta$ = score on golden context $-$ score on noise context.}\n")
    tex += "\\label{tab:ablation_sensitivity}\n\\end{table}\n"
    return tex


def fig_sensitivity(rows: list[dict]):
    labels   = [r["label"] for r in rows]
    abs_drop = [r["sensitivity_abs"] for r in rows]
    colors   = ["#F44336" if "Ours" not in l else "#2196F3" for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, abs_drop, color=colors, alpha=0.85)
    ax.set_xlabel("Golden → Noise score drop (descriptive sensitivity, not overall metric quality)")
    ax.set_title("Metric Sensitivity to Evidence Removal")
    ax.set_xlim(0, max(abs_drop) * 1.25 if abs_drop else 1)
    for bar, val in zip(bars, abs_drop):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    # legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#F44336", alpha=0.85, label="Baseline metrics"),
              Patch(color="#2196F3", alpha=0.85, label="Hybrid-Eval (Ours)")]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity_bar.png", dpi=150)
    plt.close(fig)
    log.info("Saved sensitivity_bar.png")


def fig_condition_profile(rows: list[dict]):
    """Line chart: each metric's score across 5 conditions."""
    conds = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
    clabels = ["Golden", "Semi-Gold", "Noise", "Counter.", "Missing"]

    fig, ax = plt.subplots(figsize=(10, 5))
    styles = ["-o", "--s", "-.^", ":D", "-v", "--*"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548"]
    for i, r in enumerate(rows):
        vals = [r[c] for c in conds]
        ax.plot(clabels, vals, styles[i % len(styles)],
                color=colors[i % len(colors)],
                label=r["label"], linewidth=2, markersize=6)

    ax.set_ylabel("Score")
    ax.set_title("Metric Scores Across Perturbation Conditions")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "condition_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved condition_profile.png")


def main():
    pert = load_pert_summary()
    if not pert:
        return

    rows = compute_sensitivity(pert)

    (ABLATION_DIR / "sensitivity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Sensitivity data saved")

    for r in rows:
        log.info("  %-20s  golden=%.3f  noise=%.3f  Δ=%.3f  rel=%.1f%%",
                 r["label"], r["golden"], r["noise"],
                 r["sensitivity_abs"], r["sensitivity_rel"] * 100)

    md = write_markdown(rows)
    (TABLES_DIR / "ablation_sensitivity.md").write_text(md, encoding="utf-8")
    log.info("Markdown → ablation_sensitivity.md")

    tex = write_latex(rows)
    (TABLES_DIR / "ablation_sensitivity.tex").write_text(tex, encoding="utf-8")
    log.info("LaTeX → ablation_sensitivity.tex")

    fig_sensitivity(rows)
    fig_condition_profile(rows)
    log.info("Ablation analysis complete → %s", ABLATION_DIR)


if __name__ == "__main__":
    main()
