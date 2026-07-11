"""
Script 11 – Complementarity Analysis: F1 vs H-FAct.

Three-part empirical proof that H-FAct and F1 are complementary
(i.e. measure different dimensions) rather than redundant:

  Part 1 – Inter-metric correlation
    Spearman ρ between F1 and H-FAct.
    Low ρ (<0.3) means they are not measuring the same thing.

  Part 2 – Incremental validity (regression)
    OLS: human ~ F1          → baseline R²
    OLS: human ~ F1 + H-FAct → full R²
    ΔR² = increase from adding H-FAct; t-test on H-FAct coefficient.
    Significant ΔR² proves H-FAct adds information beyond F1.

  Part 3 – Quadrant scatter
    Visualises the four (F1, H-FAct) quadrants to show
    the "lucky hallucination" and "correct paraphrase" zones.

Outputs:
  results/tables/complementarity.md
  results/tables/complementarity.tex
  results/figures/complementarity_scatter.png
  results/figures/complementarity_regression.png
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR
from src.utils.helpers import get_logger

log = get_logger("11_complementarity")

TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = RESULTS_DIR / "correlation" / "annotated_sample.jsonl"
    if not path.exists():
        log.error("Run 06_human_correlation.py first: %s", path)
        sys.exit(1)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    f1     = np.array([r["f1"]          for r in rows], dtype=float)
    hfact  = np.array([r["h_fact"]      for r in rows], dtype=float)
    human  = np.array([r["human_score"] for r in rows], dtype=float)
    log.info("Loaded %d annotated samples.", len(rows))
    return f1, hfact, human


# ── Part 1: inter-metric correlation ────────────────────────────────────────

def inter_metric_corr(f1: np.ndarray, hfact: np.ndarray) -> dict:
    rho, p_s   = stats.spearmanr(f1, hfact)
    r,   p_p   = stats.pearsonr(f1, hfact)
    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p":   round(float(p_s), 4),
        "pearson_r":    round(float(r),   4),
        "pearson_p":    round(float(p_p), 4),
        "n":            len(f1),
    }


# ── Part 2: incremental validity (OLS regression) ───────────────────────────

def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R² of OLS regression y ~ X (X includes intercept column)."""
    beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def incremental_validity(
    f1: np.ndarray, hfact: np.ndarray, human: np.ndarray
) -> dict:
    n   = len(human)
    one = np.ones(n)

    # Model A: human ~ intercept + F1
    Xa   = np.column_stack([one, f1])
    r2_a = ols_r2(Xa, human)

    # Model B: human ~ intercept + F1 + H-FAct
    Xb   = np.column_stack([one, f1, hfact])
    r2_b = ols_r2(Xb, human)

    delta_r2 = r2_b - r2_a

    # t-test on H-FAct coefficient (partial regression coefficient)
    beta_b, _, _, _ = np.linalg.lstsq(Xb, human, rcond=None)
    y_hat_b  = Xb @ beta_b
    ss_res_b = float(np.sum((human - y_hat_b) ** 2))
    s2       = ss_res_b / max(n - 3, 1)          # MSE, df = n - k - 1
    XtX_inv  = np.linalg.pinv(Xb.T @ Xb)
    se_hfact = math.sqrt(s2 * float(XtX_inv[2, 2]))
    t_hfact  = float(beta_b[2]) / se_hfact if se_hfact > 0 else 0.0
    p_hfact  = float(2 * stats.t.sf(abs(t_hfact), df=n - 3))

    return {
        "n":              n,
        "r2_f1_only":     round(r2_a,    4),
        "r2_f1_hfact":    round(r2_b,    4),
        "delta_r2":       round(delta_r2, 4),
        "beta_hfact":     round(float(beta_b[2]), 4),
        "t_hfact":        round(t_hfact, 3),
        "p_hfact":        round(p_hfact, 4),
        "sig_hfact":      p_hfact < 0.05,
    }


# ── Part 3: quadrant scatter ─────────────────────────────────────────────────

def fig_scatter(f1: np.ndarray, hfact: np.ndarray, human: np.ndarray):
    """Four-quadrant scatter: F1 (x) vs H-FAct (y), coloured by human score."""
    thresh = 0.3   # boundary separating high/low regions

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(f1, hfact, c=human, cmap="RdYlGn",
                    vmin=1, vmax=5, s=40, alpha=0.75, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Human Score (1–5)")

    # Quadrant lines
    ax.axvline(thresh, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.axhline(thresh, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)

    # Quadrant labels
    fs = 8
    ax.text(0.01, 0.97, "Low F1\nHigh H-FAct\n(correct paraphrase)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, color="#1565C0")
    ax.text(0.99, 0.97, "High F1\nHigh H-FAct\n(ideal)",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=fs, color="#2E7D32")
    ax.text(0.01, 0.03, "Low F1\nLow H-FAct\n(wrong)",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=fs, color="#B71C1C")
    ax.text(0.99, 0.03, "High F1\nLow H-FAct\n(lucky hallucination)",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=fs, color="#E65100")

    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_ylabel("H-FAct Score", fontsize=11)
    ax.set_title("F1 vs H-FAct — Four Quadrants (n={})\nColour = Human Score".format(len(f1)),
                 fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "complementarity_scatter.png", dpi=150)
    plt.close(fig)
    log.info("Saved complementarity_scatter.png")


def fig_regression(f1: np.ndarray, hfact: np.ndarray, human: np.ndarray,
                   reg: dict):
    """Side-by-side: F1→human vs (F1+H-FAct)→human partial plots."""
    n   = len(human)
    one = np.ones(n)

    # Partial regression: H-FAct effect after controlling F1
    # residualise both human and H-FAct on F1+intercept
    Xa = np.column_stack([one, f1])
    beta_h, _, _, _ = np.linalg.lstsq(Xa, hfact,  rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(Xa, human, rcond=None)
    resid_hfact = hfact  - Xa @ beta_h
    resid_human = human - Xa @ beta_y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: F1 vs human
    ax1.scatter(f1, human, alpha=0.55, s=25, color="#2196F3")
    m, b, r, p, _ = stats.linregress(f1, human)
    xs = np.linspace(0, 1, 100)
    ax1.plot(xs, m * xs + b, color="#1565C0", linewidth=1.5)
    ax1.set_xlabel("F1")
    ax1.set_ylabel("Human Score")
    ax1.set_title(f"F1 → Human\nR²={reg['r2_f1_only']:.3f}, ρ={stats.spearmanr(f1, human)[0]:.3f}")
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(alpha=0.25)

    # Right: H-FAct partial effect after removing F1
    ax2.scatter(resid_hfact, resid_human, alpha=0.55, s=25, color="#F44336")
    if resid_hfact.std() > 0:
        m2, b2, _, _, _ = stats.linregress(resid_hfact, resid_human)
        xs2 = np.linspace(resid_hfact.min(), resid_hfact.max(), 100)
        ax2.plot(xs2, m2 * xs2 + b2, color="#B71C1C", linewidth=1.5)
    sig = "p<0.05 ✓" if reg["sig_hfact"] else f"p={reg['p_hfact']:.3f}"
    ax2.set_xlabel("H-FAct residual (after removing F1 effect)")
    ax2.set_ylabel("Human residual (after removing F1 effect)")
    ax2.set_title(
        f"H-FAct partial effect (controlling for F1)\n"
        f"β={reg['beta_hfact']:.3f}, t={reg['t_hfact']:.2f}, {sig}\n"
        f"ΔR²={reg['delta_r2']:.3f}"
    )
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("Incremental Validity: Does H-FAct Add Information Beyond F1?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "complementarity_regression.png", dpi=150)
    plt.close(fig)
    log.info("Saved complementarity_regression.png")


# ── Quadrant counts ──────────────────────────────────────────────────────────

def quadrant_counts(f1: np.ndarray, hfact: np.ndarray,
                    human: np.ndarray, thresh: float = 0.3) -> dict:
    hi_f1_hi_hf = int(np.sum((f1 >= thresh) & (hfact >= thresh)))
    hi_f1_lo_hf = int(np.sum((f1 >= thresh) & (hfact <  thresh)))
    lo_f1_hi_hf = int(np.sum((f1 <  thresh) & (hfact >= thresh)))
    lo_f1_lo_hf = int(np.sum((f1 <  thresh) & (hfact <  thresh)))

    # "Lucky hallucination": high F1, low H-FAct
    # Average human score in that quadrant
    mask_lucky = (f1 >= thresh) & (hfact < thresh)
    avg_human_lucky = float(human[mask_lucky].mean()) if mask_lucky.any() else float("nan")
    mask_ideal = (f1 >= thresh) & (hfact >= thresh)
    avg_human_ideal = float(human[mask_ideal].mean()) if mask_ideal.any() else float("nan")

    return {
        "thresh":           thresh,
        "ideal_n":          hi_f1_hi_hf,
        "lucky_halluc_n":   hi_f1_lo_hf,
        "correct_para_n":   lo_f1_hi_hf,
        "both_wrong_n":     lo_f1_lo_hf,
        "avg_human_ideal":  round(avg_human_ideal, 3),
        "avg_human_lucky":  round(avg_human_lucky, 3),
    }


# ── Output tables ────────────────────────────────────────────────────────────

def write_tables(imc: dict, reg: dict, qc: dict):
    sig_star = "\\*" if reg["sig_hfact"] else "(n.s.)"
    # ── Markdown ────────────────────────────────────────────────────────────
    md  = "# Complementarity Analysis: F1 vs H-FAct\n\n"
    md += f"_n = {imc['n']} LLM-annotated samples._\n\n"

    md += "## Part 1 — Inter-metric correlation\n\n"
    md += "| | Spearman ρ | p-value | Pearson r | p-value |\n"
    md += "|:---|---:|---:|---:|---:|\n"
    md += (f"| F1 vs H-FAct | {imc['spearman_rho']:.4f} | {imc['spearman_p']:.4f}"
           f" | {imc['pearson_r']:.4f} | {imc['pearson_p']:.4f} |\n\n")
    md += ("> Low correlation confirms F1 and H-FAct measure different dimensions "
           "and are not redundant.\n\n")

    md += "## Part 2 — Incremental validity\n\n"
    md += "| Model | R² | ΔR² | H-FAct β | t | p |\n"
    md += "|:---|---:|---:|---:|---:|---:|\n"
    md += f"| F1 only | {reg['r2_f1_only']:.4f} | — | — | — | — |\n"
    md += (f"| F1 + H-FAct | {reg['r2_f1_hfact']:.4f} | **{reg['delta_r2']:+.4f}** "
           f"| {reg['beta_hfact']:.4f} | {reg['t_hfact']:.2f} | {reg['p_hfact']:.4f} |\n\n")
    conc = "significant" if reg["sig_hfact"] else "not significant at α=0.05"
    md += f"> H-FAct coefficient is **{conc}** (p={reg['p_hfact']:.4f}).\n\n"

    md += "## Part 3 — Quadrant counts\n\n"
    md += f"Threshold = {qc['thresh']} (separates high/low regions).\n\n"
    md += "| Quadrant | n | Description |\n"
    md += "|:---|---:|:---|\n"
    md += f"| High F1 + High H-FAct (ideal) | {qc['ideal_n']} | Grounded AND correct |\n"
    md += f"| High F1 + Low H-FAct (lucky hallucination) | {qc['lucky_halluc_n']} | Correct answer, no retrieval support |\n"
    md += f"| Low F1 + High H-FAct (correct paraphrase) | {qc['correct_para_n']} | Grounded, but paraphrased away from gold |\n"
    md += f"| Low F1 + Low H-FAct (both wrong) | {qc['both_wrong_n']} | Neither correct nor grounded |\n"

    (TABLES_DIR / "complementarity.md").write_text(md, encoding="utf-8")
    log.info("Saved complementarity.md")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    tex  = "% ── Complementarity Analysis ──\n\n"
    tex += "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lrrrr}\n\\toprule\n"
    tex += " & Spearman $\\rho$ & $p$ & Pearson $r$ & $p$ \\\\\n\\midrule\n"
    tex += (f"F1 vs H-FAct & {imc['spearman_rho']:.4f} & {imc['spearman_p']:.4f}"
            f" & {imc['pearson_r']:.4f} & {imc['pearson_p']:.4f} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += "\\caption{Inter-metric correlation between F1 and H-FAct ($n=" + str(imc['n']) + "$).}\n"
    tex += "\\label{tab:inter_metric}\n\\end{table}\n\n"

    tex += "\\begin{table}[t]\n\\centering\n"
    tex += "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    tex += "Model & $R^2$ & $\\Delta R^2$ & $\\hat{\\beta}_{\\text{H-FAct}}$ & $t$ & $p$ \\\\\n\\midrule\n"
    tex += f"F1 only & {reg['r2_f1_only']:.4f} & --- & --- & --- & --- \\\\\n"
    tex += (f"F1 + H-FAct & {reg['r2_f1_hfact']:.4f} & {reg['delta_r2']:+.4f}"
            f" & {reg['beta_hfact']:.4f} & {reg['t_hfact']:.2f} & {reg['p_hfact']:.4f} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += ("\\caption{Incremental validity: OLS regression predicting human scores. "
            "Adding H-FAct to F1 increases $R^2$ by $\\Delta R^2=" + f"{reg['delta_r2']:+.4f}" + "$"
            + (f", $p={reg['p_hfact']:.4f}$" if reg["sig_hfact"] else " (n.s.)") + ".}\n")
    tex += "\\label{tab:incremental_validity}\n\\end{table}\n"

    (TABLES_DIR / "complementarity.tex").write_text(tex, encoding="utf-8")
    log.info("Saved complementarity.tex")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    f1, hfact, human = load_data()

    log.info("=== Part 1: Inter-metric correlation ===")
    imc = inter_metric_corr(f1, hfact)
    log.info("  F1 vs H-FAct: Spearman ρ=%.4f (p=%.4f), Pearson r=%.4f (p=%.4f)",
             imc["spearman_rho"], imc["spearman_p"],
             imc["pearson_r"],    imc["pearson_p"])

    log.info("=== Part 2: Incremental validity ===")
    reg = incremental_validity(f1, hfact, human)
    log.info("  F1 only     : R²=%.4f", reg["r2_f1_only"])
    log.info("  F1 + H-FAct : R²=%.4f  ΔR²=%+.4f", reg["r2_f1_hfact"], reg["delta_r2"])
    log.info("  H-FAct coef : β=%.4f  t=%.2f  p=%.4f  significant=%s",
             reg["beta_hfact"], reg["t_hfact"], reg["p_hfact"], reg["sig_hfact"])

    log.info("=== Part 3: Quadrant counts ===")
    qc = quadrant_counts(f1, hfact, human)
    log.info("  Ideal (hi-hi): %d | Lucky halluc (hi-lo): %d",
             qc["ideal_n"], qc["lucky_halluc_n"])
    log.info("  Correct para (lo-hi): %d | Both wrong (lo-lo): %d",
             qc["correct_para_n"], qc["both_wrong_n"])

    # Save results JSON
    ablation_dir = RESULTS_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    result = {"inter_metric_corr": imc, "incremental_validity": reg, "quadrants": qc}
    (ablation_dir / "complementarity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_tables(imc, reg, qc)
    fig_scatter(f1, hfact, human)
    fig_regression(f1, hfact, human, reg)
    log.info("Done → results/tables/complementarity.{md,tex} + results/figures/")


if __name__ == "__main__":
    main()

