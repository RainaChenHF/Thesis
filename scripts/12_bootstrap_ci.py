#!/usr/bin/env python3
"""
12_bootstrap_ci.py — Bootstrap 95 % confidence intervals
=========================================================
Reads existing prediction JSONL files (no API calls needed) and computes
bootstrap 95 % CIs for:
  1. Baseline metrics  (n=500, 8 metrics)
  2. Perturbation metrics per condition  (n=500 × 5 conditions)
  3. Human-correlation Spearman ρ  (n=100, 8 metrics)

Outputs
-------
  results/bootstrap/baseline_ci.json
  results/bootstrap/perturbation_ci.json
  results/bootstrap/correlation_ci.json
  results/tables/baseline_ci.tex
  results/tables/baseline_ci.md
"""
from __future__ import annotations
import json, random, pathlib, math, sys
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, RANDOM_SEED

B = 2000          # bootstrap resamples
ALPHA = 0.05      # 95 % CI
rng = np.random.RandomState(RANDOM_SEED)

OUT_DIR = RESULTS_DIR / "bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = RESULTS_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = ["em", "f1", "joint_hit", "epp_f1", "h_fact", "maa"]
METRIC_DISPLAY = {
    "em": "EM", "f1": "F1", "joint_hit": "JHit@3",
    "epp_f1": "EPP-F1", "h_fact": "H-FAct", "maa": "MAA",
}
CORR_METRICS = ["em", "f1", "joint_hit", "epp_f1", "h_fact", "maa"]


# ── helpers ────────────────────────────────────────────────────────────────
def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def boot_mean_ci(values: np.ndarray, b: int = B) -> dict:
    """Bootstrap CI for a sample mean."""
    n = len(values)
    point = float(np.mean(values))
    means = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(values[idx])
    lo = float(np.percentile(means, 100 * ALPHA / 2))
    hi = float(np.percentile(means, 100 * (1 - ALPHA / 2)))
    se = float(np.std(means, ddof=1))
    return {"mean": round(point, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "se": round(se, 4)}


def boot_spearman_ci(x: np.ndarray, y: np.ndarray, b: int = B) -> dict:
    """Bootstrap CI for Spearman ρ."""
    n = len(x)
    rho_point, p_point = stats.spearmanr(x, y)
    rhos = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos[i] = r if not math.isnan(r) else 0.0
    lo = float(np.percentile(rhos, 100 * ALPHA / 2))
    hi = float(np.percentile(rhos, 100 * (1 - ALPHA / 2)))
    return {"rho": round(float(rho_point), 4),
            "p": round(float(p_point), 6),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


# ── 1. baseline CI ────────────────────────────────────────────────────────
def baseline_ci() -> dict:
    print("[1/3] Baseline metric CIs  (n=500) …")
    rows = load_jsonl(RESULTS_DIR / "baseline" / "predictions.jsonl")
    result: dict[str, Any] = {"n": len(rows), "B": B, "alpha": ALPHA}
    for m in METRIC_KEYS:
        vals = np.array([float(r.get(m, 0)) for r in rows])
        result[m] = boot_mean_ci(vals)
        print(f"  {METRIC_DISPLAY[m]:>8s}  {result[m]['mean']:.4f}  "
              f"[{result[m]['ci_lo']:.4f}, {result[m]['ci_hi']:.4f}]")
    with open(OUT_DIR / "baseline_ci.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ── 2. perturbation CI ────────────────────────────────────────────────────
CONDS = ["golden", "semi_golden", "noise", "counterfactual", "missing"]
COND_FILE = {
    "golden": "golden_predictions.jsonl",
    "semi_golden": "semi_golden_predictions.jsonl",
    "noise": "noise_predictions.jsonl",
    "counterfactual": "counterfactual_predictions.jsonl",
    "missing": "missing_predictions.jsonl",
}

def perturbation_ci() -> dict:
    print("[2/3] Perturbation metric CIs  (5 × n=500) …")
    result: dict[str, Any] = {"B": B, "alpha": ALPHA}
    for cond in CONDS:
        fpath = RESULTS_DIR / "perturbation" / COND_FILE[cond]
        rows = load_jsonl(fpath)
        cond_res: dict[str, Any] = {"n": len(rows)}
        for m in METRIC_KEYS:
            vals = np.array([float(r.get(m, 0)) for r in rows])
            cond_res[m] = boot_mean_ci(vals)
        # HR-P for counterfactual / missing
        if cond in ("counterfactual", "missing"):
            hr_vals = np.array([float(r.get("hallucinated", 0)) for r in rows])
            cond_res["hr_p"] = boot_mean_ci(hr_vals)
        result[cond] = cond_res
        print(f"  {cond:>16s}  n={len(rows)}")
    with open(OUT_DIR / "perturbation_ci.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ── 3. correlation CI ─────────────────────────────────────────────────────
def correlation_ci() -> dict:
    print("[3/3] Spearman ρ CIs  (n=100) …")
    rows = load_jsonl(RESULTS_DIR / "correlation" / "annotated_sample.jsonl")
    human = np.array([float(r["human_score"]) for r in rows])
    result: dict[str, Any] = {"n": len(rows), "B": B, "alpha": ALPHA}
    for m in CORR_METRICS:
        vals = np.array([float(r.get(m, 0)) for r in rows])
        ci = boot_spearman_ci(vals, human)
        result[m] = ci
        print(f"  {METRIC_DISPLAY[m]:>8s}  ρ={ci['rho']:.4f}  "
              f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  p={ci['p']:.6f}")
    with open(OUT_DIR / "correlation_ci.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ── LaTeX + Markdown tables ───────────────────────────────────────────────
def write_baseline_table(data: dict) -> None:
    """Baseline results with 95 % CI."""
    # LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline Results with 95\% Bootstrap CIs ($B{=}2000$, $n{=}500$).}",
        r"\label{tab:baseline-ci}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Mean} & \textbf{95\% CI} & \textbf{SE} \\",
        r"\midrule",
    ]
    for m in METRIC_KEYS:
        d = data[m]
        name = METRIC_DISPLAY[m]
        ci_str = f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
        lines.append(
            f"{name} & {d['mean']:.3f} & {ci_str} & {d['se']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "baseline_ci.tex").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    # Markdown
    md = ["# Baseline Results with 95% Bootstrap CIs", "",
          "| Metric | Mean | 95% CI | SE |",
          "|:---|---:|:---|---:|"]
    for m in METRIC_KEYS:
        d = data[m]
        md.append(f"| {METRIC_DISPLAY[m]} | {d['mean']:.4f} "
                  f"| [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] "
                  f"| {d['se']:.4f} |")
    md.append(f"\n_B={B}, n={data['n']}, seed={RANDOM_SEED}_")
    (TABLE_DIR / "baseline_ci.md").write_text("\n".join(md) + "\n",
                                              encoding="utf-8")
    print(f"  -> {TABLE_DIR / 'baseline_ci.tex'}")
    print(f"  -> {TABLE_DIR / 'baseline_ci.md'}")


def write_correlation_table(data: dict) -> None:
    """Spearman ρ with 95 % CI."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Spearman $\rho$ with 95\% Bootstrap CIs ($B{=}2000$, $n{=}100$).}",
        r"\label{tab:corr-ci}",
        r"\small",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{$\rho$} & \textbf{$p$} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for m in CORR_METRICS:
        d = data[m]
        name = METRIC_DISPLAY[m]
        p_str = "$<$0.001" if d["p"] < 0.001 else f"{d['p']:.3f}"
        ci_str = f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
        lines.append(f"{name} & {d['rho']:.3f} & {p_str} & {ci_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "correlation_ci.tex").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    # Markdown
    md = ["# Spearman ρ with 95% Bootstrap CIs", "",
          "| Metric | ρ | p-value | 95% CI |",
          "|:---|---:|---:|:---|"]
    for m in CORR_METRICS:
        d = data[m]
        p_str = "<0.001" if d["p"] < 0.001 else f"{d['p']:.4f}"
        md.append(f"| {METRIC_DISPLAY[m]} | {d['rho']:.4f} | {p_str} "
                  f"| [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] |")
    md.append(f"\n_B={B}, n={data['n']}, seed={RANDOM_SEED}_")
    (TABLE_DIR / "correlation_ci.md").write_text("\n".join(md) + "\n",
                                                  encoding="utf-8")
    print(f"  -> {TABLE_DIR / 'correlation_ci.tex'}")
    print(f"  -> {TABLE_DIR / 'correlation_ci.md'}")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Bootstrap 95% Confidence Intervals")
    print(f"  B={B}  seed={RANDOM_SEED}  α={ALPHA}")
    print("=" * 50)

    bl = baseline_ci()
    perturbation_ci()
    corr = correlation_ci()

    print("\nGenerating tables …")
    write_baseline_table(bl)
    write_correlation_table(corr)

    print("\nDone.")


if __name__ == "__main__":
    main()
