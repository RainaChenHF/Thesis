#!/usr/bin/env python3
"""
Script 13 -- Multiple comparison correction (Bonferroni) + Williams test
for dependent correlations.

Inputs:
  results/comparison/correlation.json        -- 19 baseline metrics rho+p
  results/correlation/summary.json           -- our metrics rho+p (h_fact v1, MAA, EPP-F1, hits)
  results/correlation_v3/annotated_sample_v3.jsonl  -- per-sample h_fact_v3 (regenerated 2026-06)
  results/comparison/predictions.jsonl       -- per-sample bertscore + others
  results/correlation/annotated_sample.jsonl -- per-sample human_score + h_fact_v1

Outputs:
  results/comparison/multiple_comparison.json
  results/tables/multiple_comparison.md
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from scipy.stats import spearmanr, t as t_dist

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR

ALPHA = 0.05


def williams_t(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """
    Williams (1959) test for comparing two dependent correlations
    r(X1, Y) vs r(X1, Z) when Y and Z are themselves correlated by r(Y, Z).

    Convention here:
      r12 = corr(human, A)
      r13 = corr(human, B)
      r23 = corr(A, B)

    Returns: (t_stat, two_sided_p)
    """
    det_R = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    if det_R <= 0:
        return float("nan"), float("nan")
    mean_r = (r12 + r13) / 2.0
    numerator = (r12 - r13) * math.sqrt((n - 1) * (1 + r23))
    denom_sq = (
        2 * ((n - 1) / (n - 3)) * det_R
        + mean_r**2 * (1 - r23) ** 3
    )
    if denom_sq <= 0:
        return float("nan"), float("nan")
    t_stat = numerator / math.sqrt(denom_sq)
    df = n - 3
    p_two = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    return t_stat, p_two


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    # 1) Collect rho + p for every metric we want to include in the correction.
    cmp_corr = json.loads((RESULTS_DIR / "comparison" / "correlation.json").read_text())
    our_corr = json.loads((RESULTS_DIR / "correlation" / "summary.json").read_text())

    metrics: dict[str, dict] = {}
    for name, v in cmp_corr.items():
        metrics[name] = {"rho": v["rho"], "p_raw": v["pvalue"], "source": "comparison"}
    for name, v in our_corr.items():
        # don't overwrite a comparison entry; the "our" file is for h_fact v1, MAA, hits, EPP-F1
        if name in metrics:
            continue
        metrics[name] = {"rho": v["rho"], "p_raw": v["pvalue"], "source": "ours"}

    # h_fact_v3: derived from summary_v3.json (already computed Spearman on 100 samples)
    v3_summary = json.loads((RESULTS_DIR / "summary_v3.json").read_text())
    if "h_fact_v3_rho" in v3_summary:
        # we'll recompute p from scratch below for Williams; for Bonferroni we need p_raw
        # use spearmanr on raw v3 per-sample data
        v3_path = RESULTS_DIR / "correlation_v3" / "annotated_sample_v3.jsonl"
        if v3_path.exists():
            v3_rows = load_jsonl(v3_path)
            hs, v3s = [], []
            for r in v3_rows:
                if r.get("human_score") is not None and r.get("h_fact_v3") is not None:
                    hs.append(r["human_score"]); v3s.append(r["h_fact_v3"])
            rho_v3, p_v3 = spearmanr(hs, v3s)
            metrics["h_fact_v3"] = {
                "rho": float(rho_v3),
                "p_raw": float(p_v3),
                "source": "ours_v3",
            }
            print(f"[v3] recomputed h_fact_v3 rho={rho_v3:.4f}  p={p_v3:.2e}  (n={len(hs)})")
        else:
            print("[!] correlation_v3/annotated_sample_v3.jsonl not found — h_fact_v3 row will be skipped")

    N = len(metrics)
    alpha_bonf = ALPHA / N

    # 2) Bonferroni-adjusted p and significance flag
    for name, v in metrics.items():
        v["p_bonf"] = min(1.0, v["p_raw"] * N)  # adjusted p-value
        v["sig_raw_05"] = v["p_raw"] < ALPHA
        v["sig_bonf"] = v["p_raw"] < alpha_bonf

    # 3) Williams test: H-FAct v3 vs BERTScore on shared sample set
    williams_out = None
    v3_path = RESULTS_DIR / "correlation_v3" / "annotated_sample_v3.jsonl"
    bs_path = RESULTS_DIR / "comparison" / "predictions.jsonl"
    if v3_path.exists() and bs_path.exists():
        v3_by_qid = {r["question_id"]: r for r in load_jsonl(v3_path)}
        bs_by_qid = {r["question_id"]: r for r in load_jsonl(bs_path)}
        common = sorted(set(v3_by_qid) & set(bs_by_qid))
        hs, v3, bs = [], [], []
        for qid in common:
            r3 = v3_by_qid[qid]; rb = bs_by_qid[qid]
            if (r3.get("human_score") is not None
                and r3.get("h_fact_v3") is not None
                and rb.get("bertscore") is not None):
                hs.append(r3["human_score"])
                v3.append(r3["h_fact_v3"])
                bs.append(rb["bertscore"])
        n = len(hs)
        r12, _ = spearmanr(hs, v3)   # human vs v3
        r13, _ = spearmanr(hs, bs)   # human vs bertscore
        r23, _ = spearmanr(v3, bs)   # v3 vs bertscore
        t_stat, p_w = williams_t(float(r12), float(r13), float(r23), n)
        williams_out = {
            "comparison": "H-FAct v3 vs BERTScore (human as reference)",
            "n": n,
            "rho_v3_vs_human": round(float(r12), 4),
            "rho_bertscore_vs_human": round(float(r13), 4),
            "rho_v3_vs_bertscore": round(float(r23), 4),
            "t_stat": round(float(t_stat), 4),
            "p_two_sided": round(float(p_w), 4),
            "df": n - 3,
            "interpretation": (
                "Two correlations statistically indistinguishable (p > 0.05)"
                if p_w > 0.05
                else "Significant difference (p < 0.05)"
            ),
        }
        print(f"[Williams] n={n}  rho_v3={r12:.4f}  rho_bs={r13:.4f}  rho_v3_bs={r23:.4f}")
        print(f"[Williams] t={t_stat:.4f}  p={p_w:.4f}")
    else:
        print("[!] Williams test skipped: v3 per-sample data or bertscore data missing")

    # 4) Write outputs
    out_dir = RESULTS_DIR / "comparison"
    out_dir.mkdir(exist_ok=True)
    out = {
        "alpha": ALPHA,
        "n_metrics": N,
        "alpha_bonferroni": alpha_bonf,
        "per_metric": metrics,
        "williams_v3_vs_bertscore": williams_out,
    }
    (out_dir / "multiple_comparison.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5) Markdown table
    rows_sorted = sorted(metrics.items(), key=lambda kv: -abs(kv[1]["rho"]))
    md = [
        f"# Multiple Comparison Correction (Bonferroni, α={ALPHA})",
        "",
        f"N = {N} metrics  →  Bonferroni-adjusted threshold α/N = {alpha_bonf:.5f}",
        "",
        f"`**` = passes Bonferroni  ·  `*` = passes raw α=0.05 only  ·  blank = not significant",
        "",
        "| Metric | ρ | p (raw) | p (Bonferroni-adj) | Sig |",
        "|:---|---:|---:|---:|:---:|",
    ]
    for name, v in rows_sorted:
        mark = "**" if v["sig_bonf"] else ("*" if v["sig_raw_05"] else "")
        md.append(
            f"| {name} | {v['rho']:.4f} | {v['p_raw']:.4f} | {v['p_bonf']:.4f} | {mark} |"
        )
    md.append("")
    if williams_out:
        md += [
            "## Williams test: H-FAct v3 vs BERTScore",
            "",
            f"- ρ(human, H-FAct v3)   = **{williams_out['rho_v3_vs_human']:.4f}**",
            f"- ρ(human, BERTScore)   = **{williams_out['rho_bertscore_vs_human']:.4f}**",
            f"- ρ(H-FAct v3, BERTScore) = {williams_out['rho_v3_vs_bertscore']:.4f}",
            f"- n = {williams_out['n']}, df = {williams_out['df']}",
            f"- t = {williams_out['t_stat']:.4f}, **p = {williams_out['p_two_sided']:.4f}**",
            "",
            f"_{williams_out['interpretation']}_",
            "",
        ]
    (RESULTS_DIR / "tables" / "multiple_comparison.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(f"\n✓ wrote {out_dir / 'multiple_comparison.json'}")
    print(f"✓ wrote {RESULTS_DIR / 'tables' / 'multiple_comparison.md'}")


if __name__ == "__main__":
    main()
