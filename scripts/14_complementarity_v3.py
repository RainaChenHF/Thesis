#!/usr/bin/env python3
"""
Script 14 -- Recompute complementarity quadrants and incremental validity
using H-FAct v3 (instead of v1 which the stale results/ablation/complementarity.json used).

Why: the existing complementarity.json was computed against h_fact (v1).  This
script writes a v3 variant so the "lucky hallucination" quadrant statistic
(avg_human_lucky) can be interpreted on the actual reported version.

Inputs:
  results/correlation_v3/annotated_sample_v3.jsonl   (h_fact_v3 + human_score per sample)
Output:
  results/ablation/complementarity_v3.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR

THRESH = 0.3


def main():
    rows = [json.loads(l) for l in open(RESULTS_DIR / "correlation_v3" / "annotated_sample_v3.jsonl")]
    f1 = np.array([r["f1"] for r in rows])
    hf3 = np.array([r["h_fact_v3"] for r in rows])
    hf1 = np.array([r["h_fact"] for r in rows])
    human = np.array([r["human_score"] for r in rows])

    n = len(rows)

    # Inter-metric correlation
    rho_f1_v3, p_f1_v3 = spearmanr(f1, hf3)
    pr_f1_v3, ppr_f1_v3 = pearsonr(f1, hf3)

    # Incremental validity: F1 + v3 vs F1 alone (OLS R^2)
    # Use simple OLS via numpy lstsq for clarity
    def r2(X, y):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
        y_hat = X1 @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

    r2_f1 = r2(f1.reshape(-1, 1), human)
    r2_f1_v3 = r2(np.column_stack([f1, hf3]), human)
    delta_r2 = r2_f1_v3 - r2_f1

    # Quadrants (F1 x H-FAct v3)
    hi_f1_hi_v3 = int(np.sum((f1 >= THRESH) & (hf3 >= THRESH)))
    hi_f1_lo_v3 = int(np.sum((f1 >= THRESH) & (hf3 <  THRESH)))
    lo_f1_hi_v3 = int(np.sum((f1 <  THRESH) & (hf3 >= THRESH)))
    lo_f1_lo_v3 = int(np.sum((f1 <  THRESH) & (hf3 <  THRESH)))

    mask_ideal = (f1 >= THRESH) & (hf3 >= THRESH)
    mask_lucky = (f1 >= THRESH) & (hf3 <  THRESH)
    mask_para  = (f1 <  THRESH) & (hf3 >= THRESH)
    mask_both  = (f1 <  THRESH) & (hf3 <  THRESH)

    avg_human_ideal = float(human[mask_ideal].mean()) if mask_ideal.any() else float("nan")
    avg_human_lucky = float(human[mask_lucky].mean()) if mask_lucky.any() else float("nan")
    avg_human_para  = float(human[mask_para ].mean()) if mask_para .any() else float("nan")
    avg_human_both  = float(human[mask_both ].mean()) if mask_both .any() else float("nan")

    # Same with v1 for side-by-side comparison
    hi_f1_lo_v1 = int(np.sum((f1 >= THRESH) & (hf1 <  THRESH)))
    mask_lucky_v1 = (f1 >= THRESH) & (hf1 <  THRESH)
    avg_human_lucky_v1 = float(human[mask_lucky_v1].mean()) if mask_lucky_v1.any() else float("nan")

    out = {
        "n": n,
        "thresh": THRESH,
        "inter_metric_corr": {
            "spearman_rho": round(float(rho_f1_v3), 4),
            "spearman_p": round(float(p_f1_v3), 6),
            "pearson_r": round(float(pr_f1_v3), 4),
            "pearson_p": round(float(ppr_f1_v3), 6),
        },
        "incremental_validity_v3": {
            "r2_f1_only": round(float(r2_f1), 4),
            "r2_f1_plus_v3": round(float(r2_f1_v3), 4),
            "delta_r2": round(float(delta_r2), 4),
        },
        "quadrants_v3": {
            "thresh": THRESH,
            "ideal_n": hi_f1_hi_v3,
            "lucky_halluc_n": hi_f1_lo_v3,
            "correct_para_n": lo_f1_hi_v3,
            "both_wrong_n": lo_f1_lo_v3,
            "avg_human_ideal": round(avg_human_ideal, 4),
            "avg_human_lucky": round(avg_human_lucky, 4),
            "avg_human_correct_para": round(avg_human_para, 4),
            "avg_human_both_wrong": round(avg_human_both, 4),
        },
        "quadrants_v1_for_compare": {
            "lucky_halluc_n": hi_f1_lo_v1,
            "avg_human_lucky": round(avg_human_lucky_v1, 4),
        },
        "interpretation_note": (
            "v3 'lucky hallucination' quadrant (high F1, low H-FAct v3): "
            f"{hi_f1_lo_v3} samples with avg human score {avg_human_lucky:.3f}. "
            "If avg_human_lucky is materially HIGHER than avg_human_ideal "
            f"({avg_human_ideal:.3f}), it does NOT mean H-FAct is wrong — it means "
            "H-FAct is identifying cases the F1 token-overlap calls correct but where "
            "evidence support is missing, while humans (here, LLM-simulated) still "
            "give credit for the plausible answer. Stage-three discussion in §10.4 "
            "should report both quadrants explicitly."
        ),
    }
    out_path = RESULTS_DIR / "ablation" / "complementarity_v3.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n✓ wrote {out_path}")


if __name__ == "__main__":
    main()
