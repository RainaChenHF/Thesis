"""
Script 05 – Complexity stratification.

Reads baseline predictions, classifies each question L1-L5,
computes per-level metric averages, saves stratified results.

Output: results/stratified/summary.json
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR
from src.utils.helpers import get_logger
from src.metrics.hybrid_eval import complexity_weighted_score

log = get_logger("05_stratified")


def main():
    pred_path = RESULTS_DIR / "baseline" / "predictions.jsonl"
    if not pred_path.exists():
        log.error("Run script 03 first: %s not found", pred_path)
        sys.exit(1)

    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        level = r.get("complexity", "L3")
        buckets[level].append(r)

    out: dict[str, dict] = {}
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        grp = buckets.get(level, [])
        def mean(key): return round(sum(r.get(key, 0) for r in grp) / max(len(grp), 1), 4)
        out[level] = {
            "n": len(grp),
            "em":         mean("em"),
            "f1":         mean("f1"),
            "joint_hit":  mean("joint_hit"),
            "epp_f1":     mean("epp_f1"),
            "h_fact":     mean("h_fact"),
            "maa":        mean("maa"),
        }
        log.info("  %s (n=%d)  EM=%.3f  H-FAct=%.3f  JHit=%.3f",
                 level, len(grp), out[level]["em"],
                 out[level]["h_fact"], out[level]["joint_hit"])

    out_dir = RESULTS_DIR / "stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute CWRS (Complexity-Weighted Reasoning Score) using H-FAct per level
    cwrs_input = {level: [r.get("h_fact", 0.0) for r in buckets.get(level, [])]
                  for level in ["L1", "L2", "L3", "L4", "L5"]}
    cwrs = complexity_weighted_score(cwrs_input)
    out["cwrs"] = cwrs
    log.info("  CWRS (H-FAct weighted by complexity): %.4f", cwrs)

    (out_dir / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved → %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
