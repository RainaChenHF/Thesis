#!/usr/bin/env python3
"""Generate a v3-aware metric ranking figure without overwriting old artifacts.

The original advisor demo script (scripts/14_advisor_demo.py) writes
results/figures/metric_ranking.png using the original H-FAct value
(rho=0.624). This companion script inserts the canonical H-FAct v3 value
from results/correlation_v3/summary_v3.json and writes:

  results/figures/metric_ranking_v3.png

The script uses Pillow instead of matplotlib so it can run in lightweight
environments where the original plotting stack is unavailable.
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def font(size, bold=False):
    candidates = [
        "Arial Bold" if bold else "Arial",
        "Calibri Bold" if bold else "Calibri",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


corr_baseline = load_json(RES / "correlation" / "summary.json")
corr_comparison = load_json(RES / "comparison" / "correlation.json")
corr_bootstrap = load_json(RES / "bootstrap" / "correlation_ci.json")
summary_v3 = load_json(RES / "correlation_v3" / "summary_v3.json")


METRICS = [
    ("EM", "em", "Lexical", False, "baseline"),
    ("F1", "f1", "Lexical", False, "baseline"),
    ("BLEU-1", "bleu1", "Lexical", False, "comparison"),
    ("ROUGE-L", "rouge_l", "Lexical", False, "comparison"),
    ("BERTScore proxy", "bertscore", "Semantic", False, "comparison"),
    ("FActScore", "factscore", "Factual", False, "comparison"),
    ("RAGAS Faith.", "ragas_faithfulness", "LLM-Judge variant", False, "comparison"),
    ("RAGAS Relev.", "ragas_relevancy", "LLM-Judge variant", False, "comparison"),
    ("RAGAS Ctx-P", "ragas_ctx_precision", "LLM-Judge variant", False, "comparison"),
    ("RAGAS Ctx-R", "ragas_ctx_recall", "LLM-Judge variant", False, "comparison"),
    ("AIS", "ais", "Factual", False, "comparison"),
    ("RGB Noise", "rgb_noise", "RAG-Robust", False, "comparison"),
    ("RGB Integration", "rgb_integration", "RAG-Robust", False, "comparison"),
    ("RGB Neg-Reject", "rgb_neg_rejection", "RAG-Robust", False, "comparison"),
    ("G-Eval Consist.", "geval_consistency", "LLM-Judge variant", False, "comparison"),
    ("G-Eval Relev.", "geval_relevance", "LLM-Judge variant", False, "comparison"),
    ("G-Eval Coher.", "geval_coherence", "LLM-Judge variant", False, "comparison"),
    ("G-Eval Correct.", "geval_correctness", "LLM-Judge variant", False, "comparison"),
    ("G-Eval Avg", "geval_avg", "LLM-Judge variant", False, "comparison"),
    ("H-FAct", "h_fact", "Factual", True, "baseline"),
    ("H-FAct v3", "h_fact_v3", "Factual", True, "v3"),
    ("MAA", "maa", "Attribution", True, "baseline"),
    ("EPP-F1", "epp_f1", "Evidence", True, "baseline"),
    ("JHit@3", "joint_hit", "Retrieval", True, "baseline"),
]


def get_corr(key, source):
    if source == "v3":
        return summary_v3["h_fact_v3"]["rho"], None
    if source == "baseline":
        item = corr_baseline[key]
        return item["rho"], corr_bootstrap.get(key)
    item = corr_comparison[key]
    return item["rho"], corr_bootstrap.get(key)


rows = []
for name, key, cat, is_ours, source in METRICS:
    rho, ci = get_corr(key, source)
    rows.append({
        "name": name,
        "cat": cat,
        "is_ours": is_ours,
        "rho": float(rho),
        "ci_lo": None if ci is None else ci.get("ci_lo"),
        "ci_hi": None if ci is None else ci.get("ci_hi"),
    })

rows.sort(key=lambda r: r["rho"], reverse=True)


# Canvas and chart geometry.
W, H = 2200, 1580
left, right = 310, 2040
top, bottom = 150, 1420
axis_min, axis_max = -0.4, 1.0
bar_h = 34
row_gap = 18

img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

title_font = font(42)
label_font = font(27)
small_font = font(22)
tiny_font = font(20)
legend_font = font(25)


def x_pos(value):
    ratio = (value - axis_min) / (axis_max - axis_min)
    return int(left + ratio * (right - left))


def text_size(text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


# Background and guides.
top_tier_x = x_pos(0.5)
draw.rectangle([top_tier_x, top - 25, right, bottom + 10], fill=(232, 244, 252))
draw.line([x_pos(0), top - 25, x_pos(0), bottom + 10], fill=(20, 20, 20), width=3)
draw.line([top_tier_x, top - 25, top_tier_x, bottom + 10], fill=(145, 165, 175), width=3)

for tick in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    x = x_pos(tick)
    draw.line([x, bottom + 5, x, bottom + 18], fill=(0, 0, 0), width=2)
    tick_label = f"{tick:.1f}"
    tw, th = text_size(tick_label, small_font)
    draw.text((x - tw / 2, bottom + 24), tick_label, fill=(0, 0, 0), font=small_font)

draw.rectangle([left, top - 25, right, bottom + 10], outline=(0, 0, 0), width=3)

title = "Metric Alignment with LLM-Simulated Judgment"
tw, th = text_size(title, title_font)
draw.text(((W - tw) / 2, 35), title, fill=(0, 0, 0), font=title_font)

axis_label = "Spearman rho with LLM-simulated judgment"
tw, th = text_size(axis_label, label_font)
draw.text(((W - tw) / 2, H - 72), axis_label, fill=(0, 0, 0), font=label_font)

draw.text((right - 160, top + 10), "top-tier\nalignment", fill=(21, 101, 192), font=small_font)


# Bars.
# Keep the highest-correlated metrics at the top, matching the thesis table.
plot_rows = rows
y = top
for r in plot_rows:
    y_mid = y + bar_h / 2
    name_w, _ = text_size(r["name"], label_font)
    draw.text((left - name_w - 18, y + 1), r["name"], fill=(0, 0, 0), font=label_font)

    if r["is_ours"]:
        color = (21, 101, 192)
    elif r["rho"] < 0:
        color = (198, 40, 40)
    else:
        color = (144, 164, 174)

    x0 = x_pos(0)
    x1 = x_pos(r["rho"])
    draw.rectangle([min(x0, x1), y, max(x0, x1), y + bar_h], fill=color)

    lo, hi = r["ci_lo"], r["ci_hi"]
    if lo is not None and hi is not None:
        xl, xr = x_pos(lo), x_pos(hi)
        draw.line([xl, y_mid, xr, y_mid], fill=(0, 0, 0), width=3)
        draw.line([xl, y_mid - 10, xl, y_mid + 10], fill=(0, 0, 0), width=3)
        draw.line([xr, y_mid - 10, xr, y_mid + 10], fill=(0, 0, 0), width=3)

    value_label = f"{r['rho']:.3f}"
    label_x = max(x0, x1) + 26
    if r["rho"] < 0:
        label_x = x0 + 26
    draw.text((label_x, y - 1), value_label, fill=(0, 0, 0), font=small_font)

    y += bar_h + row_gap


# Legend.
legend_x, legend_y = 1460, 1260
draw.rectangle([legend_x - 24, legend_y - 24, right - 20, legend_y + 128], outline=(190, 190, 190), width=2)
legend_items = [
    ((21, 101, 192), "Our metrics"),
    ((144, 164, 174), "Existing metrics (rho >= 0)"),
    ((198, 40, 40), "Failed variants (rho < 0)"),
]
for i, (color, label) in enumerate(legend_items):
    yy = legend_y + i * 42
    draw.rectangle([legend_x, yy, legend_x + 52, yy + 24], fill=color)
    draw.text((legend_x + 70, yy - 5), label, fill=(0, 0, 0), font=legend_font)


out_path = FIG / "metric_ranking_v3.png"
img.save(out_path)

print(f"Wrote {out_path}")
print("Top 5:")
for i, r in enumerate(rows[:5], start=1):
    print(f"{i}. {r['name']}: {r['rho']:.4f}")

