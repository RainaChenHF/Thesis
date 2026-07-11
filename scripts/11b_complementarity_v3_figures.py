"""
Generate H-FAct v3 complementarity figures for Chapter 6.5.

This script is intentionally separate from 11_complementarity_analysis.py.
It reads the v3 aligned subset and writes new figure names so that the
original H-FAct figures are not overwritten.

Inputs:
  results/correlation_v3/canonical_v3_4run.jsonl

Outputs:
  results/figures/complementarity_scatter_v3.png
  results/figures/complementarity_regression_v3.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = ImageDraw = ImageFont = None


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
V3_SAMPLE_PATH = RESULTS_DIR / "correlation_v3" / "canonical_v3_4run.jsonl"


def load_v3_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not V3_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing v3 annotated sample: {V3_SAMPLE_PATH}")

    rows = [
        json.loads(line)
        for line in V3_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    f1 = np.array([row["f1"] for row in rows], dtype=float)
    hfact_v3 = np.array([row["h_fact_v3"] for row in rows], dtype=float)
    judgment = np.array([row["human_score"] for row in rows], dtype=float)
    return f1, hfact_v3, judgment


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if stats is not None:
        return float(stats.spearmanr(x, y)[0])
    rx = average_ranks(x)
    ry = average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if stats is not None:
        result = stats.linregress(x, y)
        return float(result.slope), float(result.intercept)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def incremental_validity(
    f1: np.ndarray, hfact_v3: np.ndarray, judgment: np.ndarray
) -> dict[str, float | bool]:
    n = len(judgment)
    ones = np.ones(n)

    x_f1 = np.column_stack([ones, f1])
    r2_f1 = ols_r2(x_f1, judgment)

    x_full = np.column_stack([ones, f1, hfact_v3])
    r2_full = ols_r2(x_full, judgment)
    delta_r2 = r2_full - r2_f1

    beta_full, _, _, _ = np.linalg.lstsq(x_full, judgment, rcond=None)
    y_hat = x_full @ beta_full
    ss_res = float(np.sum((judgment - y_hat) ** 2))
    df = max(n - x_full.shape[1], 1)
    mse = ss_res / df
    xtx_inv = np.linalg.pinv(x_full.T @ x_full)
    se_hfact = math.sqrt(mse * float(xtx_inv[2, 2]))
    t_hfact = float(beta_full[2]) / se_hfact if se_hfact > 0 else 0.0
    p_hfact = float(2 * stats.t.sf(abs(t_hfact), df=df)) if stats is not None else float("nan")

    return {
        "n": n,
        "r2_f1_only": round(r2_f1, 4),
        "r2_full": round(r2_full, 4),
        "delta_r2": round(delta_r2, 4),
        "beta_hfact": round(float(beta_full[2]), 4),
        "t_hfact": round(t_hfact, 3),
        "p_hfact": round(p_hfact, 4),
        "sig_hfact": bool(p_hfact < 0.05) if not math.isnan(p_hfact) else False,
    }


def require_pil() -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError(
            "Neither matplotlib nor PIL is available. Install matplotlib from "
            "requirements.txt or use a Python environment with Pillow."
        )


def judgment_color(value: float) -> tuple[int, int, int]:
    # Red -> yellow -> green for scores in [1, 5].
    t = min(max((value - 1.0) / 4.0, 0.0), 1.0)
    if t < 0.5:
        u = t / 0.5
        r, g, b = 220, int(70 + 170 * u), 70
    else:
        u = (t - 0.5) / 0.5
        r, g, b = int(220 - 160 * u), 190, 90
    return r, g, b


def draw_axes(
    draw: "ImageDraw.ImageDraw",
    box: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(30, 30, 30), width=2)
    for i in range(6):
        tx = left + (right - left) * i / 5
        ty = bottom - (bottom - top) * i / 5
        draw.line((tx, bottom, tx, bottom + 5), fill=(30, 30, 30), width=1)
        draw.line((left - 5, ty, left, ty), fill=(30, 30, 30), width=1)
        xv = xlim[0] + (xlim[1] - xlim[0]) * i / 5
        yv = ylim[0] + (ylim[1] - ylim[0]) * i / 5
        draw.text((tx - 15, bottom + 8), f"{xv:.1f}", fill=(40, 40, 40))
        draw.text((left - 45, ty - 7), f"{yv:.1f}", fill=(40, 40, 40))
        if i not in (0, 5):
            draw.line((tx, top, tx, bottom), fill=(225, 225, 225), width=1)
            draw.line((left, ty, right, ty), fill=(225, 225, 225), width=1)
    draw.text((left + (right - left) / 2 - 45, bottom + 35), x_label, fill=(20, 20, 20))
    draw.text((left, top - 72), title, fill=(20, 20, 20))
    draw.text((left, top - 26), y_label, fill=(20, 20, 20))


def to_pixel(
    x: float,
    y: float,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[int, int]:
    left, top, right, bottom = box
    px = left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
    py = bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)
    return int(px), int(py)


def plot_scatter_pil(f1: np.ndarray, hfact_v3: np.ndarray, judgment: np.ndarray) -> Path:
    require_pil()
    threshold = 0.3
    img = Image.new("RGB", (1050, 900), "white")
    draw = ImageDraw.Draw(img)
    box = (95, 115, 875, 805)
    xlim = (-0.05, 1.05)
    ylim = (-0.05, 1.05)

    draw_axes(
        draw,
        box,
        "F1 Score",
        "H-FAct v3 Score",
        f"F1 vs H-FAct v3 - Four Quadrants (n={len(f1)})",
        xlim,
        ylim,
    )
    px, _ = to_pixel(threshold, 0, box, xlim, ylim)
    _, py = to_pixel(0, threshold, box, xlim, ylim)
    draw.line((px, box[1], px, box[3]), fill=(140, 140, 140), width=2)
    draw.line((box[0], py, box[2], py), fill=(140, 140, 140), width=2)

    for x, y, z in zip(f1, hfact_v3, judgment):
        cx, cy = to_pixel(float(x), float(y), box, xlim, ylim)
        color = judgment_color(float(z))
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=color, outline=(255, 255, 255))

    draw.text((105, 125), "Low F1 / High H-FAct v3\n(supported mismatch)", fill=(21, 101, 192))
    draw.text((690, 125), "High F1 / High H-FAct v3\n(aligned)", fill=(46, 125, 50))
    draw.text((105, 740), "Low F1 / Low H-FAct v3\n(weak answer and support)", fill=(183, 28, 28))
    draw.text((685, 740), "High F1 / Low H-FAct v3\n(weak grounding)", fill=(230, 81, 0))
    draw.text((95, 60), "Colour = LLM-simulated judgment", fill=(20, 20, 20))

    # Simple color legend.
    lx, ly = 910, 160
    draw.text((lx, ly - 30), "Judgment", fill=(20, 20, 20))
    for i in range(100):
        value = 5 - 4 * i / 99
        draw.rectangle((lx, ly + i * 4, lx + 28, ly + i * 4 + 4), fill=judgment_color(value))
    draw.text((lx + 35, ly - 2), "5", fill=(20, 20, 20))
    draw.text((lx + 35, ly + 390), "1", fill=(20, 20, 20))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "complementarity_scatter_v3.png"
    img.save(out_path)
    return out_path


def plot_regression_pil(
    f1: np.ndarray, hfact_v3: np.ndarray, judgment: np.ndarray, reg: dict[str, float | bool]
) -> Path:
    require_pil()
    n = len(judgment)
    ones = np.ones(n)
    x_f1 = np.column_stack([ones, f1])
    beta_hfact, _, _, _ = np.linalg.lstsq(x_f1, hfact_v3, rcond=None)
    beta_judgment, _, _, _ = np.linalg.lstsq(x_f1, judgment, rcond=None)
    resid_hfact = hfact_v3 - x_f1 @ beta_hfact
    resid_judgment = judgment - x_f1 @ beta_judgment

    img = Image.new("RGB", (1650, 680), "white")
    draw = ImageDraw.Draw(img)
    draw.text(
        (430, 25),
        "Incremental Validity: Does H-FAct v3 Add Information Beyond F1?",
        fill=(20, 20, 20),
    )

    box1 = (90, 115, 760, 590)
    box2 = (900, 115, 1570, 590)
    draw_axes(
        draw,
        box1,
        "F1",
        "LLM-simulated judgment score",
        f"F1 -> LLM-simulated judgment  R2={reg['r2_f1_only']:.3f}, rho={spearman_rho(f1, judgment):.3f}",
        (-0.05, 1.05),
        (0.8, 5.2),
    )
    for x, y in zip(f1, judgment):
        cx, cy = to_pixel(float(x), float(y), box1, (-0.05, 1.05), (0.8, 5.2))
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(33, 150, 243))
    slope, intercept = linear_fit(f1, judgment)
    p1 = to_pixel(0, intercept, box1, (-0.05, 1.05), (0.8, 5.2))
    p2 = to_pixel(1, slope + intercept, box1, (-0.05, 1.05), (0.8, 5.2))
    draw.line((*p1, *p2), fill=(21, 101, 192), width=3)

    xmin, xmax = float(resid_hfact.min()), float(resid_hfact.max())
    ymin, ymax = float(resid_judgment.min()), float(resid_judgment.max())
    pad_x = max((xmax - xmin) * 0.1, 0.05)
    pad_y = max((ymax - ymin) * 0.1, 0.2)
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)
    if math.isnan(float(reg["p_hfact"])):
        partial_title = (
            f"H-FAct v3 partial effect  beta={reg['beta_hfact']:.3f}, "
            f"t={reg['t_hfact']:.2f}, Delta R2={reg['delta_r2']:.3f}"
        )
    else:
        p_display = "p<0.05" if bool(reg["sig_hfact"]) else f"p={reg['p_hfact']:.3f}"
        partial_title = (
            f"H-FAct v3 partial effect  beta={reg['beta_hfact']:.3f}, "
            f"t={reg['t_hfact']:.2f}, {p_display}, Delta R2={reg['delta_r2']:.3f}"
        )
    draw_axes(
        draw,
        box2,
        "H-FAct v3 residual (after removing F1 effect)",
        "LLM-simulated judgment residual",
        partial_title,
        xlim,
        ylim,
    )
    zero_x, _ = to_pixel(0, 0, box2, xlim, ylim)
    _, zero_y = to_pixel(0, 0, box2, xlim, ylim)
    draw.line((zero_x, box2[1], zero_x, box2[3]), fill=(150, 150, 150), width=1)
    draw.line((box2[0], zero_y, box2[2], zero_y), fill=(150, 150, 150), width=1)
    for x, y in zip(resid_hfact, resid_judgment):
        cx, cy = to_pixel(float(x), float(y), box2, xlim, ylim)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(244, 67, 54))
    slope2, intercept2 = linear_fit(resid_hfact, resid_judgment)
    p1 = to_pixel(xlim[0], slope2 * xlim[0] + intercept2, box2, xlim, ylim)
    p2 = to_pixel(xlim[1], slope2 * xlim[1] + intercept2, box2, xlim, ylim)
    draw.line((*p1, *p2), fill=(183, 28, 28), width=3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "complementarity_regression_v3.png"
    img.save(out_path)
    return out_path


def plot_scatter(f1: np.ndarray, hfact_v3: np.ndarray, judgment: np.ndarray) -> Path:
    if plt is None:
        return plot_scatter_pil(f1, hfact_v3, judgment)
    threshold = 0.3

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        f1,
        hfact_v3,
        c=judgment,
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        s=40,
        alpha=0.75,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="LLM-simulated judgment score (1-5)")

    ax.axvline(threshold, color="gray", linestyle="--", alpha=0.65, linewidth=0.9)
    ax.axhline(threshold, color="gray", linestyle="--", alpha=0.65, linewidth=0.9)

    label_size = 8
    ax.text(
        0.01,
        0.97,
        "Low F1\nHigh H-FAct v3\n(supported mismatch)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=label_size,
        color="#1565C0",
    )
    ax.text(
        0.99,
        0.97,
        "High F1\nHigh H-FAct v3\n(aligned)",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=label_size,
        color="#2E7D32",
    )
    ax.text(
        0.01,
        0.03,
        "Low F1\nLow H-FAct v3\n(weak answer and support)",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=label_size,
        color="#B71C1C",
    )
    ax.text(
        0.99,
        0.03,
        "High F1\nLow H-FAct v3\n(weak grounding)",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=label_size,
        color="#E65100",
    )

    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_ylabel("H-FAct v3 Score", fontsize=11)
    ax.set_title(
        f"F1 vs H-FAct v3 - Four Quadrants (n={len(f1)})\n"
        "Colour = LLM-simulated judgment",
        fontsize=11,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "complementarity_scatter_v3.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_regression(
    f1: np.ndarray, hfact_v3: np.ndarray, judgment: np.ndarray, reg: dict[str, float | bool]
) -> Path:
    if plt is None:
        return plot_regression_pil(f1, hfact_v3, judgment, reg)
    n = len(judgment)
    ones = np.ones(n)

    x_f1 = np.column_stack([ones, f1])
    beta_hfact, _, _, _ = np.linalg.lstsq(x_f1, hfact_v3, rcond=None)
    beta_judgment, _, _, _ = np.linalg.lstsq(x_f1, judgment, rcond=None)
    resid_hfact = hfact_v3 - x_f1 @ beta_hfact
    resid_judgment = judgment - x_f1 @ beta_judgment

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.scatter(f1, judgment, alpha=0.55, s=25, color="#2196F3")
    slope, intercept = linear_fit(f1, judgment)
    xs = np.linspace(0, 1, 100)
    ax1.plot(xs, slope * xs + intercept, color="#1565C0", linewidth=1.5)
    rho_f1 = spearman_rho(f1, judgment)
    ax1.set_xlabel("F1")
    ax1.set_ylabel("LLM-simulated judgment score")
    ax1.set_title(
        f"F1 -> LLM-simulated judgment\n"
        f"R2={reg['r2_f1_only']:.3f}, rho={rho_f1:.3f}"
    )
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(alpha=0.25)

    ax2.scatter(resid_hfact, resid_judgment, alpha=0.55, s=25, color="#F44336")
    if resid_hfact.std() > 0:
        slope2, intercept2 = linear_fit(resid_hfact, resid_judgment)
        xs2 = np.linspace(resid_hfact.min(), resid_hfact.max(), 100)
        ax2.plot(xs2, slope2 * xs2 + intercept2, color="#B71C1C", linewidth=1.5)

    sig = "p=n/a" if math.isnan(float(reg["p_hfact"])) else (
        "p<0.05" if bool(reg["sig_hfact"]) else f"p={reg['p_hfact']:.3f}"
    )
    ax2.set_xlabel("H-FAct v3 residual (after removing F1 effect)")
    ax2.set_ylabel("LLM-simulated judgment residual (after removing F1 effect)")
    ax2.set_title(
        "H-FAct v3 partial effect (controlling for F1)\n"
        f"beta={reg['beta_hfact']:.3f}, t={reg['t_hfact']:.2f}, {sig}\n"
        f"Delta R2={reg['delta_r2']:.3f}"
    )
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Incremental Validity: Does H-FAct v3 Add Information Beyond F1?",
        fontsize=11,
        fontweight="bold",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "complementarity_regression_v3.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    f1, hfact_v3, judgment = load_v3_data()
    reg = incremental_validity(f1, hfact_v3, judgment)

    scatter_path = plot_scatter(f1, hfact_v3, judgment)
    regression_path = plot_regression(f1, hfact_v3, judgment, reg)

    print(f"Loaded {reg['n']} v3 aligned samples.")
    print(f"F1-only R2: {reg['r2_f1_only']:.4f}")
    print(f"F1 + H-FAct v3 R2: {reg['r2_full']:.4f}")
    print(f"Delta R2: {reg['delta_r2']:.4f}")
    print(f"Saved: {scatter_path}")
    print(f"Saved: {regression_path}")


if __name__ == "__main__":
    main()
