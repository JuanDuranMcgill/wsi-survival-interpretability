#!/usr/bin/env python3
"""Fusion weight against ablation effect, one point per tissue region.

This is the paper's central interpretability result in one picture: the
architecture's own per-region weight does not predict what happens when that
region is removed. Necrosis carries the highest fusion weight in both cohorts
and its ablation costs nothing measurable; adipose is mid-ranked by weight and
produces the largest ablation effect in both.

Form: a scatter is the right choice because the claim is about the relationship
between two quantities, and the claim is that there isn't one. Bars would force
two separate rankings on the reader and make the non-relationship harder to see.

Encoding: position only. One series, so no colour encoding and no legend; every
point is directly labelled, which is feasible at 8 and 9 points. The uniform
weight and the zero-effect line are drawn as recessive references, since both
axes have a meaningful origin that is not at the corner.

Reads the committed JSONs, so the figure cannot drift from the numbers.

Usage:
    python visualization/make_weight_vs_ablation_figure.py \
        --out Figures/fig_weight_vs_ablation.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"

SHORT = {
    "Perivesical adipose tissue (extravesical fat)": "Adipose",
    "Invasive urothelial carcinoma (tumor)": "Tumour",
    "Inflammatory infiltrates (immune cells)": "Immune",
    "Normal urothelium (benign mucosa)": "Urothelium",
    "Muscularis propria (detrusor muscle)": "Muscularis",
    "Lamina propria (fibrovascular stroma)": "Lamina propria",
    "Blood vessels (vasculature)": "Vessels",
    "Necrosis": "Necrosis",
    "Adipose tissue (fat)": "Adipose",
    "Muscle tissue (smooth or skeletal muscle)": "Muscle",
    "Ductal carcinoma in situ (DCIS)": "DCIS",
    "Tumor-infiltrating lymphocytes (immune infiltrates)": "TILs",
    "Invasive breast carcinoma (tumor cells)": "Tumour",
    "Fibrous desmoplastic stroma": "Stroma",
    "Normal breast glands and lobules (TDLU)": "TDLU",
    "Necrosis or hemorrhage": "Necrosis",
}

# Regions whose ablation CI excludes zero get a filled marker; the rest are
# hollow. That is a secondary encoding carrying a statistical distinction, and
# it is also stated in the caption, so it is never colour-alone.
def panel(ax, cohort, results_root, title):
    fw = json.load(open(f"{results_root}/fusion_weight_test_{cohort}_last.json"))
    ab = json.load(open(f"{results_root}/region_ablation_{cohort}.json"))
    uniform = fw["uniform_weight"] * 100

    pts = []
    for name, v in ab["per_region"].items():
        d = v["delta_no_renorm"]
        pts.append({
            "label": SHORT.get(name, name[:14]),
            "w": fw["per_region"][name]["mean"] * 100,
            "d": d["mean"] * 100,
            "lo": d["ci95"][0] * 100,
            "hi": d["ci95"][1] * 100,
            "sig": d["ci95"][1] < 0 or d["ci95"][0] > 0,
        })

    rho, p = spearmanr([q["w"] for q in pts], [q["d"] for q in pts])

    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.axvline(uniform, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for q in pts:
        ax.errorbar(q["w"], q["d"], yerr=[[q["d"] - q["lo"]], [q["hi"] - q["d"]]],
                    fmt="none", ecolor=TEXT_SECONDARY, elinewidth=0.9,
                    capsize=2.0, zorder=2, alpha=0.85)
        ax.plot(q["w"], q["d"], "o", ms=7,
                mfc=SERIES_1 if q["sig"] else "white",
                mec=SERIES_1, mew=1.6, zorder=3)

    # Greedy label declutter: place each label right of its point, but when two
    # points are close in data space push the labels apart vertically and, if
    # still tight, flip one to the left. With 8-9 points this is enough; a
    # general solver would be overkill.
    xs_all = [q["w"] for q in pts]
    ys_all = [q["d"] for q in pts]
    xr = (max(xs_all) - min(xs_all)) or 1.0
    yr = (max(ys_all) - min(ys_all)) or 1.0
    placed = []
    for q in sorted(pts, key=lambda z: (-z["d"], z["w"])):
        dx, dy, ha = 8, 4, "left"
        for _ in range(6):
            # approximate label box centre in data units
            cx = q["w"] + (dx / 72.0) * xr * (1 if ha == "left" else -1) * 0.9
            cy = q["d"] + (dy / 72.0) * yr * 0.9
            clash = any(abs(cx - px) < 0.16 * xr and abs(cy - py) < 0.075 * yr
                        for px, py in placed)
            if not clash:
                break
            if dy > 0:
                dy = -12
            elif ha == "left":
                ha, dx, dy = "right", -8, 4
            else:
                dy -= 11
        placed.append((cx, cy))
        ax.annotate(q["label"], (q["w"], q["d"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.2, color=TEXT_PRIMARY)

    ax.set_xlabel("Region fusion weight (%)", fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_ylabel("Change in c-index when ablated (pp)", fontsize=9.5,
                  color=TEXT_PRIMARY)
    ax.set_title(f"{title}    Spearman $\\rho$ = {rho:+.2f}, $p$ = {p:.2f}",
                 fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=8)
    ax.tick_params(labelsize=8.5, colors=TEXT_SECONDARY, length=0)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)

    xs = [q["w"] for q in pts]
    pad = (max(xs) - min(xs)) * 0.28 or 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad * 1.5)
    return rho, p, uniform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="Figures/fig_weight_vs_ablation.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    out = {}
    for ax, (c, t) in zip(axes, [("blca", "(a) TCGA-BLCA"), ("brca", "(b) TCGA-BRCA")]):
        rho, p, u = panel(ax, c, args.results_root, t)
        out[c] = {"spearman_rho": float(rho), "p": float(p), "uniform_weight_pct": u}
        print(f"[{c}] weight vs ablation: rho = {rho:+.3f}, p = {p:.3f}, "
              f"uniform = {u:.2f}%")

    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=220, bbox_inches="tight")
    with open(os.path.join(args.results_root, "weight_vs_ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
