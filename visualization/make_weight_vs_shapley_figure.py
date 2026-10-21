#!/usr/bin/env python3
"""Fusion weight against exact Shapley value, one point per tissue compartment.

The paper's central interpretability result in one picture: on the same 20
retrained models per cohort, the architecture's own per-compartment weight
tracks what each compartment contributes (its Shapley value with out-of-bag
concordance as the payoff) in BLCA, and does not in BRCA.

Form and encoding follow make_weight_vs_ablation_figure.py: a scatter, position
only, every point labelled, filled markers where the Shapley interval excludes
zero. If the weights measured reliance, points would rise from left to right.

The p-value is the exact permutation p of the Spearman correlation over all R!
orderings, as reported in the paper, not scipy's asymptotic one.

Reads the committed JSONs, so the figure cannot drift from the numbers.

Usage:
    python visualization/make_weight_vs_shapley_figure.py \
        --out Figures/fig_weight_vs_shapley.pdf
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from make_weight_vs_ablation_figure import FS, GRID, SERIES_1, SHORT, TEXT_PRIMARY, TEXT_SECONDARY


def exact_perm_p(x, y):
    """Two-sided exact permutation p for Spearman's rho over all orderings of y."""
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    hits = total = 0
    for perm in itertools.permutations(ry):
        total += 1
        hits += abs(np.corrcoef(rx, perm)[0, 1]) >= obs - 1e-12
    return hits / total


# labels the greedy declutter places on a neighbour's error bar: (dx, dy, ha)
LABEL_OVERRIDE = {("blca", "Lamina propria"): (-8, 4, "right")}


def fmt_p(p):
    """As in the paper's text: three decimals below 0.05, two above."""
    return f"{p:.3f}" if p < 0.05 else f"{p:.2f}"


def panel(ax, cohort, results_root, title):
    s = json.load(open(f"{results_root}/shapley_{cohort}/region_shapley_{cohort}.json"))
    pts = []
    for name in s["region_names"]:
        v = s["per_region"][name]
        phi = v["shapley_cindex"]
        pts.append({
            "label": SHORT.get(name, name[:14]),
            "w": v["fusion_weight"]["mean"] * 100,
            "s": phi["mean"] * 100,
            "lo": phi["ci95"][0] * 100,
            "hi": phi["ci95"][1] * 100,
            "sig": phi["ci95"][0] > 0 or phi["ci95"][1] < 0,
        })
    ws, ss = [q["w"] for q in pts], [q["s"] for q in pts]
    rho = spearmanr(ws, ss).correlation
    p = exact_perm_p(ws, ss)
    uniform = 100.0 / len(pts)

    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.axvline(uniform, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for q in pts:
        ax.errorbar(q["w"], q["s"], yerr=[[q["s"] - q["lo"]], [q["hi"] - q["s"]]],
                    fmt="none", ecolor=TEXT_SECONDARY, elinewidth=FS["elw"],
                    capsize=FS["cap"], zorder=2, alpha=0.85)
        ax.plot(q["w"], q["s"], "o", ms=FS["ms"], mfc=SERIES_1 if q["sig"] else "white",
                mec=SERIES_1, mew=FS["mew"], zorder=3)

    # same greedy label declutter as the ablation figure
    xr = (max(ws) - min(ws)) or 1.0
    yr = (max(ss) - min(ss)) or 1.0
    placed = []
    for q in sorted(pts, key=lambda z: (-z["s"], z["w"])):
        dx, dy, ha = 8, 4, "left"
        for _ in range(6):
            cx = q["w"] + (dx / 72.0) * xr * (1 if ha == "left" else -1) * 0.9
            cy = q["s"] + (dy / 72.0) * yr * 0.9
            if not any(abs(cx - px) < 0.16 * xr and abs(cy - py) < 0.075 * yr for px, py in placed):
                break
            if dy > 0:
                dy = -12
            elif ha == "left":
                ha, dx, dy = "right", -8, 4
            else:
                dy -= 11
        dx, dy, ha = LABEL_OVERRIDE.get((cohort, q["label"]), (dx, dy, ha))
        placed.append((cx, cy))
        ax.annotate(q["label"], (q["w"], q["s"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=FS["annot"], color=TEXT_PRIMARY)

    ax.set_xlabel("Region fusion weight (%)", fontsize=FS["label"], color=TEXT_PRIMARY)
    ax.set_ylabel("Shapley value (pp of c-index)", fontsize=FS["label"], color=TEXT_PRIMARY)
    ax.set_title(f"{title}    Spearman $\\rho$ = {rho:+.2f}, exact $p$ = {fmt_p(p)}",
                 fontsize=FS["title"], color=TEXT_PRIMARY, loc="left", pad=8)
    ax.tick_params(labelsize=FS["tick"], colors=TEXT_SECONDARY, length=0)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)
    pad = (max(ws) - min(ws)) * 0.28 or 0.5
    ax.set_xlim(min(ws) - pad, max(ws) + pad * 1.5)
    return rho, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="Figures/fig_weight_vs_shapley.pdf")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (cohort, title) in zip(axes, [("blca", "(a) TCGA-BLCA"), ("brca", "(b) TCGA-BRCA")]):
        rho, p = panel(ax, cohort, args.results, title)
        print(f"{cohort}: rho {rho:+.3f}, exact p {p:.4f}")
    fig.tight_layout(w_pad=3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=300, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
