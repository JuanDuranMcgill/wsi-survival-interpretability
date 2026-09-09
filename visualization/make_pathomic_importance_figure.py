#!/usr/bin/env python3
"""Regenerate the pathomic feature-importance figure from the corrected run.

Replaces panels (c) and (d) of the original combined figure. Addresses the
supervisor's 1:40pm comment on that figure:

- "Simplify this figure": the original encoded importance in bar length, region
  in fill colour, direction in bar edge colour, and univariate significance in
  stars, four encodings on one mark. Here bar length is the only visual
  encoding. Region and direction live in the axis label as text, so nothing is
  carried by colour alone and no legend is needed.
- "Show effect sizes and confidence intervals rather than relying on
  significance stars": stars are gone; every bar carries its bootstrap 95% CI.
- "Replace 'higher = worse prognosis' with 'associated with higher
  model-derived risk'": the direction arrow is defined that way in the caption,
  since the surrogate regresses the model's risk score and not survival.
- "Font size is too small in the caption": base font raised to 9pt with the
  axis title at 10pt, sized for a single-column figure at 100% scale.

Reads the committed per-feature importances, so the numbers cannot drift from
what the analysis produced.

Usage:
    python visualization/make_pathomic_importance_figure.py \
        --out Figures/fig_pathomic_importance.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Palette parameters, from the design system's light mode. Single series, so one
# hue; identity is carried by the axis labels rather than by colour.
SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2df"

# Compact names for the pyradiomics first-order statistics.
METRIC_SHORT = {
    "InterquartileRange": "IQR",
    "RobustMeanAbsoluteDeviation": "robust MAD",
    "MeanAbsoluteDeviation": "MAD",
    "RootMeanSquared": "RMS",
    "90Percentile": "90th pct",
    "10Percentile": "10th pct",
    "Variance": "variance",
    "Skewness": "skewness",
    "Kurtosis": "kurtosis",
    "Entropy": "entropy",
    "Median": "median",
    "Mean": "mean",
    "Minimum": "min",
    "Maximum": "max",
    "Range": "range",
    "Uniformity": "uniformity",
    "Energy": "energy",
    "TotalEnergy": "total energy",
    "MeanAbsoluteDeviation_std": "MAD",
}

REGION_SHORT = {
    "adipose": "Adipose", "tils": "TILs", "dcis": "DCIS", "tumor": "Tumour",
    "stroma": "Stroma", "tdlu": "TDLU", "muscle": "Muscle", "vessels": "Vessels",
    "necrosis": "Necrosis", "immune": "Immune", "lamina": "Lamina propria",
    "muscularis": "Muscularis", "urothelium": "Urothelium",
}


def label_for(row) -> str:
    """Region, channel and statistic as text, plus the direction arrow."""
    region = REGION_SHORT.get(row["region"], str(row["region"]).capitalize())
    metric = METRIC_SHORT.get(row["metric"], str(row["metric"]))
    stat = {"mean": "mean", "std": "SD"}.get(str(row["stat"]), str(row["stat"]))
    d = str(row["direction"])
    # ASCII only: the arrow glyphs are absent from Helvetica, and the sign is
    # clearer anyway. Defined in the caption as direction of association with
    # the model-derived risk score.
    sign = "+risk" if "higher survival_target" in d else (
        "-risk" if "lower survival_target" in d else "n/d")
    return f"{region} {row['channel']} {metric} ({stat})  {sign}"


def panel(ax, csv_path, title, top_n):
    d = pd.read_csv(csv_path).sort_values("mean_importance", ascending=False).head(top_n)
    d = d.iloc[::-1]  # largest at the top once drawn
    y = range(len(d))
    imp = d["mean_importance"].to_numpy() * 100
    lo = (d["mean_importance"] - d["ci_lo"]).to_numpy() * 100
    hi = (d["ci_hi"] - d["mean_importance"]).to_numpy() * 100

    ax.barh(list(y), imp, height=0.62, color=SERIES_1, zorder=3)
    ax.errorbar(imp, list(y), xerr=[lo, hi], fmt="none",
                ecolor=TEXT_SECONDARY, elinewidth=0.9, capsize=2.2, zorder=4)

    ax.set_yticks(list(y))
    ax.set_yticklabels([label_for(r) for _, r in d.iterrows()],
                       fontsize=8.2, color=TEXT_PRIMARY)
    ax.set_xlabel("Importance (% of total)", fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=7)
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_xlim(0, max(imp + hi) * 1.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--threshold", default="q50")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--out", default="Figures/fig_pathomic_importance.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (cohort, title) in zip(axes, [
        ("blca", "(a) TCGA-BLCA"), ("brca", "(b) TCGA-BRCA")]):
        csv = os.path.join(args.results_root,
                           f"reliability_{cohort}_radiomic_{args.threshold}",
                           "radiomic_importance.csv")
        if not os.path.exists(csv):
            raise SystemExit(f"missing {csv}")
        panel(ax, csv, title, args.top_n)

    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
