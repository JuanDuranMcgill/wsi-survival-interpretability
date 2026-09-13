#!/usr/bin/env python3
"""Clinical feature associations with the model-derived risk score.

The clinical counterpart to make_pathomic_importance_figure.py, and the other
half of the supervisor's 1:40pm comment, which asked to split or simplify the
original four-panel figure. Two figures rather than one four-panel figure.

Why univariate Spearman rather than Random Forest impurity importance: impurity
importance is biased toward high-cardinality predictors, and the clinical
feature set makes that concrete. Impurity ranks age, weight and height highest,
all continuous with hundreds of distinct values, while the stage variables have
two to six levels. The univariate correlation is not subject to that bias and is
what the text describes.

Bar length is the only visual encoding; sign is in the position relative to
zero. Bootstrap CIs replace significance stars.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SERIES_1="#2a78d6"; TEXT_PRIMARY="#0b0b0b"; TEXT_SECONDARY="#52514e"; GRID="#e3e2df"

def pretty(name: str) -> str:
    """Readable label. Whole-word replacements only: an earlier substring rule
    turned "height" into "H&Eight"."""
    n = name.replace("_", " ")
    exact = {
        "number of lymphnodes positive by he": "Positive lymph nodes",
        "lymph node examined count": "Lymph nodes examined",
        "lymphovascular invasion present": "Lymphovascular invasion",
        "breast carcinoma estrogen receptor status": "Estrogen receptor status",
        "breast carcinoma progesterone receptor status": "Progesterone receptor status",
        "lab proc her2 neu immunohistochemistry receptor status": "HER2 status",
        "age at diagnosis": "Age at diagnosis",
        "hist of non mibc": "History of non-MIBC",
        "history of neoadjuvant treatment": "Neoadjuvant treatment",
        "pathologic t": "Pathologic T", "pathologic n": "Pathologic N",
        "pathologic m": "Pathologic M", "pathologic stage": "Pathologic stage",
        "menopause status peri 612 months since last menstrual period":
            "Menopause: perimenopausal",
        "menopause status post prior bilateral ovariectomy or 12 mo since lmp "
        "with no prior hysterectomy": "Menopause: postmenopausal",
        "menopause status pre 6 months since lmp and no prior bilateral "
        "ovariectomy and not on estrogen replacement": "Menopause: premenopausal",
        "menopause status indeterminate neither pre or postmenopausal":
            "Menopause: indeterminate",
        "neoplasm histologic grade high grade": "Histologic grade: high",
        "neoplasm histologic grade low grade": "Histologic grade: low",
    }
    if n in exact:
        return exact[n]
    n = n.replace("histological type ", "Hist. type: ")
    n = n.replace("tobacco smoking history ", "Smoking history ")
    return (n[:1].upper() + n[1:])[:38]


def panel(ax, coh, clin_csv, oob_npz, top_n, n_boot, seed):
    d = np.load(oob_npz, allow_pickle=True)
    y = pd.Series(np.asarray(d["mean_risk"], float),
                  index=[str(p) for p in d["patient_ids"]]).dropna()
    X = pd.read_csv(clin_csv, index_col=0); X.index = X.index.astype(str)
    sh = [p for p in y.index if p in X.index]
    X = X.loc[sh].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.7]
    X = X.fillna(X.median(numeric_only=True)); X = X.loc[:, X.std(ddof=0) > 1e-9]
    # Drop missingness-indicator dummies; they encode absence of a record, not
    # a clinical state, and are not interpretable as an association.
    X = X.loc[:, [c for c in X.columns if not c.endswith("_nan")]]
    yy = y.loc[sh].to_numpy(); rng = np.random.default_rng(seed); n = len(yy)
    rows = []
    for c in X.columns:
        xv = X[c].to_numpy(); r, p = spearmanr(xv, yy)
        boot = np.array([spearmanr(xv[i], yy[i])[0]
                         for i in (rng.integers(0, n, n) for _ in range(n_boot))])
        boot = boot[np.isfinite(boot)]
        rows.append({"feature": c, "rho": r, "p": p,
                     "lo": np.quantile(boot, .025), "hi": np.quantile(boot, .975)})
    df = pd.DataFrame(rows); df["a"] = df.rho.abs()
    df = df.sort_values("a", ascending=False).head(top_n).iloc[::-1]
    yv = range(len(df))
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, zorder=2)
    ax.barh(list(yv), df.rho, height=.62, color=SERIES_1, zorder=3)
    ax.errorbar(df.rho, list(yv),
                xerr=[df.rho - df.lo, df.hi - df.rho], fmt="none",
                ecolor=TEXT_SECONDARY, elinewidth=.9, capsize=2.2, zorder=4)
    ax.set_yticks(list(yv))
    ax.set_yticklabels([pretty(f) for f in df.feature], fontsize=8.2, color=TEXT_PRIMARY)
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(.7)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", default=os.path.expanduser("~/wsi-transfer/features"))
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default="Figures/fig_clinical_importance.pdf")
    a = ap.parse_args()
    plt.rcParams.update({"font.family":"sans-serif",
        "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
        "figure.facecolor":"white","axes.facecolor":"white","pdf.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax,(c,csv,t) in zip(axes, [
            ("blca","clinical_features.csv","(a) TCGA-BLCA"),
            ("brca","clinical_features_BRCA.csv","(b) TCGA-BRCA")]):
        panel(ax, c, os.path.join(a.features_root, csv),
              f"results/oob_risk_{c}.npz", a.top_n, a.n_boot, a.seed)
        ax.set_title(t, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=7)
        ax.set_xlabel("Spearman $\\rho$ with model-derived risk", fontsize=9.5,
                      color=TEXT_PRIMARY)
    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0]+".png", dpi=220, bbox_inches="tight")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
