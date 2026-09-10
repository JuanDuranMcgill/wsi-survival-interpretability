#!/usr/bin/env python3
"""Test pathomic features against observed survival, and report their correlation
structure.

Closes two parts of the supervisor's comments:

- 1:37pm "If you want to claim that adipose features are prognostic, they should
  also be tested against actual survival" and "account for correlated features
  and multiple testing".
- 1:44pm "Explain how the adipose importance percentages were calculated and
  test if the adipose result remains after accounting for [...] region size."

The surrogate importances answer a different question: they say how well a
feature predicts the *model's risk score*. That is fidelity to the backbone, not
prognosis. A feature can dominate the surrogate while carrying no independent
association with outcome. This script asks the prognostic question directly.

Two analyses:

1. **Univariate Cox against PFI**, per feature, with Benjamini-Hochberg FDR
   control across all features in the cohort. Reported per region so the adipose
   claim can be read against the others. Also reports the region-size confound:
   the number of features a region contributes, since a region with more
   features has more chances to produce a hit.

2. **Correlation structure** among the top surrogate features. Impurity
   importance splits credit arbitrarily among correlated predictors, so a block
   of near-duplicate adipose features can look like several independent signals
   when it is one. Reports the mean absolute within-region and between-region
   Spearman correlation, and the effective number of independent directions from
   an eigenvalue decomposition.

Usage:
    python analysis/pathomic_vs_survival.py --cohort brca \
        --radiomics-csv ~/wsi-transfer/features/brca/radiomics_pre_corr.csv \
        --oob-npz results/oob_risk_brca.npz \
        --importance-csv results/reliability_brca_radiomic_q50/radiomic_importance.csv \
        --out results/pathomic_vs_survival_brca.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return np.clip(adj, 0, 1)


def region_of(name: str) -> str:
    return str(name).split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--radiomics-csv", required=True)
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--importance-csv", required=True)
    ap.add_argument("--top-n-corr", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.oob_npz, allow_pickle=True)
    surv = pd.DataFrame({
        "patient_id": [str(p) for p in d["patient_ids"]],
        "t": np.asarray(d["pfi_time"], float),
        "e": np.asarray(d["pfi_event"], float),
        "wsi_risk": np.asarray(d["mean_risk"], float),
    }).dropna(subset=["t", "e"]).set_index("patient_id")

    X = pd.read_csv(args.radiomics_csv, index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated(keep="first")]
    shared = [p for p in surv.index if p in X.index]
    X = X.loc[shared].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.9]
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.std(ddof=0) > 1e-9]

    # Drop saturated / near-constant features. The radiomics table contains
    # intensity features pinned at the 8-bit ceiling of 255 for almost every
    # patient (e.g. necrosis_E_..._Maximum_mean: 5 unique values across 384
    # patients, 99% at 255). A Cox fit on such a feature is driven by a handful
    # of outliers and produces absurd hazard ratios with vanishing p-values.
    # These are data artefacts, not prognostic signals.
    modal_frac = X.apply(lambda c: (c == c.mode().iloc[0]).mean())
    n_unique = X.nunique()
    degenerate = (modal_frac > 0.5) | (n_unique < 20)
    n_degenerate = int(degenerate.sum())
    dropped_examples = list(X.columns[degenerate][:5])
    X = X.loc[:, ~degenerate]
    Z = (X - X.mean()) / X.std(ddof=0)
    S = surv.loc[shared]
    print(f"[{args.cohort}] {len(shared)} patients matched, {Z.shape[1]} features "
          f"({n_degenerate} degenerate dropped), {int(S['e'].sum())} events")

    # --- 1. univariate Cox against observed PFI -----------------------------
    rows = []
    for col in Z.columns:
        df = pd.DataFrame({"x": Z[col].to_numpy(), "_t": S["t"].to_numpy(),
                           "_e": S["e"].to_numpy()})
        try:
            cph = CoxPHFitter().fit(df, duration_col="_t", event_col="_e")
            rows.append({"feature": col, "region": region_of(col),
                         "coef": float(cph.params_["x"]),
                         "hr": float(np.exp(cph.params_["x"])),
                         "p": float(cph.summary.loc["x", "p"])})
        except Exception:
            rows.append({"feature": col, "region": region_of(col),
                         "coef": np.nan, "hr": np.nan, "p": np.nan})
    cox = pd.DataFrame(rows)
    ok = cox["p"].notna()
    cox.loc[ok, "q"] = bh_fdr(cox.loc[ok, "p"].to_numpy())

    per_region = {}
    for r, g in cox.groupby("region"):
        sig = g["q"] < 0.05
        per_region[r] = {
            "n_features": int(len(g)),
            "n_fdr_significant": int(sig.sum()),
            "frac_fdr_significant": float(sig.mean()),
            "min_q": float(g["q"].min()) if g["q"].notna().any() else None,
            "best_feature": (str(g.loc[g["q"].idxmin(), "feature"])
                             if g["q"].notna().any() else None),
            "best_hr": (float(g.loc[g["q"].idxmin(), "hr"])
                        if g["q"].notna().any() else None),
        }

    # --- 2. correlation structure of the top surrogate features -------------
    imp = pd.read_csv(args.importance_csv).sort_values(
        "mean_importance", ascending=False)
    top = [f for f in imp["feature"].tolist() if f in Z.columns][:args.top_n_corr]
    corr_rep = {"n_top_features": len(top)}
    if len(top) >= 3:
        M = Z[top].to_numpy()
        rho, _ = spearmanr(M)
        rho = np.atleast_2d(rho)
        A = np.abs(rho)
        np.fill_diagonal(A, np.nan)
        regs = np.array([region_of(f) for f in top])
        same = regs[:, None] == regs[None, :]
        np.fill_diagonal(same, False)
        ev = np.linalg.eigvalsh(np.corrcoef(M.T))
        ev = np.clip(ev, 0, None)
        corr_rep.update({
            "mean_abs_spearman_all_pairs": float(np.nanmean(A)),
            "mean_abs_spearman_within_region": float(np.nanmean(A[same])),
            "mean_abs_spearman_between_region": float(np.nanmean(A[~same & ~np.eye(len(top), dtype=bool)])),
            "n_pairs_abs_rho_above_0.8": int(np.nansum(A > 0.8) // 2),
            "effective_n_independent_directions": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "top_region_counts": {r: int((regs == r).sum()) for r in sorted(set(regs))},
            "interpretation": (
                "Impurity importance divides credit among correlated predictors, "
                "so a block of near-duplicate features within one region can "
                "appear as several independent signals. Compare the effective "
                "number of independent directions against n_top_features."),
        })

    ad = per_region.get("adipose", {})
    rep = {
        "cohort": args.cohort, "endpoint": "PFI",
        "n_patients": len(shared), "n_events": int(S["e"].sum()),
        "n_features_tested": int(ok.sum()),
        "n_degenerate_features_dropped": n_degenerate,
        "degenerate_filter": ("dropped features whose modal value covers >50% of "
                              "patients or with <20 unique values; the radiomics "
                              "table contains intensity features saturated at the "
                              "8-bit ceiling of 255"),
        "degenerate_examples": [str(c) for c in dropped_examples],
        "multiple_testing": "Benjamini-Hochberg FDR across all features in the cohort",
        "note": ("Surrogate importance measures prediction of the model's risk "
                 "score. These Cox tests measure association with observed "
                 "survival. A feature can rank high on the former with no "
                 "support on the latter."),
        "n_fdr_significant_overall": int((cox["q"] < 0.05).sum()),
        "per_region_cox": dict(sorted(
            per_region.items(), key=lambda kv: -kv[1]["frac_fdr_significant"])),
        "adipose_summary": ad,
        "correlation_structure": corr_rep,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    cox.sort_values("p").to_csv(args.out.replace(".json", "_cox.csv"), index=False)

    print(f"[{args.cohort}] FDR-significant against PFI: "
          f"{rep['n_fdr_significant_overall']} / {rep['n_features_tested']} features")
    print(f"[{args.cohort}] {'region':12} {'nFeat':>6} {'nSig':>5} {'fracSig':>8} {'min q':>10}")
    for r, v in rep["per_region_cox"].items():
        mq = "n/a" if v["min_q"] is None else f"{v['min_q']:.3g}"
        print(f"[{args.cohort}] {r:12} {v['n_features']:6d} {v['n_fdr_significant']:5d} "
              f"{v['frac_fdr_significant']:8.3f} {mq:>10}")
    c = rep["correlation_structure"]
    if "effective_n_independent_directions" in c:
        print(f"[{args.cohort}] top-{c['n_top_features']} features: mean |rho| within-region "
              f"{c['mean_abs_spearman_within_region']:.3f}, between "
              f"{c['mean_abs_spearman_between_region']:.3f}, "
              f"effective independent directions "
              f"{c['effective_n_independent_directions']:.1f}")
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
