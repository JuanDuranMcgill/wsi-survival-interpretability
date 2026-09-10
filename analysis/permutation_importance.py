#!/usr/bin/env python3
"""Permutation importance for the pathomic surrogate, and whether it agrees with
the impurity importance the paper reports.

Closes the last open part of the supervisor's 1:37pm comment: "specify the
feature-importance method, add permutation importance if needed, account for
correlated features".

The point is not a second ranking for its own sake. Impurity (mean decrease in
impurity) importance has a known failure mode: it inflates high-cardinality and
correlated predictors, and it is computed on the training data. Permutation
importance measures the drop in a held-out score when one feature is shuffled,
so it does not share that bias. If the two rankings agree, the reported ranking
is robust to the choice of method. If they disagree, the impurity ranking was an
artefact of the estimator and the paper must say so.

We already know the top features form a heavily correlated block (about 2.8 and
5.4 effective independent directions among the top thirty), which is exactly the
regime where impurity importance is least trustworthy. So this is a real check,
not a formality.

Note that permutation importance is itself degraded by correlated features, in
the opposite direction: shuffling one member of a duplicated pair leaves the
information intact through its partner, so both look unimportant. Both rankings
are therefore reported per region as well as per feature, since the regional
aggregate is stable under within-region correlation while individual features
are not.

Runtime is dominated by n_features x n_repeats forest evaluations. Defaults are
sized for a CPU node; --top-n restricts to the highest-impurity features for a
fast pass.

Usage on Narval (CPU account, does not contend with the ablation for GPU):

    sbatch --account=def-senger_cpu --time=3:00:00 --cpus-per-task=32 --mem=64G \
      --wrap "python analysis/permutation_importance.py --cohort brca \
        --radiomics-csv \$HOME/data/brca/radiomics_pre_corr.csv \
        --oob-npz results/oob_risk_brca.npz \
        --importance-csv results/reliability_brca_radiomic_q50/radiomic_importance.csv \
        --n-jobs 32 --out results/permutation_importance_brca.json"
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def region_of(name: str) -> str:
    return str(name).split("_")[0]


def load(radiomics_csv, oob_npz):
    d = np.load(oob_npz, allow_pickle=True)
    # Accept either key name ("mean_risk" from oob_risk_*.npz aggregates or
    # "risk" from patient_error_*_oob.npz per-patient outputs).
    risk_key = "mean_risk" if "mean_risk" in d else "risk"
    y = pd.Series(np.asarray(d[risk_key], float),
                  index=[str(p) for p in d["patient_ids"]]).dropna()

    X = pd.read_csv(radiomics_csv, index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated(keep="first")]
    shared = [p for p in y.index if p in X.index]
    X = X.loc[shared].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.9]
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.std(ddof=0) > 1e-9]
    return X, y.loc[shared]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--radiomics-csv", required=True)
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--importance-csv", required=True,
                    help="the impurity importances to compare against")
    ap.add_argument("--first-order-only", action="store_true",
                    help="restrict to first-order features, matching the paper's "
                         "importance analysis")
    ap.add_argument("--top-n", type=int, default=0,
                    help="restrict to the N highest-impurity features; 0 uses all")
    ap.add_argument("--n-trees", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, y = load(args.radiomics_csv, args.oob_npz)
    imp = pd.read_csv(args.importance_csv).set_index("feature")

    if args.first_order_only:
        X = X.loc[:, [c for c in X.columns if "firstorder" in c]]
    if args.top_n:
        ranked = [f for f in imp.sort_values("mean_importance", ascending=False).index
                  if f in X.columns][:args.top_n]
        X = X.loc[:, ranked]
    print(f"[{args.cohort}] {X.shape[0]} patients x {X.shape[1]} features, "
          f"target = out-of-bag risk score")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    rf = RandomForestRegressor(n_estimators=args.n_trees, n_jobs=args.n_jobs,
                               random_state=args.seed, oob_score=True)
    t0 = time.time()
    rf.fit(Xtr, ytr)
    print(f"[{args.cohort}] forest fitted in {time.time()-t0:.0f}s, "
          f"OOB R2 = {rf.oob_score_:.4f}, test R2 = {rf.score(Xte, yte):.4f}")

    t0 = time.time()
    pi = permutation_importance(rf, Xte, yte, n_repeats=args.n_repeats,
                                random_state=args.seed, n_jobs=args.n_jobs)
    print(f"[{args.cohort}] permutation importance in {time.time()-t0:.0f}s")

    df = pd.DataFrame({
        "feature": X.columns,
        "region": [region_of(c) for c in X.columns],
        "perm_mean": pi.importances_mean,
        "perm_std": pi.importances_std,
        "impurity_mean": [float(imp.at[c, "mean_importance"])
                          if c in imp.index else np.nan for c in X.columns],
    })
    both = df.dropna(subset=["impurity_mean"])
    rho, pval = spearmanr(both["perm_mean"], both["impurity_mean"])

    # Region-level aggregates are the stable comparison under within-region
    # correlation, since both methods are degraded by it at the feature level.
    reg = df.groupby("region").agg(
        n_features=("feature", "size"),
        perm_sum=("perm_mean", "sum"),
        impurity_sum=("impurity_mean", "sum")).reset_index()
    reg["perm_pct"] = 100 * reg["perm_sum"] / reg["perm_sum"].sum()
    reg["impurity_pct"] = 100 * reg["impurity_sum"] / reg["impurity_sum"].sum()
    rrho, rp = spearmanr(reg["perm_pct"], reg["impurity_pct"])

    top_perm = df.nlargest(15, "perm_mean")
    rep = {
        "cohort": args.cohort,
        "target": "out-of-bag risk score",
        "n_patients": int(X.shape[0]), "n_features": int(X.shape[1]),
        "first_order_only": bool(args.first_order_only),
        "top_n_restriction": args.top_n or None,
        "forest": {"n_trees": args.n_trees, "oob_r2": float(rf.oob_score_),
                   "test_r2": float(rf.score(Xte, yte)),
                   "test_size": args.test_size},
        "n_repeats": args.n_repeats,
        "agreement_feature_level": {"spearman_rho": float(rho), "p": float(pval),
                                    "n_features_compared": int(len(both))},
        "agreement_region_level": {"spearman_rho": float(rrho), "p": float(rp),
                                   "n_regions": int(len(reg))},
        "region_table": reg.sort_values("perm_pct", ascending=False).to_dict("records"),
        "top_15_by_permutation": top_perm.to_dict("records"),
        "n_features_with_negative_perm": int((df["perm_mean"] < 0).sum()),
        "caveat": ("Permutation importance is degraded by correlated features in "
                   "the opposite direction to impurity importance: shuffling one "
                   "member of a duplicated pair leaves the information available "
                   "through its partner, so both appear unimportant. The "
                   "region-level comparison is the more stable one."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    df.sort_values("perm_mean", ascending=False).to_csv(
        args.out.replace(".json", ".csv"), index=False)

    print(f"\n[{args.cohort}] impurity vs permutation agreement: "
          f"feature-level rho = {rho:+.3f} (p={pval:.3g}), "
          f"region-level rho = {rrho:+.3f} (p={rp:.3g})")
    print(f"[{args.cohort}] {'region':12} {'nFeat':>6} {'perm %':>8} {'impurity %':>11}")
    for r in rep["region_table"]:
        print(f"[{args.cohort}] {r['region']:12} {r['n_features']:6d} "
              f"{r['perm_pct']:8.1f} {r['impurity_pct']:11.1f}")
    print(f"[{args.cohort}] features with negative permutation importance: "
          f"{rep['n_features_with_negative_perm']} / {X.shape[1]}")
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
