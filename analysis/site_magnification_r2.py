#!/usr/bin/env python
"""
Scan magnification in the site analysis: partial R-squared of the out-of-bag risk
score.

Extends site_casemix in analysis/review_checks.py, which reports the share of
risk-score variance site explains after case mix (stage, T, N; plus ER, PR in
BRCA). Here, with the same OLS on the standardised risk score:

  - partial R-squared of magnification after case mix;
  - partial R-squared of site after case mix and magnification;
  - for reference, site after case mix (must reproduce review_checks.json) and
    magnification after case mix and site.

Magnification is the pixel size (openslide.mpp-x), ~0.25 um/px (40x) vs ~0.5
um/px (20x), from results/slide_magnification_per_slide.csv (one slide per
patient). It is coded as one indicator for ~0.5 um/px, plus one for no recorded
pixel size (7 BRCA slides, all from site OL); a sensitivity analysis drops those
patients instead.

The risk score is `risk` in patient_error_{cohort}_oob.npz, which reproduces the
paper's site values exactly (it is the oob_risk_{cohort}.npz mean_risk that
review_checks.py reads). The constants and site coding below mirror
review_checks.py, which needs lifelines and so is not imported.

    python analysis/site_magnification_r2.py --out results/site_magnification_r2.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
DATA = os.path.expanduser("~/data")
CLIN = {"blca": "clinical_features.csv", "brca": "clinical_features_BRCA.csv"}
CASEMIX = {"blca": ["pathologic_stage", "pathologic_T", "pathologic_N"],
           "brca": ["pathologic_stage", "pathologic_t", "pathologic_n",
                    "breast_carcinoma_estrogen_receptor_status",
                    "breast_carcinoma_progesterone_receptor_status"]}
MPP_20X = 0.375   # same cut as slide_magnification.mpp_class


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() or 1.0)


def load(c, slides):
    d = np.load(os.path.join(DATA, f"patient_error_{c}_oob.npz"), allow_pickle=True)
    df = pd.DataFrame({"pid": [str(p) for p in d["patient_ids"]], "t": d["pfi_time"],
                       "e": d["pfi_event"], "x": d["risk"]}).dropna()
    df = df[df.t > 0].reset_index(drop=True)
    X = pd.read_csv(os.path.join(DATA, CLIN[c]), index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated()].reindex(df.pid)
    X.index = range(len(df))
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.std() > 0]
    df["site"] = df.pid.str[5:7]
    s = slides[slides.cohort == c].set_index("patient_id")
    if s.index.duplicated().any():
        raise SystemExit(f"[{c}] more than one slide per patient in the per-slide table")
    df["mpp"] = s["mpp_x"].reindex(df.pid).to_numpy()
    if df.pid.isin(s.index).sum() != len(df):
        raise SystemExit(f"[{c}] patients missing from the per-slide table")
    return df, X


def site_dummies(df, min_n=10):
    counts = df.site.value_counts()
    site = df.site.where(df.site.isin(counts[counts >= min_n].index), "other")
    return pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)


def mag_dummies(df):
    M = pd.DataFrame({"mpp_0.5": (df.mpp >= MPP_20X).astype(float)})
    if df.mpp.isna().any():
        M["mpp_missing"] = df.mpp.isna().astype(float)
    return M.loc[:, M.std() > 0]


def partial_r2(y, base, add):
    """Share of the residual variance after `base` explained by adding `add`, with its F test."""
    def ols(M):
        A = np.column_stack([np.ones(len(y))] + ([M] if M.size else []))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(((y - A @ beta) ** 2).sum()), np.linalg.matrix_rank(A)
    rss0, k0 = ols(base)
    rss1, k1 = ols(np.column_stack([base, add]))
    df1, df2 = k1 - k0, len(y) - k1
    F = ((rss0 - rss1) / df1) / (rss1 / df2) if df1 > 0 else float("nan")
    return {"partial_r2": (rss0 - rss1) / rss0, "df": int(df1),
            "F": float(F), "p": float(stats.f.sf(F, df1, df2)) if df1 > 0 else None}


def analyse(df, X, c):
    y = zs(df.x)
    C = X[CASEMIX[c]].apply(zs).values
    S = site_dummies(df).values
    M = mag_dummies(df).values
    return {"n": int(len(df)), "n_site_levels": int(S.shape[1] + 1),
            "n_mpp_0.5": int((df.mpp >= MPP_20X).sum()), "n_mpp_missing": int(df.mpp.isna().sum()),
            "mag_terms": list(mag_dummies(df).columns),
            "site_given_casemix": partial_r2(y, C, S),
            "mag_given_casemix": partial_r2(y, C, M),
            "site_given_casemix_mag": partial_r2(y, np.column_stack([C, M]), S),
            "mag_given_casemix_site": partial_r2(y, np.column_stack([C, S]), M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default=os.path.join(RES, "slide_magnification_per_slide.csv"))
    ap.add_argument("--out", default=os.path.join(RES, "site_magnification_r2.json"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    slides = pd.read_csv(args.slides)
    saved = json.load(open(os.path.join(RES, "review_checks.json")))
    out = {"risk_score": "patient_error_{cohort}_oob.npz risk (out-of-bag), standardised",
           "magnification": f"openslide.mpp-x, ~0.5 um/px (20x) when >= {MPP_20X}, else ~0.25 um/px (40x)",
           "site": "TCGA source site, sites with < 10 patients pooled as 'other' (review_checks.site_casemix)"}
    for c in ["blca", "brca"]:
        df, X = load(c, slides)
        res = analyse(df, X, c)
        ref = saved[c]["site_casemix"]["risk_partial_r2_site"]
        res["check_site_given_casemix_vs_review_checks_json"] = {
            "computed_saved": [res["site_given_casemix"]["partial_r2"], ref],
            "match": abs(res["site_given_casemix"]["partial_r2"] - ref) < 1e-9}
        if df.mpp.isna().any():
            keep = df.mpp.notna().to_numpy()
            res["sensitivity_drop_missing_mpp"] = analyse(df[keep].reset_index(drop=True),
                                                          X[keep].reset_index(drop=True), c)
        out[c] = res
        print(f"[{c}] {json.dumps(res, indent=1)}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
