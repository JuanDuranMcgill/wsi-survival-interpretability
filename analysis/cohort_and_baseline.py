#!/usr/bin/env python3
"""Cohort descriptives, c-index confidence intervals, and a clinical Cox baseline.

Closes three supervisor comments in one pass, all from data already on disk:

- 1:18pm "Report confidence intervals for the c-index values and explain exactly
  how these values were obtained. It should be clear if they come from training,
  validation, out-of-bag, or held-out patients."
- 1:32pm "Describe the cohorts and data splits more clearly. Report exclusions,
  final patient numbers, deaths/censored cases, follow-up, handling of multiple
  slides per patient, and which patients were used for training, validation,
  explainability, and reliability analyses. Also check for TCGA site effects and
  compare the WSI model with a simple clinical Cox model if possible."

The comparison against the clinical Cox model uses the SAME bootstrap
out-of-bag protocol as the backbone, so the two c-indices are directly
comparable. An in-sample Cox fit would be measured on data it saw, which is the
error this whole revision exists to correct.

Three models are compared:
  clinical  - Cox on clinical features only
  wsi       - the graph model's out-of-bag risk score
  combined  - Cox on clinical features plus the WSI risk as one extra covariate

If `combined` does not beat `clinical`, the WSI model adds nothing over routine
clinical variables, which is the question a reviewer will ask.

Usage:
    python analysis/cohort_and_baseline.py --cohort blca \
        --oob-npz results/oob_risk_blca.npz \
        --clinical-csv /path/to/clinical_features.csv \
        --out results/cohort_and_baseline_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


# ----------------------------------------------------------------------
def cindex(t, e, r):
    """Harrell c-index. Higher risk should mean shorter time."""
    t, e, r = np.asarray(t, float), np.asarray(e, float), np.asarray(r, float)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    if len(t) < 2:
        return float("nan")
    earlier = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    conc = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0)
    n = int(earlier.sum())
    return float(conc.sum() / n) if n else float("nan")


def bootstrap_ci(t, e, r, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI for a c-index, resampling patients."""
    rng = np.random.default_rng(seed)
    n = len(t)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = cindex(t[idx], e[idx], r[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def source_site(barcode: str) -> str:
    """TCGA tissue source site: characters 6-7 of the barcode."""
    parts = str(barcode).split("-")
    return parts[1] if len(parts) > 1 else "??"


# ----------------------------------------------------------------------
def cox_oob(X, t, e, n_rounds=100, train_frac=0.8, seed=0, penalizer=0.1):
    """Bootstrap out-of-bag c-index for a Cox model.

    Mirrors the backbone's protocol: sample patients with replacement for
    training, score the out-of-bag remainder, average the per-round c-index.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    idx = np.arange(n)
    scores = []
    for _ in range(n_rounds):
        tr = rng.choice(idx, int(train_frac * n), replace=True)
        oob = np.setdiff1d(idx, np.unique(tr))
        if len(oob) < 20 or int(e[oob].sum()) < 5:
            continue
        df = X.iloc[tr].copy()
        df["_t"], df["_e"] = t[tr], e[tr]
        # A bootstrap draw can contain duplicate rows and near-constant columns;
        # drop zero-variance covariates so the fit stays conditioned.
        keep = [c for c in X.columns if df[c].std(ddof=0) > 1e-9]
        if not keep:
            continue
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df[keep + ["_t", "_e"]], duration_col="_t", event_col="_e")
            risk = cph.predict_partial_hazard(X.iloc[oob][keep]).to_numpy()
        except Exception:
            continue
        v = cindex(t[oob], e[oob], risk)
        if np.isfinite(v):
            scores.append(v)
    if not scores:
        return {"n_rounds_used": 0, "cindex_oob_mean": float("nan")}
    s = np.array(scores)
    return {
        "n_rounds_used": len(s),
        "cindex_oob_mean": float(s.mean()),
        "cindex_oob_sd": float(s.std(ddof=1)),
        "cindex_oob_ci95": [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))],
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--clinical-csv", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cox-rounds", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.oob_npz, allow_pickle=True)
    pids = np.array([str(p) for p in d["patient_ids"]])
    wsi = np.asarray(d["mean_risk"], float)
    t = np.asarray(d["pfi_time"], float)
    e = np.asarray(d["pfi_event"], float)
    oob_count = np.asarray(d["oob_count"], int)

    rep = {"cohort": args.cohort, "endpoint": "PFI",
           "source_oob_npz": os.path.abspath(args.oob_npz)}

    # --- cohort descriptives -------------------------------------------------
    ok = np.isfinite(t) & np.isfinite(e)
    ev = e[ok] == 1
    rep["cohort_descriptives"] = {
        "n_patients_modelled": int(len(pids)),
        "n_with_survival": int(ok.sum()),
        "n_events": int(ev.sum()),
        "n_censored": int((~ev).sum()),
        "censored_fraction": float((~ev).mean()),
        "pfi_time_days": {
            "median_all": float(np.median(t[ok])),
            "median_censored": float(np.median(t[ok][~ev])),
            "median_event": float(np.median(t[ok][ev])),
            "iqr_all": [float(np.quantile(t[ok], .25)), float(np.quantile(t[ok], .75))],
            "max": float(t[ok].max()),
        },
        "slides_per_patient": (
            "One slide per patient by construction: the dataset builder selects "
            "DX1 slides only and keys on the first 12 barcode characters, so "
            "each patient contributes exactly one slide."),
        "oob_rounds_per_patient": {
            "min": int(oob_count.min()), "median": int(np.median(oob_count)),
            "max": int(oob_count.max()),
            "n_never_oob": int((oob_count == 0).sum())},
        "patient_set_usage": (
            "All modelled patients enter every analysis. Backbone training and "
            "validation are per-round bootstrap splits (80% with replacement, "
            "out-of-bag remainder validated). Explainability and reliability use "
            "out-of-bag risk averaged over the rounds in which each patient was "
            "out-of-bag. Reliability additionally splits patients 60/40 into "
            "disjoint profile-discovery and profile-evaluation halves."),
    }

    # --- TCGA source-site effects -------------------------------------------
    sites = np.array([source_site(p) for p in pids])
    counts = Counter(sites)
    big = [s for s, c in counts.items() if c >= 10]
    site_rep = {"n_sites": len(counts),
                "n_sites_with_ge10_patients": len(big),
                "largest_site_share": float(max(counts.values()) / len(sites)),
                "top_sites": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8])}
    if len(big) >= 3:
        from scipy.stats import kruskal
        groups_r = [wsi[(sites == s) & np.isfinite(wsi)] for s in big]
        groups_e = [e[(sites == s) & ok] for s in big]
        try:
            hr, pr = kruskal(*[g for g in groups_r if len(g) > 1])
            he, pe = kruskal(*[g for g in groups_e if len(g) > 1])
            site_rep["kruskal_risk_by_site"] = {"H": float(hr), "p": float(pr)}
            site_rep["kruskal_event_by_site"] = {"H": float(he), "p": float(pe)}
            site_rep["interpretation"] = (
                "A small p for risk-by-site means the model's output varies with "
                "the contributing institution, which is a confounder for any "
                "morphology-based claim. Compare against event-by-site, since "
                "genuine case-mix differences between sites also produce this.")
        except Exception as ex:
            site_rep["kruskal_error"] = str(ex)
    rep["tcga_source_site"] = site_rep

    # --- WSI c-index with CI -------------------------------------------------
    lo, hi = bootstrap_ci(t[ok], e[ok], wsi[ok], n_boot=args.n_boot)
    rep["wsi_cindex"] = {
        "value": cindex(t[ok], e[ok], wsi[ok]),
        "ci95": [lo, hi],
        "n_bootstrap": args.n_boot,
        "provenance": (
            "Computed from out-of-bag predictions: each patient's risk is the "
            "mean over only those bootstrap rounds in which they were out of "
            "bag. This is neither a training nor an in-sample figure. The "
            "in-sample counterpart is reported alongside for contrast."),
        "insample_cindex_for_contrast": cindex(
            t[ok], e[ok], np.asarray(d["insample_risk"], float)[ok]),
    }

    # --- clinical Cox baseline ----------------------------------------------
    clin = pd.read_csv(args.clinical_csv, index_col=0)
    clin.index = clin.index.astype(str)
    shared = [p for p in pids if p in clin.index]
    rep["clinical_baseline"] = {"n_matched_to_clinical": len(shared)}
    if len(shared) >= 50:
        pos = {p: k for k, p in enumerate(pids)}
        sel = [pos[p] for p in shared]
        X = clin.loc[shared].apply(pd.to_numeric, errors="coerce")
        X = X.loc[:, X.notna().mean() > 0.7]
        X = X.fillna(X.median(numeric_only=True))
        X = X.loc[:, X.std(ddof=0) > 1e-9]
        X = (X - X.mean()) / X.std(ddof=0)
        tt, ee, rr = t[sel], e[sel], wsi[sel]
        m = np.isfinite(tt) & np.isfinite(ee) & np.isfinite(rr)
        X, tt, ee, rr = X[m], tt[m], ee[m], rr[m]

        clin_only = cox_oob(X, tt, ee, args.cox_rounds)
        Xc = X.copy(); Xc["wsi_risk"] = (rr - rr.mean()) / rr.std(ddof=0)
        combined = cox_oob(Xc, tt, ee, args.cox_rounds)
        wsi_here = cindex(tt, ee, rr)

        rep["clinical_baseline"].update({
            "n_used": int(len(tt)), "n_features": int(X.shape[1]),
            "feature_names": list(X.columns),
            "protocol": ("Bootstrap out-of-bag, matching the backbone: 80% of "
                         "patients sampled with replacement to fit, c-index scored "
                         "on the out-of-bag remainder, averaged over rounds."),
            "clinical_only": clin_only,
            "wsi_only_on_same_patients": wsi_here,
            "clinical_plus_wsi": combined,
            "wsi_increment_over_clinical": (
                float(combined["cindex_oob_mean"] - clin_only["cindex_oob_mean"])
                if np.isfinite(combined.get("cindex_oob_mean", np.nan)) else None),
            "interpretation": (
                "If clinical_plus_wsi does not exceed clinical_only, the WSI "
                "model adds nothing over routine clinical variables on this "
                "cohort. Read the increment against clinical_only's CI width."),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    cd = rep["cohort_descriptives"]
    print(f"[{args.cohort}] N={cd['n_patients_modelled']} events={cd['n_events']} "
          f"censored={cd['censored_fraction']:.1%} median PFI={cd['pfi_time_days']['median_all']:.0f}d")
    w = rep["wsi_cindex"]
    print(f"[{args.cohort}] WSI c-index {w['value']:.4f} "
          f"[{w['ci95'][0]:.4f}, {w['ci95'][1]:.4f}]   in-sample {w['insample_cindex_for_contrast']:.4f}")
    if "clinical_only" in rep["clinical_baseline"]:
        b = rep["clinical_baseline"]
        print(f"[{args.cohort}] clinical Cox OOB {b['clinical_only']['cindex_oob_mean']:.4f} "
              f"| WSI {b['wsi_only_on_same_patients']:.4f} "
              f"| combined {b['clinical_plus_wsi']['cindex_oob_mean']:.4f} "
              f"(increment {b['wsi_increment_over_clinical']:+.4f})")
    s = rep["tcga_source_site"]
    if "kruskal_risk_by_site" in s:
        print(f"[{args.cohort}] site effect: risk p={s['kruskal_risk_by_site']['p']:.4g}, "
              f"event p={s['kruskal_event_by_site']['p']:.4g} "
              f"({s['n_sites_with_ge10_patients']} sites with >=10 patients)")
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
