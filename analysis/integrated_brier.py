#!/usr/bin/env python3
"""Integrated Brier score for the backbone, with a Kaplan-Meier reference.

The paper reports an IPCW Brier score at a single horizon (the median follow-up
time). That is one slice of a curve, and it is hard to read on its own: a Brier
score depends on the event rate, so 0.150 in BRCA and 0.335 in BLCA say more
about censoring than about the models.

This script fixes both problems. It integrates the IPCW Brier over a range of
horizons rather than one, and it divides by the same quantity computed for a
model that knows nothing about the patient, giving the index of prediction
accuracy

    IPA = 1 - IBS(model) / IBS(Kaplan-Meier)

which is 0 for a model no better than the marginal survival curve and 1 for a
perfect one. That is comparable across cohorts in a way the raw Brier is not.

Horizon grid: from the first event time to the largest time at which the
censoring distribution still has mass above `--min-g`. Beyond that the IPCW
weights 1/G(t) explode and the estimate is dominated by a handful of patients.

Inputs are the out-of-bag risk vectors, so these are honest estimates on
patients the model did not train on.

One correction is unavoidable first. `mean_risk` averages the Cox linear
predictor across independently trained bootstrap rounds, and a Cox score has no
inherent scale: each round sets its own, and the average inherits the spread
without inheriting a meaning. Left alone, the scores imply a hazard ratio of
roughly 35,000 between the 99th and 1st percentile patient in BLCA, so
exponentiating them produces survival curves far too confident to score well
under any proper scoring rule. We therefore fit a calibration slope, the
coefficient of a univariate Cox model on the risk score alone, and evaluate
beta * r. Both are reported. The slope is fitted on the same patients, so the
recalibrated figure is optimistic; it is the ranking, not the calibration, that
this script is really testing.

Usage:
    python analysis/integrated_brier.py --cohort blca \
        --oob-npz results/oob_risk_blca.npz \
        --out results/integrated_brier_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patient_error import _breslow_baseline_cumhaz, _km_censoring, _step_eval


def brier_at(times, events, risks, horizon):
    """IPCW Brier score at one horizon. Mirrors patient_error.ipcw_brier."""
    t, e, r = times, events, risks
    ev_t, H0, shift = _breslow_baseline_cumhaz(t, e, r)
    H0_tau = _step_eval(ev_t, H0, np.array([horizon]), default=0.0)[0]
    surv = np.exp(-H0_tau * np.exp(r - shift))

    g_t, g_v = _km_censoring(t, e)
    G_tau = _step_eval(g_t, g_v, np.array([horizon]))[0]
    G_Ti = _step_eval(g_t, g_v, t)

    c = np.full(len(t), np.nan)
    died = (t <= horizon) & (e == 1)
    alive = t > horizon
    c[died] = (0.0 - surv[died]) ** 2 / G_Ti[died]
    c[alive] = (1.0 - surv[alive]) ** 2 / G_tau
    ok = np.isfinite(c)
    return float(c[ok].mean()) if ok.any() else np.nan


def horizon_grid(times, events, min_g=0.2, n=100):
    """Times over which to integrate, truncated where IPCW weights blow up."""
    g_t, g_v = _km_censoring(times, events)
    usable = g_t[g_v >= min_g]
    hi = float(usable.max()) if len(usable) else float(np.quantile(times, 0.8))
    lo = float(np.min(times[events == 1]))
    hi = max(hi, lo * 1.01)
    return np.linspace(lo, hi, n)


def integrated_brier(times, events, risks, grid):
    """Trapezoid integral of the Brier curve, normalised by the time range."""
    curve = np.array([brier_at(times, events, risks, h) for h in grid])
    ok = np.isfinite(curve)
    if ok.sum() < 2:
        return np.nan, curve
    return float(np.trapezoid(curve[ok], grid[ok]) / (grid[ok][-1] - grid[ok][0])), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--min-g", type=float, default=0.2,
                    help="truncate the grid where the censoring survival drops "
                         "below this; guards against exploding IPCW weights")
    ap.add_argument("--n-grid", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="skip the calibration slope and score the raw averaged "
                         "linear predictor; reported either way for contrast")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.oob_npz, allow_pickle=True)
    t = np.asarray(d["pfi_time"], float)
    e = np.asarray(d["pfi_event"], float)
    r = np.asarray(d["mean_risk"], float)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    null = np.zeros_like(r)          # a model that knows nothing about the patient

    import pandas as pd
    from lifelines import CoxPHFitter
    slope = float(CoxPHFitter().fit(pd.DataFrame({"r": r, "t": t, "e": e}),
                                    duration_col="t", event_col="e").params_["r"])
    r_cal = r if args.no_recalibrate else slope * r

    grid = horizon_grid(t, e, args.min_g, args.n_grid)
    ibs_raw, _ = integrated_brier(t, e, r, grid)
    ibs_model, curve_model = integrated_brier(t, e, r_cal, grid)
    ibs_null, curve_null = integrated_brier(t, e, null, grid)
    ipa = 1.0 - ibs_model / ibs_null

    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(t), len(t))
        if e[idx].sum() < 5:
            continue
        g = horizon_grid(t[idx], e[idx], args.min_g, 25)
        m, _ = integrated_brier(t[idx], e[idx], r_cal[idx], g)
        z, _ = integrated_brier(t[idx], e[idx], np.zeros(len(idx)), g)
        if np.isfinite(m) and np.isfinite(z) and z > 0:
            boot.append(1.0 - m / z)
    boot = np.asarray(boot)

    rep = {
        "cohort": args.cohort,
        "endpoint": "PFI",
        "source_oob_npz": os.path.abspath(args.oob_npz),
        "n_patients": int(len(t)),
        "n_events": int(e.sum()),
        "grid": {"t_min_days": float(grid[0]), "t_max_days": float(grid[-1]),
                 "n_points": int(len(grid)), "min_censoring_survival": args.min_g,
                 "rule": ("first event time to the last time at which the "
                          "Kaplan-Meier estimate of the censoring distribution "
                          "is at least min_g")},
        "calibration_slope": slope,
        "recalibrated": not args.no_recalibrate,
        "integrated_brier_raw_score": ibs_raw,
        "integrated_brier_model": ibs_model,
        "integrated_brier_kaplan_meier": ibs_null,
        "index_of_prediction_accuracy": ipa,
        "ipa_ci95": [float(np.quantile(boot, 0.025)),
                     float(np.quantile(boot, 0.975))] if len(boot) else None,
        "n_bootstrap_used": int(len(boot)),
        "interpretation": (
            "IPA is 0 when the model does no better than the marginal survival "
            "curve and 1 when it is perfect. Unlike the raw Brier score it does "
            "not move with the event rate, so the two cohorts can be compared."),
        "brier_curve": [{"t_days": float(a), "model": float(b), "kaplan_meier": float(c)}
                        for a, b, c in zip(grid, curve_model, curve_null)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"[{args.cohort}] n={len(t)}, events={int(e.sum())}")
    print(f"[{args.cohort}] grid {grid[0]:.0f}-{grid[-1]:.0f} days ({len(grid)} points)")
    print(f"[{args.cohort}] calibration slope = {slope:.4f}")
    print(f"[{args.cohort}] IBS raw = {ibs_raw:.4f}, IBS recalibrated = {ibs_model:.4f}, "
          f"IBS Kaplan-Meier = {ibs_null:.4f}")
    ci = rep["ipa_ci95"]
    print(f"[{args.cohort}] IPA = {ipa:+.4f}" +
          (f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
