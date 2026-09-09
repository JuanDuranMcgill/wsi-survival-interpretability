#!/usr/bin/env python3
"""Per-patient prediction error for censored survival data.

Replaces compute_patient_error.py, which had a defect: `counts[i]` only
incremented when `events[i] == 1`, so every censored patient kept the
initialised value `errors[i] = 0.0` and was scored as perfectly predicted.
Because the MED3pa high-error label is the top quantile of this distribution,
only patients who had an event could ever be labelled high-error, making the
label a partial proxy for the outcome itself.

Two metrics are provided.

1. Symmetric pairwise concordance error (PRIMARY).
   Patient i's error is the fraction of the comparable pairs they participate
   in that the model ranks discordantly. A pair (i, j) is comparable when the
   outcome ordering is known under right-censoring, which is the case when the
   patient with the shorter time had an event. Patient i participates whether
   they are the earlier-event member or the later member, so censored patients
   receive real comparable-pair counts. Risk ties count as half a discordance,
   the usual Harrell convention. This is a per-patient decomposition of the
   c-index the paper already reports.

2. IPCW Brier score at a horizon (SECONDARY, reported as extra information).
   Uses a Breslow baseline hazard to turn the Cox risk score into a survival
   probability, and inverse-probability-of-censoring weights from a
   Kaplan-Meier estimate of the censoring distribution. Reported alongside the
   primary metric, not used for the label unless explicitly requested.

Usage:
    python patient_error.py --cohort blca \
        --final-npz /path/to/final_med3pa_input.npz \
        --cdr-xlsx  /path/to/TCGA-CDR-SupplementalTableS1.xlsx \
        --out results/patient_error_blca.json

Endpoint is PFI, matching the training scripts.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_backbone(final_npz: str, risk_key: str = "mean_risk"):
    """Return (patient_ids, risks) from the backbone's final npz."""
    d = np.load(final_npz, allow_pickle=True)
    if risk_key not in d:
        raise KeyError(
            f"{risk_key!r} not in {final_npz}. Available: {sorted(d.keys())}"
        )
    pids = np.array([str(p)[:12] for p in d["patient_ids"]])
    risks = np.asarray(d[risk_key], dtype=float)
    if len(pids) != len(risks):
        raise ValueError(f"length mismatch: {len(pids)} ids vs {len(risks)} risks")
    return pids, risks


def load_survival(cdr_xlsx: str, pids: np.ndarray):
    """Align PFI time and event to `pids`. Returns (times, events, mask)."""
    df = pd.read_excel(cdr_xlsx)
    df = df[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
    df["PFI.time"] = pd.to_numeric(df["PFI.time"], errors="coerce")
    df["PFI"] = pd.to_numeric(df["PFI"], errors="coerce")
    df = df.dropna(subset=["PFI.time", "PFI"])
    # Barcodes are unique across cancer types, so no type filter is needed.
    df = df.drop_duplicates(subset="bcr_patient_barcode", keep="first")
    lut = df.set_index("bcr_patient_barcode")

    times = np.full(len(pids), np.nan)
    events = np.full(len(pids), np.nan)
    for k, pid in enumerate(pids):
        if pid in lut.index:
            times[k] = float(lut.at[pid, "PFI.time"])
            events[k] = int(lut.at[pid, "PFI"])
    mask = np.isfinite(times) & np.isfinite(events)
    return times, events, mask


# ----------------------------------------------------------------------
# Metric 1: symmetric pairwise concordance error
# ----------------------------------------------------------------------
def symmetric_concordance_error(times, events, risks):
    """Per-patient discordance rate over all comparable pairs.

    A pair (i, j) is comparable iff the patient with the smaller time had an
    event. Concordant means that patient also carries the higher risk. Ties in
    risk count as 0.5 discordance. Every patient in a comparable pair is
    charged for it, so censored patients get non-zero counts.

    Returns (error, n_comparable) as float arrays of length N. Patients in no
    comparable pair get error = nan, not 0, so they cannot be silently treated
    as perfectly predicted.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    r = np.asarray(risks, dtype=float)
    n = len(t)

    # earlier[i, j] is True when i has the strictly smaller time and i had an
    # event, which makes the pair's outcome ordering known.
    ti, tj = t[:, None], t[None, :]
    ei = e[:, None]
    earlier = (ti < tj) & (ei == 1)          # i is the earlier-event member
    comparable = earlier | earlier.T          # symmetric: either side qualifies
    np.fill_diagonal(comparable, False)

    ri, rj = r[:, None], r[None, :]
    # For a comparable pair, the model is right when the earlier-event patient
    # has the higher risk. Build discordance from the earlier member's view,
    # then symmetrise so both members are charged identically.
    disc_from_i = np.where(earlier, np.where(ri > rj, 0.0, np.where(ri == rj, 0.5, 1.0)), 0.0)
    discordance = disc_from_i + disc_from_i.T

    n_comparable = comparable.sum(axis=1).astype(float)
    total_disc = discordance.sum(axis=1)

    error = np.full(n, np.nan)
    ok = n_comparable > 0
    error[ok] = total_disc[ok] / n_comparable[ok]
    return error, n_comparable


def global_cindex(times, events, risks):
    """Harrell c-index, for cross-checking against the paper's reported value."""
    t, e, r = np.asarray(times, float), np.asarray(events, float), np.asarray(risks, float)
    ti, tj, ei = t[:, None], t[None, :], e[:, None]
    earlier = (ti < tj) & (ei == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    conc = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0)
    total = earlier.sum()
    return float(conc.sum() / total) if total else float("nan")


# ----------------------------------------------------------------------
# Metric 2: IPCW Brier score
# ----------------------------------------------------------------------
def _km_censoring(times, events):
    """Kaplan-Meier estimate of the censoring survival function G(t)."""
    order = np.argsort(times)
    t_sorted = np.asarray(times, float)[order]
    cens = (np.asarray(events, float)[order] == 0).astype(float)
    n = len(t_sorted)
    at_risk = n - np.arange(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(cens == 1, 1.0 - 1.0 / at_risk, 1.0)
    g = np.cumprod(factors)
    uniq, idx = np.unique(t_sorted, return_index=True)
    return uniq, g[np.append(idx[1:] - 1, n - 1)]


def _step_eval(grid_t, grid_v, query, default=1.0):
    """Right-continuous step function lookup with a floor to avoid 1/0."""
    out = np.full(len(query), default, dtype=float)
    pos = np.searchsorted(grid_t, query, side="right") - 1
    valid = pos >= 0
    out[valid] = grid_v[pos[valid]]
    return np.clip(out, 1e-8, None)


def _breslow_baseline_cumhaz(times, events, risks):
    """Breslow estimate of the baseline cumulative hazard H0(t)."""
    t, e, r = np.asarray(times, float), np.asarray(events, float), np.asarray(risks, float)
    exp_r = np.exp(r - r.max())  # shift for numerical stability
    ev_times = np.unique(t[e == 1])
    H0, running = [], 0.0
    for tk in ev_times:
        at_risk = t >= tk
        denom = exp_r[at_risk].sum()
        d_k = int(((t == tk) & (e == 1)).sum())
        if denom > 0:
            running += d_k / denom
        H0.append(running)
    return ev_times, np.asarray(H0), r.max()


def ipcw_brier(times, events, risks, horizon=None):
    """Per-patient IPCW Brier contribution at `horizon`.

    Returns (contrib, horizon, brier). Patients who are censored before the
    horizon contribute nan, since they carry no information at that time.
    """
    t = np.asarray(times, float)
    e = np.asarray(events, float)
    r = np.asarray(risks, float)
    if horizon is None:
        horizon = float(np.median(t))

    ev_t, H0, shift = _breslow_baseline_cumhaz(t, e, r)
    H0_tau = _step_eval(ev_t, H0, np.array([horizon]), default=0.0)[0]
    surv_tau = np.exp(-H0_tau * np.exp(r - shift))       # S(tau | x_i)

    g_t, g_v = _km_censoring(t, e)
    G_tau = _step_eval(g_t, g_v, np.array([horizon]))[0]
    G_Ti = _step_eval(g_t, g_v, t)

    contrib = np.full(len(t), np.nan)
    died_early = (t <= horizon) & (e == 1)
    survived = t > horizon
    contrib[died_early] = (0.0 - surv_tau[died_early]) ** 2 / G_Ti[died_early]
    contrib[survived] = (1.0 - surv_tau[survived]) ** 2 / G_tau
    finite = np.isfinite(contrib)
    return contrib, float(horizon), float(contrib[finite].mean()) if finite.any() else float("nan")


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def distribution_report(error, n_comparable, events):
    """Everything needed to choose a label threshold with evidence."""
    finite = np.isfinite(error)
    q = [0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    rep = {
        "n_patients": int(len(error)),
        "n_scored": int(finite.sum()),
        "n_unscored_no_comparable_pairs": int((~finite).sum()),
        "n_events": int((events == 1).sum()),
        "n_censored": int((events == 0).sum()),
        "censored_fraction": float((events == 0).mean()),
        "n_exactly_zero_error": int((error[finite] == 0).sum()),
        "frac_exactly_zero_error": float((error[finite] == 0).mean()),
        "mean_comparable_pairs": float(np.mean(n_comparable[finite])),
        "min_comparable_pairs": int(np.min(n_comparable[finite])),
        "quantiles": {f"q{int(x*100)}": float(np.quantile(error[finite], x)) for x in q},
        "histogram_20_bins": {
            "counts": np.histogram(error[finite], bins=20, range=(0, 1))[0].tolist(),
            "bin_edges": np.histogram(error[finite], bins=20, range=(0, 1))[1].tolist(),
        },
    }
    # The defect check: is the label still outcome-confounded?
    for thr_q in (0.5, 0.7, 0.8, 0.9):
        thr = float(np.quantile(error[finite], thr_q))
        lab = np.zeros(len(error), dtype=bool)
        lab[finite] = error[finite] >= thr
        n_lab = int(lab.sum())
        rep[f"label_at_q{int(thr_q*100)}"] = {
            "threshold": thr,
            "n_high_error": n_lab,
            "prevalence": float(n_lab / max(finite.sum(), 1)),
            "majority_class_accuracy": float(max(n_lab, finite.sum() - n_lab) / max(finite.sum(), 1)),
            # If this is ~1.0 the label is still an outcome proxy.
            "frac_of_high_error_that_had_event": (
                float((events[lab] == 1).mean()) if n_lab else float("nan")
            ),
            "event_rate_in_low_error": (
                float((events[finite & ~lab] == 1).mean()) if (finite & ~lab).any() else float("nan")
            ),
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--final-npz", required=True)
    ap.add_argument("--cdr-xlsx", required=True)
    ap.add_argument("--risk-key", default="mean_risk")
    ap.add_argument("--horizon", type=float, default=None,
                    help="Brier horizon in days. Default: median follow-up time.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pids, risks = load_backbone(args.final_npz, args.risk_key)
    times, events, mask = load_survival(args.cdr_xlsx, pids)
    n_missing = int((~mask).sum())
    pids, risks, times, events = pids[mask], risks[mask], times[mask], events[mask]
    print(f"[{args.cohort}] {len(pids)} patients aligned, {n_missing} dropped for missing PFI")

    error, n_comp = symmetric_concordance_error(times, events, risks)
    cidx = global_cindex(times, events, risks)
    print(f"[{args.cohort}] global c-index = {cidx:.4f}  (cross-check against the paper)")
    print(f"[{args.cohort}] mean per-patient error = {np.nanmean(error):.4f}")

    brier_contrib, horizon, brier = ipcw_brier(times, events, risks, args.horizon)
    print(f"[{args.cohort}] IPCW Brier at t={horizon:.0f}d = {brier:.4f}")

    rep = distribution_report(error, n_comp, events)
    rep.update({
        "cohort": args.cohort,
        "endpoint": "PFI",
        "source_npz": os.path.abspath(args.final_npz),
        "risk_key": args.risk_key,
        "n_dropped_missing_survival": n_missing,
        "global_cindex": cidx,
        "primary_metric": "symmetric_pairwise_concordance_error",
        "ipcw_brier": {"horizon_days": horizon, "score": brier},
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{args.cohort}] wrote {args.out}")

    # Per-patient table, which is what the MED3pa scripts consume.
    csv_path = args.out.replace(".json", "_per_patient.csv")
    pd.DataFrame({
        "patient_id": pids,
        "error": error,
        "n_comparable_pairs": n_comp,
        "ipcw_brier_contrib": brier_contrib,
        "pfi_time": times,
        "pfi_event": events.astype(int),
        "risk": risks,
    }).to_csv(csv_path, index=False)
    print(f"[{args.cohort}] wrote {csv_path}")

    npz_path = args.out.replace(".json", ".npz")
    np.savez(npz_path, patient_ids=pids, error=error,
             n_comparable_pairs=n_comp, risk=risks,
             pfi_time=times, pfi_event=events.astype(int))
    print(f"[{args.cohort}] wrote {npz_path}  (drop-in replacement for med3pa_error_input.npz)")


if __name__ == "__main__":
    main()
