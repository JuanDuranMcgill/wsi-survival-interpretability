#!/usr/bin/env python3
"""Imbalance-aware evaluation for the MED3pa secondary classifier and profiles.

The paper evaluated the secondary classifier with plain accuracy against a
0.50 null. With a top-quantile high-error label the class balance is fixed by
construction, so a classifier that always predicts the majority class scores
`max(p, 1-p)` without learning anything. At the 80th percentile that is 0.80,
which is above every Caution profile the paper reports.

This module provides the metrics that are actually interpretable under
imbalance, plus the per-profile reporting the reviewers asked for:

- AUROC, tested against 0.50, which IS the valid null for that metric
- balanced accuracy and average precision
- the majority-class accuracy stated explicitly, so accuracy is comparable
- per profile: patient N, high-error prevalence inside the profile, coverage
- per profile: the ACTUAL survival-model error in that subgroup, with a
  bootstrap CI, which is the number a clinician would care about and which the
  paper never reported

Import these from the med3pa driver scripts rather than reimplementing them.
"""
from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


# ----------------------------------------------------------------------
# Secondary-classifier metrics
# ----------------------------------------------------------------------
def classifier_metrics(y_true, y_score, n_perm: int = 2000, seed: int = 0) -> dict:
    """Imbalance-aware metrics for the high-error classifier.

    `y_score` should be a continuous score or probability, not a hard label.
    The AUROC p-value is a permutation test against the 0.50 null, which
    avoids assuming normality of the bootstrap accuracy distribution.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    y, s = y[ok], s[ok]
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return {"n": int(n), "error": "degenerate label, metrics undefined"}

    prevalence = float(y.mean())
    majority_acc = float(max(prevalence, 1.0 - prevalence))
    auroc = float(roc_auc_score(y, s))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = roc_auc_score(rng.permutation(y), s)
    # Two-sided, with the +1 correction so p is never exactly 0.
    p = float((np.sum(np.abs(null - 0.5) >= abs(auroc - 0.5)) + 1) / (n_perm + 1))

    yhat = (s >= np.quantile(s, 1.0 - prevalence)).astype(int)
    return {
        "n": int(n),
        "prevalence_high_error": prevalence,
        "majority_class_accuracy": majority_acc,
        "plain_accuracy": float((yhat == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "auroc": auroc,
        "auroc_p_vs_0.5": p,
        "auroc_n_permutations": int(n_perm),
        "average_precision": float(average_precision_score(y, s)),
        "average_precision_baseline": prevalence,
        "note": (
            "Accuracy must be read against majority_class_accuracy, not 0.50. "
            "AUROC is the threshold-free metric; its null is 0.50."
        ),
    }


# ----------------------------------------------------------------------
# Actual survival-model error within a subgroup
# ----------------------------------------------------------------------
def subgroup_cindex(times, events, risks, mask=None):
    """Harrell c-index restricted to the patients selected by `mask`."""
    t = np.asarray(times, float)
    e = np.asarray(events, float)
    r = np.asarray(risks, float)
    if mask is not None:
        m = np.asarray(mask, bool)
        t, e, r = t[m], e[m], r[m]
    if len(t) < 2:
        return float("nan"), 0
    ti, tj, ei = t[:, None], t[None, :], e[:, None]
    earlier = (ti < tj) & (ei == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    conc = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0)
    total = int(earlier.sum())
    return (float(conc.sum() / total) if total else float("nan")), total


def _bootstrap_ci(fn, n, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def profile_report(
    mask,
    high_error_label,
    patient_error,
    times,
    events,
    risks,
    n_total=None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Everything the reviewers asked to see for one Trust/Caution profile.

    `mask` is a boolean array over the evaluation cohort selecting the patients
    who satisfy the profile rule.
    """
    m = np.asarray(mask, bool)
    n_in = int(m.sum())
    n_total = int(len(m) if n_total is None else n_total)
    if n_in == 0:
        return {"n_patients": 0, "coverage_pct": 0.0, "error": "empty profile"}

    y = np.asarray(high_error_label).astype(int)
    err = np.asarray(patient_error, dtype=float)
    t = np.asarray(times, float)
    e = np.asarray(events, float)
    r = np.asarray(risks, float)

    cidx_in, n_pairs_in = subgroup_cindex(t, e, r, m)
    idx_in = np.flatnonzero(m)

    def _boot_cidx(sample_idx):
        sel = idx_in[sample_idx]
        c, _ = subgroup_cindex(t[sel], e[sel], r[sel])
        return c

    ci_lo, ci_hi = _bootstrap_ci(_boot_cidx, n_in, n_boot=n_boot, seed=seed)

    err_in = err[m]
    finite = np.isfinite(err_in)

    return {
        "n_patients": n_in,
        "coverage_pct": float(100.0 * n_in / n_total),
        "n_events_in_profile": int((e[m] == 1).sum()),
        "n_censored_in_profile": int((e[m] == 0).sum()),
        # class balance inside the subgroup, which the paper never reported
        "high_error_prevalence_in_profile": float(y[m].mean()),
        "high_error_prevalence_overall": float(y.mean()),
        "majority_class_accuracy_in_profile": float(max(y[m].mean(), 1 - y[m].mean())),
        # the actual survival-model performance in this subgroup
        "survival_model_cindex_in_profile": cidx_in,
        "survival_model_cindex_ci95": [ci_lo, ci_hi],
        "n_comparable_pairs_in_profile": n_pairs_in,
        "mean_patient_error_in_profile": float(err_in[finite].mean()) if finite.any() else float("nan"),
        "caveat": (
            "A profile describes the average error behaviour of a subgroup. It "
            "does not imply every patient satisfying the rule is high-error, nor "
            "that low-error patients cannot satisfy it."
        ),
    }


# ----------------------------------------------------------------------
# Locked discovery / evaluation split
# ----------------------------------------------------------------------
def discovery_eval_split(patient_ids, frac_discovery=0.6, seed=20260908, stratify=None):
    """Deterministic patient-level split so profile discovery never sees the
    evaluation set. Returns (discovery_mask, eval_mask).

    Pass `stratify` (for example the high-error label, or the event indicator)
    to keep its proportion equal across the two halves.
    """
    pids = np.asarray(patient_ids)
    n = len(pids)
    rng = np.random.default_rng(seed)
    disc = np.zeros(n, dtype=bool)

    if stratify is None:
        idx = rng.permutation(n)
        disc[idx[: int(round(frac_discovery * n))]] = True
    else:
        strat = np.asarray(stratify)
        for level in np.unique(strat):
            pos = np.flatnonzero(strat == level)
            pos = pos[rng.permutation(len(pos))]
            disc[pos[: int(round(frac_discovery * len(pos)))]] = True

    if disc.all() or (~disc).all():
        raise ValueError("split produced an empty side; check frac_discovery")
    return disc, ~disc
