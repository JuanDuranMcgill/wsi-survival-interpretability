"""Within-cancer-type metrics (design decision 6): only pairs of patients of the same type are compared.

cindex_within      Harrell's C over same-type comparable pairs (pairs pooled over types, not C-indices averaged).
cox_loss_stratified  Cox partial likelihood with risk sets inside each type (the paper's cox_ph_loss per type).
patient_error_within  the paper's symmetric_concordance_error with pairs restricted to the same type.
"""
import numpy as np


def _pairs(t, e, r):
    """(concordant + 0.5 * tied risk, comparable) over pairs where the earlier patient had an event; an event and a
    censoring at the same time count as comparable (censored assumed to live longer), as the paper's fast_cindex."""
    ti, tj = t[:, None], t[None, :]
    earlier = ((ti < tj) | ((ti == tj) & (e[None, :] == 0))) & (e[:, None] == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    num = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0).sum()
    return num, earlier.sum()


def cindex_within(time, event, risk, stratum):
    t, e, r, s = (np.asarray(x) for x in (time, event, risk, stratum))
    t, e, r = t.astype(float), e.astype(int), r.astype(float)
    num = den = 0.0
    for g in np.unique(s):
        m = s == g
        a, b = _pairs(t[m], e[m], r[m])
        num, den = num + a, den + b
    return num / den if den else float("nan")


def cox_loss_stratified(risk, time, event, stratum):
    import torch
    s = np.asarray(stratum)
    total = torch.zeros((), dtype=risk.dtype, device=risk.device)
    for g in np.unique(s):
        idx = torch.as_tensor(np.flatnonzero(s == g), device=risk.device)
        rk, tm, ev = risk[idx], time[idx], event[idx]
        order = torch.argsort(tm, descending=True)
        rk, ev = rk[order], ev[order]
        total = total - ((rk - torch.logcumsumexp(rk, dim=0)) * ev).sum()
    return total / max(float(event.sum()), 1.0)


def patient_error_within(time, event, risk, stratum):
    from analysis.patient_error import symmetric_concordance_error
    s = np.asarray(stratum)
    err, n = np.full(len(s), np.nan), np.zeros(len(s))
    for g in np.unique(s):
        m = s == g
        err[m], n[m] = symmetric_concordance_error(np.asarray(time)[m], np.asarray(event)[m], np.asarray(risk)[m])
    return err, n
