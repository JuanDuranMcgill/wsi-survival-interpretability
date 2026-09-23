#!/usr/bin/env python3
"""Conformal lower bounds on survival time, and whether they hold per subgroup.

The reliability leg of the paper asks for whom the model can be trusted, and
MED3pa answers it heuristically: it finds subgroups where a derived error label
is predictable, and the paper shows those subgroups do not differ measurably in
concordance. Conformal prediction answers a sharper version of the question.
It wraps the model and gives every patient a lower bound on time to progression
together with a coverage promise: at least 1 - alpha of patients progress no
earlier than their bound. The promise is marginal, over the cohort. Whether it
also holds inside clinically meaningful subgroups is exactly "for whom is the
model reliable", now with a measurable target.

Method: fixed-cutoff conformalized survival analysis (Candes, Lei and Ren 2023)
in the general right-censoring form of Sesia and Svetnik (2025), which is the
form TCGA needs: the censoring time of a patient who progressed is not recorded.
It is imputed from an estimated censoring distribution, given that it exceeds
the observed event time. Hong (2026) applied the same construction to UNI2
pathology survival models on five other TCGA cohorts and CPTAC, and found
marginal coverage near target but not within risk tertiles; BLCA and BRCA were
not among them.

The target is T wedge c0, survival truncated at a horizon c0, because on the
patients whose censoring time is at least c0 that quantity is fully observed.
A lower bound on T wedge c0 below c0 is also a lower bound on T.

Procedure, repeated over random train / calibration / test splits:

1. On the training part, fit a Cox model on the out-of-bag risk score and take
   each patient's conditional alpha-quantile of survival time, capped at c0.
2. On the calibration part, keep the patients whose censoring time is at least
   c0, observed or imputed, and score each as predicted quantile minus observed
   T wedge c0. Weighted split conformal on those scores, with weights 1 / P(C >=
   c0 | x) from the censoring model, gives the correction eta.
3. On the test part, each patient's bound is their predicted quantile minus
   eta, clipped to [0, c0].

Coverage on real data is estimated by inverse probability of censoring
weighting, since the true survival time of a censored patient is unknown:
the mean over test patients of 1{observed time >= bound} / G(bound-). The same
estimator gives coverage inside each subgroup.

Three comparators run on the same test patients: the model's quantile with no
conformal correction, which shows why calibration is needed; a single bound for
everyone from the Kaplan-Meier curve, which shows what the model adds; and
conformal applied to the censored time directly, the simple valid but
conservative alternative.

Two censoring models: marginal Kaplan-Meier, which assumes censoring does not
depend on the patient, and a Cox model of censoring on the risk score. The
Sesia-Svetnik double robustness means coverage is approximately right if either
the censoring model or the survival model is right.

Usage (local; needs the out-of-bag risk vector and the clinical table):

    python analysis/conformal_survival.py --cohort blca \\
        --oob-npz results/oob_risk_blca.npz \\
        --clinical-csv ~/wsi-transfer/features/clinical_features.csv \\
        --out results/conformal_survival_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")


# ── Censoring distribution ───────────────────────────────────────────────────

class KMCensoring:
    """Kaplan-Meier estimate of G(t) = P(C > t), censoring as the event.

    At tied times, progression is taken to occur before censoring, the usual
    convention, so a patient progressing at t is not at risk of censoring at t.
    """

    def fit(self, t, e):
        t, e = np.asarray(t, float), np.asarray(e, int)
        self.times = np.unique(t)
        g, vals = 1.0, []
        for u in self.times:
            n_risk = np.sum(t >= u) - np.sum((t == u) & (e == 1))
            d_c = np.sum((t == u) & (e == 0))
            if n_risk > 0:
                g *= 1.0 - d_c / n_risk
            vals.append(g)
        self.vals = np.array(vals)
        return self

    def surv(self, x, t, left=False):
        """G(t), or G(t-) = P(C >= t) with left=True. x is ignored."""
        t = np.atleast_1d(np.asarray(t, float))
        side = "left" if left else "right"
        pos = np.searchsorted(self.times, t, side=side) - 1
        out = np.ones_like(t)
        ok = pos >= 0
        out[ok] = self.vals[pos[ok]]
        return np.clip(out, 1e-8, 1.0)


class CoxCensoring:
    """Censoring hazard depending on the risk score: G(t | x) = G0(t)^exp(g x)."""

    def fit(self, t, e, x):
        df = pd.DataFrame({"x": x, "t": t, "c": 1 - np.asarray(e, int)})
        self.cph = CoxPHFitter(penalizer=0.01).fit(df, "t", "c")
        bs = self.cph.baseline_survival_
        self.times = bs.index.values.astype(float)
        self.base = bs.values.ravel()
        self.gamma = float(self.cph.params_["x"])
        self.xmean = float(self.cph._norm_mean["x"])
        return self

    def surv(self, x, t, left=False):
        x = np.atleast_1d(np.asarray(x, float))
        t = np.broadcast_to(np.atleast_1d(np.asarray(t, float)), x.shape)
        side = "left" if left else "right"
        pos = np.searchsorted(self.times, t, side=side) - 1
        g0 = np.ones_like(t, dtype=float)
        ok = pos >= 0
        g0[ok] = self.base[pos[ok]]
        return np.clip(g0 ** np.exp(self.gamma * (x - self.xmean)), 1e-8, 1.0)


# ── Survival model: conditional alpha-quantile from a Cox fit ────────────────

def cox_quantiles(t_tr, e_tr, x_tr, x_new, alpha, c0):
    df = pd.DataFrame({"x": x_tr, "t": t_tr, "e": e_tr})
    cph = CoxPHFitter(penalizer=0.01).fit(df, "t", "e")
    bs = cph.baseline_survival_
    times, s0 = bs.index.values.astype(float), bs.values.ravel()
    beta, xm = float(cph.params_["x"]), float(cph._norm_mean["x"])
    S = s0[None, :] ** np.exp(beta * (np.asarray(x_new)[:, None] - xm))
    below = S <= 1.0 - alpha
    q = np.where(below.any(axis=1), times[np.argmax(below, axis=1)], np.inf)
    return np.minimum(q, c0)


# ── Weighted split conformal quantile (Tibshirani et al. 2019) ───────────────

def weighted_quantile_with_test(scores, w_cal, w_test, level):
    """(level)-quantile of sum_i p_i delta_{s_i} + p_test delta_{+inf}.

    Returns one threshold per test point, since the test weight enters the
    normalisation.
    """
    order = np.argsort(scores)
    s, w = scores[order], w_cal[order]
    cw = np.cumsum(w)
    tot = cw[-1] + np.asarray(w_test)                          # per test point
    need = level * tot
    idx = np.searchsorted(cw, need, side="left")
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


# ── One split ────────────────────────────────────────────────────────────────

def one_split(t, e, x, tr, ca, te, alpha, c0s, cens, rng):
    q_ca_all = {c0: cox_quantiles(t[tr], e[tr], x[tr], x[ca], alpha, c0) for c0 in c0s}
    q_te_all = {c0: cox_quantiles(t[tr], e[tr], x[tr], x[te], alpha, c0) for c0 in c0s}
    G = KMCensoring().fit(t[tr], e[tr]) if cens == "km" else CoxCensoring().fit(t[tr], e[tr], x[tr])

    # Kaplan-Meier survival of T on the training part: a single bound for everyone.
    # Standard convention at tied times: events before censoring.
    km_t = np.unique(t[tr][e[tr] == 1])
    km_s, s_ = [], 1.0
    for u in km_t:
        s_ *= 1.0 - np.sum((t[tr] == u) & (e[tr] == 1)) / np.sum(t[tr] >= u)
        km_s.append(s_)
    km_s = np.array(km_s)
    out = {}
    for c0 in c0s:
        q_ca, q_te = q_ca_all[c0], q_te_all[c0]
        tc, ec, xc = t[ca], e[ca], x[ca]
        # Membership of the calibration patient in {C >= c0}: known for censored
        # patients and for patients progressing after c0; imputed otherwise.
        member = np.where(ec == 0, tc >= c0, True).astype(bool)
        need = (ec == 1) & (tc < c0)
        if need.any():
            p_keep = (G.surv(xc[need], np.full(need.sum(), c0), left=True)
                      / G.surv(xc[need], tc[need]))
            member[need] = rng.random(need.sum()) < np.clip(p_keep, 0, 1)
        V = q_ca[member] - np.minimum(tc[member], c0)
        w_cal = 1.0 / G.surv(xc[member], np.full(member.sum(), c0), left=True)
        w_te = 1.0 / G.surv(x[te], np.full(len(te), c0), left=True)
        if len(V) == 0:
            eta = np.full(len(te), np.inf)
        else:
            eta = weighted_quantile_with_test(V, w_cal, w_te, 1.0 - alpha)
        lpb = np.clip(q_te - eta, 0.0, c0)

        # comparators
        lpb_uncal = np.clip(q_te, 0.0, c0)
        hit = np.where(km_s <= 1.0 - alpha)[0]
        flat = km_t[hit[0]] if len(hit) else np.inf
        lpb_flat = np.full(len(te), min(flat, c0))
        V_naive = q_ca - np.minimum(tc, c0)                    # censored time, no correction
        k = int(np.ceil((1 - alpha) * (len(V_naive) + 1)))
        eta_naive = np.sort(V_naive)[k - 1] if k <= len(V_naive) else np.inf
        lpb_naive = np.clip(q_te - eta_naive, 0.0, c0)

        out[c0] = {"lpb": lpb, "lpb_uncal": lpb_uncal, "lpb_flat": lpb_flat,
                   "lpb_naive": lpb_naive, "n_cal_member": int(member.sum())}
    return out


def ipcw_coverage(t, lpb, Geval, x=None):
    """Estimated P(T >= bound): mean of 1{t >= bound} / G(bound- | x).

    With the marginal Kaplan-Meier censoring model this assumes censoring does
    not depend on the patient; a simulation with patient-dependent censoring
    shows it then reads 3 to 4 points low. With the Cox censoring model, x is
    the risk score and that bias is removed when the censoring model holds.
    """
    if len(t) == 0:
        return np.nan
    return float(np.mean((t >= lpb) / Geval.surv(x, lpb, left=True)))


# ── Driver ───────────────────────────────────────────────────────────────────

def run(t, e, x, groups, alpha, c0s, n_reps, cens, seed, fracs=(0.5, 0.25, 0.25), n_boot=0):
    rng = np.random.default_rng(seed)
    n = len(t)
    Geval = KMCensoring().fit(t, e)                            # for evaluation, full data
    Geval_x = CoxCensoring().fit(t, e, x)                      # patient-aware evaluation
    pooled = {c0: {m: [] for m in ("lpb", "lpb_uncal", "lpb_flat", "lpb_naive")} for c0 in c0s}
    pooled_idx, rep_cov, members, tertile_pool = [], {c0: {m: [] for m in pooled[c0]} for c0 in c0s}, [], []
    for rep in range(n_reps):
        perm = rng.permutation(n)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        tr, ca, te = perm[:a], perm[a:b], perm[b:]
        res = one_split(t, e, x, tr, ca, te, alpha, c0s, cens, rng)
        pooled_idx.append(te)
        # risk tertile defined within the training part, then applied to test
        cuts = np.quantile(x[tr], [1 / 3, 2 / 3])
        tertile_pool.append(np.digitize(x[te], cuts))
        for c0 in c0s:
            for m in pooled[c0]:
                pooled[c0][m].append(res[c0][m])
                rep_cov[c0][m].append(ipcw_coverage(t[te], res[c0][m], Geval))
            members.append(res[c0]["n_cal_member"])
    idx = np.concatenate(pooled_idx)
    tert = np.concatenate(tertile_pool)
    groups = dict(groups)
    groups["model risk tertile"] = np.array(["low", "mid", "high"])[tert]

    # Patient-level bootstrap for the Cox-censoring coverage estimates. Patients
    # are resampled with replacement; each pooled test record is weighted by how
    # often its patient was drawn, and the evaluation censoring model is refitted
    # on the resample. The bounds themselves are held fixed, so the interval
    # reflects which patients were observed and the censoring model's fit, not a
    # rerun of the conformal procedure. A separate generator leaves the point
    # estimates unchanged.
    brng = np.random.default_rng(seed + 1)
    boot_w, boot_G = [], []
    for _ in range(n_boot):
        draw = brng.integers(0, n, n)
        cnt = np.bincount(draw, minlength=n).astype(float)
        boot_w.append(cnt[idx])
        boot_G.append(CoxCensoring().fit(t[draw], e[draw], x[draw]))

    def boot_cov(bounds, msk=None):
        """Bootstrap draws of the Cox-censoring coverage on the records in msk."""
        msk = np.ones(len(idx), bool) if msk is None else msk
        ti, xi, bi = t[idx][msk], x[idx][msk], bounds[msk]
        hit = (ti >= bi).astype(float)
        out = np.empty(n_boot)
        for b in range(n_boot):
            w = boot_w[b][msk]
            out[b] = np.sum(w * hit / boot_G[b].surv(xi, bi, left=True)) / np.sum(w)
        return out

    def ci(draws):
        return [float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))]

    summary = {}
    for c0 in c0s:
        L = {m: np.concatenate(pooled[c0][m]) for m in pooled[c0]}
        methods = {}
        for m in L:
            rc = np.array(rep_cov[c0][m])
            methods[m] = {
                "coverage_ipcw_pooled": ipcw_coverage(t[idx], L[m], Geval),
                "coverage_ipcw_pooled_cox_censoring": ipcw_coverage(t[idx], L[m], Geval_x, x[idx]),
                "coverage_ipcw_rep_mean": float(np.nanmean(rc)),
                "coverage_ipcw_rep_2.5_97.5": [float(np.nanquantile(rc, 0.025)),
                                               float(np.nanquantile(rc, 0.975))],
                "median_bound_days": float(np.median(L[m])),
                "frac_bound_positive": float(np.mean(L[m] > 0)),
            }
            if n_boot and m in ("lpb", "lpb_flat"):
                methods[m]["coverage_cox_censoring_boot_95ci"] = ci(boot_cov(L[m]))
        sub = {}
        for gname, labels in groups.items():
            lab = labels if gname == "model risk tertile" else np.asarray(labels)[idx]
            rows, draws = {}, {}
            for level in pd.unique(lab):
                if pd.isna(level):
                    continue
                msk = lab == level
                npat = len(np.unique(idx[msk]))
                if npat < 15:
                    continue
                rows[str(level)] = {
                    "n_patients": int(npat),
                    "coverage_ipcw": ipcw_coverage(t[idx][msk], L["lpb"][msk], Geval),
                    "coverage_ipcw_cox_censoring": ipcw_coverage(
                        t[idx][msk], L["lpb"][msk], Geval_x, x[idx][msk]),
                    "median_bound_days": float(np.median(L["lpb"][msk])),
                }
                if n_boot:
                    draws[str(level)] = boot_cov(L["lpb"], msk)
                    rows[str(level)]["coverage_cox_censoring_boot_95ci"] = ci(draws[str(level)])
                    rows[str(level)]["boot_frac_below_target"] = float(
                        np.mean(draws[str(level)] < 1 - alpha))
            # contrast between the two ends of each grouping, on the same resamples
            pairs = {"stage": ("III-IV", "I-II"), "nodal status": ("N+", "N0"),
                     "T category": ("T2-T4", "T1"), "model risk tertile": ("high", "low")}
            if n_boot and gname in pairs and all(k in draws for k in pairs[gname]):
                hi, lo = pairs[gname]
                dd = draws[hi] - draws[lo]
                rows[f"difference {hi} minus {lo}"] = {
                    "estimate": rows[hi]["coverage_ipcw_cox_censoring"]
                    - rows[lo]["coverage_ipcw_cox_censoring"],
                    "boot_95ci": ci(dd)}
            sub[gname] = rows
        rho = pd.Series(x[idx]).corr(pd.Series(L["lpb"]), method="spearman")
        summary[f"{c0:.0f}"] = {"horizon_c0_days": float(c0), "methods": methods,
                                "subgroups": sub,
                                "spearman_risk_vs_bound": float(rho)}
    return summary


def load_cohort(oob_npz, clinical_csv):
    d = np.load(oob_npz, allow_pickle=True)
    df = pd.DataFrame({"pid": [str(p) for p in d["patient_ids"]],
                       "t": np.asarray(d["pfi_time"], float),
                       "e": np.asarray(d["pfi_event"], float),
                       "x": np.asarray(d["mean_risk"], float)}).dropna()
    df = df[df.t > 0].reset_index(drop=True)
    X = pd.read_csv(clinical_csv, index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated()]
    cols = {c.lower(): c for c in X.columns}
    groups = {}
    st = X.reindex(df.pid)[cols["pathologic_stage"]].values if "pathologic_stage" in cols else None
    if st is not None:
        groups["stage"] = np.where(np.isnan(st), None, np.where(st <= 2, "I-II", "III-IV"))
    nn_ = X.reindex(df.pid)[cols["pathologic_n"]].values if "pathologic_n" in cols else None
    if nn_ is not None:
        groups["nodal status"] = np.where(np.isnan(nn_), None, np.where(nn_ == 0, "N0", "N+"))
    if "pathologic_t" in cols and set(pd.unique(X[cols["pathologic_t"]].dropna())) <= {1, 2, 3, 4}:
        tt = X.reindex(df.pid)[cols["pathologic_t"]].values
        groups["T category"] = np.where(np.isnan(tt), None, np.where(tt <= 1, "T1", "T2-T4"))
    site = df.pid.str[5:7]
    top = site.value_counts()
    groups["source site"] = np.where(site.isin(top[top >= 25].index), site, "other").astype(object)
    return df, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--clinical-csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--horizons", default="q25,q50,q75",
                    help="cutoffs c0, as quantiles of observed time or in days")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--censoring", default="km,cox")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--boot", type=int, default=1000,
                    help="patient-level bootstrap resamples for coverage intervals")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df, groups = load_cohort(args.oob_npz, os.path.expanduser(args.clinical_csv))
    t, e, x = df.t.values, df.e.values.astype(int), df.x.values
    c0s = [float(np.quantile(t, float(h[1:]) / 100)) if h.startswith("q") else float(h)
           for h in args.horizons.split(",")]
    print(f"[{args.cohort}] n={len(t)}, events={e.sum()}, horizons {[round(c) for c in c0s]} days")

    rep = {"cohort": args.cohort, "n_patients": int(len(t)), "n_events": int(e.sum()),
           "alpha": args.alpha, "target_coverage": 1 - args.alpha, "reps": args.reps,
           "splits": "train 50% / calibration 25% / test 25%, repeated",
           "bootstrap": (f"{args.boot} patient-level resamples, Cox-censoring estimates only; "
                         "censoring model refitted per resample, bounds held fixed"),
           "method": ("fixed-cutoff conformalized survival analysis (Candes, Lei & Ren "
                      "2023) with censoring times imputed for patients who progressed "
                      "(Sesia & Svetnik 2025); weighted split conformal"),
           "by_censoring_model": {}}
    for cens in args.censoring.split(","):
        res = run(t, e, x, groups, args.alpha, c0s, args.reps, cens, args.seed,
                  n_boot=args.boot if cens == "cox" else 0)
        rep["by_censoring_model"][cens] = res
        for c0, v in res.items():
            m = v["methods"]
            print(f"[{args.cohort}] censoring={cens:3} c0={c0:>5}d  conformal cov "
                  f"{m['lpb']['coverage_ipcw_pooled']:.3f}/"
                  f"{m['lpb']['coverage_ipcw_pooled_cox_censoring']:.3f} (median bound {m['lpb']['median_bound_days']:.0f}d, "
                  f"{m['lpb']['frac_bound_positive']:.0%} >0) | uncalibrated "
                  f"{m['lpb_uncal']['coverage_ipcw_pooled']:.3f} | flat KM "
                  f"{m['lpb_flat']['coverage_ipcw_pooled']:.3f} ({m['lpb_flat']['median_bound_days']:.0f}d) "
                  f"| naive {m['lpb_naive']['coverage_ipcw_pooled']:.3f} "
                  f"({m['lpb_naive']['median_bound_days']:.0f}d) | rho(risk,bound) {v['spearman_risk_vs_bound']:+.2f}")
    rep["caveat"] = (
        "Coverage on real data is estimated by inverse probability of censoring "
        "weighting, reported under a marginal Kaplan-Meier and a Cox censoring "
        "model; each estimate is only as good as its censoring model, and TCGA "
        "follow-up varies by contributing site. In simulation with "
        "patient-dependent censoring the bounds held at 0.90 while the marginal "
        "estimate read 3 to 4 points low. "
        "The out-of-bag risk scores are a symmetric function of the whole cohort, "
        "which keeps patients exchangeable, but the guarantee is approximate rather "
        "than exact, as for other bootstrap-based conformal constructions.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
