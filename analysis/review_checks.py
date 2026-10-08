#!/usr/bin/env python3
"""Checks requested in review, all on committed or local CPU data.

1. Weight-Shapley agreement: exact permutation p-values for the Spearman
   correlations over 8 or 9 compartments, intervals from resampling rounds, and
   a test of the BLCA-BRCA difference.
2. The risk score against clinical variables: hazard ratio of the out-of-bag
   risk score adjusted for clinical variables, likelihood-ratio test, and the
   share of risk-score variance the clinical variables explain.
3. Source site with case mix: variance in the risk score explained by site after
   stage, T, N (and ER, PR in BRCA), and a likelihood-ratio test for site in a
   Cox model of the outcome adjusted for the same variables.
4. The high-error label against follow-up, event status, comparable pairs and
   site, and a check of whether pathomic features still predict it after
   removing between-site differences.
5. Aggregate enrichment of pathomic associations with PFI under a permutation
   null that keeps the correlation between features (outcomes shuffled jointly).

Outputs results/review_checks.json.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results")
FEAT = os.path.expanduser("~/wsi-transfer/features")
CLIN = {"blca": "clinical_features.csv", "brca": "clinical_features_BRCA.csv"}
CASEMIX = {"blca": ["pathologic_stage", "pathologic_T", "pathologic_N"],
           "brca": ["pathologic_stage", "pathologic_t", "pathologic_n",
                    "breast_carcinoma_estrogen_receptor_status",
                    "breast_carcinoma_progesterone_receptor_status"]}
rng = np.random.default_rng(20261007)


# ── 1. agreement over compartments ───────────────────────────────────────────

def exact_spearman_p(a, b):
    a, b = stats.rankdata(a), stats.rankdata(b)
    obs = abs(stats.spearmanr(a, b).correlation)
    n, hits, tot = len(a), 0, 0
    for perm in itertools.permutations(range(n)):
        r = abs(stats.spearmanr(a, b[list(perm)]).correlation)
        hits += r >= obs - 1e-12
        tot += 1
    return hits / tot


def round_arrays(d):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "round_*.json")))]
    return np.array([r["fusion_weights"] for r in recs]), np.array([r["phi_perf"] for r in recs])


def agreement(n_boot=5000):
    out = {}
    runs = {"bootstrap": "shapley_{c}", "graph_diffusion": "shapley_graph_{c}",
            "held_out_sites": "shapley_sites_{c}"}
    draws = {}
    for run, pat in runs.items():
        for c in ("blca", "brca"):
            W, P = round_arrays(os.path.join(RES, pat.format(c=c)))
            rho = stats.spearmanr(W.mean(0), P.mean(0)).correlation
            bs = []
            for _ in range(n_boot):
                i = rng.integers(0, len(W), len(W))
                bs.append(stats.spearmanr(W[i].mean(0), P[i].mean(0)).correlation)
            bs = np.array(bs)
            draws[(run, c)] = bs
            out[f"{run}_{c}"] = {
                "n_rounds": int(len(W)), "rho": float(rho),
                "exact_permutation_p": float(exact_spearman_p(W.mean(0), P.mean(0))),
                "round_resampling_95": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]}
        d = draws[(run, "blca")] - draws[(run, "brca")]
        out[f"{run}_blca_minus_brca"] = {
            "estimate": out[f"{run}_blca"]["rho"] - out[f"{run}_brca"]["rho"],
            "round_resampling_95": [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))],
            "one_sided_p_difference_le_0": float(np.mean(d <= 0))}
    out["note"] = ("Intervals resample rounds, not patients: they describe how stable the "
                   "agreement is across retrained models on these patients.")
    return out


# ── shared loading ───────────────────────────────────────────────────────────

def load(c):
    d = np.load(os.path.join(RES, f"oob_risk_{c}.npz"), allow_pickle=True)
    df = pd.DataFrame({"pid": [str(p) for p in d["patient_ids"]], "t": d["pfi_time"],
                       "e": d["pfi_event"], "x": d["mean_risk"]}).dropna()
    df = df[df.t > 0].reset_index(drop=True)
    X = pd.read_csv(os.path.join(FEAT, CLIN[c]), index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated()].reindex(df.pid)
    X.index = range(len(df))
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.std() > 0]
    df["site"] = df.pid.str[5:7]
    return df, X


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() or 1.0)


def cox_ll(T, E, Z, penalizer):
    data = Z.copy(); data["_t"] = T; data["_e"] = E
    f = CoxPHFitter(penalizer=penalizer).fit(data, "_t", "_e")
    return f, f.log_likelihood_


# ── 2. risk score against clinical variables ─────────────────────────────────

def risk_vs_clinical(c, penalizer=0.1):
    df, X = load(c)
    Zc = X.apply(zs)
    full = Zc.copy(); full["risk"] = zs(df.x)
    f_full, ll1 = cox_ll(df.t.values, df.e.values, full, penalizer)
    _, ll0 = cox_ll(df.t.values, df.e.values, Zc, penalizer)
    lr = 2 * (ll1 - ll0)
    s = f_full.summary.loc["risk"]
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    r2_in = LinearRegression().fit(Zc, df.x).score(Zc, df.x)
    r2_cv = cross_val_score(LinearRegression(), Zc, df.x, cv=5, scoring="r2").mean()
    return {"n": int(len(df)), "events": int(df.e.sum()), "n_clinical": int(Zc.shape[1]),
            "penalizer": penalizer,
            "hr_per_sd_adjusted": float(np.exp(s["coef"])),
            "hr_95": [float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))],
            "lr_chi2": float(lr), "lr_p": float(stats.chi2.sf(lr, 1)),
            "risk_r2_from_clinical_in_sample": float(r2_in),
            "risk_r2_from_clinical_5fold": float(r2_cv),
            "note": ("Likelihood-ratio test between penalised fits with the same penaliser; "
                     "approximate under the ridge penalty.")}


# ── 3. source site with case mix ─────────────────────────────────────────────

def site_casemix(c, min_n=10, penalizer=0.01):
    df, X = load(c)
    counts = df.site.value_counts()
    site = df.site.where(df.site.isin(counts[counts >= min_n].index), "other")
    D = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
    C = X[CASEMIX[c]].apply(zs)
    y = zs(df.x)
    def ols(M):
        A = np.column_stack([np.ones(len(y)), M])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(((y - A @ beta) ** 2).sum())
        return rss, A.shape[1]
    rss0, k0 = ols(C.values)
    rss1, k1 = ols(np.column_stack([C.values, D.values]))
    tss = float(((y - y.mean()) ** 2).sum())
    F = ((rss0 - rss1) / (k1 - k0)) / (rss1 / (len(y) - k1))
    _, ll0 = cox_ll(df.t.values, df.e.values, C.reset_index(drop=True), penalizer)
    _, ll1 = cox_ll(df.t.values, df.e.values, pd.concat([C, D], axis=1).reset_index(drop=True), penalizer)
    lr = 2 * (ll1 - ll0)
    return {"n_sites_modelled": int(D.shape[1] + 1), "casemix": CASEMIX[c],
            "risk_r2_casemix": 1 - rss0 / tss, "risk_r2_casemix_plus_site": 1 - rss1 / tss,
            "risk_partial_r2_site": (rss0 - rss1) / rss0,
            "risk_site_F": float(F), "risk_site_p": float(stats.f.sf(F, k1 - k0, len(y) - k1)),
            "outcome_site_lr_chi2": float(lr), "outcome_site_df": int(D.shape[1]),
            "outcome_site_p": float(stats.chi2.sf(lr, D.shape[1]))}


# ── 4. what the high-error label tracks ──────────────────────────────────────

def error_label(c, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict, train_test_split
    pe = pd.read_csv(os.path.join(RES, f"patient_error_{c}_oob_per_patient.csv"))
    pe["site"] = pe.patient_id.str[5:7]
    lab = (pe.error > pe.error.median()).astype(int).values
    out = {"n": int(len(pe)), "label": "error above the median"}
    for col in ("pfi_time", "pfi_event", "n_comparable_pairs"):
        auc = roc_auc_score(lab, pe[col].values)
        out[f"auroc_{col}"] = float(max(auc, 1 - auc))
    out["spearman_error_followup_censored"] = float(stats.spearmanr(
        pe.error[pe.pfi_event == 0], pe.pfi_time[pe.pfi_event == 0]).correlation)
    out["spearman_error_followup_events"] = float(stats.spearmanr(
        pe.error[pe.pfi_event == 1], pe.pfi_time[pe.pfi_event == 1]).correlation)
    S = pd.get_dummies(pe.site, dtype=float)
    p = cross_val_predict(LogisticRegression(max_iter=2000), S, lab, cv=5, method="predict_proba")[:, 1]
    out["auroc_site_5fold"] = float(roc_auc_score(lab, p))
    F = pe[["pfi_time", "pfi_event", "n_comparable_pairs"]]
    p = cross_val_predict(RandomForestClassifier(500, random_state=seed), F, lab, cv=5,
                          method="predict_proba")[:, 1]
    out["auroc_followup_event_pairs_5fold"] = float(roc_auc_score(lab, p))

    # pathomic features, raw against within-site centred, on a 60/40 split
    R = pd.read_csv(os.path.join(FEAT, c, "radiomics_pre_corr.csv"), index_col=0)
    R.index = [str(i)[:12] for i in R.index]
    R = R[~R.index.duplicated()].reindex(pe.patient_id).apply(pd.to_numeric, errors="coerce")
    R = R.loc[:, R.notna().mean() > 0.9]
    R = R.fillna(R.median()).loc[:, lambda d: d.std() > 1e-9]
    R.index = range(len(pe))
    tr, te = train_test_split(np.arange(len(pe)), test_size=0.4, stratify=lab, random_state=seed)
    def fit_eval(M):
        uni = [abs(roc_auc_score(lab[tr], M.iloc[tr][col]) - 0.5) for col in M.columns]
        top = M.columns[np.argsort(uni)[::-1][:30]]
        rf = RandomForestClassifier(500, random_state=seed, n_jobs=-1).fit(M.iloc[tr][top], lab[tr])
        return float(roc_auc_score(lab[te], rf.predict_proba(M.iloc[te][top])[:, 1]))
    Rc = R - R.groupby(pe.site.values).transform("mean")
    out["pathomic_auroc_raw"] = fit_eval(R)
    out["pathomic_auroc_within_site_centred"] = fit_eval(Rc)
    out["pathomic_note"] = ("Simplified check, not a MED3pa rerun: random forest on the 30 "
                            "features most associated with the label in a 60% split, AUROC "
                            "on the remaining 40%, with features raw and centred within site.")
    return out


# ── 5. aggregate enrichment under a correlation-preserving null ──────────────

def score_test_p(Xs, t, e):
    """Univariate Cox score-test p-values for every column of standardised Xs."""
    order = np.argsort(t, kind="stable")
    t, e, X = t[order], e[order], Xs[order]
    n = len(t)
    S1 = np.cumsum(X[::-1], 0)[::-1]
    S2 = np.cumsum((X ** 2)[::-1], 0)[::-1]
    N = np.arange(n, 0, -1)[:, None]
    first = np.searchsorted(t, t, side="left")
    ev = np.where(e == 1)[0]
    m1 = S1[first[ev]] / N[first[ev]]
    m2 = S2[first[ev]] / N[first[ev]]
    U = (X[ev] - m1).sum(0)
    I = (m2 - m1 ** 2).sum(0)
    chi = U ** 2 / np.maximum(I, 1e-12)
    return stats.chi2.sf(chi, 1)


def enrichment(c, n_perm=2000):
    df, _ = load(c)
    R = pd.read_csv(os.path.join(FEAT, c, "radiomics_pre_corr.csv"), index_col=0)
    R.index = [str(i)[:12] for i in R.index]
    R = R[~R.index.duplicated()]
    keep = [p for p in df.pid if p in R.index]
    S = df.set_index("pid").loc[keep]
    R = R.loc[keep].apply(pd.to_numeric, errors="coerce")
    R = R.loc[:, R.notna().mean() > 0.9]
    R = R.fillna(R.median()).loc[:, lambda d: d.std(ddof=0) > 1e-9]
    modal = R.apply(lambda col: (col == col.mode().iloc[0]).mean())
    R = R.loc[:, ~((modal > 0.5) | (R.nunique() < 20))]
    Xs = ((R - R.mean()) / R.std(ddof=0)).values
    t, e = S.t.values.astype(float), S.e.values.astype(int)
    p = score_test_p(Xs, t, e)
    obs = {a: int((p < a).sum()) for a in (0.05, 0.01, 0.001)}
    null = {a: [] for a in obs}
    for _ in range(n_perm):
        i = rng.permutation(len(t))
        pp = score_test_p(Xs, t[i], e[i])
        for a in obs:
            null[a].append(int((pp < a).sum()))
    res = {"n_patients": len(keep), "n_features": int(Xs.shape[1]), "n_perm": n_perm,
           "test": "univariate Cox score test; outcomes permuted jointly across patients"}
    for a in obs:
        nl = np.array(null[a])
        res[f"alpha_{a}"] = {"observed": obs[a], "expected_independent": a * Xs.shape[1],
                             "null_mean": float(nl.mean()), "null_95th": float(np.quantile(nl, .95)),
                             "null_99th": float(np.quantile(nl, .99)),
                             "perm_p": float((1 + (nl >= obs[a]).sum()) / (1 + n_perm))}
    return res


def main():
    out = {"agreement": agreement()}
    for c in ("blca", "brca"):
        print(f"[{c}] clinical / site / error / enrichment ...", file=sys.stderr)
        out[c] = {"risk_vs_clinical": risk_vs_clinical(c), "site_casemix": site_casemix(c),
                  "error_label": error_label(c), "enrichment": enrichment(c)}
    path = os.path.join(RES, "review_checks.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
