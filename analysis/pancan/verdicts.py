"""Pass rules of the IER audit for the pan-carcinoma extension (design decision 13). Frozen before the first curve
point (git tag pancan-rules-frozen); never changed to make a check pass.

I faithful    Spearman rho between fusion weights and risk-score Shapley shares > 0, exact one-sided permutation
              p < 0.05 over the compartments. With 4 compartments only a perfect ranking passes (p = 1/24); with 5,
              rho >= 0.9 (p = 5/120).
I prognostic  Spearman rho between risk-score and C-index Shapley values > 0, patient-bootstrap 95% CI excluding 0.
E             at least one pathomic feature with Benjamini-Hochberg q < 0.05 against PFI (Cox stratified by type).
R predictable error-classifier AUROC > 0.5 with permutation p below the Bonferroni level over 16 tests.
R differs     at least one recurring profile (>= 5% of runs) whose in-profile C-index 95% interval excludes the
              cohort value.
Curve         at a curve size, a check passes if it passes in at least 2 of the 3 repeats.
"""
from itertools import permutations

import numpy as np
from scipy.stats import rankdata


def _rho(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p_spearman(a, b):
    """Exact one-sided p: share of orderings of b whose Spearman rho with a is at least the observed one."""
    obs = _rho(a, b)
    b = np.asarray(b, dtype=float)
    vals = [_rho(a, b[list(p)]) for p in permutations(range(len(b)))]
    return float(np.mean(np.array(vals) >= obs - 1e-12))


def faithful(weights, local_share, alpha=0.05):
    rho, p = _rho(weights, local_share), perm_p_spearman(weights, local_share)
    return {"rho": rho, "p": p, "pass": bool(rho > 0 and p < alpha)}


def prognostic(rho, ci95):
    return bool(rho > 0 and ci95[0] > 0)


def e_check(qvalues, alpha=0.05):
    return bool(np.any(np.asarray(qvalues, dtype=float) < alpha))


def r_predictable(auroc, perm_p, n_tests=16, alpha=0.05):
    return bool(auroc > 0.5 and perm_p < alpha / n_tests)


def r_differs(profiles, cohort_c, min_run_share=0.05):
    return any(p["run_share"] >= min_run_share and (p["ci"][0] > cohort_c or p["ci"][1] < cohort_c)
               for p in profiles)


def curve_pass(repeat_verdicts, need=2):
    return bool(sum(bool(v) for v in repeat_verdicts) >= need)
