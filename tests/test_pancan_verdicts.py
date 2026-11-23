import numpy as np
import pandas as pd
import pytest

from analysis.pancan.curve import curve_subsets
from analysis.pancan.verdicts import (curve_pass, e_check, faithful, perm_p_spearman, r_differs,
                                      r_predictable)


def test_exact_permutation_p_of_a_perfect_ranking_is_one_over_r_factorial():
    assert perm_p_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1 / 24)
    assert perm_p_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1 / 120)


def test_faithful_needs_positive_rho_and_p_below_005():
    assert faithful([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])["pass"]          # p = 1/24
    assert not faithful([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.4, 0.3])["pass"]      # one swap: p = 4/24
    assert faithful([1, 2, 3, 4, 5], [1, 2, 3, 5, 4])["pass"]                     # rho 0.9, p = 5/120
    assert not faithful([4, 3, 2, 1], [1, 2, 3, 4])["pass"]


def test_e_r_and_curve_rules():
    assert e_check([0.2, 0.049]) and not e_check([0.05, 0.3])
    assert r_predictable(0.6, 0.001, n_tests=16) and not r_predictable(0.6, 0.004, n_tests=16)
    assert not r_predictable(0.5, 0.0001)
    prof = [{"run_share": 0.04, "ci": (0.7, 0.8)}, {"run_share": 0.10, "ci": (0.55, 0.64)}]
    assert r_differs(prof, cohort_c=0.65)
    assert not r_differs([{"run_share": 0.10, "ci": (0.60, 0.70)}], cohort_c=0.65)
    assert curve_pass([True, False, True]) and not curve_pass([True, False, False])


def test_curve_subsets_hit_the_event_targets_within_each_type():
    rng = np.random.default_rng(0)
    c = pd.DataFrame({"patient": [f"P{i}" for i in range(4000)],
                      "type": rng.choice(["A", "B", "C"], 4000, p=[0.5, 0.3, 0.2]),
                      "pfi_event": (rng.random(4000) < 0.3).astype(int)})
    subs = curve_subsets(c, sizes=(150, 600), repeats=3)
    for (size, rep), pts in subs.items():
        s = c[c["patient"].isin(pts)]
        assert abs(s["pfi_event"].sum() - size) <= 3
        share_all = c.groupby("type")["pfi_event"].sum() / c["pfi_event"].sum()
        share_sub = s.groupby("type")["pfi_event"].sum() / s["pfi_event"].sum()
        assert (abs(share_all - share_sub) < 0.02).all()
    assert set(subs[(150, 0)]) != set(subs[(150, 1)])
    assert subs == curve_subsets(c, sizes=(150, 600), repeats=3)
