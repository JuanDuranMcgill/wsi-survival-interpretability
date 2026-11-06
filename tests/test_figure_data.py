"""Every number plotted in the audit figures equals the number printed in the paper."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualization"))
import figure_data as fd  # noqa: E402


@pytest.mark.parametrize("cohort,w_use,use_out", [("blca", 0.95, 0.88), ("brca", 0.73, -0.27)])
def test_interp_rhos(cohort, w_use, use_out):
    s = fd.interp_summary(cohort)
    assert round(s["rho_weight_use"], 2) == w_use
    assert round(s["rho_use_outcome"], 2) == use_out


def test_brca_use_vs_outcome_regions():
    s = fd.interp_summary("brca")
    assert round(s["use_share"]["DCIS"], 1) == 18.2
    assert round(s["use_share"]["Necrosis"], 1) == 12.6
    assert round(s["shapley_pp"]["Adipose"], 2) == 2.44
    assert round(s["shapley_pp"]["Muscle"], 2) == 2.20


def test_planted_hit_rates():
    h = fd.planted_hit_rates()
    expected = {"Shapley": (51, 2.3), "Deletion": (41, 2.5), "Weights": (36, 3.1),
                "Integrated gradients": (32, 3.3), "Gradient x input": (18, 4.2),
                "ABMIL attention": (0, 5.9)}
    for k, (pct, rank) in expected.items():
        # The paper prints percentages with round() (32.5 -> 32) and mean ranks half-up (3.25 -> 3.3);
        # fd.pct_label / fd.rank_label reproduce that, so figure labels match the text.
        assert fd.pct_label(h[k][0]) == f"{pct}%", k
        assert fd.rank_label(h[k][1]) == f"{rank:.1f}", k


def test_coverage_intervals_match_paper():
    c = fd.coverage_by_stage("brca")
    late = c[c.stage == "III-IV"].sort_values("horizon_days").iloc[-2:]
    assert [(round(a, 2), round(b, 2)) for a, b in zip(late.lo, late.hi)] == [(0.69, 0.88), (0.70, 0.88)]


@pytest.mark.parametrize("cohort,n,nom,minq,feat_rho", [("blca", 851, 120, 0.064, 0.28),
                                                       ("brca", 875, 132, 0.149, 0.18)])
def test_explain_summary(cohort, n, nom, minq, feat_rho):
    e = fd.explain_summary(cohort)
    assert e["n_tested"] == n
    assert e["n_nominal"] == nom
    assert e["n_fdr"] == 0
    assert round(e["min_q"], 3) == minq
    assert round(e["region_rho"], 2) == 0.76
    assert round(e["feature_rho"], 2) == feat_rho


def test_blca_adipose_share_moves():
    e = fd.explain_summary("blca")
    assert round(e["adipose_share_impurity"]) == 17
    assert round(e["adipose_share_perm"]) == 61


def test_profiles():
    p = fd.profiles()
    assert len(p) == 159
    d = p[p["differs"]]
    assert len(d) == 3
    assert sorted(round(x, 2) for x in d["cindex"]) == [0.33, 0.37, 0.72]


def test_auroc():
    a = fd.auroc_by_arm()
    row = a[(a.cohort == "brca") & (a.arm == "pathomic") & (a.thr == "q50")]
    assert round(float(row.auroc.iloc[0]), 3) == 0.671


def test_coverage_stage_brca():
    c = fd.coverage_by_stage("brca")
    late = c[c.stage == "III-IV"].sort_values("horizon_days")
    assert [round(x, 2) for x in late.coverage.iloc[-2:]] in ([0.79, 0.78], [0.78, 0.79])
