import numpy as np
import pytest

from analysis.pancan.planted import planted_z_within, ranks_of


def test_planted_z_is_standardised_within_type_and_zero_when_absent():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(60, 8)) + np.repeat([[5.0], [-5.0]], 30, axis=0)  # types differ a lot
    means[[3, 40]] = np.nan                                                     # compartment absent
    types = np.array(["A"] * 30 + ["B"] * 30)
    z = planted_z_within(means, types, seed=1)
    assert z[3] == 0 and z[40] == 0
    for t in "AB":
        m = (types == t) & ~np.isnan(means).any(1)
        assert z[m].mean() == pytest.approx(0, abs=1e-9) and z[m].std() == pytest.approx(1, abs=1e-9)


def test_ranks_of_puts_the_planted_compartment_first_when_it_leads():
    per_region = {"tumour": {"fusion_weight": {"mean": 0.2}, "shapley_cindex": {"mean": 0.01},
                             "local_attribution_share": {"mean": 0.1}},
                  "stroma": {"fusion_weight": {"mean": 0.5}, "shapley_cindex": {"mean": 0.08},
                             "local_attribution_share": {"mean": 0.7}},
                  "fat": {"fusion_weight": {"mean": 0.3}, "shapley_cindex": {"mean": -0.01},
                          "local_attribution_share": {"mean": 0.2}}}
    r = ranks_of(per_region, "stroma")
    assert r == {"fusion_weight": 1, "shapley_cindex": 1, "local_attribution_share": 1}
    assert ranks_of(per_region, "fat")["fusion_weight"] == 2
