import numpy as np
import pytest

from analysis.pancan.metrics import cindex_within, patient_error_within


def test_cindex_within_matches_the_papers_cindex_on_one_stratum():
    pytest.importorskip("torch")  # region_ablation imports torch
    from analysis.region_ablation import fast_cindex
    rng = np.random.default_rng(0)
    t = rng.integers(1, 30, 200).astype(float)  # ties in time on purpose
    e = (rng.random(200) < 0.4).astype(int)
    r = np.round(rng.normal(size=200), 1)       # ties in risk on purpose
    assert cindex_within(t, e, r, np.zeros(200)) == pytest.approx(fast_cindex(t, e, r))


def test_cindex_within_ignores_pairs_across_types():
    # type A dies early and gets high risks: every cross-type pair is concordant (pooled C = 5/6),
    # but within A the order is wrong and within B it is right (within-type C = 1/2)
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 1, 1])
    r = np.array([10.0, 11.0, 1.0, 0.0])
    s = np.array(["A", "A", "B", "B"])
    assert cindex_within(t, e, r, s) == pytest.approx(0.5)
    assert cindex_within(t, e, r, np.zeros(4)) == pytest.approx(5 / 6)


def test_cindex_within_skips_a_type_without_comparable_pairs():
    t = np.array([1.0, 2.0, 5.0, 6.0])
    e = np.array([1, 1, 0, 0])
    r = np.array([2.0, 1.0, 0.0, 9.0])
    assert cindex_within(t, e, r, np.array(["A", "A", "B", "B"])) == pytest.approx(1.0)


def test_patient_error_within_counts_only_same_type_pairs():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 1, 1])
    r = np.array([10.0, 11.0, 1.0, 0.0])
    err, n = patient_error_within(t, e, r, np.array(["A", "A", "B", "B"]))
    assert n.tolist() == [1, 1, 1, 1]
    assert err.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_cox_loss_stratified_is_the_event_weighted_sum_of_per_type_losses():
    torch = pytest.importorskip("torch")
    from analysis.pancan.metrics import cox_loss_stratified
    from analysis.region_ablation import cox_ph_loss
    g = torch.Generator().manual_seed(0)
    risk = torch.randn(12, generator=g)
    time = torch.rand(12, generator=g) * 100
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    strat = np.array(["A"] * 6 + ["B"] * 6)  # B has no events and must contribute nothing
    a = slice(0, 6)
    want = cox_ph_loss(risk[a], time[a], event[a]) * event[a].sum() / event.sum()
    assert cox_loss_stratified(risk, time, event, strat).item() == pytest.approx(want.item(), rel=1e-6)
