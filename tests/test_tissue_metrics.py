"""Tests for tissue classifier metrics, calibration and the fixed pass bars."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.bars import (class_bars, merges_for, normal_epithelium_bar,  # noqa: E402
                                  close_merges, premerge_small, tumour_organ_bars)
from analysis.tissue.metrics import (combine, ece, fit_temperature, recall_precision,  # noqa: E402
                                     threshold_for_precision)


def test_combine_rows_sum_to_one():
    rng = np.random.default_rng(0)
    pt = rng.random(50)
    pr = rng.random((50, 8))
    pr /= pr.sum(1, keepdims=True)
    out = combine(pt, pr)
    assert out.shape == (50, 9)
    assert np.allclose(out.sum(1), 1, atol=1e-6)
    assert np.allclose(out[:, 0], pt)


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20000, 3))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    assert abs(fit_temperature(3 * z, y) - 3) <= 0.3


def test_ece_zero_when_perfect():
    y = np.array([0, 1, 2, 1])
    assert ece(np.eye(3)[y], y) == 0


def test_recall_precision():
    t = recall_precision(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b", "c"]).set_index("class")
    assert t.loc["a", "recall"] == 0.5 and t.loc["a", "precision"] == 1.0
    assert abs(t.loc["b", "precision"] - 2 / 3) < 1e-9
    assert t.loc["c", "n_true"] == 0 and np.isnan(t.loc["c", "recall"])


def test_threshold_for_precision():
    p = np.array([0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.65] + [0.5] * 5)
    is_k = np.array([1] * 9 + [0] + [0] * 5, dtype=bool)
    assert threshold_for_precision(p, is_k, 0.90) == 0.6
    assert threshold_for_precision(np.array([0.9, 0.8]), np.array([False, False]), 0.9) is None


def _table(rows):
    return pd.DataFrame(rows, columns=["test_set", "class", "n_true", "recall", "precision"])


def test_bar_ignores_sets_without_the_class():
    ok = _table([("kather7k", "vessels", 0, np.nan, 0.0), ("bcss", "vessels", 40, 0.90, 0.85)])
    assert class_bars(ok) == {"vessels": True}
    bad = _table([("kather7k", "stroma", 400, 0.84, 0.95), ("bcss", "stroma", 900, 0.95, 0.95)])
    assert class_bars(bad) == {"stroma": False}
    low_prec = _table([("bcss", "fat", 50, 0.95, 0.79)])
    assert class_bars(low_prec) == {"fat": False}


def test_tumour_organ_bar():
    t = tumour_organ_bars({"LUAD": 0.86, "KIRC": 0.86, "THCA": 0.84},
                          {"LUAD": 0.10, "KIRC": 0.11, "THCA": 0.01})
    assert t.loc["LUAD", "passes"] and not t.loc["KIRC", "passes"] and not t.loc["THCA", "passes"]


def test_normal_bar():
    assert normal_epithelium_bar(True, {"LUAD": 0.049, "KIRC": 0.0})
    assert not normal_epithelium_bar(True, {"LUAD": 0.05, "KIRC": 0.0})
    assert not normal_epithelium_bar(False, {"LUAD": 0.0})


def test_merges_for():
    assert merges_for({"vessels", "fat"}) == {"vessels": "stroma", "fat": "other"}
    assert merges_for(set()) == {}


def test_threshold_balanced_ignores_prevalence():
    # 1,000 positives spread over [0, 1); 10 negatives all at 0.95. Plain precision is >= 0.9 at any
    # threshold (positives dominate); at equal prevalence only thresholds above the negatives qualify.
    p = np.r_[np.linspace(0, 0.999, 1000), np.full(10, 0.95)]
    is_k = np.r_[np.ones(1000, bool), np.zeros(10, bool)]
    assert threshold_for_precision(p, is_k, 0.90) < 0.01
    assert threshold_for_precision(p, is_k, 0.90, balanced=True) > 0.95


def test_premerge_small_classes():
    counts = {"vessels": 5, "stroma": 5000, "fat": 300, "other": 40}
    assert premerge_small(counts, 100) == {"vessels": "stroma"}  # 'other' has no merge target


def test_close_merges_follows_chains():
    # muscle went into stroma, then stroma failed and went into other: muscle must end in other too
    assert close_merges({"muscle": "stroma", "vessels": "stroma", "stroma": "other"}) == {
        "muscle": "other", "vessels": "other", "stroma": "other"}
    assert close_merges({}) == {}
