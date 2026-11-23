"""Share of run tiles kept by each class's confidence cut (analysis/tissue/label_run.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.label_run import kept_counts  # noqa: E402


def test_kept_counts_per_class():
    label = np.array([0, 0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.95, 0.05], [0.2, 0.8], [0.4, 0.6]])
    thr = np.array([0.9, np.nan])  # class 1 has no cut: all its tiles are kept
    n_lab, n_kept = kept_counts(label, probs, thr)
    assert n_lab.tolist() == [3, 2] and n_kept.tolist() == [2, 2]
