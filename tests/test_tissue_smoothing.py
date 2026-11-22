"""Neighbour smoothing of tile probabilities (analysis/tissue/smooth_test.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.smooth_test import neighbour_mean  # noqa: E402


def test_neighbour_mean_uses_present_neighbours_in_same_image():
    image = np.array(["a", "a", "a", "b"])
    gx = np.array([0, 1, 3, 1])
    gy = np.array([0, 0, 0, 0])
    P = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    S = neighbour_mean(image, gx, gy, P, radius=1)
    assert np.allclose(S[0], [0.5, 0.5])   # itself and (1, 0)
    assert np.allclose(S[1], [0.5, 0.5])   # itself and (0, 0); (3, 0) is two tiles away
    assert np.allclose(S[2], [0.5, 0.5])   # alone
    assert np.allclose(S[3], [0.2, 0.8])   # image b is separate
    assert np.allclose(neighbour_mean(image, gx, gy, P, radius=0), P)
