"""The saved tissue classifier file gives (n, 9) probabilities; merged-away classes stay zero."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.classes import CLASSES  # noqa: E402
from analysis.tissue.train_eval import load_classifier  # noqa: E402


def test_load_classifier_probabilities(tmp_path):
    rng = np.random.default_rng(0)
    classes2 = ["normal_epithelium", "stroma", "immune", "necrosis", "fat", "other"]  # vessels, muscle merged
    path = tmp_path / "w.npz"
    np.savez(path, coef1=rng.normal(size=(1, 512)), b1=np.array([0.1]), T1=1.5,
             coef2=rng.normal(size=(6, 512)), b2=np.zeros(6), T2=2.0, classes2=np.array(classes2),
             classes=np.array(CLASSES), thresholds=np.full(9, np.nan),
             merged=json.dumps({"vessels": "stroma", "muscle": "stroma"}), feature="conch")
    X = rng.normal(size=(20, 512))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    P = load_classifier(path)(X)
    assert P.shape == (20, 9)
    assert np.allclose(P.sum(1), 1, atol=1e-6)
    assert np.all(P[:, CLASSES.index("vessels")] == 0) and np.all(P[:, CLASSES.index("muscle")] == 0)
    z1 = (X @ np.load(path)["coef1"].T + 0.1).ravel() / 1.5
    assert np.allclose(P[:, 0], 1 / (1 + np.exp(-z1)))
