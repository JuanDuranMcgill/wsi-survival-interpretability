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


def test_saved_classifier_loads_with_object_class_names(tmp_path):
    # sklearn keeps string labels as an object array in classes_; the saved file must still load
    # with allow_pickle=False (the Beast labelling job failed on exactly this).
    from types import SimpleNamespace
    from analysis.tissue.train_eval import save_classifier
    rng = np.random.default_rng(1)
    m1 = SimpleNamespace(coef_=rng.normal(size=(1, 512)), intercept_=np.array([0.0]))
    m2 = SimpleNamespace(coef_=rng.normal(size=(3, 512)), intercept_=np.zeros(3),
                         classes_=np.array(["immune", "other", "stroma"], dtype=object))
    model = {"m1": m1, "m2": m2, "T1": 1.0, "T2": 1.0, "merged": {"vessels": "stroma"}}
    path = tmp_path / "w.npz"
    save_classifier(path, model, {c: None for c in CLASSES})
    P = load_classifier(path)(rng.normal(size=(5, 512)))
    assert P.shape == (5, 9) and np.allclose(P.sum(1), 1)


def test_load_classifier_with_two_step2_classes(tmp_path):
    # after merges step 2 can be binary (immune vs other); sklearn then stores one coefficient row
    rng = np.random.default_rng(2)
    path = tmp_path / "w.npz"
    coef2 = rng.normal(size=(1, 512))
    np.savez(path, coef1=np.zeros((1, 512)), b1=np.array([0.0]), T1=1.0,  # p_tumour = 0.5
             coef2=coef2, b2=np.array([0.3]), T2=2.0, classes2=np.array(["immune", "other"]),
             classes=np.array(CLASSES), thresholds=np.full(9, np.nan), merged=json.dumps({}), feature="conch")
    X = rng.normal(size=(6, 512))
    P = load_classifier(path)(X)
    assert P.shape == (6, 9) and np.allclose(P.sum(1), 1)
    p_other = 1 / (1 + np.exp(-((X @ coef2.T).ravel() + 0.3) / 2.0))
    rest = P[:, 1:].sum(1)
    assert np.allclose(P[:, CLASSES.index("other")] / rest, p_other)
