import json

import numpy as np

from analysis.pancan.bags import compartments, make_bag
from analysis.tissue.classes import CLASSES


def _weights(tmp_path, classes2):
    p = tmp_path / "w.npz"
    np.savez(p, classes2=np.array(classes2), classes=np.array(CLASSES), merged=json.dumps({}))
    return p


def test_compartments_follow_the_label_set_and_drop_other(tmp_path):
    main = compartments(_weights(tmp_path, ["fat", "immune", "normal_epithelium", "other"]))
    assert main == ["tumour", "normal_epithelium", "immune", "fat"]
    stroma = compartments(_weights(tmp_path, ["fat", "immune", "normal_epithelium", "other", "stroma"]))
    assert stroma == ["tumour", "normal_epithelium", "stroma", "immune", "fat"]


def test_make_bag_groups_tiles_by_label_caps_them_and_drops_other():
    n = 50
    label = np.array([CLASSES.index("tumour")] * 30 + [CLASSES.index("fat")] * 5 + [CLASSES.index("other")] * 15)
    feats = np.arange(n, dtype=np.float32)[:, None].repeat(4, 1)
    coords = np.stack([np.arange(n), np.arange(n)], 1)
    bag = make_bag(feats, coords, label, ["tumour", "immune", "fat"], cap=10, seed=1)
    assert list(bag) == ["tumour", "immune", "fat"]
    tf, tc = bag["tumour"]
    assert len(tf) == 10 and set(tf[:, 0].astype(int)) <= set(range(30))
    assert (tf[:, 0] == tc[:, 0]).all()  # features and coordinates stay paired
    assert len(bag["immune"][0]) == 0
    assert sorted(bag["fat"][0][:, 0].astype(int).tolist()) == list(range(30, 35))


def test_make_bag_is_reproducible():
    label = np.zeros(100, dtype=int)
    feats = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    coords = np.zeros((100, 2))
    a = make_bag(feats, coords, label, ["tumour"], cap=7, seed=3)["tumour"][0]
    b = make_bag(feats, coords, label, ["tumour"], cap=7, seed=3)["tumour"][0]
    assert np.array_equal(a, b)
