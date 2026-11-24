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


def test_neighbour_context_counts_the_tissues_around_each_tile():
    from analysis.pancan.bags import neighbour_context
    T, I, O = CLASSES.index("tumour"), CLASSES.index("immune"), CLASSES.index("other")
    coords = np.array([(x * 256, 0) for x in range(5)])
    label = np.array([T, I, T, O, T])
    ctx = neighbour_context(coords, label, np.array([0, 2]), radius=1)
    assert ctx.shape == (2, len(CLASSES))
    assert ctx[0, I] == 1.0                                 # tile 0's only neighbour is immune
    assert ctx[1, I] == 0.5 and ctx[1, O] == 0.5            # tile 2 sits between immune and other
    assert np.allclose(ctx.sum(1), 1.0)


def test_knn_edges_are_symmetric_without_self_loops():
    from analysis.pancan.bags import knn_edges
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 10000, (40, 2))
    e = knn_edges(coords, k=5)
    pairs = set(map(tuple, e.T.tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert all(i != j for i, j in pairs)
    deg = np.bincount(e[0], minlength=40)
    assert (deg >= 5).all()
    assert knn_edges(coords[:1], k=5).shape == (2, 0)
