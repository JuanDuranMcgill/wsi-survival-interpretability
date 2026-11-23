import numpy as np
import pytest

from analysis.pancan.info_kinds import family_shapley, region_dispersion, spatial_features
from analysis.tissue.classes import CLASSES

T, S, I = CLASSES.index("tumour"), CLASSES.index("stroma"), CLASSES.index("immune")


def _grid(rows):
    """rows of class letters -> (coords, labels) on a 256 px grid."""
    code = {"T": T, "S": S, "I": I}
    coords, labels = [], []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                coords.append((x * 256, y * 256))
                labels.append(code[ch])
    return np.array(coords), np.array(labels)


def test_spatial_features_count_neighbours_and_tumour_pieces():
    coords, labels = _grid(["TT.I",
                            "TT.I",
                            "...."])
    f = spatial_features(coords, labels)
    assert f["tumour_to_tumour"] == pytest.approx(1.0)      # each tumour tile's neighbours are all tumour
    assert f["tumour_to_immune"] == 0.0
    assert f["tumour_pieces_per_100"] == pytest.approx(100 / 4)  # one piece of 4 tiles
    assert f["largest_tumour_piece_share"] == 1.0
    coords, labels = _grid(["TIT"])
    f = spatial_features(coords, labels)
    assert f["tumour_to_immune"] == 1.0 and f["immune_to_tumour"] == 1.0
    assert f["tumour_pieces_per_100"] == pytest.approx(100.0)   # two pieces of one tile


def test_region_dispersion_is_zero_when_all_regions_look_alike():
    coords = np.array([(x * 256, y * 256) for x in range(32) for y in range(32)])
    same = np.ones((len(coords), 4))
    assert region_dispersion(coords, same, tiles_per_region=16) == pytest.approx(0.0)
    split = np.where((coords[:, 0] < 16 * 256)[:, None], 0.0, 2.0) * np.ones((1, 4))
    assert region_dispersion(coords, split, tiles_per_region=16) > 0


def test_family_shapley_recovers_additive_values():
    names = ["a", "b", "c"]
    gain = {"a": 0.05, "b": 0.02, "c": 0.0}
    v = {frozenset(s): 0.5 + sum(gain[n] for n in s)
         for s in [(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]}
    phi = family_shapley(v, names)
    assert phi == pytest.approx(gain)
