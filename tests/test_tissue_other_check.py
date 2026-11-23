import numpy as np

from analysis.tissue.other_check import edge_mask


def test_edge_mask_only_centre_of_3x3_block_is_interior():
    coords = np.array([(x * 256, y * 256) for x in range(3) for y in range(3)], dtype=np.int32)
    edge = edge_mask(coords)
    centre = (coords[:, 0] == 256) & (coords[:, 1] == 256)
    assert not edge[centre].any()
    assert edge[~centre].all()


def test_edge_mask_lone_tile_is_edge():
    assert edge_mask(np.array([[1024, 2048]], dtype=np.int32)).tolist() == [True]
