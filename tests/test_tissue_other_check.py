import numpy as np

from analysis.tissue.other_check import edge_mask


def test_edge_mask_only_centre_of_3x3_block_is_interior():
    coords = np.array([(x * 256, y * 256) for x in range(3) for y in range(3)], dtype=np.int32)
    edge = edge_mask(coords)
    centre = (coords[:, 0] == 256) & (coords[:, 1] == 256)
    assert not edge[centre].any()
    assert edge[~centre].all()


def test_step2_rows_gtex_normal_adds_each_gtex_tile_once_as_normal_tissue():
    import pandas as pd
    from analysis.tissue.train_eval import step2_rows
    idx = pd.DataFrame({"tile_id": ["k1", "k2", "g1", "g1", "t1", "g2"],
                        "source": ["kather100k", "kather100k", "gtex", "gtex", "tcgaut", "gtex"],
                        "label": ["stroma", "muscle", "not_tumour", "not_tumour", "tumour", "not_tumour"],
                        "split": ["train"] * 5 + ["valid"],
                        "type": ["COAD", "COAD", "LUAD", "LUSC", "LUAD", "LIHC"],
                        "tissue": ["", "", "Lung", "Lung", "", "Liver"]})
    _, y = step2_rows(idx, {"muscle": "stroma"}, "train", gtex_normal=True)
    assert sorted(y.tolist()) == ["normal_epithelium", "stroma", "stroma"]
    _, y0 = step2_rows(idx, {"muscle": "stroma"}, "train")
    assert sorted(y0.tolist()) == ["stroma", "stroma"]


def test_edge_mask_lone_tile_is_edge():
    assert edge_mask(np.array([[1024, 2048]], dtype=np.int32)).tolist() == [True]
