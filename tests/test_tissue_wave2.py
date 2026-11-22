"""Second-wave tissue data: label maps and split extension (analysis/tissue)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.classes import BRUMMER_MAP, CLASSES, TIGER_MAP, brummer_group  # noqa: E402
from analysis.tissue.splits import extend_splits  # noqa: E402


def test_tiger_map():
    # 1 invasive tumour, 2 tumour-associated stroma, 3 in-situ tumour, 4 healthy glands,
    # 5 necrosis, 6 inflamed stroma (dropped: stroma or immune), 7 rest (dropped: mixed)
    assert TIGER_MAP == {0: None, 1: "tumour", 2: "stroma", 3: "tumour", 4: "normal_epithelium",
                         5: "necrosis", 6: None, 7: None}
    assert set(v for v in TIGER_MAP.values() if v) <= set(CLASSES)


def test_extend_splits_keeps_existing_sites():
    existing = {"A1": "train", "A2": "test"}
    groups = ["A1", "A2", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    out = extend_splits(existing, groups, ["BRCA"] * len(groups))
    assert out["A1"] == "train" and out["A2"] == "test"      # a known site keeps its split
    assert set(out) == set(groups)
    assert {out[g] for g in groups[2:]} <= {"train", "valid", "test"}
    assert extend_splits(existing, groups, ["BRCA"] * len(groups)) == out  # deterministic


def test_brummer_map_and_groups():
    # folders of Brummer et al.'s kidney tiles; 'empty' (background) and 'other' (mixed, incl. fat) dropped
    assert BRUMMER_MAP == {"cancer": "tumour", "normal": "normal_epithelium", "stroma": "stroma",
                           "blood": "necrosis", "empty": None, "other": None}
    assert brummer_group("TCGA-B0-5402_89123_4714_empty_7773.png") == ("B0", "TCGA-B0-5402")
    assert brummer_group("FIMM6_62_ML1412181.mrxs_TumorTile 80.png") == (
        "brummer:FIMM6_62_ML1412181", "FIMM6_62_ML1412181")
