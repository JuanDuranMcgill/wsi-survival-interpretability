"""Tests for the tissue classifier helpers (analysis/tissue: classes, tiles, splits)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.tissue.classes import (CLASSES, GTEX_TISSUES, KATHER_MAP, MERGE,  # noqa: E402
                                     BCSS_MAP, tcgaut_type)
from analysis.tissue.splits import assign_groups, tss_of  # noqa: E402
from analysis.tissue.tiles import (center_crop, labelled_tiles, otsu_threshold,  # noqa: E402
                                   tissue_fraction)

COHORT = "BLCA BRCA CESC COAD ESCA HNSC KIRC KIRP LIHC LUAD LUSC OV PAAD PRAD STAD THCA UCEC".split()
TCGAUT_LABELS = [  # Zenodo 5889558 archive names
    "Bladder_Urothelial_Carcinoma", "Breast_invasive_carcinoma",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma", "Colon_adenocarcinoma",
    "Esophageal_carcinoma", "Head_and_Neck_squamous_cell_carcinoma",
    "Kidney_renal_clear_cell_carcinoma", "Kidney_renal_papillary_cell_carcinoma",
    "Liver_hepatocellular_carcinoma", "Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma",
    "Ovarian_serous_cystadenocarcinoma", "Pancreatic_adenocarcinoma", "Prostate_adenocarcinoma",
    "Stomach_adenocarcinoma", "Thyroid_carcinoma", "Uterine_Corpus_Endometrial_Carcinoma",
    "Glioblastoma_multiforme", "Sarcoma", "Thymoma", "Kidney_Chromophobe",
]


def test_class_order():
    assert CLASSES == ["tumour", "normal_epithelium", "stroma", "immune", "necrosis", "fat",
                       "vessels", "muscle", "other"]


def test_kather_map_covers_nine_codes():
    assert set(KATHER_MAP) == {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}
    assert set(KATHER_MAP.values()) <= set(CLASSES)
    assert KATHER_MAP["TUM"] == "tumour" and KATHER_MAP["NORM"] == "normal_epithelium"
    assert KATHER_MAP["BACK"] == "other" and KATHER_MAP["MUC"] == "other"


def test_bcss_map_fixed_codes():
    fixed = {1: "tumour", 2: "stroma", 3: "immune", 4: "necrosis", 6: "necrosis", 9: "fat",
             13: "normal_epithelium", 18: "vessels"}
    for code, cls in fixed.items():
        assert BCSS_MAP[code] == cls
    assert BCSS_MAP[0] is None
    assert all(v is None or v in CLASSES for v in BCSS_MAP.values())


def test_tcgaut_types_cover_cohort():
    mapped = {tcgaut_type(x) for x in TCGAUT_LABELS}
    assert set(COHORT) <= mapped
    assert tcgaut_type("Glioblastoma multiforme") is None
    assert tcgaut_type("Kidney_Chromophobe") is None
    assert tcgaut_type("bladder urothelial carcinoma") == "BLCA"


def test_gtex_tissues_cover_cohort():
    assert set(GTEX_TISSUES) == set(COHORT)
    assert all(len(v) >= 1 for v in GTEX_TISSUES.values())


def test_merge_table_complete():
    assert set(MERGE) == set(CLASSES) - {"tumour", "other"}
    assert set(MERGE.values()) <= {"stroma", "other"}
    assert MERGE["stroma"] == "other" and MERGE["vessels"] == "stroma"


def test_center_crop():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[48, 48] = 200
    out = center_crop(img, 128)
    assert out.shape == (128, 128, 3)
    assert out[0, 0, 0] == 200 and out[1, 1, 0] == 0


def test_otsu_between_modes():
    gray = np.concatenate([np.full(500, 60), np.full(500, 220)]).astype(np.uint8)
    thr = otsu_threshold(gray)
    assert 60 <= thr < 220


def test_tissue_fraction_rejects_blank():
    white = np.full((256, 256, 3), 255, dtype=np.uint8)
    assert tissue_fraction(white, 200) < 0.5
    half = white.copy()
    half[:, :128] = 0
    assert abs(tissue_fraction(half, 200) - 0.5) <= 0.01


def test_labelled_tiles_cover_rule():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:, :256] = 1
    mask[:, 256:] = 2
    mask[256:, 256:] = 0
    assert set(labelled_tiles(mask, 256)) == {(0, 0, 1), (0, 256, 1), (256, 0, 2)}
    mixed = np.ones((256, 256), dtype=np.uint8)
    mixed[:, 179:] = 2  # 70% code 1, 30% code 2
    assert labelled_tiles(mixed, 256) == []


def test_tss_of():
    assert tss_of("TCGA-A2-A0T2-01Z-00-DX1") == "A2"
    assert tss_of("TCGA-BH-A0AU") == "BH"


def test_one_site_map_for_two_sources():
    rng = np.random.default_rng(1)
    sites = [f"S{i:02d}" for i in range(40)]
    strata = ["BRCA" if i < 20 else "LUAD" for i in range(40)]
    groups = sites * 3  # the same site appears in several sources and tiles
    strat = strata * 3
    m1 = assign_groups(groups, strat, seed=0)
    order = rng.permutation(len(groups))
    m2 = assign_groups([groups[i] for i in order], [strat[i] for i in order], seed=0)
    assert m1 == m2
    assert set(m1) == set(sites) and set(m1.values()) <= {"train", "valid", "test"}
    for s in ("BRCA", "LUAD"):
        got = [m1[g] for g, t in zip(sites, strata) if t == s]
        assert abs(got.count("test") - 3) <= 1 and abs(got.count("valid") - 3) <= 1
    assert assign_groups(groups, strat, seed=1) != m1
