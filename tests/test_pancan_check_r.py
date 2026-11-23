import numpy as np
import pandas as pd
import pytest

from analysis.pancan.check_r_core import clinical_matrix, oob_risk, profile_cindex


def _round(path, patients, cidx, risk):
    path.mkdir(parents=True)
    np.savez(path / "oob.npz", patient=np.array(patients), time=np.arange(len(patients), dtype=float) + 1,
             event=np.ones(len(patients)), type=np.array(["A"] * len(patients)),
             risk=np.array(risk, dtype=float), cindex=np.array(cidx), region_weights=np.zeros((0, 2)))


def test_oob_risk_uses_the_epoch_with_the_best_mean_cindex_and_averages_over_rounds(tmp_path):
    # epoch 2 has the best mean C over the two rounds (0.6 and 0.7 vs 0.65 and 0.5)
    _round(tmp_path / "round_000", ["P1", "P2"], [0.65, 0.6], [[9, 9], [1.0, 2.0]])
    _round(tmp_path / "round_001", ["P2", "P3"], [0.5, 0.7], [[9, 9], [3.0, 5.0]])
    r = oob_risk(tmp_path).set_index("patient")
    assert r.loc["P1", "risk"] == 1.0 and r.loc["P3", "risk"] == 5.0
    assert r.loc["P2", "risk"] == pytest.approx(2.5) and r.loc["P2", "n_oob"] == 2
    assert int(r.attrs["epoch"]) == 2


def test_clinical_matrix_codes_stage_grade_sex_and_type():
    c = pd.DataFrame({"patient": ["P1", "P2", "P3"], "type": ["OV", "BRCA", "BRCA"], "age": [60.0, None, 40.0],
                      "sex": ["FEMALE", "FEMALE", "MALE"], "ajcc_pathologic_tumor_stage": [None, "Stage IIA", None],
                      "clinical_stage": ["Stage IIIC", None, None], "histological_grade": ["G3", None, "G1"]})
    X = clinical_matrix(c)
    assert X.loc["P1", "stage"] == 3 and X.loc["P2", "stage"] == 2 and X.loc["P3", "stage_na"] == 1
    assert X.loc["P1", "grade"] == 3 and X.loc["P2", "grade_na"] == 1
    assert X.loc["P3", "male"] == 1 and X.loc["P1", "type_OV"] == 1 and X.loc["P2", "type_OV"] == 0
    assert not X.isna().any().any()


def test_profile_cindex_is_within_type():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 1, 1])
    r = np.array([10.0, 11.0, 1.0, 0.0])
    s = np.array(["A", "A", "B", "B"])
    c, ci = profile_cindex(np.ones(4, bool), t, e, r, s, n_boot=50)
    assert c == pytest.approx(0.5) and len(ci) == 2
