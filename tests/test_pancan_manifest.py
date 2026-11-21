"""Rules of the pan-cancer manifest: slide-name parsing, one slide per patient, eligibility."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
import pancan_manifest as pm  # noqa: E402


def test_parse_slide_name():
    p = pm.parse_slide_name("TCGA-38-4631-01Z-00-DX1.5e0c873a-9c4c-4e0b-bf2e-e3cd8b760761.svs")
    assert p == {"patient": "TCGA-38-4631", "tss_code": "38", "sample_code": "01",
                 "slide_tag": "DX1"}


def test_parse_slide_name_metastasis_and_dx2():
    p = pm.parse_slide_name("TCGA-EE-A2GJ-06Z-00-DX2.abc.svs")
    assert p["sample_code"] == "06" and p["slide_tag"] == "DX2" and p["tss_code"] == "EE"


def _slides(rows):
    return pd.DataFrame(rows, columns=["file_name", "patient", "sample_code", "slide_tag",
                                       "file_size"])


def test_one_slide_per_patient_keeps_dx1_only():
    s = _slides([
        ["A-DX1", "P1", "01", "DX1", 10],
        ["A-DX2", "P1", "01", "DX2", 10],
        ["B-DX2", "P2", "01", "DX2", 10],   # no DX1: patient dropped
    ])
    chosen = pm.select_one_slide_per_patient(s)
    assert list(chosen["file_name"]) == ["A-DX1"]


def test_one_slide_per_patient_prefers_primary_then_name():
    s = _slides([
        ["Z-06-DX1", "P1", "06", "DX1", 10],
        ["Y-01-DX1", "P1", "01", "DX1", 10],
        ["X-01-DX1", "P1", "01", "DX1", 10],
    ])
    chosen = pm.select_one_slide_per_patient(s)
    assert list(chosen["file_name"]) == ["X-01-DX1"]
    assert int(chosen["n_dx1_candidates"].iloc[0]) == 3


def test_eligibility_reasons():
    pat = pd.DataFrame({
        "patient": ["P1", "P2", "P3", "P4", "P5"],
        "in_cdr": [True, False, True, True, True],
        "Redaction": [None, None, "Redacted", None, None],
        "PFI": [1.0, None, 0.0, None, 0.0],
        "PFI.time": [100.0, None, 50.0, 30.0, 20.0],
        "type": ["BLCA", None, "BLCA", "LAML", "PCPG"],
    })
    out = pm.add_eligibility(pat)
    assert list(out["exclusion"]) == ["", "not_in_cdr", "redacted", "no_pfi",
                                      "pfi_not_recommended"]
    assert list(out["eligible"]) == [True, False, False, False, False]
