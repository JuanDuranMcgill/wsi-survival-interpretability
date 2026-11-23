import numpy as np
import pandas as pd

from analysis.pancan.cohort import build_cohort
from analysis.pancan.splits import site_rounds


def _manifest():
    return pd.DataFrame({
        "patient": ["P1", "P2", "P3", "P4"],
        "file_name": ["P1-DX1.a.svs", "P2-DX1.b.svs", "P3-DX1.c.svs", "P4-DX1.d.svs"],
        "type": ["LUAD", "LUAD", "BRCA", "GBM"],
        "tss_code": ["05", "44", "A2", "02"],
        "tss_institution": ["Inst A", None, "Inst B", "Inst C"],
        "PFI": [1.0, 0.0, 1.0, 1.0], "PFI.time": [100.0, 2000.0, 300.0, 50.0],
        "age_at_initial_pathologic_diagnosis": [60.0, 70.0, 50.0, 40.0], "gender": ["MALE", "FEMALE", "FEMALE", "MALE"],
        "ajcc_pathologic_tumor_stage": ["Stage I", "Stage II", "Stage III", None],
        "in_cohort": [True, True, True, False]})


def test_build_cohort_keeps_cohort_rows_and_names_unmapped_sites():
    c = build_cohort(_manifest())
    assert c["patient"].tolist() == ["P1", "P2", "P3"]
    assert c["slide"].tolist() == ["P1-DX1.a", "P2-DX1.b", "P3-DX1.c"]
    assert c["site"].tolist() == ["Inst A", "TSS-44", "Inst B"]
    assert c["pfi_event"].tolist() == [1, 0, 1]


def _cohort(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in ["LUAD", "BRCA", "KIRC"]:
        for s in range(12):
            for k in range(10):
                rows.append({"patient": f"{t}-{s}-{k}", "type": t, "site": f"site{s}" if t != "KIRC" else f"k{s}",
                             "pfi_time": float(rng.integers(10, 3000)), "pfi_event": int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_site_rounds_never_split_a_site_within_a_round():
    c = _cohort()
    r = site_rounds(c, n_rounds=3)
    m = r.merge(c[["patient", "site"]], on="patient")
    assert (m.groupby(["round", "site"])["in_train"].nunique() == 1).all()


def test_site_rounds_give_every_type_oob_patients_within_bounds():
    c = _cohort()
    r = site_rounds(c, n_rounds=3, oob_frac=0.2, max_frac=0.4)
    m = r.merge(c[["patient", "type"]], on="patient")
    oob = 1 - m.groupby(["round", "type"])["in_train"].mean()
    assert (oob > 0).all() and (oob <= 0.4).all()


def test_site_rounds_reach_a_type_whose_small_sites_are_shared_with_a_filled_type():
    rows = [{"patient": f"OV-big-{k}", "type": "OV", "site": "big"} for k in range(54)]
    for s in range(3):  # small sites shared with BRCA
        rows += [{"patient": f"OV-s{s}-{k}", "type": "OV", "site": f"shared{s}"} for k in range(15)]
        rows += [{"patient": f"BRCA-s{s}-{k}", "type": "BRCA", "site": f"shared{s}"} for k in range(5)]
    for s in range(20):
        rows += [{"patient": f"BRCA-{s}-{k}", "type": "BRCA", "site": f"b{s}"} for k in range(10)]
    c = pd.DataFrame(rows).assign(pfi_time=100.0, pfi_event=1)
    r = site_rounds(c, n_rounds=10).merge(c[["patient", "type"]], on="patient")
    oob = 1 - r.groupby(["round", "type"])["in_train"].mean()
    assert (oob.xs("OV", level="type") > 0).all()
    assert (oob <= 0.4).all()


def test_site_rounds_are_reproducible_and_differ_between_rounds():
    c = _cohort()
    a, b = site_rounds(c, n_rounds=2), site_rounds(c, n_rounds=2)
    assert a.equals(b)
    r0 = a[a["round"] == 0].set_index("patient")["in_train"]
    r1 = a[a["round"] == 1].set_index("patient")["in_train"]
    assert not r0.equals(r1)


def test_type_batches_are_single_type_cover_everyone_once_and_respect_the_size():
    from analysis.pancan.splits import type_batches
    types = np.array(["A"] * 70 + ["B"] * 33 + ["C"] * 1)
    batches = type_batches(types, 32, np.random.default_rng(0))
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(103))  # the lone C patient cannot form a risk set
    assert all(len(set(types[b])) == 1 and 2 <= len(b) <= 32 for b in batches)
