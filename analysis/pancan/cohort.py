"""The pan-carcinoma cohort table: one slide per patient, PFI from TCGA-CDR, hospital (Beast, any env).

  python -I cohort.py   -> /data/processed/juan/pancan/cohort.csv (patient-level; never in git)
"""
import sys
from pathlib import Path

import pandas as pd

MANIFEST = Path("/data/processed/juan/pancan/manifest_patients.csv")
COHORT = Path("/data/processed/juan/pancan/cohort.csv")


def build_cohort(man):
    """Rows with in_cohort; site = TCGA tissue-source institution (TSS code when unmapped), as in the manifest."""
    m = man[man["in_cohort"].astype(bool)].copy()
    out = pd.DataFrame({
        "patient": m["patient"].values,
        "slide": m["file_name"].str.replace(".svs", "", regex=False).values,
        "type": m["type"].values,
        "site": m["tss_institution"].fillna("TSS-" + m["tss_code"].astype(str)).values,
        "pfi_time": m["PFI.time"].astype(float).values,
        "pfi_event": m["PFI"].astype(int).values,
        "age": m["age_at_initial_pathologic_diagnosis"].values,
        "sex": m["gender"].values,
        "stage": m["ajcc_pathologic_tumor_stage"].values})
    assert out["patient"].is_unique, "one slide per patient"
    return out.reset_index(drop=True)


if __name__ == "__main__":
    c = build_cohort(pd.read_csv(MANIFEST))
    c.to_csv(COHORT, index=False)
    print(len(c), "patients,", c["pfi_event"].sum(), "PFI events,", c["type"].nunique(), "types,",
          c["site"].nunique(), "sites ->", COHORT, file=sys.stderr)
