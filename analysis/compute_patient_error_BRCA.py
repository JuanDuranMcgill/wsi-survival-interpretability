#!/usr/bin/env python3
# compute_patient_error_BRCA.py
#
# Computes concordance-based per-patient error for BRCA patients.
# For each patient i, error = fraction of valid comparable pairs (i,j)
# where the model ranked them incorrectly (discordant).
#
# Inputs:
#   final_med3pa_input.npz   — patient_ids + mean_risk (from merge_bootstrap_BRCA.py)
#   TCGA-CDR Excel           — PFI and PFI.time for BRCA patients
#
# Output:
#   med3pa_error_input.npz   — patient_ids + per-patient concordance error
#
# Usage:
#   python compute_patient_error_BRCA.py

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
FINAL_FILE  = "/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate_BRCA/final_med3pa_input.npz"
ERROR_FILE  = "/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate_BRCA/med3pa_error_input.npz"
CDR_EXCEL   = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"

# -------------------------------------------------------
# Step 1 — Load mean risk scores
# -------------------------------------------------------
print(">>> Loading mean risk scores from bootstrap merge...")
data      = np.load(FINAL_FILE, allow_pickle=True)
pids_raw  = np.array([str(p)[:12] for p in data["patient_ids"]])
mean_risk = np.asarray(data["mean_risk"], dtype=np.float64)
print(f"    Patients with risk scores: {len(pids_raw)}")

# -------------------------------------------------------
# Step 2 — Load survival data from TCGA CDR
# -------------------------------------------------------
print(">>> Loading BRCA survival data from TCGA CDR...")
surv_df = pd.read_excel(CDR_EXCEL)
surv_df = surv_df[surv_df["type"].str.contains("BRCA", na=False)]
surv_df = surv_df[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
surv_df["PFI.time"] = pd.to_numeric(surv_df["PFI.time"], errors="coerce")
surv_df["PFI"]      = pd.to_numeric(surv_df["PFI"],      errors="coerce")
surv_df = surv_df.dropna(subset=["PFI.time", "PFI"])
surv_df["bcr_patient_barcode"] = surv_df["bcr_patient_barcode"].astype(str).str[:12]
surv_df = surv_df.set_index("bcr_patient_barcode")
print(f"    BRCA patients in CDR: {len(surv_df)}")

# -------------------------------------------------------
# Step 3 — Align patients
# -------------------------------------------------------
print(">>> Aligning risk scores with survival data...")
valid_mask = np.array([pid in surv_df.index for pid in pids_raw])
pids      = pids_raw[valid_mask]
risk      = mean_risk[valid_mask]
times     = np.array([surv_df.loc[pid, "PFI.time"] for pid in pids], dtype=np.float64)
events    = np.array([surv_df.loc[pid, "PFI"]      for pid in pids], dtype=np.int32)
n         = len(pids)
print(f"    Aligned patients: {n}  (dropped {(~valid_mask).sum()} with no survival data)")
print(f"    Events: {events.sum()}  |  Censored: {(events==0).sum()}")

# -------------------------------------------------------
# Step 4 — Compute per-patient concordance error (vectorised)
# -------------------------------------------------------
print(">>> Computing per-patient concordance error...")

# For each pair (i,j), a valid comparable pair exists when:
#   event_i=1 and t_i < t_j  →  i should have HIGHER risk than j
#   event_j=1 and t_j < t_i  →  j should have HIGHER risk than i
# We count discordant pairs per patient.

errors = np.zeros(n, dtype=np.float64)

for i in range(n):
    n_concordant = 0
    n_discordant = 0

    ri = risk[i]
    ti = times[i]
    ei = events[i]

    rj = risk       # all others (we'll mask self out)
    tj = times
    ej = events

    # Case A: patient i had event, died before j → i should have higher risk
    if ei == 1:
        mask_A = (tj > ti)                      # j survived past i's death
        mask_A[i] = False
        diff_A = ri - rj[mask_A]
        n_concordant += int((diff_A > 0).sum())
        n_discordant += int((diff_A < 0).sum())

    # Case B: patient j had event, died before i → j should have higher risk
    mask_B = (ej == 1) & (tj < ti)
    mask_B[i] = False
    diff_B = rj[mask_B] - ri                    # j's risk minus i's risk
    n_concordant += int((diff_B > 0).sum())
    n_discordant += int((diff_B < 0).sum())

    total = n_concordant + n_discordant
    errors[i] = n_discordant / total if total > 0 else 0.0

print(f"    Error range: [{errors.min():.4f}, {errors.max():.4f}]")
print(f"    Mean error:   {errors.mean():.4f}  |  Std: {errors.std():.4f}")

# -------------------------------------------------------
# Step 5 — Save
# -------------------------------------------------------
np.savez(ERROR_FILE, patient_ids=pids, error=errors)
print(f"\n>>> Saved: {ERROR_FILE}")
print(f"    Patients: {n}")
print(f"\n>>> Done. You can now run radiomic_importance_bootstrap_BRCA.py and bootstrap_med3pa_clinical_BRCA.py")