#!/usr/bin/env python3
# Copied unchanged on 2026-11-21 from Trillium /scratch/sorkwos/multimodality/graph_subset_generation/filter_radiomics_firstorder.py (last modified 2026-04-22 16:12); only this line was added.
# filter_radiomics_firstorder.py

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Config
# -------------------------------------------------------
RADIOMICS_DIR          = "/home/sorkwos/links/scratch/radiomics_output"
OUTPUT_DIR             = "/home/sorkwos/links/scratch/radiomics_output"
OUTPUT_CSV             = os.path.join(OUTPUT_DIR, "radiomics_firstorder_504.csv")
VARIANCE_THRESHOLD     = 0.01
NAN_PATIENT_THRESHOLD  = 0.50
NAN_FEATURE_THRESHOLD  = 0.20

# -------------------------------------------------------
# Step 0 — Load all per-patient NPZ files
# -------------------------------------------------------
print(">>> Loading per-patient NPZ files...")
npz_files = sorted([
    os.path.join(RADIOMICS_DIR, f)
    for f in os.listdir(RADIOMICS_DIR)
    if f.endswith("_radiomics.npz")
])
print(f"  Found {len(npz_files)} patients.")

all_patient_ids = []
all_vectors     = []
feature_names   = None

for path in npz_files:
    data = np.load(path, allow_pickle=True)
    pid  = str(data["patient_id"])
    vec  = data["feature_vector"]
    if feature_names is None:
        feature_names = list(data["feature_names"])
    all_patient_ids.append(pid)
    all_vectors.append(vec)

df = pd.DataFrame(all_vectors, columns=feature_names, index=all_patient_ids)
df.index.name = "patient_id"
print(f"  Raw shape: {df.shape}  ({df.shape[0]} patients x {df.shape[1]} features)")

# -------------------------------------------------------
# Step 1 — First-order filter
# -------------------------------------------------------
print("\n>>> Step 1: Keeping first-order features only...")
before = df.shape[1]
df = df.loc[:, df.columns.str.contains("firstorder")]
print(f"  Dropped {before - df.shape[1]} non-first-order features")
print(f"  Remaining: {df.shape[1]} first-order features")

# -------------------------------------------------------
# Step 2 — Drop patients with too many missing features
# -------------------------------------------------------
print("\n>>> Step 2: Dropping patients with too many NaN features...")
nan_frac_patients = df.isna().mean(axis=1)
before = len(df)
df = df[nan_frac_patients <= NAN_PATIENT_THRESHOLD]
print(f"  Dropped {before - len(df)} patients (>{NAN_PATIENT_THRESHOLD*100:.0f}% missing features)")
print(f"  Remaining: {len(df)} patients")

# -------------------------------------------------------
# Step 3 — Drop features with too many missing values
# -------------------------------------------------------
print("\n>>> Step 3: Dropping features missing in too many patients...")
nan_frac_features = df.isna().mean(axis=0)
before = df.shape[1]
df = df.loc[:, nan_frac_features <= NAN_FEATURE_THRESHOLD]
print(f"  Dropped {before - df.shape[1]} features (>{NAN_FEATURE_THRESHOLD*100:.0f}% missing across patients)")
print(f"  Remaining: {df.shape[1]} features")

# -------------------------------------------------------
# Step 4 — Fill remaining NaNs with column median
# -------------------------------------------------------
print("\n>>> Step 4: Filling remaining NaN values with column median...")
n_filled = df.isna().sum().sum()
df = df.fillna(df.median())
print(f"  Filled {n_filled} NaN cells")

# -------------------------------------------------------
# Step 5 — Drop low-variance features
# -------------------------------------------------------
print("\n>>> Step 5: Dropping low-variance features...")
variances = df.var(axis=0)
before = df.shape[1]
df = df.loc[:, variances > VARIANCE_THRESHOLD]
print(f"  Dropped {before - df.shape[1]} features (variance < {VARIANCE_THRESHOLD})")
print(f"  Remaining: {df.shape[1]} features")

# -------------------------------------------------------
# Save
# -------------------------------------------------------
print(f"\n>>> Saving...")
df.to_csv(OUTPUT_CSV)
print(f"  Saved: {OUTPUT_CSV}")
print(f"  Final shape: {df.shape}  ({df.shape[0]} patients x {df.shape[1]} features)")
print("\n>>> Done.")