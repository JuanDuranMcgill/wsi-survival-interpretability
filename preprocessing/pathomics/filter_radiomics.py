#!/usr/bin/env python3
# Copied unchanged on 2026-11-21 from Trillium /scratch/sorkwos/multimodality/graph_subset_generation/filter_radiomics.py (last modified 2026-04-08 17:01); only this line was added.
# filter_radiomics.py
# Post-processing script for radiomic features.
# Aggregates all per-patient NPZ files, applies variance and
# correlation filtering, and saves a clean CSV ready for Med3PA.
#
# Usage: python filter_radiomics.py

import os
import json
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Config
# -------------------------------------------------------
RADIOMICS_DIR = "/home/sorkwos/links/scratch/radiomics_output"
OUTPUT_DIR    = "/home/sorkwos/links/scratch/radiomics_output"
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "radiomics_filtered.csv")

VARIANCE_THRESHOLD    = 0.01   # drop features with variance below this
CORRELATION_THRESHOLD = 0.90   # drop one of any pair correlated above this
NAN_PATIENT_THRESHOLD = 0.50   # drop patients missing more than 50% of features
NAN_FEATURE_THRESHOLD = 0.20   # drop features missing in more than 20% of patients


# -------------------------------------------------------
# Step 1 — Load all per-patient NPZ files
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

# Build dataframe: rows = patients, columns = features
df = pd.DataFrame(all_vectors, columns=feature_names, index=all_patient_ids)
df.index.name = "patient_id"

print(f"  Raw shape: {df.shape}  ({df.shape[0]} patients x {df.shape[1]} features)")


# -------------------------------------------------------
# Step 2 — Drop patients with too many missing features
# -------------------------------------------------------
print("\n>>> Step 1: Dropping patients with too many NaN features...")
nan_frac_patients = df.isna().mean(axis=1)
before = len(df)
df = df[nan_frac_patients <= NAN_PATIENT_THRESHOLD]
print(f"  Dropped {before - len(df)} patients (>{NAN_PATIENT_THRESHOLD*100:.0f}% missing features)")
print(f"  Remaining: {len(df)} patients")


# -------------------------------------------------------
# Step 3 — Drop features with too many missing values
# -------------------------------------------------------
print("\n>>> Step 2: Dropping features missing in too many patients...")
nan_frac_features = df.isna().mean(axis=0)
before = df.shape[1]
df = df.loc[:, nan_frac_features <= NAN_FEATURE_THRESHOLD]
print(f"  Dropped {before - df.shape[1]} features (>{NAN_FEATURE_THRESHOLD*100:.0f}% missing across patients)")
print(f"  Remaining: {df.shape[1]} features")


# -------------------------------------------------------
# Step 4 — Fill remaining NaNs with column median
# -------------------------------------------------------
print("\n>>> Step 3: Filling remaining NaN values with column median...")
df = df.fillna(df.median())


# -------------------------------------------------------
# Step 5 — Drop low-variance features
# -------------------------------------------------------
print("\n>>> Step 4: Dropping low-variance features...")
variances = df.var(axis=0)
before = df.shape[1]
df = df.loc[:, variances > VARIANCE_THRESHOLD]
print(f"  Dropped {before - df.shape[1]} features (variance < {VARIANCE_THRESHOLD})")
print(f"  Remaining: {df.shape[1]} features")


# -------------------------------------------------------
# Step 5b — Save pre-correlation version for RF importance
# -------------------------------------------------------
pre_corr_csv = os.path.join(OUTPUT_DIR, "radiomics_pre_corr.csv")
df.to_csv(pre_corr_csv)
print(f"\n>>> Saved pre-correlation CSV: {pre_corr_csv}")
print(f"  Shape: {df.shape}")


# -------------------------------------------------------
# Step 6 — Drop highly correlated features
# -------------------------------------------------------
print("\n>>> Step 5: Dropping highly correlated features...")
corr_matrix = df.corr().abs()

# Upper triangle only to avoid duplicates
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find columns where any correlation exceeds threshold
to_drop = [
    col for col in upper.columns
    if any(upper[col] > CORRELATION_THRESHOLD)
]

before = df.shape[1]
df = df.drop(columns=to_drop)
print(f"  Dropped {before - df.shape[1]} features (correlation > {CORRELATION_THRESHOLD})")
print(f"  Remaining: {df.shape[1]} features")


# -------------------------------------------------------
# Step 7 — Save
# -------------------------------------------------------
print(f"\n>>> Saving filtered feature matrix...")
df.to_csv(OUTPUT_CSV)

print(f"  Saved: {OUTPUT_CSV}")
print(f"  Final shape: {df.shape}  ({df.shape[0]} patients x {df.shape[1]} features)")
print("\n>>> Done.")