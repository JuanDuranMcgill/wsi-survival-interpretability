import os
import numpy as np
import pandas as pd

SAVE_ROOT = "/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate"
FINAL_FILE = os.path.join(SAVE_ROOT, "final_med3pa_input.npz")

EXCEL = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"

print(">>> Loading model outputs...")
data = np.load(FINAL_FILE, allow_pickle=True)

patient_ids = data["patient_ids"]
risks = data["mean_risk"]

# 🔴 clean patient IDs (first 12 chars)
patient_ids = [str(pid)[:12] for pid in patient_ids]

print(">>> Loading survival data...")
surv_df = pd.read_excel(EXCEL)

surv_df = surv_df[surv_df["type"].str.contains("BLCA|HNSC|LUAD|BRCA|UCEC", na=False)]
surv_df = surv_df[["bcr_patient_barcode", "PFI", "PFI.time"]]
surv_df["PFI.time"] = pd.to_numeric(surv_df["PFI.time"], errors="coerce")
surv_df = surv_df.dropna(subset=["PFI.time"])

# 🔴 align patients
times = []
events = []

missing = 0

for pid in patient_ids:
    row = surv_df.loc[surv_df["bcr_patient_barcode"] == pid]

    if row.empty:
        times.append(np.nan)
        events.append(np.nan)
        missing += 1
    else:
        times.append(float(row["PFI.time"].values[0]))
        events.append(int(row["PFI"].values[0]))

times = np.array(times)
events = np.array(events)

print(f">>> Missing survival entries: {missing}")

# remove invalid patients
valid_mask = ~np.isnan(times)

times = times[valid_mask]
events = events[valid_mask]
risks = risks[valid_mask]
patient_ids = np.array(patient_ids)[valid_mask]

# 🔴 compute error
print(">>> Computing per-patient error...")

N = len(times)
errors = np.zeros(N)
counts = np.zeros(N)

for i in range(N):
    for j in range(N):

        if i == j:
            continue

        if times[i] < times[j] and events[i] == 1:
            counts[i] += 1

            if risks[i] <= risks[j]:
                errors[i] += 1

valid = counts > 0
errors[valid] = errors[valid] / counts[valid]

print(">>> Done")

# 🔴 save
out_path = os.path.join(SAVE_ROOT, "med3pa_error_input.npz")

np.savez(
    out_path,
    patient_ids=patient_ids,
    error=errors,
    risk=risks
)

print(f">>> Saved: {out_path}")