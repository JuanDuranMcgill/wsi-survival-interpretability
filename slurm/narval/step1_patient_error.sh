#!/bin/bash
# Step 1 of the runbook: recompute per-patient error (fixed censoring definition).
# Cluster: Narval (Alliance).  Do NOT use this script on Trillium — paths differ.
#
# Submit:
#   sbatch --account=def-senger_cpu --time=0:30:00 \
#          --cpus-per-task=4 --mem=16G \
#          slurm/narval/step1_patient_error.sh
#
# Prerequisite: both final_med3pa_input.npz files must have been transferred
# from Trillium via Globus before this runs (see envs/narval.md).
#SBATCH --job-name=step1_patient_error
#SBATCH --output=/scratch/sorkwos/slurm_logs/step1_patient_error_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/step1_patient_error_%j.err

set -euo pipefail

REPO=/home/sorkwos/wsi-survival-interpretability
CDR=/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx

# Data paths — Narval layout (Trillium paths do NOT exist here)
BLCA_NPZ=/scratch/sorkwos/med3pa_bootstrap/final_med3pa_input.npz
BRCA_NPZ=/scratch/sorkwos/med3pa_bootstrap_intermediate_BRCA/final_med3pa_input.npz

module load python/3.11 scipy-stack
source "${REPO}/venv/bin/activate"

echo ">>> step1_patient_error started on $(hostname) at $(date)"

python "${REPO}/analysis/patient_error.py" \
    --cohort blca \
    --final-npz "${BLCA_NPZ}" \
    --cdr-xlsx  "${CDR}" \
    --out       "${REPO}/results/patient_error_blca.json"

python "${REPO}/analysis/patient_error.py" \
    --cohort brca \
    --final-npz "${BRCA_NPZ}" \
    --cdr-xlsx  "${CDR}" \
    --out       "${REPO}/results/patient_error_brca.json"

echo ">>> step1_patient_error finished at $(date)"
