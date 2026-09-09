#!/bin/bash
# Step 2 — BRCA clinical MED3PA reliability (Narval / Alliance)
# Cluster: Narval. Do NOT use on Trillium — paths and module names differ.
#
# BRCA has 1060 patients vs BLCA's 379; backbone round count: 106.
# Required env var: QUANTILE  — 50, 70, 80, or 90
#
# Submit via submit_all.sh, or manually:
#   QUANTILE=50 sbatch --job-name=s2_brca_clin_q50 slurm/narval/step2/brca_clinical.sh
#
# Data prerequisites in ~/data/:
#   patient_error_brca_oob.npz     (copy from Mac: results/patient_error_brca_oob.npz)
#   clinical_features_BRCA.csv     (copy from Mac: ~/data/clinical_features_BRCA.csv)

#SBATCH --account=def-senger_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/step2_brca_clinical_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/step2_brca_clinical_%j.err

set -euo pipefail

Q="${QUANTILE:-50}"
case "$Q" in
    50) QFLOAT=0.5 ;;
    70) QFLOAT=0.7 ;;
    80) QFLOAT=0.8 ;;
    90) QFLOAT=0.9 ;;
    *)  echo "ERROR: QUANTILE must be 50, 70, 80, or 90 (got $Q)"; exit 1 ;;
esac

REPO="$HOME/wsi-survival-interpretability"
DATA="$HOME/data"
OUTDIR="$REPO/results/reliability_brca_clinical_q${Q}"

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

cd "$REPO"
mkdir -p "$OUTDIR"

echo "[$(date)] BRCA clinical q${Q} starting on $(hostname)"
echo "  cpus     : $SLURM_CPUS_PER_TASK"
echo "  mem      : $SLURM_MEM_PER_NODE MB"
echo "  error_npz: $DATA/patient_error_brca_oob.npz"
echo "  clinical : $DATA/clinical_features_BRCA.csv"
echo "  outdir   : $OUTDIR"

python analysis/brca_clinical_linear_med3pa.py \
    --error-npz             "$DATA/patient_error_brca_oob.npz" \
    --clinical-csv          "$DATA/clinical_features_BRCA.csv" \
    --round-count           106 \
    --save-root             "$DATA/med3pa_bootstrap_intermediate_BRCA" \
    --outdir                "$OUTDIR" \
    --med3pa-error-quantile "$QFLOAT" \
    --med3pa-n-runs         200 \
    --med3pa-n-parallel     10 \
    --fi-n-jobs             10 \
    --discovery-frac        0.6 \
    --split-seed            20260908

echo "[$(date)] BRCA clinical q${Q} done. Results in $OUTDIR"
