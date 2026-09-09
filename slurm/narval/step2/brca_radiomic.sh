#!/bin/bash
# Step 2 — BRCA radiomic MED3PA reliability (Narval / Alliance)
# Cluster: Narval. Do NOT use on Trillium — paths and module names differ.
#
# BRCA radiomic is the largest combination: 1060 patients, first-order
# radiomic features, 1000 bootstrap forests. 16h / 96G requested.
# --n-jobs is set to 14 to leave 2 CPUs for OS + SLURM bookkeeping.
#
# Required env var: QUANTILE  — 50, 70, 80, or 90
#
# Submit via submit_all.sh, or manually:
#   QUANTILE=50 sbatch --job-name=s2_brca_rad_q50 slurm/narval/step2/brca_radiomic.sh
#
# Data prerequisites in ~/data/:
#   patient_error_brca_oob.npz          (copy from Mac: results/patient_error_brca_oob.npz)
#   brca/radiomics_pre_corr.csv         (copy from Mac: ~/data/brca/radiomics_pre_corr.csv)

#SBATCH --account=def-senger_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/step2_brca_radiomic_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/step2_brca_radiomic_%j.err

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
OUTDIR="$REPO/results/reliability_brca_radiomic_q${Q}"

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

cd "$REPO"
mkdir -p "$OUTDIR"

echo "[$(date)] BRCA radiomic q${Q} starting on $(hostname)"
echo "  cpus        : $SLURM_CPUS_PER_TASK"
echo "  mem         : $SLURM_MEM_PER_NODE MB"
echo "  error_npz   : $DATA/patient_error_brca_oob.npz"
echo "  radiomics   : $DATA/brca/radiomics_pre_corr.csv"
echo "  outdir      : $OUTDIR"

python analysis/brca_radiomic_linear_med3pa.py \
    --error-npz             "$DATA/patient_error_brca_oob.npz" \
    --radiomics-csv         "$DATA/brca/radiomics_pre_corr.csv" \
    --round-count           106 \
    --save-root             "$DATA/med3pa_bootstrap_intermediate_BRCA" \
    --outdir                "$OUTDIR" \
    --med3pa-error-quantile "$QFLOAT" \
    --n-jobs                14 \
    --n-bootstrap           1000 \
    --top-n-med3pa          30 \
    --n-runs                200 \
    --n-parallel            10 \
    --discovery-frac        0.6 \
    --split-seed            20260908

echo "[$(date)] BRCA radiomic q${Q} done. Results in $OUTDIR"
