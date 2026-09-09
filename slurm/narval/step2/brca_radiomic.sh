#!/bin/bash
#SBATCH --account=def-senger_cpu
#SBATCH --job-name=s2_brca_rad_q${QUANTILE:-50}
#SBATCH --output=logs/step2_brca_radiomic_q%j.out
#SBATCH --error=logs/step2_brca_radiomic_q%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Step 2 — BRCA radiomic MED3PA reliability (all 2a-2f changes applied)
#
# BRCA radiomic is the largest combination: 1060 patients, first-order
# radiomic features, 1000 bootstrap forests. 16h and 96G requested.
#
# Required env vars:
#   QUANTILE  — 50, 70, 80, or 90
#
# Data must be present in ~/data/ before this job runs:
#   patient_error_brca_oob.npz
#   radiomics_output_BRCA/radiomics_pre_corr.csv   (subdirectory matters)

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
mkdir -p logs "$OUTDIR"

echo "[$(date)] Starting BRCA radiomic q${Q} on $SLURMD_NODENAME"

python analysis/brca_radiomic_linear_med3pa.py \
    --error-npz          "$DATA/patient_error_brca_oob.npz" \
    --radiomics-csv      "$DATA/brca/radiomics_pre_corr.csv" \
    --round-count        106 \
    --save-root          "$DATA/med3pa_bootstrap_intermediate_BRCA" \
    --outdir             "$OUTDIR" \
    --med3pa-error-quantile "$QFLOAT" \
    --n-jobs             14 \
    --n-bootstrap        1000 \
    --top-n-med3pa       30 \
    --n-runs             200 \
    --n-parallel         10 \
    --discovery-frac     0.6 \
    --split-seed         20260908

echo "[$(date)] Done. Results in $OUTDIR"
