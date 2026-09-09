#!/bin/bash
#SBATCH --account=def-senger_cpu
#SBATCH --job-name=s2_brca_clin_q${QUANTILE:-50}
#SBATCH --output=logs/step2_brca_clinical_q%j.out
#SBATCH --error=logs/step2_brca_clinical_q%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G

# Step 2 — BRCA clinical MED3PA reliability (all 2a-2f changes applied)
#
# BRCA has 1060 patients vs BLCA's 379; more memory requested.
# Backbone round count: 106.
#
# Required env vars:
#   QUANTILE  — 50, 70, 80, or 90
#
# Data must be present in ~/data/ before this job runs:
#   patient_error_brca_oob.npz
#   clinical_features_BRCA.csv

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
mkdir -p logs "$OUTDIR"

echo "[$(date)] Starting BRCA clinical q${Q} on $SLURMD_NODENAME"

python analysis/brca_clinical_linear_med3pa.py \
    --error-npz      "$DATA/patient_error_brca_oob.npz" \
    --clinical-csv   "$DATA/clinical_features_BRCA.csv" \
    --round-count    106 \
    --save-root      "$DATA/med3pa_bootstrap_intermediate_BRCA" \
    --outdir         "$OUTDIR" \
    --med3pa-error-quantile "$QFLOAT" \
    --med3pa-n-runs  200 \
    --med3pa-n-parallel 10 \
    --fi-n-jobs      10 \
    --discovery-frac 0.6 \
    --split-seed     20260908

echo "[$(date)] Done. Results in $OUTDIR"
