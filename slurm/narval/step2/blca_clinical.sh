#!/bin/bash
#SBATCH --account=def-senger_cpu
#SBATCH --job-name=s2_blca_clin_q${QUANTILE:-50}
#SBATCH --output=logs/step2_blca_clinical_q%j.out
#SBATCH --error=logs/step2_blca_clinical_q%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# Step 2 — BLCA clinical MED3PA reliability (all 2a-2f changes applied)
#
# Required env vars:
#   QUANTILE  — integer label for the quantile: 50, 70, 80, or 90
#               (maps to --med3pa-error-quantile 0.50, 0.70, 0.80, 0.90)
#
# Data must be present in ~/data/ before this job runs:
#   patient_error_blca_oob.npz         (from oob_risk.py + patient_error.py)
#   clinical_features.csv              (from Trillium; transfer via scp/Globus)
#
# Submit with:
#   QUANTILE=50 sbatch slurm/narval/step2/blca_clinical.sh
#   QUANTILE=70 sbatch slurm/narval/step2/blca_clinical.sh
#   QUANTILE=80 sbatch slurm/narval/step2/blca_clinical.sh
#   QUANTILE=90 sbatch slurm/narval/step2/blca_clinical.sh
#
# Or use slurm/narval/step2/submit_all.sh to launch all 16 combinations.

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
OUTDIR="$REPO/results/reliability_blca_clinical_q${Q}"

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

cd "$REPO"
mkdir -p logs "$OUTDIR"

echo "[$(date)] Starting BLCA clinical q${Q} on $SLURMD_NODENAME"
echo "  error_npz : $DATA/patient_error_blca_oob.npz"
echo "  clinical  : $DATA/clinical_features.csv"
echo "  outdir    : $OUTDIR"
echo "  quantile  : $QFLOAT"

python analysis/blca_clinical_linear_med3pa.py \
    --error-npz      "$DATA/patient_error_blca_oob.npz" \
    --clinical-csv   "$DATA/clinical_features.csv" \
    --round-count    102 \
    --save-root      "$DATA/med3pa_bootstrap_intermediate" \
    --outdir         "$OUTDIR" \
    --med3pa-error-quantile "$QFLOAT" \
    --med3pa-n-runs  200 \
    --med3pa-n-parallel 10 \
    --fi-n-jobs      10 \
    --discovery-frac 0.6 \
    --split-seed     20260908

echo "[$(date)] Done. Results in $OUTDIR"
