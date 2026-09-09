#!/bin/bash
# Submit all 16 Step 2 jobs: 4 arms × 4 quantiles.
# Cluster: Narval (Alliance). Do NOT use on Trillium.
#
# Run from the repo root:
#   cd ~/wsi-survival-interpretability
#   bash slurm/narval/step2/submit_all.sh
#
# Prerequisites — complete ALL of these before running:
#
# 1. Populate ~/data/ (copy from your local Mac):
#      scp results/patient_error_blca_oob.npz  narval:~/data/
#      scp results/patient_error_brca_oob.npz  narval:~/data/
#      scp ~/data/clinical_features.csv        narval:~/data/
#      scp ~/data/clinical_features_BRCA.csv   narval:~/data/
#      rsync -avz ~/data/blca/  narval:~/data/blca/
#      rsync -avz ~/data/brca/  narval:~/data/brca/
#
# 2. Install MED3pa + GPU deps on the LOGIN NODE (one-time, ~10 min):
#      module load python/3.11 scipy-stack
#      source ~/wsi-survival-interpretability/venv/bin/activate
#      pip install --no-index numpy pandas scikit-learn scipy openpyxl joblib matplotlib
#      pip install med3pa==1.0.4 --no-deps
#      pip install torch --index-url https://download.pytorch.org/whl/cu118
#      pip install xgboost ray tqdm pyyaml relib checkpointer
#    (internet access required — do this on the login node, never on compute)
#
# Note: #SBATCH --job-name does NOT expand env vars on Alliance clusters.
# Job names are passed explicitly via --job-name in the sbatch call below.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/scratch/sorkwos/slurm_logs"

mkdir -p "$LOG_DIR"

declare -A ARMS=(
    [blca_clinical]="$SCRIPTS_DIR/blca_clinical.sh"
    [brca_clinical]="$SCRIPTS_DIR/brca_clinical.sh"
    [blca_radiomic]="$SCRIPTS_DIR/blca_radiomic.sh"
    [brca_radiomic]="$SCRIPTS_DIR/brca_radiomic.sh"
)

QUANTILES=(50 70 80 90)

for arm in blca_clinical brca_clinical blca_radiomic brca_radiomic; do
    for q in "${QUANTILES[@]}"; do
        JOB_NAME="s2_${arm}_q${q}"
        JOB_ID=$(QUANTILE=$q sbatch \
            --job-name="$JOB_NAME" \
            --parsable \
            "${ARMS[$arm]}")
        echo "Submitted $arm q${q} → job $JOB_ID (name: $JOB_NAME)"
    done
done

echo ""
echo "All 16 jobs submitted. Monitor with:"
echo "  squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R'"
echo "  tail -f $LOG_DIR/step2_*.out"
echo ""
echo "Primary results (q50) will be in:"
echo "  results/reliability_blca_clinical_q50/reliability_blca_clinical.json"
echo "  results/reliability_brca_clinical_q50/reliability_brca_clinical.json"
echo "  results/reliability_blca_radiomic_q50/reliability_blca_radiomic.json"
echo "  results/reliability_brca_radiomic_q50/reliability_brca_radiomic.json"
