#!/bin/bash
# Submit all 16 Step 2 jobs: 4 arms × 4 quantiles.
#
# Prerequisites (must be done before running this script):
#
#   1. Populate ~/data/ with:
#        patient_error_blca_oob.npz       ← from results/ on this machine
#        patient_error_brca_oob.npz       ← from results/ on this machine
#        clinical_features.csv            ← from Trillium (scp or Globus)
#        clinical_features_BRCA.csv       ← from Trillium
#        radiomics_output/radiomics_pre_corr.csv        ← from Trillium
#        radiomics_output_BRCA/radiomics_pre_corr.csv   ← from Trillium
#
#   2. Install MED3pa dependencies on the login node (one-time):
#        module load python/3.11 scipy-stack
#        source ~/wsi-survival-interpretability/venv/bin/activate
#        pip install --no-index numpy pandas scikit-learn scipy openpyxl joblib matplotlib
#        pip install med3pa==1.0.4 --no-deps
#        pip install torch --index-url https://download.pytorch.org/whl/cu118
#        pip install xgboost ray tqdm pyyaml relib checkpointer
#
#   3. Run this script from the repo root:
#        cd ~/wsi-survival-interpretability
#        bash slurm/narval/step2/submit_all.sh
#
# All 16 jobs run independently; quantile 0.5 (q50) is the primary result.

set -euo pipefail

SCRIPTS_DIR="$(dirname "$0")"
mkdir -p logs

ARMS=(blca_clinical brca_clinical blca_radiomic brca_radiomic)
QUANTILES=(50 70 80 90)

for arm in "${ARMS[@]}"; do
    for q in "${QUANTILES[@]}"; do
        JOB_ID=$(QUANTILE=$q sbatch --parsable "$SCRIPTS_DIR/${arm}.sh")
        echo "Submitted $arm q${q}: job $JOB_ID"
    done
done

echo ""
echo "All 16 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/step2_*"
