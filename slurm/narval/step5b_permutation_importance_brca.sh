#!/bin/bash
#SBATCH --job-name=perm_imp_brca
#SBATCH --account=def-senger_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/perm_imp_brca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/perm_imp_brca_%j.err

# Step 5b — Permutation importance, BRCA cohort (CPU-only, does not contend with ablation).
# Checks whether impurity importance (reported in paper) and permutation importance agree.
# Output: results/permutation_importance_brca.json + .csv (no patient data; safe to commit).
#
# OOB NPZ lives outside the repo (/home/sorkwos/data/) and is never committed.

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"

mkdir -p /scratch/sorkwos/slurm_logs

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

echo ">>> CPUs: $SLURM_CPUS_PER_TASK"
echo ">>> Python: $(which python)"

python "$REPO/analysis/permutation_importance.py" \
    --cohort           brca \
    --radiomics-csv    "$HOME/data/brca/radiomics_pre_corr.csv" \
    --oob-npz          "$HOME/data/patient_error_brca_oob.npz" \
    --importance-csv   "$REPO/results/reliability_brca_radiomic_q50/radiomic_importance.csv" \
    --first-order-only \
    --n-trees          1000 \
    --n-repeats        10 \
    --n-jobs           32 \
    --out              "$REPO/results/permutation_importance_brca.json"

echo ">>> Done."
