#!/bin/bash
#SBATCH --job-name=subsets_brca_r1gap
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/subsets_brca_r1gap_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/subsets_brca_r1gap_%j.err

# Round 1 for the 6 BRCA subset configs that never had one (Step 1 only timed
# `full` + one keep= config; Step 2 assumed round 1 existed everywhere).
# See docs/COMPARTMENT_EXPERIMENTS_STATUS.md, issue #2.
set -euo pipefail
module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> subsets BRCA round-1 gap fill (6 configs) started on $(hostname) at $(date)"

python analysis/compartment_subsets.py --cohort brca \
  --rounds-dir /scratch/sorkwos/subsets_brca \
  --round-start 1 --round-end 1 \
  --config "keep=Adipose+Muscle tissue" \
  --config "keep=Adipose+Muscle tissue+Blood vessels" \
  --config "keep=Invasive breast+Necrosis+Adipose" \
  --config "keep=Normal breast+Necrosis+Ductal carcinoma" \
  --config "drop=Necrosis" \
  --config "drop=Adipose"

echo ">>> subsets BRCA round-1 gap fill finished at $(date)"
