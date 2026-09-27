#!/bin/bash
#SBATCH --job-name=subsets_blca_r1gap2
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/subsets_blca_r1gap2_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/subsets_blca_r1gap2_%j.err

# Retry of the 2 BLCA subset configs (drop=Necrosis, drop=Perivesical adipose)
# that OOM'd in subsets_blca_round1_gap.sh — these keep 7 of 8 regions, close
# to the full model's footprint. Lower batch size for headroom. The other 4
# configs already completed and are skipped automatically if rerun.
set -euo pipefail
module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> subsets BLCA round-1 gap retry (drop configs, batch-size 4) started on $(hostname) at $(date)"

python analysis/compartment_subsets.py --cohort blca \
  --rounds-dir /scratch/sorkwos/subsets_blca \
  --round-start 1 --round-end 1 \
  --batch-size 4 \
  --config "drop=Necrosis" \
  --config "drop=Perivesical adipose"

echo ">>> subsets BLCA round-1 gap retry finished at $(date)"
