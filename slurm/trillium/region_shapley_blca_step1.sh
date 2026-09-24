#!/bin/bash
#SBATCH --job-name=shapley_blca_step1
#SBATCH --output=/scratch/sorkwos/slurm_logs/shapley_blca_step1_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/shapley_blca_step1_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=03:00:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> region_shapley BLCA step1 (round 1 timing) started on $(hostname) at $(date)"

python analysis/region_shapley.py --cohort blca \
  --rounds-dir /scratch/sorkwos/shapley_blca \
  --round-start 1 --round-end 1

echo ">>> region_shapley BLCA step1 finished at $(date)"
