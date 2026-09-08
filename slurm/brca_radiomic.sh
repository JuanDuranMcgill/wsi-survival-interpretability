#!/bin/bash
#SBATCH --job-name=brca_rad_med3pa
#SBATCH --output=/home/sorkwos/links/scratch/slurm_logs/brca_rad_med3pa_%j.out
#SBATCH --error=/home/sorkwos/links/scratch/slurm_logs/brca_rad_med3pa_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=11:00:00

set -euo pipefail

module load gcc opencv/4.12.0
source ~/envs/conch_env/bin/activate

echo ">>> brca_radiomic_med3pa started on $(hostname) at $(date)"

python /home/sorkwos/links/scratch/multimodality/graph_subset_generation/brca_radiomic_linear_med3pa.py \
  --n-bootstrap 500000 \
  --n-jobs 24 \
  --n-runs 1000 \
  --n-parallel 24 \
  --outdir /home/sorkwos/links/scratch/multimodality/graph_subset_generation/results/brca_radiomic_med3pa_500k

echo ">>> brca_radiomic_med3pa finished at $(date)"
