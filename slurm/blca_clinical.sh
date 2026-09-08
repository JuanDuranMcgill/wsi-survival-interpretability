#!/bin/bash
#SBATCH --job-name=blca_clin_med3pa
#SBATCH --output=/home/sorkwos/links/scratch/slurm_logs/blca_clin_med3pa_%j.out
#SBATCH --error=/home/sorkwos/links/scratch/slurm_logs/blca_clin_med3pa_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=11:00:00

set -euo pipefail

module load gcc opencv/4.12.0
source ~/envs/conch_env/bin/activate

echo ">>> blca_clinical_med3pa started on $(hostname) at $(date)"

python /home/sorkwos/links/scratch/multimodality/graph_subset_generation/blca_clinical_linear_med3pa.py \
  --fi-n-bootstrap 500000 \
  --fi-n-jobs 24 \
  --med3pa-n-runs 1000 \
  --med3pa-n-parallel 24 \
  --outdir /home/sorkwos/links/scratch/multimodality/graph_subset_generation/results/blca_clinical_med3pa_500k

echo ">>> blca_clinical_med3pa finished at $(date)"
