#!/bin/bash
#SBATCH --job-name=subsets_brca_step1
#SBATCH --output=/scratch/sorkwos/slurm_logs/subsets_brca_step1_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/subsets_brca_step1_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> compartment_subsets BRCA step1 (round 1, full + one keep config) started on $(hostname) at $(date)"

python analysis/compartment_subsets.py --cohort brca \
  --rounds-dir /scratch/sorkwos/subsets_brca \
  --round-start 1 --round-end 1 \
  --config full \
  --config "keep=Adipose"

echo ">>> compartment_subsets BRCA step1 finished at $(date)"
