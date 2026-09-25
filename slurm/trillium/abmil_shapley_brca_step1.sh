#!/bin/bash
#SBATCH --job-name=abmil_brca_step1
#SBATCH --output=/scratch/sorkwos/slurm_logs/abmil_brca_step1_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/abmil_brca_step1_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:30:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> abmil_compartment_shapley BRCA step1 (round 1 timing) started on $(hostname) at $(date)"

python analysis/abmil_compartment_shapley.py --cohort brca \
  --rounds-dir /scratch/sorkwos/abmil_brca \
  --round-start 1 --round-end 1

echo ">>> abmil_compartment_shapley BRCA step1 finished at $(date)"
