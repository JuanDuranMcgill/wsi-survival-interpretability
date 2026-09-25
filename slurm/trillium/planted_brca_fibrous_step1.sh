#!/bin/bash
#SBATCH --job-name=planted_brca_fib_s1
#SBATCH --output=/scratch/sorkwos/slurm_logs/planted_brca_fibrous_step1_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/planted_brca_fibrous_step1_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> planted_signal BRCA/Fibrous step1 (round 1, graph+abmil) started on $(hostname) at $(date)"

python analysis/planted_signal.py --cohort brca --planted Fibrous \
  --models graph,abmil \
  --rounds-dir /scratch/sorkwos/planted_brca_fibrous \
  --round-start 1 --round-end 1

echo ">>> planted_signal BRCA/Fibrous step1 finished at $(date)"
