#!/bin/bash
#SBATCH --job-name=ga2m_pilot_aggregate
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_pilot_aggregate_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_pilot_aggregate_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability
python model/aggregate_pilot_results.py
