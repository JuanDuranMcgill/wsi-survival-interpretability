#!/bin/bash
#SBATCH --job-name=ga2m_attribution_vs_ablation
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_attribution_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_attribution_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability
python model/ga2m_attribution_vs_ablation.py
