#!/bin/bash
#SBATCH --job-name=ga2m_sweep
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_sweep_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_sweep_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability
python model/debug_ga2m_sweep.py
