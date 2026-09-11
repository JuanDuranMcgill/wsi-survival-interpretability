#!/bin/bash
#SBATCH --job-name=ga2m_synth_check
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_synth_check_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_synth_check_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> GA2M synthetic identifiability check started on $(hostname) at $(date)"
python model/validate_ga2m_synthetic.py
echo ">>> finished at $(date), exit code $?"
