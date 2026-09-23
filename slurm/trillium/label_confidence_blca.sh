#!/bin/bash
#SBATCH --job-name=label_confidence_blca
#SBATCH --output=/scratch/sorkwos/slurm_logs/label_confidence_blca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/label_confidence_blca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> label_confidence_audit (BLCA) started on $(hostname) at $(date)"

python analysis/label_confidence_audit.py --cohort blca \
  --jsonl-dir /home/sorkwos/links/scratch/blca_jsons \
  --patients-file results/modelled_patients_blca.txt \
  --workers "$SLURM_CPUS_PER_TASK" \
  --out /scratch/sorkwos/label_confidence_blca.json

echo ">>> label_confidence_audit (BLCA) finished at $(date)"
