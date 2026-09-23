#!/bin/bash
#SBATCH --job-name=label_confidence_brca
#SBATCH --output=/scratch/sorkwos/slurm_logs/label_confidence_brca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/label_confidence_brca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> label_confidence_audit (BRCA) started on $(hostname) at $(date)"

python analysis/label_confidence_audit.py --cohort brca \
  --jsonl-dir /home/sorkwos/links/scratch/brca_jsons \
  --patients-file results/modelled_patients_brca.txt \
  --workers "$SLURM_CPUS_PER_TASK" \
  --out /scratch/sorkwos/label_confidence_brca.json

echo ">>> label_confidence_audit (BRCA) finished at $(date)"
