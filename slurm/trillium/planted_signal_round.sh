#!/bin/bash
#SBATCH --job-name=planted_round
#SBATCH --output=/scratch/sorkwos/slurm_logs/planted_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/planted_round_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=06:00:00

# COHORT / PLANTED / ROUNDS_DIR / ROUND_START / ROUND_END / MODELS passed via sbatch --export
set -euo pipefail
: "${COHORT:?}"; : "${PLANTED:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"; : "${MODELS:?}"
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> planted $COHORT/$PLANTED models=$MODELS rounds $ROUND_START-$ROUND_END started on $(hostname) at $(date)"
python analysis/planted_signal.py --cohort "$COHORT" --planted "$PLANTED" \
  --models "$MODELS" --rounds-dir "$ROUNDS_DIR" \
  --round-start "$ROUND_START" --round-end "$ROUND_END"
echo ">>> planted $COHORT/$PLANTED models=$MODELS rounds $ROUND_START-$ROUND_END finished at $(date)"
