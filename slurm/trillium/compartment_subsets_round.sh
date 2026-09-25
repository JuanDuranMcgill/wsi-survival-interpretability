#!/bin/bash
#SBATCH --job-name=subsets_round
#SBATCH --output=/scratch/sorkwos/slurm_logs/subsets_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/subsets_round_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=06:00:00

# COHORT / ROUNDS_DIR / ROUND_START / ROUND_END / CONFIG passed via sbatch --export
set -euo pipefail
: "${COHORT:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"; : "${CONFIG:?}"
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> subsets $COHORT config=[$CONFIG] rounds $ROUND_START-$ROUND_END started on $(hostname) at $(date)"
python analysis/compartment_subsets.py --cohort "$COHORT" \
  --rounds-dir "$ROUNDS_DIR" --round-start "$ROUND_START" --round-end "$ROUND_END" \
  --config "$CONFIG"
echo ">>> subsets $COHORT config=[$CONFIG] rounds $ROUND_START-$ROUND_END finished at $(date)"
