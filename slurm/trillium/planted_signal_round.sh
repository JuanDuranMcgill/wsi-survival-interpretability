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
# MODELS must be a SINGLE model name (graph or abmil), never a comma-joined
# list — Slurm's --export parser splits on commas, so "graph,abmil" here
# silently truncates to "graph". This is exactly the bug that caused BLCA
# planted-signal ABMIL (rounds 2-10) to never run the first time around (see
# docs/COMPARTMENT_EXPERIMENTS_STATUS.md, issue #1). Submit one job per model.
set -euo pipefail
: "${COHORT:?}"; : "${PLANTED:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"; : "${MODELS:?}"
BATCH_SIZE="${BATCH_SIZE:-16}"

case "$MODELS" in
  *,*) echo "ABORT: MODELS='$MODELS' contains a comma — see header for why this" \
            "silently breaks. Submit one job per model." >&2
       exit 1 ;;
esac

source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> planted $COHORT/$PLANTED models=$MODELS rounds $ROUND_START-$ROUND_END started on $(hostname) at $(date)"
python analysis/planted_signal.py --cohort "$COHORT" --planted "$PLANTED" \
  --models "$MODELS" --rounds-dir "$ROUNDS_DIR" \
  --round-start "$ROUND_START" --round-end "$ROUND_END" \
  --batch-size "$BATCH_SIZE"
echo ">>> planted $COHORT/$PLANTED models=$MODELS rounds $ROUND_START-$ROUND_END finished at $(date)"
