#!/bin/bash
#SBATCH --job-name=planted_round
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/scratch/sorkwos/slurm_logs/planted_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/planted_round_%j.err

# Reusable planted-signal round runner for Narval.
# Cluster: Narval (Alliance Canada). Do NOT use on Trillium (different account
# names, different module system, different env).
#
# COHORT / PLANTED / MODELS / ROUNDS_DIR / ROUND_START / ROUND_END passed via
# sbatch --export. MODELS must be a SINGLE model name (graph or abmil), never
# a comma-joined list — Slurm's --export parser splits on commas, so a value
# like "graph,abmil" here would silently truncate to "graph". This is exactly
# the bug that caused BLCA planted-signal ABMIL (rounds 2-10) to never run on
# Trillium (see docs/COMPARTMENT_EXPERIMENTS_STATUS.md, issue #1). Submit one
# job per model instead.
set -euo pipefail
: "${COHORT:?}"; : "${PLANTED:?}"; : "${MODELS:?}"; : "${ROUNDS_DIR:?}"
: "${ROUND_START:?}"; : "${ROUND_END:?}"

case "$MODELS" in
  *,*) echo "ABORT: MODELS='$MODELS' contains a comma — this defeats the whole" \
            "point of this script (see header). Submit one job per model." >&2
       exit 1 ;;
esac

module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> planted $COHORT/$PLANTED model=$MODELS rounds $ROUND_START-$ROUND_END started on $(hostname) at $(date)"
echo ">>> Torch: $(python -c 'import torch; print(torch.__version__)')  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

python analysis/planted_signal.py --cohort "$COHORT" --planted "$PLANTED" \
  --models "$MODELS" --rounds-dir "$ROUNDS_DIR" \
  --round-start "$ROUND_START" --round-end "$ROUND_END"

echo ">>> planted $COHORT/$PLANTED model=$MODELS rounds $ROUND_START-$ROUND_END finished at $(date)"
