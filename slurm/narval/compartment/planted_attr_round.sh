#!/bin/bash
#SBATCH --job-name=planted_attr_round
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/scratch/sorkwos/slurm_logs/planted_attr_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/planted_attr_round_%j.err

# Section E (reviewer-driven): gradient attributions in the planted-signal
# test. COHORT / PLANTED / MODELS / ROUNDS_DIR / ROUND_START / ROUND_END
# passed via sbatch --export. MODELS must be a SINGLE model name (graph or
# abmil), never comma-joined — see planted_round.sh's header for why.
#
# Start the graph job for a planting FIRST — it creates planted_feature.npz
# within the first minute or two of starting (before any training), which the
# abmil job for the same planting also needs. Don't launch the abmil job
# until that file exists.
set -euo pipefail
: "${COHORT:?}"; : "${PLANTED:?}"; : "${MODELS:?}"; : "${ROUNDS_DIR:?}"
: "${ROUND_START:?}"; : "${ROUND_END:?}"
BATCH_SIZE="${BATCH_SIZE:-16}"

case "$MODELS" in
  *,*) echo "ABORT: MODELS='$MODELS' contains a comma — see header. Submit one job per model." >&2
       exit 1 ;;
esac

module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> planted_attr $COHORT/$PLANTED model=$MODELS rounds $ROUND_START-$ROUND_END batch=$BATCH_SIZE started on $(hostname) at $(date)"
echo ">>> Torch: $(python -c 'import torch; print(torch.__version__)')  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python analysis/planted_signal.py --cohort "$COHORT" --planted "$PLANTED" \
  --models "$MODELS" --attributions --rounds-dir "$ROUNDS_DIR" \
  --round-start "$ROUND_START" --round-end "$ROUND_END" \
  --batch-size "$BATCH_SIZE"

echo ">>> planted_attr $COHORT/$PLANTED model=$MODELS rounds $ROUND_START-$ROUND_END finished at $(date)"
