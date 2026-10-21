#!/bin/bash
#SBATCH --job-name=review_round
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/scratch/sorkwos/slurm_logs/review_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/review_round_%j.err

# Review-response runs K-N (docs/COMPARTMENT_EXPERIMENTS_PLAN.md). One round per
# job; walltime is set by slurm/launch_review_runs.sh with --time.
#
# EXP / COHORT / ROUNDS_DIR / ROUND_START / ROUND_END passed via sbatch --export.
# For EXP=N also PLANTED and MODELS. MODELS must be a SINGLE model name (graph or
# abmil), never comma-joined — Slurm's --export splits on commas (see
# planted_round.sh's header). Optional BATCH_SIZE (default 16; pass 8 or 4 if
# the graph model OOMs on the 40GB A100, as it did before).
#   K  permuted-outcome control         analysis/permuted_outcome.py
#   L  Shapley, also at the last epoch  analysis/region_shapley.py --last-epoch
#   M  ABMIL, area-normalised attention analysis/abmil_compartment_shapley.py --last-epoch
#   N  planted outcomes                 analysis/planted_signal.py --last-epoch --attributions
#   S  Shapley, absent compartments masked out of attention (branch masked-attention-shapley)
#                                       analysis/region_shapley.py --mask-absent --last-epoch
set -euo pipefail
: "${EXP:?}"; : "${COHORT:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"
BATCH_SIZE="${BATCH_SIZE:-16}"

module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> review $EXP $COHORT ${PLANTED:-} ${MODELS:-} rounds $ROUND_START-$ROUND_END batch=$BATCH_SIZE started on $(hostname) at $(date)"
echo ">>> commit $(git rev-parse --short HEAD)  Torch: $(python -c 'import torch; print(torch.__version__)')  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

COMMON=(--cohort "$COHORT" --rounds-dir "$ROUNDS_DIR" --round-start "$ROUND_START" --round-end "$ROUND_END"
        --batch-size "$BATCH_SIZE")
case "$EXP" in
  K) python analysis/permuted_outcome.py "${COMMON[@]}" ;;
  L) python analysis/region_shapley.py "${COMMON[@]}" --last-epoch ;;
  S) python analysis/region_shapley.py "${COMMON[@]}" --mask-absent --last-epoch ;;
  M) python analysis/abmil_compartment_shapley.py "${COMMON[@]}" --last-epoch ;;
  N) : "${PLANTED:?}"; : "${MODELS:?}"
     case "$MODELS" in
       *,*) echo "ABORT: MODELS='$MODELS' contains a comma. Submit one job per model." >&2; exit 1 ;;
     esac
     python analysis/planted_signal.py "${COMMON[@]}" --planted "$PLANTED" --models "$MODELS" \
       --last-epoch --attributions ;;
  *) echo "ABORT: unknown EXP='$EXP' (expected K, L, M, N or S)" >&2; exit 1 ;;
esac

echo ">>> review $EXP $COHORT ${PLANTED:-} ${MODELS:-} rounds $ROUND_START-$ROUND_END finished at $(date)"
