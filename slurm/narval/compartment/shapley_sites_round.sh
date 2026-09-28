#!/bin/bash
#SBATCH --job-name=shapley_sites_round
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/scratch/sorkwos/slurm_logs/shapley_sites_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/shapley_sites_round_%j.err

# COHORT / ROUNDS_DIR / ROUND_START / ROUND_END passed via sbatch --export.
# Optional BATCH_SIZE override (default: script default of 16 — this is the
# same model/training as region_shapley.py's "full" run, which already
# completed clean at 16 for both cohorts, unlike region_shapley_graph.py).
set -euo pipefail
: "${COHORT:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"
BATCH_SIZE="${BATCH_SIZE:-16}"

module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> shapley_sites $COHORT rounds $ROUND_START-$ROUND_END batch=$BATCH_SIZE started on $(hostname) at $(date)"
echo ">>> Torch: $(python -c 'import torch; print(torch.__version__)')  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python analysis/region_shapley_sites.py --cohort "$COHORT" \
  --rounds-dir "$ROUNDS_DIR" --round-start "$ROUND_START" --round-end "$ROUND_END" \
  --batch-size "$BATCH_SIZE"

echo ">>> shapley_sites $COHORT rounds $ROUND_START-$ROUND_END finished at $(date)"
