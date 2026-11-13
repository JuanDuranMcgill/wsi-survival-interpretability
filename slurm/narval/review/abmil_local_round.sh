#!/bin/bash
#SBATCH --job-name=abmil_local
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/scratch/sorkwos/slurm_logs/abmil_local_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/abmil_local_%j.err

# ABMIL rounds (run M settings, as EXP=M in review_round.sh) reading the embeddings
# from the node's local SSD instead of /scratch. When /scratch is slow, training
# reads it at ~70 MB/s and the GPU sits idle; copying the feature files once with
# 16 parallel streams takes minutes, and the epochs then read from local disk.
# Only the files the dataset loads are copied (*DX1*.pt, not *_A.pt), under the
# same names and region folders, so the patient order and the bootstrap splits are
# unchanged. $SLURM_TMPDIR is wiped when the job ends; /scratch is only read.
# The round checkpoints every epoch, so a job cut off by the time limit is resumed
# by resubmitting it (or by an afternotok copy of it).
#
#   sbatch --time=10:00:00 --export=ALL,COHORT=brca,ROUNDS_DIR=/scratch/$USER/abmil_v3_brca,ROUND_START=4,ROUND_END=4 \
#       slurm/narval/review/abmil_local_round.sh

set -euo pipefail
: "${COHORT:?}"; : "${ROUNDS_DIR:?}"; : "${ROUND_START:?}"; : "${ROUND_END:?}"
BATCH_SIZE="${BATCH_SIZE:-16}"

module load python/3.11 scipy-stack
source "$HOME/wsi-survival-interpretability/venv/bin/activate"
cd "$HOME/wsi-survival-interpretability"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> abmil_local $COHORT rounds $ROUND_START-$ROUND_END batch=$BATCH_SIZE started on $(hostname) at $(date)"
echo ">>> commit $(git rev-parse --short HEAD)"

# Region folders in the order the code uses (the first one defines the patient list).
mapfile -t SRC_DIRS < <(python -c "
import sys; sys.path.insert(0, 'analysis')
from compartment_common import default_class_dirs
print('\n'.join(default_class_dirs('$COHORT')))")
LOCAL="$SLURM_TMPDIR/emb"
CLASS_ARGS=()
t0=$(date +%s)
for src in "${SRC_DIRS[@]}"; do
    dst="$LOCAL/$(basename "$src")"
    mkdir -p "$dst"
    find "$src" -maxdepth 1 -name '*DX1*.pt' ! -name '*_A.pt' -print0 \
        | xargs -0 -P 16 -n 8 cp -t "$dst"
    n_src=$(find "$src" -maxdepth 1 -name '*DX1*.pt' ! -name '*_A.pt' | wc -l)
    n_dst=$(find "$dst" -maxdepth 1 -name '*.pt' | wc -l)
    b_src=$(find "$src" -maxdepth 1 -name '*DX1*.pt' ! -name '*_A.pt' -printf '%s\n' | awk '{s+=$1} END {print s}')
    b_dst=$(find "$dst" -maxdepth 1 -name '*.pt' -printf '%s\n' | awk '{s+=$1} END {print s}')
    if [ "$n_src" != "$n_dst" ] || [ "$b_src" != "$b_dst" ]; then
        echo "ABORT: copy of $(basename "$src") incomplete ($n_dst/$n_src files, $b_dst/$b_src bytes)" >&2; exit 1
    fi
    echo "    copied $(basename "$src"): $n_dst files, $((b_dst / 1000000000)) GB"
    CLASS_ARGS+=(--class_dir "$dst")
done
echo ">>> copied to $LOCAL in $(( $(date +%s) - t0 )) s"

python analysis/abmil_compartment_shapley.py --cohort "$COHORT" --rounds-dir "$ROUNDS_DIR" \
    --round-start "$ROUND_START" --round-end "$ROUND_END" --batch-size "$BATCH_SIZE" --last-epoch \
    "${CLASS_ARGS[@]}"

echo ">>> abmil_local $COHORT rounds $ROUND_START-$ROUND_END finished at $(date)"
