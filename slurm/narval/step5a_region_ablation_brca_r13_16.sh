#!/bin/bash
#SBATCH --job-name=ablation_brca_r13
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/ablation_brca_r13_16_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ablation_brca_r13_16_%j.err

# Step 5a — BRCA region ablation, rounds 7–12 (parallel shard).
# Run alongside r13_16 and r17_20; merge with analysis/merge_brca_ablation.py after all finish.

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"
BASE="/scratch/sorkwos/UNI2_classwise_embeddings_BRCA"
CDR_XLSX="/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx"
SAVE_TMP="/scratch/sorkwos/ablation_tmp_brca_r13"
OUT="$REPO/results/region_ablation_brca_partial_r13_16.json"

mkdir -p /scratch/sorkwos/slurm_logs "$SAVE_TMP"

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

python "$REPO/analysis/region_ablation.py" \
    --cohort brca \
    --class_dir "$BASE/Adipose tissue (fat)" \
    --class_dir "$BASE/Blood vessels (vasculature)" \
    --class_dir "$BASE/Ductal carcinoma in situ (DCIS)" \
    --class_dir "$BASE/Fibrous desmoplastic stroma" \
    --class_dir "$BASE/Invasive breast carcinoma (tumor cells)" \
    --class_dir "$BASE/Muscle tissue (smooth or skeletal muscle)" \
    --class_dir "$BASE/Necrosis or hemorrhage" \
    --class_dir "$BASE/Normal breast glands and lobules (TDLU)" \
    --class_dir "$BASE/Tumor-infiltrating lymphocytes (immune infiltrates)" \
    --cdr-xlsx    "$CDR_XLSX" \
    --rounds      20 \
    --round-start 13 \
    --round-end   16 \
    --epochs      15 \
    --batch-size  4 \
    --save-root   "$SAVE_TMP" \
    --out         "$OUT" \
    --resume      "$OUT"

echo ">>> Done. Output: $OUT"
