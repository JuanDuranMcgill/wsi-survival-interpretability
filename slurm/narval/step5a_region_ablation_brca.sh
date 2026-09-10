#!/bin/bash
#SBATCH --job-name=ablation_brca
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/ablation_brca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ablation_brca_%j.err

# Step 5a — Region ablation, BRCA cohort.
# Cluster: Narval (Alliance Canada). Do NOT use on Trillium.
#
# Prerequisites:
#   1. Classwise embeddings transferred from Trillium via Globus:
#        /scratch/sorkwos/UNI2_classwise_embeddings_BRCA/   (9 region subfolders)
#   2. CDR spreadsheet at:
#        /home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx
#   3. venv built and torch installed (done during Step 2 setup).
#      Verify: module load python/3.11 scipy-stack
#              source ~/wsi-survival-interpretability/venv/bin/activate
#              python -c "import torch; print(torch.cuda.is_available())"
#              → should print True on a GPU node.
#
# Submit from the repo root:
#   cd ~/wsi-survival-interpretability
#   sbatch slurm/narval/step5a_region_ablation_brca.sh
#
# Output: results/region_ablation_brca.json (committed to git after the run)
# Temp checkpoints: /scratch/sorkwos/ablation_tmp_brca/ (cleaned up round by round)
#
# Note: BRCA has ~1060 slides vs ~375 for BLCA, so this job is longer.
# 24h is a safe upper bound for 20 rounds × 15 epochs.

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"
BASE="/scratch/sorkwos/UNI2_classwise_embeddings_BRCA"
CDR_XLSX="/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx"
SAVE_TMP="/scratch/sorkwos/ablation_tmp_brca"
OUT="$REPO/results/region_ablation_brca.json"
# Partial-state file written after each completed round.
# If this file exists when the job starts, rounds already in it are skipped.
PARTIAL="$REPO/results/region_ablation_brca_partial.json"

mkdir -p /scratch/sorkwos/slurm_logs
mkdir -p "$SAVE_TMP"

module load python/3.11 scipy-stack
source "$REPO/venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sanity checks before committing GPU hours
echo ">>> Python: $(which python)"
echo ">>> Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ">>> CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ">>> GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo ">>> CDR file exists: $(test -f "$CDR_XLSX" && echo yes || echo NO — ABORT)"
for region in \
    "Adipose tissue (fat)" \
    "Blood vessels (vasculature)" \
    "Ductal carcinoma in situ (DCIS)" \
    "Fibrous desmoplastic stroma" \
    "Invasive breast carcinoma (tumor cells)" \
    "Muscle tissue (smooth or skeletal muscle)" \
    "Necrosis or hemorrhage" \
    "Normal breast glands and lobules (TDLU)" \
    "Tumor-infiltrating lymphocytes (immune infiltrates)"; do
    count=$(find "$BASE/$region" -name "*.pt" 2>/dev/null | wc -l)
    echo "  [$count .pt files] $region"
done
echo ""

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
    --cdr-xlsx  "$CDR_XLSX" \
    --rounds    20 \
    --epochs    15 \
    --batch-size 4 \
    --save-root "$SAVE_TMP" \
    --out       "$OUT" \
    --resume    "$PARTIAL"

echo ">>> Done. Output written to $OUT"
