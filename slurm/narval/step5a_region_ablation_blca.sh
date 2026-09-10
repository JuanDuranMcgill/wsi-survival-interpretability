#!/bin/bash
#SBATCH --job-name=ablation_blca
#SBATCH --account=def-senger_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sorkwos/slurm_logs/ablation_blca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ablation_blca_%j.err

# Step 5a — Region ablation, BLCA cohort.
# Cluster: Narval (Alliance Canada). Do NOT use on Trillium.
#
# Prerequisites:
#   1. Classwise embeddings transferred from Trillium via Globus:
#        /scratch/sorkwos/UNI2_classwise_embeddings/   (8 region subfolders, ~500 GB)
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
#   sbatch slurm/narval/step5a_region_ablation_blca.sh
#
# Output: results/region_ablation_blca.json (committed to git after the run)
# Temp checkpoints: /scratch/sorkwos/ablation_tmp_blca/ (cleaned up round by round)

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"
BASE="/scratch/sorkwos/UNI2_classwise_embeddings"
CDR_XLSX="/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx"
SAVE_TMP="/scratch/sorkwos/ablation_tmp_blca"
OUT="$REPO/results/region_ablation_blca.json"
# Partial-state file written after each completed round.
# If this file exists when the job starts, rounds already in it are skipped.
PARTIAL="$REPO/results/region_ablation_blca_partial.json"

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
    "Perivesical adipose tissue (extravesical fat)" \
    "Invasive urothelial carcinoma (tumor)" \
    "Normal urothelium (benign mucosa)" \
    "Inflammatory infiltrates (immune cells)" \
    "Lamina propria (fibrovascular stroma)" \
    "Blood vessels (vasculature)" \
    "Muscularis propria (detrusor muscle)" \
    "Necrosis"; do
    count=$(find "$BASE/$region" -name "*.pt" 2>/dev/null | wc -l)
    echo "  [$count .pt files] $region"
done
echo ""

python "$REPO/analysis/region_ablation.py" \
    --cohort blca \
    --class_dir "$BASE/Perivesical adipose tissue (extravesical fat)" \
    --class_dir "$BASE/Invasive urothelial carcinoma (tumor)" \
    --class_dir "$BASE/Normal urothelium (benign mucosa)" \
    --class_dir "$BASE/Inflammatory infiltrates (immune cells)" \
    --class_dir "$BASE/Lamina propria (fibrovascular stroma)" \
    --class_dir "$BASE/Blood vessels (vasculature)" \
    --class_dir "$BASE/Muscularis propria (detrusor muscle)" \
    --class_dir "$BASE/Necrosis" \
    --cdr-xlsx  "$CDR_XLSX" \
    --rounds    20 \
    --epochs    15 \
    --batch-size 4 \
    --save-root "$SAVE_TMP" \
    --out       "$OUT" \
    --resume    "$PARTIAL"

echo ">>> Done. Output written to $OUT"
