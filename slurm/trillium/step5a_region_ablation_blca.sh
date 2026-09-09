#!/bin/bash
#SBATCH --job-name=ablation_blca
#SBATCH --output=/home/sorkwos/links/scratch/bootstrap_logs/ablation_blca_%j.out
#SBATCH --error=/home/sorkwos/links/scratch/bootstrap_logs/ablation_blca_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --account=def-senger

# Step 5a — Region ablation, BLCA cohort.
# Cluster: Trillium (SciNet). Do NOT use on Narval.
#
# Run from the repo root:
#   cd ~/wsi-survival-interpretability
#   sbatch slurm/trillium/step5a_region_ablation_blca.sh
#
# Prerequisites:
#   - The classwise embeddings must be present under
#     /home/sorkwos/links/scratch/UNI2_classwise_embeddings/
#   - TCGA-CDR Excel must be at the path below.
#   - conch_env must include: torch, numpy, pandas, scipy, openpyxl

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"
BASE="/home/sorkwos/links/scratch/UNI2_classwise_embeddings"
CDR_XLSX="/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
SAVE_TMP="/scratch/sorkwos/ablation_tmp_blca"
OUT="$REPO/results/region_ablation_blca.json"

mkdir -p "$(dirname "$OUT")"
mkdir -p "$SAVE_TMP"
mkdir -p /home/sorkwos/links/scratch/bootstrap_logs

source /home/sorkwos/envs/conch_env/bin/activate

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
  --cdr-xlsx "$CDR_XLSX" \
  --rounds 20 \
  --epochs 15 \
  --batch-size 16 \
  --save-root "$SAVE_TMP" \
  --out "$OUT"

echo ">>> Done. Output: $OUT"
