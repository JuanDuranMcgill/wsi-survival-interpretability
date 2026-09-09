#!/bin/bash
#SBATCH --job-name=ablation_brca
#SBATCH --output=/home/sorkwos/links/scratch/bootstrap_logs_BRCA/ablation_brca_%j.out
#SBATCH --error=/home/sorkwos/links/scratch/bootstrap_logs_BRCA/ablation_brca_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --account=def-senger

# Step 5a — Region ablation, BRCA cohort.
# Cluster: Trillium (SciNet). Do NOT use on Narval.
#
# Run from the repo root:
#   cd ~/wsi-survival-interpretability
#   sbatch slurm/trillium/step5a_region_ablation_brca.sh
#
# Prerequisites:
#   - The classwise embeddings must be present under
#     /home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/
#   - TCGA-CDR Excel must be at the path below.
#   - conch_env must include: torch, numpy, pandas, scipy, openpyxl

set -euo pipefail

REPO="$HOME/wsi-survival-interpretability"
BASE="/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA"
CDR_XLSX="/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
SAVE_TMP="/scratch/sorkwos/ablation_tmp_brca"
OUT="$REPO/results/region_ablation_brca.json"

mkdir -p "$(dirname "$OUT")"
mkdir -p "$SAVE_TMP"
mkdir -p /home/sorkwos/links/scratch/bootstrap_logs_BRCA

source /home/sorkwos/envs/conch_env/bin/activate

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
  --cdr-xlsx "$CDR_XLSX" \
  --rounds 20 \
  --epochs 15 \
  --batch-size 16 \
  --save-root "$SAVE_TMP" \
  --out "$OUT"

echo ">>> Done. Output: $OUT"
