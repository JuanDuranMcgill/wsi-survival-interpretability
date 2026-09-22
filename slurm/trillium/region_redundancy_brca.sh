#!/bin/bash
#SBATCH --job-name=region_redundancy_brca
#SBATCH --output=/scratch/sorkwos/slurm_logs/region_redundancy_brca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/region_redundancy_brca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00

set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> region_redundancy (BRCA) started on $(hostname) at $(date)"

python analysis/region_redundancy.py --cohort brca \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Adipose tissue (fat)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Blood vessels (vasculature)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Ductal carcinoma in situ (DCIS)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Fibrous desmoplastic stroma" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Invasive breast carcinoma (tumor cells)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Muscle tissue (smooth or skeletal muscle)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Necrosis or hemorrhage" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Normal breast glands and lobules (TDLU)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Tumor-infiltrating lymphocytes (immune infiltrates)" \
  --out /scratch/sorkwos/region_redundancy_brca.json

echo ">>> region_redundancy (BRCA) finished at $(date)"
