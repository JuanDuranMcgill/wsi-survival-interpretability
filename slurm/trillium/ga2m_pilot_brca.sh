#!/bin/bash
#SBATCH --job-name=ga2m_pilot_brca
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_pilot_brca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_pilot_brca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=18:00:00

set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> GA2M pilot (BRCA, 10 rounds) started on $(hostname) at $(date)"

python model/train_ga2m.py --cohort brca \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Adipose tissue (fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Ductal carcinoma in situ (DCIS)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Fibrous desmoplastic stroma" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Invasive breast carcinoma (tumor cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Muscle tissue (smooth or skeletal muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Necrosis or hemorrhage" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Normal breast glands and lobules (TDLU)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Tumor-infiltrating lymphocytes (immune infiltrates)" \
  --bootstrap_rounds 10 \
  --epochs 30 \
  --seed_offset 0 \
  --lam_int 1e-3 \
  --save_root /scratch/sorkwos/ga2m_pilot/brca

echo ">>> GA2M pilot (BRCA) finished at $(date)"
