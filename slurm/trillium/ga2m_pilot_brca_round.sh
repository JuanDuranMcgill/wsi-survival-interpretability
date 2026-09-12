#!/bin/bash
#SBATCH --job-name=ga2m_pilot_brca_r
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_pilot_brca_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_pilot_brca_round_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=03:00:00

# One round only. SEED_OFFSET is passed in via `sbatch --export=SEED_OFFSET=k`
# so global_round = SEED_OFFSET + 1 covers a distinct round per invocation.
# Each job gets its own save_root (job_<k>/) to avoid every job writing the
# same round_1.npz filename into a shared directory.

set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

: "${SEED_OFFSET:?SEED_OFFSET must be set via sbatch --export=SEED_OFFSET=k}"

echo ">>> GA2M pilot (BRCA, round seed_offset=${SEED_OFFSET}) started on $(hostname) at $(date)"

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
  --bootstrap_rounds 1 \
  --epochs 30 \
  --seed_offset "${SEED_OFFSET}" \
  --lam_int 1e-3 \
  --save_root "/scratch/sorkwos/ga2m_pilot/brca/job_${SEED_OFFSET}"

echo ">>> GA2M pilot (BRCA, seed_offset=${SEED_OFFSET}) finished at $(date)"
