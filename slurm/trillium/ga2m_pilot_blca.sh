#!/bin/bash
#SBATCH --job-name=ga2m_pilot_blca
#SBATCH --output=/scratch/sorkwos/slurm_logs/ga2m_pilot_blca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/ga2m_pilot_blca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=16:00:00

set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> GA2M pilot (BLCA, 10 rounds) started on $(hostname) at $(date)"

python model/train_ga2m.py --cohort blca \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis" \
  --bootstrap_rounds 10 \
  --epochs 30 \
  --seed_offset 0 \
  --lam_int 1e-3 \
  --save_root /scratch/sorkwos/ga2m_pilot/blca

echo ">>> GA2M pilot (BLCA) finished at $(date)"
