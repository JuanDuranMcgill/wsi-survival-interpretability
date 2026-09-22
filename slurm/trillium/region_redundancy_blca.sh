#!/bin/bash
#SBATCH --job-name=region_redundancy_blca
#SBATCH --output=/scratch/sorkwos/slurm_logs/region_redundancy_blca_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/region_redundancy_blca_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> region_redundancy (BLCA) started on $(hostname) at $(date)"

python analysis/region_redundancy.py --cohort blca \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)" \
  --class-dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis" \
  --out /scratch/sorkwos/region_redundancy_blca.json

echo ">>> region_redundancy (BLCA) finished at $(date)"
