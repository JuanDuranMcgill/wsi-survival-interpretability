#!/bin/bash
#SBATCH --job-name=shapley_aggregate
#SBATCH --output=/scratch/sorkwos/slurm_logs/shapley_aggregate_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/shapley_aggregate_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

python analysis/region_shapley.py --cohort blca --aggregate \
  --rounds-dir /scratch/sorkwos/shapley_blca \
  --out /scratch/sorkwos/region_shapley_blca.json

python analysis/region_shapley.py --cohort brca --aggregate \
  --rounds-dir /scratch/sorkwos/shapley_brca \
  --out /scratch/sorkwos/region_shapley_brca.json
