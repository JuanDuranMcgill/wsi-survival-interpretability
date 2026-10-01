#!/bin/bash
#SBATCH --job-name=nested_aggregate
#SBATCH --output=/scratch/sorkwos/slurm_logs/nested_aggregate_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/nested_aggregate_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

python analysis/compartment_subsets.py --cohort brca --aggregate \
  --rounds-dir /scratch/sorkwos/nested_brca --out /scratch/sorkwos/nested_subsets_brca.json
