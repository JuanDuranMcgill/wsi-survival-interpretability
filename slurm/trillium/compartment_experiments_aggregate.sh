#!/bin/bash
#SBATCH --job-name=compartment_aggregate
#SBATCH --output=/scratch/sorkwos/slurm_logs/compartment_aggregate_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/compartment_aggregate_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

set -euo pipefail
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

python analysis/abmil_compartment_shapley.py --cohort blca --aggregate \
  --rounds-dir /scratch/sorkwos/abmil_blca --out /scratch/sorkwos/abmil_shapley_blca.json
python analysis/abmil_compartment_shapley.py --cohort brca --aggregate \
  --rounds-dir /scratch/sorkwos/abmil_brca --out /scratch/sorkwos/abmil_shapley_brca.json

python analysis/compartment_subsets.py --cohort blca --aggregate \
  --rounds-dir /scratch/sorkwos/subsets_blca --out /scratch/sorkwos/compartment_subsets_blca.json
python analysis/compartment_subsets.py --cohort brca --aggregate \
  --rounds-dir /scratch/sorkwos/subsets_brca --out /scratch/sorkwos/compartment_subsets_brca.json
