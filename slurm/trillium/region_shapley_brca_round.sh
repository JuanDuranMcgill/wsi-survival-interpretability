#!/bin/bash
#SBATCH --job-name=shapley_brca_r
#SBATCH --output=/scratch/sorkwos/slurm_logs/shapley_brca_round_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/shapley_brca_round_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=06:00:00

# ROUND_START / ROUND_END passed via sbatch --export=ROUND_START=k,ROUND_END=j
# Walltime is more conservative than BLCA's: no direct BRCA timing yet, and
# BRCA has ~2.8x the slides (1060 vs 379), so rounds are sized smaller (4
# rather than 5) with extra buffer per job.
set -euo pipefail
: "${ROUND_START:?ROUND_START must be set via sbatch --export}"
: "${ROUND_END:?ROUND_END must be set via sbatch --export}"
source /home/sorkwos/envs/conch_env/bin/activate
cd /home/sorkwos/wsi-survival-interpretability

echo ">>> region_shapley BRCA rounds ${ROUND_START}-${ROUND_END} started on $(hostname) at $(date)"

python analysis/region_shapley.py --cohort brca \
  --rounds-dir /scratch/sorkwos/shapley_brca \
  --round-start "${ROUND_START}" --round-end "${ROUND_END}"

echo ">>> region_shapley BRCA rounds ${ROUND_START}-${ROUND_END} finished at $(date)"
