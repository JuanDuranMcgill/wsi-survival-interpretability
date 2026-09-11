#!/bin/bash
#SBATCH --job-name=step1_patient_error
#SBATCH --output=/scratch/sorkwos/slurm_logs/step1_patient_error_%j.out
#SBATCH --error=/scratch/sorkwos/slurm_logs/step1_patient_error_%j.err
#SBATCH --account=def-senger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00

set -euo pipefail

source /home/sorkwos/envs/conch_env/bin/activate

cd /home/sorkwos/wsi-survival-interpretability

echo ">>> Step 1 (BLCA) started on $(hostname) at $(date)"
python analysis/patient_error.py --cohort blca \
  --final-npz /scratch/sorkwos/med3pa_bootstrap/final_med3pa_input.npz \
  --cdr-xlsx  /home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx \
  --out results/patient_error_blca.json

echo ">>> Step 1 (BRCA) started at $(date)"
python analysis/patient_error.py --cohort brca \
  --final-npz /scratch/sorkwos/med3pa_bootstrap_intermediate_BRCA/final_med3pa_input.npz \
  --cdr-xlsx  /home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx \
  --out results/patient_error_brca.json

echo ">>> Step 1 finished at $(date)"
