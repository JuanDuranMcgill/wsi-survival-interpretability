#!/bin/bash
# submit_bootstrap_BRCA_top_up.sh
#
# Submits 11 new SLURM jobs to complete the remaining 52 BRCA bootstrap rounds.
# - 10 jobs of 5 rounds each  (job_10 to job_19)
# -  1 job of 2 rounds        (job_20)
# Seeds 101–152 (non-overlapping with original seeds 1–100).
# Save roots: job_10 through job_20 (new folders, nothing existing is touched).
# Max epochs: 20 (covers real learning curve, faster than 30).
#
# Usage: bash submit_bootstrap_BRCA_top_up.sh

SCRIPT_DIR="/home/sorkwos/links/scratch/multimodality/graph_subset_generation"
LOG_DIR="/home/sorkwos/links/scratch/bootstrap_logs_BRCA"
BASE_SAVE="/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate_BRCA"

mkdir -p "$LOG_DIR"

# First 10 jobs: 5 rounds each
for IDX in 0 1 2 3 4 5 6 7 8 9; do
    JOB_ID=$((IDX + 10))          # job_10 through job_19
    SEED_OFFSET=$((100 + IDX * 5)) # 100, 105, 110, ..., 145
    N_ROUNDS=5
    SAVE_ROOT="${BASE_SAVE}/job_${JOB_ID}"
    JOB_SCRIPT="/tmp/brca_topup_job${JOB_ID}.sh"

    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=brca_topup_job${JOB_ID}
#SBATCH --output=${LOG_DIR}/topup_job${JOB_ID}_%j.out
#SBATCH --error=${LOG_DIR}/topup_job${JOB_ID}_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --account=def-senger

source /home/sorkwos/envs/conch_env/bin/activate

python ${SCRIPT_DIR}/interpret_graphtrans_late_int_sparse_m3save_BRCA.py \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Adipose tissue (fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Ductal carcinoma in situ (DCIS)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Fibrous desmoplastic stroma" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Invasive breast carcinoma (tumor cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Muscle tissue (smooth or skeletal muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Necrosis or hemorrhage" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Normal breast glands and lobules (TDLU)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Tumor-infiltrating lymphocytes (immune infiltrates)" \
  --bootstrap_rounds ${N_ROUNDS} \
  --epochs 20 \
  --save_root ${SAVE_ROOT} \
  --seed_offset ${SEED_OFFSET}
JOBEOF

    sbatch "$JOB_SCRIPT"
    echo "Submitted job_${JOB_ID} (seed_offset=${SEED_OFFSET}, rounds=${N_ROUNDS}, save_root=${SAVE_ROOT})"
done

# Last job: 2 rounds
JOB_ID=20
SEED_OFFSET=150
N_ROUNDS=2
SAVE_ROOT="${BASE_SAVE}/job_${JOB_ID}"
JOB_SCRIPT="/tmp/brca_topup_job${JOB_ID}.sh"

cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=brca_topup_job${JOB_ID}
#SBATCH --output=${LOG_DIR}/topup_job${JOB_ID}_%j.out
#SBATCH --error=${LOG_DIR}/topup_job${JOB_ID}_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --account=def-senger

source /home/sorkwos/envs/conch_env/bin/activate

python ${SCRIPT_DIR}/interpret_graphtrans_late_int_sparse_m3save_BRCA.py \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Adipose tissue (fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Ductal carcinoma in situ (DCIS)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Fibrous desmoplastic stroma" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Invasive breast carcinoma (tumor cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Muscle tissue (smooth or skeletal muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Necrosis or hemorrhage" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Normal breast glands and lobules (TDLU)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA/Tumor-infiltrating lymphocytes (immune infiltrates)" \
  --bootstrap_rounds ${N_ROUNDS} \
  --epochs 20 \
  --save_root ${SAVE_ROOT} \
  --seed_offset ${SEED_OFFSET}
JOBEOF

sbatch "$JOB_SCRIPT"
echo "Submitted job_${JOB_ID} (seed_offset=${SEED_OFFSET}, rounds=${N_ROUNDS}, save_root=${SAVE_ROOT})"

echo ""
echo "=== All 11 top-up jobs submitted ==="
echo "    job_10 to job_19 : 5 rounds each (seeds 101-150)"
echo "    job_20            : 2 rounds      (seeds 151-152)"
echo "    Total new rounds  : 52"
echo "    Expected total    : 48 (existing complete) + 52 (new) = 100"