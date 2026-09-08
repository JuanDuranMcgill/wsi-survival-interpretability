#!/bin/bash
# submit_bootstrap_BRCA.sh
# Submits 10 parallel SLURM jobs, each running 10 bootstrap rounds (30 epochs each).
# Total: 100 bootstrap rounds across the BRCA dataset (9 regions).
# Usage: bash submit_bootstrap_BRCA.sh

SCRIPT_DIR="/home/sorkwos/links/scratch/multimodality/graph_subset_generation"
LOG_DIR="/home/sorkwos/links/scratch/bootstrap_logs_BRCA"
BASE_SAVE="/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate_BRCA"

mkdir -p "$LOG_DIR"

for JOB in 0 1 2 3 4 5 6 7 8 9; do
    SEED_OFFSET=$((JOB * 10))
    SAVE_ROOT="${BASE_SAVE}/job_${JOB}"
    JOB_SCRIPT="/tmp/bootstrap_brca_job${JOB}.sh"

    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=brca_bootstrap_job${JOB}
#SBATCH --output=${LOG_DIR}/job${JOB}_%j.out
#SBATCH --error=${LOG_DIR}/job${JOB}_%j.err
#SBATCH --time=18:00:00
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
  --bootstrap_rounds 10 \
  --epochs 30 \
  --save_root ${SAVE_ROOT} \
  --seed_offset ${SEED_OFFSET}
JOBEOF

    sbatch "$JOB_SCRIPT"
    echo "Submitted job ${JOB} (seed_offset=${SEED_OFFSET}, save_root=${SAVE_ROOT})"
done