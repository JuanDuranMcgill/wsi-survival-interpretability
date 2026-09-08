#!/bin/bash
# submit_bootstrap.sh
# Submits 6 parallel SLURM jobs, each running 17 bootstrap rounds.
# Usage: bash submit_bootstrap.sh

SCRIPT_DIR="/home/sorkwos/links/scratch/multimodality/graph_subset_generation"
LOG_DIR="/home/sorkwos/links/scratch/bootstrap_logs"
BASE_SAVE="/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate"

mkdir -p "$LOG_DIR"

for JOB in 0 1 2 3 4 5; do
    SEED_OFFSET=$((JOB * 17))
    SAVE_ROOT="${BASE_SAVE}/job_${JOB}"
    JOB_SCRIPT="/tmp/bootstrap_job${JOB}.sh"

    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=bootstrap_job${JOB}
#SBATCH --output=${LOG_DIR}/job${JOB}_%j.out
#SBATCH --error=${LOG_DIR}/job${JOB}_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=24
#SBATCH --account=def-senger

source /home/sorkwos/envs/conch_env/bin/activate

python ${SCRIPT_DIR}/interpret_graphtrans_late_int_sparse_m3save.py \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis" \
  --bootstrap_rounds 17 \
  --epochs 30 \
  --save_root ${SAVE_ROOT} \
  --seed_offset ${SEED_OFFSET}
JOBEOF

    sbatch "$JOB_SCRIPT"
    echo "Submitted job ${JOB} (seed_offset=${SEED_OFFSET}, save_root=${SAVE_ROOT})"
done