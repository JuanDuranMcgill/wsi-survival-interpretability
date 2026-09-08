#!/bin/bash
# submit_radiomics_BRCA.sh
# Submits 8 parallel SLURM jobs, each processing 1/8 of the BRCA slides (~133 each).
#
# Usage: bash submit_radiomics_BRCA.sh

SCRIPT_DIR="/home/sorkwos/links/scratch/multimodality/graph_subset_generation"
LOG_DIR="/home/sorkwos/links/scratch/radiomics_brca_logs"
mkdir -p $LOG_DIR

for i in 1 2 3 4 5 6 7 8; do
    sbatch --job-name=radiomics_brca${i} \
           --output=${LOG_DIR}/batch${i}_%j.out \
           --error=${LOG_DIR}/batch${i}_%j.err \
           --time=11:42:00 \
           --nodes=1 \
           --cpus-per-task=6 \
           --account=def-senger \
           --wrap="source /home/sorkwos/envs/conch_env/bin/activate && python ${SCRIPT_DIR}/find_pairs_BRCA.py --run --batch ${i} --total 8"
    echo "Submitted batch ${i}"
done