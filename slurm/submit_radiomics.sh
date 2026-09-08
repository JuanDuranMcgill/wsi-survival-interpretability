#!/bin/bash
# submit_radiomics.sh
# Submits 4 parallel SLURM jobs, each processing 1/4 of the slides.
#
# Usage: bash submit_radiomics.sh

SCRIPT_DIR="/home/sorkwos/links/scratch/multimodality/graph_subset_generation"
LOG_DIR="/home/sorkwos/links/scratch/radiomics_logs"
mkdir -p $LOG_DIR

sbatch --job-name=radiomics_batch1 \
       --output=${LOG_DIR}/batch1_%j.out \
       --error=${LOG_DIR}/batch1_%j.err \
       --time=11:59:00 \
       --nodes=1 \
       --cpus-per-task=6 \
       --account=def-senger \
       --wrap="source /home/sorkwos/envs/conch_env/bin/activate && python ${SCRIPT_DIR}/find_pairs.py --run --batch 1 --total 4"
echo "Submitted batch 1"

sbatch --job-name=radiomics_batch2 \
       --output=${LOG_DIR}/batch2_%j.out \
       --error=${LOG_DIR}/batch2_%j.err \
       --time=11:59:00 \
       --nodes=1 \
       --cpus-per-task=6 \
       --account=def-senger \
       --wrap="source /home/sorkwos/envs/conch_env/bin/activate && python ${SCRIPT_DIR}/find_pairs.py --run --batch 2 --total 4"
echo "Submitted batch 2"

sbatch --job-name=radiomics_batch3 \
       --output=${LOG_DIR}/batch3_%j.out \
       --error=${LOG_DIR}/batch3_%j.err \
       --time=11:59:00 \
       --nodes=1 \
       --cpus-per-task=6 \
       --account=def-senger \
       --wrap="source /home/sorkwos/envs/conch_env/bin/activate && python ${SCRIPT_DIR}/find_pairs.py --run --batch 3 --total 4"
echo "Submitted batch 3"

sbatch --job-name=radiomics_batch4 \
       --output=${LOG_DIR}/batch4_%j.out \
       --error=${LOG_DIR}/batch4_%j.err \
       --time=11:59:00 \
       --nodes=1 \
       --cpus-per-task=6 \
       --account=def-senger \
       --wrap="source /home/sorkwos/envs/conch_env/bin/activate && python ${SCRIPT_DIR}/find_pairs.py --run --batch 4 --total 4"
echo "Submitted batch 4"