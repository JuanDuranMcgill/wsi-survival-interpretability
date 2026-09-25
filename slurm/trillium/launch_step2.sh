#!/bin/bash
# One-off driver: submits all Step 2 (rounds 2-10) jobs for the three compartment
# experiments, in priority order B (subsets) -> A (ABMIL) -> C (planted signal).
# Round 1 for every piece was already run as Step 1; --rounds-dir values below
# match those Step 1 runs exactly so finished round-1 data is reused, not redone.
#
# Batch sizes deviate from the original suggestion wherever Step 1 timing showed
# a config/model would not fit its share of a 6h job:
#   - B/BRCA "full", "drop=Necrosis", "drop=Adipose" (9 and 8 of 9 regions) ran at
#     ~11345s (~3.15h) per round in Step 1 -> 1 round/job instead of 3.
#   - C/BRCA graph model ran at ~7178-9076s (~2-2.5h) per round in Step 1 ->
#     2 rounds/job instead of 3.
# Every other piece keeps the suggested batch size; Step 1 timing showed comfortable
# margin under 6h for all of them.
set -euo pipefail
cd /home/sorkwos/wsi-survival-interpretability
LOG=/scratch/sorkwos/slurm_logs
mkdir -p "$LOG"

sub() { sbatch --parsable "$@"; }

echo "=== B: compartment_subsets ==="

BLCA_CONFIGS=(
  "full"
  "keep=Invasive urothelial"
  "keep=Invasive urothelial+Perivesical adipose"
  "keep=Invasive urothelial+Perivesical adipose+Inflammatory"
  "keep=Invasive urothelial+Necrosis+Perivesical adipose"
  "keep=Blood vessels+Muscularis+Lamina"
  "drop=Necrosis"
  "drop=Perivesical adipose"
)
for cfg in "${BLCA_CONFIGS[@]}"; do
  for range in "2 6" "7 10"; do
    read -r rs re <<< "$range"
    jid=$(sub --export=ALL,COHORT=blca,ROUNDS_DIR=/scratch/sorkwos/subsets_blca,ROUND_START=$rs,ROUND_END=$re,CONFIG="$cfg" \
      slurm/trillium/compartment_subsets_round.sh)
    echo "  B/blca [$cfg] rounds $rs-$re -> job $jid"
  done
done

# BRCA: light configs (<=3 regions) get 3 rounds/job as suggested;
# heavy configs (full, drop=*, 8-9 regions) get 1 round/job (Step 1 showed ~3.15h/round).
BRCA_LIGHT_CONFIGS=(
  "keep=Adipose"
  "keep=Adipose+Muscle tissue"
  "keep=Adipose+Muscle tissue+Blood vessels"
  "keep=Invasive breast+Necrosis+Adipose"
  "keep=Normal breast+Necrosis+Ductal carcinoma"
)
BRCA_HEAVY_CONFIGS=(
  "full"
  "drop=Necrosis"
  "drop=Adipose"
)
for cfg in "${BRCA_LIGHT_CONFIGS[@]}"; do
  for range in "2 4" "5 7" "8 10"; do
    read -r rs re <<< "$range"
    jid=$(sub --export=ALL,COHORT=brca,ROUNDS_DIR=/scratch/sorkwos/subsets_brca,ROUND_START=$rs,ROUND_END=$re,CONFIG="$cfg" \
      slurm/trillium/compartment_subsets_round.sh)
    echo "  B/brca [$cfg] rounds $rs-$re -> job $jid"
  done
done
for cfg in "${BRCA_HEAVY_CONFIGS[@]}"; do
  for r in 2 3 4 5 6 7 8 9 10; do
    jid=$(sub --export=ALL,COHORT=brca,ROUNDS_DIR=/scratch/sorkwos/subsets_brca,ROUND_START=$r,ROUND_END=$r,CONFIG="$cfg" \
      slurm/trillium/compartment_subsets_round.sh)
    echo "  B/brca [$cfg] round $r -> job $jid  (heavy config: 1 round/job)"
  done
done

echo "=== A: abmil_compartment_shapley ==="

for range in "2 4" "5 7" "8 10"; do
  read -r rs re <<< "$range"
  jid=$(sub --export=ALL,COHORT=blca,ROUNDS_DIR=/scratch/sorkwos/abmil_blca,ROUND_START=$rs,ROUND_END=$re \
    slurm/trillium/abmil_shapley_round.sh)
  echo "  A/blca rounds $rs-$re -> job $jid"
done
for r in 2 3 4 5 6 7 8 9 10; do
  jid=$(sub --export=ALL,COHORT=brca,ROUNDS_DIR=/scratch/sorkwos/abmil_brca,ROUND_START=$r,ROUND_END=$r \
    slurm/trillium/abmil_shapley_round.sh)
  echo "  A/brca round $r -> job $jid"
done

echo "=== C: planted_signal ==="

for range in "2 4" "5 7" "8 10"; do
  read -r rs re <<< "$range"
  jid=$(sub --export=ALL,COHORT=blca,PLANTED=Necrosis,ROUNDS_DIR=/scratch/sorkwos/planted_blca_necrosis,ROUND_START=$rs,ROUND_END=$re,MODELS="graph,abmil" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/blca/Necrosis rounds $rs-$re -> job $jid"
  jid=$(sub --export=ALL,COHORT=blca,PLANTED=Lamina,ROUNDS_DIR=/scratch/sorkwos/planted_blca_lamina,ROUND_START=$rs,ROUND_END=$re,MODELS="graph,abmil" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/blca/Lamina rounds $rs-$re -> job $jid"
done

# BRCA: graph and abmil as separate jobs in the same --rounds-dir.
# graph: 2 rounds/job (Step 1: ~2-2.5h/round). abmil: 1 round/job (Step 1: >2.5h/round, unfinished).
for range in "2 3" "4 5" "6 7" "8 9" "10 10"; do
  read -r rs re <<< "$range"
  jid=$(sub --export=ALL,COHORT=brca,PLANTED=Necrosis,ROUNDS_DIR=/scratch/sorkwos/planted_brca_necrosis,ROUND_START=$rs,ROUND_END=$re,MODELS="graph" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/brca/Necrosis graph rounds $rs-$re -> job $jid"
  jid=$(sub --export=ALL,COHORT=brca,PLANTED=Fibrous,ROUNDS_DIR=/scratch/sorkwos/planted_brca_fibrous,ROUND_START=$rs,ROUND_END=$re,MODELS="graph" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/brca/Fibrous graph rounds $rs-$re -> job $jid"
done
for r in 2 3 4 5 6 7 8 9 10; do
  jid=$(sub --export=ALL,COHORT=brca,PLANTED=Necrosis,ROUNDS_DIR=/scratch/sorkwos/planted_brca_necrosis,ROUND_START=$r,ROUND_END=$r,MODELS="abmil" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/brca/Necrosis abmil round $r -> job $jid"
  jid=$(sub --export=ALL,COHORT=brca,PLANTED=Fibrous,ROUNDS_DIR=/scratch/sorkwos/planted_brca_fibrous,ROUND_START=$r,ROUND_END=$r,MODELS="abmil" \
    slurm/trillium/planted_signal_round.sh)
  echo "  C/brca/Fibrous abmil round $r -> job $jid"
done

echo "=== done submitting ==="
