#!/bin/bash
# Aggregates the review-response runs K-N after slurm/launch_review_runs.sh step 2
# has finished, and copies their round files into results/ as in
# docs/COMPARTMENT_EXPERIMENTS_PLAN.md. Reads JSON only, so it runs in seconds on
# a login node:
#
#   bash slurm/aggregate_review_runs.sh narval        # or trillium
#
# Optional second argument restricts the experiments, e.g. "M N". Never copies
# round_*_local.npz, planted_feature.npz or synthetic_labels.npz (patient-level).
# Each summary's n_rounds says how many rounds it saw; check it against 10 (K, M,
# N) or 20 (L) before using the numbers.
set -euo pipefail
CLUSTER="${1:?usage: aggregate_review_runs.sh narval|trillium [\"K L M N\"]}"
EXPS="${2:-K L M N}"
cd "$(dirname "$0")/.."
if [ "$CLUSTER" = narval ]; then
  module load python/3.11 scipy-stack
  source "$HOME/wsi-survival-interpretability/venv/bin/activate"
else
  source /home/sorkwos/envs/conch_env/bin/activate
fi
SCR="${SCR:-/scratch/sorkwos}"

copy_rounds() {  # source dir, destination dir under results/
  mkdir -p "$2"
  cp "$1"/round_[0-9][0-9][0-9].json "$2"/
  echo "  copied $(ls "$2"/round_*.json | wc -l) round files to $2"
}

for exp in $EXPS; do
  for cohort in blca brca; do
    case "$exp" in
      K) python analysis/permuted_outcome.py --cohort "$cohort" --aggregate --rounds-dir "$SCR/permuted_$cohort" \
           --real-run "results/shapley_$cohort/region_shapley_$cohort.json" \
           --out "results/permuted_outcome_$cohort.json"
         copy_rounds "$SCR/permuted_$cohort" "results/permuted_outcome_$cohort" ;;
      L) python analysis/region_shapley.py --cohort "$cohort" --aggregate --rounds-dir "$SCR/shapley_last_$cohort" \
           --out "results/region_shapley_last_$cohort.json"
         copy_rounds "$SCR/shapley_last_$cohort" "results/shapley_last_$cohort" ;;
      M) python analysis/abmil_compartment_shapley.py --cohort "$cohort" --aggregate \
           --rounds-dir "$SCR/abmil_v2_$cohort" --out "results/abmil_shapley_v2_$cohort.json"
         copy_rounds "$SCR/abmil_v2_$cohort" "results/abmil_v2_$cohort" ;;
      N) if [ "$cohort" = blca ]; then plantings="Necrosis:necrosis Lamina:lamina"
         else plantings="Necrosis:necrosis Fibrous:fibrous"; fi
         for pl in $plantings; do
           planted=${pl%%:*}; slug=${pl##*:}; src=$SCR/planted_v3_${cohort}_$slug; dst=results/planted_v3_${cohort}_$slug
           python analysis/planted_signal.py --cohort "$cohort" --planted "$planted" --models graph,abmil \
             --aggregate --rounds-dir "$src" --out "$dst.json"
           for m in graph abmil; do copy_rounds "$src/$m" "$dst/$m"; done
           cp "$src/synthetic_meta.json" "$dst/"
           cp "$src"/summary_*.json "$dst/" 2>/dev/null || true
         done ;;
      *) echo "unknown experiment $exp" >&2; exit 1 ;;
    esac
  done
done
echo ">>> done. Copy results/ back (rsync or git) and commit."
