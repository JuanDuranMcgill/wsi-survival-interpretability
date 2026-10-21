#!/bin/bash
# Submits the review-response runs K-N (docs/COMPARTMENT_EXPERIMENTS_PLAN.md) on
# Narval or Trillium, one round per job, in two steps as before:
#
#   bash slurm/launch_review_runs.sh narval 1     # round 1 of everything (timing + sanity)
#   bash slurm/launch_review_runs.sh narval 2     # every remaining round
#   bash slurm/launch_review_runs.sh narval all   # both at once, unattended
#
# Optional third argument restricts the experiments, e.g. "M N".
#
# Step 1 submits round 1 of K, L and M, and round 1 of the graph model for each
# planting, which writes that planting's planted_feature.npz before any training.
# Check the round-1 files before step 2 (stop conditions in the plan: baseline
# above 0.5, efficiency gap below 1e-9, head check below 1e-3; K's baseline near
# 0.5). Step 2 submits the rest, including every ABMIL planted round, and skips any
# piece whose round 1 is missing.
#
# "all" submits round 1 and the remaining rounds together; the remaining rounds of
# each piece wait on its round 1 with --dependency=afterok, so they start only if
# round 1 exits cleanly (and are cancelled if it fails). It cannot check the stop
# conditions, so check the round-1 files when you are back; use it once round 1 has
# already passed for this code on some cluster.
#
# Patient order is sorted since 2026-09-28, so these runs draw the same splits on
# either cluster and can be pooled, but they do not pair with rounds from runs
# before that date. All output goes to fresh directories for that reason.
#
# Walltimes are per round, sized from the Narval timings in
# docs/NARVAL_ROUND_TIME_BUDGETS.txt (ABMIL ~4h/round BLCA, 6-8h BRCA) with room
# for the extra last-epoch audit. Revise them from the step-1 "seconds" field.
set -euo pipefail
CLUSTER="${1:?usage: launch_review_runs.sh narval|trillium 1|2|all [\"K L M N\"]}"
STEP="${2:?usage: launch_review_runs.sh narval|trillium 1|2|all [\"K L M N\"]}"
EXPS="${3:-K L M N}"
case "$CLUSTER" in narval|trillium) ;; *) echo "cluster must be narval or trillium" >&2; exit 1 ;; esac
case "$STEP" in 1|2|all) ;; *) echo "step must be 1, 2 or all" >&2; exit 1 ;; esac

cd "$(dirname "$0")/.."
SCRIPT="slurm/$CLUSTER/review/review_round.sh"
SCR="${SCR:-/scratch/sorkwos}"
mkdir -p "$SCR/slurm_logs"

# refuse to run code without the where-present attention fix
grep -q "attention_share_area_norm_where_present" analysis/abmil_compartment_shapley.py || {
  echo "ABORT: this checkout lacks the where-present attention fix; git pull first." >&2; exit 1; }
echo ">>> $CLUSTER step $STEP, experiments: $EXPS, commit $(git rev-parse --short HEAD)"

walltime() {  # experiment, cohort, model
  case "$CLUSTER/$1/$2/$3" in
    # Narval: sized from docs/NARVAL_ROUND_TIME_BUDGETS.txt (ABMIL ~4h/round BLCA,
    # 6-8h BRCA); Narval is several times slower than Trillium for these jobs
    narval/K/blca/*) echo 06:00:00 ;;  narval/K/brca/*) echo 10:00:00 ;;
    narval/L/blca/*) echo 08:00:00 ;;  narval/L/brca/*) echo 12:00:00 ;;
    narval/M/blca/*) echo 12:00:00 ;;  narval/M/brca/*) echo 16:00:00 ;;
    narval/N/blca/graph) echo 08:00:00 ;;  narval/N/brca/graph) echo 12:00:00 ;;
    narval/N/blca/abmil) echo 12:00:00 ;;  narval/N/brca/abmil) echo 16:00:00 ;;
  esac
  case "$CLUSTER/$1/$2/$3" in narval/*) return ;; esac
  case "$1/$2/$3" in
    # Trillium: revised 2026-10-20 from step-1 seconds (~2-3x observed):
    # BLCA graph ~28 min, BRCA graph ~55 min, M blca 38 min, M brca 85 min;
    # 81 step-2 rounds all finished under 40% of these
    K/blca/*) echo 01:30:00 ;;  K/brca/*) echo 02:30:00 ;;
    L/blca/*) echo 01:30:00 ;;  L/brca/*) echo 02:30:00 ;;
    M/blca/*) echo 02:00:00 ;;  M/brca/*) echo 04:00:00 ;;
    N/blca/graph) echo 01:30:00 ;;  N/brca/graph) echo 02:30:00 ;;
    N/blca/abmil) echo 02:00:00 ;;  N/brca/abmil) echo 04:00:00 ;;
  esac
}

# submit EXP COHORT DIR ROUND [PLANTED MODELS]
submit() {
  local exp=$1 cohort=$2 dir=$3 r=$4 planted=${5:-} models=${6:-}
  local exports="ALL,EXP=$exp,COHORT=$cohort,ROUNDS_DIR=$dir,ROUND_START=$r,ROUND_END=$r"
  [ -n "$planted" ] && exports="$exports,PLANTED=$planted,MODELS=$models"
  [ -n "${BATCH_SIZE:-}" ] && exports="$exports,BATCH_SIZE=$BATCH_SIZE"
  local name="rv_${exp}_${cohort}${planted:+_$planted}${models:+_$models}_r$r"
  local dep=()
  [ -n "${DEP:-}" ] && dep=(--dependency="afterok:$DEP" --kill-on-invalid-dep=yes)
  local jid
  jid=$(sbatch --parsable --job-name="$name" --time="$(walltime "$exp" "$cohort" "${models:-x}")" \
        "${dep[@]}" --export="$exports" "$SCRIPT")
  LAST_JID=$jid
  echo "  $name -> job $jid${DEP:+ (after $DEP)}"
}

# step 1 must start from empty directories: rounds written by older code lack the
# new fields, and the aggregate would silently drop them
fresh() {
  if [ "$STEP" != 2 ] && compgen -G "$1/round_*.json" > /dev/null && [ -z "${FORCE:-}" ]; then
    echo "  SKIP $1: already has round files (move them aside, or FORCE=1)" >&2; return 1
  fi
}
have_round1() {
  [ -f "$1/round_001.json" ] || { echo "  SKIP $1: no round_001.json yet (run step 1 first)" >&2; return 1; }
}
rounds() { if [ "$STEP" = 1 ]; then echo 1; else seq 2 "$1"; fi; }

for exp in $EXPS; do
  case "$exp" in
    K|L|M)
      for cohort in blca brca; do
        case "$exp" in K) dir=$SCR/permuted_$cohort; n=10 ;;
                       L) dir=$SCR/shapley_last_$cohort; n=20 ;;
                       M) dir=$SCR/abmil_v2_$cohort; n=10 ;; esac
        echo "=== $exp $cohort ($dir)"
        if [ "$STEP" = all ]; then
          fresh "$dir" || continue
          DEP=; submit "$exp" "$cohort" "$dir" 1
          DEP=$LAST_JID; for r in $(seq 2 $n); do submit "$exp" "$cohort" "$dir" "$r"; done; DEP=
          continue
        fi
        if [ "$STEP" = 1 ]; then fresh "$dir" || continue; else have_round1 "$dir" || continue; fi
        for r in $(rounds $n); do submit "$exp" "$cohort" "$dir" "$r"; done
      done ;;
    N)
      for p in "blca Necrosis necrosis" "blca Lamina lamina" "brca Necrosis necrosis" "brca Fibrous fibrous"; do
        read -r cohort planted slug <<< "$p"
        dir=$SCR/planted_v3_${cohort}_$slug
        echo "=== N $cohort $planted ($dir)"
        if [ "$STEP" = all ]; then
          fresh "$dir/graph" || continue
          DEP=; submit N "$cohort" "$dir" 1 "$planted" graph
          DEP=$LAST_JID
          for r in $(seq 2 10); do submit N "$cohort" "$dir" "$r" "$planted" graph; done
          for r in $(seq 1 10); do submit N "$cohort" "$dir" "$r" "$planted" abmil; done
          DEP=
        elif [ "$STEP" = 1 ]; then
          fresh "$dir/graph" || continue
          submit N "$cohort" "$dir" 1 "$planted" graph
        else
          have_round1 "$dir/graph" || continue
          [ -f "$dir/planted_feature.npz" ] || { echo "  SKIP: no planted_feature.npz" >&2; continue; }
          for r in $(seq 2 10); do submit N "$cohort" "$dir" "$r" "$planted" graph; done
          for r in $(seq 1 10); do submit N "$cohort" "$dir" "$r" "$planted" abmil; done
        fi
      done ;;
    *) echo "unknown experiment $exp" >&2; exit 1 ;;
  esac
done
echo ">>> done submitting. squeue -u sorkwos to watch."
