# Compartment experiments: status and what's left

Companion to `docs/COMPARTMENT_EXPERIMENTS_PLAN.md`. Written after Step 1 (round 1
everywhere) plus a first Step 2 batch (rounds 2-10) run on Trillium. Committed
results reflect exactly what's described here as "done"; everything else in
this file is open and could be picked up on another cluster (e.g. Narval).

## Before running anything on a different cluster

Narval and Trillium do not share `/home` or `/scratch`. Two things must exist
on the new cluster before any of this can run there:

1. **The data**: the class-wise UNI-2 embeddings (`BLCA_BASE`/`BRCA_BASE` in
   `analysis/region_ablation.py`, currently
   `/home/sorkwos/links/scratch/UNI2_classwise_embeddings[_BRCA]` on Trillium)
   and the TCGA-CDR survival table (`CDR_DEFAULT` in
   `analysis/compartment_common.py`). Neither was verified to exist on Narval
   as of this writing — check first, or pass `--class_dir` / `--cdr-xlsx` to
   point at wherever they end up living there.
2. **The environment**: `conch_env` (or equivalent) with the same package set,
   rebuilt fresh on Narval — it is not shared with Trillium.

Anything produced on Narval lands in *Narval's* `/scratch`, not Trillium's.
Before a final aggregate, copy Narval's `round_*.json` files back into the
matching directory under Trillium's `/scratch/sorkwos/...` (or straight into
this repo's `results/...` tree) so one aggregate run sees all rounds together.
Never copy back `round_*_local.npz`, `planted_feature.npz`, or
`synthetic_labels.npz` — patient-level, not for git, and not needed for
aggregation.

## What's committed and done

- **A (ABMIL), both cohorts**: full 10/10 rounds, aggregated.
  `results/abmil_shapley_{blca,brca}.json` + `results/abmil_{blca,brca}/round_*.json`.
- **B (subsets)**, `full` and one `keep=` config per cohort (the two Step 1
  timing configs): full 10/10 rounds. All other 6 configs per cohort: 9/10
  rounds (rounds 2-10 only — see gap below). Aggregated regardless; the
  aggregate JSON's own `n_rounds`/`n_paired_rounds` fields say which configs
  are short a round.
  `results/compartment_subsets_{blca,brca}.json` + per-config round files under
  `results/compartment_subsets_{blca,brca}/<config-slug>/`.
- **C (planted signal)**: partial only, not aggregated yet (too incomplete to
  summarize meaningfully). Whatever rounds exist are committed under
  `results/planted_{cohort}_{planting}/{graph,abmil}/round_*.json`, plus each
  planting's `synthetic_meta.json`.

## Known issues

### 1. Bug: BLCA planted-signal ABMIL never ran (rounds 2-10)

The Step 2 launcher passed `--export=...,MODELS=graph,abmil,...` to `sbatch`
for the two BLCA planted jobs (Necrosis, Lamina). Slurm's `--export` parser
splits on commas to separate `KEY=VALUE` pairs, so `MODELS=graph,abmil` was
silently truncated to `MODELS=graph` — confirmed by the jobs' own printed
startup line (`models=graph`, not `models=graph,abmil`). Every BLCA planted
Step 2 job completed cleanly (exit 0, no errors) but only ever trained the
graph model. **ABMIL rounds 2-10 for both BLCA plantings do not exist and
never ran.** BRCA planted jobs were not affected — graph and abmil were
already separate jobs there, so no comma ever reached `--export`.

Fix for anywhere this gets resubmitted (Narval included): pass a single model
per job (`--models graph` or `--models abmil` in separate jobs, the same
pattern already used for BRCA), never a comma-joined value through
`sbatch --export`.

Needed:
```bash
python analysis/planted_signal.py --cohort blca --planted Necrosis --models abmil \
  --rounds-dir <dir>/planted_blca_necrosis --round-start 2 --round-end 10
python analysis/planted_signal.py --cohort blca --planted Lamina --models abmil \
  --rounds-dir <dir>/planted_blca_lamina --round-start 2 --round-end 10
```
(Split across multiple jobs/rounds as GPU availability allows — BLCA ABMIL ran
at ~2000-2100s/round in Step 1, so even all 9 rounds in one job is a safe fit
under a 6h wall.)

### 2. Gap: 12 of 16 subset configs never had a round-1

Step 1 only timed `full` + one `keep=` config per cohort; Step 2 assumed round
1 already existed for every config and only requested rounds 2-10. The other
6 BLCA configs and 6 BRCA configs are missing round 1 specifically (not a bug
— just never run). They currently report as 9/10 rounds each.

Needed, one round each:
```bash
# BLCA (6 configs)
python analysis/compartment_subsets.py --cohort blca --rounds-dir <dir>/subsets_blca \
  --round-start 1 --round-end 1 \
  --config "keep=Invasive urothelial+Perivesical adipose" \
  --config "keep=Invasive urothelial+Perivesical adipose+Inflammatory" \
  --config "keep=Invasive urothelial+Necrosis+Perivesical adipose" \
  --config "keep=Blood vessels+Muscularis+Lamina" \
  --config "drop=Necrosis" \
  --config "drop=Perivesical adipose"

# BRCA (6 configs)
python analysis/compartment_subsets.py --cohort brca --rounds-dir <dir>/subsets_brca \
  --round-start 1 --round-end 1 \
  --config "keep=Adipose+Muscle tissue" \
  --config "keep=Adipose+Muscle tissue+Blood vessels" \
  --config "keep=Invasive breast+Necrosis+Adipose" \
  --config "keep=Normal breast+Necrosis+Ductal carcinoma" \
  --config "drop=Necrosis" \
  --config "drop=Adipose"
```
(These can be one job each, or split — each config's round 1 was ~500-2000s in
the comparable configs already timed, well under 6h either way.)

### 3. Stalled in Trillium's queue (not failed, just not started)

As of this writing, 31 Step 2 jobs have been sitting `PENDING` with `TIME=0:00`
for 14+ hours — the whole `compute` partition has 750+ jobs queued across all
users, and this account's fair-share priority dropped after the initial batch.
No error, no timeout; Slurm gives no ETA. These are exactly the C jobs still
needed anyway, so re-running the same work fresh on Narval (rather than
waiting) covers this gap too. Trillium's own copies may still complete on
their own — check before merging to avoid double work.

Still needed (round coverage as of this writing, in addition to the ABMIL gap
above):
```bash
# BLCA planted, graph model (Necrosis has rounds 1-7, Lamina has rounds 1-4)
python analysis/planted_signal.py --cohort blca --planted Necrosis --models graph \
  --rounds-dir <dir>/planted_blca_necrosis --round-start 8 --round-end 10
python analysis/planted_signal.py --cohort blca --planted Lamina --models graph \
  --rounds-dir <dir>/planted_blca_lamina --round-start 5 --round-end 10

# BRCA planted, both models, both plantings (only round 1 exists for any of these)
python analysis/planted_signal.py --cohort brca --planted Necrosis --models graph \
  --rounds-dir <dir>/planted_brca_necrosis --round-start 2 --round-end 10
python analysis/planted_signal.py --cohort brca --planted Necrosis --models abmil \
  --rounds-dir <dir>/planted_brca_necrosis --round-start 2 --round-end 10
python analysis/planted_signal.py --cohort brca --planted Fibrous --models graph \
  --rounds-dir <dir>/planted_brca_fibrous --round-start 2 --round-end 10
python analysis/planted_signal.py --cohort brca --planted Fibrous --models abmil \
  --rounds-dir <dir>/planted_brca_fibrous --round-start 2 --round-end 10
```
Step 1 timing: BRCA planted graph ran ~2-2.5h/round, BRCA planted abmil ran
~1-3.4h/round (highly variable, likely node contention) — size batches
accordingly for whatever Narval's own wall-time limits look like; nothing here
has any inter-round dependency, so any split works and finished rounds are
always skipped on resubmit.

## After all of the above land

Re-run the aggregate commands from `docs/COMPARTMENT_EXPERIMENTS_PLAN.md` for
B (to pick up the now-complete round-1s) and C (first time, once every
planting has enough rounds), then commit the refreshed aggregate JSONs and any
new per-round files — same exclusions as always: no `*_local.npz`,
`planted_feature.npz`, or `synthetic_labels.npz`.

## Status update (2026-09-27) — Narval progress, and what's still open

Everything above was picked up on Narval starting 2026-09-25. Two things came
out of that run worth recording here:

- **Real per-round timing on Narval is much slower than this doc's original
  Trillium-based estimates** for the ABMIL model specifically: BLCA planted
  ABMIL ran at ~4h/round on Narval (vs. the ~2000-2100s/round quoted above),
  and BRCA planted ABMIL at ~6-8h/round (vs. ~1-3.4h/round). Whatever cluster
  picks up the remaining work, size job wall-time off these numbers, not the
  original Trillium ones.
- **New OOM failure mode**: the graph model's attention layers and the
  `drop=`-style subset configs (7 of 8 regions, close to the full model's
  footprint) can use nearly the entire 40GB card. Two BLCA planted-graph jobs
  and 2 BLCA subset `drop=` configs OOM'd at the default `--batch-size 16` on
  Narval; both `slurm/narval/compartment/planted_round.sh` and
  `slurm/trillium/planted_signal_round.sh` (also
  `compartment_subsets_round.sh`) now accept an optional `BATCH_SIZE`
  override (default 16) for exactly this — pass `BATCH_SIZE=4` if it recurs.

### Done (on Narval, committed to this repo's `results/` once copied back)

- BRCA subsets round-1 gap — all 6 configs (see issue #2 above).
- BLCA planted graph gap — Necrosis rounds 8-10, Lamina rounds 5-10 (issue #3).

### Still outstanding as of this writing

Queued and running on Narval right now, **and available to run on Trillium in
parallel if you start a session there** — whichever cluster gets to a given
round/config first wins; the other side's run on the same round is simply
wasted compute, not a correctness problem (every script skips rounds already
on disk). One job per round/config, matching the one-model-per-job
convention from issue #1:

```bash
# BLCA subsets round 1, the 2 configs that OOM'd (drop= configs, wider region set)
sbatch --export=COHORT=blca,ROUNDS_DIR=/scratch/sorkwos/subsets_blca,ROUND_START=1,ROUND_END=1,CONFIG="drop=Necrosis",BATCH_SIZE=4 \
  slurm/trillium/compartment_subsets_round.sh
sbatch --export=COHORT=blca,ROUNDS_DIR=/scratch/sorkwos/subsets_blca,ROUND_START=1,ROUND_END=1,CONFIG="drop=Perivesical adipose",BATCH_SIZE=4 \
  slurm/trillium/compartment_subsets_round.sh

# BLCA planted ABMIL, Necrosis — rounds 4-10 (rounds 1-3 already done)
for r in 4 5 6 7 8 9 10; do
  sbatch --export=COHORT=blca,PLANTED=Necrosis,MODELS=abmil,ROUNDS_DIR=/scratch/sorkwos/planted_blca_necrosis,ROUND_START=$r,ROUND_END=$r \
    slurm/trillium/planted_signal_round.sh
done

# BLCA planted ABMIL, Lamina — rounds 4-10 (rounds 1-3 already done)
for r in 4 5 6 7 8 9 10; do
  sbatch --export=COHORT=blca,PLANTED=Lamina,MODELS=abmil,ROUNDS_DIR=/scratch/sorkwos/planted_blca_lamina,ROUND_START=$r,ROUND_END=$r \
    slurm/trillium/planted_signal_round.sh
done

# BRCA planted graph — both plantings, round 10 only (rounds 1-9 already done)
sbatch --export=COHORT=brca,PLANTED=Necrosis,MODELS=graph,ROUNDS_DIR=/scratch/sorkwos/planted_brca_necrosis,ROUND_START=10,ROUND_END=10 \
  slurm/trillium/planted_signal_round.sh
sbatch --export=COHORT=brca,PLANTED=Fibrous,MODELS=graph,ROUNDS_DIR=/scratch/sorkwos/planted_brca_fibrous,ROUND_START=10,ROUND_END=10 \
  slurm/trillium/planted_signal_round.sh

# BRCA planted ABMIL, Necrosis — round 10 only (rounds 1-9 already done)
sbatch --export=COHORT=brca,PLANTED=Necrosis,MODELS=abmil,ROUNDS_DIR=/scratch/sorkwos/planted_brca_necrosis,ROUND_START=10,ROUND_END=10 \
  slurm/trillium/planted_signal_round.sh

# BRCA planted ABMIL, Fibrous — rounds 6 and 10 only (1-5, 7-9 already done)
sbatch --export=COHORT=brca,PLANTED=Fibrous,MODELS=abmil,ROUNDS_DIR=/scratch/sorkwos/planted_brca_fibrous,ROUND_START=6,ROUND_END=6 \
  slurm/trillium/planted_signal_round.sh
sbatch --export=COHORT=brca,PLANTED=Fibrous,MODELS=abmil,ROUNDS_DIR=/scratch/sorkwos/planted_brca_fibrous,ROUND_START=10,ROUND_END=10 \
  slurm/trillium/planted_signal_round.sh
```

If you decide to keep only one cluster running, cancel the other side's
matching jobs first (`squeue -u sorkwos` on that cluster) to avoid paying for
duplicate GPU-hours — but running both is safe, just wasteful.
