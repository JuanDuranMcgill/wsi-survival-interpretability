# Compartment experiments: three runs in parallel

Three independent experiments. They share no outputs and can run at the same
time on separate GPUs. Every script writes one JSON per completed round, so jobs
can be split over round ranges, killed at walltime and resubmitted without
losing finished rounds. All three were tested end to end on a synthetic dataset
in the same file format.

Environment and data are as for `region_shapley.py` (default Trillium paths for
the class-wise UNI-2 embeddings and the TCGA-CDR table). Write to `/scratch`,
not `/home`.

## A. The audit on a standard model: `analysis/abmil_compartment_shapley.py`

Gated ABMIL on the same tiles, patients and out-of-bag splits. Removing a
compartment drops its tiles from the bag. Compares attention share (what a
heatmap shows) with exact Shapley values and deletion.

```bash
python analysis/abmil_compartment_shapley.py --cohort blca --rounds-dir /scratch/$USER/abmil_blca --round-start 1 --round-end 10
python analysis/abmil_compartment_shapley.py --cohort brca --rounds-dir /scratch/$USER/abmil_brca --round-start 1 --round-end 10
# after all rounds:
python analysis/abmil_compartment_shapley.py --cohort blca --aggregate --rounds-dir /scratch/$USER/abmil_blca --out results/abmil_shapley_blca.json
python analysis/abmil_compartment_shapley.py --cohort brca --aggregate --rounds-dir /scratch/$USER/abmil_brca --out results/abmil_shapley_brca.json
```

Stop conditions on round 1: `head_check_max_abs_diff` below 1e-3 (the script
raises otherwise), `efficiency_gap` below 1e-9, `empty_coalition_cindex` 0.5,
baseline concordance above 0.5.

## B. Which compartments a model needs: `analysis/compartment_subsets.py`

Retrains the region-fusion model on subsets of compartments, same out-of-bag
patients per round as the full model. Reports concordance at the selected epoch
and at the last epoch (no selection), and the paired difference from `full`.

Tier 1 first (per cohort, 10 rounds each). Quote the configs: names contain spaces.

BLCA (Shapley order: invasive UC, adipose, inflammatory, normal urothelium,
necrosis, lamina propria, muscularis, vessels):

```bash
python analysis/compartment_subsets.py --cohort blca --rounds-dir /scratch/$USER/subsets_blca --round-start 1 --round-end 10 \
  --config full \
  --config "keep=Invasive urothelial" \
  --config "keep=Invasive urothelial+Perivesical adipose" \
  --config "keep=Invasive urothelial+Perivesical adipose+Inflammatory" \
  --config "keep=Invasive urothelial+Necrosis+Perivesical adipose" \
  --config "keep=Blood vessels+Muscularis+Lamina" \
  --config "drop=Necrosis" \
  --config "drop=Perivesical adipose"
```

BRCA (Shapley order: adipose, muscle, vessels, TILs, invasive BC, stroma, DCIS,
necrosis, normal glands):

```bash
python analysis/compartment_subsets.py --cohort brca --rounds-dir /scratch/$USER/subsets_brca --round-start 1 --round-end 10 \
  --config full \
  --config "keep=Adipose" \
  --config "keep=Adipose+Muscle tissue" \
  --config "keep=Adipose+Muscle tissue+Blood vessels" \
  --config "keep=Invasive breast+Necrosis+Adipose" \
  --config "keep=Normal breast+Necrosis+Ductal carcinoma" \
  --config "drop=Necrosis" \
  --config "drop=Adipose"
```

These can be split across GPUs by giving each job a subset of `--config` flags
and/or of rounds; all jobs of a cohort share one `--rounds-dir`. Tier 2, if
time allows: `drop=` for each remaining compartment.

```bash
python analysis/compartment_subsets.py --cohort blca --aggregate --rounds-dir /scratch/$USER/subsets_blca --out results/compartment_subsets_blca.json
python analysis/compartment_subsets.py --cohort brca --aggregate --rounds-dir /scratch/$USER/subsets_brca --out results/compartment_subsets_brca.json
```

Reading: the top-k keep-sets are chosen from this cohort's Shapley ranking, so
they carry some selection optimism. The pre-specified tumour+necrosis+adipose
set and the bottom-three control do not. `last_epoch` is the unbiased
comparison.

## C. The audit where the answer is known: `analysis/planted_signal.py`

Replaces the outcome with a semi-synthetic one driven by one chosen compartment
(real embeddings and patients, synthetic labels; censored fraction matched to
the real cohort; true-risk concordance 0.75). Runs the region-fusion model and
ABMIL on it and asks whether weights / attention, deletion and Shapley rank the
planted compartment first.

Two plantings per cohort: necrosis (the compartment the weights over-rate) and
one the weights under-rate.

```bash
python analysis/planted_signal.py --cohort blca --planted Necrosis --rounds-dir /scratch/$USER/planted_blca_necrosis --round-start 1 --round-end 10
python analysis/planted_signal.py --cohort blca --planted Lamina    --rounds-dir /scratch/$USER/planted_blca_lamina   --round-start 1 --round-end 10
python analysis/planted_signal.py --cohort brca --planted Necrosis --rounds-dir /scratch/$USER/planted_brca_necrosis --round-start 1 --round-end 10
python analysis/planted_signal.py --cohort brca --planted Fibrous  --rounds-dir /scratch/$USER/planted_brca_fibrous  --round-start 1 --round-end 10
# after all rounds, for each:
python analysis/planted_signal.py --cohort blca --planted Necrosis --aggregate --rounds-dir /scratch/$USER/planted_blca_necrosis --out results/planted_blca_necrosis.json
```

The first call writes `synthetic_meta.json` with the oracle concordance and
event count; check it before the rounds run on. If a model's baseline stays
below about 0.58, the signal is too weak to learn at this event count: rerun
that planting in a fresh directory with `--censor-frac 0.5` and report both.

## What to commit

Commit: the aggregate JSONs under `results/`, the per-round `round_*.json`
files, and `synthetic_meta.json` / `summary_*.json` for the planted runs.

Do not commit: any `*_local.npz`, `planted_feature.npz` or
`synthetic_labels.npz`. These hold patient-level data.
