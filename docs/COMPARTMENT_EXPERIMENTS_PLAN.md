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

## D. Shapley with the graph diffusion on: `analysis/region_shapley_graph.py`

The main model uses the k-NN graph diffusion in training only; every risk score
in the paper, and every run of `region_ablation.py` / `region_shapley.py` and
the experiments above, is computed without it. This reruns the Shapley analysis
with the diffusion used in training *and* evaluation, to test whether it
changes which compartments the model relies on. Same protocol as
`region_shapley.py` otherwise; 20 rounds per cohort; run all rounds of a cohort
on one cluster.

Needs the precomputed graph files: `<slide>_A.pt` beside each slide's embedding
file, or a root passed as `--adj-dir` with one subdirectory per compartment
(named like the embedding directories). A round stops if fewer than 90% of
compartment-slides with at least 10 tiles have a usable graph.

```bash
python analysis/region_shapley_graph.py --cohort blca --rounds-dir /scratch/$USER/shapley_graph_blca --round-start 1 --round-end 20
python analysis/region_shapley_graph.py --cohort brca --rounds-dir /scratch/$USER/shapley_graph_brca --round-start 1 --round-end 20
python analysis/region_shapley_graph.py --cohort blca --aggregate --rounds-dir /scratch/$USER/shapley_graph_blca --out results/region_shapley_graph_blca.json
python analysis/region_shapley_graph.py --cohort brca --aggregate --rounds-dir /scratch/$USER/shapley_graph_brca --out results/region_shapley_graph_brca.json
```

Commit the aggregate JSONs and the `round_*.json` files (copy them to
`results/shapley_graph_{cohort}/`). This script writes no patient-level files.

## Reviewer-driven runs (2026-09-28)

All scripts below were tested end to end on synthetic data. The dataset now
sorts slides by name (`region_ablation.MultiRegionDataset`), so patient order
and every split are the same on every cluster from here on; runs before this
date used filesystem order. Run every experiment's rounds on one cluster.

### E. Gradient attributions in the planted-signal test

`planted_signal.py --attributions` adds gradient x input and integrated
gradients per compartment to every round, for both models, so the planted test
compares weights / attention, deletion, Shapley and two gradient methods on the
same models. Fresh directories (do not reuse the earlier planted runs):

```bash
for P in "blca Necrosis necrosis" "blca Lamina lamina" "brca Necrosis necrosis" "brca Fibrous fibrous"; do set -- $P
  python analysis/planted_signal.py --cohort $1 --planted $2 --models graph --attributions \
    --rounds-dir /scratch/$USER/planted_attr_$1_$3 --round-start 1 --round-end 10
  python analysis/planted_signal.py --cohort $1 --planted $2 --models abmil --attributions \
    --rounds-dir /scratch/$USER/planted_attr_$1_$3 --round-start 1 --round-end 10
done
# one model per job; after all rounds, per planting:
python analysis/planted_signal.py --cohort blca --planted Necrosis --models graph,abmil --aggregate \
  --rounds-dir /scratch/$USER/planted_attr_blca_necrosis --out results/planted_attr_blca_necrosis.json
```
Start one graph job per planting first and launch the rest once
`planted_feature.npz` exists in that directory. Check on round 1:
`ig_completeness_gap_in_risk_sd` (graph) below ~0.05 and
`ig_completeness_gap_max_abs` (ABMIL) small.

### F. Shapley with whole sites held out: `analysis/region_shapley_sites.py`

20 rounds per cohort; each round holds out whole TCGA source sites (>= 20% of
patients, >= 20 events) and trains on the rest. Held-out sites are recorded in
`split_XXX.json`.

```bash
python analysis/region_shapley_sites.py --cohort blca --rounds-dir /scratch/$USER/shapley_sites_blca --round-start 1 --round-end 20
python analysis/region_shapley_sites.py --cohort brca --rounds-dir /scratch/$USER/shapley_sites_brca --round-start 1 --round-end 20
python analysis/region_shapley_sites.py --cohort blca --aggregate --rounds-dir /scratch/$USER/shapley_sites_blca --out results/region_shapley_sites_blca.json
python analysis/region_shapley_sites.py --cohort brca --aggregate --rounds-dir /scratch/$USER/shapley_sites_brca --out results/region_shapley_sites_brca.json
```
Commit the aggregates, `round_*.json` and `split_*.json`, not the
`*_local.npz`.

### G. Nested subset selection: `analysis/nested_subsets.py`

10 rounds per cohort. Each round ranks compartments by Shapley value on an
inner hold-out of its own in-bag patients, then trains full, top-1, top-2 and
top-3 on the in-bag set and evaluates on out-of-bag patients. About five
trainings per round.

```bash
python analysis/nested_subsets.py --cohort blca --rounds-dir /scratch/$USER/nested_blca --round-start 1 --round-end 10
python analysis/nested_subsets.py --cohort brca --rounds-dir /scratch/$USER/nested_brca --round-start 1 --round-end 10
python analysis/compartment_subsets.py --cohort blca --aggregate --rounds-dir /scratch/$USER/nested_blca --out results/nested_subsets_blca.json
python analysis/compartment_subsets.py --cohort brca --aggregate --rounds-dir /scratch/$USER/nested_brca --out results/nested_subsets_brca.json
```
Commit the aggregates, every config's `round_*.json` and `selection/round_*.json`.

### H. Tiles for pathologist labelling: `preprocessing/export_label_tiles.py`

CPU only; needs the classifier JSONL and the WSIs (OpenSlide), so it runs where
they live. 25 tiles per predicted class per cohort (200 BLCA, 225 BRCA).

```bash
python preprocessing/export_label_tiles.py --cohort blca --jsonl-dir /home/sorkwos/links/scratch/blca_jsons \
  --svs-root /home/sorkwos/links/scratch/TCGA-BLCA-p2 --svs-root /home/sorkwos/links/scratch/TCGA-BLCA/WSI/output_folder1 \
  --patients-file results/modelled_patients_blca.txt --out /scratch/$USER/label_tiles_blca
python preprocessing/export_label_tiles.py --cohort brca --jsonl-dir /home/sorkwos/links/scratch/brca_jsons \
  --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-1 --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-2 \
  --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-3 --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-4 \
  --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-5 --svs-root /home/sorkwos/links/scratch/TCGA-BRCA-new \
  --patients-file results/modelled_patients_brca.txt --out /scratch/$USER/label_tiles_brca
```
Do not commit the tiles, the sheets or `key.csv`. Send Juan the two
`labelling_sheet.xlsx` files (with their `tiles/` folders); keep `key.csv`
until the pathologists return their sheets, then run
`analysis/label_agreement.py`.

### I. External cohort (CPTAC-BRCA): plan only

Not run. Needs, in order: CPTAC-BRCA diagnostic slides and matching outcome
data from the CPTAC data portals; tiling, CONCH classification with the same
nine classes and prompt, and UNI-2 embedding with the existing preprocessing
scripts; a PFI-equivalent endpoint, which CPTAC may not provide at the same
definition; then the region-fusion model trained on all of TCGA-BRCA and
evaluated on CPTAC, with the Shapley analysis on the CPTAC patients. Weeks of
data work before any GPU time; treat as future work unless the data are already
available.

## Review-response runs (2026-10-07)

All tested end to end on synthetic data. Fresh directories only; run every
experiment's rounds on one cluster. Slides are sorted, so splits are
reproducible. Commit the aggregates and every `round_*.json`; never commit
`*_local.npz`, `planted_feature.npz` or `synthetic_labels.npz`.

Priority order: K, L, M, N.

### K. Permuted-outcome control: `analysis/permuted_outcome.py`

Retrains the region-fusion model on outcomes shuffled across patients (a fresh
shuffle per round), to test whether the fusion-weight ordering depends on the
outcome at all. 10 rounds per cohort.

```bash
python analysis/permuted_outcome.py --cohort blca --rounds-dir /scratch/$USER/permuted_blca --round-start 1 --round-end 10
python analysis/permuted_outcome.py --cohort brca --rounds-dir /scratch/$USER/permuted_brca --round-start 1 --round-end 10
python analysis/permuted_outcome.py --cohort blca --aggregate --rounds-dir /scratch/$USER/permuted_blca \
  --real-run results/shapley_blca/region_shapley_blca.json --out results/permuted_outcome_blca.json
python analysis/permuted_outcome.py --cohort brca --aggregate --rounds-dir /scratch/$USER/permuted_brca \
  --real-run results/shapley_brca/region_shapley_brca.json --out results/permuted_outcome_brca.json
```
Expected: baseline concordance near 0.5. Copy round files to `results/permuted_outcome_{cohort}/`.

### L. Shapley at the last epoch: `analysis/region_shapley.py --last-epoch`

The original Shapley run selects each round's epoch on the out-of-bag patients.
This rerun records the same quantities for the last epoch too. 20 rounds per cohort.

```bash
python analysis/region_shapley.py --cohort blca --last-epoch --rounds-dir /scratch/$USER/shapley_last_blca --round-start 1 --round-end 20
python analysis/region_shapley.py --cohort brca --last-epoch --rounds-dir /scratch/$USER/shapley_last_brca --round-start 1 --round-end 20
python analysis/region_shapley.py --cohort blca --aggregate --rounds-dir /scratch/$USER/shapley_last_blca --out results/region_shapley_last_blca.json
python analysis/region_shapley.py --cohort brca --aggregate --rounds-dir /scratch/$USER/shapley_last_brca --out results/region_shapley_last_brca.json
```
Copy round JSONs (not `*_local.npz`) to `results/shapley_last_{cohort}/`.

### M. ABMIL with area-normalised attention and last epoch: `abmil_compartment_shapley.py --last-epoch`

Records attention per unit of tissue (attention share divided by tile share,
renormalised per patient) and the tile share itself, at the selected and the
last epoch. 10 rounds per cohort.

```bash
python analysis/abmil_compartment_shapley.py --cohort blca --last-epoch --rounds-dir /scratch/$USER/abmil_v2_blca --round-start 1 --round-end 10
python analysis/abmil_compartment_shapley.py --cohort brca --last-epoch --rounds-dir /scratch/$USER/abmil_v2_brca --round-start 1 --round-end 10
python analysis/abmil_compartment_shapley.py --cohort blca --aggregate --rounds-dir /scratch/$USER/abmil_v2_blca --out results/abmil_shapley_v2_blca.json
python analysis/abmil_compartment_shapley.py --cohort brca --aggregate --rounds-dir /scratch/$USER/abmil_v2_brca --out results/abmil_shapley_v2_brca.json
```
Copy round JSONs to `results/abmil_v2_{cohort}/`.

### N. Planted outcomes with last epoch and area-normalised attention

Same four plantings as before, both models, 10 rounds, with `--last-epoch
--attributions`. One model per job.

```bash
for P in "blca Necrosis necrosis" "blca Lamina lamina" "brca Necrosis necrosis" "brca Fibrous fibrous"; do set -- $P
  python analysis/planted_signal.py --cohort $1 --planted $2 --models graph --last-epoch --attributions \
    --rounds-dir /scratch/$USER/planted_v3_$1_$3 --round-start 1 --round-end 10
  python analysis/planted_signal.py --cohort $1 --planted $2 --models abmil --last-epoch --attributions \
    --rounds-dir /scratch/$USER/planted_v3_$1_$3 --round-start 1 --round-end 10
done
# per planting, after all rounds:
python analysis/planted_signal.py --cohort blca --planted Necrosis --models graph,abmil --aggregate \
  --rounds-dir /scratch/$USER/planted_v3_blca_necrosis --out results/planted_v3_blca_necrosis.json
```
Start one graph job per planting first; launch the rest once
`planted_feature.npz` exists. Copy round JSONs, `synthetic_meta.json` and
`summary_*.json` to `results/planted_v3_<cohort>_<planting>/`.
