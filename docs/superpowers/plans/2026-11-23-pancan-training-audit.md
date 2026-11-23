# Pan-carcinoma training and IER audit: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Juan wants it done here, fast).
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the paper's two models (per-tissue weighted model, ABMIL) trained on the 17 carcinoma types and the full
IER audit (I, E, R) run on them, with within-type metrics and hospital-held-out rounds, ready to run the day the
preprocessing ends (about 2026-12-03) and tested end to end on the slides already finished.

**Architecture:** a new package `analysis/pancan/` that reuses the paper's model and audit functions
(`analysis/region_ablation.py`, `region_shapley.py`, `patient_error.py`, `planted_signal.py`, the MED3pa drivers)
and replaces only what the pooled design breaks: data loading from the run's files, Cox loss and C-index within
cancer type, hospital-split rounds, and any hard-coded BLCA/BRCA lists. Each label set (main, stroma-kept) is a
config value, so every step runs twice.

**Tech stack:** Python 3.12 on the MEDomics-Beast; torch; numpy/pandas/scikit-learn; lifelines; med3pa 1.0.4
(`--no-deps`, as on Narval); pyradiomics, SimpleITK, scikit-image, openslide for pathomics.

**Spec:** `docs/superpowers/specs/2026-11-20-pancancer-audit-extension-design.md` (decisions 1-16); tissue labels:
`docs/superpowers/specs/2026-11-22-tissue-classifier-design.md` and `docs/TISSUE_CLASSIFIER_RESULTS.md`.

## Global constraints

- The running preprocessing job is not touched: no GPU use before it ends unless Juan approves; `/data/storage` is
  not read during the run; writes only in `*/juan` folders; the `pilot` env gets no new packages (new env `audit`).
- No patient-level data in git (cohort tables with patient IDs and outcomes stay on the Beast; only aggregates in
  `results/pancan/`).
- Pass rules are those of design decision 13, coded once in `analysis/pancan/verdicts.py` and frozen (git tag)
  before the first curve point. Never changed to make a check pass.
- Every C-index is within cancer type (comparable pairs of the same type only); every Cox fit is stratified by type.
- Same training settings as the paper's main runs (AdamW lr 3e-5, weight decay 1e-3, region weights lr 1e-3,
  warmup then cosine, entropy term 1e-3) unless the dry run shows they do not converge.

## Review focus

1. A cancer type with no OOB events in a round: within-type C-index must skip it, not count it as 0.5 or crash.
2. A slide with a compartment that has no tiles (e.g. no fat): the model's zero token, Shapley coalitions and pathomic
   features must all treat it as absent, the same way.
3. Hospital split where one hospital dominates a type (OV 54%, LIHC 41%, COAD 38%): every round must still have
   OOB patients of every type, or say which types it lacks.
4. Patients with two candidate slides: exactly one slide per patient, chosen as in the manifest.
5. The two label sets must never mix: file names, caches and results carry the label set's name.

---

### Task 1: audit environment on the Beast
- Create `/home/users/juan/envs/audit` (venv, Python 3.12), install torch (CUDA build matching the pilot env),
  numpy, pandas, scikit-learn, scipy, lifelines, `med3pa==1.0.4 --no-deps` plus its listed dependencies, scikit-image,
  SimpleITK, pyradiomics (version in `preprocessing/pathomics/README.md`, or the nearest that builds on 3.12),
  openslide-python. Record exact versions in `envs/beast.md`.
- Verify: one import line for each package prints its version.

### Task 2: cohort table and hospital-split rounds
**Files:** create `analysis/pancan/cohort.py`, `analysis/pancan/splits.py`; test `tests/test_pancan_splits.py`.
- `build_cohort() -> DataFrame[patient, slide, type, site, pfi_time, pfi_event]` from the manifest
  (`in_cohort == True`) and TCGA-CDR; written to `/data/processed/juan/pancan/cohort.csv` (not git).
- `site_rounds(cohort, n_rounds, oob_frac=0.2, seed=0) -> DataFrame[round, patient, in_train]`: per round and per
  type, whole sites (TSS) go to OOB until about `oob_frac` of the type's patients are OOB; seed = round.
- Tests: no site is both train and OOB within a round; every type has OOB patients in every round (or the round
  is flagged); same seed gives the same split.

### Task 3: within-type metrics and stratified Cox loss
**Files:** create `analysis/pancan/metrics.py`; test `tests/test_pancan_metrics.py`.
- `cindex_within(time, event, risk, stratum) -> float`: Harrell's C over comparable pairs of the same stratum.
- `cox_loss_stratified(risk, time, event, stratum) -> tensor`: sum over strata of the paper's `cox_ph_loss`, divided
  by the total number of events.
- `patient_error_within(time, event, risk, stratum)`: the paper's `symmetric_concordance_error` with pairs restricted
  to the same stratum.
- Tests: on two strata, `cindex_within` equals the event-weighted pair count of per-stratum C-indices and ignores
  cross-type pairs (a constructed case where pooled C is 1.0 and within-type C is 0.5); loss equals the sum of
  per-stratum losses; a stratum with no events contributes nothing.

### Task 4: bag cache from the run's files
**Files:** create `analysis/pancan/bags.py`; test `tests/test_pancan_bags.py`.
- `make_bag(emb_path, label_path, compartments, cap=1024, seed=0) -> dict[compartment -> (feats float16 n x 1536,
  coords)]`: tiles grouped by their argmax label; at most `cap` per compartment, drawn with a seed (Claude's call, so
  6,940 slides fit: about 110 GB per label set on `/data/processed`).
- `compartments(variant)`: main = tumour, normal epithelium, immune, fat; stroma = the same plus stroma (read from
  the label set's weights file, so a relabelling cannot silently change them).
- Tests: cap respected; tiles land in their label's compartment; "other" tiles are dropped; same seed, same bag.

### Task 5: training driver (our model and ABMIL)
**Files:** create `analysis/pancan/train.py`; test `tests/test_pancan_train.py` (CPU, synthetic bags).
- Reuses `region_ablation.IPGGraphFormer(num_regions=len(compartments))` and `abmil_compartment_shapley.GatedABMIL`.
- Batches of 32 patients drawn from one cancer type at a time, so every risk set is within type (Claude's call; with
  mixed batches of 16 most pairs would cross types).
- CLI: `--model ours|abmil --variant main|stroma --rounds a:b --subset <csv> --epochs 30 --device cuda|cpu`.
- Writes per round and epoch: OOB risks, region weights (ours), OOB within-type C-index, to
  `/data/processed/juan/pancan/runs/<variant>/<model>/<subset>/round_###/`.
- Test: two epochs on synthetic bags run on CPU, the loss falls, and the files are written.

### Task 6: OOB risk and check I (Shapley over compartments)
**Files:** create `analysis/pancan/check_i.py`; test `tests/test_pancan_check_i.py`.
- OOB risk per patient = mean over the rounds where the patient was OOB, at the epoch with the best mean
  within-type OOB C-index (as in the paper).
- Exact Shapley over all compartment coalitions with the paper's `coalition_risks` / `shapley_from_table`; payoff =
  `cindex_within` on the round's OOB patients; patient bootstrap as `patient_bootstrap_shapley.py`.
- Test: Shapley values sum to v(all) minus v(none) on a synthetic table.

### Task 7: check R (patient error and MED3pa)
**Files:** create `analysis/pancan/check_r.py`; test `tests/test_pancan_check_r.py`.
- Error from `patient_error_within`; MED3pa as in `*_clinical_linear_med3pa.py` (q50/q70/q80/q90, 200 runs, the
  same split rule), clinical arm = age, sex, cancer type, stage, grade where recorded; pathomic arm after Task 8.
- In-profile C-index = `cindex_within`.
- Test: the high-error label at q50 marks half the patients with a defined error.

### Task 8: pathomic features and check E
**Files:** adapt `preprocessing/pathomics/` into `analysis/pancan/pathomics.py`; create `analysis/pancan/check_e.py`.
- Tiles from the run's coordinates and the new labels, only tiles at or above their class's confidence cut; same
  first-order (and GLCM, as in the paper) features of H and E channels; at least 5 tiles per compartment, at most
  1,000, drawn with a seed (the paper's draw had none).
- Measure the speed on 5 slides from `/data/scratch` as soon as the run moves one there or the run ends.
- Check E: Cox per feature stratified by type, Benjamini-Hochberg; surrogate random forest as in the paper.

### Task 9: frozen pass rules, events-curve subsets, planted signal
**Files:** create `analysis/pancan/verdicts.py`, `analysis/pancan/curve.py`; tests for both.
- `verdicts.py`: the rules of decision 13 as functions of the check outputs; git tag `pancan-rules-frozen`.
- `curve.py`: subsets at 150, 300, 600, 1,200 events, 3 repeats, patients sampled within type in proportion to
  events, seed fixed; written once to `results/pancan/curve_subsets_counts.csv` (counts only) and the patient lists
  on the Beast.
- Planted signal: `planted_signal.synth_outcome` on the pan-carcinoma bags, one or two effect sizes.

### Task 10: dry run on finished slides (CPU)
- All of the above on the slides finished so far, 2 rounds, 3 epochs, CPU, both label sets: every file is written
  and every check returns a verdict. Numbers from this run are not results.

### Task 11: full run (after the preprocessing ends)
- Relabel all slides, build bags, train on the GPU (Juan approves the job), run the audit and the curve.
