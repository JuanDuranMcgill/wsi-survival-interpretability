# Design: running the IER audit where the data is large enough (TCGA pan-cancer)

Status: draft for Juan's approval (2026-11-20; updated 2026-11-22). Extends the AIIM
manuscript (Overleaf project 6a1dc8fa08f8c0d184f393aa, `manuscript_aim.tex`). One paper, not
a second.

**What changed on 2026-11-22** (for Juan to check):
- Compute moved to the MEDomics-Beast. Preprocessing of all 6,940 slides started on
  2026-11-22 and should end about 2026-12-03 (decision 10).
- Tissue labels come from a classifier trained on public labelled tiles, not from CONCH
  prompts. There are two label sets, the main one and a stroma-kept variant, and the audit
  runs on both (decision 2).
- The events curve stops at 1,999 events, the carcinoma cohort's total (decision 8).
- The pathomic scripts are in the repo, and slides are kept rather than streamed
  (decisions 14 and 15).
- The paper reports what failed on the way, not only what worked (decision 11).

## Intent and why

The point of this work is patients. AI survival models will only help patients if
researchers can trust what the models' explanations say before the models reach the
clinic. The paper built a tool for researchers, the IER audit, that tests three
explanations: tissue weights (I), surrogate feature rankings (E) and flagged
reliability groups (R). On TCGA bladder (163 PFI events) and breast cancer (139)
most explanations failed, and the paper concludes that cohorts with under 200
events are too small.

Pathologists who read the paper liked it, but it is only a negative result. The
extension turns it into a positive, actionable one: run the same tool where the
data is large enough, and show at how many events each explanation starts to pass.
Researchers then know what it takes to build models whose explanations can be
trusted, which is the step that matters for patients. The message has to be simple enough
for pathologists who will not read the methods: with enough data, does a model pass more of
the checks, and how much data does that take?

## What the literature review found (notes with checked references in `docs/lit/`)

- Sample-size theory supports the paper. 139-163 events give the overall C-index to
  about +/-0.05 (Collins 2016: >= 100, preferably >= 200 events), but anything that
  splits the events (subgroups, interactions) needs about 4x more (Brookes 2004;
  Gelman). A ~25-parameter Cox model needs 500-900 events by Riley 2019/2020.
  Detecting a C-index difference of 0.10 between halves needs about 310 events.
  Target for the extension: roughly 600-1,000+ events.
- No single TCGA cancer type is large enough. With slides required, the largest are
  GBM (~330 PFI events) and SKCM (~285). TCGA-CDR has 3,996 PFI events in all
  (Liu 2018), but fewer among patients with a diagnostic FFPE slide. Exact counts from
  the manifest (2026-11-20, `results/pancan/manifest_by_type.csv`): 9,752 patients have
  a DX slide, 9,362 are eligible with 3,208 PFI events, and the 22 types with at least
  50 events give 8,562 patients and 2,974 events. GBM and OV lose the most (OV: 106 of
  587 patients have a DX slide).
- Pooling has known traps: a pooled C-index rewards recognising the cancer type
  (Song 2025; Wulczyn 2020 used within-type pairs), tissue-source-site effects
  inflate scores (Howard 2021; site-split folds lowered PORPOISE from 0.606 to
  0.576), and tissue-type importance can encode cancer type.
- Explanation stability needs large samples elsewhere (Ein-Dor 2006: thousands;
  Legha 2026: 1,100-1,800), and nobody has measured how explanation reliability in
  computational pathology grows with cohort size. This is a gap the paper can fill.

## Goals

1. A pan-cancer TCGA cohort with enough events, on which the full audit (I, E, R)
   runs with the same code and the same pass/fail criteria as the current paper.
2. An events curve: the audit rerun on growing subsets of that cohort, showing for
   each check the event count at which it starts to pass.
3. A power curve for the interpretability check from planted signals of known size
   at each event count, so a failure can be read as "not enough data" or "wrong
   explanation".
4. The paper's sample-size claim grounded in theory (Harrell, Peduzzi, Riley,
   Collins), cited in the Introduction or Discussion.
5. Paper text: a new Results part for the pan-cancer audit and the events curve,
   with the Discussion, Limitations and Conclusion updated. The current BLCA/BRCA
   results stay as they are and become the small-data contrast.

## Non-goals

- Building the best-performing model. A stronger model is welcome but not the aim.
- Claiming clinical readiness of any model.
- Re-running the existing BLCA/BRCA analyses.

## Anti-goals (failure even if it "works")

- A pass that comes from the cancer type or the hospital rather than the patient.
- Any leakage between training and evaluation patients, or between hospitals across
  folds when a site split is claimed.
- Changing the audit's criteria to make checks pass.
- Patient-level data in git.

## Approaches considered

- **A. TCGA pan-cancer plus an events curve (chosen by Juan, with B).** Public data,
  our own pipeline (UNI-2 features, tissue labels from our own classifier, our checks); answers when
  explanations become trustworthy. Cost: about 7x the slides of the current paper to
  tile, label and encode, and many training runs for the curve.
- **B. A large external single-cancer cohort (chosen by Juan, with A).** DACHS or
  TransSCOT colorectal (2,000-3,000 patients) or PLCO (breast 1,012, colorectal 749,
  prostate 1,095 patients), with the Kather lab's public colorectal model where it
  fits. Cleanest test of the audit on a large single cancer outside TCGA, but access
  is by application, takes months and sometimes costs money. Applications start now
  and A runs meanwhile; B enters the paper if access arrives in time.
- **C. A stronger model on the same two cancers.** Dropped: the event count, not the
  model, is the binding limit, so the checks that split the data would still fail.

## Decisions (Claude's calls unless marked)

1. **Cohort.** TCGA cancer types with diagnostic FFPE WSIs and PFI in TCGA-CDR
   (Liu 2018), keeping types with at least 50 PFI events among patients with a
   slide, so that within-type evaluation is meaningful. One slide per patient, chosen
   as in the current pipeline. **Carcinomas only (Juan, 2026-11-21):** the 5
   non-carcinoma types (GBM, LGG, SARC, SKCM, MESO) are dropped, leaving 17 types,
   6,940 patients and 1,999 PFI events. Reasons: published pan-cancer tissue sets
   (OpenTME/HETA 2026, Lunit SCOPE, Kather CRC, Diao 2021) cover carcinomas only, and
   3 of the 4 types without any stage are non-carcinomas. Within-type clinical
   C-index with age, sex and stage: 0.638 for the 17 types
   (`results/pancan/clinical_variables_check.csv`).
2. **Tissue types.** Revised 2026-11-22 (design
   `docs/superpowers/specs/2026-11-22-tissue-classifier-design.md`, results
   `docs/TISSUE_CLASSIFIER_RESULTS.md`). The starting set was 8 classes that every
   carcinoma has or can have: tumour, normal epithelium, stroma, immune infiltrate,
   necrosis or haemorrhage, fat, vessels, muscle. CONCH prompts labelled only about half
   of the tiles correctly (balanced accuracy 0.52 on Kather's colon tiles). Each tile is
   therefore labelled by a two-step logistic regression on its UNI-2 embedding (Juan chose
   UNI-2 over CONCH). It was trained on public tiles that pathologists had labelled and
   tested on hospitals, donors and slides it never saw. Classes that missed the bars fixed
   in advance were merged by a fixed table.
   - **Main labels:** tumour, normal epithelium, immune and fat. Stroma, vessels, muscle
     and necrosis go to "other", which belongs to no tissue compartment. "Other" holds 60%
     of the tiles labelled so far (486 slides, `results/pancan/tissue_threshold_kept.csv`),
     so the main model does not see most stroma.
   - **Stroma-kept labels (Juan, 2026-11-22):** stroma, with vessels and muscle, is its own
     compartment, and "other" then holds 14% of the tiles (555 slides,
     `results/pancan/tissue_threshold_kept_stroma.csv`). Stroma narrowly misses its bar
     (BCSS recall 0.843 against 0.85, kidney lower), and the paper says so.
   - The audit runs on both label sets. Tumour against normal tissue passes in all 17
     organs.
   The pathologist review of labels is not repeated at scale. In its place are the held-out
   tests and a comparison of each patient's tumour and stroma shares with TCGA's own
   pathologist estimates (Spearman above 0.05 in 11 and 14 of the 16 types with enough
   patients). Some of the classifier's training tiles come from TCGA slides of this cohort.
   The classifier never sees survival and is fixed before any survival model is trained,
   so this is not outcome leakage.
3. **Features.** UNI-2 tile embeddings (256 px tiles at full resolution, float16), computed
   once per slide by the Beast run (`preprocessing/beast_runner/`), which also saves CONCH
   embeddings.
4. **Models.** Our per-tissue-type weighted model, on the compartments of decision 2 (four
   in the main labels, five with stroma), trained pan-cancer with a Cox
   partial likelihood stratified by cancer type, so it cannot win by ranking cancer
   types. ABMIL trained the same way as the second model. Same training settings as
   the current main runs unless a pilot shows they do not converge.
5. **Splits.** Patient-level out-of-bag rounds as now, with whole tissue-source sites
   held out together (site-split), stratified by cancer type.
6. **Metrics.** Every C-index (performance, Shapley payoff for the outcome side of
   check I, MED3pa in-profile comparisons) is computed on within-cancer-type
   comparable pairs only. Patient error for check R likewise uses within-type pairs.
7. **Audit settings.** Unchanged from the paper: Shapley over all tissue-type
   coalitions, surrogate random forest with Benjamini-Hochberg FDR, MED3pa with
   clinical (including cancer type and stage) and pathomic arms at q50/q70/q80/q90,
   Bonferroni over the tests.
8. **Events curve.** Target event counts 150, 300, 600, 1,200 and all (1,999 in the 17
   carcinoma types; 2,400 was dropped because it exceeds the cohort, Claude's call),
   sampled by patient and stratified by cancer type, 3 repeats per size below "all". At
   each size, record each check's verdict and effect size. The run processes slides in an
   order stratified by type, so the slides finished at any point are a fair sample. The
   training and audit code is built and tested on them while the run continues, and the
   smallest curve points can run before it ends (Claude's call).
9. **Planted signal.** The paper's planted-outcome test, at each curve size, with one
   or two effect sizes.
10. **Compute.** Revised 2026-11-22: the MEDomics-Beast (one 96 GB GPU, shared with
    another user). Preprocessing reads each slide once (tiling, CONCH and UNI-2
    embeddings), about 280 GPU-hours for the 6,940 slides by the run's own estimate. It
    started on 2026-11-22 and should end about 2026-12-03 (`docs/PANCAN_RUN_STATUS.md`).
    Training runs on the Beast too, roughly 20 rounds for the full cohort and fewer
    (e.g. 5) per curve point, unless a pilot shows it is too slow; Trillium and Narval are
    the fallback (Claude's call). Juan approves every GPU job.
11. **Paper.** All text through the paper-writing-check skill; numbers traced to
    `results/`. The paper shows what failed on the way as well as the path that works
    (Juan, 2026-11-22): for tissue labels, the CONCH prompts, a classifier on CONCH
    embeddings, merging failed classes and neighbour smoothing, plus the choices made after
    seeing test results (listed in `docs/TISSUE_CLASSIFIER_RESULTS.md`).
12. **External cohort (B).** Juan applies for access now: PLCO through NCI's CDAS,
    and the Kather lab for DACHS (or TransSCOT) colorectal, as a collaboration.
    Once data arrive, the same pipeline and audit run unchanged on that cohort (one
    cancer type, so ordinary C-index), with our model and ABMIL trained on it, and
    the Kather model's risk scores audited where its training patients are not the
    ones evaluated. The paper is written so that B can be added as one more Results
    part without restructuring.

## Settled after the builder check (Claude's calls; Juan to check)

13. **Pass rules, fixed before anything runs.** Each check uses exactly the test in
    the paper's Table 1, coded once in `analysis/pancan/verdicts.py` and frozen
    before the first curve point:
    - I (faithful): Spearman rho between weights and risk-score Shapley values > 0
      with permutation p < 0.05. I (prognostic): Spearman rho between risk-score and
      C-index Shapley values > 0 with a patient-bootstrap 95% CI excluding 0.
    - E: at least one pathomic feature with Benjamini-Hochberg q < 0.05 against PFI
      (Cox stratified by cancer type), across all features of the cohort.
    - R (predictable): error-classifier AUROC above 0.5 with permutation p below the
      Bonferroni level over the 16 tests. R (differs): at least one recurring profile
      (>= 5% of runs) whose in-profile C-index 95% interval excludes the cohort value.
    - At a curve size, a check passes if it passes in at least 2 of the 3 repeats;
      the pass fraction and effect sizes are reported alongside.
14. **Pathomic features.** In scope for all slides, with the same first-order
    pyradiomics features per tissue type. The extraction scripts are in
    `preprocessing/pathomics/` (copied from Trillium, commit 62e6468). They were written
    for BLCA, BRCA and the old tile files, and are adapted to the run's tile positions and
    the new labels; they read slides, so they run on all slides only after the run ends.
    Only tiles above their class's confidence cut are used (the cut at which 90% of the
    kept tiles are truly that class; each label set has its own cuts). Extraction is CPU
    work, so its speed is measured early (Claude's call).
15. **Slide storage.** Settled 2026-11-22: all slides are kept on the Beast
    (`/data/storage/juan/pancan_slides/`, the rest on `/data/scratch`), so pathomic
    features can be computed after the run. Outputs (about 1.2 TB) go on
    `/data/processed`.

## Open: Juan's answer needed

16. **GPU budget and what to cut first.** Preprocessing is measured (decision 10);
    training is not yet, and a pilot on finished slides sizes it. Proposed order of cuts
    if it does not fit: the stroma-kept labels only on the full cohort (not on the curve
    or the planted test), then ABMIL only on the full cohort, then one planted effect
    size, then 2 repeats per curve size instead of 3, then fewer rounds per curve point.

## Left to the builder

Job scripts, batch sizes and time limits; how to subsample within types; the pilot
design that sizes the compute; file layout under `results/pancan/`.
