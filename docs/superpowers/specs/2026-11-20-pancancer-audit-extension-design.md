# Design: running the IER audit where the data is large enough (TCGA pan-cancer)

Status: draft for Juan's approval (2026-11-20). Extends the AIIM manuscript
(Overleaf project 6a1dc8fa08f8c0d184f393aa, `manuscript_aim.tex`). One paper, not a second.

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
trusted, which is the step that matters for patients.

## What the literature review found (notes with checked references in the
scratchpad `lit/` folder; to be copied into `docs/lit/` with this design)

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
  our own pipeline (CONCH labels, UNI-2 features, our checks); answers when
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
2. **Tissue types.** A shared pan-cancer set that every included cancer has or can
   have: tumour, stroma, immune infiltrate, necrosis or haemorrhage, adipose, blood
   vessels, normal epithelium or parenchyma, muscle. CONCH zero-shot with one prompt
   per class, as now. The label check (pathologist review) is not repeated at scale;
   the paper states this.
3. **Features.** UNI-2 tile embeddings, as now (`preprocessing/embed_all_tiles_uni2.py`,
   `preprocessing/classify_all_tiles_multi_6.py` generalised to the shared classes).
4. **Models.** Our per-tissue-type weighted model, trained pan-cancer with a Cox
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
8. **Events curve.** Target event counts 150, 300, 600, 1,200, 2,400 and all
   (2,974 in the 22 included types), sampled by patient and stratified by cancer type, 3 repeats per
   size below "all". At each size, record each check's verdict and effect size.
9. **Planted signal.** The paper's planted-outcome test, at each curve size, with one
   or two effect sizes.
10. **Compute.** Trillium and Narval. Juan approves every GPU batch before
    submission. Embeddings staged on node-local SSD (Narval lesson). Rough order of
    magnitude, to be refined by a pilot: tiling, labelling and encoding about 9,750
    slides, plus roughly 20 training rounds for the full cohort and fewer (e.g. 5)
    per curve point; likely several hundred to over a thousand GPU-hours.
11. **Paper.** All text through the paper-writing-check skill; numbers traced to
    `results/`.
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
    pyradiomics features per tissue type. First task: find the extraction scripts on
    Trillium (`find_pairs.py`, `extract_radiomics.py`, `filter_radiomics*.py`), copy
    them into the repo, and rebuild them if they are gone.
15. **Slide storage.** Check project and nearline quotas on both clusters first. If
    under about 15 TB is free, stream the slides: download, tile, label, encode and
    extract features, then delete the slide, keeping only tiles' outputs.

## Open: Juan's answer needed

16. **GPU budget and what to cut first.** At the BRCA pace on Narval, the full plan
    could need several thousand GPU-hours. Proposed order of cuts if the pilot shows
    it does not fit: ABMIL only on the full cohort (not on the curve or the planted
    test), then one planted effect size, then 2 repeats per curve size instead of 3,
    then fewer rounds per curve point.

## Left to the builder

Job scripts, batch sizes and time limits; exact CONCH prompts for the shared classes;
how to subsample within types; the pilot design that sizes the compute; file layout
under `results/pancan/`.
