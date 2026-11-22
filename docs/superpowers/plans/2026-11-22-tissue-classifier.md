# Tissue Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A two-step logistic-regression tissue classifier on saved CONCH embeddings, trained and tested on public labelled tiles plus GTEx normal tissue, that labels every tile of the 6,940-slide pan-carcinoma run.

**Architecture:** Pure helpers (classes, tile cutting, splits, metrics, pass bars) live in `analysis/tissue/` and are unit-tested on the laptop with numpy and pandas only. Beast scripts in the same folder fetch and tile the public data, embed it with the run's exact CONCH and UNI-2 code on the shared GPU, train and test, and label the run's slides. Scripts are copied to `/home/users/juan/tissue/` on the Beast (not the Beast's repo checkout, which the running job writes to).

**Tech Stack:** Python; numpy, pandas (laptop and Beast); scikit-learn 1.9.1, torch, CONCH, timm, openslide (Beast env `/home/users/juan/envs/pilot`); idc-index, s5cmd and wsidicom (Beast env `/home/users/juan/envs/dltest`; wsidicom to be installed there); huggingface_hub 0.36.2.

**Spec:** `docs/superpowers/specs/2026-11-22-tissue-classifier-design.md`

## Global Constraints

- Never touch the running preprocessing job: no edits under `preprocessing/beast_runner/`, no restart, no `git` in `/home/users/juan/wsi-survival-interpretability` on the Beast, no packages installed into `/home/users/juan/envs/pilot`.
- The GPU may be shared with the run for the embedding task only (Juan, 2026-11-22).
- Write only in `/home/users/juan/tissue/`, `/data/scratch/juan/datasets/`, `/data/processed/juan/pancan/tissue_labels/` and `/data/processed/juan/pancan/tissue/`.
- No patient-level data in git: tiles, embeddings and per-tile labels stay on the Beast. Git gets code, tests, classifier weights (`results/pancan/tissue_classifier_weights.npz`) and aggregate results.
- Embedding must match `preprocessing/beast_runner/pancan_runner.py` exactly: CONCH `create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")` preprocess, `.half()` input, `torch.amp.autocast("cuda", dtype=torch.float16)`, `encode_image` output L2-normalised; UNI-2 as built there with `Resize(224)`, ImageNet normalisation.
- Classes, in this order: `tumour, normal_epithelium, stroma, immune, necrosis, fat, vessels, muscle, other`.
- Bars: class recall ≥ 0.85 and precision ≥ 0.80 on every test set that labels it; tumour recall ≥ 0.85 per type on TCGA-UT test sites; GTEx tiles called tumour ≤ 0.10 per organ; normal epithelium: TCGA-UT tumour tiles called normal epithelium < 0.05 in every type; pathomic threshold = calibrated probability where validation precision reaches 0.90, one per class. All bars on the 64 µm crop.
- Merge table (fixed): vessels, muscle, immune → stroma; necrosis, fat, normal_epithelium, stroma → other. Applied once, then retrain once.
- Splits: whole TSS codes for TCGA-UT and BCSS together (one site map for both, since they share TCGA patients); whole donors for GTEx; Kather 100K train, CRC-VAL-HE-7K test. Fractions 0.70 / 0.15 / 0.15, seed 0.
- Commit on the laptop after each task; push only with Juan's OK.

## Review Focus

- A TCGA patient in both TCGA-UT and BCSS lands in train in one and test in the other: one TSS-to-split map serves both sources (Task 1 test `test_one_site_map_for_two_sources`).
- A class missing from a test set (vessels in Kather, everything but tumour in TCGA-UT) is judged as failing or crashes the bars: bars use only test sets where the class has true tiles (Task 2 test `test_bar_ignores_sets_without_the_class`).
- A TCGA-UT label that is not one of our 17 types (or COAD and READ merged as one label) is silently mislabelled: unknown labels map to `None` and are dropped, and the 17 types are all present (Task 1 test `test_tcgaut_types_cover_cohort`).
- GTEx tiles of glass, pen or folds count as "not tumour" tissue: tiles under 50% tissue by Otsu are rejected (Task 1 test `test_tissue_fraction_rejects_blank`).
- A run slide being written (`*.tmp`) or a feature file without a `conch` key crashes the labelling or gets a half label: such files are skipped and listed (Task 7 verification).

---

### Task 1: Classes, tile cutting and splits

**Files:**
- Create: `analysis/tissue/__init__.py`, `analysis/tissue/classes.py`, `analysis/tissue/tiles.py`, `analysis/tissue/splits.py`
- Test: `tests/test_tissue_helpers.py`

**Interfaces:**
- Produces:
  - `classes.CLASSES: list[str]` (order above); `classes.NONTUMOUR = CLASSES[1:]`
  - `classes.KATHER_MAP: dict[str, str]` — TUM→tumour, NORM→normal_epithelium, STR→stroma, LYM→immune, DEB→necrosis, ADI→fat, MUS→muscle, BACK→other, MUC→other
  - `classes.BCSS_MAP: dict[int, str | None]` — BCSS mask codes; the 8 named in the spec are fixed, the rest set from the BCSS code table on download (Task 3) to `"other"` or `None` (dropped); code 0 (outside ROI) is `None`
  - `classes.tcgaut_type(label: str) -> str | None` — TCGA-UT cancer-type label to our type code, `None` for types outside the 17
  - `classes.GTEX_TISSUES: dict[str, list[str]]` — our type code → GTEx tissue names (HNSC → oesophagus mucosa and minor salivary gland); names checked against the IDC index in Task 4
  - `classes.MERGE: dict[str, str]` — the fixed merge table
  - `tiles.center_crop(img: np.ndarray, size: int) -> np.ndarray`
  - `tiles.otsu_threshold(gray: np.ndarray) -> float`
  - `tiles.tissue_fraction(rgb: np.ndarray, thr: float) -> float` — share of pixels darker than `thr` in grey
  - `tiles.labelled_tiles(mask: np.ndarray, tile: int, min_cover: float = 0.8) -> list[tuple[int, int, int]]` — (x, y, code) of non-overlapping tiles whose most common code covers ≥ `min_cover`; code 0 never returned
  - `splits.tss_of(barcode: str) -> str` — "TCGA-A2-A0T2..." → "A2"
  - `splits.assign_groups(groups: Sequence[str], strata: Sequence[str], fractions=(0.70, 0.15, 0.15), seed: int = 0) -> dict[str, str]` — each group (site or donor) to "train" / "valid" / "test", shuffled within its stratum (most common type of the group), deterministic for a seed

- [ ] **Step 1: Write the failing tests** in `tests/test_tissue_helpers.py`:
  - `test_kather_map_covers_nine_codes`: `set(KATHER_MAP) == {"ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"}` and all values in `CLASSES`.
  - `test_tcgaut_types_cover_cohort`: the 17 cohort codes (`BLCA BRCA CESC COAD ESCA HNSC KIRC KIRP LIHC LUAD LUSC OV PAAD PRAD STAD THCA UCEC`) each appear as `tcgaut_type(x)` for some TCGA-UT label; `tcgaut_type("Glioblastoma multiforme") is None`.
  - `test_merge_table_complete`: every class but tumour and other is a key of `MERGE`; every value is `"stroma"` or `"other"`.
  - `test_center_crop`: a 224×224×3 array with a marked centre returns 128×128×3 whose [0,0] equals input [48,48].
  - `test_tissue_fraction_rejects_blank`: an all-white 256×256 tile has fraction < 0.5; a tile half dark has 0.5 ± 0.01.
  - `test_labelled_tiles_cover_rule`: a 512×512 mask, left half code 1, right half code 2 except a 256×256 block of code 0 → tiles (0,0,1), (0,256,1), (256,0,2) and the block skipped; a tile 70% code 1 and 30% code 2 is not returned.
  - `test_one_site_map_for_two_sources`: `assign_groups` over the union of sites gives each site one split; same seed gives the same map; fractions within ±1 site per stratum.
- [ ] **Step 2: Run them and see them fail.** Run: `python -m pytest tests/test_tissue_helpers.py -q` — Expected: import errors.
- [ ] **Step 3: Implement the three modules** with the signatures above (numpy and pandas only; Otsu from a 256-bin histogram).
- [ ] **Step 4: Run the tests.** Expected: all pass.
- [ ] **Step 5: Commit** `analysis/tissue/ tests/test_tissue_helpers.py` — "Tissue classifier helpers: classes, tile cutting, site and donor splits".

### Task 2: Metrics, calibration, two-step combination and pass bars

**Files:**
- Create: `analysis/tissue/metrics.py`, `analysis/tissue/bars.py`
- Test: `tests/test_tissue_metrics.py`

**Interfaces:**
- Consumes: `classes.CLASSES`, `classes.MERGE`
- Produces:
  - `metrics.combine(p_tumour: np.ndarray, p_rest: np.ndarray) -> np.ndarray` — (n,) and (n, 8) → (n, 9) in `CLASSES` order: `[p_tumour, (1 - p_tumour) * p_rest]`
  - `metrics.fit_temperature(logits: np.ndarray, y: np.ndarray) -> float` — T in [0.05, 20] minimising NLL of `softmax(logits / T)` (binary logits passed as (n, 2))
  - `metrics.ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float`
  - `metrics.recall_precision(y_true: Sequence[str], y_pred: Sequence[str], classes: list[str]) -> pd.DataFrame` — columns class, n_true, recall, precision; recall NaN when n_true = 0
  - `metrics.threshold_for_precision(p: np.ndarray, is_k: np.ndarray, target: float = 0.90) -> float | None` — lowest threshold whose precision ≥ target, None if never reached
  - `bars.class_bars(table: pd.DataFrame) -> dict[str, bool]` — table columns test_set, class, n_true, recall, precision
  - `bars.tumour_organ_bars(recall_by_type: dict[str, float], gtex_false_tumour: dict[str, float]) -> pd.DataFrame` — per type: recall, false_tumour, passes
  - `bars.normal_epithelium_bar(passes_on_sets: bool, tumour_called_normal: dict[str, float]) -> bool`
  - `bars.merges_for(failed: set[str]) -> dict[str, str]` — the MERGE entries for the failed classes

- [ ] **Step 1: Write the failing tests** in `tests/test_tissue_metrics.py`:
  - `test_combine_rows_sum_to_one`: random inputs → rows sum to 1 within 1e-6, column 0 equals `p_tumour`.
  - `test_temperature_recovers_scale`: labels sampled from `softmax(z)` with `logits = 3 * z` → fitted T within 3 ± 0.3.
  - `test_ece_zero_when_perfect`: one-hot probabilities on the true class → 0.
  - `test_threshold_for_precision`: scores where precision reaches 0.9 at 0.6 → 0.6; unreachable → None.
  - `test_bar_ignores_sets_without_the_class`: vessels with n_true = 0 in "kather7k" and recall 0.9, precision 0.85 in "bcss" → passes; recall 0.84 in one set → fails.
  - `test_tumour_organ_bar`: recall 0.86 and false tumour 0.10 → passes; 0.11 → fails.
  - `test_normal_bar`: 0.049 in every type passes; one type at 0.05 fails.
  - `test_merges_for`: `{"vessels","fat"}` → `{"vessels":"stroma","fat":"other"}`.
- [ ] **Step 2: Run them and see them fail.** Run: `python -m pytest tests/test_tissue_metrics.py -q`
- [ ] **Step 3: Implement** with the constants from Global Constraints as module-level names (`RECALL_MIN = 0.85`, `PRECISION_MIN = 0.80`, `TUMOUR_RECALL_MIN = 0.85`, `GTEX_FALSE_TUMOUR_MAX = 0.10`, `NORMAL_FROM_TUMOUR_MAX = 0.05`, `PATHOMIC_PRECISION = 0.90`). Temperature by golden-section search on log T.
- [ ] **Step 4: Run all tissue tests.** Run: `python -m pytest tests/test_tissue_helpers.py tests/test_tissue_metrics.py -q` — Expected: all pass.
- [ ] **Step 5: Commit** — "Tissue classifier metrics, calibration and fixed pass bars".

### Task 3: Fetch and tile the public labelled sets (Beast)

**Files:**
- Create: `analysis/tissue/fetch_public.py`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `/data/scratch/juan/datasets/tissue/tiles/<source>/...png` and `/data/scratch/juan/datasets/tissue/tile_index.csv` with columns `tile_id, source, path, version, label, group, type, split`. `source` ∈ {tcgaut, kather100k, kather7k, bcss}; `version` ∈ {full, crop64}; `label` ∈ CLASSES; `group` = TSS (tcgaut, bcss) or "kather"; `type` = our type code or "" (kather is "COAD", bcss "BRCA").

- [ ] **Step 1: Download** with `huggingface_hub.snapshot_download` (env `HF_HOME=/data/scratch/juan/cache/huggingface`, `HF_HUB_OFFLINE` unset): `dakomura/tcga-ut` external-split tars (about 6.1 GB), `1aurent/NCT-CRC-HE` split `NCT_CRC_HE_100K_NONORM`, `MedOtter/BCSS`. Kather 7K is the copy already in `/data/scratch/juan/datasets/kather_crc_val_7k/`. Record the BCSS code table and set the open `BCSS_MAP` entries; recommit Task 1 if it changes.
- [ ] **Step 2: Tile.** TCGA-UT: keep tiles whose label maps to one of the 17 types (`tcgaut_type`), label tumour, versions full (256 px, 128 µm) and crop64 (centre 128 px). Kather: full (224 px) and crop64 (centre 128 px). BCSS: `labelled_tiles(mask, 256)` at 0.25 µm/px as crop64, and `labelled_tiles` on the mask downsampled 2x, image tiles of 512 px resized to 256, as full. TCGA-UT keys must yield the patient barcode; stop and report if they do not.
- [ ] **Step 3: Split.** One `assign_groups` call over the union of TCGA-UT and BCSS TSS codes, strata = each site's most common type. Kather 100K → train, 7K → test.
- [ ] **Step 4: Verify.** Run on the Beast: `/home/users/juan/envs/pilot/bin/python -I fetch_public.py --summary`. Expected: a table of tiles per source × split × label × version; all 17 types present in TCGA-UT test; no TSS in two splits (printed check "sites in more than one split: 0"); 20 random tiles per source saved to `/data/scratch/juan/datasets/tissue/peek/` and looked at.
- [ ] **Step 5: Commit** `analysis/tissue/fetch_public.py` (and `classes.py` if changed) — "Fetch and tile TCGA-UT, Kather and BCSS for the tissue classifier".

### Task 4: Fetch and tile GTEx normal tissue (Beast)

**Files:**
- Create: `analysis/tissue/fetch_gtex.py`

**Interfaces:**
- Consumes: `classes.GTEX_TISSUES`, `tiles.otsu_threshold`, `tiles.tissue_fraction`, `splits.assign_groups`.
- Produces: GTEx rows appended to `tile_index.csv` with `source = gtex`, `label = "not_tumour"`, `group` = donor ID, `type` = our type code (one row per tile per type it stands in for).

- [ ] **Step 1: Install wsidicom into the dltest env only** (`/home/users/juan/envs/dltest/bin/pip install wsidicom`). Check it opens one slide of each photometric group (RGB, YBR_ICT, YBR_FULL_422); stop and report if YBR_FULL_422 fails.
- [ ] **Step 2: Select slides** with `idc_index.IDCClient` (`sm_index`, collection `gtex`): match `GTEX_TISSUES` names (print the IDC tissue vocabulary and fix the dict if names differ), 20 donors per tissue, one slide each, seed 0. Download with s5cmd to `/data/scratch/juan/datasets/gtex/`.
- [ ] **Step 3: Tile** each slide at the level nearest 0.5 µm/px: tiles of 64 µm (crop64) and 128 µm (full) resized to 256 px, kept if `tissue_fraction ≥ 0.5` with the slide's Otsu threshold; at most 300 tiles per slide per version, random with seed 0. Record the slide's µm/px.
- [ ] **Step 4: Split** donors with `assign_groups` (strata = tissue).
- [ ] **Step 5: Verify.** Run: `/home/users/juan/envs/dltest/bin/python -I fetch_gtex.py --summary`. Expected: tiles per tissue × split; every one of the 17 types has GTEx test tiles; 20 random tiles per tissue saved to `peek/` and looked at (no glass, no pen).
- [ ] **Step 6: Commit** — "Fetch and tile GTEx normal tissue for the tissue classifier".

### Task 5: Embed the public tiles (Beast, shared GPU)

**Files:**
- Create: `analysis/tissue/embed_public.py`

**Interfaces:**
- Consumes: `tile_index.csv`.
- Produces: `/data/scratch/juan/datasets/tissue/emb/<source>.pt` = `{"tile_id": list[str], "conch": float16 (n, 512), "uni2": float16 (n, 1536)}`.

- [ ] **Step 1: Write** `embed_public.py`, copying the model construction and transforms from `preprocessing/beast_runner/pancan_runner.py` (not importing it). Batches of 256, DataLoader with 8 workers (the run already uses 16), resume by skipping sources whose output exists.
- [ ] **Step 2: Reproduce the run's embedding first.** Take one finished run slide, re-embed 512 of its tiles from the raw slide with this script's code path, and compare with its saved `conch`: Expected cosine similarity ≥ 0.999 on every tile. Stop if not.
- [ ] **Step 3: Run** in the background on the Beast with `nohup`, logging to `/data/processed/juan/pancan/tissue/logs/embed.log`. Watch the run's ledger rate before and during; expected slowdown only while embedding runs.
- [ ] **Step 4: Verify.** Expected: one file per source, row counts equal to the index, CONCH norms 1.00 ± 0.01, no non-finite values.
- [ ] **Step 5: Commit** — "Embed public tissue tiles with the run's CONCH and UNI-2 code".

### Task 6: Train, calibrate, test and apply the bars (Beast)

**Files:**
- Create: `analysis/tissue/train_eval.py`
- Create (results, git): `results/pancan/tissue_classifier_weights.npz`, `results/pancan/tissue_classifier_tests.csv`, `results/pancan/tissue_classifier_summary.json`
- Create: `docs/TISSUE_CLASSIFIER_RESULTS.md`

**Interfaces:**
- Consumes: embeddings, `tile_index.csv`, Task 1 and 2 helpers.
- Produces: `load_classifier(path) -> Callable[[np.ndarray], np.ndarray]` in `train_eval.py`, mapping (n, 512) CONCH float arrays to (n, 9) calibrated probabilities in `CLASSES` order (classes merged away get probability folded into their target and stay as columns of 0); the weights file holds `coef1, b1, T1, coef2, b2, T2, merged` and per-class thresholds.

- [ ] **Step 1: Train step 1** (tumour or not) on train rows, both versions: positives tumour from tcgaut, kather100k, bcss; negatives everything else plus gtex. `LogisticRegression(class_weight="balanced", max_iter=3000)`, C chosen from `[0.01, 0.1, 1, 10]` by validation log-loss (tcgaut, bcss, gtex valid rows). Sample weights so that each type contributes equally among tcgaut and gtex rows.
- [ ] **Step 2: Train step 2** on non-tumour train rows of kather100k and bcss (8 classes), C chosen the same way on bcss valid rows.
- [ ] **Step 3: Calibrate** T1 and T2 with `fit_temperature` on valid rows; per-class thresholds with `threshold_for_precision` on valid rows.
- [ ] **Step 4: Test** on test rows, crop64 for the bars and full reported alongside: `recall_precision` per test set (kather7k, bcss, tcgaut); tumour recall per type (tcgaut); GTEx false tumour per type; source shortcut (share of bcss test non-tumour tiles called tumour against the share of GTEx breast test tiles called tumour); share of each class among tcgaut tumour tiles per type; ECE before and after. Same tests for a UNI-2 version, reported only.
- [ ] **Step 5: Apply the bars** (`class_bars`, `tumour_organ_bars`, `normal_epithelium_bar`). If any class fails: apply `merges_for(failed)`, retrain once, retest, report both rounds. The spec's second-wave data (SegPath, TIGER) is not in this plan; a failed class is reported so Juan can decide on it.
- [ ] **Step 6: Verify.** Run: `/home/users/juan/envs/pilot/bin/python -I train_eval.py`. Expected: the three results files written; the summary lists every bar with its number and pass or fail.
- [ ] **Step 7: Write `docs/TISSUE_CLASSIFIER_RESULTS.md`** in plain language (what was trained, test numbers, bars, merges, limits), every number traced to the results files. Commit code, results and doc.

### Task 7: Plausibility check and labels for the run's slides

**Files:**
- Create: `analysis/tissue/plausibility.py`, `analysis/tissue/label_run.py`
- Create (results, git): `results/pancan/tissue_plausibility_by_type.csv`
- Modify: `docs/HANDOFF_2026-11-21.md` (section 3), `docs/TISSUE_CLASSIFIER_RESULTS.md`

**Interfaces:**
- Consumes: `load_classifier`, run feature files `/data/processed/juan/pancan/features/<type>/<slide>_emb16.pt` (key `conch`), the manifest.
- Produces: `/data/processed/juan/pancan/tissue_labels/<type>/<slide>.pt` = `{"label": uint8 (n,) index into CLASSES, "probs": float16 (n, 9), "classifier": sha256 of the weights file}`.

- [ ] **Step 1: `label_run.py`:** label every finished slide; skip `*.tmp`, files without `conch`, and slides already labelled with the same classifier hash; print the skipped list. Re-runnable after the run ends (about 2026-12-03).
- [ ] **Step 2: `plausibility.py`:** fetch from the GDC API, for cohort patients, the frozen-section slide fields `percent_tumor_nuclei`, `percent_necrosis`, `percent_stromal_cells`, `percent_lymphocyte_infiltration` (mean of top and bottom sections); Spearman within type against the predicted tumour, necrosis, stroma and immune shares; save a 20-tile grid per class per type as PNG to `/data/processed/juan/pancan/tissue/grids/`.
- [ ] **Step 3: Verify.** Expected: labels for every finished slide with no unexpected skips; the plausibility table has 17 rows; any type with tumour correlation ≤ 0 is named in the results doc.
- [ ] **Step 4: Update the handoff and results doc,** commit, and ask Juan before pushing.
