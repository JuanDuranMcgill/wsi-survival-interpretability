# IER Audit Figures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Fig. 1 (the IER audit at a glance) and Figs. 2–4 (one zoom per leg), export them as PDFs, then put them in the paper.

**Architecture:**
- Data panels are made in Python from `results/`. Every value goes through `figure_data.py`, which is pinned to the paper's numbers by tests.
- Diagrams are built with pptxgenjs, one editable `.pptx` per figure, with the data panels placed in as images.
- PowerPoint (COM) exports each deck to a vector PDF and a PNG for checking.
- Overleaf upload and LaTeX edits come last.

**Tech Stack:**
- Python 3.11: matplotlib, pandas, pytest.
- Node 24 with pptxgenjs.
- PowerPoint through COM on Windows.
- Overleaf through the browser pane.

**Spec:** `docs/superpowers/specs/2026-11-06-figures-design.md`

## Global Constraints

- **Width:** final width is about 6.3 in. Draw at final size. Fonts are 7–10 pt and never below 6.5 pt.
- **Palette:**
  - navy `1F2A5C` for model and structure;
  - card fill `E8EDF6`;
  - red `C0392B` for fails or differs;
  - green `2E8B57` for passes;
  - grey `7F8C8D` for neutral.
- **Font:** Calibri, both in the slides and in matplotlib.
- **Greyscale:** pass/fail is shown by ✓/✗ and a label, never by colour alone.
- **No slide chrome:** no footer, no navigation bar, no slide title, no accent stripes.
- **Fig. 1:** no findings.
- **Numbers:** every figure number equals the paper's.
- **(R) has no direction:** "more or less reliable" and "interval excludes cohort value", not "fail".
- **(E) shows both levels:** region order holds; feature order is weak (+0.28 / +0.18); the share is not robust (17% → 61% BLCA).

## Review Focus

- **A number in a panel differs from the paper** (rounding, or the wrong run or cohort). Expected: tests pin every plotted value to the paper. Check: Task 2 tests.
- **Text too small or clipped after scaling to text width.** Expected: no text under 6.5 pt and nothing cut off. Check: the Task 4/5 visual check on the PNG render at 100%.
- **The slide image is presented as more than it is.** The compartment map has no per-colour legend, so the caption says "colours mark the compartments" and names none. Check: Task 4 caption text.
- **An overwhelming panel.** Expected: each zoom (a) has at most 4 steps and each (b) at most 3 mini-plots. Check: the Task 6 fresh-eyes review.
- **An Overleaf breakage.** Expected: `fig:pipeline` and `fig:wsi_highlights` labels still resolve, and the compile has 0 errors. Check: Task 7 compile log.

---

### Task 1: Setup and image assets
**Files:**
- Create: `paper_figures/assets/` (`he_tiles.jpg`, `compartment_map.jpg`, `brca_slide.jpg`)
- Create: `paper_figures/package.json`
- Create: `paper_figures/.gitignore` (ignores `node_modules`)

- [ ] **Install:** `pip install --user matplotlib pytest`, then `npm install pptxgenjs` in `paper_figures/`. Expected: imports succeed.
- [ ] **Compartment map:** take the deck's `image49.png`, downscale to 1600 px wide and save as JPEG quality 85.
- [ ] **H&E views:** export a crop of Overleaf's `Figures/combined_panel` through a page canvas at 900 px or less, base64 chunks, then the transcript. This gives a real TCGA H&E thumbnail and a BRCA slide.
  - Ruling if blocked: use the uncoloured H&E areas of the compartment map.
  - Do not use the deck's `image47.jpg`; its source is unknown.

### Task 2: `figure_data.py` — every plotted value, pinned to the paper
**Files:**
- Create: `visualization/figure_data.py`
- Create: `tests/test_figure_data.py`

**Interfaces — produces:**
- `interp_summary(cohort) -> dict`:
  - `rho_weight_use`
  - `rho_use_outcome`
  - `use_share`: {region: %}
  - `shapley_pp`: {region: pp}
- `planted_hit_rates() -> dict` mapping measure → (pct_first, mean_rank). Covers our model over both runs and all four plantings, plus ABMIL's summed attention.
- `explain_summary(cohort) -> dict`:
  - `n_tested`
  - `n_nominal`
  - `n_fdr`
  - `min_q`
  - `region_rho`
  - `feature_rho`
  - `adipose_share_impurity`
  - `adipose_share_perm`
- `profiles() -> DataFrame` with columns `cohort`, `arm`, `thr`, `rule`, `n`, `cindex`, `lo`, `hi`, `cohort_value`, `differs`.
- `auroc_by_arm() -> DataFrame`
- `coverage_by_stage(cohort) -> DataFrame`

- [ ] **Write the tests with the paper's values:**
  - `rho_weight_use` is 0.95 (BLCA) and 0.73 (BRCA); `rho_use_outcome` is 0.88 and −0.27.
  - BRCA `use_share`: DCIS 18.2, necrosis 12.6. BRCA `shapley_pp`: adipose 2.44, muscle 2.20.
  - Planted hit rates: Shapley (51, 2.3), deletion (41, 2.5), weights (36, 3.1), grad×input (18, 4.2), IG (32, 3.3); ABMIL summed attention 0%.
  - Explainability:
    - BLCA: 851 tested, 120 nominal, 0 FDR, min_q 0.064.
    - BRCA: 875, 132, 0, 0.149.
    - region_rho 0.76; feature_rho 0.28 / 0.18; BLCA adipose shares 17 → 61.
  - Profiles: `len(profiles()) == 159`, `differs.sum() == 3`, and the differing rows have C-indices {0.37, 0.33, 0.72}.
  - AUROC: BRCA pathomic q50 is 0.671.
  - BRCA coverage for stage III–IV at the two longer horizons is 0.79 and 0.78.
  - All values are rounded as the paper prints them.
- [ ] **Run** `python -m pytest tests/test_figure_data.py -q`. Expected: FAIL, module not found.
- [ ] **Implement the loaders** from the `results/` files named in the spec.
  - If a value can only be reproduced by a different but defensible aggregation, rule on it in the ledger.
  - Never change a test value to match the code without first finding where the paper's number comes from.
- [ ] **Run the tests again.** Expected: all pass. Then commit.

### Task 3: Data panels (b)
**Files:**
- Create: `visualization/make_audit_panels.py`. It writes `paper_figures/panels/fig{2,3,4}b.{svg,png}`.

**Interfaces:**
- Consumes: Task 2 functions.
- Produces: `fig2b.svg`/`.png` (3.0 × 2.6 in), `fig3b` (3.0 × 2.6 in), `fig4b` (3.0 × 2.8 in).

**Panel contents:**
- **fig2b**:
  - Per cohort, two ρ markers with ✓/✗.
  - A BRCA mini-bar of "moves the score" against "carries concordance".
  - A hit-rate bar chart by measure.
- **fig3b**:
  - A funnel of features: tested → nominal → FDR, per cohort.
  - A region-vs-feature agreement strip.
  - The adipose share under the two estimators.
- **fig4b**:
  - A forest plot of in-profile C-index: the 3 that differ plus 5 others, with the cohort line.
  - The AUROC by arm.
  - Coverage by stage against the 0.90 target.

- [ ] **Generate the panels,** then render the PNGs and inspect them. Expected: fonts at least 6.5 pt (assert on `rcParams` and on every `Text` artist), and no overlaps. Commit.

### Task 4: Fig. 1 deck
**Files:**
- Create: `paper_figures/build_fig1.js` (writes `paper_figures/fig1_ier_overview.pptx`)
- Create: `paper_figures/export.ps1`. It exports any listed `.pptx` to `.pdf` and to a 300-dpi PNG through PowerPoint COM (`SaveAs` 32; `Slides(1).Export`).

**Layout:** 7.0 × 7.6 in.
- **Top band:** the pipeline, using the assets from Task 1.
- **Middle:** one weight node per compartment. The I connector attaches there.
- **Right:** the risk score, where E attaches, and patient error, where R attaches.
- **Below:** three leg panels (claim / check / passes if) and two preliminary-check chips.
- **Bottom:** one tie-together line.

- [ ] **Build and export the deck.** Then render the PNG and inspect it against the Global Constraints. Fix anything, re-export and commit.

### Task 5: Zoom decks (Figs. 2–4)
**Files:**
- Create: `paper_figures/build_zooms.js` (writes `fig2_interp_zoom.pptx`, `fig3_explain_zoom.pptx`, `fig4_reliab_zoom.pptx`)

**Layout:** 7.0 × 3.4 in. Panel (a), the step diagram, sits on the left, about 3.6 in wide. Panel (b) is the Task 3 SVG (or the PNG if SVG fails in the export), about 3.2 in wide.

- [ ] **Build, export and inspect each deck.** Then commit.

### Task 6: Fresh-eyes review of the four PNGs
- [ ] **Run a subagent review** against the spec's anti-goals and the Review Focus. Fix Critical and Important findings and re-export. Commit.

### Task 7: Overleaf
- [ ] **Upload** `fig1_ier_overview.pdf`, `fig2_interp_zoom.pdf`, `fig3_explain_zoom.pdf` and `fig4_reliab_zoom.pdf` to `Figures/`.
  - Route: Overleaf's import from a public URL, after pushing to GitHub. **Ask Juan before pushing.**
  - Fallback: base64 upload through the page.
- [ ] **LaTeX edits,** guarded as before:
  - Replace the `fig:pipeline` TikZ body with `\includegraphics{Figures/fig1_ier_overview}` and a new caption, keeping the label.
  - Add Figs. 2–4 at the start of Results 3.2–3.4, labelled `fig:zoom_interp`, `fig:zoom_explain` and `fig:zoom_reliab`, each cited in the text.
  - Move `fig:wsi_highlights` to the appendix next to `tab:weights`, without its side boxes.
- [ ] **Compile.** Expected: 0 errors and no undefined or duplicated labels. Run `check_paper.py`; expect no number lost. Update the handoff note and commit.
