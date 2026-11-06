# Figures that explain the IER audit — design

## Intent

Juan wants a reader to understand the whole audit from the figures alone, quickly and without being overwhelmed. The first figure matters most.

The current Fig. 1 (`fig:pipeline`, TikZ) carries the right information:
- the model pipeline;
- where each leg of the audit attaches.

It is too minimal, though. The page limit is gone, so the figures can be bigger and there can be more of them.

Inspiration comes from the IER section of Juan's FMT deck: `FMT_Nov6_2026_final.pptx`, slides with badges 23–36, at positions 46–59.
- Badge 25, "From slide to tissue compartments": real H&E → CONCH → the slide coloured by compartment → UNI-2 → one vector per tile.
- Badge 26, "The compartment model": tiles of each tissue → a summary per tissue → one weight per tissue → Σ → risk score.
- The rest of the section shows the style that works:
  - large I/E/R cards;
  - the claim in quotes ("This tissue matters most.");
  - "moves the score most ≠ makes predictions correct";
  - the forest plot of flagged groups;
  - "120 → 0" features before and after correction.

## Goals

1. **Fig. 1: the IER audit at a glance.** It is cited in the Introduction and contains no findings.
   - Full pipeline: real WSI → tiles → CONCH compartments → UNI-2 embeddings → per-compartment encoder → cross-region attention → fusion with one learned weight per compartment → risk score.
   - Each leg is drawn where it attaches:
     - (I) at the fusion weights;
     - (E) at the risk score, through a surrogate on clinical and pathomic features;
     - (R) at each patient's prediction error.
   - Each leg shows the claim as usually read (in quotes), how it is checked (a few small visual steps), and what counts as passing.
   - Two small preliminary checks: performance context and tissue labels.
   - It is drawn in a new shape, not the old two-row layout. Juan: "same information, but extended and better".
2. **Figs. 2–4: one zoom per leg (I, E, R).** Each is placed at the start of its Results subsection and is not super detailed.
   - Panel (a): how the check works, in 3–4 simple steps.
   - Panel (b): what it found, using real numbers.
3. **Every number in a figure is already in the paper** and is regenerated from `results/` by a script. Nothing is retyped from the slides.
4. **The paper text cites and captions each figure.** Fig. 1 replaces `fig:pipeline`, keeping the label so existing references still work.

## Non-goals

- No new analyses and no new numbers.
- The existing data figures (weight-vs-Shapley scatter, pathomic importance, clinical importance, ablation) stay as supporting detail. The zooms do not replace them.
- No Python library or tooling yet. That is Juan's next project.

## Anti-goals (failure even if it compiles)

- **Overwhelming panels:** too many numbers or arrows, or tiny text.
- **Findings in Fig. 1:** the Introduction carries no results.
- **Wrong numbers:** any number that differs from the paper.
- **Slide-deck chrome:** McGill footer, navigation bar or slide titles inside a paper figure.
- **"Did not improve prediction" framing:** performance is context.
- **"Everything is garbage" tone:** the message is that explanations need checks.

## Constraints

- **Print size:** full text width of the elsarticle single-column layout, about 6.3 in. Figures are drawn at their final size, so fonts are 7–10 pt in the figure and never below 6.5 pt. Colour must survive greyscale (pass/fail also uses ✓/✗ and labels, not colour alone).
- **Style:** Juan's deck, adapted for print.
  - Navy (`1F2A5C`-like) for the model and structure, light blue-grey cards, red for "fails / differs", green for "passes", grey for neutral.
  - Sans font (Calibri or Arial).
  - No accent stripes and no slide chrome.
- **Medium (Juan chose approach A):**
  - Diagrams are built as slides in a new PowerPoint deck that Juan can edit, then exported to vector PDF through PowerPoint.
  - Data panels are built by Python scripts in `visualization/` from `results/`, saved as PDF and PNG, and placed into the deck as images. Vector if PowerPoint keeps SVG, otherwise PNG at 600 dpi.
  - Output: one PDF per figure.
- **Getting the figures into Overleaf:** Claude uploads the figures later. The Overleaf page blocks fetching local files.
  - The route goes through a public URL that Overleaf imports, for example the public GitHub repo. That needs Juan's OK to push the figure files, so it is decided at that step.
  - Base64 through the browser is the fallback for small files.
- **Real WSI imagery available on this laptop** (the raw `.svs` files are on the clusters, behind Duo):
  - the deck's bladder slide coloured by compartment, 2999×2340 (`ppt/media/image49.png`);
  - an H&E tile crop (`image47.jpg`);
  - Overleaf's `Figures/combined_panel`: BLCA and BRCA slides with the two highest-weighted compartments overlaid, with a known legend.
- **Numbers available in `results/`:**
  - Shapley values (`shapley_*/region_shapley_*.json`);
  - planted outcomes (`planted_*`, `planted_attr_*`, `planted_v3_*`);
  - surrogate–outcome Cox tests (`pathomic_vs_survival_*_cox.csv`: 851 BLCA and 875 BRCA rows, which match the appendix);
  - reliability profiles (`reliability_*_q*/`);
  - conformal coverage (`conformal_survival_*.json`).

## Approaches considered

- **A. PowerPoint diagrams + Python data panels (chosen by Juan).** Diagrams Juan can edit, and data panels that cannot drift from the results. Cost: an export step and an upload step.
- **B. TikZ in LaTeX.** Native to the paper and cheap to transfer as text, but slow to iterate and hard for Juan to edit.
- **C. Everything in matplotlib.** Fully reproducible, but the diagrams come out stiff and every edit needs code.

## Figure content (decided with Juan)

### Fig. 1 — The IER audit at a glance (about full page)

1. **Pipeline band:**
   - Real slide → a few tiles → the same slide coloured by compartment (the deck image).
   - UNI-2 vector per tile → one encoder per compartment → cross-region attention → one learned weight per compartment (w₁…w_R as visible nodes) → Σ → risk score.
2. **Three leg panels, connected by arrows to their attachment points:**

   | Leg | Claim as usually read | Check | Passes if |
   |---|---|---|---|
   | (I) | "This tissue drives the risk" | Hide tissues in every combination; compare the weights with what the risk score depends on and with what the outcome depends on; known-signal test; held-out hospitals | The weights rank tissues as model use does, and model use ranks them as outcome does |
   | (E) | "These features explain the risk" | A second importance estimator; each feature tested against observed outcome with FDR | The order survives the estimator change and features survive FDR |
   | (R) | "The model is more or less reliable for these patients" | Censoring-aware error per patient; rules found on discovery patients; C-index inside each group on held-out patients; conformal bounds checked by subgroup | The group's C-index interval excludes the cohort value |

3. **Two small preliminary checks:**
   - performance context: clinical-only Cox, ABMIL, Kaplan–Meier;
   - tissue labels: classifier confidence.
4. **One line tying it together:** every explanation is checked against the model and against what happened to patients.

### Fig. 2 — Zoom on (I), Results 3.2

- **(a) How it works:**
  1. one weight per compartment;
  2. hide compartments in every combination (2⁸ / 2⁹ coalitions);
  3. two payoffs: each patient's risk score (what the model uses) and out-of-bag concordance (outcome-based contribution);
  4. two comparisons: faithful? prognostic?
  - Side box: plant a known signal in one compartment, then ask which measure ranks it first.
- **(b) What it found:**
  - Per cohort, the two Spearman ρ with pass/fail: BLCA +0.95 / +0.88; BRCA +0.73 / −0.27.
  - In BRCA: moves the score most (DCIS 18.2%, necrosis 12.6%) ≠ carries concordance (adipose +2.44, muscle +2.20 pp).
  - Known-signal hit rates by measure: Shapley 51%, deletion 41%, weights 36%, integrated gradients 32%, gradient×input 18%; summed ABMIL attention never first.

### Fig. 3 — Zoom on (E), Results 3.3

- **(a)** Risk score → surrogate forest on clinical and pathomic features → feature ranking → check 1, change the estimator; check 2, test each feature against outcome with FDR.
- **(b)**
  - The region order holds (adipose first under both estimators), but the share moves: 17% → 61% in BLCA.
  - The outcome funnel: BLCA 851 tested → 120 nominal → 0 after FDR; BRCA 875 → 132 → 0. The strongest q is 0.064 in BLCA and 0.149 in BRCA.

### Fig. 4 — Zoom on (R), Results 3.4

- **(a)** Error per patient over comparable pairs → high-error label → MED3pa rule ("Stage I?") on discovery patients → on held-out patients, two questions: is error predictable (AUROC), and is the C-index different inside the group?
- **(b)**
  - Forest plot of in-profile C-index with 95% intervals against the cohort line, with the 3 of 159 that differ in red.
  - The AUROC by arm: BRCA pathomic 0.671, and the rest.
  - Bound coverage by stage, BRCA III–IV 0.78–0.79 against the 0.90 target.

## Calls made after the builder check (Claude's, for Juan to confirm)

1. **(R) has no direction, as in the paper.**
   - Claim in Fig. 1: "The model is more or less reliable for these patients".
   - Check: the C-index inside the group against the cohort.
   - Passes if: the interval excludes the cohort value.
   - Fig. 4(a) uses a generic rule example that implies no error direction, because the Stage I profile was flagged on the lower-error side yet had a lower C-index.
   - In Fig. 4(b), the 3 profiles that differ are highlighted and labelled "interval excludes cohort value", not "fail".
2. **`fig:wsi_highlights` (the slide panel with the old I/E/R findings boxes) moves to the appendix** next to the weights table, without its side boxes.
   - The boxes only repeated results that Figs. 2–4 now show.
   - Its label and the Results reference keep working.
   - Its slide images can also feed Fig. 1 and Fig. 2.
3. **Fig. 3(b) shows both levels of the estimator check, as the paper does:**
   - region order holds (ρ = +0.76 in both cohorts, adipose first);
   - feature order agrees only weakly (ρ = +0.28 BLCA, +0.18 BRCA);
   - the share is not robust (17% → 61% BLCA).

   No single tick overstates it.

## Left to the builder

- Exact layout, sizes and arrows within the style above.
- Which profiles appear in the forest plot: the 3 that differ plus a readable handful of others.
- Icon choices.
- Caption wording: plain, in the paper's style.
- How data panels are embedded: SVG or 600-dpi PNG.
