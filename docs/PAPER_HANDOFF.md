# Paper handoff: AIIM submission (state on 2026-10-22)

Read this first when picking the paper up again, human or Claude. It records
where everything lives, what was done in the October 2026 revision, and what is
left before submission.

## The paper

- **Title:** "Towards Trustworthy Explanations of Whole-Slide Image Survival
  Models: The IER Audit" (since 2026-11-05; previously "Auditing Whole-Slide
  Image Survival Models: Interpretability, Explainability, and Reliability
  Profiling").
- **Target journal:** Artificial Intelligence in Medicine (Elsevier). The
  backups are Computers in Biology and Medicine and Journal of Pathology
  Informatics. Medical Image Analysis is a stretch target.
- **Framing:** the IER audit. Each explanation a WSI survival model reports is
  paired with a check of whether it means what it appears to mean:
  - **I:** per-compartment fusion weights or attention, checked against exact
    Shapley values.
  - **E:** surrogate feature importance, checked against outcome with FDR
    control.
  - **R:** MED3pa error profiles, checked by the concordance inside each
    profile, plus conformal bounds.
- **Status:** content complete. An independent review rated it "ready after
  light edits", and those edits are applied. It compiles with 0 errors.

## Where everything is

**Overleaf** (the source of truth for the text), in the project titled
"Towards Trustworthy WSI Survival Analysis: …". Use the project last modified
in October 2026, not the one whose title ends in "original".

| File | Content |
|---|---|
| `manuscript_aim.tex` | Main document: the elsarticle AIIM version |
| `highlights.txt` | The 5 highlights, kept identical to the manuscript; Elsevier wants this as a separate file |
| `cover_letter.tex` | Cover letter; compile it on its own |
| `references.bib` | Bibliography; all 49 cited entries verified as real, metadata corrected, DOIs added |
| `samplepaper.tex` | Old LNCS-format draft, kept only as a reference for what the AIIM version builds on. Nothing from it goes into the manuscript or cover letter |
| `Figures/` | Includes `fig_weight_vs_shapley.pdf`, the new main-text figure |

**This repo:**

- Code is on `main`. All review-run results are under `results/`:
  - K: `permuted_outcome_*`
  - L: `region_shapley_last_*`, `shapley_last_*`
  - M: `abmil_shapley_v2_*`, `abmil_v2_*`
  - N: `planted_v3_*`
  - S: `region_shapley_masked_*`, `shapley_masked_*`
- `visualization/make_weight_vs_shapley_figure.py` regenerates the new figure
  from `results/shapley_{blca,brca}/`. It uses exact permutation p-values, which
  match the paper: 0.022 and 0.10.
- `slurm/launch_review_runs.sh` and `slurm/aggregate_review_runs.sh` launch and
  aggregate runs K–N and S on Narval or Trillium. See
  `docs/COMPARTMENT_EXPERIMENTS_PLAN.md`, sections K–N and S.

## What the October 2026 revision did

1. **Ran the reviewer-response experiments**:
   - K: shuffled outcomes.
   - L: Shapley at the last epoch.
   - M: ABMIL attention per unit of tissue.
   - N: planted outcomes, third run.
   - S: Shapley with absent compartments masked out of attention, so that
     removal equals absence. This needs `--mask-absent`.

   All of them support the main result.
2. **Wrote those results into the paper**, with new rows in the Runs appendix
   table.
3. **Cut the main text from about 17k to about 10k words.** Heavy detail moved
   verbatim to "Supplementary methods" and "Supplementary results" appendix
   sections; nothing was deleted.
4. **Added the weight-vs-Shapley figure** to the main text.
5. **Brought the paper in line with AIIM's guide for authors**:
   - abstract at 250 words or fewer, IER-structured
   - a Funding section in Elsevier's format
   - the sex/gender definition stated
   - British spelling throughout
   - every table and figure cited, with an adequate caption
6. **Ran an independent fresh-eyes review** and applied its fixes:
   - corrected a wrong count (three BLCA tests below 0.50, not four)
   - replaced a wrong sentence about profile intervals with the real facts
   - defined jargon
   - made claims consistent across the abstract, highlights, Table 1,
     Results, Discussion and Conclusion
   - gave each Results section an opening question

Every edit was checked automatically so that no number, label, citation or
cross-reference was lost.

## Reframing applied on 2026-10-27

- **Central claim:**
  - Before: the weights matched reliance in BLCA but not BRCA.
  - Now: the weights match what the model uses in both cohorts, but what it uses carries outcome signal only in BLCA.
- **What the two measures are:**
  - The concordance-payoff Shapley value is "outcome-based contribution".
  - The label-free per-patient Shapley share (`local_attribution_share`, risk score as payoff) is "what the model uses". It is now a "Model use" column in `tab:shapley`.
- **Performance is context, not a goal.** The paper's aim is to understand the model's predictions, not to improve them. The model is described as performing "comparably to ABMIL", never as "did not improve prediction".
- **Novelty is narrowed.** New citations: HIPPO (`kaczmarzyk2024hippo`), `albuquerque2024characterizing`, `zhou2022feature` and DPsurv (`xing2025dpsurv`).
- Full text and number sources: `docs/DRAFT_AIIM_REFRAMING.md`.

**Linear, no-repetition pass (also 2026-10-27).** Text was moved, not deleted; every number is still in the paper.
- **Table 1:** the "what the check changed" column moved to a new summary table, `tab:findings`, at the start of the Discussion.
- **Results 3.2:**
  - The main-text ablation subsection was removed because it duplicated Appendix `app:ablation`. A pointer sentence keeps the definition of "pp".
  - `tab:weights` moved to a new appendix section, `app:weights`.
  - The diffusion, masked-removal and last-epoch details moved to a new appendix section, `app:robust`, with one summary sentence left in the main text.
- **Discussion, from 9 subsections to 5:**
  - Principal findings, which now includes the clinical-variables paragraph and the summary table.
  - Interpretability, which now includes the source-site paragraph; `sec:bounds` moved with it.
  - Explainability and reliability.
  - Clinical use and reporting.
  - Limitations and future work.
- **Two leftovers of the old framing were fixed:** the clinical-use paragraph and reporting recommendation 2.
- **Result:** it compiles with 0 errors and no undefined or duplicated labels; the PDF is 66 pages and the main text is about 11,200 words.

**Proofreading pass (2026-10-27).** Three readers checked the whole paper; 79 verified fixes were applied.
- Old-framing leftovers were removed.
- Every number now names its cohort.
- Pointers to moved material were repaired.
- Undefined abbreviations are now spelled out: AUROC, PFI, ABMIL, UC, BC and E-MAD.
- Repeated sentences were removed.
- All Spearman p-values are now exact permutation values, as in Results 3.2:
  - weights vs model use: 0.001 (BLCA) and 0.031 (BRCA);
  - model use vs outcome: 0.007 (BLCA) and 0.49 (BRCA);
  - in the robustness appendix: 0.046, 0.10, 0.022 and 0.10.
- The abstract is a single paragraph (as in 37 of the 40 latest AIIM papers on PubMed), 245 words.

**Introduction rewrite (2026-11-04).** First step of a linear rewrite in which each section does only its own job (design: `docs/superpowers/specs/2026-11-04-linear-rewrite-design.md`).
- **New order:** context → problem → gap → hypothesis → what we did (IER audit, Table 1) → where we applied it → contributions.
- **Hypothesis stated explicitly:** the explanations a WSI survival model reports mean what they appear to mean (compartment weights, top surrogate features, flagged subgroups); the audit tests this.
- **No results in the Introduction any more:** the findings it used to preview are all still in Results and the Discussion.
- **What stayed the same:** all 31 citations, Table 1 and Fig. 1 are unchanged.
- **Length:** prose went from 1,366 to 928 words.
- **Compile:** 0 errors and no undefined or duplicated labels; the PDF is now 65 pages.
- "One feature extractor (UNI-2)" and "two architectures" were in the Introduction's old scope sentence, which was dropped. Both are now in Limitations.

**Rest of the paper (2026-11-04).** The same rules were applied to Methods, Results, the Discussion and the Conclusion. There were 57 edits, the main text went from 11,201 to 10,709 words, and the PDF from 65 to 63 pages. Plan: `docs/superpowers/plans/2026-11-04-linear-rewrite-rest.md`.
- **Methods holds only methods:**
  - The label-decisiveness result moved to Results 3.1.
  - The logit range moved to Results 3.2.
  - The 106-versus-100-round detail moved to Appendix `app:training`.
  - The "why labels don't matter" argument was merged into Limitations.
  - The MED3pa classifier definition and the source-site test design moved in from Results.
- **Results holds only findings.**
  - Interpretive markers went from 26 to 1, and the one left is a figure caption.
  - The closing "central result" paragraph and the "In short" summary are gone.
  - Section 3.1 is now "Performance Context and Tissue Labels", matching the rows of Table 1.
- **The Discussion states the answer once.** Principal findings answers the Introduction's hypothesis: "held only in part". The later subsections give the reasons rather than restating it. The clinical-use, surrogate and recommendation-2 paragraphs no longer repeat results.
- **Data size leads the Limitations:** event counts, and the shuffled-outcome weight ordering as a sign of what a model learns from so few events. "More events" was added to future work.
- **The Conclusion** follows the hypothesis, then I/E/R, then the limits (under 200 events per cohort, no external validation).
- **Unchanged:** the abstract (it already ran linearly) and every number. All 470 distinct numbers and all 53 citations are still in the paper.
- **Typo fixed:** "fea2ture" in the manuscript highlights; `highlights.txt` was already correct.
- **Compile:** 0 errors, no undefined or duplicated labels, and the same 4 known BibTeX warnings.
- **Fresh review:** an independent reviewer compared the old and new text, found nothing that changes a number or a result, and 15 wording fixes were applied: overstatements softened, cited work described as before, setup sentences restored, and remaining repeats cut. The live Overleaf SHA is f0ad0b3f.

**Thesis carried through (2026-11-05).** One idea now runs from title to Discussion: an explanation is a hypothesis until it is checked.
- **Title:** "Towards Trustworthy Explanations of Whole-Slide Image Survival Models: The IER Audit". The cover letter, README and LAST_HANDOFF were updated to match.
- **Abstract:**
  - The planted-signal sentence is shorter.
  - New closing sentence: unchecked explanations are hypotheses, not findings, and with fewer than two hundred events per cohort, settling them needs more data.
  - The abstract is about 249 words, under the 250-word limit.
- **Introduction:**
  - It now says that many models are trained and explained on single TCGA cohorts with at most a few hundred events (`chen2022pancancer`, `duran2025beyond`, `eijpe2026dimafx`).
  - The thesis line is in the gap paragraph.
- **Discussion:**
  - Principal findings has a short paragraph, "Explanations are hypotheses until checked".
  - Limitations notes that small data limits the field, not only this study, citing the same three papers.
- **Unchanged:** no number, citation or result. The live Overleaf SHA is 630d2152, it compiles with 0 errors, and the PDF is 64 pages.

**Audit framed as a tool (2026-11-06)**, following section 2 of `docs/TOOL_REFRAMING_NOTES.md`:
- **Introduction:** the gap sentence now asks for a procedure "any group can run on its own model". Contribution 1 is "a reusable procedure with a reference implementation". Contribution 4 is "a demonstration of the full audit".
- **Methods:** new subsection "Running the audit on another model" (`sec:portability`), saying what each check needs.
- **Discussion:** new paragraph "Relation to other tools" comparing the audit with HIPPO, xMIL and "Beyond attention heatmaps" (`jamshididaji2026beyond`, added to `references.bib`; metadata checked on Crossref and arXiv). It also states the audit's limit.
- **Cover letter:** one added sentence on the tool.
- **Abstract and highlights:** unchanged.
- **Compile:** 0 errors, no undefined references, 69 pages.

**New figures (2026-11-06).** Design: `docs/superpowers/specs/2026-11-06-figures-design.md`. Plan and ledger: `docs/superpowers/plans/2026-11-06-ier-figures.md`.
- **Fig. 1 (`fig:pipeline`), the IER audit at a glance:**
  - Top: the pipeline, real slide → tiles → compartments → encoders → one weight each → risk score.
  - Bottom: the I, E and R panels drawn where each attaches, each with "what it seems to say", "how the audit checks it" and "passes if".
  - It contains no findings and replaces the TikZ figure.
- **Figs. 2–4 (`fig:zoom_interp`, `fig:zoom_explain`, `fig:zoom_reliab`):** one zoom per leg at the start of Results 3.2, 3.3 and 3.4. Panel (a) explains the check; panel (b) shows what it found.
- **`fig:wsi_highlights` (the slide panel)** moved to Appendix `app:weights`, without its old I/E/R side boxes.
- **Sources** are in `paper_figures/`:
  - editable PowerPoint decks: `fig1_ier_overview.pptx` and `fig{2,3,4}_*_zoom.pptx`, built by `build_fig1.js` and `build_zooms.js`;
  - `export.ps1` exports PDF and PNG through PowerPoint;
  - data panels come from `visualization/make_audit_panels.py`, which reads `visualization/figure_data.py`;
  - `tests/test_figure_data.py` (11 tests) pins every plotted number to the paper, using the paper's rounding.
- **Images:**
  - The H&E thumbnail is TCGA-FJ-A3ZE, cut from `combined_panel`, with a median filter to remove the old overlay dots.
  - The compartment map comes from the FMT deck and is a different BLCA slide; the caption says so.
  - Its per-colour legend is unknown; ask Juan if names are wanted.
- **Size:** the text column is 390 pt (5.4 in). The figures are placed 6.3–6.4 in wide, centred into the margins, so print text stays at 6.5 pt or more. Elsevier re-sets them at 190 mm anyway.
- **Overleaf:**
  - The PDFs were imported from the public GitHub repo as linked files, so they refresh from the URL.
  - The live SHA is 2a839796. It compiles with 0 errors, no undefined or duplicated labels and no float too large, and the PDF is 68 pages.
  - Not yet checked by eye in the PDF.

## Facts looked up from `results/` (now in the paper)

- **The 3 MED3pa profiles (of 159) whose concordance interval excludes the
  cohort value:**
  - Two are the same rule at two thresholds, BRCA stage I, with *lower*
    concordance (0.37, 0.33).
  - One is a BRCA age band, with *higher* concordance (0.72).
  - Source: `results/reliability_*/reliability_*.json`.
- **Unrescaled integrated Brier score:** 0.544 (BLCA) and 0.306 (BRCA), against
  Kaplan-Meier's 0.361 and 0.270. Source: `results/integrated_brier_*.json`.

## Still to do before submission

**Plan for framing the audit as a tool (2026-11-06):** see `docs/TOOL_REFRAMING_NOTES.md`.
- **Recommended order:** paper first, library during review.
- **Venue check:** AIIM confirmed. Computers in Biology and Medicine is dropped as a backup.
- **Paper changes:** a few text edits.
- **Licence:** still needed.

**Only the author can do these:**

- [ ] Read the compiled PDF once, start to finish.
- [ ] Add full postal addresses for the four affiliations. AIIM requires
      them; they are not filled in.
- [ ] Confirm the Funding sentence that the funders had no role in the study.
- [ ] Choose a licence for this repo. It is public and has no LICENSE file.
- [ ] Get approval from all co-authors: Saieg, St-Jean, Enger and Vallières.

**On submission day:**

- [ ] Tag a GitHub release matching the submitted version.
- [ ] Export each figure as a separate file (Elsevier requires this).
- [ ] Complete the competing-interests declaration in the Elsevier tool (a
      Word file).
- [ ] Upload `highlights.txt` as its own file and the compiled cover letter.

**Optional:**

- [ ] Add a p-value column to the AUROC table (`tab:auroc`).
- [ ] Move the model-description paragraph above Table 1 in the Introduction.

## Practical notes for whoever resumes

- **Editing Overleaf with Claude in Chrome:**
  - Edit the open file by dispatching changes through its CodeMirror view:
    `document.querySelector('.cm-content').cmView.view`.
  - Match old text with whitespace-tolerant patterns, and check that each
    pattern matches exactly once.
  - Before applying, simulate the result and diff the set of numbers, labels,
    citations and `\ref`s, so nothing is lost.
  - The JavaScript tool truncates long outputs. To read long text, post it with
    a `BroadcastChannel` to a second Overleaf-origin tab and read that tab with
    `get_page_text`, which handles about 45k characters per call.
- **Clusters:**
  - Narval and Trillium require Duo multi-factor login, which an agent cannot
    complete.
  - Trillium GPU jobs must be submitted from `trig-login01`.
  - The group's Trillium fair-share fell to about 0.1 after the October runs,
    so expect long queues there.
  - On Narval's 40 GB A100 the graph model needs batch size 8.
- **Patient order** has been sorted since 2026-09-28, so splits match across
  clusters. Runs from before that date do not pair with newer ones.
