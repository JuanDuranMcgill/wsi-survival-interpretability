# Paper handoff: AIIM submission (state on 2026-10-22)

Read this first when picking the paper up again, human or Claude. It records
where everything lives, what was done in the October 2026 revision, and what is
left before submission.

## The paper

- **Title:** "Auditing Whole-Slide Image Survival Models: Interpretability,
  Explainability, and Reliability Profiling".
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
| `samplepaper.tex` | An earlier LNCS-format draft; kept for reference |
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
