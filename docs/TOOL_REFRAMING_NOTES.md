# Reframing the paper as a tool: notes (2026-11-06)

Read this when you have energy for it. Nothing here is urgent, and none of it changes the experiments.

## 1. Decisions taken

- **Journal: Artificial Intelligence in Medicine (AIIM).** A venue check on 2026-11-06 confirmed it as the best fit (summary in section 6).
- **Framing:** the IER audit becomes something other groups can run on their own WSI survival models, not only a study of our model.
- **Experiments:** unchanged.

## 2. What "reframe as a tool" means for the paper

Today the paper reads as "we audited our model". The reframed paper reads as "here is an audit any group can run on their WSI survival model; we show it on two models and two cohorts". The experiments become the demonstration.

In practice this means a few text changes, not a rewrite:

- **Introduction, contributions:** state the audit as a reusable procedure with code.
- **Methods:** one short paragraph on applying the audit to another model, listing what each check needs.
  - I: a model whose slides can be re-scored with tissue compartments removed.
  - E and R: out-of-bag risk scores, survival times and patient features.
- **Discussion:** one short paragraph on how others can use it, and its limits.
- **Comparison with other tools.** AIIM asks for one, so place the audit against:
  - HIPPO;
  - xMIL;
  - "Beyond attention heatmaps" (Jamshidi Idaji, Hense et al., *Medical Image Analysis* 2026, doi 10.1016/j.media.2026.104148). This is the closest published work. Check that the paper cites it.
- **Cover letter:** one sentence on the tool.

## 3. The one decision: paper first or library first

**My recommendation: paper first, library during review.**

How it works:

1. Make the light text changes in section 2. Keep the claim modest: "an audit procedure with a reference implementation". Don't write "a library" until one exists.
2. Finish the to-dos already in `docs/PAPER_HANDOFF.md`, then submit to AIIM:
   - addresses;
   - funding sentence;
   - licence;
   - co-author approval.
3. While the paper is under review, which usually takes months, build the library.
4. In the revision, point reviewers to the released library.

Why:

- The paper is finished. Waiting for a library could delay it by months.
- Reviewers will ask for changes anyway, and the revision is the natural moment to point to the library.
- AIIM does not require software, so a reference implementation is enough to submit.
- It splits one big job into two small ones.

The other way, library first, gives a stronger submission but delays it. It is also the larger, riskier job.

## 4. Steps, in order (paper-first path)

1. Text changes on Overleaf (section 2): about one session of work with Claude.
2. Licence for the repo (MIT or BSD-3 are common for research code).
3. In `docs/PAPER_HANDOFF.md`, remove *Computers in Biology and Medicine* from the backups (see section 6).
4. The author-only to-dos from `docs/PAPER_HANDOFF.md`, then submit.
5. Build the library (section 5).
6. Revision: cite the library.
7. Later, optionally, a short software paper on the library alone, in one of these:
   - JOSS: free, but the repo must have been public for 6+ months first.
   - A JAMIA Open Application Note: up to 2,000 words.

## 5. What the library would need (later)

Today each check is a script tied to BLCA/BRCA, the cluster paths and our two models.

- **E and R:** easy. They only need risk scores, survival times and features, which any model gives.
- **I:** the core. It re-scores each slide with some compartments removed:
  - For MIL models, dropping a compartment's tiles works for any architecture; `analysis/abmil_compartment_shapley.py` already does this.
  - Exact Shapley stays cheap because there are only 8–9 compartments (256–512 subsets).
- **Planted-signal test:** the hardest part, because it retrains the user's model. The library would need them to supply a training function. It could come in a second version.

## 6. Venue check summary (2026-11-06)

- **AIIM: best fit.**
  - Its exclusion targets "mere application" of known methods, and a reusable audit is a method.
  - About 23% of its papers since 2023 are on explainability and 36% on imaging.
  - It has published close work: Eijpe 2026 (already cited), Parimbelli 2023, Cabitza 2026, Lee 2026, and the MedCAT tool paper.
- **JAMIA: poor fit.**
  - About half its papers are EHR, NLP or LLM work, and only 4 of about 1,060 since 2023 mention whole-slide images.
  - Its 4,000-word limit would cut about 63% of the paper.
  - MED3pa I (Vallières' lab) is in JAMIA (June 2026), which makes JAMIA Open a good home for the library later.
- **Medical Image Analysis:** stretch. It published "Beyond attention heatmaps", but it expects a new method and many datasets.
- **Journal of Pathology Informatics:** a good backup, and the natural home if the library becomes the main product.
- **Computers in Biology and Medicine:** drop it as a backup. Web of Science put it on hold in April 2024, and its coverage was reportedly ended in November 2025.
- **Journal of Biomedical Informatics:** poor fit. Its scope excludes imaging, and it limits papers to 6,000 words.

Sources:
- AIIM scope: https://shop.elsevier.com/journals/artificial-intelligence-in-medicine/0933-3657
- JAMIA instructions: https://academic.oup.com/jamia/pages/General_Instructions
- MED3pa I: https://doi.org/10.1093/jamia/ocag034
- "Beyond attention heatmaps": https://doi.org/10.1016/j.media.2026.104148
- CIBM status: https://phdtalks.org/2026/01/computers-in-biology-and-medicine-journal-elsevier.html
- JOSS rules: https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future
