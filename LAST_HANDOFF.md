# Last handoff (2026-10-22)

The work laptop used for this project was returned. Everything needed to carry
on is in two places:

- **Overleaf** holds the manuscript, highlights, cover letter and references.
- **This repo** holds the code, every result, and the notes below.

For the full state, see **[docs/PAPER_HANDOFF.md](docs/PAPER_HANDOFF.md)**:
where each file is, what changed, and the to-do list.

## Where things stand

The paper is ready to submit to *Artificial Intelligence in Medicine* once the
author has done these:

1. Read the PDF once.
2. Confirm the MICCAI workshop paper's status. It is marked `TO CONFIRM` in
   `cover_letter.tex`; cite it if it was published.
3. Add the affiliation postal addresses.
4. Confirm the funder-role sentence.
5. Choose a licence for this repo.
6. Get approval from all co-authors.

## Decisions taken, and why

| Decision | Why |
|---|---|
| Kept the IER framing as the paper's spine, with an IER-structured abstract | It is the contribution; an earlier abstract had dropped the E and R legs |
| Moved detail to "Supplementary methods/results" appendices instead of deleting it | The main text went from about 12.3k to 10k prose words with no result lost |
| Called the model "region-based", not "graph-based" | The graph diffusion is used only in training; no audited model uses a graph at prediction |
| Wrote "matched in BLCA, not in BRCA" instead of "weights do not measure reliance" | The weights do track Shapley values in BLCA (ρ = +0.81); the claim had to match the evidence |
| Added masked removal (run S) | Removing a compartment did not exactly equal a slide lacking it; with masking it does, and the result held |
| Called attention "summed over each compartment" in the claims | Summed attention grows with area; per-tissue attention (run M/N) found the driver in 15% of rounds against 18% for Shapley |
| Left these review suggestions out | A p-value column for the AUROC table needs another data pull; moving the model paragraph above Table 1 is structural; the abstract is at the 250-word limit |

## How the paper was checked

- Every batch of edits was dry-run first. Each edit had to match exactly one
  place, and the full set of numbers, labels, citations and cross-references
  was compared before and after, so none was lost.
- All 49 references were verified against Crossref, PMLR, arXiv and the
  publishers.
- An independent agent with no history of the project reviewed the whole
  paper. Its fixes were applied, and the facts it asked about were looked up
  in `results/`.

## To resume with a new Claude session

Open the repo in Claude Code (or any assistant that can read it) and paste:

```text
I'm finishing the paper "Auditing Whole-Slide Image Survival Models:
Interpretability, Explainability, and Reliability Profiling" for Artificial
Intelligence in Medicine. Read LAST_HANDOFF.md and docs/PAPER_HANDOFF.md in
this repo first. The manuscript text is in my Overleaf project
(manuscript_aim.tex, highlights.txt, cover_letter.tex, references.bib), not in
this repo. If you can use my browser (Claude in Chrome), edit Overleaf the way
docs/PAPER_HANDOFF.md describes, dry-running every batch of edits and checking
that no number, label, citation or reference is lost. Keep answers short and
plain. Ask before pushing or submitting anything. Start by telling me which
to-do items remain.
```

## Working preferences

- Keep answers short and plain, without walls of text.
- Be careful with edits: verify against the data before changing a claim, and
  never lose a result.
- Ask before pushing, submitting or sending anything.
