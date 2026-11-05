# Linear Rewrite of Methods, Results, Discussion and Conclusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply the approved Introduction rules to the rest of `manuscript_aim.tex`:
- Methods: how.
- Results: findings without interpretation.
- Discussion: meaning and limits, including data size and overfitting.
- Conclusion: short.

**Architecture:**
- Make targeted, anchor-based edits to a local copy of the live document. The copy has SHA f80a4e2b, taken after the Introduction was applied.
- Check the result against the old document with one script.
- Apply it to Overleaf as one checksum-guarded replacement, then recompile.

**Tech Stack:** Python for edits and checks; Overleaf through the browser pane.

**Spec:** `docs/superpowers/specs/2026-11-04-linear-rewrite-design.md`. The Introduction sample was approved by Juan on 2026-11-04.

## Global Constraints

- No number or finding changes, and no result removed. Moving text between Methods, Results, the Discussion and the appendix is fine.
- Every `\cite` key, `\label` and `\ref` survives and resolves; compile has 0 errors; British spelling.
- **Abstract:** change it only if it conflicts with the new text. Juan has already worked through the abstract.
- **Highlights:** fix the typo "fea2ture" there.

## Edits

**Methods**
- Move the label-decisiveness finding to Results 3.1.
- Merge "Why the central result does not depend on label accuracy" into the Discussion's label limitation.
- Move the logit range to Results 3.2.
- Move the 106-versus-100-round detail to Appendix `app:training`.
- Strip the result from the held-out-hospitals motivation.
- Move the MED3pa secondary-classifier definition here from Results.

**Results**
- Remove every interpretive clause ("therefore", "this is why", "so the signal…", "appears to depend", "would mislead").
- Remove the closing "central result" paragraph and the "In short" summary.
- Each removed idea lives in the Discussion; add it there if it is missing.

**Discussion**
- State the main answer once, in Principal findings.
- Write the other subsections as the reasons behind it, not restatements.
- Add data size and overfitting as the frame for the limits, using only existing results: event counts, the shuffled-outcome control, and the clinical-variable variance fitted versus held out.
- Add "one feature extractor (UNI-2)" and "two architectures" to Limitations.

**Conclusion:** the answer, what it means, and what is needed next, in about 150 words.

## Review Focus

- **A number disappears with a cut sentence.** Check: every number token in the old main text is still in the new paper (`number_check`).
- **A citation or label is lost, or a `\ref` dangles.** Check: `cite_check`, `label_check`, and the compile log.
- **Interpretation remains in Results.** Check: interpretive-marker count in Results falls to at most 2, and each remaining one is listed and justified.
- **A claim in the Discussion is new rather than moved.** Check: every Discussion addition is traceable to an existing result (manual read).
- **Length falls without losing content.** Check: words per section, old versus new.

### Task 1: `check_paper.py old new` (run old-vs-old first; expect the interpretation check to FAIL)
### Task 2: `edits_rest.py` produces `manuscript_rest.tex`; run the checks until ALL PASS
### Task 3: Show Juan the summary; apply with a guarded full-document replacement; recompile; add a handoff note; commit locally
