# Introduction Rewrite (style sample) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Introduction of `manuscript_aim.tex` with a short, linear one that sets up the question without results, as the style sample for the rest of the paper.

**Architecture:**
- Pull the current Introduction from Overleaf.
- Draft the new one in a local `.tex` file.
- Check it with a script.
- Show it to Juan.
- Apply it to Overleaf with a checksum-guarded replacement.
- Recompile.

**Tech Stack:** LaTeX (elsarticle) in Overleaf, edited through the browser pane; Python for the local checks.

**Spec:** `docs/superpowers/specs/2026-11-04-linear-rewrite-design.md`

## Global Constraints

- Each section does only its own job. The Introduction covers why it matters, the hypothesis, and what we did.
- Introduction order:
  1. context
  2. problem
  3. gap
  4. hypothesis
  5. what we did (IER audit; Table 1 without findings)
  6. where we applied it
  7. contributions, phrased as things we provide
- No results, interpretation, mini-arguments or restating in the Introduction.
- Hypothesis: "the explanations a WSI survival model reports — the weight of each tissue compartment, the features a surrogate ranks highest, the patient subgroups flagged as more or less reliable — mean what they appear to mean; the audit tests this."
- No number or finding changes anywhere; no result removed from the paper.
- Every `\cite` key in the current Introduction is still cited somewhere in the paper.
- Every `\label` and `\ref` resolves; 0 compile errors; British spelling.

## Review Focus

- **A citation that only the old Introduction carried silently disappears.** Expect every old key to be present in the new text or elsewhere in the paper. Check: Task 2, `cite_check`.
- **A result sneaks back in through the contributions or the Table 1 caption.** Expect no ρ, concordance, AUROC or percentage values in the Introduction, and no "we found" phrasing. Check: Task 2, `no_results_check`.
- **Table 1 loses a row or its `tab:audit` label, or `fig:pipeline` loses its reference.** Expect the same rows and labels as before. Check: Task 2, `label_check`.
- **Text elsewhere that points into the Introduction breaks.** Expect every `\ref` in the paper to still resolve after the swap. Check: Task 3, compile log has no undefined references.
- **The new Introduction is not actually shorter or more linear.** Expect fewer words than the current Introduction prose, about 1,380. Check: Task 2, word count printed.

---

### Task 1: Fetch the current Introduction

**Files:**
- Create: `<scratchpad>/intro/intro_old.tex`. This is the text from `\section{Introduction}` up to, but not including, `\section{Methods}`, copied from the live Overleaf document, together with its SHA-256.

- [ ] **Step 1:** Read the Introduction from the open `manuscript_aim.tex` in the browser pane, through `cmView.view`, and save it exactly.
- [ ] **Step 2:** Record the full-document SHA-256 for the guarded replacement in Task 3.

### Task 2: Draft and check the new Introduction

**Files:**
- Create: `<scratchpad>/intro/intro_new.tex`
- Create: `<scratchpad>/tools/check_intro.py`

**Interfaces:**
- Consumes: `intro_old.tex` from Task 1.
- Produces: an `intro_new.tex` that passes all checks, for Task 3.

- [ ] **Step 1: Write `check_intro.py old new paper`.** It prints:
  - `cite_check`: old keys missing from both the new Introduction and the rest of the paper. The pass condition is an empty set.
  - `label_check`: old `\label`s, the Table 1 rows (counted by `\hline`-separated rows) and the `\ref{fig:pipeline}`. All must be present.
  - `no_results_check`: matches of `\\rho`, `concordance of`, `AUROC`, `\d+\s*\\%`, `0\.\d{2,}` and "we found" in the new prose. The pass condition is no matches.
  - The words of new versus old prose, with the table excluded.
- [ ] **Step 2: Run it on old versus old** to confirm it flags results. Expected: `no_results_check` reports the current ρ and concordance values.
- [ ] **Step 3: Write `intro_new.tex`** in the Global Constraints order. Table 1 stays as it is now (questions and checks only). The related-work citations go into one lean "Problem" paragraph.
- [ ] **Step 4: Run `check_intro.py` on old versus new.** Expected: all checks pass, and the new prose word count is below the old one.

### Task 3: Juan reviews; apply to Overleaf

- [ ] **Step 1:** Show Juan the new Introduction as readable text and wait for his approval or edits.
- [ ] **Step 2:** Apply it with a guarded replacement. Overleaf's SHA-256 must equal Task 1's value, the old Introduction must occur exactly once, and the result must hash to the expected value.
- [ ] **Step 3:** Recompile. Expected: 0 errors, 0 undefined citations or references, 0 multiply defined labels.
- [ ] **Step 4:** Add a line to `docs/PAPER_HANDOFF.md` and commit it locally. Push only if Juan asks.

### Follow-up (separate plan, after Juan approves the sample)

Apply the same rules to Methods, Results, Discussion, Conclusion and the abstract, with the same checks generalised: citations, labels, every number still present, and section word counts.
