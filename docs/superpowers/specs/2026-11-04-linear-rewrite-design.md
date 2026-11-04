# Linear rewrite of the AIIM manuscript — design (short)

**Intent.** Make the paper a linear, engaging read whose point is easy to follow:
- Explanations of WSI survival models look convincing.
- The IER audit checks them.
- Most do not hold up as commonly read.
- More data is needed.

Juan found the current text verbose, repetitive and interpretive, with results showing up in the Introduction.

**Approach chosen.** Option 3: rewrite the Introduction first as a style sample. Once Juan approves its style and length, apply the same rules to every other section in one pass.

The options considered were:
1. Section by section with approval each time: most control, about five rounds.
2. The whole paper in one pass: fastest, but a wrong style would have to be corrected everywhere at once.
3. Introduction first, then the rest: the style is agreed once and then applied consistently.

**Rules (goals).**
- Each section does only its own job:
  - Introduction: why it matters, the hypothesis, what we did.
  - Methods: how.
  - Results: what we found, without interpretation.
  - Discussion: what it means and its limits, including data size and overfitting.
- Writing is plain and linear. There are no mini-arguments, no restating, and no result before Results.
- The Introduction runs in this order:
  1. Context: WSI survival models now report explanations that clinicians would see.
  2. Problem: such explanations are rarely checked and often fail elsewhere.
  3. Gap: there is no procedure to check each kind before reporting it.
  4. Hypothesis.
  5. What we did: the IER audit, with Table 1 listing each explanation and its check, without findings.
  6. Where we applied it: two models and two TCGA cohorts.
  7. Contributions, phrased as things we provide.

**Hypothesis (Claude's proposal, for Juan to confirm).** The explanations a WSI survival model reports mean what they appear to mean:
- the weight of each tissue compartment;
- the features a surrogate ranks highest;
- the patient subgroups flagged as more or less reliable.

The audit tests this.

**Non-goals.**
- No new analyses.
- No change to any number or finding.
- No result is removed from the paper; results live in Results and the Discussion.

**Anti-goals (failure even if it compiles).**
- Blind compression.
- Losing a citation.
- Results or interpretation in the Introduction.
- A paper that reads as "everything is garbage" rather than "explanations need checking".

**Constraints.**
- AIIM; British spelling.
- Every `\cite` in the current Introduction is kept, in a lean related-work part or moved where it belongs.
- Every `\label` and `\ref` still resolves.
- It compiles with 0 errors.
- Edits go into Overleaf with checksum-guarded replacements.

**Left to the writer.** The exact wording, within these rules.
