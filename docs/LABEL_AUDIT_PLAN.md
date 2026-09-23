# Auditing the tissue labels before rebuilding them

## Why

Every result in the paper is computed over tissue compartments defined by a
single zero-shot CONCH call per tile: one prompt template, `"an H&E image of
{class}"`, and an argmax with no confidence threshold. The paper audits
everything downstream of those labels and takes the labels themselves on faith.
For a paper whose argument is that explanations should be tested rather than
believed, that is an inconsistency a reviewer can find.

No pathologist-labelled patches exist, so label accuracy cannot be measured.
Label ambiguity can, because the classifier saved the full softmax over classes
for every tile.

## Step 1: measure the labels we have

`analysis/label_confidence_audit.py`. No GPU, reads the saved JSONL.

Per compartment: top-1 probability, margin to the runner-up, near-tie rate,
entropy; the runner-up distribution, to compare against what the pathologists
flagged (stroma misclassified in BLCA, tumour underestimated in BRCA); and what
a confidence threshold would remove, both as a share of tiles and as the number
of slides still holding enough tiles of that compartment.

The number to watch first is necrosis. It carries the highest fusion weight in
both cohorts and the central claim turns on it. If necrosis is largely near-ties,
the paper has to say so. If it is assigned decisively, the label-noise argument
gains evidence it currently lacks.

## Step 2, conditional on step 1: rebuild the regions and re-audit

- Prompt ensembling: several templates per class with averaged text embeddings,
  which is how CONCH was evaluated in its own paper.
- An uncertain bin for tiles below a confidence threshold chosen from step 1,
  rather than forcing every tile into a class.
- Retrain, then rerun the fusion weights, the ablation and the redundancy
  analysis on the new regions.

That is days of GPU across both cohorts, and it is worth it only if step 1
shows the labels are ambiguous enough to matter. Either outcome is reportable.
If the weight-versus-ablation mismatch survives cleaner labels, the label-noise
argument becomes a tested result. If it shrinks, label noise was part of the
effect, which the paper currently argues cannot happen, and that must be found
by us before a reviewer finds it.

## Caveat that travels with every number

A softmax probability is not an accuracy. These figures measure how decisively
the classifier chose, which bounds the trust owed to the compartment names
without establishing it.
