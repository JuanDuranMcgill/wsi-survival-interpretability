# Do tissue compartments carry duplicate information?

## Why

The paper's ablation section closes with a caveat it does not resolve:

> ablation measures each region's unique contribution, and information duplicated
> across regions will be invisible to it; the near-zero deltas are consistent
> either with a region carrying no signal or with its signal being available
> elsewhere.

That ambiguity sits directly under the paper's central claim. Necrosis has the
highest fusion weight in both cohorts and an ablation effect indistinguishable
from zero. We read that as "the weight does not track reliance". A reviewer can
read it as "ablation cannot see necrosis because its signal is duplicated in the
other compartments". Both readings fit the evidence as it stands.

## The cheap version, which is also the better one

The first instinct was to reload trained checkpoints and measure redundancy in
the learned 384-dim region embeddings `h_r`. That turns out to be a bad plan:
the training script deletes each checkpoint after converting it to that epoch's
npz, so BLCA has none at all and BRCA retains only seven rounds, surviving as
debris from jobs killed at walltime.

It is also unnecessary. Redundancy between compartments is in large part a
property of the *input representation*, and the per-region UNI-2 tile features
are still on disk for both cohorts. Mean-pooling them per region gives one
1,536-dim vector per patient per compartment, with no model, no GPU and no
training. That covers BLCA and BRCA rather than BRCA alone.

## What is measured

For each patient and region, mean-pool the UNI-2 features of that region's
tiles. Then ask, per region r: how much of `x_r` is predictable from the other
regions' pooled embeddings?

- Reduce each region to `k` principal components (default 16), fit within the
  training fold only.
- Ridge-regress the other regions' components onto region r's components.
- Report out-of-fold R², averaged over r's components weighted by the variance
  each explains. High R² means region r is redundant given the others.

**The control matters more than the headline.** Two compartments on one slide
share staining, scanner and fixation, which produces correlation with no shared
biology. The paper already reports strong TCGA source-site effects on the risk
score, so this is not hypothetical. Every quantity is therefore computed twice:

1. on the raw pooled embeddings, and
2. after subtracting, for each patient, the mean of that patient's own region
   embeddings, which removes the slide-level component.

The gap between the two is the part of the apparent redundancy that is a
slide-level technical signature rather than shared tissue content.

Also reported, because it is easier to explain than a cross-validated R²: the
mean absolute correlation between each pair of regions on their leading
components.

## How to read the result, including what it cannot settle

The test is one-sided, and the write-up must say so.

- **High redundancy** confirms the confound. Ablating one compartment leaves its
  information available through the others, so a near-zero delta says nothing
  about reliance, and the paper's central comparison has to be softened.
- **Low redundancy** is evidence for the paper's reading but does not close the
  question, because cross-region self-attention can create redundancy the inputs
  did not have.

If input redundancy comes back low, the seven surviving BRCA checkpoints become
worth spending: repeating the measurement on the learned `h_r` would test
whether the model manufactured the duplication. That is a follow-up, not a
prerequisite.

## Cost

I/O bound, no GPU. 379 x 8 file reads for BLCA and 1,060 x 9 for BRCA, each
streamed and reduced to a mean immediately, so memory stays flat.

## Outputs

`results/region_redundancy_{cohort}.json`, holding per-region out-of-fold R²
with and without slide-level centering, the pairwise correlation matrix, tile
counts per region (a region pooled from very few tiles has a noisy mean and its
number should not be over-read), and the number of patients contributing.
