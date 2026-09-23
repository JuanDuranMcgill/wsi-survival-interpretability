# Exact Shapley values over tissue compartments

## Why

The paper tests the fusion weights against single-region deletion, and deletion
has a limitation the paper states: it measures what a region adds given all the
others, so information shared between two regions is credited to neither. The
redundancy analysis showed that does not explain the necrosis result, but it
leaves deletion as the only functional measure.

Shapley values remove the limitation. A region's value averages what it adds over
every combination of the other regions, so shared information is split between
the regions that carry it. For two regions that carry identical information,
deletion scores each at zero, while Shapley gives each half the credit and marks
the pair with a negative interaction.

## Why it is exact here, and why that matters

On images Shapley values are approximated, because the players are thousands of
patches. Here they are 8 (BLCA) or 9 (BRCA) tissue compartments, so all 256 or
512 coalitions are evaluated and the values are exact. That is an advantage of
modelling at the compartment level that patch-level models do not have.

## What is computed, from one table of coalition values per round

- Global Shapley values with out-of-bag concordance as the payoff, in the spirit
  of SAGE (Covert, Lundberg and Lee 2020). They sum to the model's concordance
  minus 0.5.
- Per-patient Shapley values on the risk score, summarised as each region's
  share of the attribution: on the same scale as a fusion weight, and with a
  spread across patients that shows how far one cohort-level weight is from
  describing individuals.
- The Shapley interaction index for every pair (Grabisch and Roubens 1999).
  Necrosis with adipose is the pair of interest.
- Single-region deletion from the same table, so all attributions come from the
  same trained models.

Removal matches region_ablation.py exactly, and training matches it exactly,
including the per-round seed, so round k uses the same out-of-bag patients as
round k of the published ablation.

## Validation, before any cluster time

- Games with known answers: an additive game returns its coefficients with zero
  interaction; pure synergy and pure redundancy return +1 and -1 interaction
  with the credit split equally; efficiency holds to machine precision.
- The cached-embedding shortcut reproduces the real model and every
  single-region deletion from region_ablation.py to about 6e-8, including a
  patient with a missing region.
- End to end on synthetic slides with a planted signal in one region: that
  region takes +42.6 points of Shapley value and 60% of the per-patient
  attribution; the noise regions sit near zero. Resume, checkpoint cleanup and
  aggregation all work.

## Cost

Training dominates, as in the ablation, and each round is cheaper than an
ablation round: the ablation reran the full model twice per region, whereas this
encodes each out-of-bag patient once and scores every coalition through the
attention and fusion head only.

## Reading the result

If Shapley ranks adipose high and necrosis near zero, the central claim rests on
three functional measures that agree with each other and disagree with the
weights. If Shapley credits necrosis substantially, its value must come from
coalitions deletion cannot see, and the interaction index will say which
partners it depends on. Either way the paper learns something it cannot learn
from deletion alone.
