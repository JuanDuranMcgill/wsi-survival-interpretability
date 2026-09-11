# Design spec: region-level GA²M head (Step 6)

An alternative aggregation head for the existing backbone, in which each tissue
region contributes an explicit scalar to the risk score and each region *pair*
contributes an explicit interaction scalar. Region encoders are untouched.

**Why.** Our ablation showed that the current per-region fusion weight does not
predict what happens when a region is removed: necrosis carries the highest
weight in both cohorts and its ablation costs nothing measurable, while adipose
is mid-ranked by weight and produces the largest effect. The cause is
structural. Cross-region self-attention mixes the region embeddings before the
weighted sum, so the weight applies to an already-mixed representation and no
longer corresponds to one tissue type.

The naive fix, dropping the attention for a pure additive model, loses the
cross-region interaction the model can currently represent. That objection is
answered by the GA²M family (Lou et al., KDD 2013; GAMI-Net; NODE-GA²M), which
keeps interactions but makes each one its own readable term:

```
risk(patient) = Σ_r f_r(h_r) + Σ_{r<s} f_rs(h_r, h_s)
```

The patch-level precedent in pathology is Additive MIL (Javed et al. 2022),
already cited in the paper as reference [9]. The region-level version for
survival appears to be open ground.

---

## 1. Architecture

Keep unchanged: `MultiRegionDataset`, `forward_bag` (per-region projection,
k-NN graph, graph attention, attention pooling), producing `h_r ∈ R^384` for
each of the R regions.

Replace: `cross_region_attn` → `cross_region_norm` → `region_weights` weighted
sum → `head`.

With:

**Main effects.** One small MLP per region, `384 → 64 → 1`, GELU. R of them.

**Interaction terms.** A low-rank bilinear form rather than an MLP per pair.
Per-pair MLPs would cost more parameters than the attention block they replace,
which defeats one of the motivations. Instead, with rank `k = 16`:

```
f_rs(h_r, h_s) = ⟨ A_r h_r , B_s h_s ⟩ + ⟨ A_s h_s , B_r h_r ⟩
```

where `A_r, B_r ∈ R^{k×384}` are per-region projections shared across all pairs
involving r. Parameter cost is `2·R·k·384` (about 98k at R=8, k=16) against
roughly 590k for the 4-head attention plus norms, so this is a genuine
reduction. The symmetric form keeps `f_rs = f_sr`.

**Sparsity on interactions.** An L1 penalty on the per-pair output magnitudes,
weight `λ_int` (start at 1e-3, tune on the pilot). Pairs that go to zero are a
finding in themselves: they say which regions genuinely interact.

### Identifiability — do not skip this

An additive decomposition is not unique without centering constraints. A
constant can move freely between main effects and interactions, and any function
of `h_r` alone can move from `f_rs` into `f_r`. Without fixing this, the
per-region attributions are arbitrary and the entire purpose is lost.

Enforce, per training batch:

1. Centre each main effect: subtract its batch mean from `f_r` outputs.
2. Centre each interaction to have zero batch mean, **and** zero conditional
   mean with respect to each of its two marginals. In practice, after computing
   the pairwise term for a batch, subtract its mean, then subtract the mean
   within bins of `f_r` and of `f_s`. GAMI-Net calls this the marginal clarity
   constraint; a simpler batch-level double-centering is sufficient here.
3. Carry a single global intercept absorbing the removed constants.

Record the centering in the output so it is reproducible.

**Heredity constraint (optional, GAMI-Net style).** Only allow an interaction
`f_rs` if both `f_r` and `f_s` are non-negligible. Simplifies the reported
interaction map. Try without first.

---

## 2. Training protocol

Identical to the existing backbone so the two are directly comparable:

- Cox loss, same optimiser settings, same epochs per round
- Same bootstrap protocol: `train_frac = 0.8` with replacement, out-of-bag
  remainder for validation
- **Same `seed_offset` scheme**, so round *n* of the GA²M run sees the same
  patients as round *n* of the original run. This makes the comparison paired,
  which matters at these effect sizes.
- PFI endpoint

**Pilot first: 10 rounds per cohort.** Report out-of-bag c-index, in-sample
c-index and wall time per round before committing to the full run. Three
outcomes:

- OOB c-index within ~0.02 of 0.62 / 0.61 → proceed to the full run
- OOB clearly better → proceed, and this becomes a performance result too
- OOB clearly worse, or training unstable → stop and report; a cleanly
  interpretable model that costs accuracy is still a reportable trade-off

**Full run: ~100 rounds per cohort**, matching 102 (BLCA) and 100 (BRCA).

---

## 3. What to save

Per round, per out-of-bag patient:

- `main_effects[R]` — each region's scalar contribution
- `interactions[R][R]` — each pair's scalar contribution (symmetric, zero diagonal)
- `intercept`, `risk` (their sum), `patient_id`

Per round: out-of-bag and in-sample c-index, epochs, seed.

Aggregate to `results/ga2m_{cohort}.json`:

- mean absolute main effect per region, across out-of-bag patients and rounds,
  with bootstrap CI over rounds — **this is the new attribution, replacing the
  fusion weight**
- mean absolute interaction per pair, same treatment — the validated
  interaction map replacing the raw attention matrix
- fraction of pairs driven to zero by the L1 penalty
- **exact ablation**: c-index recomputed with all terms containing region r
  dropped. No zeroing, no masking, no attention-mixing ambiguity. Compare the
  resulting ordering against the current ablation ordering.

---

## 4. The comparisons that decide whether this was worth it

1. **Accuracy.** Paired out-of-bag c-index, GA²M against the current model, same
   rounds and seeds. Wilcoxon signed-rank over rounds.
2. **Overfitting.** In-sample minus out-of-bag gap. Current model: 0.99 vs 0.62
   (BLCA) and 0.99 vs 0.61 (BRCA). If GA²M narrows that, it is the headline.
3. **Attribution validity.** Spearman between the GA²M main effects and its own
   exact ablation deltas. The current architecture gives ρ = −0.69 (BLCA) and
   +0.03 (BRCA), i.e. no positive relationship. If GA²M gives a strong positive
   ρ, the paper has both a negative result and its fix.
4. **Agreement with existing evidence.** Does adipose still lead? Three methods
   already say it should.
5. **Interaction structure.** Which pairs survive the L1 penalty, and does
   necrosis appear in interactions rather than main effects? That would explain
   the current result: high attention, no independent contribution.

---

## 5. Risks

- **Identifiability is the main one.** Get the centering wrong and the
  attributions are meaningless in a way that is not obvious from the loss.
  Validate on synthetic data first: generate `y = g1(h_1) + g12(h_1,h_2) + noise`
  with known components and confirm recovery.
- Low-rank bilinear may underfit relative to attention. `k` is the knob.
- 28 or 36 interaction terms on 379 patients will overfit without the L1
  penalty. Do not skip it.
- This is a new model, not a re-analysis. If it underperforms, that is a real
  result for the Discussion, not a failure.

---

## 6. Relationship to the paper

If the pilot goes well and the full run completes, this most likely becomes a
second paper rather than part of this revision — a new architecture is a
different contribution from an audit framework. For *this* submission it belongs
in Future Work, with the pilot cited if available.

But if attribution validity (comparison 3) comes out strongly positive, it is
worth reconsidering. "Per-region weights are not attributions, and here is an
architecture where they are" is a substantially stronger paper than the first
half alone.
