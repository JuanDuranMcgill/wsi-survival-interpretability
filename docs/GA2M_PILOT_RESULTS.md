# GA²M pilot results (10 rounds/cohort) and what they mean

Status: pilot complete for both cohorts. Full ~100-round run **not started** —
this is the decision point the design doc's stop rule calls for.

Everything below is real, computed data from actual pilot runs on Trillium.
No numbers in this document are estimated or invented.

---

## 0. Recap: what had to be fixed before this pilot could run

Two bugs surfaced and were fixed before these results are trustworthy. Full
detail is in the git history; short version because it matters for how much
to trust what follows:

1. **Centering bug (implementation, in the original ask).** The first
   implementation centered interactions by binning on the main-effect
   *output*, per a misreading of Section 1. That's wrong — conditioning on a
   many-to-one, moving-target quantity rather than the input. Symptom:
   attribution recovery on synthetic data came out *negatively* correlated
   with ground truth (r ≈ -0.4) despite the model fitting the target almost
   exactly. Fixed with the exact closed-form centering of the bilinear
   projections `u_r = A_r(h_r)`, `v_r = B_r(h_r)` (see `model/ga2m_head.py`
   module docstring for the derivation). Confirmed correct by an oracle test
   (hand-set exact ground-truth weights, no training: r = 0.997 / 0.9997).

2. **Synthetic-test-harness bug (mine, found by the other session).** After
   fixing #1, the synthetic validator still failed — held-out MSE ~2
   (worse than predicting the mean), looking exactly like a training-dynamics
   failure. Survived extensive rank/learning-rate/regularization sweeps
   because it wasn't a training problem at all: the test's ground-truth
   direction vectors were drawn from the same per-call RNG as the data, so
   the train and eval splits silently had *different true functions*. Fixed
   by drawing truth vectors from fixed, data-seed-independent generators
   (`model/validate_ga2m_synthetic.py`).

**Current synthetic validation state (both required to pass, both do):**

| Scenario | Held-out MSE | r (main effect) | r (interaction) |
|---|---|---|---|
| Independent regions | 0.0027 | +0.981 | +0.999 |
| Correlated regions (ρ=0.6), plain centering | 0.0010 | +0.966 | +0.999 |

(1.0 = predicting the mean; noise floor ≈ 0.01.) The residualized-centering
fallback for correlated regions exists in `ga2m_head.py` but isn't needed —
plain centering already passes at ρ=0.6.

---

## 1. What ran

Per-region encoder (proj/pos_mlp/beta/agt_layers/attn_pool/forward_bag) is
byte-identical to the existing backbone (imported as a library, not
reimplemented), same bootstrap split logic and `seed_offset` scheme, same
Cox loss, same PFI endpoint, same `--epochs 30`. Only the aggregation head
differs: GA²M (main-effect MLP + rank-16 bilinear interaction per pair,
λ_int=1e-3) in place of
`cross_region_attn → cross_region_norm → region_weights weighted sum → head`.

- **BLCA**: 10 rounds, seed_offset 0 (global rounds 1-10), one sequential
  SLURM job (`slurm/trillium/ga2m_pilot_blca.sh`), `--save_root
  /scratch/sorkwos/ga2m_pilot/blca`.
- **BRCA**: 10 rounds, seed_offset 0-9 (global rounds 1-10), **parallelized
  into 10 single-round SLURM jobs** (`slurm/trillium/ga2m_pilot_brca_round.sh`,
  one per `seed_offset` via `sbatch --export=SEED_OFFSET=k`), each writing to
  its own `--save_root /scratch/sorkwos/ga2m_pilot/brca/job_<k>` to avoid
  every job producing an identically-named `round_1.npz` in a shared
  directory. This was purely a wall-clock decision (time pressure on the
  session, not a methodological choice) — BLCA was left running sequentially
  since it was already most of the way through and moving fast; BRCA hadn't
  produced a single round yet after 30+ minutes and is the heavier cohort
  (~3x the patients), so it got split.

Both completed cleanly; no crashes, no timeouts, no anomalies in the SLURM
accounting.

---

## 2. Per-round results

### BLCA (target OOB c-index: 0.62, from the paper's Table 1)

| seed | OOB c-index | in-sample c-index | n_oob | wall time |
|---|---|---|---|---|
| 1 | 0.6426 | 0.9889 | 161 | 29.7 min |
| 2 | 0.6030 | 0.9892 | 165 | 38.8 min |
| 3 | 0.5433 | 0.9853 | 173 | 37.3 min |
| 4 | 0.6365 | 0.9895 | 173 | 39.5 min |
| 5 | 0.5959 | 0.9899 | 164 | 38.6 min |
| 6 | 0.6101 | 0.9883 | 172 | 39.2 min |
| 7 | 0.5927 | 0.9902 | 162 | 38.9 min |
| 8 | 0.5950 | 0.9876 | 168 | 39.3 min |
| 9 | 0.5690 | 0.9899 | 176 | 40.1 min |
| 10 | 0.5589 | 0.9883 | 176 | 39.2 min |

**Mean OOB c-index: 0.5947 ± 0.0299.** Mean in-sample: 0.9887 ± 0.0014.
Overfitting gap: 0.394. Mean wall time/round: 38.1 min (round 1 ran faster,
29.7 min, likely a warm-cache/first-allocation effect — all subsequent
rounds cluster around 37-40 min).

### BRCA (target OOB c-index: 0.61)

| seed | OOB c-index | in-sample c-index | n_oob | wall time |
|---|---|---|---|---|
| 1 | 0.5668 | 0.9924 | 474 | 59.9 min |
| 2 | 0.6097 | 0.9926 | 461 | 73.5 min |
| 3 | 0.6003 | 0.9915 | 483 | 73.2 min |
| 4 | 0.5457 | 0.9896 | 470 | 72.3 min |
| 5 | 0.5833 | 0.9915 | 479 | 55.7 min |
| 6 | 0.6041 | 0.9913 | 465 | 56.5 min |
| 7 | 0.5108 | 0.9937 | 488 | 65.9 min |
| 8 | 0.5829 | 0.9920 | 471 | 53.7 min |
| 9 | 0.6618 | 0.9910 | 462 | 61.8 min |
| 10 | 0.5971 | 0.9890 | 468 | 70.3 min |

**Mean OOB c-index: 0.5862 ± 0.0383.** Mean in-sample: 0.9914 ± 0.0013.
Overfitting gap: 0.405. Mean wall time/round: 64.3 min.

Raw per-round `.npz` files (patient-level, gitignored, never committed):
`/scratch/sorkwos/ga2m_pilot/blca/round_{1..10}.npz`,
`/scratch/sorkwos/ga2m_pilot/brca/job_{0..9}/round_1.npz`.
Aggregate JSON (no patient-level data): `results/ga2m_pilot_stop_rule.json`.

---

## 3. Stop-rule evaluation (Section 2's three-way decision)

> - OOB c-index within ~0.02 of 0.62/0.61 → proceed to the full run
> - OOB clearly better → proceed, and this becomes a performance result too
> - OOB clearly worse, or training unstable → stop and report

**Neither cohort cleanly hits any of the three.** Both miss "within 0.02" by
almost exactly the same margin:

- BLCA: 0.62 − 0.5947 = **0.0253** (target minus mean)
- BRCA: 0.61 − 0.5862 = **0.0238**

Both misses are smaller than that cohort's own round-to-round standard
deviation (0.0299 BLCA, 0.0383 BRCA) — i.e. within about 1σ of target, not a
large or clearly-worse gap. No round in either cohort looks unstable (BLCA
range 0.543-0.643, BRCA range 0.511-0.662 — ordinary bootstrap spread, not
divergence). **This is a genuine judgment call, not resolvable from these 10
rounds alone** — the full ~100-round run would materially tighten the
confidence interval around the mean and likely settle it either way.

**Overfitting gap did not narrow.** GA²M: 0.394 (BLCA) / 0.405 (BRCA).
Original model (reported in the design doc): ~0.37 (0.99 vs 0.62) / ~0.38
(0.99 vs 0.61). Essentially unchanged, possibly marginally worse. If this
holds at full scale, Section 4.2's "if GA²M narrows that gap, it is the
headline" does not happen.

---

## 4. Attribution validity (Section 4.3)

Two versions of this comparison exist. **The self-consistency test (4a) is
the decisive one** — it asks whether GA²M's own attribution agrees with
GA²M's own behavior, using nothing from the original architecture. The
cross-model comparison (4b) against the *original* architecture's separate
ablation study is a different question (does GA²M's attribution agree with
a different model's ablation?) and should not drive the proceed/stop
decision, per direct instruction — it's kept below only as context for how
the two framings differ.

### 4a. Self-consistency — GA²M's own exact ablation vs. GA²M's own main effect (comparison 3, decisive)

For each pilot round and each region r, recomputed OOB c-index with every
term touching r dropped (`ablated_risk = risk − main_effects[:,r] −
interactions[:,r,:].sum(1)`, using only what's already saved in the pilot's
`.npz` files — no model, no GPU). `delta_r = baseline_cindex − ablated_cindex_r`,
averaged across the 10 rounds, Spearman'd against GA²M's own mean
`|main effect|` per region. Sanity check first: reconstructing risk from
`intercept + main_effects + interactions` and recomputing c-index from
scratch matched the saved values to ~1e-6 (float32 rounding) in all 20
rounds — the saved data is internally consistent.

| | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|
| BLCA | +0.381 | 0.352 | +0.657 | 0.077 |
| BRCA | −0.250 | 0.516 | −0.310 | 0.417 |

**Neither cohort clears significance.** This is materially weaker than the
cross-model framing below — on its own terms, GA²M's main-effect magnitude
does not reliably predict what GA²M's own ablation says matters, in either
cohort, at 10 rounds. BRCA's self-consistency is actually negative (weak,
non-significant). This is the honest answer to "does the architecture do
what it was built for": **not yet demonstrated, at this round count.**
Full data (per-region deltas, per-round detail): `results/ga2m_self_ablation.json`.

**Interaction sparsity (same data):** the L1 penalty (λ_int=1e-3) drove
**0 of 28 pairs (BLCA)** and **0 of 36 pairs (BRCA)** below 5% of that
cohort's max pair magnitude — every interaction pair stayed active, nothing
got pruned. Section 4.5's "which pairs survive the L1 penalty" has nothing
to report at this λ; the full run likely needs a materially stronger
λ_int if pair-level sparsity is meant to be informative.

**Necrosis check** (Section 4.5's specific speculation — "high attention, no
independent contribution"):

| | main-effect rank | interaction-involvement rank | interactions > main effect? |
|---|---|---|---|
| BLCA (Necrosis) | 2/8 | 2/8 | No — identical rank, high in both |
| BRCA (Necrosis or hemorrhage) | 2/9 | **1/9** | Mildly yes — top interaction region, still rank-2 main effect |

Necrosis is not a region with a small main effect hidden behind large
interactions in either cohort — it has a *large* main effect (rank 2 of 8-9)
in both. BRCA shows a mild version of the hypothesized split (necrosis is
the single most interaction-heavy region, edging out its own high main
effect); BLCA shows no split at all.

### 4b. Cross-model comparison (context only, not decisive)

**Reference:** `results/weight_vs_ablation.json` (already committed, from the
existing 20-round region-ablation study of the *original* architecture)
reports that architecture's fusion-weight-vs-ablation Spearman correlation:
**ρ = −0.69 (BLCA)**, **ρ = +0.03 (BRCA)** — the fusion weight tells you
nothing about, or actively misleads about, what ablation says matters. This
was the motivation for Section 1's redesign.

**GA²M main-effect magnitude vs. the *original model's* ablation deltas**
(20 rounds, different model than the 10-round GA²M pilot — not a paired
comparison):

### BLCA

| region | \|main effect\| | ablation importance (−Δc-index) |
|---|---|---|
| Invasive urothelial carcinoma (tumor) | 1.0264 | 0.0096 |
| Necrosis | 0.9186 | 0.0011 |
| Perivesical adipose tissue (extravesical fat) | 0.8781 | 0.0098 |
| Muscularis propria (detrusor muscle) | 0.7471 | 0.0008 |
| Inflammatory infiltrates (immune cells) | 0.7448 | 0.0044 |
| Normal urothelium (benign mucosa) | 0.6879 | 0.0023 |
| Blood vessels (vasculature) | 0.6758 | −0.0018 |
| Lamina propria (fibrovascular stroma) | 0.6312 | −0.0000 |

**Spearman ρ = +0.667 (p=0.071), Pearson r = +0.730 (p=0.040).**

### BRCA

| region | \|main effect\| | ablation importance (−Δc-index) |
|---|---|---|
| Ductal carcinoma in situ (DCIS) | 1.3980 | 0.0058 |
| Necrosis or hemorrhage | 1.3564 | −0.0035 |
| Normal breast glands and lobules (TDLU) | 1.0912 | −0.0003 |
| Tumor-infiltrating lymphocytes (immune infiltrates) | 1.0111 | 0.0014 |
| Adipose tissue (fat) | 0.9427 | 0.0094 |
| Muscle tissue (smooth or skeletal muscle) | 0.8674 | 0.0067 |
| Invasive breast carcinoma (tumor cells) | 0.8338 | 0.0003 |
| Fibrous desmoplastic stroma | 0.8157 | 0.0001 |
| Blood vessels (vasculature) | 0.8082 | 0.0006 |

**Spearman ρ = −0.067 (p=0.865), Pearson r = −0.119 (p=0.760).**

**Reading, with 4a's caveat in mind:** against the *original* model's
ablation, BLCA looks like a real flip (−0.69 → +0.67/+0.73) and BRCA looks
flat (+0.03 → −0.07). But 4a shows that against **GA²M's own** ablation, BLCA
drops to a non-significant +0.38 and BRCA goes to −0.25. The cross-model
version is the more flattering framing and the self-consistency version is
the more honest one — **go with 4a for the actual decision.** This section
is retained because it's what the fusion-weight comparison was originally
run against, and the gap between the two versions is itself informative
(some of the apparent BLCA improvement is specific to which ablation study
it's compared against, not a property of GA²M alone).

**Caveats on this cross-model version specifically:**
- The ablation reference used 20 rounds of the *original* architecture; GA²M
  used 10 rounds of itself — not a paired comparison, unlike 4a which is
  fully self-contained.
- n=8 (BLCA) / n=9 (BRCA) regions is a small sample for a correlation test —
  wide CIs, low power, in both 4a and 4b.
- Ten rounds is not the full ensemble the paper would report from.

Full data: `results/ga2m_attribution_vs_ablation.json` (4b) and
`results/ga2m_self_ablation.json` (4a, the decisive one).

---

## 5. Resource budget for the full run, if you proceed

At observed per-round wall time, ~100 rounds/cohort:

- **BLCA**: 38.1 min/round × 100 ≈ **63.5 GPU-hours**
- **BRCA**: 64.3 min/round × 100 ≈ **107 GPU-hours**

Both pilots together consumed ~17 GPU-hours in well under 2 hours of
wall-clock by running rounds in parallel (BLCA sequential + BRCA as 10
concurrent single-round jobs). The same approach (single-round jobs,
`seed_offset` 0..99, one `--save_root .../job_<k>` each) scales directly to
the full run; Trillium admitted all 10 BRCA jobs within about 15-20 minutes
of submission during this session, for reference on likely queue behavior,
though that is not a guarantee for a much larger batch.

---

## 6. Open questions for whoever decides on the full run

1. **Self-consistency (4a) — the decisive test — does not clear significance
   in either cohort at 10 rounds** (BLCA ρ=+0.38 p=0.35, BRCA ρ=−0.25
   p=0.52), and the stop rule doesn't cleanly resolve either. Neither cohort
   makes a clear case to proceed on the numbers alone. Is 100 rounds/cohort
   still the right next step given this, or does the small-n (8-9 regions)
   noise mean this needs a fundamentally larger signal to ever resolve, full
   run or not?
2. λ_int=1e-3 produced zero interaction sparsity (0/28, 0/36 pairs near
   zero) in the pilot. Worth increasing λ_int materially before the full run
   if Section 4.5's pruning/heredity analysis is meant to say anything —
   tuning this now is cheap; discovering it's still all-dense after 100
   rounds is not.
3. Section 4.5's "necrosis: high attention, no independent contribution"
   speculation isn't borne out — necrosis has a large main effect in both
   cohorts, not a small one. Worth revisiting that framing before the full
   run's writeup leans on it.
4. Worth deciding now: full run at rank=16 as speced, or is it worth a quick
   rank sensitivity check first given the pilot's small-n region correlation
   is noisy either way?

---

## 7. File index

| What | Path |
|---|---|
| GA²M head (centering fix) | `model/ga2m_head.py` |
| Training driver | `model/train_ga2m.py` |
| Synthetic validator (fixed) | `model/validate_ga2m_synthetic.py` |
| Diagnostic scripts from the debugging chain | `model/debug_ga2m_*.py` |
| Reference synthetic test (other session) | `analysis/ga2m_reference_synthetic.py` |
| Pilot aggregation script | `model/aggregate_pilot_results.py` |
| Self-ablation script (decisive, comparison 3) | `model/ga2m_self_ablation.py` |
| Attribution-vs-ablation script (cross-model, context only) | `model/ga2m_attribution_vs_ablation.py` |
| SLURM: pilots | `slurm/trillium/ga2m_pilot_{blca,brca}.sh`, `ga2m_pilot_brca_round.sh` |
| SLURM: validation/debug jobs | `slurm/trillium/ga2m_*_check.sh` |
| Pilot stop-rule summary (JSON, no patient data) | `results/ga2m_pilot_stop_rule.json` |
| Self-ablation results (JSON, decisive) | `results/ga2m_self_ablation.json` |
| Attribution-vs-ablation (JSON, cross-model, context only) | `results/ga2m_attribution_vs_ablation.json` |
| Raw per-round pilot output (patient-level, NOT in git) | `/scratch/sorkwos/ga2m_pilot/` |
