# Co-author's cluster analyses: results (2026-11-12)

Answers to `docs/HANDOFF_SHIRIN_CLUSTER_TASKS.md`, computed on Trillium. Tasks 1 and 2
are complete. Task 3 has the magnification counts; the partial R-squared part is left
for Narval, where the out-of-bag risk file is. Task 4 is a check and an estimate; nothing was run.
No existing result was overwritten.

| Task | Status | Outputs |
|---|---|---|
| 1. Patient-level bootstrap of the Shapley correlations | done | `results/patient_bootstrap_shapley_{blca,brca}.json`, `analysis/patient_bootstrap_shapley.py` |
| 2. Comparable pairs per patient | whole cohort done; MED3pa evaluation split not rebuilt | `results/comparable_pairs_{blca,brca}.json`, `analysis/comparable_pairs.py` |
| 3. Scan magnification | counts done; partial R-squared not run | `results/magnification_{blca,brca}.json`, `analysis/slide_magnification.py` |
| 4. ABMIL per-patient risks | none exist; rerun estimated, not started | — |

## 1. Patient-level bootstrap of the Shapley correlations

Input: the main Shapley run on Trillium scratch (`/scratch/sorkwos/shapley_{blca,brca}`,
20 rounds each; its round JSONs are identical to those in `results/shapley_{blca,brca}/`).
B = 1,000 replicates per cohort, patients resampled with replacement and entered in
each round with their multiplicity as weights. Weighted concordance, exact Shapley
values and local shares were recomputed per round, then averaged over rounds as the
aggregate does. Fusion weights are fixed.

Check: with every patient weighted once, the script reproduces every saved coalition
c-index exactly (largest difference 0), the local shares to 6e-10, and the paper's
correlations.

| Correlation across compartments | Paper / point | Bootstrap median | 95% interval |
|---|---|---|---|
| **BLCA** (8 compartments, 379 patients) | | | |
| Fusion weight vs outcome Shapley | +0.81 | +0.64 | +0.17 to +0.90 |
| Model use (local share) vs outcome Shapley | +0.88 | +0.69 | +0.12 to +0.95 |
| Fusion weight vs model use | +0.95 | +0.95 | +0.90 to +0.98 |
| **BRCA** (9 compartments, 1,060 patients) | | | |
| Fusion weight vs outcome Shapley | −0.60 | −0.40 | −0.78 to +0.20 |
| Model use (local share) vs outcome Shapley | −0.27 | −0.13 | −0.67 to +0.48 |
| Fusion weight vs model use | +0.73 | +0.73 | +0.60 to +0.82 |

Headline contrast, BLCA minus BRCA for model use vs outcome Shapley (replicate b of
BLCA paired with replicate b of BRCA): point estimate +1.15, bootstrap median +0.79,
95% interval +0.02 to +1.45. 2.2% of replicates are at or below 0.

How to read this:
- **BLCA:** both correlations with outcome Shapley stay positive, but the intervals are
  wide (lower ends +0.12 and +0.17).
- **BRCA:** both intervals include 0.
- **The BLCA–BRCA difference** barely excludes 0.
- **Fusion weight vs model use** is stable in both cohorts. Neither quantity depends on
  outcomes, so resampling patients barely moves it.
- **The medians are closer to 0 than the point estimates** for the two Shapley
  correlations. Resampling adds noise to each compartment's mean Shapley value, which
  tends to weaken a rank correlation over 8 or 9 compartments. The percentile
  intervals are reported as computed, with no bias correction.

## 2. Comparable pairs per patient, by censoring status

Pair counts depend only on PFI time and event, not on the risk scores. So they were
computed from the CDR table for the same patients, with the same definition as
`analysis/patient_error.py`. Check: the counts reproduce
`results/patient_error_{cohort}_oob.json` exactly (number of patients and events,
unscored patients 6 / 37, mean 199.6 / 147.8 and minimum 2 / 1 among scored patients).

| | n | Median | IQR | Min | Max | With 0 pairs | With < 10 pairs |
|---|---|---|---|---|---|---|---|
| **BLCA** event | 163 | 332 | 268–343 | 170 | 372 | 0 | 0 |
| BLCA censored | 216 | 121 | 98–154 | 0 | 163 | 6 | 22 |
| BLCA all | 379 | 161 | 115–329 | 0 | 372 | 6 | 22 |
| **BRCA** event | 139 | 593 | 405–883 | 144 | 1022 | 0 | 0 |
| BRCA censored | 921 | 67 | 40–107 | 0 | 139 | 37 | 75 |
| BRCA all | 1060 | 81 | 44–124 | 0 | 1022 | 37 | 75 |

In both cohorts, every patient with fewer than 10 pairs is censored. In BRCA that's
8% of censored patients (75 of 921).

**The MED3pa evaluation split was not rebuilt.** `discovery_eval_split` stratifies on
the high-error label, which comes from `patient_error_{cohort}_oob.npz`. That file is
on Narval (`/home/sorkwos/data/`), not on Trillium. `analysis/comparable_pairs.py` can
be extended there: run the split with seed 20260908, check for 151 and 424 evaluation
patients, and summarise the same counts.

## 3. Scan magnification by cohort and site

Source: the slide files on Trillium scratch (`/scratch/sorkwos/TCGA-*`), read with
openslide. All 379 BLCA and 1,060 BRCA slides were found. Magnification is the
`openslide.objective-power` tag, else `aperio.AppMag`. The pixel size comes from
`openslide.mpp-x`.

**The magnification tag and the pixel size disagree for some slides.** Tile area
follows the pixel size, so that is the measure to use:

| | Tag 40x, ~0.25 µm/px | Tag 20x, ~0.25 µm/px | Tag 20x, ~0.5 µm/px | Not recorded |
|---|---|---|---|---|
| BLCA (379) | 372 | 5 | 2 | 0 |
| BRCA (1,060) | 919 | 83 | 51 | 7 |

By pixel size:
- **BLCA:** 377 slides at ~0.25 µm/px (40x-equivalent) and 2 at ~0.5 µm/px (20x).
- **BRCA:** 1,002 at ~0.25 µm/px, 51 at ~0.5 µm/px, and 7 with no value.

By the tag alone, BLCA is 372 at 40x and 7 at 20x, and BRCA is 919 at 40x, 134 at 20x
and 7 not recorded.

**Magnification is almost entirely nested within site.**
- **BLCA:** 0 of 32 sites mix magnifications. The 20x-tagged slides are all of site C4
  (5 slides, tagged 20x but 0.25 µm/px) and all of site HQ (2 slides, 0.5 µm/px).
- **BRCA:** 1 of 38 sites mixes: AC has 47 slides at 40x and 1 at 20x. The 20x slides
  are all of A8 (83 slides, tagged 20x but 0.25 µm/px), all of AO (47), HN (2) and
  MS (1), plus that one AC slide. The 7 slides with no metadata are all from site OL,
  whose other 14 slides are 40x.

**The partial R-squared extension was not run; it is left for Narval.** The site
analysis (`analysis/review_checks.py`, `site_casemix`) needs the out-of-bag risk
scores. They are in `patient_error_{cohort}_oob.npz` (field `risk`, with
`patient_ids`, `pfi_time` and `pfi_event`) or the `_per_patient.csv` beside it. The
MED3pa runs read these from `/home/sorkwos/data/` on Narval. Neither file is on
Trillium: `/home/sorkwos/data/` doesn't exist, and home, scratch and project hold only
the summary JSONs. So the paper's 15% / 14% could not be reproduced here. Because magnification is nested within site, expect
three things:
- magnification's partial R-squared, adjusted for case mix, can't exceed site's;
- adding magnification will barely change site's partial R-squared once site is in the
  model, since the two are almost the same variable;
- what's really testable is whether magnification accounts for site's share, i.e. how
  much site adds once magnification is in the model.

Use pixel size, not the tag, for this. On Narval, reproduce the site result first,
then add magnification. Narval needs the per-slide magnification, which the counts JSON
doesn't hold. Either run `analysis/slide_magnification.py` there, if the slides are on
Narval, or extend it here to write a per-slide table and copy that across. Don't commit
the table.

## 4. Surrogate (E) and MED3pa (R) checks on ABMIL: what exists

**Nothing usable exists.** Every ABMIL run on Trillium scratch saved only round-level
JSONs:
- `abmil_{blca,brca}`, `abmil_v2_{blca,brca}`: 10 rounds each;
- `planted_{cohort}_{planting}/abmil`: 2 to 8 rounds;
- `planted_attr_*` and `planted_v3_*`: empty `abmil/` folders here, since run N's
  ABMIL rounds ran on Narval.

The round JSONs hold no per-patient arrays, and there are no npz, csv or checkpoint
files. `analysis/abmil_compartment_shapley.py` saves no per-patient risks, unlike
`region_shapley.py`'s `round_###_local.npz`.

**Estimate for a rerun (not started):**
- **Code:** a small change. After `coalition_risk` builds the table, save `patient_ids`,
  times, events and the full-model risk per round, as `region_shapley.py` does. About
  10 lines.
- **Compute:** 10 GPU rounds per cohort, 20 jobs in total. On Trillium's timings from
  run M, that's about 40 min per BLCA round and 85 min per BRCA round, so ~6.5 + ~14 =
  **~21 GPU-hours**. Each job asks for 2 h (BLCA) or 4 h (BRCA).
- **Splits:** patient order is sorted and seeds come from the round number, so a rerun
  reproduces run M's splits. Its out-of-bag risks would pair with M's round results.
- **Coverage:** with ~42% of patients out-of-bag per round, about 0.4% of patients are
  never out-of-bag in 10 rounds (roughly 2 BLCA and 4 BRCA patients).
- **Queue:** the K–M runs in October used about 66 GPU-hours and dropped def-senger's
  Trillium fair-share from 0.48 to 0.10 for a while. 21 GPU-hours would cost less, but
  check `sshare -A def-senger -u sorkwos` first.

Juan decides whether to run it.
