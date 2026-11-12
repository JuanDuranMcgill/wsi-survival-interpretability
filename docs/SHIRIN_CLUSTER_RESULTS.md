# Co-author's cluster analyses: results (2026-11-12)

Answers to `docs/HANDOFF_SHIRIN_CLUSTER_TASKS.md`, computed on Trillium, plus the parts
that needed the out-of-bag risk file, run on Narval. Tasks 1 to 3 are complete. Task 4 is
a check and an estimate; nothing was run.
No existing result was overwritten.

| Task | Status | Outputs |
|---|---|---|
| 1. Patient-level bootstrap of the Shapley correlations | done | `results/patient_bootstrap_shapley_{blca,brca}.json`, `analysis/patient_bootstrap_shapley.py` |
| 2. Comparable pairs per patient | done (MED3pa evaluation split on Narval) | `results/comparable_pairs_{blca,brca}.json`, `results/comparable_pairs_{blca,brca}_med3pa_eval.json`, `analysis/comparable_pairs.py` |
| 3. Scan magnification | done (partial R-squared on Narval) | `results/site_magnification_r2.json`, `analysis/site_magnification_r2.py`, `results/magnification_{blca,brca}.json`, `analysis/slide_magnification.py` |
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

### On the MED3pa evaluation split (Narval)

Rebuilt on Narval from `/home/sorkwos/data/patient_error_{blca,brca}_oob.npz` with
`python analysis/comparable_pairs.py --med3pa-split`. Outputs:
`results/comparable_pairs_{blca,brca}_med3pa_eval.json`.

**There are two evaluation splits per cohort, not one.** The MED3pa scripts inner-join
the error file onto their feature CSV, and `discovery_eval_split` runs on that row order.
The clinical and radiomic CSVs order patients differently, so each arm gets a different
split. Both have 151 (BLCA) and 424 (BRCA) evaluation patients, but they share only
65 and 156 of them. Both are reported.

Check: each rebuilt split matches its saved `reliability_{cohort}_{arm}.json` in error
threshold, evaluation size and evaluation prevalence. Re-applying the saved evaluation
profiles reproduces every one exactly (patients, events, comparable pairs, mean error):
8/8 for BLCA clinical, 20/20 for BRCA clinical and 14/14 for BRCA radiomic. The BLCA
radiomic run saved no evaluation profiles, so that split is checked only by size,
threshold and prevalence.

Pairs behind each evaluation patient's error label. The error is computed over the
whole cohort, so these are whole-cohort counts restricted to the evaluation patients:

| | n | Median | IQR | Min | Max | With 0 pairs | With < 10 pairs |
|---|---|---|---|---|---|---|---|
| **BLCA clinical** event | 61 | 334 | 268–343 | 170 | 371 | 0 | 0 |
| BLCA clinical censored | 90 | 120 | 95–151 | 0 | 162 | 3 | 11 |
| BLCA clinical all | 151 | 157 | 108–326 | 0 | 371 | 3 | 11 |
| **BLCA radiomic** event | 62 | 332 | 259–343 | 200 | 371 | 0 | 0 |
| BLCA radiomic censored | 89 | 121 | 101–155 | 0 | 163 | 3 | 7 |
| BLCA radiomic all | 151 | 161 | 113–290 | 0 | 371 | 3 | 7 |
| **BRCA clinical** event | 58 | 618 | 455–872 | 147 | 994 | 0 | 0 |
| BRCA clinical censored | 366 | 77 | 42–114 | 0 | 139 | 16 | 26 |
| BRCA clinical all | 424 | 84 | 44–124 | 0 | 994 | 16 | 26 |
| **BRCA radiomic** event | 63 | 582 | 362–809 | 144 | 1022 | 0 | 0 |
| BRCA radiomic censored | 361 | 63 | 38–110 | 0 | 139 | 14 | 34 |
| BRCA radiomic all | 424 | 81 | 42–124 | 0 | 1022 | 14 | 34 |

The split looks like the whole cohort, as stratification should make it. Every patient
with fewer than 10 pairs is censored. Patients with 0 pairs have no error; MED3pa
labels them low-error.

The JSONs also hold counts of pairs among evaluation patients only, which is what each
profile's c-index uses. Those are smaller: BRCA clinical censored, median 34 (IQR
19–48), 23 with 0 and 38 with fewer than 10.

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

### On Narval: site result reproduced, then magnification added

**The site result reproduces exactly.** `site_casemix` from `analysis/review_checks.py`
was run with `risk` from `/home/sorkwos/data/patient_error_{cohort}_oob.npz` in place of
`oob_risk_{cohort}.npz`'s `mean_risk`. The clinical CSVs are the same-named files in
`~/data/`, since the script's `~/wsi-transfer/features` was a Mac path. Every value
matches `results/review_checks.json` to the last digit:

| | n | Site levels (≥ 10 patients, rest "other") | R² case mix | R² case mix + site | Partial R² of site |
|---|---|---|---|---|---|
| BLCA | 378 | 14 | 0.0856 | 0.2191 | **0.146** |
| BRCA | 1047 | 20 | 0.1046 | 0.2259 | **0.136** |

So the npz `risk` is the risk score the paper used.

**Magnification barely matters once site is in the model.** Narval has no slides, so
the pixel size comes from Trillium's per-slide table
(`results/slide_magnification_per_slide.csv`, one slide per patient). Magnification is
an indicator for ~0.5 µm/px (≥ 0.375). In BRCA, a second indicator marks the 7 slides
with no pixel size, all from site OL. Same OLS on the standardised risk score as
`site_casemix`. Script: `analysis/site_magnification_r2.py`. Output:
`results/site_magnification_r2.json`.

| Partial R² of the risk score | BLCA (n = 378) | BRCA (n = 1047) |
|---|---|---|
| Site, after case mix (paper; reproduced) | 0.146 (p = 1e-7) | 0.136 (p = 2e-22) |
| **Magnification, after case mix** | **0.000** (p = 0.97) | **0.011** (p = 0.003) |
| **Site, after case mix and magnification** | **0.147** (p = 1e-7) | **0.127** (p = 2e-20) |
| Magnification, after case mix and site | 0.001 (p = 0.55) | 0.002 (p = 0.39) |

How to read this:
- **BLCA:** only 2 slides are at ~0.5 µm/px, so magnification explains nothing, and
  site's share is unchanged.
- **BRCA:** magnification explains 1.1% of the risk score after case mix. That is
  significant, but less than a tenth of site's share. With magnification in the
  model, site still explains 12.7%, down from 13.6%. Once site is in, magnification
  adds 0.2% (p = 0.39).
- **Site's association with the risk score is not a magnification effect.** Mostly it
  is something else about the sites.
- **Sensitivity:** dropping BRCA's 7 slides with no pixel size (n = 1,040) gives the
  same picture. Magnification after case mix is 0.007; site after case mix and
  magnification is 0.128.
- The site variable pools sites with fewer than 10 patients into "other", as
  `site_casemix` does. So magnification isn't exactly nested in it (BLCA's 2 HQ slides
  fall in "other"). That's why "magnification after site" is not exactly 0.

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
