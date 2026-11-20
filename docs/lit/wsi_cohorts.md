# WSI survival: event counts, cohorts, pan-cancer models (lit note, 2026-11-20)

Purpose: decide where to re-run the tissue-type explanation audit (now BLCA 163 PFI events, BRCA 139) with more events or a stronger model.

Legend: **[V]** = checked against the primary source (full text, publisher/Crossref/Europe PMC record, or official repo/data portal). **[S]** = from a secondary page or a search snippet; the primary text could not be opened. **[NV]** = could not verify; do not cite without checking.
Access problems: PMC served a CAPTCHA for several articles (not bypassed); nature.com, cell.com, sciencedirect and thecvf.com refused automated fetches. Where that blocked a number, it is marked.

---

## 1. TCGA event counts (TCGA-CDR)

**Source [V]:** Liu J, Lichtenberg T, Hoadley KA, et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. *Cell* 2018;173(2):400–416.e11. doi:10.1016/j.cell.2018.02.052 (PMC6066282). Table 3 gives events/censored per endpoint and per cancer type, plus endpoint recommendations.

- 11,160 patients, 33 cancer types; overall median follow-up 22.1 months (GBM and LAML about 12 months, KICH about 48) [V].
- Sum of Table 3 (my calculation): **OS 3,622 events** (33 types); **PFI 3,996 events** (32 types; LAML has no PFI) [V numbers, own sum]. The per-type N column sums to 11,160, which matches the paper.
- Cross-check: GDC "vital_status = Dead" counts (API, queried 2026-11-20) match Table 3 OS events within 1 for all projects checked (BLCA 182 vs 181, BRCA 152 vs 151, GBM 492 vs 491, OV 349 vs 349).
- Recommendations (Table 3): PFI is not recommended for PCPG and is unavailable for LAML. OS is not recommended for DLBC, PCPG, TGCT or THYM. BRCA OS is marked "use with caution" (longer follow-up needed); BRCA PFI is recommended.
- Diagnostic FFPE slides: **9,752 TCGA cases have ≥1 "Diagnostic Slide"** in GDC (API facet, queried 2026-11-20) [V]. GDC case lists are not identical to the CDR (e.g. GDC TCGA-TGCT now has 263 cases, CDR has 134), so join by barcode.

### Per-type table, sorted by PFI events

"PFI with slide (rough)" = PFI events × min(1, dx-slide cases / CDR N). It assumes events are missing at random with respect to slide availability. It reproduces this project's real counts closely (BLCA 166 vs actual 163; BRCA 140 vs actual 139), but it is an **estimate**. Compute the exact values on the cluster by joining `TCGA-CDR-SupplementalTableS1.xlsx` with the GDC slide manifest.

| Type | N (CDR) | OS events | PFI events | GDC cases with dx slide | PFI with slide (rough) |
|---|---|---|---|---|---|
| GBM | 596 | 491 | 506 | 389 | ~330 |
| OV | 587 | 349 | 414 | 106 | ~75 |
| SKCM | 470 | 216 | 309 | 433 | ~285 |
| LUAD | 522 | 188 | 213 | 478 | ~195 |
| HNSC | 528 | 223 | 198 | 450 | ~169 |
| LGG | 515 | 125 | 192 | 491 | ~183 |
| LIHC | 377 | 132 | 185 | 365 | ~179 |
| BLCA | 412 | 181 | 177 | 386 | 163 (actual) |
| KIRC | 537 | 177 | 162 | 513 | ~155 |
| LUSC | 504 | 219 | 149 | 478 | ~141 |
| BRCA | 1097 | 151 | 145 | 1062 | 139 (actual) |
| STAD | 443 | 172 | 143 | 416 | ~134 |
| SARC | 261 | 99 | 139 | 254 | ~135 |
| UCEC | 548 | 91 | 124 | 505 | ~114 |
| COAD | 459 | 102 | 123 | 451 | ~121 |
| PAAD | 185 | 100 | 110 | 183 | ~109 |
| PRAD | 500 | 10 | 93 | 403 | ~75 |
| ESCA | 185 | 77 | 87 | 156 | ~73 |
| CESC | 307 | 71 | 71 | 269 | ~62 |
| MESO | 87 | 74 | 61 | 75 | ~53 |
| KIRP | 291 | 44 | 58 | 275 | ~55 |
| THCA | 507 | 16 | 52 | 506 | ~52 |
| ACC | 92 | 34 | 49 | 56 | ~30 |
| READ | 170 | 26 | 39 | 165 | ~38 |
| UCS | 57 | 35 | 37 | 57 | ~37 |
| TGCT | 134 | 4 | 35 | 261* | ≤35 |
| UVM | 80 | 23 | 30 | 80 | ~30 |
| CHOL | 45 | 22 | 23 | 39 | ~20 |
| THYM | 124 | 9 | 22 | 121 | ~21 |
| PCPG | 179 | 6 | 21 | 176 | ~21 |
| KICH | 113 | 13 | 17 | 109 | ~16 |
| DLBC | 48 | 9 | 12 | 44 | ~11 |
| LAML | 200 | 133 | NA | – | – |

\*GDC TGCT count exceeds the CDR N; see note above.

**Takeaways**
- Only GBM (~330) and SKCM (~285) clearly exceed BLCA's PFI events among patients with slides. LUAD (~195), LGG (~183), LIHC (~179) and HNSC (~169) are similar to BLCA. OV has 414 PFI events, but only 106 TCGA-OV cases have FFPE diagnostic slides.
- No single TCGA cancer type gets far beyond 200–330 PFI events. "Large" in TCGA therefore means pooling. Nine types have ≥150 PFI events in the CDR: BLCA, GBM, HNSC, KIRC, LGG, LIHC, LUAD, OV, SKCM, totalling 2,356 (own sum, before the slide filter). Sixteen types have ≥100 PFI events, totalling 3,289.
- Combined cohorts inflate within-"type" signal too. For example, GBMLGG merges two grades with very different survival (GBM: 506 PFI events of 596; LGG: 192 of 515).

---

## 2. Pan-cancer and multi-cancer WSI survival models

| Model / paper | Data, endpoint | Pooled or per type? | Evaluation | Reported C-index | Code / weights |
|---|---|---|---|---|---|
| **Wulczyn et al. 2020** PLoS ONE 15(6):e0233678, doi:10.1371/journal.pone.0233678 [V] | 10 TCGA types (BLCA, BRCA, COAD, HNSC, KIRC, LIHC, LUAD, LUSC, OV, STAD); DSS; 4,880 cases (train+tune) and 1,216 test cases; test DSS events 284 in total, 13–76 per type | Per type (hyperparameters tuned per study, 500 models) | Test split 2:1:1. The "combined" C-index sums concordant and informative pairs **only within the same study**, i.e. a stratified C-index | DLS alone, combined: **61.1** (57.2–65.1). Per type: BLCA 54.0, BRCA 72.0, COAD 70.9, HNSC 58.2, KIRC 71.1, LIHC 71.3, LUAD 46.3, LUSC 62.1, OV 53.9, STAD 68.7. Baseline (stage, age, sex) 66.9; baseline+DLS 70.6 (Δ 3.7, 1.0–6.5). Significant in 5 of 10 types in multivariable Cox | Pseudocode only; no code or weights |
| **PORPOISE**, Chen RJ et al. 2022 *Cancer Cell* 40:865–878.e6, doi:10.1016/j.ccell.2022.07.004 [V citation] | 14 TCGA types, 5,720 patients; survival in 4 discrete bins (repo: `survival_months`) | **Per type** | 5-fold CV | Overall c-Index MMF 0.645, AMIL (WSI only) 0.585, SNN 0.607 (PMC version) [S]. The ScienceDirect version reads 0.644 / 0.578 / 0.606 and the arXiv v1 0.630 / 0.585 / 0.597 [S]. KIRP MMF 0.816 [S]. Full per-type table not opened (PMC CAPTCHA) | github.com/mahmoodlab/PORPOISE: code + 5-fold splits; no checkpoints [V] |
| Howard, Kather & Pearson, *Cancer Cell* 2023;41(1):5–6, doi:10.1016/j.ccell.2022.10.025 [V citation] | Re-analysis of PORPOISE on 8 types | – | **Site-preserved CV** | Repo table: average MMF 0.606 → 0.576, MIL 0.567 → 0.557, SNN 0.575 → 0.554 when slides from the same site are not split between train and test. COAD MMF 0.575 → 0.482 [V repo: github.com/fmhoward/PORPOISE_SITE] | Code |
| **MultiSurv**, Vale-Silva & Rohr 2021 *Sci Rep* 11:13505, doi:10.1038/s41598-021-92799-4 [V] | 33 TCGA types, 11,081 patients after preprocessing; clinical, omics, WSI | **One pooled model** (split 80/10/10, stratified by type) | Test set, time-dependent C (Ctd), computed on the whole test set | WSI only Ctd **0.569** (0.543–0.597). Clinical only 0.809. Best (clinical+mRNA) **0.822**. Per type (≥20 test patients): THCA 0.988, KIRP 0.959, COAD 0.953 … SARC 0.589, LUSC 0.554 | github.com/luisvalesilva/multisurv |
| **Cheerla & Gevaert 2019** *Bioinformatics* 35(14):i446–i454, doi:10.1093/bioinformatics/btz342 [V] | 20 TCGA types; clinical, mRNA, miRNA, WSI | One pooled model; single-type models trained for comparison | 85/15 split stratified by type | Pooled C-index 0.78 (0.784 in conclusion). Per type ranged 0.66 (LUSC) to 0.93 (KICH) in the pooled model, vs 0.51–0.78 for single-type models. WSI-only not reported | Code on GitHub (not checked) |
| **PROGPATH**, Yuan W et al. 2025 *Signal Transduct Target Ther* 10:285, doi:10.1038/s41392-025-02374-w [V] | 15 TCGA types: 6,670 patients, 7,999 WSIs. External: 17 cohorts, 4,441 patients (PLCO, CPTAC, hospitals). DSS (TCGA, PLCO), OS (CPTAC, others) | **One pooled model**. Cancer type is a one-hot input token that drives a router; clinical variables are fused | 5-fold CV stratified by type | "Mean" C-index TCGA **0.731**, PLCO 0.727, CPTAC 0.664, other clinical cohorts 0.677. Text unclear whether this is pooled or averaged over types [V] | github.com/Valeyards/ProgPath: weights released, Virchow2 features, CC BY-NC-SA 4.0 [V] |
| **Song S et al. 2025** "Multimodal Survival Modeling in the Age of Foundation Models", arXiv:2505.07683 [V, preprint] | 32 TCGA types, 7,982 patients, 2,192 deaths; OS; UNI2-h tile embeddings mean-pooled | One pooled model | 5-fold CV; headline C over the whole test fold | Histology only **0.754** pooled, vs per type BRCA 0.648, KIRC 0.685, UCEC 0.703, THCA 0.749, LGG 0.753, HNSC 0.636, LUSC 0.598, LUAD 0.617. **The pooled value is ≥ every per-type value.** Multimodal pooled 0.795 | – |
| **CHIEF**, Wang X et al. *Nature* 2024;634:970–978, doi:10.1038/s41586-024-07894-z [V citation] | Survival in 7 types (COADREAD, LUSC, BRCA, GBM, UCEC, LUAD, RCC), TCGA + 15 validation cohorts [S] | Per type | – | C-indices **not retrieved** (PMC CAPTCHA) [NV] | github.com/hms-dbmi/CHIEF: gated weights (CHIEF_CTransPath, CHIEF_pretraining), RCC survival example notebook; no fine-tuned survival weights listed [V] |
| **TITAN**, Ding T et al. *Nat Med* 2025, doi:10.1038/s41591-025-03982-3 [S citation] | DSS on 6 TCGA cohorts: BRCA 1,022, BLCA 360, CRC 545, KIRC 502, UCEC 504, NSCLC 844 patients | Per type; linear Cox on frozen slide embeddings | 5-fold **site-preserving** CV | TITAN beats next best (CHIEF) by +3.62% on average; per-cohort values not retrieved [V arXiv:2411.19666] | huggingface.co/MahmoodLab/TITAN (gated, CC-BY-NC-ND); needs CONCH v1.5 features; no survival code [V] |
| **Prov-GigaPath**, Xu H et al. *Nature* 2024;630:181–188, doi:10.1038/s41586-024-07441-w [V citation] | Benchmark: 9 subtyping + 17 pathomics tasks | – | – | No survival in the original benchmark, as far as found [S] | HF weights (gated) |
| **UNI** (Chen RJ et al. *Nat Med* 2024;30:850–862, doi:10.1038/s41591-024-02857-3) and **CONCH** (Lu MY et al. *Nat Med* 2024;30:863–874, doi:10.1038/s41591-024-02856-4) [V citations] | Encoders | – | – | Original papers: no survival task found [S]. Used as feature extractors in the survival benchmarks below | HF (gated) |
| **MUSK**, Xiang J et al. *Nature* 2025;638:769–778, doi:10.1038/s41586-024-08378-w [S citation] | Prognosis in 16 types [S] | – | – | C-index not retrieved [NV] | huggingface.co/xiangjx/musk, base model only, CC-BY-NC-ND [V] |
| **MCAT**, Chen RJ et al. ICCV 2021 [S] | TCGA BLCA, BRCA, GBMLGG, LUAD, UCEC (repo) | Per type, WSI + genomics | 5-fold CV, random | Original values not opened (thecvf 403). SurvPath's re-run: MCAT overall 0.594 | github.com/mahmoodlab/MCAT: "Model checkpoints for MCAT are uploaded in the results directory" [V] |
| **SurvPath**, Jaume G et al. CVPR 2024, arXiv:2304.06819 [V] | DSS; BRCA 869, BLCA 359, COADREAD 296, HNSC 392, STAD 317 | Per type | 5-fold, **stratified by site** | SurvPath overall 0.629. **WSI-only ABMIL 0.554** (BRCA 0.493, BLCA 0.518, COADREAD 0.630, HNSC 0.580, STAD 0.550); TransMIL 0.556; MCAT 0.594 | github.com/mahmoodlab/SurvPath: code + splits, CTransPath features, no checkpoints [V] |
| **GPFM**, Ma J et al. arXiv:2407.18449 [V, preprint] | 11 TCGA survival tasks + CPTAC-LUAD; ABMIL on frozen features | Per type | 5-fold, label-stratified (not site-preserving) | GPFM mean C-index 0.726 across 12 tasks (includes GBMLGG); per-encoder values not retrieved | – |
| **PathBench**, arXiv:2505.20202 [V, preprint] | 7 in-house survival tasks (breast, gastric, CRC) | Per task | – | Best per task 0.66–0.77 | Private data, leaderboard only |
| Gustafsson FK et al. 2026, arXiv:2604.24679 [V, preprint] | Swedish breast cohorts (see section 3) | Per cohort; train on one, test external | External | Best RFS **0.678** (H-optimus-1), PFS 0.702; CONCH v1.5 and UNI2 close | Data not public |

### Does pooling across cancer types inflate the C-index?
Yes, and the evidence is consistent:
1. **Case-mix heterogeneity.** A C-index rises with the spread of true risk in the sample (Vergouwe Y, Moons KGM, Steyerberg EW. *Am J Epidemiol* 2010;172(8):971–980, doi:10.1093/aje/kwq223 [V]). Cancer types differ hugely in baseline hazard: in the CDR, the share of patients with a PFI event ranges from 85% in GBM (506/596) to about 10% in THCA (52/507), with BRCA at 13% (145/1,097). Any score that "knows" the type ranks most across-type pairs correctly for free.
2. **Direct demonstration.** In Song et al. 2025 (UNI2 histology), the pooled C-index (0.754) is at or above every per-type value (0.598–0.753). In MultiSurv (best model, clinical+mRNA, pooled Ctd 0.822), the per-type examples given in the text span 0.554 (LUSC) to 0.988 (THCA). In Cheerla 2019 the pooled model reaches 0.78. PROGPATH even feeds cancer type to the model as an input token.
3. **Correct alternative.** Wulczyn 2020 computes the combined C-index only over within-type pairs (a stratified C-index). The result, 61.1, is close to typical single-type WSI values. Use this, or report per-type C-indices with CIs plus a stratified Cox model / stratified C.
4. **Site confounding inside TCGA.** TCGA slide sources (tissue source sites) carry learnable signatures that inflate survival accuracy (Howard FM et al. *Nat Commun* 2021;12:4423, doi:10.1038/s41467-021-24698-1 [V]). Site-preserved CV lowered PORPOISE's MMF average from 0.606 to 0.576 (Howard 2023 letter/repo). With site-stratified folds, WSI-only ABMIL in SurvPath is 0.554; the BRCA value of 0.493 is chance level. Label-stratified benchmarks (GPFM 0.726) are higher, but cohorts and endpoints differ, so the gap is not purely due to the split.
5. **Merged cohorts.** GBMLGG, NSCLC (LUAD+LUSC), RCC (KIRC+KIRP+KICH) and COADREAD are small-scale pools, and the same inflation applies to them.

---

## 3. Larger non-TCGA WSI cohorts with survival

| Cohort | Cancer | Size | Endpoint, events | Access | Source |
|---|---|---|---|---|---|
| **PLCO** pathology images | Breast 1,012 subjects / 2,909 images; colorectal 749 / 2,777; prostate 1,095 / 2,999; lung 492 / 1,521; ovarian 227 / 874; bladder 284 / 481; adenoma 768 / 1,103; male breast 18 / 56 (Aperio AT2, 40x) | Long trial follow-up with mortality. PROGPATH used PLCO for DSS (3,804 WSIs: BLCA 466, BRCA 1,893, LUAD 176, CRC 1,269). Events not on catalog [NV] | CDAS project application + data agreements; delivered by NCI CGR file share; no fee stated | cdas.cancer.gov/learn/plco/images/pathology [V] |
| **NLST** pathology | Lung | ~450–451 patients, 1,225 slides (standard package) | Outcomes via CDAS; events not verified [NV] | Slides free on TCIA / IDC; clinical data via CDAS | cdas.cancer.gov/learn/nlst/images [V]; TCIA [S] |
| **CPTAC** (CPTAC-2/3) | ~10 types (LUAD, LSCC, CCRCC, PDA, UCEC, GBM, HNSCC, CM, BRCA, COAD, OV…) | 2,208 cases in GDC | **700 deaths in total, all in CPTAC-3** (incl. 122 AML). By disease type: gliomas 155, adenomas/adenocarcinomas 146, ductal/lobular 130, squamous 74, melanoma 61. Every solid type has <160 deaths, and follow-up is short | WSIs public (TCIA/IDC); clinical data via GDC/PDC | GDC API, queried 2026-11-20 [V] |
| **DACHS** | CRC (Germany) | 2,271 patients in Jiang 2024 (4,428 total minus external cohorts; table columns 908/908/455). Median follow-up 6.3 years | OS, DFS, DSS; event counts in appendix, not read [NV] | Via study PIs (DKFZ, Brenner/Hoffmeister); "provided by the corresponding study Principal Investigators" | Jiang X et al. *Lancet Digit Health* 2024;6(1):e33–e43, doi:10.1016/S2589-7500(23)00208-X [V] |
| **MCO** | CRC (Australia, 1994–2010) | ~1,500 patients; 1,395 used by Jiang 2024; median follow-up 4.9 years (capped at 5) | OS and DSS; events [NV] | SREDH consortium application, data-use agreement, **access fees**, 6-monthly reporting | Jiang 2024 [V]; sredhconsortium.org page [S] |
| **TransSCOT** (SCOT trial) | CRC stage III/high-risk II | 3,182 in translational arm; 2,856 with two sections | Events not reported in source [NV] | "upon reasonable request" plus IRB/custodian approval | Henriksen AL, Skrede OJ et al. arXiv:2602.22347 [V, preprint] |
| **QUASAR 2, VICTOR, Gloucester, Aker** | CRC | DoMore-v1-CRC: 828 distinct-outcome training + 1,645 tuning patients; 920 UK validation; 1,122 Norwegian (capecitabine) validation. QUASAR 2 validation n=1,075 | Cancer-specific survival; death counts [NV] | Via trial groups (Oxford/Oslo); not public | Skrede OJ et al. *Lancet* 2020;395:350–360, doi:10.1016/S0140-6736(19)32998-8 [V]; Kleppe A et al. *Lancet Oncol* 2022;23(9):1221–1232, doi:10.1016/S1470-2045(22)00391-6 [S] |
| YCR-BCIP | CRC (Yorkshire) | Cited with QUASAR in Kather-lab funding; size [NV] | [NV] | Leeds (Quirke) collaboration | Jiang 2024 acknowledgments [V mention only] |
| **SurGen / SR386** | CRC (Scotland) | 427 cases (1 WSI each) | **161 deaths (38%)**, but only a binary 5-year label; censored follow-up times not recorded, so a C-index is limited | **Public**: EBI BioImage Archive doi:10.6019/S-BIAD1285 | Myles C et al. *GigaScience* 2025;14:giaf086 [S]; arXiv:2502.04946 [V] |
| **HANCOCK** | Head & neck (Erlangen) | 763 patients; 701 with primary-tumour H&E WSI; follow-up up to 14 years | OS (5-year OS 77.3%); recurrence; death count not stated (computable from the public data) | **Public**: hancock.research.fau.eu (TCIA mirror planned) | Dörrich M et al. *Nat Commun* 2025;16:7163, doi:10.1038/s41467-025-62386-6 [V] |
| **LEOPARD** | Prostate (Radboud) | 508 public training patients + 100 hidden test (1,723 slides total) | Biochemical recurrence (time + event); count not verified [NV] | **Public**: AWS `s3://leopard-challenge`, CC BY-NC-SA | registry.opendata.aws/leopard [V]; Grisi C et al. arXiv:2603.14187 [V] |
| PLCO prostate (LEOPARD external) / TCGA-PRAD | Prostate | PLCO 723 patients, 1,971 slides; TCGA-PRAD 298 patients with **62 BCR events** | BCR | CDAS / public | arXiv:2603.14187 [V] |
| **Swedish breast** (SöS-BC-4, KS-Solna, SCAN-B-Lund) | Breast | 2,315 / 1,857 / 1,262 patients | RFS events **351 / 389 / 226**; PFS 144 / 145 / 88; mean follow-up 7.7–9.1 years | **Not public** | arXiv:2604.24679 [V] |
| CPS-II (+ CPS-3, PLCO, TCGA) for HiPS | Breast | Sizes not retrieved [NV] | Survival | American Cancer Society request; PLCO via CDAS | Amgad M et al. *Nat Med* 2024;30:85–97, doi:10.1038/s41591-023-02643-7 [S] |
| CHIMERA (MICCAI 2025) | Bladder NMIBC, prostate | Small (prostate training 95 cases; bladder 368 patients in total) | Recurrence | Public training data | chimera.grand-challenge.org [S] |
| PANDA | Prostate | – | **No survival** (grading only) | Public | – |

Practical reading: only PLCO (breast, CRC, prostate), DACHS, MCO, TransSCOT/QUASAR and the Swedish breast cohorts plausibly reach several hundred events in **one** cancer type. All of them require an application, and the Swedish ones are not shareable. Among fully public sets, none clearly exceeds ~200 events: HANCOCK, SurGen (binary only) and LEOPARD fall short, and CPTAC is <160 per type.

---

## 4. Published models with public weights that could be re-run with one tissue type removed

The audit needs (a) held-out risk scores and (b) the model rerun without one tissue type. For a frozen-feature MIL model, (b) means dropping that type's tile features from the bag at inference, or retraining without them.

| Candidate | Why | Caveats |
|---|---|---|
| **KatherLab DACHS survival model** (Jiang 2024), github.com/KatherLab/crc-models-2022 `DACHS_model/survival_prediction.pth`, MIT; runs with marugoto (`survival` branch). RetCCL tile features → attMIL [V] | Trained on 2,271 DACHS patients, so it is stronger than any TCGA-only CRC model. **TCGA-CRC (565) was an external cohort**, so its scores are genuinely held out: OS C 0.64 (0.52–0.74), DSS 0.69 (0.54–0.81). MCO external: OS 0.65, DSS 0.70 [V] | TCGA-CRC events are few (COAD 123 + READ 39 PFI); the endpoint differs (OS/DSS); requires RetCCL features, not UNI-2 |
| **PROGPATH**, github.com/Valeyards/ProgPath (weights, CC BY-NC-SA 4.0). Virchow2 tile features (2,560-d) + clinical (age, sex, stage) + cancer-type one-hot → risk [V] | Pooled 15-type model; input is a bag of tile features, so tissue-type tiles can be dropped | Released weights were probably trained on all of TCGA, so TCGA scores would be in-sample: use external cohorts (PLCO, CPTAC, SR-CRC demo). Clinical and cancer-type inputs carry much of the signal; ablating tissue may barely move the risk |
| **CHIEF**, github.com/hms-dbmi/CHIEF (gated weights, AGPL/GPLv3 non-commercial) [V] | Strong slide encoder; RCC survival example included | No fine-tuned survival checkpoints listed; you would train a survival head |
| **MCAT** checkpoints, github.com/mahmoodlab/MCAT [V] | Released TCGA checkpoints | Needs genomics input; random (not site-stratified) 5-fold splits; ResNet-50 features |
| **PORPOISE / SurvPath / Patho-Bench / STAMP** (code + splits; no survival checkpoints) [V] | Retraining recipes with established splits. Patho-Bench offers 95 tasks with canonical train/test splits and coxnet; STAMP lists Cox survival and UNI-v2/Virchow-v2 support | You retrain, i.e. the same as the current pipeline |
| Encoders: UNI2, CONCH v1.5/TITAN, Virchow2, MUSK, Prov-GigaPath (all gated, non-commercial) | A stronger encoder can raise the C-index | Not survival models |

---

## 5. Main pitfalls of pan-cancer pooling (for the paper)
1. A pooled C-index mixes between-type pairs, which are easy (type predicts hazard), with within-type pairs, which are the real task. Report a stratified C-index (within-type pairs only, as in Wulczyn 2020) and per-type values with CIs.
2. Explanations inherit the same confound. If tissue composition differs by cancer type (adipose in BRCA, muscle in BLCA), a pooled model's tissue-type importance can encode the cancer-type label. Ablate within type, or stratify the Cox baseline hazard by type.
3. Endpoint heterogeneity: PFI quality varies by type (not recommended for PCPG, absent for LAML); follow-up is short (median 22.1 months) and differs by type.
4. Site/batch signatures (TCGA tissue source sites) inflate results; use site-preserved folds.
5. Event imbalance: GBM (~330 PFI events with slides) and SKCM (~285) contribute the most events, so a pooled audit is driven by a few aggressive types. OV has many events but few FFPE slides.
6. Cancer type as an input (PROGPATH) or proxied by morphology (any WSI model) makes "importance" of tissues partly a type detector.

---

## Gaps (not verified)
- Exact PFI events among TCGA patients with FFPE slides for types other than BLCA/BRCA (estimates only; compute on the cluster).
- CHIEF, TITAN, MUSK per-cohort survival C-indices; PORPOISE full per-type table; MCAT original table.
- Event counts for PLCO, NLST, DACHS, MCO, TransSCOT, QUASAR 2, YCR-BCIP, LEOPARD, HANCOCK, CPS-II.

## Reference URLs
- Liu 2018 CDR: https://doi.org/10.1016/j.cell.2018.02.052 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC6066282/
- GDC API queries (2026-11-20): https://api.gdc.cancer.gov/cases (facets project.project_id, filter files.experimental_strategy = "Diagnostic Slide"; CPTAC vital_status)
- Wulczyn 2020: https://doi.org/10.1371/journal.pone.0233678
- Chen 2022 PORPOISE: https://doi.org/10.1016/j.ccell.2022.07.004 ; https://github.com/mahmoodlab/PORPOISE
- Howard 2023 letter: https://doi.org/10.1016/j.ccell.2022.10.025 ; https://github.com/fmhoward/PORPOISE_SITE
- Howard 2021 site signatures: https://doi.org/10.1038/s41467-021-24698-1
- Vale-Silva 2021 MultiSurv: https://doi.org/10.1038/s41598-021-92799-4 ; https://github.com/luisvalesilva/multisurv
- Cheerla 2019: https://doi.org/10.1093/bioinformatics/btz342
- PROGPATH 2025: https://doi.org/10.1038/s41392-025-02374-w ; https://github.com/Valeyards/ProgPath
- Song 2025: https://arxiv.org/abs/2505.07683
- CHIEF: https://doi.org/10.1038/s41586-024-07894-z ; https://github.com/hms-dbmi/CHIEF
- TITAN: https://doi.org/10.1038/s41591-025-03982-3 ; https://arxiv.org/abs/2411.19666 ; https://github.com/mahmoodlab/TITAN
- Prov-GigaPath: https://doi.org/10.1038/s41586-024-07441-w
- UNI: https://doi.org/10.1038/s41591-024-02857-3 ; CONCH: https://doi.org/10.1038/s41591-024-02856-4
- MUSK: https://doi.org/10.1038/s41586-024-08378-w ; https://github.com/lilab-stanford/MUSK
- SurvPath: https://arxiv.org/abs/2304.06819 ; https://github.com/mahmoodlab/SurvPath
- MCAT: https://github.com/mahmoodlab/MCAT
- GPFM: https://arxiv.org/abs/2407.18449 ; PathBench: https://arxiv.org/abs/2505.20202
- Gustafsson 2026 breast benchmark: https://arxiv.org/abs/2604.24679
- Vergouwe 2010: https://doi.org/10.1093/aje/kwq223
- Jiang 2024 (DACHS/MCO): https://doi.org/10.1016/S2589-7500(23)00208-X ; https://github.com/KatherLab/crc-models-2022
- Kather 2019 PLoS Med (DACHS n=409 validation): https://doi.org/10.1371/journal.pmed.1002730
- Skrede 2020: https://doi.org/10.1016/S0140-6736(19)32998-8 ; Kleppe 2022: https://doi.org/10.1016/S1470-2045(22)00391-6
- TransSCOT (Henriksen/Skrede 2026): https://arxiv.org/abs/2602.22347
- PLCO images: https://cdas.cancer.gov/learn/plco/images/pathology ; NLST: https://cdas.cancer.gov/learn/nlst/images
- SurGen: https://arxiv.org/abs/2502.04946 ; https://doi.org/10.6019/S-BIAD1285
- HANCOCK: https://doi.org/10.1038/s41467-025-62386-6 ; https://hancock.research.fau.eu/
- LEOPARD: https://registry.opendata.aws/leopard/ ; https://arxiv.org/abs/2603.14187
- HiPS: https://doi.org/10.1038/s41591-023-02643-7 ; https://github.com/PathologyDataScience/HiPS
- Patho-Bench: https://github.com/mahmoodlab/Patho-Bench ; STAMP: https://github.com/KatherLab/STAMP (protocol doi:10.1038/s41596-024-01047-2)
