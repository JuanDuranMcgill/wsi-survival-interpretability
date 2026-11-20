# Explanation reliability and sample size: literature note

Prepared 2026-11-20 for the IER audit paper (WSI survival models, TCGA cohorts with 139-163 events).

Verification rule used: every reference below was checked online for title, authors, year, venue and DOI/URL
(publisher page, PubMed/Europe PMC record, Crossref, arXiv or proceedings page). Numbers quoted from a paper
were read in its abstract or record. Numbers marked **[own calculation]** are my arithmetic from standard
formulas, not from the literature. Items I could not verify are listed in section 7.

---

## 0. Bottom line

1. **Small samples make explanations unstable, not just tests underpowered.** With few samples, many
   near-equally-good models exist (Rashomon effect, underspecification), and they rank features differently.
   The classic quantitative result is from cancer outcome gene signatures: Ein-Dor et al. (2006) estimated that
   *several thousand* patients are needed for two independently derived prognostic gene lists to overlap by 50%.
2. **Instability shrinks with n, but slowly, and the n needed depends on what is being explained.**
   Population-level summaries stabilise before individual-level ones (Legha et al. 2026: 1100 vs 1800 patients
   in one case study). Feature rankings degrade when data are cut more than when features are cut
   (Lee et al. 2025). Model-induced SHAP disagreement is larger on smaller datasets (Hwang et al. 2026).
3. **Inference for explanations exists** (CIs and tests for Shapley/LOCO/permutation importance, confidence
   intervals for ranks), but it is rarely used in computational pathology, and none of it has been calibrated
   for MIL survival models on ~150 events.
4. **Pathology-specific evaluations (xMIL, Additive MIL, the 2026 MedIA benchmark) test faithfulness, not
   sample-size dependence.** I found no study that measures how MIL explanation reliability changes with cohort
   size. TCGA adds a known confounder (tissue-source-site signatures).
5. **Subgroup C-index comparisons are badly underpowered at this size.** Interaction effects need roughly 4x
   the sample of main effects (Brookes et al. 2004). Precise C estimates need 100-200+ events in the whole
   sample (Collins et al. 2016), which a subgroup of a 150-event cohort cannot have.
   **[own calculation]**: with ~150 events, a subgroup-vs-rest C-index difference must be about 0.15-0.25 to be
   detected with 80% power.

---

## 1. Instability of feature importance and attribution with small samples

### 1a. Why: many good models, different explanations

| Reference | What it shows | Link |
|---|---|---|
| Breiman L. Statistical Modeling: The Two Cultures. *Statistical Science* 2001;16(3):199-231. | Coins the "Rashomon effect": many datasets admit many approximately equally good models, which can tell different stories about which variables matter. | https://doi.org/10.1214/ss/1009213726 (PDF: https://www2.math.uu.se/~thulin/mm/breiman.pdf) |
| D'Amour A, Heller K, Moldovan D, ..., Sculley D. Underspecification Presents Challenges for Credibility in Modern Machine Learning. *JMLR* 2022;23(226):1-61. arXiv 2011.03395 (2020). | A pipeline is underspecified when it can return many predictors with equivalent held-out performance that behave differently. Examples include medical imaging, EHR risk prediction and medical genomics. | https://jmlr.org/papers/v23/20-1335.html |
| Fisher A, Rudin C, Dominici F. All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *JMLR* 2019;20(177):1-81. | "Model class reliance": the range of permutation importance across all well-performing models; a variable can be important for one good model and unimportant for another. | https://www.jmlr.org/papers/v20/18-760.html |
| Dong J, Rudin C. Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence* 2020;2(12):810-824. | Variable importance clouds; warns against overclaiming importance from a single model. | https://doi.org/10.1038/s42256-020-00264-0 |
| Marx C, Calmon F, Ustun B. Predictive Multiplicity in Classification. *ICML* 2020 (PMLR 119). | Near-equally-accurate models disagree on individual predictions; measures "ambiguity" and "discrepancy". | https://arxiv.org/abs/1909.06677 |
| Donnelly J, Katta S, Rudin C, Browne EP. The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance. *NeurIPS* 2023 (spotlight). | Single-model importance need not reflect the data distribution; proposes a bootstrap-over-Rashomon-set estimand that is stable to data perturbation. | https://arxiv.org/abs/2309.13775 ; https://neurips.cc/virtual/2023/poster/71545 |
| Krishna S, Han T, Gu A, Wu S, Jabbari S, Lakkaraju H. The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective. *TMLR* 2024. arXiv 2202.01602 (2022). | Explanation methods (SHAP, LIME, gradients, etc.) frequently disagree on the same model; 84% of interviewed practitioners hit this; 86% of user-study responses resolved it by arbitrary heuristics or not at all. Not a sample-size study, but shows "the explanation" is not unique. | https://arxiv.org/abs/2202.01602 |

### 1b. Empirical evidence that stability depends on sample size

| Reference | Finding relevant to n | Link |
|---|---|---|
| Ein-Dor L, Zuk O, Domany E. Thousands of samples are needed to generate a robust gene list for predicting outcome in cancer. *PNAS* 2006;103(15):5923-5928. | Main cause of non-overlapping prognostic gene lists is small training sets. Using "probably approximately correct" sorting, a typical 50% overlap between two breast-cancer outcome gene lists would need **several thousand** patients. The closest analogue to the paper's (E) test: "which features predict survival" is a ranking problem. | https://doi.org/10.1073/pnas.0601231103 |
| Michiels S, Koscielny S, Hill C. Prediction of cancer outcome with microarrays: a multiple random validation strategy. *Lancet* 2005;365(9458):488-492. | Re-analysed 7 outcome-signature studies: the gene list depended strongly on which patients were in the training set; 5 of 7 did not classify better than chance. | https://doi.org/10.1016/S0140-6736(05)17866-0 |
| Riley RD, Collins GS. Stability of clinical prediction models developed using statistical or machine learning methods. *Biometrical Journal* 2023;65(8):2200302. | Different development samples of the same size give different models; proposes instability plots and bootstrap instability measures as a routine check. | https://doi.org/10.1002/bimj.202200302 |
| Riley RD, Pate A, Dhiman P, Archer L, Martin GP, Collins GS. Clinical prediction models and the multiverse of madness. *BMC Medicine* 2023;21:502. | Smaller development datasets produce more divergent "multiverse" models; recommends large datasets, especially for subgroup reliability and fairness checks. | https://doi.org/10.1186/s12916-023-03212-y |
| Legha A, Ensor J, Whittle R, Archer L, Van Calster B, Christodoulou E, Snell KIE, Sadatsafavi M, Collins GS, Riley RD. Sequential sample size calculations and learning curves safeguard the robust development of a clinical prediction model for individuals. *J Clin Epidemiol* 2026;191:112117. arXiv 2509.15134. | Case study: fixed calculation said 342 patients; sequential bootstrapping needed **1100** for population-level stability and **1800** for individual-level stability. Shows the n needed depends on the level of the claim. | https://doi.org/10.1016/j.jclinepi.2025.112117 ; https://arxiv.org/abs/2509.15134 |
| Lee Y, Baruzzo G, Kim J, Seo J, Di Camillo B. Validity of Feature Importance in Low-Performing Machine Learning for Tabular Biomedical Data. *IEEE Access* 2025;13:176381-176391. | Removing samples ("data cutting") shifted feature ranks more than removing features; feature importance can stay valid at low performance **if n is adequate**; concludes insufficient data drives poor performance in tabular medical data. | https://doi.org/10.1109/ACCESS.2025.3618851 ; https://arxiv.org/abs/2409.13342 |
| Hwang H, Lee S, Rosenblatt L, Whang SE, Stoyanovich J. Explanation Multiplicity in SHAP: Characterization and Assessment. arXiv 2601.12654 (2026; listed as accepted at NeurIPS 2026 Evaluations & Datasets track). | SHAP outputs vary across repeated runs; **model-induced disagreement is generally larger on smaller datasets**, explainer-induced larger on larger ones; rank metrics expose top-feature changes that L2 hides. Advises treating single-run SHAP as a sample from a distribution. | https://arxiv.org/abs/2601.12654 |
| Ning Y, Li S, Ng YY, ..., Liu N. Variable importance analysis with interpretable machine learning for fair risk prediction (ShapleyVIC). *PLOS Digital Health* 2024;3(7):e0000542. | Notes black-box importance is unstable with limited samples; ShapleyVIC reproduced findings in subsamples of n=2,500 and 500 where logistic regression lost power. | https://doi.org/10.1371/journal.pdig.0000542 |
| Nogueira S, Sechidis K, Brown G. On the Stability of Feature Selection Algorithms. *JMLR* 2018;18(174):1-54. | Formal stability measure for selected feature sets with confidence intervals and tests; useful to quantify (E) stability across bootstraps. | https://jmlr.org/papers/v18/17-514.html |
| Strobl C, Boulesteix AL, Zeileis A, Hothorn T. Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics* 2007;8:25. | RF importance is biased toward predictors with more categories/scales, and bootstrap sampling contributes; relevant to the RF surrogate in (E). | https://doi.org/10.1186/1471-2105-8-25 |
| Molnar C, et al. General Pitfalls of Model-Agnostic Interpretation Methods for Machine Learning Models. In: *xxAI - Beyond Explainable AI*, LNAI 13200, Springer 2022, pp. 39-68. | Lists ignoring model/estimation uncertainty and interpreting a model that does not generalise as pitfalls. | https://doi.org/10.1007/978-3-031-04083-2_4 |

Adjacent "small n" literature (same mechanism, different field):

| Reference | Finding | Link |
|---|---|---|
| Varoquaux G. Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage* 2018;180(Pt A):68-77. | With ~100 samples, CV accuracy has error bars of about ±10%; fold spread underestimates true uncertainty. | https://doi.org/10.1016/j.neuroimage.2017.06.061 |
| Button KS, Ioannidis JPA, Mokrysz C, Nosek BA, Flint J, Robinson ESJ, Munafò MR. Power failure: why small sample size undermines the reliability of neuroscience. *Nat Rev Neurosci* 2013;14:365-376. | Low power gives both missed effects and inflated, poorly reproducible "significant" effects (winner's curse). | https://doi.org/10.1038/nrn3475 |
| Marek S, et al. Reproducible brain-wide association studies require thousands of individuals. *Nature* 2022;603:654-660. | Feature-outcome associations across many features only replicated reliably at n in the thousands. (Contested for cross-validated multivariate models: Spisak, Bingel, Wager, *Nature* 2023, https://pmc.ncbi.nlm.nih.gov/articles/PMC9995263.) | https://doi.org/10.1038/s41586-022-04492-9 |

---

## 2. Statistical inference for explanations

### 2a. Confidence intervals and tests for importance

| Reference | What it provides | Link |
|---|---|---|
| Williamson BD, Gilbert PB, Carone M, Simon N. Nonparametric variable importance assessment using machine learning techniques. *Biometrics* 2021;77(1):9-22. | Algorithm-agnostic, population-level importance (difference in R²-type predictiveness) with valid CIs when ML is used for estimation. | https://doi.org/10.1111/biom.13392 |
| Williamson BD, Gilbert PB, Simon NR, Carone M. A general framework for inference on algorithm-agnostic variable importance. *JASA* 2023;118(543):1645-1658. | Extends to general predictiveness measures (e.g. AUC-type), CIs and a test of zero importance. Directly applicable to "does this tissue type add discrimination?" | https://doi.org/10.1080/01621459.2021.2003200 |
| Williamson BD, Feng J. Efficient nonparametric statistical inference on population feature importance using Shapley values. *ICML* 2020, PMLR 119:10282-10291. | Shapley population variable importance (SPVIM): Shapley values of a *performance* measure, with asymptotic CIs and tests. Conceptually the inferential version of the paper's "Shapley on the C-index". | https://proceedings.mlr.press/v119/williamson20a.html |
| Covert I, Lundberg S, Lee SI. Understanding Global Feature Contributions With Additive Importance Measures (SAGE). *NeurIPS* 2020. | Shapley values of model loss (global predictive power); framework unifying permutation/LOCO-type measures. | https://papers.nips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html |
| Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-Free Predictive Inference for Regression. *JASA* 2018;113(523):1094-1111. | Introduces LOCO (leave-one-covariate-out) importance with distribution-free inference. | https://doi.org/10.1080/01621459.2017.1307116 |
| Verdinelli I, Wasserman L. Feature Importance: A Closer Look at Shapley Values and LOCO. *Statistical Science* 2024;39(4):623-636. | Shapley values do not remove the effect of correlated features; questions the Shapley axioms for importance; recommends a modified LOCO. Relevant because tissue-type proportions are correlated. | https://doi.org/10.1214/24-STS937 |
| Watson DS, Wright MN. Testing conditional independence in supervised learning algorithms. *Machine Learning* 2021;110(8):2107-2129. | Conditional predictive impact (CPI): knockoff-based test of a feature's conditional importance with type I error control; R package `cpi`. | https://arxiv.org/abs/1901.09917 (journal record: https://ora.ox.ac.uk/objects/uuid:7699767d-de4e-4d90-962b-fa4892929cf5) |
| Altmann A, Toloşi L, Sander O, Lengauer T. Permutation importance: a corrected feature importance measure. *Bioinformatics* 2010;26(10):1340-1347. | PIMP: permutes the outcome to give p-values for RF importance. Directly usable for the RF surrogate in (E). | https://doi.org/10.1093/bioinformatics/btq134 |
| Molnar C, Freiesleben T, König G, et al. Relating the Partial Dependence Plot and Permutation Feature Importance to the Data Generating Process. *xAI 2023*, CCIS, Springer, pp. 456-479. | Separates Monte-Carlo variance from model (refit) variance; CIs for PFI that include refitting uncertainty. | https://doi.org/10.1007/978-3-031-44064-9_24 |
| Neuhof B, Benjamini Y. Confident Feature Ranking. *AISTATS* 2024, PMLR 238:1468-1476. | Simultaneous CIs for feature ranks from importance values; example where ranks were unstable at n=50. Useful for reporting "top tissue type" claims with uncertainty. | https://proceedings.mlr.press/v238/neuhof24a.html |

### 2b. Multiple testing and power vs n

| Reference | Point | Link |
|---|---|---|
| Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS B* 1995;57(1):289-300. | BH procedure. When only one or a few features carry signal, BH's first threshold equals Bonferroni's (α/m). | https://doi.org/10.1111/j.2517-6161.1995.tb02031.x |
| Pawitan Y, Michiels S, Koscielny S, Gusnanto A, Ploner A. False discovery rate, sensitivity and sample size for microarray studies. *Bioinformatics* 2005;21(13):3017-3024. | FDR depends on the proportion of true signals, effect size, variability and **sample size**; small studies have high FDR, and controlling FDR alone can give an unacceptably high false-negative rate. Recommends reporting sensitivity (FNR) curves with FDR. | https://doi.org/10.1093/bioinformatics/bti448 |
| Jung SH. Sample size for FDR-control in microarray data analysis. *Bioinformatics* 2005;21(14):3097-3104. | Closed-form sample size for a target number of true discoveries at a given FDR. | https://doi.org/10.1093/bioinformatics/bti456 |
| Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. *Biometrics* 1983;39(2):499-503. | Power for Cox models depends on the number of **events**, not patients. | https://doi.org/10.2307/2531021 |
| Hsieh FY, Lavori PW. Sample-size calculations for the Cox proportional hazards regression model with nonbinary covariates. *Control Clin Trials* 2000;21(6):552-560. | Events needed for a continuous covariate: D = (z₁₋α/₂ + z₁₋β)² / (σ² (log HR)²). | https://doi.org/10.1016/s0197-2456(00)00104-5 |

**[own calculation]** Using the Hsieh-Lavori formula with a standardised feature (σ = 1), 80% power, and a
Bonferroni-type threshold α = 0.05/m (the BH threshold when only one feature is truly associated):

| m features tested | Events for HR 1.2/SD | Events for HR 1.3/SD | Events for HR 1.5/SD | Smallest detectable HR/SD with 139 events | with 163 events |
|---|---|---|---|---|---|
| 1 | 236 | 114 | 48 | 1.27 | 1.25 |
| 10 | 400 | 193 | 81 | 1.36 | 1.33 |
| 30 | 478 | 231 | 97 | 1.40 | 1.37 |
| 100 | 562 | 271 | 114 | 1.44 | 1.40 |

Reading: with 139-163 events and tens of candidate features, only features with a per-SD hazard ratio of about
1.35-1.45 or more are reliably detectable after FDR correction. Univariable Cox test assumed; a surrogate-ranked
feature tested on the same data would also need selection-aware inference (sample splitting), which costs more power.

---

## 3. Pathology / MIL-specific evidence

### 3a. Attention and attribution faithfulness in MIL

| Reference | Finding | Link |
|---|---|---|
| Jain S, Wallace BC. Attention is not Explanation. *NAACL-HLT* 2019, pp. 3543-3556. | Attention weights often do not correlate with gradient importance, and different attention distributions can give the same prediction (NLP). | https://doi.org/10.18653/v1/N19-1357 |
| Wiegreffe S, Pinter Y. Attention is not not Explanation. *EMNLP-IJCNLP* 2019, pp. 11-20. | Rebuttal: whether attention explains depends on the definition; proposes tests including variance across random seeds. | https://doi.org/10.18653/v1/D19-1002 |
| Javed SA, Juyal D, Padigela H, Taylor-Weiner A, Yu L, Prakash A. Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology. *NeurIPS* 2022 (Adv. Neural Inf. Process. Syst. 35). | Additive MIL gives exact per-patch contributions; its spatial credit assignment matched pathologist-used regions and improved on classical attention heatmaps. | https://arxiv.org/abs/2206.01794 ; https://proceedings.nips.cc/paper_files/paper/2022/file/82764461a05e933cc2fd9d312e107d12-Paper-Conference.pdf |
| Early J, Evers C, Ramchurn S. Model Agnostic Interpretability for Multiple Instance Learning. *ICLR* 2022. | Model-agnostic MIL explanations (MILLI) beat inherently interpretable MIL (attention) by up to 30% in interpretability accuracy. | https://arxiv.org/abs/2201.11701 |
| Hense J, et al. xMIL: Insightful Explanations for Multiple Instance Learning in Histopathology. *NeurIPS* 2024 (Adv. Neural Inf. Process. Syst. 37), pp. 8300-8328. | Patch-flipping faithfulness on 4 histopathology datasets; attention is not faithful; xMIL-LRP most faithful, especially for biomarker tasks. | https://doi.org/10.52202/079017-0266 ; https://arxiv.org/abs/2406.04280 |
| Jamshidi Idaji M, Hense J, ..., Müller KR. Beyond attention heatmaps: How to get better explanations for multiple instance learning models in histopathology. *Medical Image Analysis* 2026;113:104148. arXiv 2603.08328. | Benchmarks 6 explanation methods across classification, regression **and survival** tasks, several MIL architectures and 2 encoders; perturbation, LRP and IG beat attention and gradient saliency. Does not (per abstract) vary dataset size. | https://doi.org/10.1016/j.media.2026.104148 ; https://arxiv.org/abs/2603.08328 |
| Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity Checks for Saliency Maps. *NeurIPS* 2018. | Some saliency methods are nearly independent of the model weights and of the training labels. | https://arxiv.org/abs/1810.03292 |
| Arun N, et al. Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging. *Radiology: AI* 2021;3(6):e200267. | Saliency maps from models with the same architecture but different initialisation (repeatability) and from different architectures (reproducibility) agreed poorly for most methods. | https://doi.org/10.1148/ryai.2021200267 |
| Mammadov A, Le Folgoc L, Hocquet G, Gori P. Reducing Variability of Multiple Instance Learning Methods for Digital Pathology. *MICCAI* 2025, LNCS pp. 506-516. | Run-to-run test AUC of MIL methods varies by up to 10-15 points from initialisation, batch order and learning rate, more than the gap between methods. (My inference, not tested there: explanations from a single run carry this variance too.) | https://doi.org/10.1007/978-3-032-04981-0_48 ; https://arxiv.org/abs/2507.00292 |

### 3b. Spurious signal in TCGA WSIs

| Reference | Finding | Link |
|---|---|---|
| Howard FM, Dolezal J, Kochanny S, ..., Kather JN, Cipriani N, Grossman RL, Pearson AT. The impact of site-specific digital histology signatures on deep learning model accuracy and bias. *Nat Commun* 2021;12:4423. | Tissue-submitting site is detectable from TCGA histology even after colour normalisation; site signatures bias survival, mutation and stage prediction; site-preserved splits remove optimism. | https://doi.org/10.1038/s41467-021-24698-1 |
| Dehkharghanian T, et al. Biased data, biased AI: deep networks predict the acquisition site of TCGA images. *Diagn Pathol* 2023;18:67. | Features from networks never trained for it identify the acquisition site (70% DenseNet, >86% KimiaNet) across 8,579 slides from >140 sites. | https://doi.org/10.1186/s13000-023-01355-3 |
| Schmitt M, et al. Hidden Variables in Deep Learning Digital Pathology and Their Potential to Cause Batch Effects: Prediction Model Study. *J Med Internet Res* 2021;23(2):e23436. | CNNs learn patient age, slide preparation date, slide origin and scanner type (dermatopathology). | https://doi.org/10.2196/23436 |

### 3c. Context: tissue-level survival explanations

| Reference | Relevance | Link |
|---|---|---|
| Kather JN, et al. Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLoS Med* 2019;16(1):e1002730. | Nine tissue classes; a "deep stroma score" was an independent prognostic factor. Precedent for tissue-type-level prognostic claims. | https://doi.org/10.1371/journal.pmed.1002730 |
| Chen RJ, et al. Pan-cancer integrative histology-genomic analysis via multimodal deep learning (PORPOISE). *Cancer Cell* 2022;40(8):865-878. | Uses attention heatmaps and integrated gradients on 14 TCGA cohorts to make prognostic morphology claims (e.g. TIL presence favourable in 9 of 14 cancers). Example of explanation claims at similar cohort sizes without per-claim inferential tests. | https://doi.org/10.1016/j.ccell.2022.07.004 |
| Lamane A, Diab AR, Wu RC, Lotter W. SAGE: Semantic Explainability of Attention-Based Survival Models in Computational Pathology. arXiv 2608.02803 (2026; iMIMIC workshop). | Concept-level (25 histological concepts) global explanations of ABMIL survival models on 7 TCGA cohorts; abstract reports no significance testing. Closest recent related work. | https://arxiv.org/abs/2608.02803 |

**Gap found:** none of the MIL explanation papers above reports how faithfulness or stability of explanations
changes with the number of slides or events. Benchmarks vary task, architecture and encoder, not cohort size.

---

## 4. Subgroup / reliability analysis: how many events to show a C-index differs

### 4a. MED3pa

Lefebvre O, Camirand Lemyre F, Ethier JF, Chikouche LH, Amriou L, Poenaru D, Vallières M. Predictive performance
precision analysis in medicine: identification of low-confidence predictions at patient and profile levels
(MED3pa I). *JAMIA* 2026;33(6):1100-1111. https://doi.org/10.1093/jamia/ocag034 (preprint: medRxiv
10.1101/2025.08.22.25334254; code https://github.com/MEDomicsLab/MED3pa).

- Individual (IPC), aggregated/profile (APC) and mixed (MPC) confidence; profiles from a tree; performance shown
  as metrics-by-declaration-rate curves.
- Evaluated on large datasets (e.g. 16,271 MIMIC-IV admissions; 148,587 training admissions for one-year mortality).
- In the main text I read, there is **no minimum profile size and no statistical test or CI for profile-level
  performance**; results are point estimates (supplement not read). So the IER audit's (R) test of whether a
  flagged profile's C-index actually differs adds something MED3pa itself does not provide, and its failure at
  ~150 events is what the power literature below predicts.

### 4b. General subgroup-analysis power

| Reference | Quantitative guidance | Link |
|---|---|---|
| Brookes ST, Whitely E, Egger M, Smith GD, Mulheran PA, Peters TJ. Subgroup analyses in randomized trials: risks of subgroup-specific analyses; power and sample size for the interaction test. *J Clin Epidemiol* 2004;57(3):229-236. | A trial with 80% power for the overall effect had only **29% power** for an interaction of the same size; about **4x** the sample is needed, more for smaller interactions. | https://doi.org/10.1016/j.jclinepi.2003.08.009 |
| Gelman A. "You need 16 times the sample size to estimate an interaction than to estimate a main effect." Blog, 15 Mar 2018; also Gelman, Hill, Vehtari, *Regression and Other Stories* (2020), Sec. 16.4. | 16x if the interaction is half the main effect; 4x if equal size. | https://statmodeling.stat.columbia.edu/2018/03/15/need16/ |
| Rothwell PM. Subgroup analysis in randomised controlled trials: importance, indications, and interpretation. *Lancet* 2005;365:176-186. | Post-hoc subgroup findings should be treated sceptically; power for subgroups or pool data. | https://doi.org/10.1016/S0140-6736(05)17709-5 |
| Lipkovich I, Dmitrienko A, D'Agostino RB Sr. Tutorial in biostatistics: data-driven subgroup identification and analysis in clinical trials. *Stat Med* 2017;36(1):136-196. | Data-driven subgroup search needs selection-aware inference (honest estimation, resampling) to avoid optimism. | https://doi.org/10.1002/sim.7064 |
| Chung Y, Kraska T, Polyzotis N, Tae KH, Whang SE. Slice Finder: Automated Data Slicing for Model Validation. *ICDE* 2019, pp. 1550-1553. | ML slice discovery that treats each problematic slice as a hypothesis, requiring significance and effect size; the extended version adds false-discovery control. | https://doi.org/10.1109/ICDE.2019.00139 ; https://arxiv.org/abs/1807.06068 |

### 4c. Precision of the C-index vs events

| Reference | Guidance | Link |
|---|---|---|
| Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Stat Med* 2016;35(2):214-226. | Validation of survival models needs **≥100 events, ideally ≥200**, for unbiased, precise performance estimates. | https://doi.org/10.1002/sim.6787 |
| Vergouwe Y, Steyerberg EW, Eijkemans MJ, Habbema JD. Substantial effective sample sizes were required for external validation studies of predictive logistic regression models. *J Clin Epidemiol* 2005;58(5):475-483. | ≥100 events and 100 non-events; specific hypotheses may need substantially more. | https://doi.org/10.1016/j.jclinepi.2004.06.017 |
| Pavlou M, et al. Estimation of required sample size for external validation of risk models for binary outcomes. *Stat Methods Med Res* 2021;30. | SE < 0.025 for the C-statistic needs **60-170 events** (true C 0.64-0.85, prevalence 0.05-0.3). | https://doi.org/10.1177/09622802211007522 |
| Riley RD, Collins GS, Ensor J, et al. Minimum sample size calculations for external validation of a clinical prediction model with a time-to-event outcome. *Stat Med* 2022;41(7):1280-1295. | Simulation-based sizing for survival-model validation (mainly calibration precision). | https://doi.org/10.1002/sim.9275 |
| Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441. | Development sample-size criteria beyond events-per-variable. | https://doi.org/10.1136/bmj.m441 |
| Kang L, Chen W, Petrick NA, Gallas BD. Comparing two correlated C indices with right-censored survival outcome: a one-shot nonparametric approach. *Stat Med* 2015;34(4):685-703. | Variance/covariance of Harrell's C and a z-test (R `compareC`). | https://doi.org/10.1002/sim.6370 |
| Hartman N, Kim S, He K, Kalbfleisch JD. Pitfalls of the concordance index for survival outcomes. *Stat Med* 2023;42(13):2179-2190. | C-index limitations are amplified for survival outcomes. | https://doi.org/10.1002/sim.9717 |
| Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 1982;143(1):29-36. | SE formula for AUC used below. | https://doi.org/10.1148/radiology.143.1.7063747 |

**[own calculation]** Approximate minimum detectable difference (80% power, two-sided α = 0.05) between the
C-index in a flagged subgroup and in the rest, assuming 150 events total, C ≈ 0.65, independent groups, Hanley-McNeil
SE as a binary-outcome stand-in for Harrell's C (censoring will make the true SE similar or larger):

| Subgroup share of patients | Subgroup events | SE(C) subgroup | SE(C) rest | Detectable ΔC |
|---|---|---|---|---|
| 10% | 15 | 0.085 | 0.028 | 0.25 |
| 20% | 30 | 0.060 | 0.030 | 0.19 |
| 30% | 45 | 0.049 | 0.032 | 0.16 |
| 50% | 75 | 0.038 | 0.038 | 0.15 |

(Assumes 500 non-events; with 250 or 850 non-events the detectable ΔC for a 20-30% subgroup stays 0.16-0.20;
with C = 0.60 or 0.70 it is about 0.16.) A data-driven subgroup also needs honest (held-out) estimation, which
roughly halves the usable events again.

---

## 5. Quantitative guidance at a glance

| Claim type | Guidance from literature | Source |
|---|---|---|
| Stable ranked list of prognostic features | Thousands of patients for 50% overlap of gene lists (breast cancer) | Ein-Dor 2006 |
| Population-level vs individual-level model stability | 1100 vs 1800 patients in one case study (vs 342 from fixed formula) | Legha 2026 |
| Precise overall C-index | ≥100 events, ideally ≥200 | Collins 2016; Vergouwe 2005 |
| C-statistic SE < 0.025 | 60-170 events | Pavlou 2021 |
| Interaction / subgroup difference | ~4x (equal-size) to 16x (half-size) the main-effect sample | Brookes 2004; Gelman 2018 |
| CV error bars at n≈100 | about ±10% accuracy | Varoquaux 2018 |
| Feature-outcome association after FDR, ~150 events | per-SD HR ≳ 1.35-1.45 detectable for 10-100 features | **[own calculation]**, Hsieh & Lavori 2000 |
| Subgroup-vs-rest ΔC, ~150 events | ΔC ≳ 0.15-0.25 detectable | **[own calculation]**, Hanley & McNeil 1982 |

---

## 6. Gaps the paper could fill

1. **No study measures explanation reliability vs cohort size for MIL survival models.** The IER audit could run a
   learning-curve analysis: subsample events (e.g. 25/50/75/100%), refit, and track pass rates of I, E and R.
   Extrapolating where pass rates plateau would give the first "events needed" estimate for WSI explanations
   (the same logic as Ein-Dor's PAC sorting and Legha's sequential learning curves).
2. **Separate "explanation is wrong" from "test is underpowered".** Report, for each audit test, the minimum
   detectable effect at the available events (as in the tables above). A failed test with a large MDE is
   inconclusive, not a refutation. Low power also inflates the explanations that do pass (Button 2013).
3. **Ground-truth simulation.** Plant a known tissue-type effect of controlled size in TCGA-like data and measure
   the audit's power curve vs events; no pathology paper does this for explanations.
4. **Model multiplicity across seeds/folds.** Quantify Rashomon-set spread of tissue-type Shapley values across
   retraining (Donnelly 2023; Mammadov 2025 shows MIL run-to-run variance is large), separating model-induced
   from data-induced instability (Hwang 2026).
5. **Inference-ready versions of each test**: Williamson-Feng SPVIM or Williamson 2023 for (I), PIMP/CPI with
   sample splitting for (E), and Kang 2015-style variance plus held-out profile estimation for (R) and MED3pa,
   which itself reports no profile-level CIs.
6. **Site effects as a competing explanation** at small n (Howard 2021): site-preserved splits could be part of
   the audit.

---

## 7. Not verified or deliberately left out

- **Campanella et al., Nat Med 2019;25(8):1301-1309 (doi 10.1038/s41591-019-0508-1)**: verified as a paper, but I
  could not open the full text to confirm any statement on how many slides are needed, so no number is cited.
- **MED3pa supplementary material** not read; a minimum profile size or test may be defined there.
- **Jamshidi Idaji 2026**: only the abstract was read; I cannot confirm whether the full text studies dataset size.
- **Hwang et al. 2026** is an arXiv preprint whose comments say accepted at NeurIPS 2026; not checked on the
  NeurIPS site.
- Left out because unverifiable or not peer reviewed: a GitHub SHAP-stability churn study, a ResearchGate
  class-imbalance SHAP preprint (claimed "N > 500" threshold), arXiv 2605.15154 (RoSHAP), arXiv 2605.21492.
- Plumb et al. "Towards a More Rigorous Science of Blindspot Discovery in Image Classification Models"
  (arXiv 2207.04104) exists but I did not confirm its venue or any statistical-validity content.
- Pavlou et al. 2021: volume confirmed (30) but issue/pages not checked.
