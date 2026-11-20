# Sample-size theory for survival prediction models: grounding for the "under ~200 events is too small" claim

Prepared 2026-11-20. I checked every reference below for title, authors, year, venue and DOI against Europe PMC (PubMed metadata), the full text where it was open access, or Crossref. Section 8 says how each one was checked. Where I could not check the exact wording, the text says so.

Cohort context used in the calculations (from `docs/SHIRIN_CLUSTER_RESULTS.md`): BLCA 379 patients and 163 progression events (event fraction 0.43); BRCA 1,060 patients and 139 events (event fraction 0.13).

---

## 0. Bottom line

| Question | Answer (events) | Main sources |
|---|---|---|
| Rules of thumb for a Cox model | 10 events per variable (EPV) (Peduzzi 1995), 15 (Harrell, p < m/15), 20 or more when predictors are low-prevalence binary (Ogundimu 2016). Vittinghoff and McCulloch (2007) show it can sometimes go below 10 | §1 |
| What 139 to 163 events support | about 14 to 16 parameters at EPV 10; 9 to 11 at m/15; **only about 4 to 8 parameters** under Riley criterion (i) when C is 0.65 to 0.70 | §1, §2, §6 |
| (a) Cox model, ~25 clinical parameters | 250 (EPV 10), 375 (m/15), 500 (EPV 20); Riley criteria: **about 510 events if C ≈ 0.70, about 880 if C ≈ 0.65** (Riley's own 25-parameter survival example needs 577 events, EPP 23) | §2, §7 |
| (b) Estimate a C-index to ±0.05 (95% CI) | **about 125 to 150 events.** ±0.025 needs about 520 to 600. Validation guidance: at least 100, preferably 200 or more events (Collins 2016) | §3, §7 |
| (c) Detect a subgroup difference | Interaction needs **4 times** the events of a same-size main effect (Brookes 2001/2004), and 16 times if it is half the size (Gelman). For a C-index difference between two equal halves: **about 310 events for ΔC = 0.10 and about 1,250 for ΔC = 0.05** (80% power). For a ratio of hazard ratios of 2: about 260 events; for 1.5: about 760 | §4, §7 |
| Deep models on whole-slide-image features | No closed-form EPV exists. Riley's criteria only give a **lower bound**. Flexible machine-learning models needed more than 10 times the EPV of logistic regression (van der Ploeg 2014). In survival data they needed about 1.5 times (random forests) and about 2 times (neural nets) the regression minimum (Infante 2023) | §5 |

---

## 1. Events-per-variable (EPV) rules for Cox and survival models

**Harrell (Regression Modeling Strategies, RMS).**
- RMS course notes (hbiostat.org/rmsc, §4.4 "Overfitting and Limits on Number of Predictors"), verbatim bullets: "p should be < m/15"; "p is number of parameters in full model or number of *candidate* parameters in a stepwise analysis, and m is the effective sample size". For failure (survival) time, the "limiting sample size" m is the "number of failures". Footnote: this is approximate, because censored observations may boost the effective sample size somewhat. The notes say the rule was "Derived from simulations to find minimum sample size so that apparent discrimination = validated discrimination", that "If true R² is high, many parameters can be estimated from smaller samples", and that it "Ignores sample size needed just to estimate the intercept or … the underlying distribution function". The rule is cited there to Harrell 1984 and 1985, Peduzzi 1995 and 1996, Smith 1992, Vittinghoff 2006 [sic; the paper is 2007] and van der Ploeg 2014. URL: https://hbiostat.org/rmsc/multivar
- Book: Harrell FE Jr. *Regression Modeling Strategies*, 2nd ed. Springer, 2015. doi:10.1007/978-3-319-19425-7. (The publisher's chapter PDF was behind a login, so I did not read the printed wording.)
- Riley et al. 2019 (below) summarise it this way: "Harrell generally recommends at least 15 EPP".
- **Not verified:** the "m/10 or m/20" wording often attributed to Harrell, Lee & Mark 1996. The 1996 paper exists (metadata below), but I could not reach its full text, and searches did not turn up that sentence. Cite the m/15 rule from RMS instead, or say "10 to 20 EPV" with a secondary source.
- Harrell FE Jr, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Stat Med* 1996;15(4):361-387. doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4. PMID 8668867. The abstract covers overfitting, the C index, calibration, and bootstrap or cross-validation validation.
- Origin of the rule: Harrell FE, Lee KL, Califf RM, Pryor DB, Rosati RA. Regression modelling strategies for improved prognostic prediction. *Stat Med* 1984;3(2):143-152. doi:10.1002/sim.4780030207. PMID 6463451. (Metadata verified only.)

**Concato et al. 1995 (Part I) and Peduzzi et al. 1995 (Part II), the Cox simulation studies.**
- Concato J, Peduzzi P, Holford TR, Feinstein AR. Importance of events per independent variable in proportional hazards analysis. I. Background, goals, and general strategy. *J Clin Epidemiol* 1995;48(12):1495-1501. doi:10.1016/0895-4356(95)00510-2. PMID 8543963. Describes the design: 673 patients from a coronary artery bypass trial, with Monte Carlo resampling.
- Peduzzi P, Concato J, Feinstein AR, Holford TR. Importance of events per independent variable in proportional hazards regression analysis. II. Accuracy and precision of regression estimates. *J Clin Epidemiol* 1995;48(12):1503-1510. doi:10.1016/0895-4356(95)00048-8. PMID 8543964.
  - Design: 673 patients, 252 deaths and 7 predictors, so 36 EPV in the full data. 500 simulations at each EPV of 2, 5, 10, 15, 20 and 25.
  - Finding: as EPV fell, coefficients became biased, 90% CI coverage was wrong, variance estimates failed, and Z tests lost validity. The abstract concludes: "the value of EPV = 10 seems most prudent".
- Companion paper for logistic regression: Peduzzi P, Concato J, Kemper E, Holford TR, Feinstein AR. A simulation study of the number of events per variable in logistic regression analysis. *J Clin Epidemiol* 1996;49(12):1373-1379. doi:10.1016/S0895-4356(96)00236-3. PMID 8970487. No major problems at EPV of 10 or more.

**Vittinghoff & McCulloch 2007, relaxing the rule.**
- Vittinghoff E, McCulloch CE. Relaxing the rule of ten events per variable in logistic and Cox regression. *Am J Epidemiol* 2007;165(6):710-718. doi:10.1093/aje/kwk052. PMID 17182981.
- From the abstract: "a range of circumstances in which coverage and bias were within acceptable levels despite less than 10 EPV", plus "other factors that were as influential as or more influential than EPV". Their conclusion is that the rule can be relaxed, particularly for sensitivity analyses of confounding control. The often-quoted "5 to 9 EPV is usually fine" is **not** in the abstract; I did not read the full text.
- Note: this paper concerns bias and coverage of regression coefficients, not predictive accuracy.

**Later critiques.**
- van Smeden M, de Groot JA, Moons KGM, Collins GS, Altman DG, Eijkemans MJC, Reitsma JB. No rationale for 1 variable per 10 events criterion for binary logistic regression analysis. *BMC Med Res Methodol* 2016;16:163. doi:10.1186/s12874-016-0267-3. PMCID PMC5122171. Conclusion: "The current evidence supporting EPV rules for binary logistic regression is weak". Problems at low EPV also depend on total sample size and on separation.
- van Smeden M, Moons KGM, de Groot JA, Collins GS, Altman DG, Eijkemans MJC, Reitsma JB. Sample size for binary logistic prediction models: beyond events per variable criteria. *Stat Methods Med Res* 2019;28(8):2455-2474. doi:10.1177/0962280218784726. PMCID PMC6710621. Finding: "EPV does not have a strong relation with metrics of predictive performance". Out-of-sample performance is better approximated by the number of predictors, the total sample size and the events fraction.
- Ogundimu EO, Altman DG, Collins GS. Adequate sample size for developing prediction models is not simply related to events per variable. *J Clin Epidemiol* 2016;76:175-182. doi:10.1016/j.jclinepi.2016.02.031. PMCID PMC5045274. This is a Cox model study using more than 2 million general-practice records. "EPV ≥ 20 generally eliminates bias in regression coefficients when many low-prevalence predictors are included in a Cox model". With variable selection, "EPV = 10 was clearly a poor choice … EPV ≥ 25 was a better choice" (full text).
- Riley et al. 2019 (§2): "others identify situations where at least 20 EPP or up to 50 EPP are required".

**Bottom line for EPV:** the rules range from 10 to 20 or more events per candidate parameter. Here "candidate" means every parameter examined, including dummy codes, spline terms and interactions. The modern view is that no single EPV works, and that the required number depends on the model's R² (equivalently its C), the event fraction and p.

---

## 2. Riley et al. minimum sample size for developing a time-to-event model (pmsampsize)

**References.**
- Riley RD, Snell KIE, Ensor J, Burke DL, Harrell FE Jr, Moons KGM, Collins GS. Minimum sample size for developing a multivariable prediction model: PART II, binary and time-to-event outcomes. *Stat Med* 2019;38(7):1276-1296. doi:10.1002/sim.7992. PMCID PMC6519266. (Full text read.)
- Riley RD, Snell KIE, Ensor J, et al. Part I, continuous outcomes. *Stat Med* 2019;38(7):1262-1275. doi:10.1002/sim.7993.
- Riley RD, Ensor J, Snell KIE, Harrell FE Jr, Martin GP, Reitsma JB, Moons KGM, Collins G, van Smeden M. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441. doi:10.1136/bmj.m441. PMID 32188600. (Metadata verified. bmj.com returned 403, so the content summary below comes from the 2019 Stat Med paper, which the BMJ article summarises.)

**The three criteria for binary and time-to-event outcomes** (2019 Part II abstract, verbatim):
1. "small optimism in predictor effect estimates as defined by a global shrinkage factor of ≥0.9";
2. "small absolute difference of ≤ 0.05 in the model's apparent and adjusted Nagelkerke's R²";
3. "precise estimation of the overall risk in the population". Their example is a 95% CI for cumulative incidence at a key time point within ±0.05.

The minimum sample size is the largest n that meets all three.

**Key formulas** (2019 Part II, equations 11 and 26):
- Criterion (i): n = p / [(S − 1) · ln(1 − R²_CS,adj / S)], with S = 0.9.
- Criterion (ii): S ≥ R²_CS,adj / (R²_CS,adj + δ · max R²_CS), with δ = 0.05. If this S is above 0.9, re-solve criterion (i) with it.
- For survival outcomes the events follow from n × event rate × mean follow-up.
- **p is the number of candidate predictor parameters.** Verbatim: "researchers should define p as the total number of parameters due to all predictors considered (screened), and not just the subset that are included in the final model … As Harrell notes, the value of p should be honest". A categorical predictor with k categories counts as k − 1 parameters.
- **When R² is unknown:** "we suggest that sample sizes be derived assuming the value of R²_CS,adj corresponds to an R²_Nagelkerke of 0.15". R²_Nagelkerke of 0.1 to 0.2 is typical in low signal-to-noise settings.
- **Getting R² from a reported C-index** (via Jinks et al. 2015):
  - D = 5.50(C − 0.5) + 10.26(C − 0.5)³
  - R²_D = (π/8)D² / (π²/6 + (π/8)D²)
  - Treat R²_D as a stand-in for Royston's R², convert it to O'Quigley's R², then compute LR = −E · ln(1 − R²_OQ) and R²_CS = 1 − exp(−LR/n).

**Worked time-to-event example in the paper** (recurrent VTE, almost exactly the "25 clinical variables" case):
- Inputs: C = 0.69, giving D = 1.115, R²_D = 0.229, R²_OQ = 0.328 and R²_CS,adj = 0.051; **25 predictor parameters**.
- Result: "n = 4285.5 and thus 4286 participants … ≈ 577 outcome events, and thus an EPP of 577/25 ≈ 23. This is over twice the 'EPP of at least 10' rule of thumb. … an EPP of 10 only ensures a shrinkage factor of 0.79".
- Fewer parameters lower the total: with about 2,000 participants, p must drop to 12.
- The paper's other example (Chagas, binary, p = 24, high R²) needed only EPP 4.8: "This reinforces why rules of thumb (eg, 10 EPP) should be avoided."
- The authors advise against shrinking n by accepting S < 0.9; they say to reduce p instead, for example with "unsupervised learning techniques such as principal component analysis … which are blinded to the outcome data".

**My re-implementation.** I re-implemented the equations and checked them against the paper's numbers: n = 4285.5 for p = 25 and R²_CS = 0.051; D = 1.115, R²_D = 0.229 and R²_OQ = 0.328 for C = 0.69. Script: `C:\Users\sorkw\AppData\Local\Temp\claude\fac5aeb26ec781c0\67fa56ad-d900-43a5-8b9b-d66628bbc1eb\scratchpad\lit\riley_calc.py`. Output: `...\scratchpad\lit\riley_calc_out.txt`.
- With the C-index route, the required events come out almost independent of cohort size: E ≈ 9p / [−ln(1 − R²_OQ)] for S = 0.9.

| C (assumed, validated) | R²_OQ | events for p = 5 | p = 10 | **p = 25** | EPP | max p that 163 / 139 events support |
|---|---|---|---|---|---|---|
| 0.60 | 0.110 | 386 | 772 | **1,930** | 77 | 2.1 / 1.8 |
| 0.65 | 0.225 | 175 | 351 | **~880** | 35 | 4.6 / 3.9 |
| 0.70 | 0.354 | 102 | 203 | **~510** | 20 | 8.0 / 6.8 |
| 0.75 | 0.481 | 67 | 135 | **~340** | 14 | 12.1 / 10.2 |

- With the default R²_Nagelkerke = 0.15, p = 25 and all three criteria:
  - BLCA-like inputs: event fraction 0.43, an assumed mean follow-up of 2 years, risk at 3 years. This gives about 672 events (EPP 27). Criterion (iii) alone needs about 194 events.
  - BRCA-like inputs: event fraction 0.13, an assumed mean follow-up of 3.5 years, risk at 5 years. This gives about 275 events (EPP 11).
  - Caveat: in pmsampsize's exponential approximation, max R²_CS depends on the event rate and therefore on the time units chosen. Treat the C-based rows above as the more defensible numbers.

**Machine learning.** Infante et al. 2023 (§5) state that Riley's guidance "works with traditional regression models and it is not an option for ML techniques".

---

## 3. External validation and precision of the C-index

**Rules of thumb for validation studies.**
- Vergouwe Y, Steyerberg EW, Eijkemans MJC, Habbema JDF. Substantial effective sample sizes were required for external validation studies of predictive logistic regression models. *J Clin Epidemiol* 2005;58(5):475-483. doi:10.1016/j.jclinepi.2004.06.017. PMID 15845334.
  - Detecting a drop in c from 0.83 to 0.73 (80% power) needed 81 to 106 events.
  - "We suggest a minimum of 100 events and 100 nonevents for external validation samples."
- Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Stat Med* 2016;35(2):214-226. doi:10.1002/sim.6787. PMCID PMC4738418.
  - Six Cox-based prognostic models (QRISK2 and others) were resampled from a large UK general-practice database.
  - "externally validating a prognostic model requires a minimum of 100 events, preferably 200 or more events".
  - Full text: "at 100 events, over 80% of simulations yield estimates of the c-index within 5% of the true value and over 60% … within 2.5%". The mean CI width drops steeply up to 50 to 100 events, then flattens.
  - The standard error of Royston's D falls roughly as 1/√events: empirical SE 0.148 at 100 events, 0.066 at 500 and 0.046 at 1,000 (QRISK2 men). The model-based SE of D is too large, and bootstrap SEs are recommended.
- Riley RD, Collins GS, Ensor J, Archer L, Booth S, Mozumder SI, Rutherford MJ, van Smeden M, Lambert PC, Snell KIE. Minimum sample size calculations for external validation of a clinical prediction model with a time-to-event outcome. *Stat Med* 2022;41(7):1280-1295. doi:10.1002/sim.9275. PMID 34915593.
  - A simulation-based framework that targets the CI width for the calibration slope, with discrimination and net benefit also checked.
  - The linear-predictor distribution can be approximated from the C-index or D. Stata and R code are provided.
- Riley RD, Snell KIE, Archer L, Ensor J, Debray TPA, van Calster B, van Smeden M, Collins GS. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. *BMJ* 2024;384:e074821. doi:10.1136/bmj-2023-074821. PMCID PMC11778934. (Full text read.)
  - Rules of thumb: "at least 100 events and 100 non-events" for the c statistic and calibration slope, and "a minimum of 200 events and 200 non-events to derive calibration plots", but these "are problematic, because they are not specific to either the model or the clinical setting".
  - The **target CI width for the c statistic is 0.1 (that is, ±0.05)**, using Newcombe's SE formula.
  - Worked binary example (C = 0.77, prevalence 0.43): **347 participants (149 events) for the c statistic**, against 949 participants (408 events) for the calibration slope. "This sample size is much larger than the rule of thumb of 100 (or 200) events."
  - For time-to-event outcomes they recommend simulation, because closed forms are hard to derive.
- Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Stat Med* 2021;40(19):4230-4251. doi:10.1002/sim.9025. (Metadata verified.)

**Standard error of C: theory.**
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 1982;143(1):29-36. doi:10.1148/radiology.143.1.7063747. Gives a closed-form SE(AUC) from the numbers of events and non-events.
- Newcombe RG. Confidence intervals for an effect size measure based on the Mann-Whitney statistic. Part 2: asymptotic methods and evaluation. *Stat Med* 2006;25(4):559-573. doi:10.1002/sim.2324. This is the SE formula used in Riley's pmvalsampsize for the c statistic.
- Pencina MJ, D'Agostino RB. Overall C as a measure of discrimination in survival analysis: model specific population value and confidence interval estimation. *Stat Med* 2004;23(13):2109-2123. doi:10.1002/sim.1802. Gives an asymptotic CI for Harrell's C, based on its link to a modified Kendall's tau.
- Kang L, Chen W, Petrick NA, Gallas BD. Comparing two correlated C indices with right-censored survival outcome: a one-shot nonparametric approach. *Stat Med* 2015;34(4):685-703. doi:10.1002/sim.6370. PMCID PMC4314453. Gives the variance of the difference between two C-indices on the same patients, which is smaller than for independent groups because of the covariance.
- Jinks RC, Royston P, Parmar MKB. Discrimination-based sample size calculations for multivariable prognostic models for time-to-event data. *BMC Med Res Methodol* 2015;15:82. doi:10.1186/s12874-015-0078-y. PMCID PMC4603804.
  - They assume Var(D) = λ/e, so the **SE falls as 1/√events**, with λ_m = 2.66 + 1.26·D^1.9 − 1.65·(D·cens)^1.3.
  - Events needed for a 95% CI half-width w are e = λ(w/1.96)⁻². Worked example: 444 events for ±0.2 on D when D = 1.3.

**C-index precision by number of events: my own Monte Carlo.** This is not a published result; label it as the authors' own computation if used.
- Setup: Cox/exponential data, linear predictor ~ N(0, σ), independent exponential censoring, Harrell's C evaluated with the true linear predictor (a fixed model, as in validation). 400 replicates per cell.
- Script: `C:\Users\sorkw\AppData\Local\Temp\claude\fac5aeb26ec781c0\67fa56ad-d900-43a5-8b9b-d66628bbc1eb\scratchpad\lit\cindex_sim.py`. Output: `...\scratchpad\lit\cindex_sim_out.txt`.

| events | SD(C), C≈0.65, 40% events | ±1.96·SD | SD(C), C≈0.70, 40% events | ±1.96·SD | ±1.96·SD, C≈0.65, 15% events |
|---|---|---|---|---|---|
| 100 | 0.032 | **±0.063** | 0.028 | ±0.055 | ±0.064 |
| 139-140 | 0.026 | **±0.050** | 0.025 | ±0.048 | ±0.051 |
| 163 | 0.024 | **±0.047** | 0.023 | ±0.045 | ±0.044 |
| 200 | 0.022 | ±0.043 | 0.022 | ±0.042 | ±0.039 |
| 500 | 0.014 | ±0.027 | 0.012 | ±0.024 | ±0.028 |
| 1000 | 0.010 | ±0.019 | 0.009 | ±0.017 | ±0.020 |

- In this simulation, SD(C) ≈ 0.29 to 0.31/√E, so the CI half-width ≈ 0.6/√E.
- The Hanley-McNeil binary formula gives similar or slightly wider values: ±0.072, ±0.056, ±0.051, ±0.032 and ±0.023 at 100, 163, 200, 500 and 1,000 events (C = 0.65, event fraction 0.43).
- If C is estimated only on a held-out half of the cohort, the events per evaluation are halved and the half-width grows by about √2. For example, the evaluation halves in `docs/SHIRIN_CLUSTER_RESULTS.md` (151 BLCA and 424 BRCA patients) hold roughly 55 to 65 events, which gives about ±0.075 to ±0.08.

---

## 4. Subgroup differences and interactions

- Brookes ST, Whitley E, Peters TJ, Mulheran PA, Egger M, Davey Smith G. Subgroup analyses in randomised controlled trials: quantifying the risks of false-positives and false-negatives. *Health Technol Assess* 2001;5(33):1-56. doi:10.3310/hta5330. PMID 11701102.
  - Simulations covered continuous, binary and survival outcomes.
  - "For an interaction of the same magnitude as the overall effect, the inflation factor was 4, and this increased dramatically to … ≥100 for more subtle interactions of < 20% of the overall effect"; "the commonly used 'rule of four' may not always be sufficient".
- Brookes ST, Whitely E, Egger M, Smith GD, Mulheran PA, Peters TJ. Subgroup analyses in randomized trials: risks of subgroup-specific analyses; power and sample size for the interaction test. *J Clin Epidemiol* 2004;57(3):229-236. doi:10.1016/j.jclinepi.2003.08.009. PMID 15066682.
  - "a trial with 80% power for the overall effect had only 29% power to detect an interaction effect of the same magnitude … sample sizes should be inflated fourfold".
  - Subgroup-specific tests found a spurious "effect in one subgroup only" in 7% to 64% of simulations.
- Gelman A. "You need 16 times the sample size to estimate an interaction than to estimate a main effect." *Statistical Modeling, Causal Inference, and Social Science* blog, 15 March 2018. https://statmodeling.stat.columbia.edu/2018/03/15/need16/
  - Follow-up post, 9 November 2023: https://statmodeling.stat.columbia.edu/2023/11/09/you-need-16-times-the-sample-size-to-estimate-an-interaction-than-to-estimate-a-main-effect-explained/ . The same argument appears in Gelman A, Hill J, Vehtari A. *Regression and Other Stories*. Cambridge University Press, 2020, §16.4. doi:10.1017/9781139161879.
  - Logic: in a balanced design the SE of an interaction is twice the SE of the main effect. An interaction the same size as the main effect therefore needs 4 times the sample, and one half that size needs 16 times.
  - Gelman notes the factor depends on how "half the size" is defined; one definition gives 4 times. (The blog page returned 403 to my fetcher; the content was confirmed from search-index summaries of both posts.)
- Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. *Biometrics* 1983;39(2):499-503. doi:10.2307/2531021. PMID 6354290. Required events: E = (z₁₋α/₂ + z₁₋β)² / [q(1 − q)(ln HR)²], where q is the prevalence of the binary covariate.

**Applying this to Cox interactions (my arithmetic with Schoenfeld; 80% power, two-sided α = 0.05, two equal subgroups and a 50/50 binary factor):**
- The interaction variance is 4 times the main-effect variance, so E_int = 16 × 7.85 / (ln R)², where R is the ratio of hazard ratios.
- R = 2 needs about **262 events**; R = 1.5 needs about **764 events**. (A main effect of HR 2 needs 65 events.)
- With 151 events (the midpoint of 139 to 163), the smallest detectable R is about 2.5.
- Unequal subgroups cost more. A 10% versus 90% split needs 0.25/0.09 ≈ 2.8 times more events.

**Applying this to a difference in C-index between two independent, equal subgroups (from my simulation in §3):**
- SD(C₁ − C₂) ≈ 0.63/√E, which is about twice the SD of the pooled C. So 4 times the events are needed for the same precision, matching the interaction logic.
- 80% power: ΔC = 0.10 needs about **310 events**; ΔC = 0.05 needs about **1,250 events**.
- 95% CI half-width of ±0.05 on the difference needs about **610 events**.
- With 139 to 163 events, the minimum detectable ΔC is about 0.14 to 0.15.
- Comparing two models on the same patients is more efficient (Kang 2015), because of the positive covariance.

---

## 5. Counting "parameters" for deep models and high-dimensional inputs

- **Standard definition.** p counts every candidate parameter examined using the outcome: dummy codes, nonlinear terms and interactions included (Riley 2019; RMS §4.4).
- **Outcome-blind data reduction does not "spend" degrees of freedom.** The RMS notes ask "why does it not create phantom degrees of freedom", and Riley 2019 suggests outcome-blinded principal component analysis to reduce p.
  - *Inference for whole-slide images (mine, not a published claim):* frozen foundation-model tile embeddings pretrained without the survival outcome are outcome-blind feature extraction. Every weight trained on the survival loss counts: attention pooling, projection layers and the risk head. A gated-attention multiple-instance-learning head on 1,024-dimensional embeddings has on the order of 10⁵ to 10⁶ weights, so nominal EPV is far below 1.
- **Effective degrees of freedom** for regularised or adaptive models are smaller than the nominal count but must be estimated:
  - Ye J. On measuring and correcting the effects of data mining and model selection. *J Am Stat Assoc* 1998;93(441):120-131. doi:10.1080/01621459.1998.10474094. Introduces "generalized degrees of freedom": the sensitivity of the fitted values to perturbing each y, estimated by refitting. (Metadata from the publisher listing and citing papers; Crossref was rate-limited, so the issue number was not checked.)
  - The same idea underlies the van Houwelingen and le Cessie heuristic shrinkage factor S = (LR − p)/LR: van Houwelingen JC, le Cessie S. Predictive value of statistical models. *Stat Med* 1990;9(11):1303-1325. doi:10.1002/sim.4780091109 (Crossref verified). This is what Riley's criterion (i) is built on.
- **Penalisation does not rescue small samples:** Riley RD, Snell KIE, Martin GP, Whittle R, Archer L, Sperrin M, Collins GS. Penalization and shrinkage methods produced unreliable clinical prediction models especially when sample size was small. *J Clin Epidemiol* 2021;132:88-96. doi:10.1016/j.jclinepi.2020.12.005. PMCID PMC8026952. "They are more unreliable when needed most".
- **Machine-learning models are "data hungry":** van der Ploeg T, Austin PC, Steyerberg EW. Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints. *BMC Med Res Methodol* 2014;14:137. doi:10.1186/1471-2288-14-137. PMCID PMC4289553.
  - Logistic regression reached a stable AUC at about 20 to 50 EPV.
  - "RF, SVM and NN models showed instability and a high optimism even with >200 events per variable"; such models "may need over 10 times as many events per variable".
- **Survival-specific machine-learning evidence:** Infante G, Miceli R, Ambrogi F. Sample size and predictive performance of machine learning methods with survival data: a simulation study. *Stat Med* 2023;42(30):5657-5675. doi:10.1002/sim.9931. PMID 37947168. (Full text read from the University of Milan repository.)
  - "random forests requires at least 150% the minimum sample size suggested for traditional regression models while neural networks requires about 200% the minimum sample size"; "a double or triple sample size to obtain the same performance level of standard regression".
  - Riley's guidance "is not an option for ML techniques".
- **Reviews of practice:**
  - Dhiman P, Ma J, Andaur Navarro CL, et al. (incl. Riley RD, Van Calster B, Moons KGM, Collins GS). Methodological conduct of prognostic prediction models developed using machine learning in oncology: a systematic review. *BMC Med Res Methodol* 2022;22:101. doi:10.1186/s12874-022-01577-x. PMCID PMC8991704.
    - Median EPV: 8 for regression models, 3.4 for other machine-learning models, 1.7 for ensembles.
    - Sample size was justified in 5 of 62 studies (8%).
  - Balki I, Amirabadi A, Levman J, Martel AL, et al. Sample-size determination methodologies for machine learning in medical imaging research: a systematic review. *Can Assoc Radiol J* 2019;70(4):344-353. doi:10.1016/j.carj.2019.06.002. PMID 31522841. Only 4 of 22 studies discussed sample-size methods; the rest used pre hoc model-based or post hoc learning-curve approaches.
- **Riley/Collins on AI and stability:**
  - Riley RD, Ensor J, Snell KIE, Archer L, Whittle R, Dhiman P, Alderman J, Liu X, Kirton L, Manson-Whitton J, van Smeden M, Moons KGM, Nirantharakumar K, Cazier J-B, Denniston AK, Van Calster B, Collins GS. Importance of sample size on the quality and utility of AI-based prediction models for healthcare. *Lancet Digit Health* 2025;7(6):100857. doi:10.1016/j.landig.2025.01.013. PMID 40461350.
    - Abstract: "Most AI studies do not provide a rationale for their chosen sample sizes and frequently rely on datasets that are inadequate". It gives seven reasons why inadequate sample size harms training and evaluation.
    - (The full text returned 403 to my fetcher. I could not check its specific advice on counting parameters for deep models, so do not cite it for that.)
  - Riley RD, Collins GS. Stability of clinical prediction models developed using statistical or machine learning methods. *Biom J* 2023;65(8):e2200302. doi:10.1002/bimj.202200302. PMCID PMC10952221. Proposes bootstrap instability plots and the mean absolute prediction error at the individual level.
  - Riley RD, Collins GS, Archer L, Whittle R, Legha A, Kirton L, Dhiman P, Sadatsafavi M, Adderley NJ, Alderman J, Martin GP, Ensor J. A decomposition of Fisher's information to inform sample size for developing or updating fair and precise clinical prediction models, part 2: time-to-event outcomes. *Diagn Progn Res* 2025;9:33. doi:10.1186/s41512-025-00204-9. Addresses precision of individual-level risk and subgroup fairness; software module pmstabilityss.
  - Pavlou M, Ambler G, Qu C, Seaman SR, White IR, Omar RZ. An evaluation of sample size requirements for developing risk prediction models with binary outcomes. *BMC Med Res Methodol* 2024;24:146. doi:10.1186/s12874-024-02268-5. Finds the Riley calibration-slope formula underestimates n when C > 0.8, also for time-to-event outcomes. R package samplesizedev.
- **High-dimensional survival (omics):**
  - Pang H, Jung SH. Sample size considerations of prediction-validation methods in high-dimensional data for survival outcomes. *Genet Epidemiol* 2013;37(3):276-282. doi:10.1002/gepi.21721. PMCID PMC3763900.
  - Dobbin KK, Simon RM. Sample size planning for developing classifiers using high-dimensional DNA microarray data. *Biostatistics* 2007;8(1):101-117. doi:10.1093/biostatistics/kxj036.
  - Both are power-based or model-based rather than EPV-based.

**Takeaway for the paper:** for a deep survival model on whole-slide-image features, a Riley calculation for a simple "core" model is a defensible **lower bound**. Examples of a core model are a Cox model on the model's risk score plus clinical covariates, or a Cox model on a handful of outcome-blind summary features. The machine-learning literature suggests multiplying it by about 2 (Infante 2023) to more than 10 (van der Ploeg 2014) for flexible learners. Learning curves or simulation are the recommended way to decide the size of an extension.

---

## 6. What the theory implies for 139 to 163 events

| Criterion | What 139 to 163 events allow |
|---|---|
| EPV 10 (Peduzzi) | 14 to 16 candidate parameters |
| p < m/15 (Harrell) | 9 to 11 |
| EPV 20 (Ogundimu, low-prevalence predictors) | 7 to 8 |
| Riley criterion (i), C = 0.70 / 0.65 / 0.60 | **about 7 to 8 / 4 to 5 / about 2 parameters** |
| Riley criterion (iii) (BLCA-like, 3-year risk to ±0.05) | needs about 190 events, so 163 falls just short |
| Validation rule (Collins 2016; Vergouwe 2005) | meets the minimum of 100; below the preferred 200 |
| C-index 95% CI on the whole cohort | about ±0.045 to ±0.05, at the edge of Riley's target width of 0.1 |
| C-index on a held-out half (about 55 to 65 events) | about ±0.075 to ±0.08 |
| Subgroup ΔC detectable at 80% power (two equal halves) | about 0.14 to 0.15 |
| Ratio of hazard ratios detectable (equal subgroups) | about 2.5 |

**Reading this:** 139 to 163 events are enough to report an overall C-index with a CI of about ±0.05. They are not enough for anything that splits the events: subgroup or high-error-patient comparisons, interactions, or per-feature attributions that act like many parameters. This supports framing "~200 events" as a floor that comes from validation guidance (Collins 2016; Riley 2024). Checks that compare subgroups need roughly 4 times that.

---

## 7. Event targets for an extension (80% power, α = 0.05)

**(a) Cox model with about 25 clinical parameters (development):**
- Rules of thumb: 250 (EPV 10), 375 (EPV 15), 500 (EPV 20).
- Riley criteria: about **510 events if C ≈ 0.70, about 880 if C ≈ 0.65**. Riley's published 25-parameter survival example needs 577 events (EPP 23).
- Planning figure: **about 500 to 900 events.**

**(b) Validate a C-index to ±0.05 (95% CI width 0.1):**
- About **125 to 150 events** for a fixed model with C ≈ 0.65 to 0.70 (my simulation). This matches Riley 2024 (149 events in a binary example with C = 0.77) and the ≥100 to 200 event rule of Collins 2016.
- ±0.025 needs about **520 to 600 events**.
- Also checking calibration typically needs more, for example 408 events in Riley 2024's example.

**(c) Detect a subgroup difference:**
- Interaction the same size as the main effect: 4 times the main-effect sample (Brookes). Half the size: about 16 times (Gelman).
- Ratio of hazard ratios 2.0: about **260 events**. Ratio 1.5: about **760 events**.
- ΔC between equal halves: 0.10 needs about **310 events**; 0.05 needs about **1,250 events**.
- Unequal subgroups (10/90) need about 2.8 times more.

---

## 8. Verification log

| Reference | How verified |
|---|---|
| Peduzzi 1995; Concato 1995; Peduzzi 1996; Vittinghoff 2007; Harrell, Lee & Mark 1996; Harrell 1984 | Europe PMC/PubMed metadata and abstracts |
| van Smeden 2016; van Smeden 2019; Ogundimu 2016 | Europe PMC abstracts. Ogundimu 2016 and van Smeden 2016 full text also read |
| Riley 2019 Part II | Full text (PMC6519266) read; numbers quoted from it |
| Riley 2020 BMJ | Metadata only (bmj.com returned 403) |
| Riley 2019 Part I; Riley 2021 binary validation | Metadata |
| Riley 2022 time-to-event validation | Abstract |
| Riley 2024 BMJ part 3 | Full text read (PMC11778934) |
| Collins 2016 | Full text read (PMC4738418) |
| Vergouwe 2005 | Abstract |
| Jinks 2015 | Full text read (PMC4603804) |
| Pencina & D'Agostino 2004; Kang 2015; Hanley & McNeil 1982; Newcombe 2006; Schoenfeld 1983 | Europe PMC metadata and abstracts |
| Brookes 2001 and 2004 | Abstracts |
| Gelman blog | URLs from search; page fetch returned 403; content from search summaries |
| Regression and Other Stories | Crossref |
| van der Ploeg 2014; Dhiman 2022; Balki 2019; Riley & Collins 2023; Riley 2021 (penalisation); Pavlou 2024; Pang & Jung 2013; Dobbin & Simon 2007; Riley 2025 Lancet Digit Health | Europe PMC abstracts |
| Infante 2023 | Full text PDF from the University of Milan repository |
| Riley 2025 Diagn Progn Res | Open-access PDF from Oxford Research Archive (abstract read) |
| Ye 1998 | Publisher DOI listing; issue number not checked |
| van Houwelingen & le Cessie 1990 | Crossref |
| RMS rule p < m/15 | hbiostat.org/rmsc course notes (the online version of the book) |
| RMS 2nd ed. book metadata | Riley 2024 reference list |

**Not verified:**
- the exact "m/10 or m/20" sentence in Harrell, Lee & Mark 1996;
- any "5 to 9 EPV" numbers from Vittinghoff 2007's full text;
- the parameter-counting advice inside Riley 2025 *Lancet Digit Health*.
