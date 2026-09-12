#!/usr/bin/env python3
"""The actual point of GA2M (DESIGN_GA2M_HEAD.md Section 4.3): does its main
effect attribution correlate with the *exact ablation* delta, unlike the
current architecture's fusion weight (rho=-0.69 BLCA, +0.03 BRCA -- no
positive relationship)?

This uses the already-computed region_ablation_{cohort}.json (independent
ablation study, larger n_rounds) as the validity reference, and the GA2M
pilot's own main_effects as the attribution being tested. Not a full
apples-to-apples comparison (ablation used 20 original-model rounds, GA2M
pilot used 10 GA2M rounds) -- flagged in the output -- but it's the first
real look at whether GA2M's decomposition tracks what ablation says, which
is the entire motivation for building it.
"""
import glob
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr


def load_ga2m_main_effects(paths, region_names):
    """Mean |main effect| per region, across all OOB patients and rounds."""
    all_main = []
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        names = list(d["region_names"])
        assert names == region_names, f"region order mismatch: {names} vs {region_names}"
        all_main.append(d["main_effects"])  # (n_oob_this_round, R)
    stacked = np.concatenate(all_main, axis=0)  # (total_oob_across_rounds, R)
    return np.abs(stacked).mean(axis=0), np.abs(stacked).std(axis=0)


def load_ablation_importance(ablation_json_path, region_names):
    with open(ablation_json_path) as f:
        d = json.load(f)
    importance = []
    for name in region_names:
        mean_delta = d["per_region"][name]["delta_renorm"]["mean"]
        importance.append(-mean_delta)  # more negative delta = more important -> flip sign
    return np.array(importance), d["baseline_cindex"]["mean"]


def run(cohort, region_names, ga2m_glob, ablation_path):
    ga2m_mean, ga2m_std = load_ga2m_main_effects(sorted(glob.glob(ga2m_glob)), region_names)
    importance, baseline_c = load_ablation_importance(ablation_path, region_names)

    rho, p = spearmanr(ga2m_mean, importance)
    r, p_pearson = pearsonr(ga2m_mean, importance)

    print(f"\n{'='*78}\n=== {cohort.upper()}: GA2M main effect vs exact ablation importance ===\n{'='*78}")
    print(f"{'region':50s} {'|main eff|':>12} {'ablation Δ(-)':>14}")
    order = np.argsort(-ga2m_mean)
    for i in order:
        print(f"{region_names[i]:50s} {ga2m_mean[i]:12.4f} {importance[i]:14.4f}")
    print(f"\n  Spearman rho = {rho:+.3f}  (p={p:.3f})")
    print(f"  Pearson  r   = {r:+.3f}  (p={p_pearson:.3f})")
    print(f"  For comparison, the CURRENT architecture's fusion-weight vs "
          f"ablation rho was reported as -0.69 (BLCA) / +0.03 (BRCA) -- no "
          f"positive relationship at all.")
    return {
        "cohort": cohort, "region_names": region_names,
        "ga2m_mean_abs_main_effect": ga2m_mean.tolist(),
        "ga2m_std_abs_main_effect": ga2m_std.tolist(),
        "ablation_importance": importance.tolist(),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "pearson_r": float(r), "pearson_p": float(p_pearson),
        "caveat": "ablation study used 20 original-model rounds; GA2M pilot used "
                  "10 GA2M rounds -- not a strictly paired comparison, first look only",
    }


blca_regions = ["Perivesical adipose tissue (extravesical fat)",
                "Invasive urothelial carcinoma (tumor)",
                "Normal urothelium (benign mucosa)",
                "Inflammatory infiltrates (immune cells)",
                "Lamina propria (fibrovascular stroma)",
                "Blood vessels (vasculature)",
                "Muscularis propria (detrusor muscle)",
                "Necrosis"]

brca_regions = ["Adipose tissue (fat)",
                "Blood vessels (vasculature)",
                "Ductal carcinoma in situ (DCIS)",
                "Fibrous desmoplastic stroma",
                "Invasive breast carcinoma (tumor cells)",
                "Muscle tissue (smooth or skeletal muscle)",
                "Necrosis or hemorrhage",
                "Normal breast glands and lobules (TDLU)",
                "Tumor-infiltrating lymphocytes (immune infiltrates)"]

out = {}
out["blca"] = run("blca", blca_regions,
                   "/scratch/sorkwos/ga2m_pilot/blca/round_*.npz",
                   "/home/sorkwos/wsi-survival-interpretability/results/region_ablation_blca.json")
out["brca"] = run("brca", brca_regions,
                   "/scratch/sorkwos/ga2m_pilot/brca/job_*/round_1.npz",
                   "/home/sorkwos/wsi-survival-interpretability/results/region_ablation_brca.json")

with open("/scratch/sorkwos/ga2m_pilot/ga2m_attribution_vs_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n>>> wrote /scratch/sorkwos/ga2m_pilot/ga2m_attribution_vs_ablation.json")
