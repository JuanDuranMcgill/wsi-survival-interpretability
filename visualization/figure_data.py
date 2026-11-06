"""Values plotted in the IER audit figures, read from results/.

Every function returns numbers exactly as the paper reports them (before
rounding); tests/test_figure_data.py pins each one to the printed value.
"""
import csv
import glob
import json
import os

import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Short display names, matched by keyword against the full CONCH class names.
SHORT = [
    ("dcis", "DCIS"), ("necrosis", "Necrosis"), ("adipose", "Adipose"), ("perivesical", "Adipose"),
    ("muscularis", "Muscularis"), ("muscle", "Muscle"), ("lymphocytes", "TILs"),
    ("inflammatory", "Immune"), ("lamina", "Lamina propria"), ("normal urothelium", "Normal urothelium"),
    ("normal breast", "Normal glands"), ("fibrous", "Stroma"), ("vessels", "Vessels"),
    ("invasive", "Tumour"),
]


def short_name(full):
    low = full.lower()
    for key, name in SHORT:
        if key in low:
            return name
    return full


def half_up(x, nd=0):
    """Round half away from zero, as the paper prints numbers (3.25 -> 3.3)."""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def pct_label(x):
    """Percent of rounds, printed as the paper does (Python round: 32.5 -> 32)."""
    return f"{round(x)}%"


def rank_label(x):
    """Mean rank, printed as the paper does (half-up: 3.25 -> 3.3)."""
    return f"{half_up(x, 1):.1f}"


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def interp_summary(cohort):
    d = _load(f"shapley_{cohort}/region_shapley_{cohort}.json")
    agree = d["agreement_across_regions"]
    per = d["per_region"]
    return {
        "rho_weight_use": agree["fusion_weight_vs_local_share"]["spearman_rho"],
        "rho_use_outcome": agree["local_share_vs_shapley"]["spearman_rho"],
        "rho_weight_outcome": agree["fusion_weight_vs_shapley"]["spearman_rho"],
        "use_share": {short_name(r): 100 * v["local_attribution_share"]["mean"] for r, v in per.items()},
        "shapley_pp": {short_name(r): 100 * v["shapley_cindex"]["mean"] for r, v in per.items()},
        "weight_pct": {short_name(r): 100 * v["fusion_weight"]["mean"] for r, v in per.items()},
    }


PLANTINGS = ["blca_lamina", "blca_necrosis", "brca_fibrous", "brca_necrosis"]


def _pool(files, model, key):
    """Pool per-round ranks over files: (% of rounds ranked first, mean rank)."""
    ranks = []
    for f in files:
        ranks += _load(f)["models"][model]["planted_rank"][key]["per_round"]
    return 100 * sum(r == 1 for r in ranks) / len(ranks), sum(ranks) / len(ranks)


def planted_hit_rates():
    both = [f"planted_{p}.json" for p in PLANTINGS] + [f"planted_attr_{p}.json" for p in PLANTINGS]
    run2 = [f"planted_attr_{p}.json" for p in PLANTINGS]
    return {
        "Shapley": _pool(both, "graph", "shapley"),
        "Deletion": _pool(both, "graph", "deletion"),
        "Weights": _pool(both, "graph", "weight_or_attention"),
        "Integrated gradients": _pool(run2, "graph", "integrated_gradients"),
        "Gradient x input": _pool(run2, "graph", "grad_x_input"),
        "ABMIL attention": _pool(both, "abmil", "weight_or_attention"),
    }


def explain_summary(cohort):
    with open(os.path.join(RESULTS, f"pathomic_vs_survival_{cohort}_cox.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    p = [float(r["p"]) for r in rows]
    q = [float(r["q"]) for r in rows]
    perm = _load(f"permutation_importance_{cohort}.json")
    adip = next(r for r in perm["region_table"] if r["region"] == "adipose")
    return {
        "n_tested": len(rows),
        "n_nominal": sum(x < 0.05 for x in p),
        "n_fdr": sum(x < 0.05 for x in q),
        "min_q": min(q),
        "region_rho": perm["agreement_region_level"]["spearman_rho"],
        "feature_rho": perm["agreement_feature_level"]["spearman_rho"],
        "adipose_share_impurity": adip["impurity_pct"],
        "adipose_share_perm": adip["perm_pct"],
    }


def _cohort_cindex(cohort):
    return _load(f"cohort_and_baseline_{cohort}.json")["wsi_cindex"]["value"]


def profiles():
    """Profiles recurring in at least 5% of MED3pa runs, across all 16 tests."""
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "reliability_*_q*", "reliability_*.json"))):
        tag = os.path.basename(os.path.dirname(f)).split("_")  # reliability, cohort, arm, qNN
        cohort, arm, thr = tag[1], ("pathomic" if tag[2] == "radiomic" else tag[2]), tag[3]
        cval = _cohort_cindex(cohort)
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for pr in d["top_profiles_eval"]:
            if pr.get("run_freq_pct", 0) < 5:
                continue
            lo, hi = pr["survival_model_cindex_ci95"]
            out.append(dict(cohort=cohort, arm=arm, thr=thr, rule=pr["readable_rule"], profile=pr["profile"],
                            n=pr["n_patients"], cindex=pr["survival_model_cindex_in_profile"], lo=lo, hi=hi,
                            cohort_value=cval, differs=bool(hi < cval or lo > cval)))
    return pd.DataFrame(out)


def auroc_by_arm():
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "reliability_*_q*", "reliability_*.json"))):
        tag = os.path.basename(os.path.dirname(f)).split("_")
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)["classifier_metrics_on_eval"]
        out.append(dict(cohort=tag[1], arm=("pathomic" if tag[2] == "radiomic" else tag[2]), thr=tag[3],
                        auroc=m["auroc"], p=m["auroc_p_vs_0.5"]))
    return pd.DataFrame(out)


def coverage_by_stage(cohort):
    """Conformal coverage by stage group under the Cox censoring model (the paper's primary)."""
    d = _load(f"conformal_survival_{cohort}.json")["by_censoring_model"]["cox"]
    out = []
    for h, v in d.items():
        for stage, s in v["subgroups"]["stage"].items():
            if stage.startswith("difference"):
                continue
            lo, hi = s["coverage_cox_censoring_boot_95ci"]
            out.append(dict(horizon_days=float(v["horizon_c0_days"]), stage=stage,
                            coverage=s["coverage_ipcw_cox_censoring"], lo=lo, hi=hi, n=s["n_patients"]))
    return pd.DataFrame(out)
