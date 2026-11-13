#!/usr/bin/env python3
"""E and R on ABMIL next to our model: the numbers behind docs/ABMIL_ER_RESULTS.md.

Reads only summaries in results/:
  - abmil_oob_risk_{cohort}[_selected_epoch].json and patient_error_{cohort}[_abmil]_oob.json
    (out-of-bag concordance);
  - reliability_[abmil_]{cohort}_{arm}_q50/ importance files (E: compartment or feature
    ranking by surrogate importance, the leading one's share, surrogate out-of-bag
    fidelity), q50 as in the paper;
  - the 16 reliability JSONs per model (R: AUROC with Hanley-McNeil 95% intervals,
    Bonferroni over 16 tests on the permutation p, profiles recurring in >= 5% of runs and
    how many have an in-profile c-index interval excluding the cohort value).

The cohort value for ABMIL's profiles is ABMIL's own out-of-bag c-index on the risk the
MED3pa runs used (last epoch), as patient_error.py reports it.

    python analysis/abmil_er_report.py --out results/abmil_er_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "visualization"))
import figure_data as fd  # noqa: E402

COHORTS = ["blca", "brca"]
N_TESTS = 16


def load(name):
    return json.load(open(os.path.join(RES, name)))


def rdir(model, cohort, arm, q="q50"):
    return os.path.join(RES, f"reliability_{'abmil_' if model == 'abmil' else ''}{cohort}_{arm}_{q}")


def explain(model, cohort, arm):
    d = rdir(model, cohort, arm)
    if arm == "radiomic":
        t = pd.read_csv(os.path.join(d, "region_summary.csv"))
        names, imp = t["region"], t["sum_mean_importance"]
    else:
        t = pd.read_csv(os.path.join(d, "clinical_importance_survival.csv"))
        names, imp = t["feature"], t["mean_importance"]
    share = imp / imp.sum()
    order = share.sort_values(ascending=False).index
    fi = json.load(open(os.path.join(d, f"reliability_{cohort}_{arm}.json")))["feature_importance_summary"]
    return {"unit": "compartment" if arm == "radiomic" else "clinical feature",
            "ranking": [[str(names[i]), float(share[i])] for i in order],
            "leading": str(names[order[0]]), "leading_share": float(share[order[0]]),
            "surrogate_oob_corr_mean": fi["mean_oob_corr"], "surrogate_oob_corr_std": fi["std_oob_corr"],
            "n_bootstrap": fi["n_bootstrap"]}


def rank_agreement(a, b):
    sa, sb = dict(a["ranking"]), dict(b["ranking"])
    common = sorted(set(sa) & set(sb))
    rho, p = stats.spearmanr([sa[k] for k in common], [sb[k] for k in common])
    return {"n_common": len(common), "spearman_rho": float(rho), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "abmil_er_comparison.json"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    out = {"oob_cindex": {}, "explain": {}, "reliability": {}}
    abmil_cidx = {}
    for c in COHORTS:
        last, sel = load(f"abmil_oob_risk_{c}.json"), load(f"abmil_oob_risk_{c}_selected_epoch.json")
        pe = load(f"patient_error_{c}_abmil_oob.json")
        abmil_cidx[c] = pe["global_cindex"]
        out["oob_cindex"][c] = {
            "ours": load(f"patient_error_{c}_oob.json")["global_cindex"],
            "abmil_last_epoch": last["oob_cindex_pooled"],
            "abmil_selected_epoch": sel["oob_cindex_pooled"],
            "abmil_mean_round_last_epoch": last["mean_round_cindex"],
            "abmil_mean_round_selected_epoch": sel["mean_round_cindex"],
            "abmil_n_rounds": last["n_rounds"], "abmil_n_scored": last["n_patients_scored"],
            "abmil_n_never_oob": last["n_never_oob"], "abmil_n_cohort": last["n_cohort"],
            "check_last_epoch_equals_patient_error": abs(last["oob_cindex_pooled"] - pe["global_cindex"]) < 1e-12,
        }
        out["explain"][c] = {}
        for arm in ["clinical", "radiomic"]:
            ours, ab = explain("main", c, arm), explain("abmil", c, arm)
            out["explain"][c][arm] = {"ours": ours, "abmil": ab, "ranking_agreement": rank_agreement(ours, ab)}

    alpha = 0.05 / N_TESTS
    for model in ["main", "abmil"]:
        a = fd.auroc_by_arm(model)
        if len(a) != N_TESTS:
            raise SystemExit(f"{model}: {len(a)} reliability JSONs, expected {N_TESTS}")
        a["bonferroni"] = a["p"] < alpha
        p = fd.profiles(model, cohort_cindex=abmil_cidx if model == "abmil" else None)
        key = "ours" if model == "main" else "abmil"
        out["reliability"][key] = {
            "auroc": a.to_dict(orient="records"),
            "bonferroni_alpha": alpha, "n_bonferroni": int(a["bonferroni"].sum()),
            "n_hm_ci_above_0.5": int((a["hm_lo"] > 0.5).sum()),
            "n_profiles_ge5pct": int(len(p)), "n_profiles_differ": int(p["differs"].sum()) if len(p) else 0,
            "profiles_differ": p[p["differs"]].to_dict(orient="records") if len(p) else [],
            "cohort_value": {c: (abmil_cidx[c] if model == "abmil" else fd._cohort_cindex(c)) for c in COHORTS},
        }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({"oob_cindex": out["oob_cindex"],
                      **{k: {kk: v[kk] for kk in ("n_bonferroni", "n_hm_ci_above_0.5", "n_profiles_ge5pct",
                                                   "n_profiles_differ")}
                         for k, v in out["reliability"].items()}}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
