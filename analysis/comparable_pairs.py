#!/usr/bin/env python
"""
Comparable pairs per patient, by censoring status.

The reliability check's per-patient error (analysis/patient_error.py) is the
discordance rate over the comparable pairs a patient belongs to: (i, j) is
comparable when the patient with the strictly shorter time had an event, and
both members are charged. The pair count depends only on PFI time and event, not
on the risk scores, so it is computed here from the CDR table for the cohort's
patients, with the same definition as symmetric_concordance_error.

The cohort is the set of patients in the main Shapley run's out-of-bag files
(379 BLCA, 1,060 BRCA), the same patients as results/patient_error_{cohort}_oob.json;
the script checks its mean, minimum and number of unscored patients against that
file.

    python analysis/comparable_pairs.py --out-dir results

With --med3pa-split, the same counts on the MED3pa evaluation split instead
(written to results/comparable_pairs_{cohort}_med3pa_eval.json). The split is
rebuilt as the MED3pa scripts build it: the feature CSV inner-joined with
patient_error_{cohort}_oob.npz (row order of the CSV), the high-error label at the
median of the finite errors (unscored patients get 0), then discovery_eval_split
(60/40, seed 20260908) stratified on that label. The clinical and radiomic arms
order patients differently, so each has its own split; both are reported. Each
arm is checked against its saved reliability JSON (threshold, evaluation size and
prevalence) and, where the JSON lists them, by re-applying its evaluation
profiles and matching their patient, event and pair counts and mean error.

    python analysis/comparable_pairs.py --med3pa-split --out-dir results
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_metrics import discovery_eval_split, profile_report  # noqa: E402

DATA = os.path.expanduser("~/data")
# Defaults of the four MED3pa scripts (analysis/{cohort}_{arm}_linear_med3pa.py).
MED3PA_ARMS = {
    "blca": {"clinical": ("clinical_features.csv", np.float64, "all_profiles_dir.csv"),
             "radiomic": ("blca/radiomics_pre_corr.csv", np.float32, "all_profiles_dir_radiomic.csv")},
    "brca": {"clinical": ("clinical_features_BRCA.csv", np.float64, "all_profiles_dir.csv"),
             "radiomic": ("brca/radiomics_pre_corr.csv", np.float32, "all_profiles_dir_radiomic.csv")},
}


def n_comparable(times, events):
    t = np.asarray(times, float); e = np.asarray(events, float)
    earlier = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    comp = earlier | earlier.T
    np.fill_diagonal(comp, False)
    return comp.sum(axis=1)


def describe(x):
    x = np.asarray(x)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "median": float(med), "iqr": [float(q1), float(q3)],
            "min": int(x.min()), "max": int(x.max()), "mean": float(x.mean()),
            "n_zero": int((x == 0).sum()), "n_lt_10": int((x < 10).sum())}


def by_status(counts, events):
    events = np.asarray(events).astype(int)
    return {"event": describe(counts[events == 1]), "censored": describe(counts[events == 0]),
            "all": describe(counts)}


def whole_cohort(args):
    # Imported here: compartment_common pulls in torch, which the split mode does not need.
    from patient_error import load_survival
    from compartment_common import CDR_DEFAULT
    cdr_xlsx = args.cdr_xlsx or CDR_DEFAULT

    for cohort in ["blca", "brca"]:
        ids = np.unique(np.concatenate([
            np.load(p)["patient_ids"].astype(str)
            for p in glob.glob(os.path.join(args.shapley_root, f"shapley_{cohort}", "round_*_local.npz"))]))
        times, events, mask = load_survival(cdr_xlsx, ids)
        ids, times, events = ids[mask], times[mask], events[mask]
        counts = n_comparable(times, events)

        saved = json.load(open(os.path.join(args.results, f"patient_error_{cohort}_oob.json")))
        scored = counts > 0
        check = {"n_patients": [int(len(ids)), saved["n_patients"]],
                 "n_unscored_no_comparable_pairs": [int((~scored).sum()),
                                                    saved["n_unscored_no_comparable_pairs"]],
                 "mean_comparable_pairs_scored": [float(counts[scored].mean()),
                                                  saved["mean_comparable_pairs"]],
                 "min_comparable_pairs_scored": [int(counts[scored].min()),
                                                 saved["min_comparable_pairs"]],
                 "n_events": [int(events.sum()), saved["n_events"]]}
        ok = all(abs(a - b) < 1e-6 for a, b in check.values())
        print(f"[{cohort}] check against patient_error_{cohort}_oob.json "
              f"(computed, saved): {check} -> {'OK' if ok else 'MISMATCH'}")

        out = {"cohort": cohort, "endpoint": "PFI",
               "definition": "pair (i, j) comparable when the strictly shorter time had an event; "
                             "both members charged (analysis/patient_error.py)",
               "check_vs_patient_error_oob_json": {"values_computed_saved": check, "match": ok},
               "whole_cohort": by_status(counts, events)}
        out["med3pa_eval_split"] = f"see comparable_pairs_{cohort}_med3pa_eval.json (--med3pa-split)"
        path = os.path.join(args.out_dir, f"comparable_pairs_{cohort}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite")
        json.dump(out, open(path, "w"), indent=2)
        print(f"[{cohort}] {json.dumps(out['whole_cohort'])}\nwrote {path}")


def med3pa_split(cohort, arm, error_npz, results):
    """Rebuild one MED3pa arm's split; return (eval patient ids, eval mask, aligned frame, check)."""
    csv, x_dtype, profiles_csv = MED3PA_ARMS[cohort][arm]
    feats_df = pd.read_csv(os.path.join(DATA, csv), index_col="patient_id")
    feats_df.index = feats_df.index.astype(str)
    oob = np.load(error_npz, allow_pickle=True)
    oob_df = pd.DataFrame({"risk": np.asarray(oob["risk"], np.float64),
                           "error": np.asarray(oob["error"], np.float64),
                           "pfi_time": np.asarray(oob["pfi_time"], np.float64),
                           "pfi_event": np.asarray(oob["pfi_event"], np.float64)},
                          index=[str(p)[:12] for p in oob["patient_ids"]])
    merged = feats_df.join(oob_df, how="inner")
    feature_names = [c for c in merged.columns if c not in oob_df.columns]
    ids = merged.index.to_numpy().astype(str)
    err = merged["error"].to_numpy()

    finite = np.isfinite(err)
    thresh = float(np.quantile(err[finite], 0.5))
    label = np.zeros(len(err), dtype=int)
    label[finite] = (err[finite] >= thresh).astype(int)
    _, ev = discovery_eval_split(ids, frac_discovery=0.6, seed=20260908, stratify=label)

    saved = json.load(open(os.path.join(results, f"reliability_{cohort}_{arm}_q50",
                                        f"reliability_{cohort}_{arm}.json")))
    check = {"error_threshold": [thresh, saved["error_threshold"]],
             "n_eval": [int(ev.sum()), saved["discovery_eval_split"]["n_eval"]],
             "eval_prevalence_high_error": [float(label[ev].mean()),
                                            saved["classifier_metrics_on_eval"]["prevalence_high_error"]]}
    ok = all(abs(a - b) < 1e-9 for a, b in check.values())

    # Re-apply the saved evaluation profiles (as apply_profile_to_X does) and match their reports.
    profiles = saved.get("top_profiles_eval", [])
    n_match = 0
    if profiles:
        conds_by_profile = pd.read_csv(os.path.join(results, f"reliability_{cohort}_{arm}_q50", profiles_csv),
                                       index_col="profile")["_conditions_median"]
        X_eval = merged.loc[ev, feature_names].to_numpy(dtype=x_dtype)
        col = {f: i for i, f in enumerate(feature_names)}
        t, e, r = (merged[k].to_numpy()[ev] for k in ("pfi_time", "pfi_event", "risk"))
        for tp in profiles:
            mask = np.ones(int(ev.sum()), dtype=bool)
            for key, th in ast.literal_eval(conds_by_profile[tp["profile"]]).items():
                feature, direction = key.rsplit("_", 1)
                if feature in col:
                    x = X_eval[:, col[feature]]
                    mask &= (x > th) if direction == "UP" else (x <= th)
            pr = profile_report(mask, label[ev], err[ev], t, e, r)
            n_match += (all(pr[k] == tp[k] for k in ("n_patients", "n_events_in_profile",
                                                       "n_comparable_pairs_in_profile"))
                        and abs(pr["mean_patient_error_in_profile"] - tp["mean_patient_error_in_profile"]) < 1e-9)
    check = {"values_computed_saved": check, "eval_profiles_reproduced": [n_match, len(profiles)],
             "match": bool(ok and n_match == len(profiles))}
    return ids[ev], merged, ev, check


def eval_split(args):
    for cohort in ["blca", "brca"]:
        error_npz = os.path.join(DATA, f"patient_error_{cohort}_oob.npz")
        oob = np.load(error_npz, allow_pickle=True)
        oob_ids = np.array([str(p)[:12] for p in oob["patient_ids"]])
        whole = np.asarray(oob["n_comparable_pairs"])
        if not np.array_equal(whole, n_comparable(oob["pfi_time"], oob["pfi_event"])):
            raise SystemExit(f"[{cohort}] n_comparable_pairs in {error_npz} does not match n_comparable")
        whole_by_id = dict(zip(oob_ids, whole))

        out = {"cohort": cohort, "endpoint": "PFI",
               "definition": "pair (i, j) comparable when the strictly shorter time had an event; "
                             "both members charged (analysis/patient_error.py)",
               "split": "discovery_eval_split, frac_discovery 0.6, seed 20260908, stratified on the "
                        "high-error label (error >= median of finite errors; unscored patients 0)",
               "input_error_npz": error_npz,
               "arms": {}}
        eval_sets = {}
        for arm in ["clinical", "radiomic"]:
            ids, merged, ev, check = med3pa_split(cohort, arm, error_npz, args.results)
            print(f"[{cohort} {arm}] split check (computed, saved): {check}")
            eval_sets[arm] = set(ids)
            events = merged["pfi_event"].to_numpy()[ev]
            within = n_comparable(merged["pfi_time"].to_numpy()[ev], events)
            out["arms"][arm] = {
                "feature_csv": os.path.join(DATA, MED3PA_ARMS[cohort][arm][0]),
                "check_vs_reliability_json": check,
                "n_eval": int(ev.sum()), "n_eval_events": int(events.sum()),
                # Pairs behind each patient's error label: the error is computed on the whole cohort.
                "pairs_in_whole_cohort": by_status(np.array([whole_by_id[p] for p in ids]), events),
                # Pairs among evaluation patients only, as the per-profile c-index counts them.
                "pairs_within_eval_split": by_status(within, events),
            }
        out["n_eval_patients_in_both_arms"] = len(eval_sets["clinical"] & eval_sets["radiomic"])

        path = os.path.join(args.out_dir, f"comparable_pairs_{cohort}_med3pa_eval.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite")
        json.dump(out, open(path, "w"), indent=2)
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapley-root", default="/scratch/sorkwos")
    ap.add_argument("--cdr-xlsx", default=None, help="default: compartment_common.CDR_DEFAULT")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--med3pa-split", action="store_true",
                    help="count pairs on the MED3pa evaluation splits instead of the whole cohort")
    args = ap.parse_args()
    eval_split(args) if args.med3pa_split else whole_cohort(args)


if __name__ == "__main__":
    main()
