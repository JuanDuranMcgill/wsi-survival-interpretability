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

The MED3pa evaluation split (discovery_eval_split, stratified on the high-error
label) is not computed: the labels come from results/patient_error_{cohort}_oob.npz,
which is not on the cluster.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patient_error import load_survival  # noqa: E402
from compartment_common import CDR_DEFAULT  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapley-root", default="/scratch/sorkwos")
    ap.add_argument("--cdr-xlsx", default=CDR_DEFAULT)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    for cohort in ["blca", "brca"]:
        ids = np.unique(np.concatenate([
            np.load(p)["patient_ids"].astype(str)
            for p in glob.glob(os.path.join(args.shapley_root, f"shapley_{cohort}", "round_*_local.npz"))]))
        times, events, mask = load_survival(args.cdr_xlsx, ids)
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
        out["med3pa_eval_split"] = ("not computed: the split is stratified on the high-error label "
                                    f"from patient_error_{cohort}_oob.npz, which is not on this cluster")
        path = os.path.join(args.out_dir, f"comparable_pairs_{cohort}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite")
        json.dump(out, open(path, "w"), indent=2)
        print(f"[{cohort}] {json.dumps(out['whole_cohort'])}\nwrote {path}")


if __name__ == "__main__":
    main()
