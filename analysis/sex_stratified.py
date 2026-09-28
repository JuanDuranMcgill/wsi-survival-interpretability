#!/usr/bin/env python3
"""Discrimination of the out-of-bag risk score by sex.

The AIM guide asks for sex and gender dimensions to be analysed or declared as
a limitation. TCGA-CDR codes gender as 1 = female, 0 = male in the clinical
feature tables used here. BRCA is almost entirely female, so for BRCA the
counts are reported and no male-specific concordance is estimated.

Outputs results/sex_stratified_{cohort}.json: per sex, patients, events and
concordance with a patient bootstrap 95% interval, and for BLCA the female
minus male difference on the same resamples.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_ablation import fast_cindex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--oob-npz", required=True)
    ap.add_argument("--clinical-csv", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-events", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.oob_npz, allow_pickle=True)
    df = pd.DataFrame({"pid": [str(p) for p in d["patient_ids"]], "t": d["pfi_time"],
                       "e": d["pfi_event"], "x": d["mean_risk"]}).dropna()
    X = pd.read_csv(args.clinical_csv, index_col=0)
    X.index = [str(i)[:12] for i in X.index]
    X = X[~X.index.duplicated()]
    df["sex"] = X.reindex(df.pid)["gender"].map({1: "female", 0: "male"}).values
    df = df.dropna(subset=["sex"]).reset_index(drop=True)

    rng = np.random.default_rng(20260928)
    out = {"cohort": args.cohort, "coding": "TCGA gender, 1 = female, 0 = male",
           "boot": args.boot, "groups": {}}
    idx_by = {s: np.where(df.sex.values == s)[0] for s in ["female", "male"]}
    draws = {}
    for s, idx in idx_by.items():
        g = df.iloc[idx]
        rec = {"n_patients": int(len(g)), "n_events": int(g.e.sum())}
        if rec["n_events"] >= args.min_events:
            rec["cindex"] = float(fast_cindex(g.t.values, g.e.values, g.x.values))
            bs = []
            for _ in range(args.boot):
                b = rng.choice(idx, len(idx), replace=True)
                bs.append(fast_cindex(df.t.values[b], df.e.values[b], df.x.values[b]))
            draws[s] = np.array(bs)
            rec["cindex_boot_95ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        else:
            rec["note"] = f"fewer than {args.min_events} events; concordance not estimated"
        out["groups"][s] = rec
    if len(draws) == 2:
        diff = draws["female"] - draws["male"]
        out["female_minus_male"] = {
            "estimate": out["groups"]["female"]["cindex"] - out["groups"]["male"]["cindex"],
            "boot_95ci": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
