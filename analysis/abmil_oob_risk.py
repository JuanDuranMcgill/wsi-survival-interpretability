#!/usr/bin/env python3
"""Out-of-bag risk per patient for ABMIL, from the rounds' round_###_local.npz.

abmil_compartment_shapley.py saves, for each round, the full-model risk of every
out-of-bag patient (risk_full at the selected epoch, risk_full_last at the last).
A patient's out-of-bag risk is the mean over the rounds in which the patient is
out-of-bag, as oob_risk.py builds the main model's. Patients never out-of-bag get
no score and are left out of the output, so patient_error.py and the MED3pa
scripts see only scored patients.

    python analysis/abmil_oob_risk.py --cohort blca --rounds-dir /scratch/$USER/abmil_v3_blca \\
        --out ~/data/abmil_oob_risk_blca.npz --summary results/abmil_oob_risk_blca.json

The npz (patient-level; keep out of git) has patient_ids, mean_risk, n_oob_rounds,
pfi_time and pfi_event, so patient_error.py reads it with --risk-key mean_risk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patient_error import global_cindex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--risk-key", default="risk_full", choices=["risk_full", "risk_full_last"])
    ap.add_argument("--cohort-npz", default=None,
                    help="patient list of the cohort, to count patients never out-of-bag "
                         "(default: ~/data/patient_error_{cohort}_oob.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    for p in (args.out, args.summary):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")

    paths = sorted(glob.glob(os.path.join(args.rounds_dir, "round_*_local.npz")))
    if not paths:
        raise SystemExit(f"no round_*_local.npz in {args.rounds_dir}")
    total, count, surv = {}, {}, {}
    per_round = []
    for p in paths:
        z = np.load(p)
        ids = z["patient_ids"].astype(str)
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{p}: repeated patient ids")
        risk = np.asarray(z[args.risk_key], float)
        for pid, r, t, e in zip(ids, risk, z["times"], z["events"]):
            if pid in surv and surv[pid] != (float(t), int(e)):
                raise SystemExit(f"{pid}: survival differs between rounds")
            surv[pid] = (float(t), int(e))
            total[pid] = total.get(pid, 0.0) + float(r)
            count[pid] = count.get(pid, 0) + 1
        per_round.append({"file": os.path.basename(p), "n_oob": int(len(ids)),
                          "cindex": float(global_cindex(np.asarray(z["times"], float),
                                                        np.asarray(z["events"], float), risk))})

    pids = np.array(sorted(total))
    mean_risk = np.array([total[p] / count[p] for p in pids])
    n_rounds = np.array([count[p] for p in pids])
    times = np.array([surv[p][0] for p in pids])
    events = np.array([surv[p][1] for p in pids])
    cidx = float(global_cindex(times, events, mean_risk))

    cohort_npz = args.cohort_npz or os.path.expanduser(f"~/data/patient_error_{args.cohort}_oob.npz")
    cohort_ids = set(str(p)[:12] for p in np.load(cohort_npz, allow_pickle=True)["patient_ids"])
    never = sorted(cohort_ids - set(pids))
    outside = sorted(set(pids) - cohort_ids)

    np.savez(args.out, patient_ids=pids, mean_risk=mean_risk, n_oob_rounds=n_rounds,
             pfi_time=times, pfi_event=events)
    summary = {
        "cohort": args.cohort, "rounds_dir": os.path.abspath(args.rounds_dir), "risk_key": args.risk_key,
        "n_rounds": len(paths), "n_patients_scored": int(len(pids)), "n_events_scored": int(events.sum()),
        "n_cohort": len(cohort_ids), "n_never_oob": len(never), "n_scored_outside_cohort": len(outside),
        "oob_rounds_per_patient": {"min": int(n_rounds.min()), "median": float(np.median(n_rounds)),
                                   "max": int(n_rounds.max())},
        "oob_cindex_pooled": cidx,
        "mean_round_cindex": float(np.mean([r["cindex"] for r in per_round])),
        "per_round": per_round,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)) or ".", exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_round"}, indent=1))
    print(f"wrote {args.out} and {args.summary}")


if __name__ == "__main__":
    main()
