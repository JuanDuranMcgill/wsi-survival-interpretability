#!/usr/bin/env python
"""
Per-patient compartment attribution for the case-study candidates
(docs/HANDOFF_CASE_STUDY.md), from the main Shapley run's per-round local files.

region_shapley.py saves, per round, every out-of-bag patient's exact local Shapley
values on the risk score (phi_local, compartments x patients) in
round_###_local.npz. For each candidate this takes the rounds in which the patient
was out-of-bag and records, per compartment:

- share: |phi| / sum over compartments of |phi|, the compartment's share of the
  patient's risk score, as local_share_mean in region_shapley.py; averaged over
  rounds;
- signed_in_risk_sd: phi divided by that round's standard deviation of the full
  model's out-of-bag risk (as local_signed_mean_in_risk_sd); positive raises risk;
- cohort_share: the cohort's mean share (local_share_mean averaged over all rounds),
  for comparison.

Only the "attribution" field of results/case_study/{patient}.json is replaced.

    python analysis/case_study_attribution.py --rounds-root /scratch/$USER
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds-root", default="/scratch/sorkwos")
    ap.add_argument("--case-dir", default=os.path.join(ROOT, "results", "case_study"))
    args = ap.parse_args()

    for cohort in ["blca", "brca"]:
        regions = json.load(open(os.path.join(ROOT, "results", f"shapley_{cohort}",
                                              f"region_shapley_{cohort}.json")))["region_names"]
        rdir = os.path.join(args.rounds_root, f"shapley_{cohort}")
        rounds = []
        for p in sorted(glob.glob(os.path.join(rdir, "round_[0-9][0-9][0-9]_local.npz"))):
            d = np.load(p)
            rec = json.load(open(p.replace("_local.npz", ".json")))
            phi = d["phi_local"].astype(float)                               # (R, n)
            full = d["coalition_risks"][-1].astype(float)                   # full coalition
            absphi = np.abs(phi)
            share = absphi / np.maximum(absphi.sum(0, keepdims=True), 1e-12)
            # the saved cohort share must be reproduced from the file
            assert np.allclose(share.mean(1), rec["local_share_mean"], atol=1e-6), p
            rounds.append({"round": int(os.path.basename(p)[6:9]), "ids": d["patient_ids"].astype(str),
                           "share": share, "signed": phi / (np.std(full) or 1.0),
                           "cohort_share": np.array(rec["local_share_mean"])})
        cohort_share = np.mean([r["cohort_share"] for r in rounds], 0)

        for f in sorted(glob.glob(os.path.join(args.case_dir, "*.json"))):
            cs = json.load(open(f))
            if cs["cohort"] != cohort:
                continue
            pid = cs["patient_id"]
            sh, sg, used = [], [], []
            for r in rounds:
                hit = np.flatnonzero(r["ids"] == pid)
                if hit.size:
                    sh.append(r["share"][:, hit[0]]); sg.append(r["signed"][:, hit[0]]); used.append(r["round"])
            if not used:
                cs["attribution"] = {"available": False,
                                     "note": f"never out-of-bag in the {len(rounds)} rounds of the main Shapley run"}
            else:
                share, signed = np.mean(sh, 0), np.mean(sg, 0)
                order = np.argsort(-share)
                cs["attribution"] = {
                    "available": True,
                    "source": f"{rdir}/round_###_local.npz (main Shapley run, analysis/region_shapley.py); "
                              "the npz files are not committed",
                    "method": "share = |phi_local| / sum |phi_local| over compartments (exact local Shapley "
                              "values of the risk score), averaged over the rounds in which the patient was "
                              "out-of-bag; signed_in_risk_sd = phi_local / that round's SD of out-of-bag risk",
                    "n_rounds_total": len(rounds), "n_rounds_out_of_bag": len(used), "rounds_out_of_bag": used,
                    "share": {regions[i]: float(share[i]) for i in range(len(regions))},
                    "share_sd_across_rounds": {regions[i]: float(np.std(sh, 0)[i]) for i in range(len(regions))},
                    "signed_in_risk_sd": {regions[i]: float(signed[i]) for i in range(len(regions))},
                    "cohort_share": {regions[i]: float(cohort_share[i]) for i in range(len(regions))},
                    "top2": [regions[i] for i in order[:2]],
                }
            json.dump(cs, open(f, "w"), indent=2)
            a = cs["attribution"]
            print(f"[{cohort}] {pid}: {a.get('n_rounds_out_of_bag', 0)} rounds out-of-bag, "
                  f"top2 {a.get('top2')} "
                  f"{[round(a['share'][k], 3) for k in a.get('top2', [])]}")


if __name__ == "__main__":
    main()
