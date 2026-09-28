#!/usr/bin/env python3
"""Exact compartment Shapley values with whole source sites held out.

The model's risk score varies with TCGA source site more sharply than outcomes
do, so a compartment could look important because it carries a site-specific
technical signature. This reruns region_shapley.py with the bootstrap split
replaced by a site-held-out one: in round k, TCGA source sites are shuffled with
seed k and whole sites are held out until the held-out patients reach
--holdout-frac of the cohort with at least 20 events; the model is trained on
every patient of the remaining sites and evaluated on the held-out sites only.
A compartment whose Shapley value survives evaluation on unseen sites is not
explained by site-specific signal learned during training.

Everything else, model, schedule, removal, exact Shapley values, checks and
aggregate, is region_shapley.run_round. The held-out sites of every round are
recorded next to its round file. Rounds of this script are not paired with
those of region_shapley.py; compare the two at the level of the aggregate.

Usage (one GPU per job):

    python analysis/region_shapley_sites.py --cohort blca \\
        --rounds-dir /scratch/$USER/shapley_sites_blca --round-start 1 --round-end 20
    python analysis/region_shapley_sites.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/shapley_sites_blca --out results/region_shapley_sites_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_shapley as rs  # noqa: E402
from compartment_common import build_dataset, default_class_dirs  # noqa: E402
from region_ablation import set_seed  # noqa: E402

HOLDOUT_FRAC = 0.2
MIN_EVENTS = 20
LAST_SPLIT = {}


def site_of(sample):
    return rs.case_id(sample)[5:7]


def site_split(dataset, rnd, train_frac=None):
    """Drop-in for region_shapley.oob_split: whole sites held out."""
    sites = np.array([site_of(s) for s in dataset.samples])
    events = np.array([int(l[1]) for l in dataset.labels])
    uniq = sorted(set(sites))
    order = np.random.default_rng(rnd).permutation(len(uniq))
    held, n_held, ev_held = [], 0, 0
    for j in order:
        if n_held >= HOLDOUT_FRAC * len(sites) and ev_held >= MIN_EVENTS:
            break
        s = str(uniq[j])
        held.append(s)
        m = sites == s
        n_held += int(m.sum()); ev_held += int(events[m].sum())
    mask = np.isin(sites, held)
    oob_idx = np.where(mask)[0]
    train_idx = np.where(~mask)[0]
    # training draws follow the round seed, as in the bootstrap protocol
    np.random.seed(rnd); torch.manual_seed(rnd)
    LAST_SPLIT.clear()
    LAST_SPLIT.update({"round": rnd, "held_out_sites": held, "n_held_out": int(len(oob_idx)),
                       "n_train": int(len(train_idx)), "held_out_events": int(ev_held)})
    return train_idx, oob_idx, rnd, int(ev_held)


def main():
    global HOLDOUT_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    HOLDOUT_FRAC = args.holdout_frac
    args.train_frac = None
    os.makedirs(args.rounds_dir, exist_ok=True)

    if args.aggregate:
        names = [os.path.basename(d.rstrip("/")) for d in (args.class_dir or default_class_dirs(args.cohort))]
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        rs.aggregate(args.rounds_dir, names, args.out, args.cohort)
        with open(args.out) as f:
            out = json.load(f)
        splits = [json.load(open(os.path.join(args.rounds_dir, p)))
                  for p in sorted(os.listdir(args.rounds_dir)) if p.startswith("split_")]
        out["split"] = "whole TCGA source sites held out per round"
        out["held_out_per_round"] = splits
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return

    dataset, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/shapley_sites_tmp_{args.cohort}"
    rs.oob_split = site_split           # run_round looks this up at call time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> site-held-out Shapley {args.cohort.upper()}: {len(dataset)} slides, "
          f"{len(set(site_of(s) for s in dataset.samples))} sites, device {device}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        if os.path.exists(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json")):
            print(f"round {rnd}: on disk, skipping")
            continue
        rs.run_round(dataset, len(names), rnd, args, device)
        with open(os.path.join(args.rounds_dir, f"split_{rnd:03d}.json"), "w") as f:
            json.dump(LAST_SPLIT, f)
        print(f"  held-out sites {LAST_SPLIT['held_out_sites']} "
              f"({LAST_SPLIT['n_held_out']} patients, {LAST_SPLIT['held_out_events']} events)")


if __name__ == "__main__":
    main()
