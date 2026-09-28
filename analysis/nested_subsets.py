#!/usr/bin/env python3
"""Compartment subsets chosen without seeing the evaluation patients.

compartment_subsets.py showed that a model kept to the top one to three
compartments of the cohort's Shapley ranking matches or beats the full model.
That ranking was computed on the same patients the subsets are evaluated on,
so the result carries selection optimism. This script removes it by nesting
the selection inside each round:

1. Split round k as everywhere else (region_shapley.oob_split).
2. Inside the in-bag patients only, hold out --inner-frac of them, train the
   full model on the rest (last epoch, no selection), and rank compartments by
   exact Shapley value on the inner hold-out.
3. On the full in-bag set, train the full model and the top-1, top-2 and top-3
   compartments of that inner ranking, and evaluate all of them on the round's
   out-of-bag patients, which steps 1-2 never saw.

Step 3 uses compartment_subsets.run_round, so its records aggregate with
compartment_subsets.aggregate against the 'full' configuration trained in the
same round. The inner ranking of every round is written to
<rounds-dir>/selection/round_XXX.json, so how stable the selection is can be
reported alongside the result.

Usage (one GPU per job):

    python analysis/nested_subsets.py --cohort blca \\
        --rounds-dir /scratch/$USER/nested_blca --round-start 1 --round-end 10
    python analysis/compartment_subsets.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/nested_blca --out results/nested_subsets_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compartment_subsets as cs  # noqa: E402
import region_shapley as rs  # noqa: E402
from compartment_common import build_dataset  # noqa: E402
from region_ablation import IPGGraphFormer, collate_bags, fast_cindex, set_seed, train_one_round  # noqa: E402


def inner_ranking(dataset, train_idx, rnd, args, device):
    uniq = np.unique(train_idx)
    rng = np.random.default_rng(10_000 + rnd)
    perm = rng.permutation(uniq)
    n_val = int(round(args.inner_frac * len(uniq)))
    inner_val, inner_train = perm[:n_val], perm[n_val:]
    R = len(dataset.region_names)
    save_root = os.path.join(args.save_root, f"inner_r{rnd}")
    os.makedirs(save_root, exist_ok=True)
    tl = DataLoader(Subset(dataset, inner_train.tolist()), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_bags)
    vl = DataLoader(Subset(dataset, inner_val.tolist()), batch_size=1, shuffle=False,
                    num_workers=1, collate_fn=collate_bags)
    model = IPGGraphFormer(num_regions=R)
    train_one_round(model, tl, device, args.epochs, save_root, rnd, lr=args.lr)
    for ep in range(1, args.epochs):                       # keep only the last epoch
        p = os.path.join(save_root, f"round_{rnd}_epoch_{ep}.pt")
        if os.path.exists(p):
            os.remove(p)
    model.eval()
    E, T, Ev, _ = rs.cache_region_embeddings(model, vl, device)
    table = rs.coalition_risks(model, E, device)
    v = np.array([fast_cindex(T, Ev, table[m]) for m in range(1 << R)])
    phi = rs.shapley_from_table(v, R)
    os.remove(os.path.join(save_root, f"round_{rnd}_epoch_{args.epochs}.pt"))
    order = [int(i) for i in np.argsort(-phi)]
    return {"round": rnd, "n_inner_train": int(len(inner_train)), "n_inner_val": int(len(inner_val)),
            "inner_val_events": int(Ev.sum()), "inner_full_cindex": float(v[-1]),
            "phi_inner": phi.tolist(), "ranking": order,
            "ranking_names": [dataset.region_names[i] for i in order]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--inner-frac", type=float, default=0.3)
    ap.add_argument("--top-k", default="1,2,3")
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None)
    args = ap.parse_args()
    ks = [int(k) for k in args.top_k.split(",")]
    base, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/nested_tmp_{args.cohort}"
    sel_dir = os.path.join(args.rounds_dir, "selection")
    os.makedirs(sel_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> nested subsets {args.cohort.upper()}: {len(base)} slides, top-k {ks}, device {device}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        sel_path = os.path.join(sel_dir, f"round_{rnd:03d}.json")
        if os.path.exists(sel_path):
            sel = json.load(open(sel_path))
        else:
            train_idx, _, _, _ = rs.oob_split(base, rnd, args.train_frac)
            sel = inner_ranking(base, train_idx, rnd, args, device)
            with open(sel_path, "w") as f:
                json.dump(sel, f)
            print(f"  round {rnd} inner ranking: {[n[:18] for n in sel['ranking_names']]}")
        plan = [("full", list(range(len(names))))] + [(f"nested_top{k}", sel["ranking"][:k]) for k in ks]
        for cfg, keep_idx in plan:
            if os.path.exists(os.path.join(args.rounds_dir, cs.slug(cfg), f"round_{rnd:03d}.json")):
                continue
            cs.run_round(base, cfg, keep_idx, rnd, args, device)


if __name__ == "__main__":
    main()
