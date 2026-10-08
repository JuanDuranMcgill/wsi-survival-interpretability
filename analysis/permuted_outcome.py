#!/usr/bin/env python3
"""Does the fusion-weight ordering depend on the outcome at all?

The permutation test in the paper shows that the weight ordering is consistent
across bootstrap rounds, but every round starts from the same initialisation and
resamples the same patients, so consistency alone does not show that the
ordering is learned from the outcome. This control retrains the region-fusion
model, with the protocol of region_shapley.py, on outcomes shuffled across
patients (time and event kept together; a fresh shuffle per round). If the
weight ordering under shuffled outcomes matches the ordering under the real
outcome, it reflects the compartments themselves (tile counts, presence, the
encoder), not prognosis.

Each round writes the usual region_shapley round file; --aggregate summarises
them and compares the mean weight vector with the real-outcome run.

Usage (one GPU per job):

    python analysis/permuted_outcome.py --cohort blca \\
        --rounds-dir /scratch/$USER/permuted_blca --round-start 1 --round-end 10
    python analysis/permuted_outcome.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/permuted_blca --out results/permuted_outcome_blca.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_shapley as rs  # noqa: E402
from compartment_common import build_dataset, default_class_dirs  # noqa: E402
from region_ablation import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--last-epoch", action="store_true")
    ap.add_argument("--real-run", default=None, help="real-outcome region_shapley summary JSON to compare with")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(args.rounds_dir, exist_ok=True)
    names = [os.path.basename(d.rstrip("/")) for d in (args.class_dir or default_class_dirs(args.cohort))]

    if args.aggregate:
        from scipy.stats import spearmanr
        rs.aggregate(args.rounds_dir, names, args.out, args.cohort)
        out = json.load(open(args.out))
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.rounds_dir, "round_*.json")))]
        W = np.array([r["fusion_weights"] for r in recs])
        out["permuted_outcome"] = True
        out["weight_spread_pp"] = float((W.mean(0).max() - W.mean(0).min()) * 100)
        # consistency of the ordering across permuted rounds: mean pairwise Spearman
        pair = [spearmanr(W[i], W[j]).correlation for i in range(len(W)) for j in range(i + 1, len(W))]
        out["mean_pairwise_round_spearman"] = float(np.mean(pair))
        if args.real_run:
            real = json.load(open(args.real_run))
            wr = [real["per_region"][n]["fusion_weight"]["mean"] for n in names]
            res = spearmanr(W.mean(0), wr)
            out["spearman_mean_weights_vs_real_outcome"] = {"rho": float(res.correlation), "p": float(res.pvalue)}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print({k: out[k] for k in out if k.startswith(("weight_spread", "mean_pairwise", "spearman_mean"))})
        return

    dataset, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    real_labels = list(dataset.labels)
    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/permuted_tmp_{args.cohort}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> permuted-outcome control {args.cohort.upper()}: {len(dataset)} slides, device {device}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        if os.path.exists(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json")):
            print(f"round {rnd}: on disk, skipping")
            continue
        perm = np.random.default_rng(50_000 + rnd).permutation(len(real_labels))
        dataset.labels = [real_labels[i] for i in perm]
        rs.run_round(dataset, len(names), rnd, args, device)


if __name__ == "__main__":
    main()
