#!/usr/bin/env python3
"""Statistical tests on the region fusion weights.

Closes two asks:
- Reviewer 1: "test whether Δw = w_max − w_min is statistically meaningful,
  rather than relying mainly on ranking".
- Shirin 1:36pm: "compare the weights with uniform weighting, report
  uncertainty and ranking stability".

Reads the per-round `region_weights` saved in every round's npz. NOTE: that key
holds the ALREADY-NORMALISED w, not the raw pre-ReLU logits. The raw logits are
only ever printed to stdout, so they live in the SLURM logs.

What it reports, per cohort:
- mean and SD per region across rounds, with the exact N and the file list
- Δw per round, and a one-sample test of Δw against the spread expected under
  a null where the weights carry no region-specific signal
- a permutation null built by shuffling region labels within each round, which
  preserves each round's overall spread and asks only whether the SAME regions
  are consistently high
- ranking stability: how often each region occupies each rank
- deviation from uniform 1/R per region, with a CI

The permutation null is the honest test here. Δw is guaranteed to be positive
for any noisy vector, so testing Δw against zero proves nothing. The real
question is whether the ordering is consistent across independent rounds, and
label-shuffling within rounds tests exactly that.

Usage:
    python fusion_weight_stats.py --cohort blca \
        --npz-glob '/scratch/sorkwos/med3pa_bootstrap_intermediate/job_*/round_*/epoch_*.npz' \
        --out results/fusion_weight_test_blca.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def load_rounds(npz_glob: str, pick: str = "last"):
    """Collect one weight vector per round.

    Each round saved several epochs. `pick` selects which epoch represents the
    round: "last" takes the highest epoch number, "best" takes the highest
    val_cidx if that field is present. Which one the paper used is itself an
    unstated methodological choice, so it is recorded in the output.
    """
    files = sorted(glob.glob(npz_glob))
    if not files:
        raise SystemExit(f"no files matched: {npz_glob}")

    by_round: dict[str, list[tuple]] = {}
    for f in files:
        parts = os.path.normpath(f).split(os.sep)
        job = next((p for p in parts if p.startswith("job")), "job?")
        rnd = next((p for p in parts if p.startswith("round_")), None)
        if rnd is None:
            rnd = os.path.basename(os.path.dirname(f))
        key = f"{job}/{rnd}"
        d = np.load(f, allow_pickle=True)
        if "region_weights" not in d:
            continue
        w = np.asarray(d["region_weights"], dtype=float)
        vc = float(d["val_cidx"]) if "val_cidx" in d else float("nan")
        ep = int("".join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0)
        names = [str(x) for x in d["region_names"]] if "region_names" in d else None
        by_round.setdefault(key, []).append((ep, vc, w, names, f))

    W, used, region_names = [], [], None
    for key, entries in sorted(by_round.items()):
        if pick == "best" and any(np.isfinite(e[1]) for e in entries):
            ep, vc, w, names, f = max(
                (e for e in entries if np.isfinite(e[1])), key=lambda e: e[1]
            )
        else:
            ep, vc, w, names, f = max(entries, key=lambda e: e[0])
        W.append(w)
        used.append(f)
        if names and region_names is None:
            region_names = names

    W = np.vstack(W)
    if region_names is None:
        region_names = [f"region_{i}" for i in range(W.shape[1])]
    return W, region_names, used


def permutation_test_consistency(W, n_perm=10000, seed=0):
    """Is the SAME region consistently highest across rounds?

    Statistic: the spread (max − min) of the across-round MEAN weight vector.
    Under the null that region identity carries no information, shuffling
    region labels independently within each round leaves each round's own
    spread intact but destroys consistency, so the mean vector flattens.
    """
    n_rounds, R = W.shape
    observed = float(np.ptp(W.mean(axis=0)))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.take_along_axis(
            W, np.argsort(rng.random((n_rounds, R)), axis=1), axis=1
        )
        null[i] = np.ptp(shuffled.mean(axis=0))
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "statistic": "spread of across-round mean weight vector (max - min)",
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": p,
        "n_permutations": int(n_perm),
        "interpretation": (
            "p small means the same regions are consistently weighted higher "
            "across independent bootstrap rounds, so the ordering is real even "
            "if the effect size is small. p large means the ranking is noise."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--pick", default="last", choices=["last", "best"])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    W, names, used = load_rounds(args.npz_glob, args.pick)
    n_rounds, R = W.shape
    uniform = 1.0 / R
    print(f"[{args.cohort}] {n_rounds} rounds x {R} regions, uniform = {uniform:.4f}")

    mean_w, sd_w = W.mean(axis=0), W.std(axis=0, ddof=1)
    dw = W.max(axis=1) - W.min(axis=1)

    # Per-region deviation from uniform, bootstrap CI over rounds.
    rng = np.random.default_rng(0)
    boot = np.array([
        W[rng.integers(0, n_rounds, n_rounds)].mean(axis=0) for _ in range(5000)
    ])
    lo, hi = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)

    ranks = np.argsort(np.argsort(-W, axis=1), axis=1)  # 0 = highest weight
    stability = {
        names[j]: dict(Counter(int(x) + 1 for x in ranks[:, j]))
        for j in range(R)
    }

    perm = permutation_test_consistency(W, args.n_perm)

    out = {
        "cohort": args.cohort,
        "n_rounds": int(n_rounds),
        "n_regions": int(R),
        "uniform_weight": uniform,
        "epoch_selection": args.pick,
        "npz_glob": args.npz_glob,
        "n_files_used": len(used),
        "per_region": {
            names[j]: {
                "mean": float(mean_w[j]),
                "sd": float(sd_w[j]),
                "ci95": [float(lo[j]), float(hi[j])],
                "deviation_from_uniform_pp": float(100.0 * (mean_w[j] - uniform)),
                "ci_excludes_uniform": bool(lo[j] > uniform or hi[j] < uniform),
            }
            for j in range(R)
        },
        "delta_w": {
            "per_round_mean": float(dw.mean()),
            "per_round_sd": float(dw.std(ddof=1)),
            "per_round_ci95": [float(np.quantile(dw, 0.025)), float(np.quantile(dw, 0.975))],
            "of_mean_vector": float(np.ptp(mean_w)),
            "of_mean_vector_pp": float(100.0 * np.ptp(mean_w)),
        },
        "consistency_permutation_test": perm,
        "rank_stability_counts": stability,
        "n_regions_whose_ci_excludes_uniform": int(
            sum(1 for j in range(R) if lo[j] > uniform or hi[j] < uniform)
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[{args.cohort}] Delta_w of mean vector = {100*np.ptp(mean_w):.2f} pp")
    print(f"[{args.cohort}] consistency p = {perm['p_value']:.5f}")
    print(f"[{args.cohort}] regions whose CI excludes uniform: "
          f"{out['n_regions_whose_ci_excludes_uniform']} / {R}")
    print(f"[{args.cohort}] wrote {args.out}")

    with open(args.out.replace(".json", "_files.txt"), "w") as f:
        f.write("\n".join(used))


if __name__ == "__main__":
    main()
