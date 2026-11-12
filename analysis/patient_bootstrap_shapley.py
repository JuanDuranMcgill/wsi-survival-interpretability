#!/usr/bin/env python
"""
Patient-level bootstrap of the compartment Shapley correlations.

The intervals in results/shapley_{cohort}/region_shapley_{cohort}.json resample
training rounds. This resamples patients instead, without retraining: every
round's round_###_local.npz (written by region_shapley.py) holds each out-of-bag
patient's risk under every coalition of compartments, so each coalition's
concordance, the exact Shapley values and the per-patient local shares can be
recomputed for any reweighting of the patients.

In each of B replicates the cohort's patient IDs are drawn with replacement.
In every round, the resampled patients who are out-of-bag in that round enter
with their multiplicity w as weights. A coalition's concordance is
    v_k(w) = w' (K * S_k) w / (w' K w),
where K[i, j] = 1 when (i, j) is a comparable pair (i has the event and j's time
is later, or equal with j censored) and S_k[i, j] is 1, 0.5 or 0 as coalition
k's risk for i is above, tied with or below j's; with w = 1 this is exactly
region_ablation.fast_cindex (risks rounded to 1e-8 for ties). Shapley values per
round come from shapley_from_table; local shares are the weighted mean of
|phi_local| / sum |phi_local| over the round's resampled patients. Both are
averaged over rounds, as the aggregate does, and the three Spearman
correlations are taken across compartments. Fusion weights are model
parameters and stay fixed.

    python analysis/patient_bootstrap_shapley.py --B 1000 \
        --rounds-dir-blca /scratch/$USER/shapley_blca --rounds-dir-brca /scratch/$USER/shapley_brca \
        --out-dir results

Writes results/patient_bootstrap_shapley_{blca,brca}.json; the BLCA minus BRCA
difference of the headline contrast (local share vs Shapley), pairing replicate
b of each cohort, goes into both files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_shapley import shapley_from_table  # noqa: E402

CORRS = {
    "fusion_weight_vs_shapley": ("fusion_weight", "shapley"),
    "local_share_vs_shapley": ("local_share", "shapley"),
    "fusion_weight_vs_local_share": ("fusion_weight", "local_share"),
}
PAPER = {"blca": {"fusion_weight_vs_shapley": 0.81, "local_share_vs_shapley": 0.88,
                  "fusion_weight_vs_local_share": 0.95},
         "brca": {"fusion_weight_vs_shapley": -0.60, "local_share_vs_shapley": -0.27,
                  "fusion_weight_vs_local_share": 0.73}}


def comparable(times, events):
    """K[i, j] = 1 when (i, j) is a comparable pair, as in fast_cindex."""
    t_i, t_j = times[:, None], times[None, :]
    e_i, e_j = events[:, None].astype(bool), events[None, :].astype(bool)
    return (e_i & ((t_j > t_i) | ((t_j == t_i) & ~e_j))).astype(np.float32)


def load_rounds(rounds_dir, results_dir):
    rounds = []
    for p in sorted(glob.glob(os.path.join(rounds_dir, "round_[0-9][0-9][0-9]_local.npz"))):
        rnd = os.path.basename(p)[6:9]
        rec = json.load(open(os.path.join(results_dir, f"round_{rnd}.json")))
        d = np.load(p)
        risks = np.round(d["coalition_risks"].astype(np.float64) / 1e-8) * 1e-8
        phi = np.abs(d["phi_local"].astype(np.float64))
        rounds.append({"round": int(rnd), "ids": d["patient_ids"].astype(str),
                       "K": comparable(d["times"], d["events"]), "risks": risks,
                       "share": phi / np.maximum(phi.sum(axis=0, keepdims=True), 1e-12),
                       "fusion_w": np.array(rec["fusion_weights"]),
                       "v_perf_saved": np.array(rec["v_perf"]),
                       "share_saved": np.array(rec["local_share_mean"])})
    return rounds


def round_stats(rd, W):
    """W: (n, B) patient weights. Returns phi (B, R) and local share means (B, R)."""
    K, risks = rd["K"], rd["risks"]
    n_coal, R = risks.shape[0], rd["share"].shape[0]
    W = W.astype(np.float32)
    den = np.einsum("ib,ib->b", W, K @ W)
    v = np.empty((n_coal, W.shape[1]))
    for k in range(n_coal):
        r = risks[k]
        S = (r[:, None] > r[None, :]).astype(np.float32) + 0.5 * (r[:, None] == r[None, :])
        v[k] = np.einsum("ib,ib->b", W, (K * S) @ W) / den
    phi = shapley_from_table(v, R).T                                   # (B, R)
    share = (rd["share"] @ W / W.sum(axis=0)).T                        # (B, R)
    return phi, share, v


def correlations(fw, phi, share):
    vals = {"fusion_weight": fw, "shapley": phi, "local_share": share}
    return {name: float(spearmanr(vals[a], vals[b]).correlation) for name, (a, b) in CORRS.items()}


def run_cohort(cohort, rounds_dir, results_dir, B, seed, chunk):
    rounds = load_rounds(rounds_dir, results_dir)
    if not rounds:
        raise SystemExit(f"no round_###_local.npz in {rounds_dir}")
    cohort_ids = np.unique(np.concatenate([rd["ids"] for rd in rounds]))
    fw = np.mean([rd["fusion_w"] for rd in rounds], axis=0)
    R = rounds[0]["share"].shape[0]
    print(f"[{cohort}] {len(rounds)} rounds, {len(cohort_ids)} patients, {R} compartments")

    # point estimate: every patient once; must match the saved rounds
    phis, shares, max_dv, max_ds = [], [], 0.0, 0.0
    for rd in rounds:
        phi, share, v = round_stats(rd, np.ones((len(rd["ids"]), 1)))
        max_dv = max(max_dv, float(np.max(np.abs(v[:, 0] - rd["v_perf_saved"]))))
        max_ds = max(max_ds, float(np.max(np.abs(share[0] - rd["share_saved"]))))
        phis.append(phi[0]); shares.append(share[0])
    point = correlations(fw, np.mean(phis, 0), np.mean(shares, 0))
    print(f"[{cohort}] check vs saved rounds: max |dv| {max_dv:.2e}, max |dshare| {max_ds:.2e}")
    print(f"[{cohort}] point estimates {point} (paper {PAPER[cohort]})")

    rng = np.random.default_rng(seed)
    pos = {pid: i for i, pid in enumerate(cohort_ids)}
    counts = np.stack([np.bincount(rng.integers(0, len(cohort_ids), len(cohort_ids)),
                                   minlength=len(cohort_ids)) for _ in range(B)], axis=1)  # (N, B)
    phi_sum, share_sum = np.zeros((B, R)), np.zeros((B, R))
    t0 = time.time()
    for rd in rounds:
        W_all = counts[[pos[p] for p in rd["ids"]]]                    # (n, B)
        for s in range(0, B, chunk):
            phi, share, _ = round_stats(rd, W_all[:, s:s + chunk])
            phi_sum[s:s + chunk] += phi; share_sum[s:s + chunk] += share
        print(f"[{cohort}] round {rd['round']} done ({time.time() - t0:.0f}s)", flush=True)
    phi_b, share_b = phi_sum / len(rounds), share_sum / len(rounds)
    reps = {name: np.array([correlations(fw, phi_b[b], share_b[b])[name] for b in range(B)])
            for name in CORRS}
    return {"cohort": cohort, "n_rounds": len(rounds), "n_patients": int(len(cohort_ids)),
            "n_compartments": R, "B": B, "seed": seed,
            "check_max_abs_diff_coalition_cindex": max_dv,
            "check_max_abs_diff_local_share": max_ds,
            "paper_values": PAPER[cohort], "point": point}, reps


def summarise(x):
    x = x[np.isfinite(x)]
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return {"median": float(med), "ci95": [float(lo), float(hi)], "n_finite": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds-dir-blca", default="/scratch/sorkwos/shapley_blca")
    ap.add_argument("--rounds-dir-brca", default="/scratch/sorkwos/shapley_brca")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--chunk", type=int, default=250)
    args = ap.parse_args()

    out, reps = {}, {}
    for i, c in enumerate(["blca", "brca"]):
        out[c], reps[c] = run_cohort(c, getattr(args, f"rounds_dir_{c}"),
                                     os.path.join(args.results_root, f"shapley_{c}"),
                                     args.B, args.seed + i, args.chunk)
        out[c]["bootstrap"] = {name: summarise(reps[c][name]) for name in CORRS}

    head = "local_share_vs_shapley"
    diff = reps["blca"][head] - reps["brca"][head]
    diff_summary = {"point": out["blca"]["point"][head] - out["brca"]["point"][head],
                    **summarise(diff),
                    "fraction_of_replicates_le_0": float(np.mean(diff[np.isfinite(diff)] <= 0))}
    for c in out:
        out[c]["blca_minus_brca_local_share_vs_shapley"] = diff_summary
        path = os.path.join(args.out_dir, f"patient_bootstrap_shapley_{c}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite")
        json.dump(out[c], open(path, "w"), indent=2)
        print(f"wrote {path}")
    print(json.dumps({c: {"point": out[c]["point"], "bootstrap": out[c]["bootstrap"]} for c in out}, indent=1))
    print("BLCA - BRCA headline:", diff_summary)


if __name__ == "__main__":
    main()
