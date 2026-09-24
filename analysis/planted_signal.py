#!/usr/bin/env python3
"""The audit on an outcome whose driving compartment is known.

On real outcomes nobody knows which compartment a model should rely on, so the
paper can show that fusion weights and Shapley values disagree but not which is
right. Here the outcome is replaced by a semi-synthetic one driven by a single,
chosen compartment, and the audit is asked to find it.

Outcome. For each patient, z is the projection of the mean UNI-2 embedding of
the planted compartment's tiles onto a fixed direction, standardised over the
patients who have that compartment; patients without it get z = 0. True risk is
beta * z. Event times are exponential with hazard exp(beta * z); censoring times
are uniform, with the upper limit set so the censored fraction matches the real
cohort's, and times are rescaled so the median matches the real median. beta is
set so the true risk reaches a chosen concordance (the 'oracle'), and the same
random draws are used throughout, so the outcome is fixed across rounds. The
real embeddings, compartments and patients are kept: only the labels change.

Models. The same rounds as the paper's analyses are then run on this outcome:
our region-fusion model (region_shapley.run_round: fusion weights, deletion,
exact Shapley values) and, optionally, gated ABMIL
(abmil_compartment_shapley.run_round: attention share, deletion, Shapley).

Reading. A measure passes if it ranks the planted compartment first. Real
compartments share information (the redundancy analysis), so some credit
spreading to correlated compartments is expected and is reported, not hidden.

Per-patient synthetic labels are written to --rounds-dir only (they are derived
from patient-level data); the summary JSON holds aggregates.

Usage (Trillium, one GPU per job):

    python analysis/planted_signal.py --cohort blca --planted Necrosis \\
        --models graph,abmil --rounds-dir /scratch/$USER/planted_blca_necrosis \\
        --round-start 1 --round-end 10
    python analysis/planted_signal.py --cohort blca --planted Necrosis --aggregate \\
        --rounds-dir /scratch/$USER/planted_blca_necrosis \\
        --out results/planted_blca_necrosis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abmil_compartment_shapley as abmil  # noqa: E402
import region_shapley  # noqa: E402
from compartment_common import build_dataset, default_class_dirs, resolve_regions  # noqa: E402
from region_ablation import fast_cindex, load_slide, set_seed  # noqa: E402


def planted_feature(dataset, r, cache):
    """Mean tile embedding of compartment r per patient (nan row if absent)."""
    if os.path.exists(cache):
        return np.load(cache)["means"]
    name = dataset.region_names[r]
    means = []
    for s in dataset.samples:
        p = s[name]
        if p is None:
            means.append(None)
            continue
        f = load_slide(p)["features"]
        f = f.numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
        means.append(f.astype(np.float64).mean(0) if len(f) else None)
    dim = next(m.shape[0] for m in means if m is not None)
    out = np.array([m if m is not None else np.full(dim, np.nan) for m in means])
    np.savez_compressed(cache, means=out)
    return out


def planted_z(means, direction, seed):
    present = ~np.isnan(means).any(1)
    X = means[present]
    X = X - X.mean(0)
    if direction == "pc1":
        u = np.linalg.svd(X, full_matrices=False)[2][0]
    else:
        u = np.random.default_rng(seed).standard_normal(X.shape[1])
        u /= np.linalg.norm(u)
    proj = X @ u
    z = np.zeros(len(means))
    z[present] = (proj - proj.mean()) / (proj.std() or 1.0)
    return z, present


def synth_outcome(z, target_c, censor_frac, median_time, seed):
    rng = np.random.default_rng(seed + 1)
    e_t = rng.exponential(size=len(z))
    u_c = rng.uniform(size=len(z))

    def draw(beta):
        T = e_t * np.exp(-beta * z)
        lo, hi = 1e-9, T.max() * 50           # C = u_c * cmax; censored when C < T
        for _ in range(80):
            cmax = 0.5 * (lo + hi)
            frac = np.mean(u_c * cmax < T)
            lo, hi = (cmax, hi) if frac > censor_frac else (lo, cmax)
        C = u_c * cmax
        t = np.minimum(T, C)
        e = (T <= C).astype(int)
        return t, e

    lo, hi = 0.0, 6.0
    for _ in range(40):
        beta = 0.5 * (lo + hi)
        t, e = draw(beta)
        c = fast_cindex(t, e, beta * z)
        lo, hi = (beta, hi) if c < target_c else (lo, beta)
    t, e = draw(beta)
    scale = median_time / np.median(t)
    return t * scale, e, float(beta), float(fast_cindex(t, e, beta * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--planted", required=True, help="the compartment that drives the outcome")
    ap.add_argument("--direction", choices=["random", "pc1"], default="random")
    ap.add_argument("--target-cindex", type=float, default=0.75,
                    help="concordance of the true risk on the synthetic outcome")
    ap.add_argument("--censor-frac", type=float, default=None,
                    help="default: the real cohort's censored fraction")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", default="graph,abmil")
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=10)
    ap.add_argument("--graph-epochs", type=int, default=15)
    ap.add_argument("--graph-lr", type=float, default=3e-5)
    abmil.add_train_args(ap)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.rounds_dir, exist_ok=True)

    if args.aggregate:
        names = [os.path.basename(d.rstrip("/")) for d in (args.class_dir or default_class_dirs(args.cohort))]
        aggregate(args, names, models)
        return

    dataset, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    (r,) = resolve_regions(args.planted, names)
    real_t = np.array([l[0] for l in dataset.labels], float)
    real_e = np.array([l[1] for l in dataset.labels], int)
    censor_frac = args.censor_frac if args.censor_frac is not None else float(1 - real_e.mean())

    means = planted_feature(dataset, r, os.path.join(args.rounds_dir, "planted_feature.npz"))
    z, present = planted_z(means, args.direction, args.seed)
    t, e, beta, oracle = synth_outcome(z, args.target_cindex, censor_frac, float(np.median(real_t)), args.seed)
    dataset.labels = [(float(a), int(b)) for a, b in zip(t, e)]
    np.savez_compressed(os.path.join(args.rounds_dir, "synthetic_labels.npz"), t=t, e=e, z=z)
    meta = {"cohort": args.cohort, "planted_region": names[r], "planted_index": r,
            "direction": args.direction, "seed": args.seed, "beta": beta,
            "oracle_cindex": oracle, "target_cindex": args.target_cindex,
            "censored_fraction": float(1 - e.mean()), "n_events": int(e.sum()),
            "n_patients": len(e), "planted_presence_rate": float(present.mean())}
    with open(os.path.join(args.rounds_dir, "synthetic_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f">>> planted {names[r]!r}: beta {beta:.3f}, oracle c-index {oracle:.3f}, "
          f"events {e.sum()}/{len(e)}, present in {present.mean():.0%}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    user = os.environ.get("USER", "user")
    save_root = args.save_root or f"/scratch/{user}/planted_tmp_{args.cohort}_{r}"
    gargs = Namespace(train_frac=args.train_frac, batch_size=args.batch_size,
                      num_workers=args.num_workers, save_root=save_root,
                      epochs=args.graph_epochs, lr=args.graph_lr,
                      rounds_dir=os.path.join(args.rounds_dir, "graph"))
    aargs = Namespace(**{k: getattr(args, k.replace("-", "_")) for k in
                         ["epochs", "train_frac", "batch_size", "lr", "weight_decay",
                          "max_train_tiles", "in_dim", "num_workers"]},
                      rounds_dir=os.path.join(args.rounds_dir, "abmil"))
    os.makedirs(gargs.rounds_dir, exist_ok=True)
    os.makedirs(aargs.rounds_dir, exist_ok=True)
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        if "graph" in models and not os.path.exists(os.path.join(gargs.rounds_dir, f"round_{rnd:03d}.json")):
            region_shapley.run_round(dataset, len(names), rnd, gargs, device)
        if "abmil" in models and not os.path.exists(os.path.join(aargs.rounds_dir, f"round_{rnd:03d}.json")):
            abmil.run_round(dataset, rnd, aargs, device)


def rank_of(values, r, higher_is_more=True):
    v = np.asarray(values, float)
    order = np.argsort(-v if higher_is_more else v)
    return int(np.where(order == r)[0][0]) + 1


def aggregate(args, names, models):
    import glob
    meta = json.load(open(os.path.join(args.rounds_dir, "synthetic_meta.json")))
    r = meta["planted_index"]
    out = {"synthetic_outcome": meta, "models": {}}
    for model in models:
        d = os.path.join(args.rounds_dir, model)
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "round_*.json")))]
        if not recs:
            continue
        summ_path = os.path.join(args.rounds_dir, f"summary_{model}.json")
        if model == "graph":
            region_shapley.aggregate(d, names, summ_path, args.cohort)
            wkey = "fusion_weights"
        else:
            abmil.aggregate(d, names, summ_path, args.cohort)
            wkey = "attention_share"
        ranks = {"weight_or_attention": [rank_of(x[wkey], r) for x in recs],
                 "deletion": [rank_of(x["deletion_delta"], r, higher_is_more=False) for x in recs],
                 "shapley": [rank_of(x["phi_perf"], r) for x in recs]}
        phi = np.array([x["phi_perf"] for x in recs])
        tot = np.abs(phi).sum(1)
        out["models"][model] = {
            "n_rounds": len(recs),
            "baseline_cindex": region_shapley.summarise([x["baseline_cindex"] for x in recs]),
            "planted_rank": {k: {"mean": float(np.mean(v)),
                                 "frac_rounds_first": float(np.mean(np.array(v) == 1)),
                                 "per_round": v} for k, v in ranks.items()},
            "planted_shapley_share": region_shapley.summarise(phi[:, r] / np.where(tot == 0, np.nan, tot)),
            "summary_file": os.path.basename(summ_path),
        }
    out["reading"] = ("A measure passes if it ranks the planted compartment first "
                      "(rank 1 of R). 'weight_or_attention' is the fusion weight for the "
                      "region-fusion model and the attention share for ABMIL. "
                      "planted_shapley_share is the planted compartment's Shapley value as "
                      "a fraction of the summed absolute Shapley values over compartments.")
    out_path = args.out or os.path.join(args.rounds_dir, "planted_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[{args.cohort}] planted {meta['planted_region']!r} (oracle c-index {meta['oracle_cindex']:.3f})")
    for model, m in out["models"].items():
        pr = m["planted_rank"]
        print(f"[{args.cohort}] {model:6} baseline {m['baseline_cindex']['mean']:.3f} | planted ranked first by "
              f"weight/attention {pr['weight_or_attention']['frac_rounds_first']:.0%}, deletion "
              f"{pr['deletion']['frac_rounds_first']:.0%}, Shapley {pr['shapley']['frac_rounds_first']:.0%} "
              f"| Shapley share {m['planted_shapley_share']['mean']:.0%}")
    print(f"[{args.cohort}] wrote {out_path}")


if __name__ == "__main__":
    main()
