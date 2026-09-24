#!/usr/bin/env python3
"""Retrain the model on subsets of tissue compartments.

Deletion and Shapley values remove a compartment from a model that was trained
with it. The stricter test retrains without it (Hooker et al., 2019), and the
practical question behind it is how many compartments a WSI survival model
needs at all. Each configuration here trains the same architecture, with the
same optimiser, schedule and out-of-bag split as region_shapley.py, on a chosen
set of compartments:

    full                every compartment
    drop=<regions>      every compartment except these
    keep=<regions>      only these

<regions> is a '+'-separated list of indices or unique name fragments, e.g.
keep=Invasive+Necrosis+adipose. Round k of every configuration uses the same
patients out of bag, so each configuration is compared with the full model
round by round.

Out-of-bag concordance is recorded two ways: at the epoch chosen by out-of-bag
concordance, as in the rest of the paper, and at the last epoch, which involves
no selection and so is the unbiased comparison.

One file per completed round under <rounds-dir>/<config-slug>/, so parallel
jobs can share a rounds directory and a rerun skips finished rounds. A final
--aggregate call writes the paired comparison against the full model.

Usage (Trillium, one GPU per job):

    python analysis/compartment_subsets.py --cohort blca \\
        --config full --config "keep=Invasive+adipose" \\
        --rounds-dir /scratch/$USER/subsets_blca --round-start 1 --round-end 10
    python analysis/compartment_subsets.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/subsets_blca --out results/compartment_subsets_blca.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compartment_common import RegionView, build_dataset, resolve_regions, slug  # noqa: E402
from region_ablation import IPGGraphFormer, collate_bags, oob_cindex, set_seed, train_one_round  # noqa: E402
from region_shapley import oob_split, summarise  # noqa: E402


def parse_config(cfg, names):
    R = len(names)
    if cfg == "full":
        return list(range(R))
    kind, _, spec = cfg.partition("=")
    idx = resolve_regions(spec, names)
    if kind == "keep":
        return idx
    if kind == "drop":
        return [i for i in range(R) if i not in idx]
    raise SystemExit(f"config must be full, keep=... or drop=...: {cfg!r}")


def run_round(base, cfg, keep_idx, rnd, args, device):
    view = RegionView(base, keep_idx)
    out_dir = os.path.join(args.rounds_dir, slug(cfg))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    train_idx, oob_idx, seed_used, oob_events = oob_split(view, rnd, args.train_frac)
    print(f"\n=== {cfg}  round {rnd}  ({len(keep_idx)} regions, seed {seed_used}, "
          f"OOB n={len(oob_idx)}, events={oob_events}) ===")
    train_loader = DataLoader(Subset(view, train_idx.tolist()), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_bags)
    oob_loader = DataLoader(Subset(view, oob_idx.tolist()), batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=collate_bags)

    save_root = os.path.join(args.save_root, f"{slug(cfg)}_r{rnd}")
    os.makedirs(save_root, exist_ok=True)
    R = len(keep_idx)
    model = IPGGraphFormer(num_regions=R)
    train_one_round(model, train_loader, device, args.epochs, save_root, rnd, lr=args.lr)
    model.cpu()
    del model

    by_epoch = []
    for ep in range(1, args.epochs + 1):
        ckpt = os.path.join(save_root, f"round_{rnd}_epoch_{ep}.pt")
        m = IPGGraphFormer(num_regions=R)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.to(device)
        by_epoch.append(float(oob_cindex(m, oob_loader, device)))
        del m
        os.remove(ckpt)
    best_ep = int(np.argmax(by_epoch)) + 1
    rec = {
        "config": cfg, "kept_regions": view.region_names, "n_regions": R,
        "round": rnd, "seed_used": int(seed_used), "n_oob": int(len(oob_idx)),
        "oob_events": int(oob_events), "epochs": args.epochs,
        "best_epoch": best_ep, "cindex_best_epoch": by_epoch[best_ep - 1],
        "cindex_last_epoch": by_epoch[-1], "cindex_by_epoch": by_epoch,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"round_{rnd:03d}.json"), "w") as f:
        json.dump(rec, f)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"  best epoch {best_ep}: {rec['cindex_best_epoch']:.4f}; "
          f"last epoch: {rec['cindex_last_epoch']:.4f}; {rec['seconds']:.0f}s")
    return rec


def aggregate(rounds_dir, out_path, cohort):
    configs = {}
    for d in sorted(glob.glob(os.path.join(rounds_dir, "*"))):
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "round_*.json")))]
        if recs:
            configs[recs[0]["config"]] = {r["round"]: r for r in recs}
    if "full" not in configs:
        raise SystemExit("no 'full' configuration to compare against")
    full = configs["full"]
    out = {"cohort": cohort, "reference": "full", "configs": {},
           "reading": ("delta is configuration minus full model on the same out-of-bag "
                       "patients, round by round; negative means the configuration is "
                       "worse. 'last_epoch' involves no epoch selection and is the "
                       "unbiased comparison. Keep-sets chosen from this cohort's own "
                       "Shapley ranking carry some selection optimism; pre-specified "
                       "sets and the bottom-ranked control do not.")}
    for cfg, rr in configs.items():
        common = sorted(set(rr) & set(full))
        entry = {"kept_regions": next(iter(rr.values()))["kept_regions"],
                 "n_regions": next(iter(rr.values()))["n_regions"],
                 "n_rounds": len(rr), "n_paired_rounds": len(common)}
        for which in ("best_epoch", "last_epoch"):
            key = f"cindex_{which}"
            entry[which] = {"cindex": summarise([r[key] for r in rr.values()])}
            if cfg != "full" and len(common) >= 2:
                d = [rr[k][key] - full[k][key] for k in common]
                entry[which]["delta_vs_full"] = summarise(d)
                entry[which]["frac_rounds_at_least_full"] = float(np.mean(np.array(d) >= 0))
        out["configs"][cfg] = entry
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[{cohort}] {'configuration':44} {'n':>3} {'c-index(last)':>14} {'delta vs full [95% CI]':>28}")
    for cfg, e in sorted(out["configs"].items(), key=lambda kv: -kv[1]["last_epoch"]["cindex"]["mean"]):
        c = e["last_epoch"]["cindex"]["mean"]
        dd = e["last_epoch"].get("delta_vs_full")
        ds = (f"{dd['mean']*100:+6.2f} [{dd['ci95'][0]*100:+.2f}, {dd['ci95'][1]*100:+.2f}] pp"
              if dd else "")
        print(f"[{cohort}] {cfg[:44]:44} {e['n_regions']:>3} {c:14.4f} {ds:>28}")
    print(f"[{cohort}] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--config", action="append", default=None,
                    help="full, keep=<regions> or drop=<regions>; repeatable")
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.aggregate:
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        aggregate(args.rounds_dir, args.out, args.cohort)
        return

    base, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/subsets_tmp_{args.cohort}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = args.config or ["full"]
    plan = [(c, parse_config(c, names)) for c in configs]
    print(f">>> {args.cohort.upper()}: {len(base)} slides, device {device}")
    for c, k in plan:
        print(f"    {c}: {[names[i] for i in k]}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        for cfg, keep_idx in plan:
            if os.path.exists(os.path.join(args.rounds_dir, slug(cfg), f"round_{rnd:03d}.json")):
                print(f"{cfg} round {rnd}: on disk, skipping")
                continue
            run_round(base, cfg, keep_idx, rnd, args, device)


if __name__ == "__main__":
    main()
