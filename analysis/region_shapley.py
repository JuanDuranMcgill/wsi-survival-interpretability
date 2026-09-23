#!/usr/bin/env python3
"""Exact Shapley values and pairwise interactions over tissue compartments.

The paper compares two readings of which compartment the model relies on: the
learned fusion weight, and a single-region deletion test. Deletion has a known
limitation, which the paper states: it measures what a region adds *given all
the others*, so a region whose information is shared with another looks
unimportant even when the model uses it. The redundancy analysis showed this
does not explain the necrosis result, but it does not replace deletion with a
measure free of the limitation.

Shapley values are that measure. A region's Shapley value averages what it adds
over every possible combination of the other regions, so shared information is
split between the regions that carry it rather than credited to neither. On
images Shapley values have to be approximated, because the players are
thousands of patches. Here the players are the 8 (BLCA) or 9 (BRCA) tissue
compartments, so every one of the 2^8 = 256 or 2^9 = 512 coalitions can be
evaluated and the values are exact. That is a property of compartment-level
modelling that patch-level models do not have.

Three quantities are computed from the same 2^R table of coalition values, on
the same trained models, so they are directly comparable with each other and
with the fusion weights of those models:

- global Shapley values, with the out-of-bag concordance of the model as the
  payoff: how much concordance each compartment contributes, in the spirit of
  SAGE (Covert, Lundberg and Lee, 2020);
- per-patient Shapley values, with the patient's risk score as the payoff,
  summarised as each compartment's share of the total attribution so that it is
  on the same scale as a fusion weight, and so that the spread across patients
  shows how far one cohort-level weight is from describing individuals;
- the Shapley interaction index for every pair of compartments (Grabisch and
  Roubens, 1999), which asks whether two compartments contribute more together
  than separately. Necrosis with adipose is the pair of interest.

Removing a region means exactly what it means in region_ablation.py: its
embedding is zeroed after per-region encoding, before cross-region attention,
and again after the attention LayerNorm, with the learned fusion weights left
unnormalised. Single-region deletion therefore falls out of the same table as a
special case, and the script checks that it does.

Cost. Removal happens after each region's graph encoding, which is where the
compute is. Each out-of-bag patient's region embeddings are encoded once and
cached, and every coalition is then scored through the attention and fusion
head only, batched over patients. The GPU cost is essentially that of
region_ablation.py: training, plus one extra pass over the out-of-bag patients.

Training matches region_ablation.py exactly: same model, optimiser, schedule,
epochs and best-epoch selection, and the same seed per round, so round k here
uses the same out-of-bag patients as round k of the published ablation.

Each completed round is written to its own file, so parallel jobs over disjoint
round ranges can share one --rounds-dir, a job killed at walltime loses only
the round in progress, and a rerun skips rounds already on disk. A final
--aggregate call writes the summary.

Usage (Trillium, one GPU per job):

    python analysis/region_shapley.py --cohort blca \\
        --rounds-dir /scratch/$USER/shapley_blca --round-start 1 --round-end 5
    ...
    python analysis/region_shapley.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/shapley_blca \\
        --out /scratch/$USER/region_shapley_blca.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_ablation import (  # noqa: E402  identical training and removal
    BLCA_BASE, BLCA_REGIONS, BRCA_BASE, BRCA_REGIONS, IPGGraphFormer,
    MultiRegionDataset, collate_bags, fast_cindex, oob_cindex, set_seed,
    train_one_round)

MIN_VAL_EVENTS = 20          # as in region_ablation.py, so seeds line up
MAX_SEED_RETRIES = 20


# ── Out-of-bag split: identical to region_ablation.run_ablation ──────────────

def oob_split(dataset, rnd, train_frac):
    N = len(dataset)
    idx = np.arange(N)
    seed_attempt, retry = rnd, 0
    while True:
        np.random.seed(seed_attempt)
        random.seed(seed_attempt)
        torch.manual_seed(seed_attempt)
        train_idx = np.random.choice(idx, int(train_frac * N), replace=True)
        mask = np.ones(N, bool)
        mask[train_idx] = False
        oob_idx = idx[mask]
        if len(oob_idx) == 0:
            perm = np.random.permutation(N)
            split = int(0.8 * N)
            train_idx, oob_idx = perm[:split], perm[split:]
        val_events = sum(int(dataset.labels[i][1]) for i in oob_idx)
        if val_events >= MIN_VAL_EVENTS:
            break
        retry += 1
        if retry >= MAX_SEED_RETRIES:
            print(f"  [WARN] no seed with >={MIN_VAL_EVENTS} OOB events")
            break
        seed_attempt = rnd * 1000 + retry
    return train_idx, oob_idx, seed_attempt, int(val_events)


def case_id(sample):
    for p in sample.values():
        if p is not None:
            return os.path.basename(p)[:12]
    return "unknown"


# ── Encoding once, scoring every coalition through the head ──────────────────

@torch.no_grad()
def cache_region_embeddings(model, loader, device):
    """Per-region embeddings for every patient, plus the full-model risk."""
    model.eval()
    E, T, Ev, R_full = [], [], [], []
    for feats_list, pos_list, t, e, _, _ in loader:
        risk = model.forward_batch(feats_list, pos_list)
        for si in range(len(feats_list)):
            embs = []
            for ri in range(len(feats_list[si])):
                f, p = feats_list[si][ri], pos_list[si][ri]
                if f is None:
                    embs.append(torch.zeros(model.d_model, device=device))
                else:
                    embs.append(model.forward_bag(f, p, None, ri))
            E.append(torch.stack(embs).cpu())
        R_full.extend(risk.detach().cpu().tolist())
        T.extend(np.asarray(t).tolist())
        Ev.extend(np.asarray(e).tolist())
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.stack(E), np.array(T), np.array(Ev), np.array(R_full)


@torch.no_grad()
def head_risk(model, E, keep):
    """Risk with only the regions in `keep` present. E: (n, R, d) on device.

    Mirrors IPGGraphFormer.forward_batch_ablated with renormalize=False,
    generalised from one removed region to any set of them.
    """
    X = E.clone()
    drop = ~keep
    X[:, drop] = 0.0
    attn, _ = model.cross_region_attn(X, X, X, need_weights=False)
    X = model.cross_region_norm(X + attn)
    X[:, drop] = 0.0
    w_raw = torch.relu(model.region_weights) + 0.01
    w = w_raw / w_raw.sum()
    return model.head((w.view(1, -1, 1) * X).sum(dim=1)).squeeze(-1)


@torch.no_grad()
def coalition_risks(model, E, device, chunk=4096):
    """Risk for every coalition: array of shape (2^R, n). Bit r set = region r kept."""
    model.eval()
    n, R, _ = E.shape
    out = np.zeros((1 << R, n), dtype=np.float64)
    for mask in range(1 << R):
        keep = torch.tensor([(mask >> r) & 1 == 1 for r in range(R)], device=device)
        vals = []
        for s in range(0, n, chunk):
            vals.append(head_risk(model, E[s:s + chunk].to(device), keep).cpu())
        out[mask] = torch.cat(vals).numpy()
    return out


# ── Exact Shapley values and interaction indices from a coalition table ──────

def shapley_weights(R):
    return np.array([math.factorial(k) * math.factorial(R - k - 1) / math.factorial(R)
                     for k in range(R)])


def shapley_from_table(v, R):
    """Exact Shapley values. v: (2^R,) or (2^R, n). Returns (R,) or (R, n)."""
    w = shapley_weights(R)
    phi = np.zeros((R,) + v.shape[1:])
    for mask in range(1 << R):
        k = bin(mask).count("1")
        for r in range(R):
            if not (mask >> r) & 1:
                phi[r] += w[k] * (v[mask | (1 << r)] - v[mask])
    return phi


def interaction_from_table(v, R):
    """Shapley interaction index for every pair. Returns (R, R) or (R, R, n)."""
    I = np.zeros((R, R) + v.shape[1:])
    if R < 2:
        return I
    w = np.array([math.factorial(k) * math.factorial(R - k - 2) / math.factorial(R - 1)
                  for k in range(R - 1)])
    for r in range(R):
        for s in range(r + 1, R):
            acc = np.zeros(v.shape[1:])
            for mask in range(1 << R):
                if (mask >> r) & 1 or (mask >> s) & 1:
                    continue
                k = bin(mask).count("1")
                acc = acc + w[k] * (v[mask | (1 << r) | (1 << s)] - v[mask | (1 << r)]
                                    - v[mask | (1 << s)] + v[mask])
            I[r, s] = I[s, r] = acc
    return I


# ── One round ────────────────────────────────────────────────────────────────

def run_round(dataset, R, rnd, args, device):
    t0 = time.time()
    train_idx, oob_idx, seed_used, oob_events = oob_split(dataset, rnd, args.train_frac)
    print(f"\n=== Round {rnd}  (seed {seed_used}, OOB n={len(oob_idx)}, "
          f"events={oob_events}) ===")
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_bags)
    oob_loader = DataLoader(Subset(dataset, oob_idx.tolist()), batch_size=1,
                            shuffle=False, num_workers=1, collate_fn=collate_bags)

    os.makedirs(args.save_root, exist_ok=True)
    model = IPGGraphFormer(num_regions=R)
    train_one_round(model, train_loader, device, args.epochs, args.save_root, rnd, lr=args.lr)
    model.cpu()
    del model

    best_ep, best_c = 1, -1.0
    for ep in range(1, args.epochs + 1):
        m = IPGGraphFormer(num_regions=R)
        m.load_state_dict(torch.load(os.path.join(args.save_root,
                                                  f"round_{rnd}_epoch_{ep}.pt"),
                                     map_location="cpu"))
        m.to(device)
        c = oob_cindex(m, oob_loader, device)
        if c > best_c:
            best_c, best_ep = c, ep
        del m
    print(f"  best epoch {best_ep} (OOB c-index {best_c:.4f})")

    model = IPGGraphFormer(num_regions=R)
    model.load_state_dict(torch.load(os.path.join(args.save_root,
                                                  f"round_{rnd}_epoch_{best_ep}.pt"),
                                     map_location="cpu"))
    model.to(device).eval()

    E, times, events, risk_full = cache_region_embeddings(model, oob_loader, device)
    table = coalition_risks(model, E, device)                   # (2^R, n)
    full = (1 << R) - 1

    # the cached path must reproduce the model exactly
    head_check = float(np.max(np.abs(table[full] - risk_full)))
    if head_check > 1e-3:
        raise RuntimeError(f"cached-embedding head differs from the model by {head_check}")

    v_perf = np.array([fast_cindex(times, events, table[m]) for m in range(1 << R)])
    phi_perf = shapley_from_table(v_perf, R)
    I_perf = interaction_from_table(v_perf, R)
    eff_gap = float(abs(phi_perf.sum() - (v_perf[full] - v_perf[0])))

    phi_loc = shapley_from_table(table, R)                      # (R, n)
    I_loc = interaction_from_table(table, R)                    # (R, R, n)
    absphi = np.abs(phi_loc)
    share = absphi / np.maximum(absphi.sum(axis=0, keepdims=True), 1e-12)
    risk_sd = float(np.std(risk_full)) or 1.0

    w_raw = torch.relu(model.region_weights.detach().cpu()) + 0.01
    fusion_w = (w_raw / w_raw.sum()).numpy()

    rec = {
        "round": rnd, "seed_used": int(seed_used), "best_epoch": int(best_ep),
        "n_oob": int(len(oob_idx)), "oob_events": oob_events,
        "baseline_cindex": float(v_perf[full]),
        "empty_coalition_cindex": float(v_perf[0]),
        "head_check_max_abs_diff": head_check,
        "efficiency_gap": eff_gap,
        "fusion_weights": fusion_w.tolist(),
        "v_perf": v_perf.tolist(),
        "phi_perf": phi_perf.tolist(),
        "interaction_perf": I_perf.tolist(),
        "deletion_delta": [float(v_perf[full & ~(1 << r)] - v_perf[full]) for r in range(R)],
        "keep_only_gain": [float(v_perf[1 << r] - v_perf[0]) for r in range(R)],
        "local_share_mean": share.mean(axis=1).tolist(),
        "local_share_sd_across_patients": share.std(axis=1).tolist(),
        "local_signed_mean_in_risk_sd": (phi_loc.mean(axis=1) / risk_sd).tolist(),
        "local_interaction_mean_in_risk_sd": (I_loc.mean(axis=2) / risk_sd).tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    ids = [case_id(dataset.samples[i]) for i in oob_idx]
    np.savez_compressed(os.path.join(args.rounds_dir, f"round_{rnd:03d}_local.npz"),
                        patient_ids=np.array(ids), times=times, events=events,
                        coalition_risks=table.astype(np.float32),
                        phi_local=phi_loc.astype(np.float32))
    with open(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json"), "w") as f:
        json.dump(rec, f)

    for ep in range(1, args.epochs + 1):
        p = os.path.join(args.save_root, f"round_{rnd}_epoch_{ep}.pt")
        if os.path.exists(p):
            os.remove(p)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"  baseline {v_perf[full]:.4f}, head check {head_check:.1e}, "
          f"efficiency gap {eff_gap:.1e}, {rec['seconds']:.0f}s")
    return rec


# ── Aggregate across rounds ──────────────────────────────────────────────────

def summarise(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": m, "sd": float(a.std(ddof=1)) if n > 1 else 0.0, "se": se,
            "ci95": [m - 1.96 * se, m + 1.96 * se], "n_rounds": n}


def aggregate(rounds_dir, region_names, out_path, cohort):
    from scipy.stats import spearmanr
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(rounds_dir, "round_*.json")))]
    if not recs:
        raise SystemExit(f"no round files in {rounds_dir}")
    R = len(region_names)
    arr = lambda key: np.array([r[key] for r in recs], dtype=float)    # noqa: E731

    phi, dele, keep1 = arr("phi_perf"), arr("deletion_delta"), arr("keep_only_gain")
    fw, share, sgn = arr("fusion_weights"), arr("local_share_mean"), arr("local_signed_mean_in_risk_sd")
    Ip, Il = arr("interaction_perf"), arr("local_interaction_mean_in_risk_sd")

    per_region = {}
    for r, nm in enumerate(region_names):
        per_region[nm] = {
            "fusion_weight": summarise(fw[:, r]),
            "shapley_cindex": summarise(phi[:, r]),
            "deletion_delta": summarise(dele[:, r]),
            "keep_only_gain": summarise(keep1[:, r]),
            "local_attribution_share": summarise(share[:, r]),
            "local_signed_in_risk_sd": summarise(sgn[:, r]),
        }

    pairs = []
    for r in range(R):
        for s in range(r + 1, R):
            sp, sl = summarise(Ip[:, r, s]), summarise(Il[:, r, s])
            pairs.append({"pair": [region_names[r], region_names[s]],
                          "interaction_cindex": sp, "interaction_local_in_risk_sd": sl,
                          "ci_excludes_zero_cindex": sp["ci95"][0] > 0 or sp["ci95"][1] < 0})
    pairs.sort(key=lambda d: -abs(d["interaction_cindex"]["mean"]))

    means = lambda key: [per_region[n][key]["mean"] for n in region_names]  # noqa: E731
    def agree(a, b):
        res = spearmanr(means(a), means(b))
        return {"spearman_rho": float(res.correlation), "p": float(res.pvalue)}
    # deletion is negative when a region matters, so its cost is -delta
    cost = [-x for x in means("deletion_delta")]
    rc = spearmanr(cost, means("shapley_cindex"))

    out = {
        "cohort": cohort, "n_rounds": len(recs), "region_names": region_names,
        "removal": ("region embedding zeroed after per-region encoding, before "
                    "cross-region attention and again after its LayerNorm; fusion "
                    "weights not renormalised (as region_ablation.py, no_renorm)"),
        "payoffs": {"global": "out-of-bag concordance",
                    "local": "each patient's risk score"},
        "checks": {
            "max_head_check_abs_diff": float(arr("head_check_max_abs_diff").max()),
            "max_efficiency_gap": float(arr("efficiency_gap").max()),
            "empty_coalition_cindex": summarise(arr("empty_coalition_cindex")),
        },
        "baseline_cindex": summarise(arr("baseline_cindex")),
        "per_region": per_region,
        "pairwise_interactions_ranked": pairs,
        "agreement_across_regions": {
            "fusion_weight_vs_shapley": agree("fusion_weight", "shapley_cindex"),
            "fusion_weight_vs_local_share": agree("fusion_weight", "local_attribution_share"),
            "deletion_cost_vs_shapley": {"spearman_rho": float(rc.correlation),
                                         "p": float(rc.pvalue)},
            "local_share_vs_shapley": agree("local_attribution_share", "shapley_cindex"),
        },
        "reading": (
            "shapley_cindex is each region's exact Shapley value with out-of-bag "
            "concordance as the payoff; they sum to baseline minus the empty-model "
            "concordance of 0.5. deletion_delta is the single-region deletion from "
            "the same models, negative when the region matters. "
            "local_attribution_share is each region's share of the per-patient "
            "absolute Shapley attribution on the risk score, on the same scale as "
            "a fusion weight. A positive interaction means two regions contribute "
            "more together than the sum of their separate contributions."),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[{cohort}] {len(recs)} rounds, baseline {out['baseline_cindex']['mean']:.4f}")
    print(f"[{cohort}] checks: head {out['checks']['max_head_check_abs_diff']:.1e}, "
          f"efficiency {out['checks']['max_efficiency_gap']:.1e}, empty-model "
          f"c-index {out['checks']['empty_coalition_cindex']['mean']:.3f}")
    print(f"[{cohort}] {'region':34} {'weight':>7} {'shapley':>9} {'deletion':>9} {'loc.share':>9}")
    for nm in region_names:
        v = per_region[nm]
        print(f"[{cohort}] {nm[:34]:34} {v['fusion_weight']['mean']*100:6.2f}% "
              f"{v['shapley_cindex']['mean']*100:+8.2f}pp "
              f"{v['deletion_delta']['mean']*100:+8.2f}pp "
              f"{v['local_attribution_share']['mean']*100:8.1f}%")
    a = out["agreement_across_regions"]
    print(f"[{cohort}] Spearman  weight~shapley {a['fusion_weight_vs_shapley']['spearman_rho']:+.2f}"
          f"  deletion-cost~shapley {a['deletion_cost_vs_shapley']['spearman_rho']:+.2f}"
          f"  weight~local {a['fusion_weight_vs_local_share']['spearman_rho']:+.2f}")
    print(f"[{cohort}] strongest pairwise interactions (concordance):")
    for d in pairs[:5]:
        i = d["interaction_cindex"]
        flag = " *" if d["ci_excludes_zero_cindex"] else ""
        print(f"[{cohort}]   {d['pair'][0][:24]} x {d['pair'][1][:24]}: "
              f"{i['mean']*100:+.2f}pp [{i['ci95'][0]*100:+.2f}, {i['ci95'][1]*100:+.2f}]{flag}")
    print(f"[{cohort}] wrote {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=("/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/"
                                           "TCGA-CDR-SupplementalTableS1.xlsx"))
    ap.add_argument("--rounds-dir", required=True,
                    help="one file per completed round; shared by parallel jobs")
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None, help="temporary per-epoch checkpoints")
    ap.add_argument("--aggregate", action="store_true",
                    help="summarise the round files instead of training")
    ap.add_argument("--out", default=None, help="summary JSON, with --aggregate")
    args = ap.parse_args()

    class_dirs = args.class_dir or [os.path.join(BLCA_BASE if args.cohort == "blca" else BRCA_BASE, r)
                                    for r in (BLCA_REGIONS if args.cohort == "blca" else BRCA_REGIONS)]
    region_names = [os.path.basename(d.rstrip("/")) for d in class_dirs]
    R = len(region_names)
    os.makedirs(args.rounds_dir, exist_ok=True)

    if args.aggregate:
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        aggregate(args.rounds_dir, region_names, args.out, args.cohort)
        return

    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/shapley_tmp_{args.cohort}"
    surv = pd.read_excel(args.cdr_xlsx)
    surv = surv[surv["type"].str.contains("BLCA" if args.cohort == "blca" else "BRCA", na=False)]
    surv = surv[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
    surv["PFI.time"] = pd.to_numeric(surv["PFI.time"], errors="coerce")
    surv = surv.dropna(subset=["PFI.time"])
    dataset = MultiRegionDataset(class_dirs, surv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> {args.cohort.upper()}: {len(dataset)} slides, {R} regions, "
          f"{1 << R} coalitions per round, device {device}")
    set_seed(1337)

    for rnd in range(args.round_start, args.round_end + 1):
        if os.path.exists(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json")):
            print(f"round {rnd}: already on disk, skipping")
            continue
        run_round(dataset, R, rnd, args, device)


if __name__ == "__main__":
    main()
