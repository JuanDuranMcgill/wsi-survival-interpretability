#!/usr/bin/env python3
"""The compartment audit applied to a standard attention MIL model.

The paper's audit was developed on our own region-fusion architecture. This
script applies the same checks to gated attention-based multiple-instance
learning (ABMIL; Ilse, Tomczak and Welling, 2018), the architecture behind most
attention heatmaps in computational pathology, trained on the same UNI-2 tile
features, patients and out-of-bag splits.

ABMIL has no per-compartment weight. What it offers in its place is attention:
the share of a slide's attention that falls on each compartment's tiles, which
is what a heatmap shows. That share is compared with each compartment's exact
Shapley value and with its single-compartment deletion, exactly as the paper
compares the fusion weights.

Removing a compartment means dropping its tiles from the bag, which applies to
any MIL model. For ABMIL it is also exact and cheap: a tile's attention logit
depends only on that tile, so with A_r the summed (stabilised) exponentiated
logits of compartment r and H_r the matching weighted sum of tile embeddings,
the pooled embedding of any coalition S is sum_{r in S} H_r / sum_{r in S} A_r.
Every one of the 2^R coalitions is therefore scored from R cached pairs per
patient. A coalition with no tiles for a patient gives the pooled embedding
zero, the same constant for everyone, so the empty coalition has concordance
0.5. The script checks that the full coalition reproduces the model.

Training: Cox partial likelihood, AdamW, the out-of-bag split of
region_shapley.py for the same round, epoch chosen by out-of-bag concordance as
in the rest of the paper. During training a bag is subsampled to at most
--max-train-tiles tiles; evaluation always uses every tile.

Usage (Trillium, one GPU per job):

    python analysis/abmil_compartment_shapley.py --cohort blca \\
        --rounds-dir /scratch/$USER/abmil_blca --round-start 1 --round-end 10
    python analysis/abmil_compartment_shapley.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/abmil_blca --out results/abmil_shapley_blca.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compartment_common import build_dataset  # noqa: E402
from region_ablation import collate_bags, cox_ph_loss, fast_cindex, set_seed  # noqa: E402
from region_shapley import interaction_from_table, oob_split, shapley_from_table, summarise  # noqa: E402


class GatedABMIL(nn.Module):
    """Gated attention MIL (Ilse et al., 2018) with a Cox risk head."""

    def __init__(self, in_dim=1536, hid=256, att=128, dropout=0.25):
        super().__init__()
        self.hid = hid
        self.fc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout))
        self.att_a = nn.Sequential(nn.Linear(hid, att), nn.Tanh(), nn.Dropout(dropout))
        self.att_b = nn.Sequential(nn.Linear(hid, att), nn.Sigmoid(), nn.Dropout(dropout))
        self.att_c = nn.Linear(att, 1)
        self.head = nn.Linear(hid, 1)

    def tiles(self, x):
        h = self.fc(x)
        return h, self.att_c(self.att_a(h) * self.att_b(h)).squeeze(-1)

    def forward_bag(self, feats_by_region, device, max_tiles=None):
        xs = [f for f in feats_by_region if f is not None and f.shape[0] > 0]
        if not xs:
            return self.head(torch.zeros(self.hid, device=device)).squeeze(-1)
        x = torch.cat(xs)
        if max_tiles and x.shape[0] > max_tiles:
            x = x[torch.randperm(x.shape[0])[:max_tiles]]
        h, a = self.tiles(x.to(device).float())
        w = torch.softmax(a, dim=0)
        return self.head((w.unsqueeze(1) * h).sum(0)).squeeze(-1)

    @torch.no_grad()
    def region_stats(self, feats_by_region, device):
        """A (R,) stabilised attention mass and H (R, hid) weighted embedding sum."""
        R = len(feats_by_region)
        hs, as_ = [], []
        for f in feats_by_region:
            if f is None or f.shape[0] == 0:
                hs.append(None); as_.append(None)
                continue
            h, a = self.tiles(f.to(device).float())
            hs.append(h); as_.append(a)
        present = [a for a in as_ if a is not None]
        A = torch.zeros(R, device=device)
        H = torch.zeros(R, self.hid, device=device)
        if not present:
            return A, H
        m = torch.max(torch.stack([a.max() for a in present]))
        for r in range(R):
            if as_[r] is not None:
                e = torch.exp(as_[r] - m)
                A[r] = e.sum()
                H[r] = (e.unsqueeze(1) * hs[r]).sum(0)
        return A, H

    @torch.no_grad()
    def coalition_risk(self, A, H, keep):
        """A (n, R), H (n, R, hid), keep bool (R,) -> risk (n,)."""
        k = keep.to(A.dtype)
        num = (H * k.view(1, -1, 1)).sum(1)
        den = (A * k.view(1, -1)).sum(1, keepdim=True)
        pooled = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def oob_eval(model, loader, device):
    model.eval()
    T, E, Rk = [], [], []
    for feats_list, _, t, e, _, _ in loader:
        for si in range(len(feats_list)):
            Rk.append(float(model.forward_bag(feats_list[si], device)))
        T.extend(np.asarray(t).tolist()); E.extend(np.asarray(e).tolist())
    return fast_cindex(T, E, Rk)


def train(model, train_loader, oob_loader, device, args):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    states, by_epoch = [], []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for feats_list, _, t, e, _, _ in train_loader:
            risk = torch.stack([model.forward_bag(fl, device, args.max_train_tiles) for fl in feats_list])
            tt = torch.tensor(t, dtype=torch.float32, device=device)
            ee = torch.tensor(e, dtype=torch.float32, device=device)
            if ee.sum() == 0:
                continue
            loss = cox_ph_loss(risk, tt, ee)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        c = oob_eval(model, oob_loader, device)
        by_epoch.append(float(c))
        states.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
        print(f"  epoch {ep}/{args.epochs}  loss={tot:.3f}  OOB c-index={c:.4f}")
    best = int(np.argmax(by_epoch))
    model.load_state_dict(states[best])
    return best + 1, by_epoch


def run_round(dataset, rnd, args, device):
    R = len(dataset.region_names)
    os.makedirs(args.rounds_dir, exist_ok=True)
    t0 = time.time()
    train_idx, oob_idx, seed_used, oob_events = oob_split(dataset, rnd, args.train_frac)
    print(f"\n=== ABMIL round {rnd}  (seed {seed_used}, OOB n={len(oob_idx)}, events={oob_events}) ===")
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_bags)
    oob_loader = DataLoader(Subset(dataset, oob_idx.tolist()), batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=collate_bags)
    model = GatedABMIL(in_dim=args.in_dim)
    best_ep, by_epoch = train(model, train_loader, oob_loader, device, args)
    model.eval()

    A_all, H_all, T, E, R_full = [], [], [], [], []
    with torch.no_grad():
        for feats_list, _, t, e, _, _ in oob_loader:
            for si in range(len(feats_list)):
                A, H = model.region_stats(feats_list[si], device)
                A_all.append(A); H_all.append(H)
                R_full.append(float(model.forward_bag(feats_list[si], device)))
            T.extend(np.asarray(t).tolist()); E.extend(np.asarray(e).tolist())
    A = torch.stack(A_all); H = torch.stack(H_all)
    T, E, R_full = np.array(T), np.array(E), np.array(R_full)

    table = np.zeros((1 << R, len(T)))
    for mask in range(1 << R):
        keep = torch.tensor([(mask >> r) & 1 == 1 for r in range(R)], device=device)
        table[mask] = model.coalition_risk(A, H, keep).cpu().numpy()
    full = (1 << R) - 1
    head_check = float(np.max(np.abs(table[full] - R_full)))
    if head_check > 1e-3:
        raise RuntimeError(f"coalition path differs from the model by {head_check}")

    v = np.array([fast_cindex(T, E, table[m]) for m in range(1 << R)])
    phi = shapley_from_table(v, R)
    I = interaction_from_table(v, R)
    eff_gap = float(abs(phi.sum() - (v[full] - v[0])))
    phi_loc = shapley_from_table(table, R)                       # (R, n)
    absphi = np.abs(phi_loc)
    share = absphi / np.maximum(absphi.sum(0, keepdims=True), 1e-12)

    An = A.cpu().numpy()
    att = An / np.maximum(An.sum(1, keepdims=True), 1e-30)       # (n, R) attention mass
    present = An > 0
    from scipy.stats import spearmanr
    per_patient = []
    for i in range(len(T)):
        p = present[i]
        if p.sum() >= 3:
            rho = spearmanr(att[i, p], share[p, i]).correlation
            if np.isfinite(rho):
                per_patient.append(rho)

    rec = {
        "round": rnd, "seed_used": int(seed_used), "best_epoch": best_ep,
        "cindex_by_epoch": by_epoch, "n_oob": int(len(oob_idx)), "oob_events": int(oob_events),
        "baseline_cindex": float(v[full]), "empty_coalition_cindex": float(v[0]),
        "head_check_max_abs_diff": head_check, "efficiency_gap": eff_gap,
        "attention_share": att.mean(0).tolist(),
        "attention_share_where_present": [float(att[present[:, r], r].mean()) if present[:, r].any()
                                          else float("nan") for r in range(R)],
        "presence_rate": present.mean(0).tolist(),
        "phi_perf": phi.tolist(), "interaction_perf": I.tolist(),
        "deletion_delta": [float(v[full & ~(1 << r)] - v[full]) for r in range(R)],
        "keep_only_gain": [float(v[1 << r] - v[0]) for r in range(R)],
        "local_share_mean": share.mean(1).tolist(),
        "per_patient_attention_vs_local_shapley_rho_mean": float(np.mean(per_patient)) if per_patient else None,
        "per_patient_n": len(per_patient),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json"), "w") as f:
        json.dump(rec, f)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"  baseline {v[full]:.4f}, head check {head_check:.1e}, efficiency gap {eff_gap:.1e}, "
          f"{rec['seconds']:.0f}s")
    return rec


def aggregate(rounds_dir, region_names, out_path, cohort):
    from scipy.stats import spearmanr
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(rounds_dir, "round_*.json")))]
    if not recs:
        raise SystemExit(f"no round files in {rounds_dir}")
    arr = lambda k: np.array([r[k] for r in recs], dtype=float)  # noqa: E731
    att, phi, dele, keep1, share = (arr("attention_share"), arr("phi_perf"), arr("deletion_delta"),
                                    arr("keep_only_gain"), arr("local_share_mean"))
    per_region = {nm: {"attention_share": summarise(att[:, r]),
                       "shapley_cindex": summarise(phi[:, r]),
                       "deletion_delta": summarise(dele[:, r]),
                       "keep_only_gain": summarise(keep1[:, r]),
                       "local_attribution_share": summarise(share[:, r]),
                       "presence_rate": float(np.mean([x["presence_rate"][r] for x in recs]))}
                  for r, nm in enumerate(region_names)}
    m = lambda k: [per_region[n][k]["mean"] for n in region_names]  # noqa: E731
    sp = lambda a, b: {"spearman_rho": float(spearmanr(a, b).correlation),  # noqa: E731
                       "p": float(spearmanr(a, b).pvalue)}
    cost = [-x for x in m("deletion_delta")]
    pp = [r["per_patient_attention_vs_local_shapley_rho_mean"] for r in recs
          if r["per_patient_attention_vs_local_shapley_rho_mean"] is not None]
    out = {
        "cohort": cohort, "model": "gated ABMIL (Ilse et al., 2018)", "n_rounds": len(recs),
        "region_names": region_names,
        "removal": "a compartment's tiles are dropped from the bag; attention renormalises over the rest",
        "checks": {"max_head_check_abs_diff": float(arr("head_check_max_abs_diff").max()),
                   "max_efficiency_gap": float(arr("efficiency_gap").max()),
                   "empty_coalition_cindex": summarise(arr("empty_coalition_cindex"))},
        "baseline_cindex": summarise(arr("baseline_cindex")),
        "per_region": per_region,
        "agreement_across_regions": {
            "attention_share_vs_shapley": sp(m("attention_share"), m("shapley_cindex")),
            "attention_share_vs_deletion_cost": sp(m("attention_share"), cost),
            "deletion_cost_vs_shapley": sp(cost, m("shapley_cindex")),
        },
        "per_patient_attention_vs_local_shapley": summarise(pp) if len(pp) > 1 else None,
        "reading": ("attention_share is the mean fraction of a slide's attention on each "
                    "compartment's tiles, what a heatmap shows. shapley_cindex is the exact "
                    "Shapley value with out-of-bag concordance as payoff. deletion_delta is "
                    "negative when a compartment matters; the agreement uses its cost, -delta."),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[{cohort}] ABMIL {len(recs)} rounds, baseline {out['baseline_cindex']['mean']:.4f}; "
          f"checks head {out['checks']['max_head_check_abs_diff']:.1e}, "
          f"efficiency {out['checks']['max_efficiency_gap']:.1e}")
    print(f"[{cohort}] {'region':34} {'attention':>9} {'shapley':>9} {'deletion':>9}")
    for nm in region_names:
        v = per_region[nm]
        print(f"[{cohort}] {nm[:34]:34} {v['attention_share']['mean']*100:8.1f}% "
              f"{v['shapley_cindex']['mean']*100:+8.2f}pp {v['deletion_delta']['mean']*100:+8.2f}pp")
    a = out["agreement_across_regions"]
    print(f"[{cohort}] Spearman attention~shapley {a['attention_share_vs_shapley']['spearman_rho']:+.2f} "
          f"(p={a['attention_share_vs_shapley']['p']:.3f}), deletion-cost~shapley "
          f"{a['deletion_cost_vs_shapley']['spearman_rho']:+.2f}")
    print(f"[{cohort}] wrote {out_path}")
    return out


def add_train_args(ap):
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-train-tiles", type=int, default=4096)
    ap.add_argument("--in-dim", type=int, default=1536)
    ap.add_argument("--num-workers", type=int, default=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=10)
    add_train_args(ap)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.aggregate:
        from compartment_common import default_class_dirs
        names = [os.path.basename(d.rstrip("/")) for d in (args.class_dir or default_class_dirs(args.cohort))]
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        aggregate(args.rounds_dir, names, args.out, args.cohort)
        return

    dataset, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> ABMIL {args.cohort.upper()}: {len(dataset)} slides, {len(names)} compartments, device {device}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        if os.path.exists(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json")):
            print(f"round {rnd}: on disk, skipping")
            continue
        run_round(dataset, rnd, args, device)


if __name__ == "__main__":
    main()
