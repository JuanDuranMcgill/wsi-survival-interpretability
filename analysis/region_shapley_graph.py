#!/usr/bin/env python3
"""Exact compartment Shapley values with the graph diffusion switched on.

The main model applies a 3-hop graph diffusion over each compartment's
precomputed k-NN tile graph during training, but every risk score in the paper
is computed with the diffusion bypassed, and region_ablation.py /
region_shapley.py never use it at all. This script reruns the Shapley analysis
with the diffusion used consistently, in training and in evaluation, so the
paper can say whether the diffusion changes which compartments the model
relies on.

Everything else is region_shapley.py: same model class, optimiser, schedule,
epochs, best-epoch selection, removal of a compartment (zeroed before and after
cross-region attention, weights not renormalised), exact Shapley values,
interaction indices, checks and aggregate. As in the main training script, a
compartment's graph is used only when it has at least --min-graph-tiles tiles
and its graph file exists and matches the tile count; otherwise that
compartment is encoded without diffusion. The fraction of compartment-slides
that had a usable graph is recorded per round, and a round stops if it falls
below --min-graph-coverage, so a missing or misplaced set of graph files cannot
silently produce a graph-free run.

Graph files are looked up as <slide>_A.pt next to each slide's embedding file,
as in the training script, or under --adj-dir/<compartment>/<slide>_A.pt (one
subdirectory per compartment, since the files share basenames across
compartments).

Round k here does not share its out-of-bag patients with round k of
region_shapley.py if the two ran on different clusters (patient order follows
the filesystem listing), so compare the two runs at the level of the aggregate,
not round by round.

Usage (one GPU per job; jobs over disjoint rounds can share --rounds-dir):

    python analysis/region_shapley_graph.py --cohort blca \\
        --rounds-dir /scratch/$USER/shapley_graph_blca --round-start 1 --round-end 5
    python analysis/region_shapley_graph.py --cohort blca --aggregate \\
        --rounds-dir /scratch/$USER/shapley_graph_blca --out results/region_shapley_graph_blca.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_shapley as rs  # noqa: E402
from compartment_common import build_dataset, default_class_dirs  # noqa: E402
from region_ablation import (IPGGraphFormer, collate_bags, cox_ph_loss, fast_cindex,  # noqa: E402
                             set_seed)

ADJ_DIR = None
MIN_GRAPH_TILES = 10


class GraphIPG(IPGGraphFormer):
    """IPGGraphFormer whose forward pass uses each compartment's tile graph."""

    def forward_batch(self, feats_list, pos_list, A_list=None):
        device = next(self.parameters()).device
        out = []
        for si in range(len(feats_list)):
            embs = []
            for ri in range(len(feats_list[si])):
                f, p = feats_list[si][ri], pos_list[si][ri]
                if f is None:
                    embs.append(torch.zeros(self.d_model, device=device))
                    continue
                A = None if A_list is None else A_list[si][ri]
                embs.append(self.forward_bag(f, p, A, ri))
            embs = torch.stack(embs)
            attn, _ = self.cross_region_attn(embs.unsqueeze(0), embs.unsqueeze(0), embs.unsqueeze(0),
                                             need_weights=False)
            embs = self.cross_region_norm(embs + attn.squeeze(0))
            w_raw = torch.relu(self.region_weights) + 0.01
            w = w_raw / w_raw.sum()
            out.append((w.unsqueeze(1) * embs).sum(0))
        return self.head(torch.stack(out)).squeeze(-1)


def adj_path(slide_path):
    """<slide>_A.pt beside the slide, or --adj-dir/<compartment>/<slide>_A.pt.

    The graph files of different compartments share a basename, so a graph
    directory must keep one subdirectory per compartment, named as the
    compartment's embedding directory is.
    """
    stem = os.path.splitext(slide_path)[0]
    p = stem + "_A.pt"
    if os.path.exists(p):
        return p
    if ADJ_DIR:
        compartment = os.path.basename(os.path.dirname(slide_path))
        p = os.path.join(ADJ_DIR, compartment, os.path.basename(stem) + "_A.pt")
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=1024)
def load_adj(path):
    A = torch.load(path, map_location="cpu")["A"]
    return A.coalesce() if A.is_sparse else A


def collate_graph(batch):
    """collate_bags plus a per-slide, per-compartment adjacency list and coverage."""
    feats_list, pos_list, t, e, paths_list, ntiles_list = collate_bags(batch)
    A_list, used, eligible = [], 0, 0
    for si, paths in enumerate(paths_list):
        row = []
        for ri, p in enumerate(paths):
            n = ntiles_list[si][ri]
            A = None
            if p is not None and n >= MIN_GRAPH_TILES:
                eligible += 1
                ap = adj_path(p)
                if ap is not None:
                    cand = load_adj(ap)
                    if tuple(cand.shape) == (n, n):
                        A, used = cand, used + 1
            row.append(A)
        A_list.append(row)
    return feats_list, pos_list, t, e, paths_list, ntiles_list, A_list, (used, eligible)


def to_device(A_list, device):
    return [[None if A is None else A.to(device) for A in row] for row in A_list]


def train_one_round(model, loader, device, epochs, save_root, round_id, lr=3e-5):
    """region_ablation.train_one_round with the graph passed through."""
    model.to(device).train()
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if n != "region_weights"],
          "weight_decay": 1e-3},
         {"params": [model.region_weights], "lr": 1e-3, "weight_decay": 0.0}], lr=lr)
    warmup_steps = 3 * len(loader)
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, 0.33, 1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=5e-6)],
        milestones=[warmup_steps])
    used = eligible = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for feats_list, pos_list, t, e, _, _, A_list, (u, g) in loader:
            used += u; eligible += g
            t = torch.tensor(t, dtype=torch.float32, device=device)
            e = torch.tensor(e, dtype=torch.float32, device=device)
            risk = model.forward_batch(feats_list, pos_list, to_device(A_list, device))
            w_raw = torch.relu(model.region_weights) + 0.01
            w = w_raw / w_raw.sum()
            loss = cox_ph_loss(risk, t, e) + 1e-3 * (-(w * torch.log(w + 1e-8)).sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        torch.save(model.state_dict(), os.path.join(save_root, f"round_{round_id}_epoch_{ep}.pt"))
        print(f"  epoch {ep}/{epochs}  loss={tot:.4f}")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return used, eligible


@torch.no_grad()
def oob_eval(model, loader, device):
    model.eval()
    T, E, R, E_emb = [], [], [], []
    used = eligible = 0
    for feats_list, pos_list, t, e, _, _, A_list, (u, g) in loader:
        used += u; eligible += g
        A_dev = to_device(A_list, device)
        R.extend(model.forward_batch(feats_list, pos_list, A_dev).cpu().tolist())
        for si in range(len(feats_list)):
            embs = []
            for ri in range(len(feats_list[si])):
                f, p = feats_list[si][ri], pos_list[si][ri]
                embs.append(torch.zeros(model.d_model, device=device) if f is None
                            else model.forward_bag(f, p, A_dev[si][ri], ri))
            E_emb.append(torch.stack(embs).cpu())
        T.extend(np.asarray(t).tolist()); E.extend(np.asarray(e).tolist())
    return np.array(T), np.array(E), np.array(R), torch.stack(E_emb), used, eligible


def run_round(dataset, R, rnd, args, device):
    t0 = time.time()
    train_idx, oob_idx, seed_used, oob_events = rs.oob_split(dataset, rnd, args.train_frac)
    print(f"\n=== graph round {rnd}  (seed {seed_used}, OOB n={len(oob_idx)}, events={oob_events}) ===")
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_graph)
    oob_loader = DataLoader(Subset(dataset, oob_idx.tolist()), batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=collate_graph)
    os.makedirs(args.save_root, exist_ok=True)
    model = GraphIPG(num_regions=R)
    tr_used, tr_elig = train_one_round(model, train_loader, device, args.epochs, args.save_root, rnd, args.lr)
    cov_train = tr_used / max(tr_elig, 1)
    print(f"  graph coverage in training: {tr_used}/{tr_elig} = {cov_train:.1%}")
    if cov_train < args.min_graph_coverage:
        raise RuntimeError(f"graph coverage {cov_train:.1%} below --min-graph-coverage; "
                           "check the _A.pt files or --adj-dir")
    model.cpu()
    del model

    best_ep, best_c = 1, -1.0
    for ep in range(1, args.epochs + 1):
        m = GraphIPG(num_regions=R)
        m.load_state_dict(torch.load(os.path.join(args.save_root, f"round_{rnd}_epoch_{ep}.pt"),
                                     map_location="cpu"))
        m.to(device)
        T, E, Rk, _, _, _ = oob_eval(m, oob_loader, device)
        c = fast_cindex(T, E, Rk)
        if c > best_c:
            best_c, best_ep = c, ep
        del m
    model = GraphIPG(num_regions=R)
    model.load_state_dict(torch.load(os.path.join(args.save_root, f"round_{rnd}_epoch_{best_ep}.pt"),
                                     map_location="cpu"))
    model.to(device).eval()
    times, events, risk_full, E_emb, ev_used, ev_elig = oob_eval(model, oob_loader, device)

    # coalitions act on the graph-diffused region embeddings, exactly as the model does
    table = rs.coalition_risks(model, E_emb, device)
    full = (1 << R) - 1
    head_check = float(np.max(np.abs(table[full] - risk_full)))
    if head_check > 1e-3:
        raise RuntimeError(f"cached-embedding head differs from the model by {head_check}")
    v = np.array([fast_cindex(times, events, table[m]) for m in range(1 << R)])
    phi = rs.shapley_from_table(v, R)
    I = rs.interaction_from_table(v, R)
    phi_loc = rs.shapley_from_table(table, R)
    I_loc = rs.interaction_from_table(table, R)
    absphi = np.abs(phi_loc)
    share = absphi / np.maximum(absphi.sum(0, keepdims=True), 1e-12)
    risk_sd = float(np.std(risk_full)) or 1.0
    w_raw = torch.relu(model.region_weights.detach().cpu()) + 0.01

    rec = {
        "round": rnd, "seed_used": int(seed_used), "best_epoch": int(best_ep),
        "n_oob": int(len(oob_idx)), "oob_events": int(oob_events),
        "graph_diffusion": True, "min_graph_tiles": MIN_GRAPH_TILES,
        "graph_coverage_train": cov_train, "graph_coverage_eval": ev_used / max(ev_elig, 1),
        "baseline_cindex": float(v[full]), "empty_coalition_cindex": float(v[0]),
        "head_check_max_abs_diff": head_check,
        "efficiency_gap": float(abs(phi.sum() - (v[full] - v[0]))),
        "fusion_weights": (w_raw / w_raw.sum()).numpy().tolist(),
        "v_perf": v.tolist(), "phi_perf": phi.tolist(), "interaction_perf": I.tolist(),
        "deletion_delta": [float(v[full & ~(1 << r)] - v[full]) for r in range(R)],
        "keep_only_gain": [float(v[1 << r] - v[0]) for r in range(R)],
        "local_share_mean": share.mean(1).tolist(),
        "local_share_sd_across_patients": share.std(1).tolist(),
        "local_signed_mean_in_risk_sd": (phi_loc.mean(1) / risk_sd).tolist(),
        "local_interaction_mean_in_risk_sd": (I_loc.mean(2) / risk_sd).tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json"), "w") as f:
        json.dump(rec, f)
    for ep in range(1, args.epochs + 1):
        p = os.path.join(args.save_root, f"round_{rnd}_epoch_{ep}.pt")
        if os.path.exists(p):
            os.remove(p)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"  baseline {v[full]:.4f}, graph coverage eval {rec['graph_coverage_eval']:.1%}, "
          f"head check {head_check:.1e}, efficiency gap {rec['efficiency_gap']:.1e}, {rec['seconds']:.0f}s")
    return rec


def main():
    global ADJ_DIR, MIN_GRAPH_TILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None)
    ap.add_argument("--cdr-xlsx", default=None)
    ap.add_argument("--surv-csv", default=None, help="testing: survival table as CSV")
    ap.add_argument("--adj-dir", default=None, help="root with one subdirectory per compartment holding <slide>_A.pt, if not beside the slides")
    ap.add_argument("--min-graph-tiles", type=int, default=10)
    ap.add_argument("--min-graph-coverage", type=float, default=0.9)
    ap.add_argument("--rounds-dir", required=True)
    ap.add_argument("--round-start", type=int, default=1)
    ap.add_argument("--round-end", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ADJ_DIR, MIN_GRAPH_TILES = args.adj_dir, args.min_graph_tiles
    os.makedirs(args.rounds_dir, exist_ok=True)

    if args.aggregate:
        names = [os.path.basename(d.rstrip("/")) for d in (args.class_dir or default_class_dirs(args.cohort))]
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        rs.aggregate(args.rounds_dir, names, args.out, args.cohort)
        with open(args.out) as f:
            out = json.load(f)
        recs = [json.load(open(os.path.join(args.rounds_dir, p)))
                for p in sorted(os.listdir(args.rounds_dir)) if p.startswith("round_") and p.endswith(".json")]
        out["graph_diffusion"] = True
        out["graph_coverage_eval"] = rs.summarise([r["graph_coverage_eval"] for r in recs])
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return

    dataset, names = build_dataset(args.cohort, args.class_dir, args.cdr_xlsx, args.surv_csv)
    args.save_root = args.save_root or f"/scratch/{os.environ.get('USER', 'user')}/shapley_graph_tmp_{args.cohort}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> graph-on Shapley {args.cohort.upper()}: {len(dataset)} slides, {len(names)} regions, device {device}")
    set_seed(1337)
    for rnd in range(args.round_start, args.round_end + 1):
        if os.path.exists(os.path.join(args.rounds_dir, f"round_{rnd:03d}.json")):
            print(f"round {rnd}: on disk, skipping")
            continue
        run_round(dataset, len(names), rnd, args, device)


if __name__ == "__main__":
    main()
