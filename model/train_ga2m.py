#!/usr/bin/env python3
"""GA2M head training driver (docs/DESIGN_GA2M_HEAD.md).

Reuses the existing backbone's dataset, collate function, Cox loss, c-index,
adjacency loading, and OOM-safety helper by importing the cohort's original
training script as a library (no copy-paste of that logic, so there is no
chance of it silently drifting from the model already used for Table 1).
Only the aggregation head changes.

Per Section 2 ("Training protocol") this uses the *same* bootstrap protocol,
optimiser settings, epochs-per-round and seed_offset scheme as the original,
so that round n of this run sees exactly the same patients (same train/OOB
split) as round n of the original run — required for the paired comparisons
in Section 4.

Per Section 3 ("What to save"), each round writes, for every out-of-bag
patient: main_effects[R], interactions[R][R] (symmetric, zero diagonal),
intercept, risk, patient_id — plus the round's own OOB and in-sample c-index,
epoch count and seed.

Usage (mirrors submit_bootstrap.sh):
    python train_ga2m.py --cohort blca \\
        --class_dir ".../Perivesical adipose tissue (extravesical fat)" \\
        --class_dir ".../Invasive urothelial carcinoma (tumor)" \\
        ... (repeat --class_dir once per region) \\
        --bootstrap_rounds 10 --epochs 30 --seed_offset 0 \\
        --save_root /scratch/sorkwos/ga2m_pilot/blca
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ga2m_head import (  # noqa: E402
    GA2MHead, region_pairs, center_decomposition,
    raw_interactions_from_projections, check_sum_preserved,
)

BACKBONE_PATH = {
    "blca": os.path.join(HERE, "interpret_graphtrans_late_int_sparse_m3save.py"),
    "brca": os.path.join(HERE, "interpret_graphtrans_late_int_sparse_m3save_BRCA.py"),
}


def load_backbone(cohort: str):
    """Import the cohort's original training script as a module, to reuse its
    dataset/collate/loss/eval code verbatim rather than re-deriving it."""
    path = BACKBONE_PATH[cohort]
    spec = importlib.util.spec_from_file_location(f"backbone_{cohort}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Model: identical per-region encoder, GA2M aggregation head
# ============================================================

class GA2MModel(nn.Module):
    """Same per-region encoder as IPGGraphFormer (proj/pos_mlp/beta/agt_layers/
    attn_pool/forward_bag, copied to keep this file self-contained and because
    IPGGraphFormer bakes num_regions=8 into its __init__). Everything from
    cross_region_attn onward is replaced by GA2MHead."""

    def __init__(self, agt_layer_cls, num_regions, in_dim=1536, d_model=384,
                 agt_heads=6, agt_layers=2, gnn_hops=3, knn_k=16, dropout=0.2,
                 ga2m_rank=16, ga2m_hidden=64):
        super().__init__()
        self.gnn_hops = gnn_hops
        self.knn_k = knn_k
        self.num_regions = num_regions
        self.d_model = d_model

        self.proj = nn.ModuleList([nn.Linear(in_dim, d_model) for _ in range(num_regions)])
        self.pos_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            for _ in range(num_regions)
        ])
        self.beta = nn.ParameterList([nn.Parameter(torch.ones(gnn_hops + 1)) for _ in range(num_regions)])
        self.agt_layers = nn.ModuleList([
            nn.ModuleList([agt_layer_cls(d_model, nheads=agt_heads, emb_dropout=dropout) for _ in range(agt_layers)])
            for _ in range(num_regions)
        ])
        self.attn_pool = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
            for _ in range(num_regions)
        ])

        self.ga2m = GA2MHead(num_regions=num_regions, d_model=d_model, rank=ga2m_rank,
                              main_hidden=ga2m_hidden)

    # ---- identical to IPGGraphFormer.sparse_diffusion / forward_bag ----
    def sparse_diffusion(self, H0, A, region_idx):
        if A is None:
            return H0
        if not A.is_coalesced():
            A = A.coalesce()
        beta = self.beta[region_idx]
        Z_prev = H0
        Z = beta[0] * Z_prev
        for s in range(1, len(beta)):
            Z_prev = torch.sparse.mm(A, Z_prev)
            Z = Z + beta[s] * Z_prev
        return Z

    def forward_bag(self, feats, pos, A, region_idx):
        device = self.proj[region_idx].weight.device
        pos = pos.float().to(device)
        feats = feats.float().to(device)

        H = self.proj[region_idx](feats) + self.pos_mlp[region_idx](pos)
        N = H.size(0)
        if N > 1:
            if A is not None:
                H = self.sparse_diffusion(H, A, region_idx)
            for layer in self.agt_layers[region_idx]:
                H = layer(H)
            w = self.attn_pool[region_idx](H).squeeze(-1)
            w = torch.softmax(w, dim=0)
            pooled = (w.unsqueeze(1) * H).sum(dim=0)
        else:
            pooled = H.squeeze(0)
        return pooled

    def _region_embs_for_slide(self, feats_regions, pos_regions, A_regions):
        device = next(self.parameters()).device
        region_embs = []
        for region_idx in range(self.num_regions):
            f = feats_regions[region_idx]
            p = pos_regions[region_idx]
            A = None if A_regions is None else A_regions[region_idx]
            if f is None:
                region_embs.append(torch.zeros(self.d_model, device=device))
                continue
            region_embs.append(self.forward_bag(f, p, A, region_idx))
        return torch.stack(region_embs)  # (R, d_model)

    def _stack_region_embs(self, feats_list, pos_list, A_list):
        slide_embs = []
        for slide_idx in range(len(feats_list)):
            A_regions = None if A_list is None else A_list[slide_idx]
            slide_embs.append(self._region_embs_for_slide(
                feats_list[slide_idx], pos_list[slide_idx], A_regions))
        return torch.stack(slide_embs)  # (B, R, d_model)

    def forward_batch_components(self, feats_list, pos_list, A_list):
        """Centering is applied within this batch. Fine for training, where
        the batch is the actual training mini-batch (Section 1 says centering
        is enforced "per training batch"). NOT fine for extracting per-patient
        attributions one patient at a time — see forward_batch_raw."""
        H = self._stack_region_embs(feats_list, pos_list, A_list)
        risk, main_c, inter_c, intercept, info = self.ga2m(H)
        return risk, main_c, inter_c, intercept, info

    def forward_batch_raw(self, feats_list, pos_list, A_list):
        """Uncentered main effect + rank-k projections (u_r = A_r h_r,
        v_r = B_r h_r), no batch-relative statistics involved. Used to
        accumulate per-patient terms across an entire OOB set before
        centering once — the exact projection-based centering needs the
        projections themselves, not just the collapsed scalar interaction,
        so this returns Ah/Bh rather than a pre-summed interaction term."""
        H = self._stack_region_embs(feats_list, pos_list, A_list)
        main_raw, Ah, Bh = self.ga2m.raw_projections(H)
        return main_raw, Ah, Bh

    def forward_batch(self, feats_list, pos_list, A_list):
        """Risk-only, for drop-in compatibility with the reused
        eval_cindex_evalsurv / run_with_oom_split from the backbone module."""
        risk, *_ = self.forward_batch_components(feats_list, pos_list, A_list)
        return risk


# ============================================================
# Training (mirrors backbone.train_one_round / bootstrap_rounds)
# ============================================================

def make_split(dataset, r, seed_offset, train_frac, min_val_events=20, max_retries=20):
    """Identical to the split logic in bootstrap_rounds() in the original
    scripts, copied rather than imported because it is inline in that
    function, not a standalone helper. Must stay byte-identical so round n
    sees the same patients here as in the original run."""
    N = len(dataset)
    idx = np.arange(N)
    global_round = seed_offset + r
    seed_attempt = global_round
    retry = 0
    while True:
        np.random.seed(seed_attempt)
        random.seed(seed_attempt)
        torch.manual_seed(seed_attempt)

        train_idx = np.random.choice(idx, int(train_frac * N), replace=True)
        mask = np.ones(N, bool)
        mask[train_idx] = False
        test_idx = idx[mask]
        if len(test_idx) == 0:
            perm = np.random.permutation(N)
            split = int(0.8 * N)
            train_idx, test_idx = perm[:split], perm[split:]

        val_events = sum(int(dataset.labels[i][1]) for i in test_idx)
        if val_events >= min_val_events:
            break
        retry += 1
        if retry >= max_retries:
            break
        seed_attempt = global_round * 1000 + retry
    return train_idx, test_idx, seed_attempt


def build_optimizer(model, lr, epochs, warmup_epochs, decay_start_epoch, steps_per_epoch):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup_steps = warmup_epochs * steps_per_epoch
    hold_steps = max(0, (decay_start_epoch - 1 - warmup_epochs) * steps_per_epoch)
    decay_steps = max(1, (epochs - (decay_start_epoch - 1)) * steps_per_epoch)
    sched_warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.33, end_factor=1.0, total_iters=warmup_steps)
    sched_hold = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=hold_steps)
    sched_decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay_steps, eta_min=5e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[sched_warmup, sched_hold, sched_decay],
        milestones=[warmup_steps, warmup_steps + hold_steps],
    )
    return opt, scheduler


def in_sample_cindex(model, dataset, train_idx, backbone, device):
    """In-sample c-index on the (with-replacement) training set for this
    round, for the overfitting-gap comparison in Section 4.2."""
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, sorted(set(train_idx.tolist()))),
        batch_size=1, shuffle=False, collate_fn=backbone.collate_bags, num_workers=0,
    )
    return backbone.eval_cindex_evalsurv(model, loader, device)


@torch.no_grad()
def extract_oob_components(model, dataset, test_idx, backbone, device):
    """Per-OOB-patient main/interaction terms for Section 3's save.

    Two passes, deliberately: (1) one forward pass per OOB patient (batch
    size 1 — patients have a variable number of tiles per region, so this is
    the simplest correct way to run the per-region encoder) collecting the
    *raw*, uncentered terms; (2) a single `center_decomposition` call over
    the whole OOB set at once. Centering per single-patient batch would
    subtract each patient's own value as the "batch mean" and zero out every
    saved attribution while dumping all the signal into a meaningless
    per-patient intercept — the risk score would still be correct (centering
    is sum-preserving at any batch size) but the saved decomposition would
    be silent garbage. See GA2MModel.forward_batch_raw.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx.tolist()),
        batch_size=1, shuffle=False, collate_fn=backbone.collate_bags, num_workers=0,
    )
    pids, main_raws, Ahs, Bhs, times, events = [], [], [], [], [], []
    for feats_list, pos_list, t, e, paths, ntiles in loader:
        main_raw, Ah, Bh = model.forward_batch_raw(feats_list, pos_list, None)
        pid = os.path.basename(paths[0][0])[:12] if paths[0][0] else "missing"
        pids.append(pid)
        main_raws.append(main_raw[0].detach().cpu())
        Ahs.append(Ah[0].detach().cpu())
        Bhs.append(Bh[0].detach().cpu())
        times.append(float(t[0]))
        events.append(int(e[0]))

    main_raw_all = torch.stack(main_raws)   # (n_oob, R)
    Ah_all = torch.stack(Ahs)                # (n_oob, R, k)
    Bh_all = torch.stack(Bhs)                # (n_oob, R, k)
    main_c, inter_c, intercept, info = center_decomposition(
        main_raw_all, Ah_all, Bh_all, model.ga2m.pairs)
    risk = intercept + main_c.sum(dim=1) + inter_c.sum(dim=1)

    # Cheap, always-on sanity check: centering must not change the total.
    inter_raw_all = raw_interactions_from_projections(Ah_all, Bh_all, model.ga2m.pairs)
    ok, max_diff = check_sum_preserved(main_raw_all, inter_raw_all, main_c, inter_c, intercept)
    if not ok:
        raise RuntimeError(f"GA2M centering broke sum-preservation this round (max diff {max_diff:.3e})")

    return {
        "patient_ids": np.array(pids),
        "main_effects": main_c.numpy(),           # (n_oob, R)
        "interactions_flat": inter_c.numpy(),      # (n_oob, P)
        "intercept": float(intercept.item()),       # one scalar for the whole round
        "risk": risk.numpy(),
        "pfi_time": np.array(times),
        "pfi_event": np.array(events),
        "centering_info": info,
        "sum_preservation_max_diff": max_diff,
    }


def expand_interactions(flat, num_regions):
    """(n, P) canonical-pairs -> (n, R, R) symmetric, zero diagonal, per
    Section 3's literal save format."""
    pairs = region_pairs(num_regions)
    n = flat.shape[0]
    out = np.zeros((n, num_regions, num_regions), dtype=flat.dtype)
    for p_idx, (r, s) in enumerate(pairs):
        out[:, r, s] = flat[:, p_idx]
        out[:, s, r] = flat[:, p_idx]
    return out


def build_adjacency_list(paths, ntiles, backbone, device):
    """Real k-NN adjacency for training, mirroring the exact logic in the
    original train_one_round (resolve_adj_path + cached_graph + a 10-tile
    minimum before diffusion is attempted). Kept for fidelity: Section 2
    requires the same forward-pass mechanics as the existing backbone, not
    just the same loss/optimiser/split, so graph diffusion must actually run
    at train time here the way it does in the original."""
    A_list = []
    for slide_paths, slide_ntiles in zip(paths, ntiles):
        slide_A = []
        for p, n in zip(slide_paths, slide_ntiles):
            if p is None or n < 10:
                slide_A.append(None)
                continue
            ap = backbone.resolve_adj_path(p)
            A = backbone.cached_graph(ap) if ap else None
            slide_A.append(A.to(device, non_blocking=True) if A is not None else None)
        A_list.append(slide_A)
    return A_list


def train_one_round_ga2m(model, train_loader, device, epochs, lr, lam_int, cox_loss_fn, backbone):
    opt, scheduler = build_optimizer(model, lr, epochs, warmup_epochs=3, decay_start_epoch=12,
                                      steps_per_epoch=len(train_loader))
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feats_list, pos_list, t, e, paths, ntiles in train_loader:
            t = torch.tensor(t, dtype=torch.float32, device=device)
            e = torch.tensor(e, dtype=torch.float32, device=device)
            A_list = build_adjacency_list(paths, ntiles, backbone, device)
            risk, main_c, inter_c, intercept, _ = model.forward_batch_components(feats_list, pos_list, A_list)
            cox = cox_loss_fn(risk, t, e)
            l1 = model.ga2m.l1_interaction_penalty(inter_c)
            loss = cox + lam_int * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += float(loss.item())
            del feats_list, pos_list, A_list, t, e
        print(f"  epoch {ep:3d}/{epochs}  loss={total_loss:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", required=True)
    ap.add_argument("--bootstrap_rounds", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lam_int", type=float, default=1e-3, help="L1 penalty weight on interactions")
    ap.add_argument("--ga2m_rank", type=int, default=16)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--save_root", required=True)
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> device = {device}")

    backbone = load_backbone(args.cohort)

    excel = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
    surv_df = pd.read_excel(excel)
    surv_df = surv_df[surv_df["type"].str.contains("BLCA|HNSC|LUAD|BRCA|UCEC", na=False)]
    surv_df = surv_df[["bcr_patient_barcode", "PFI", "PFI.time"]]
    surv_df["PFI.time"] = pd.to_numeric(surv_df["PFI.time"], errors="coerce")
    surv_df = surv_df.dropna(subset=["PFI.time"])

    dataset = backbone.MultiRegionDataset(args.class_dir, surv_df)
    num_regions = len(args.class_dir)
    print(f">>> {len(dataset)} slides, {num_regions} regions: {dataset.region_names}")

    round_summaries = []

    for r in range(1, args.bootstrap_rounds + 1):
        t_round = time.time()
        train_idx, test_idx, seed_used = make_split(dataset, r, args.seed_offset, args.train_frac)
        print(f"\n=== GA2M round {r}/{args.bootstrap_rounds} (seed {seed_used}) ===")

        backbone.set_seed(seed_used)
        model = GA2MModel(backbone.AGTLayer, num_regions=num_regions,
                           ga2m_rank=args.ga2m_rank).to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx.tolist()),
            batch_size=args.batch_size, shuffle=True, collate_fn=backbone.collate_bags, num_workers=0,
        )

        model = train_one_round_ga2m(model, train_loader, device, args.epochs, args.lr, args.lam_int,
                                      cox_loss_fn=backbone.cox_ph_loss, backbone=backbone)

        oob = extract_oob_components(model, dataset, test_idx, backbone, device)
        oob_cidx = backbone.fast_cindex(oob["pfi_time"], oob["pfi_event"], oob["risk"])
        train_cidx = in_sample_cindex(model, dataset, train_idx, backbone, device)
        wall_time = time.time() - t_round

        interactions_full = expand_interactions(oob["interactions_flat"], num_regions)
        round_path = os.path.join(args.save_root, f"round_{r}.npz")
        np.savez(
            round_path,
            patient_ids=oob["patient_ids"],
            main_effects=oob["main_effects"],
            interactions=interactions_full,
            intercept=oob["intercept"],
            risk=oob["risk"],
            pfi_time=oob["pfi_time"],
            pfi_event=oob["pfi_event"],
            region_names=np.array(dataset.region_names),
            oob_cindex=oob_cidx,
            train_cindex=train_cidx,
            epochs=args.epochs,
            seed=seed_used,
            round_id=r,
            wall_time_sec=wall_time,
            lam_int=args.lam_int,
            ga2m_rank=args.ga2m_rank,
            sum_preservation_max_diff=oob["sum_preservation_max_diff"],
        )
        print(f">>> round {r}: OOB c-index={oob_cidx:.4f}  in-sample c-index={train_cidx:.4f}  "
              f"n_oob={len(oob['patient_ids'])}  wall_time={wall_time:.1f}s  -> {round_path}")

        round_summaries.append({
            "round": r, "seed": int(seed_used), "oob_cindex": float(oob_cidx),
            "train_cindex": float(train_cidx), "n_oob": int(len(oob["patient_ids"])),
            "wall_time_sec": float(wall_time),
            "sum_preservation_max_diff": oob["sum_preservation_max_diff"],
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "cohort": args.cohort,
        "endpoint": "PFI",
        "num_regions": num_regions,
        "region_names": dataset.region_names,
        "bootstrap_rounds": args.bootstrap_rounds,
        "epochs_per_round": args.epochs,
        "seed_offset": args.seed_offset,
        "lam_int": args.lam_int,
        "ga2m_rank": args.ga2m_rank,
        "centering_method": "exact_projection_centering",
        "save_root": os.path.abspath(args.save_root),
        "rounds": round_summaries,
        "mean_oob_cindex": float(np.mean([s["oob_cindex"] for s in round_summaries])),
        "std_oob_cindex": float(np.std([s["oob_cindex"] for s in round_summaries])),
        "mean_train_cindex": float(np.mean([s["train_cindex"] for s in round_summaries])),
        "mean_wall_time_sec": float(np.mean([s["wall_time_sec"] for s in round_summaries])),
    }
    summary_path = os.path.join(args.save_root, "pilot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> wrote {summary_path}")
    print(f">>> mean OOB c-index = {summary['mean_oob_cindex']:.4f} "
          f"(+/- {summary['std_oob_cindex']:.4f})  mean in-sample = {summary['mean_train_cindex']:.4f}  "
          f"mean wall time/round = {summary['mean_wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
