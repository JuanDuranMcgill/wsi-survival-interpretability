#!/usr/bin/env python3
"""
Step 5a: Region ablation for interpretability.

Trains 20 bootstrap rounds from scratch (matching the original protocol), then
evaluates each round's best-epoch model on out-of-bag (OOB) patients with one
region zeroed at a time.

Zeroing mechanism: region r's embedding is set to zero AFTER per-region
processing (projection + graph-diffusion + attention pooling) but BEFORE
cross-region self-attention. This removes region r from both its direct
contribution to the patient embedding and from influencing other regions via
cross-region attention.

Two renormalisation variants are reported:
  no_renorm — remaining weights unchanged; region r contributes w[r]*0 = 0,
              so the patient embedding = sum_{j!=r} w[j]*emb[j] and weights
              still sum to ~1 (because the original normalisation is kept).
  renorm    — w[r] forced to 0, remaining weights renormalised to sum to 1;
              answers "what if region r did not exist?"

delta_cindex = ablated_cindex - baseline_cindex (negative means the region
carries useful signal).

Outputs: results/region_ablation_{cohort}.json

Usage (Trillium, single GPU):
  source /home/sorkwos/envs/conch_env/bin/activate
  python analysis/region_ablation.py \\
    --cohort blca \\
    --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/<Region 1>" \\
    --class_dir "..." \\
    --cdr-xlsx /home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx \\
    --rounds 20 --epochs 15 \\
    --out results/region_ablation_blca.json
"""

import os
import sys
import json
import random
import time
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── repo root on sys.path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ── NumPy pickle shim (same as training script) ───────────────────────────────
import types as _types
_fake_core = _types.ModuleType("numpy._core")
_fake_core.multiarray = np.core.multiarray
_fake_core.umath = np.core.umath
sys.modules.setdefault("numpy._core", _fake_core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.umath", np.core.umath)
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_xy(coords_np: np.ndarray) -> np.ndarray:
    if coords_np.size == 0:
        return coords_np.astype(np.float32)
    xs = coords_np[:, 0].astype(np.float32)
    ys = coords_np[:, 1].astype(np.float32)
    xs = (xs - xs.min()) / max(xs.max() - xs.min(), 1.0)
    ys = (ys - ys.min()) / max(ys.max() - ys.min(), 1.0)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def load_slide(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiRegionDataset(Dataset):
    def __init__(self, class_dirs, surv_df):
        self.region_dirs = class_dirs
        self.region_names = [os.path.basename(d.rstrip("/")) for d in class_dirs]
        self.samples = []
        self.labels = []

        ref_dir = class_dirs[0]
        if any(n.endswith(".pt") for n in os.listdir(ref_dir)):
            candidates = [n for n in os.listdir(ref_dir)
                          if n.endswith(".pt") and "DX1" in n and not n.endswith("_A.pt")]
        else:
            candidates = []
            for root, _, files in os.walk(ref_dir):
                for fn in files:
                    if fn.endswith(".pt") and "DX1" in fn and not fn.endswith("_A.pt"):
                        candidates.append(fn)

        for slide_name in candidates:
            case_id = slide_name[:12]
            row = surv_df.loc[surv_df["bcr_patient_barcode"] == case_id]
            if row.empty:
                continue
            region_paths = {}
            for rname, rdir in zip(self.region_names, self.region_dirs):
                p = os.path.join(rdir, slide_name)
                region_paths[rname] = p if os.path.exists(p) else None
            t = float(row["PFI.time"].values[0])
            e = int(row["PFI"].values[0])
            self.samples.append(region_paths)
            self.labels.append((t, e))

        if not self.samples:
            raise RuntimeError("No slides matched survival table")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor(self.labels[idx][0]), torch.tensor(self.labels[idx][1])


def collate_bags(batch):
    feats_list, pos_list, times, events, paths_list, ntiles_list = [], [], [], [], [], []
    for region_paths, t, e in batch:
        feats_r, pos_r, paths_r, ntiles_r = [], [], [], []
        for rname, path in region_paths.items():
            if path is None:
                feats_r.append(None); pos_r.append(None)
                paths_r.append(None); ntiles_r.append(0)
                continue
            d = load_slide(path)
            coords = d["coords"]
            coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
            feats = d["features"]
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            feats = feats.to(torch.float32).contiguous()
            pos = torch.from_numpy(normalize_xy(coords_np)).to(torch.float32).contiguous()
            feats_r.append(feats); pos_r.append(pos)
            paths_r.append(path); ntiles_r.append(feats.shape[0])
        feats_list.append(feats_r); pos_list.append(pos_r)
        paths_list.append(paths_r); ntiles_list.append(ntiles_r)
        times.append(t.item()); events.append(int(e.item()))
    return feats_list, pos_list, np.array(times), np.array(events), paths_list, ntiles_list


# ── Model ─────────────────────────────────────────────────────────────────────

class AGTLayer(nn.Module):
    def __init__(self, dim_in, nheads=6, emb_dropout=0.15):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim_in // nheads
        self.linear_k = nn.Linear(dim_in, dim_in, bias=False)
        self.linear_q = nn.Linear(dim_in, dim_in, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_in, bias=False)
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(dim_in, dim_in, bias=False)
        self.dropout = nn.Dropout(emb_dropout)
        self.ln = nn.LayerNorm(dim_in)

    def forward(self, h):
        N = h.size(0)
        k = self.linear_k(h).reshape(N, self.nheads, self.head_dim)
        q = self.linear_q(h).reshape(N, self.nheads, self.head_dim)
        v = self.linear_v(h).reshape(N, self.nheads, self.head_dim)
        q = self.relu(q); k = self.relu(k)
        kv = torch.einsum("nhd,nhe->hde", k, v)
        num = torch.einsum("nhd,hde->nhe", q, kv)
        denom = torch.einsum("nhd,hd->n", q, k.sum(dim=0)) + 1e-6
        attn = torch.einsum("nhe,n->nhe", num, 1.0 / denom)
        return self.ln(h + self.dropout(self.linear_final(attn.reshape(N, -1))))


class IPGGraphFormer(nn.Module):
    def __init__(self, num_regions, in_dim=1536, d_model=384, agt_heads=6,
                 agt_layers=2, gnn_hops=3, dropout=0.2):
        super().__init__()
        self.num_regions = num_regions
        self.d_model = d_model
        self.gnn_hops = gnn_hops

        self.proj = nn.ModuleList([nn.Linear(in_dim, d_model) for _ in range(num_regions)])
        self.pos_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            for _ in range(num_regions)
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.ones(gnn_hops + 1)) for _ in range(num_regions)
        ])
        self.agt_layers = nn.ModuleList([
            nn.ModuleList([AGTLayer(d_model, nheads=agt_heads, emb_dropout=dropout)
                           for _ in range(agt_layers)])
            for _ in range(num_regions)
        ])
        self.attn_pool = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
            for _ in range(num_regions)
        ])
        self.region_weights = nn.Parameter(torch.ones(num_regions))
        self.cross_region_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_region_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def sparse_diffusion(self, H0, A, region_idx):
        if A is None:
            return H0
        if not A.is_coalesced():
            A = A.coalesce()
        beta = self.beta[region_idx]
        Z = beta[0] * H0
        Z_prev = H0
        for s in range(1, len(beta)):
            Z_prev = torch.sparse.mm(A, Z_prev)
            Z = Z + beta[s] * Z_prev
        return Z

    def forward_bag(self, feats, pos, A, region_idx):
        device = self.proj[region_idx].weight.device
        feats = feats.float().to(device)
        pos = pos.float().to(device)
        H = self.proj[region_idx](feats) + self.pos_mlp[region_idx](pos)
        N = H.size(0)
        if N > 1:
            if A is not None:
                H = self.sparse_diffusion(H, A, region_idx)
            for layer in self.agt_layers[region_idx]:
                H = layer(H)
            w = torch.softmax(self.attn_pool[region_idx](H).squeeze(-1), dim=0)
            pooled = (w.unsqueeze(1) * H).sum(dim=0)
        else:
            pooled = H.squeeze(0)
        return pooled

    def forward_batch(self, feats_list, pos_list):
        """Standard forward (no ablation). A_list ignored — adjacency not used at
        inference time in the original extract_risks call."""
        device = next(self.parameters()).device
        slide_embeddings = []
        for si in range(len(feats_list)):
            region_embs = []
            for ri in range(len(feats_list[si])):
                f = feats_list[si][ri]
                p = pos_list[si][ri]
                if f is None:
                    region_embs.append(torch.zeros(self.d_model, device=device))
                    continue
                region_embs.append(self.forward_bag(f, p, None, ri))
            region_embs = torch.stack(region_embs)
            reg_3d = region_embs.unsqueeze(0)
            attn_out, _ = self.cross_region_attn(reg_3d, reg_3d, reg_3d, need_weights=False)
            region_embs = self.cross_region_norm(region_embs + attn_out.squeeze(0))
            w_raw = torch.relu(self.region_weights) + 0.01
            w = w_raw / w_raw.sum()
            slide_embeddings.append((w.unsqueeze(1) * region_embs).sum(dim=0))
        H = torch.stack(slide_embeddings)
        return self.head(H).squeeze(-1)

    def forward_batch_ablated(self, feats_list, pos_list, zeroed_region, renormalize):
        """Ablation forward: zero region zeroed_region before cross-region
        attention, then evaluate with/without renormalisation."""
        device = next(self.parameters()).device
        slide_embeddings = []
        for si in range(len(feats_list)):
            region_embs = []
            for ri in range(len(feats_list[si])):
                f = feats_list[si][ri]
                p = pos_list[si][ri]
                if f is None:
                    region_embs.append(torch.zeros(self.d_model, device=device))
                    continue
                region_embs.append(self.forward_bag(f, p, None, ri))
            region_embs = torch.stack(region_embs)

            # Zero ablated region BEFORE cross-region attention.
            region_embs[zeroed_region] = 0.0

            reg_3d = region_embs.unsqueeze(0)
            attn_out, _ = self.cross_region_attn(reg_3d, reg_3d, reg_3d, need_weights=False)
            region_embs = self.cross_region_norm(region_embs + attn_out.squeeze(0))

            # Zero again after LayerNorm so position r contributes nothing.
            region_embs[zeroed_region] = 0.0

            w_raw = torch.relu(self.region_weights) + 0.01
            w = w_raw / w_raw.sum()

            if renormalize:
                w_abl = w.clone()
                w_abl[zeroed_region] = 0.0
                w_sum = w_abl.sum()
                if w_sum > 1e-8:
                    w_abl = w_abl / w_sum
                slide_emb = (w_abl.unsqueeze(1) * region_embs).sum(dim=0)
            else:
                # w still sums to 1; region r's contribution is w[r]*0 = 0.
                slide_emb = (w.unsqueeze(1) * region_embs).sum(dim=0)

            slide_embeddings.append(slide_emb)
        H = torch.stack(slide_embeddings)
        return self.head(H).squeeze(-1)


# ── C-index ───────────────────────────────────────────────────────────────────

def fast_cindex(times, events, risks, tied_tol=1e-8):
    times = np.asarray(times); events = np.asarray(events, dtype=bool)
    risks = np.asarray(risks)
    n = len(times)
    if n <= 1:
        return 0.0
    risks_q = np.round(risks / tied_tol) * tied_tol if tied_tol > 0 else risks
    order = np.argsort(-times)
    times = times[order]; events = events[order]; risks_q = risks_q[order]
    uniq = np.unique(risks_q)
    ranks = np.searchsorted(uniq, risks_q) + 1
    m = len(uniq)
    bit = np.zeros(m + 1, dtype=np.int64)

    def bit_add(i, v):
        while i <= m:
            bit[i] += v; i += i & -i

    def bit_sum(i):
        s = 0
        while i > 0:
            s += bit[i]; i -= i & -i
        return s

    ci_num = ci_den = 0.0
    i = 0
    while i < n:
        t = times[i]; j = i
        while j < n and times[j] == t:
            j += 1
        block_ranks = ranks[i:j]; block_events = events[i:j]
        total_later = bit_sum(m)
        if total_later > 0:
            for k in range(i, j):
                if not events[k]:
                    continue
                r = ranks[k]; less = bit_sum(r - 1); equal = bit_sum(r) - less
                ci_num += less + 0.5 * equal; ci_den += total_later
        cens_mask = ~block_events
        if cens_mask.any():
            cens_ranks = block_ranks[cens_mask]
            cnt = np.bincount(cens_ranks, minlength=m + 1)
            pref = np.cumsum(cnt)
            n_cens = cens_ranks.size
            for k in range(i, j):
                if not events[k]:
                    continue
                r = ranks[k]
                less = pref[r - 1] if r - 1 >= 0 else 0
                ci_num += less + 0.5 * cnt[r]; ci_den += n_cens
        for k in range(i, j):
            bit_add(ranks[k], 1)
        i = j
    return ci_num / max(ci_den, 1.0)


# ── Training ──────────────────────────────────────────────────────────────────

def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]; event = event[order]; time = time[order]
    lse = torch.logcumsumexp(risk, dim=0)
    loglik = risk - lse
    return -(loglik * event).sum() / max(event.sum(), 1.0)


def train_one_round(model, train_loader, device, epochs, save_root, round_id, lr=3e-5):
    model.to(device).train()
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if n != "region_weights"],
             "weight_decay": 1e-3},
            {"params": [model.region_weights], "lr": 1e-3, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    warmup = 3; total_steps = epochs * len(train_loader)
    warmup_steps = warmup * len(train_loader)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, 0.33, 1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=5e-6),
        ],
        milestones=[warmup_steps],
    )

    val_cidx_by_epoch = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feats_list, pos_list, t, e, paths, ntiles in train_loader:
            t = torch.tensor(t, dtype=torch.float32, device=device)
            e = torch.tensor(e, dtype=torch.float32, device=device)
            risk = model.forward_batch(feats_list, pos_list)
            w_raw = torch.relu(model.region_weights) + 0.01
            w = w_raw / w_raw.sum()
            entropy = -(w * torch.log(w + 1e-8)).sum()
            loss = cox_ph_loss(risk, t, e) + 1e-3 * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss.detach())
            del feats_list, pos_list, t, e, risk, loss
        ckpt = os.path.join(save_root, f"round_{round_id}_epoch_{ep}.pt")
        torch.save(model.state_dict(), ckpt)
        # We record NaN as placeholder; OOB c-index is computed later.
        val_cidx_by_epoch.append(float("nan"))
        print(f"  epoch {ep}/{epochs}  loss={total_loss:.4f}")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return val_cidx_by_epoch


# ── OOB evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def oob_cindex(model, loader, device):
    model.eval()
    Ts, Es, Rs = [], [], []
    for feats_list, pos_list, t, e, paths, ntiles in loader:
        risks = model.forward_batch(feats_list, pos_list).detach().cpu().numpy()
        Rs.extend(risks.tolist()); Ts.extend(t.tolist()); Es.extend(e.tolist())
    return fast_cindex(Ts, Es, Rs)


@torch.no_grad()
def oob_cindex_ablated(model, loader, device, zeroed_region, renormalize):
    model.eval()
    Ts, Es, Rs = [], [], []
    for feats_list, pos_list, t, e, paths, ntiles in loader:
        risks = model.forward_batch_ablated(
            feats_list, pos_list, zeroed_region, renormalize
        ).detach().cpu().numpy()
        Rs.extend(risks.tolist()); Ts.extend(t.tolist()); Es.extend(e.tolist())
    return fast_cindex(Ts, Es, Rs)


# ── Main ablation loop ────────────────────────────────────────────────────────

def run_ablation(dataset, num_regions, region_names, rounds, train_frac,
                 batch_size, epochs, save_root, cohort, out_path, lr=3e-5):
    os.makedirs(save_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    N = len(dataset)
    idx = np.arange(N)
    MIN_VAL_EVENTS = 20
    MAX_SEED_RETRIES = 20

    # Per-region per-round results: list of dicts
    per_region_rounds = {r: [] for r in range(num_regions)}
    baseline_rounds = []

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        print(f"\n=== Round {rnd}/{rounds} ===")

        # ── Determine OOB split (same seed logic as original training) ────────
        seed_attempt = rnd
        retry = 0
        while True:
            np.random.seed(seed_attempt)
            random.seed(seed_attempt)
            torch.manual_seed(seed_attempt)
            train_idx = np.random.choice(idx, int(train_frac * N), replace=True)
            mask = np.ones(N, bool); mask[train_idx] = False
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
                print(f"  [WARN] Could not find seed with >={MIN_VAL_EVENTS} OOB events.")
                break
            seed_attempt = rnd * 1000 + retry

        print(f"  OOB N={len(oob_idx)}, OOB events={val_events}")

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx.tolist()),
            batch_size=batch_size, shuffle=True,
            num_workers=4, prefetch_factor=2, collate_fn=collate_bags,
        )
        oob_loader = DataLoader(
            torch.utils.data.Subset(dataset, oob_idx.tolist()),
            batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_bags,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model = IPGGraphFormer(num_regions=num_regions)
        train_one_round(model, train_loader, device, epochs, save_root, rnd, lr=lr)

        # ── Pick best epoch by OOB c-index ────────────────────────────────────
        best_ep, best_c = 1, -1.0
        for ep in range(1, epochs + 1):
            ckpt = os.path.join(save_root, f"round_{rnd}_epoch_{ep}.pt")
            m = IPGGraphFormer(num_regions=num_regions)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.to(device)
            c = oob_cindex(m, oob_loader, device)
            print(f"    epoch {ep} OOB c-index={c:.4f}")
            if c > best_c:
                best_c = c; best_ep = ep
            del m
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        print(f"  Best epoch: {best_ep} (OOB c-index={best_c:.4f})")

        # ── Load best epoch model ─────────────────────────────────────────────
        best_ckpt = os.path.join(save_root, f"round_{rnd}_epoch_{best_ep}.pt")
        model_best = IPGGraphFormer(num_regions=num_regions)
        model_best.load_state_dict(torch.load(best_ckpt, map_location=device))
        model_best.to(device)

        baseline = oob_cindex(model_best, oob_loader, device)
        baseline_rounds.append(float(baseline))
        print(f"  Baseline OOB c-index (best epoch): {baseline:.4f}")

        # ── Ablation per region ───────────────────────────────────────────────
        for ri in range(num_regions):
            c_no = float(oob_cindex_ablated(model_best, oob_loader, device,
                                             zeroed_region=ri, renormalize=False))
            c_re = float(oob_cindex_ablated(model_best, oob_loader, device,
                                             zeroed_region=ri, renormalize=True))
            per_region_rounds[ri].append({
                "round": rnd,
                "baseline": float(baseline),
                "ablated_no_renorm": c_no,
                "ablated_renorm": c_re,
                "delta_no_renorm": c_no - float(baseline),
                "delta_renorm": c_re - float(baseline),
            })
            print(f"    [{region_names[ri][:40]}]  "
                  f"no_renorm={c_no:.4f} (Δ{c_no-baseline:+.4f})  "
                  f"renorm={c_re:.4f} (Δ{c_re-baseline:+.4f})")

        # ── Cleanup ───────────────────────────────────────────────────────────
        del model, model_best
        for ep in range(1, epochs + 1):
            p = os.path.join(save_root, f"round_{rnd}_epoch_{ep}.pt")
            if os.path.exists(p):
                os.remove(p)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"  Round {rnd} time: {time.time()-t0:.0f}s")

    # ── Aggregate across rounds ───────────────────────────────────────────────
    def summarise(vals):
        a = np.array(vals)
        n = len(a)
        mean = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ci95 = [mean - 1.96 * se, mean + 1.96 * se]
        return {"mean": mean, "sd": float(np.std(a, ddof=1)), "se": se,
                "ci95": ci95, "n_rounds": n}

    per_region_summary = {}
    for ri in range(num_regions):
        rname = region_names[ri]
        deltas_no = [d["delta_no_renorm"] for d in per_region_rounds[ri]]
        deltas_re = [d["delta_renorm"] for d in per_region_rounds[ri]]
        per_region_summary[rname] = {
            "delta_no_renorm": summarise(deltas_no),
            "delta_renorm": summarise(deltas_re),
            "rounds": per_region_rounds[ri],
        }

    result = {
        "cohort": cohort,
        "n_rounds": rounds,
        "epochs_per_round": epochs,
        "train_frac": train_frac,
        "n_slides": len(dataset),
        "num_regions": num_regions,
        "region_names": region_names,
        "ablation_method": (
            "Region r's embedding is zeroed after per-region processing "
            "(projection + graph-diffusion + attention pooling) and before "
            "cross-region self-attention, so region r also does not participate "
            "in other regions' cross-region attention. After cross-region "
            "LayerNorm, position r is zeroed again. "
            "no_renorm: original normalised weights kept (w still sums to 1, "
            "region r contributes w[r]*0=0). "
            "renorm: w[r] forced to 0 and remaining weights renormalised to sum to 1."
        ),
        "evaluation": "out-of-bag (OOB) patients only; in-sample c-index is not reported",
        "baseline_cindex": summarise(baseline_rounds),
        "per_region": per_region_summary,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n>>> Saved: {out_path}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

BLCA_REGIONS = [
    "Perivesical adipose tissue (extravesical fat)",
    "Invasive urothelial carcinoma (tumor)",
    "Normal urothelium (benign mucosa)",
    "Inflammatory infiltrates (immune cells)",
    "Lamina propria (fibrovascular stroma)",
    "Blood vessels (vasculature)",
    "Muscularis propria (detrusor muscle)",
    "Necrosis",
]

BRCA_REGIONS = [
    "Adipose tissue (fat)",
    "Blood vessels (vasculature)",
    "Ductal carcinoma in situ (DCIS)",
    "Fibrous desmoplastic stroma",
    "Invasive breast carcinoma (tumor cells)",
    "Muscle tissue (smooth or skeletal muscle)",
    "Necrosis or hemorrhage",
    "Normal breast glands and lobules (TDLU)",
    "Tumor-infiltrating lymphocytes (immune infiltrates)",
]

BLCA_BASE = "/home/sorkwos/links/scratch/UNI2_classwise_embeddings"
BRCA_BASE = "/home/sorkwos/links/scratch/UNI2_classwise_embeddings_BRCA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class_dir", action="append", default=None,
                    help="Override default class dirs (repeat for each region).")
    ap.add_argument("--cdr-xlsx", default=None,
                    help="Path to TCGA-CDR Excel. Defaults to Trillium location.")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15,
                    help="Epochs per round. Original training used 30; 15 is enough for ablation.")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save-root", default=None,
                    help="Temp dir for per-round .pt checkpoints. Cleaned up after each round.")
    ap.add_argument("--out", required=True,
                    help="Output JSON path, e.g. results/region_ablation_blca.json")
    args = ap.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    if args.cdr_xlsx is None:
        # Trillium default
        args.cdr_xlsx = (
            "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/"
            "TCGA-CDR-SupplementalTableS1.xlsx"
        )

    if args.class_dir is not None:
        class_dirs = args.class_dir
    elif args.cohort == "blca":
        class_dirs = [os.path.join(BLCA_BASE, r) for r in BLCA_REGIONS]
    else:
        class_dirs = [os.path.join(BRCA_BASE, r) for r in BRCA_REGIONS]

    if args.save_root is None:
        args.save_root = f"/scratch/sorkwos/ablation_tmp_{args.cohort}"

    region_names = [os.path.basename(d.rstrip("/")) for d in class_dirs]
    num_regions = len(class_dirs)

    # ── Survival table ────────────────────────────────────────────────────────
    cohort_filter = "BLCA" if args.cohort == "blca" else "BRCA"
    print(f">>> Loading survival table from {args.cdr_xlsx}")
    surv_df = pd.read_excel(args.cdr_xlsx)
    surv_df = surv_df[surv_df["type"].str.contains(cohort_filter, na=False)]
    surv_df = surv_df[["bcr_patient_barcode", "PFI", "PFI.time"]]
    surv_df["PFI.time"] = pd.to_numeric(surv_df["PFI.time"], errors="coerce")
    surv_df = surv_df.dropna(subset=["PFI.time"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f">>> Building {args.cohort.upper()} dataset ({num_regions} regions)...")
    dataset = MultiRegionDataset(class_dirs, surv_df)
    print(f">>> Slides: {len(dataset)}")

    set_seed(1337)

    run_ablation(
        dataset=dataset,
        num_regions=num_regions,
        region_names=region_names,
        rounds=args.rounds,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_root=args.save_root,
        cohort=args.cohort,
        out_path=args.out,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
