#!/usr/bin/env python3
# interpret_graphtrans_12.py
# Using precomputed KNN graph (adjacency matrix) to speed up the model
# Precomputing graph is done in precompute_knn_graphs.py

print(">>> Script started.")

import os
job_id = os.environ.get("SLURM_JOB_ID", "interactive")
SAVE_ROOT = None  # set from --save_root argument in main()
import sys
import random
import time
import argparse
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pycox.evaluation import EvalSurv
import hnswlib
import types
import shutil   # <-- ADDED (ONLY NEW IMPORT)
from datetime import datetime

LOG_F = None

# ============================================================
# SLURM TMPDIR STAGING (ADDED)
# ============================================================

SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR", None)
STAGED_ADJ_DIR = None


'''
def stage_adjacency_matrices(slide_paths):
    global STAGED_ADJ_DIR

    if SLURM_TMPDIR is None:
        print(">>> SLURM_TMPDIR not set — using Lustre adjacency files.")
        return

    STAGED_ADJ_DIR = os.path.join(SLURM_TMPDIR, "adjacency_matrices")
    os.makedirs(STAGED_ADJ_DIR, exist_ok=True)

    print(f">>> Staging adjacency matrices to {STAGED_ADJ_DIR}")
    t0 = time.time()
    copied = 0

    for path in slide_paths:
        src = f"{os.path.splitext(path)[0]}_A.pt"
        if not os.path.exists(src):
            continue
        dst = os.path.join(STAGED_ADJ_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    print(f">>> Staged {copied} adjacency matrices in {time.time() - t0:.2f} sec")

def stage_adjacency_matrices(slide_paths):
    global STAGED_ADJ_DIR

    if SLURM_TMPDIR is None:
        print(">>> SLURM_TMPDIR not set — using Lustre adjacency files.")
        return

    TAR_SRC = "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/adjacency_matrices.tar"
    if not os.path.exists(TAR_SRC):
        print(">>> adjacency_matrices.tar not found — using Lustre adjacency files.")
        return

    STAGED_ADJ_DIR = os.path.join(SLURM_TMPDIR, "adjacency_matrices")
    os.makedirs(STAGED_ADJ_DIR, exist_ok=True)

    marker = os.path.join(STAGED_ADJ_DIR, ".EXTRACTED")
    if os.path.exists(marker):
        print(">>> Adjacency matrices already staged.")
        return

    print(f">>> Staging adjacency matrices via TAR to {STAGED_ADJ_DIR}")
    t0 = time.time()

    tar_dst = os.path.join(SLURM_TMPDIR, "adjacency_matrices.tar")
    shutil.copyfile(TAR_SRC, tar_dst)

    import tarfile
    with tarfile.open(tar_dst, "r") as tar:
        tar.extractall(STAGED_ADJ_DIR)

    open(marker, "w").close()  # extraction marker

    print(f">>> Staged adjacency matrices in {time.time() - t0:.2f} sec")

'''
def stage_adjacency_matrices(slide_paths, class_dir):

    global STAGED_ADJ_DIR

    if SLURM_TMPDIR is None:
        print(">>> SLURM_TMPDIR not set — using Lustre adjacency files.")
        return

    #TAR_SRC = "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/adjacency_matrices.tar"

    class_name = os.path.basename(class_dir.rstrip("/"))
    TAR_SRC = os.path.join(
        os.path.dirname(class_dir),
        f"{class_name}.tar"
    )


    if not os.path.exists(TAR_SRC):
        print(">>> adjacency_matrices.tar not found — using Lustre adjacency files.")
        return

    # infer class folder name from the slides you loaded (same as --class_dir basename)
    if not slide_paths:
        print(">>> No slide paths provided — using Lustre adjacency files.")
        return
    class_name = os.path.basename(os.path.dirname(slide_paths[0]))

    STAGED_ADJ_DIR = os.path.join(SLURM_TMPDIR, "adjacency_matrices")
    os.makedirs(STAGED_ADJ_DIR, exist_ok=True)

    marker = os.path.join(STAGED_ADJ_DIR, f".EXTRACTED_{class_name}")
    if os.path.exists(marker):
        print(">>> Adjacency matrices already staged for this class.")
        return

    # only extract what we need for this run (saves time + avoids huge staging)
    needed = {os.path.basename(os.path.splitext(p)[0]) + "_A.pt" for p in slide_paths}

    print(f">>> Staging adjacency matrices (class='{class_name}') via TAR to {STAGED_ADJ_DIR}")
    t0 = time.time()

    tar_dst = os.path.join(SLURM_TMPDIR, "adjacency_matrices.tar")
    shutil.copyfile(TAR_SRC, tar_dst)

    staged = 0
    '''
    import tarfile
    with tarfile.open(tar_dst, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if not base.endswith("_A.pt"):
                continue
            if base not in needed:
                continue

            # ensure the member belongs to the requested class folder, regardless of tar layout:
            #   adjacency_matrices/<class>/..., <class>/..., or even absolute-ish paths
            parts = m.name.split("/")
            if class_name not in parts:
                continue

            m.name = base  # flatten into STAGED_ADJ_DIR
            tar.extract(m, STAGED_ADJ_DIR)
            staged += 1
    '''


    import tarfile
    with tarfile.open(tar_dst, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue

            base = os.path.basename(m.name)

            # only adjacency matrices
            if not base.endswith("_A.pt"):
                continue

            # only the ones needed for this run
            if base not in needed:
                continue

            # flatten into STAGED_ADJ_DIR
            m.name = base
            tar.extract(m, STAGED_ADJ_DIR)
            staged += 1


    open(marker, "w").close()
    print(f">>> Staged {staged} adjacency matrices in {time.time() - t0:.2f} sec")


def resolve_adj_path(slide_path):
    name = os.path.basename(os.path.splitext(slide_path)[0] + "_A.pt")
    if STAGED_ADJ_DIR is not None:
        local = os.path.join(STAGED_ADJ_DIR, name)
        if os.path.exists(local):
            return local
    p = f"{os.path.splitext(slide_path)[0]}_A.pt"
    if os.path.exists(p):
        return p
    return None


def cleanup_staged_adjacency():
    global STAGED_ADJ_DIR
    if STAGED_ADJ_DIR and os.path.exists(STAGED_ADJ_DIR):
        print(f">>> Cleaning up staged adjacency matrices: {STAGED_ADJ_DIR}")
        shutil.rmtree(STAGED_ADJ_DIR, ignore_errors=True)
        STAGED_ADJ_DIR = None


def log_result(msg: str):
    global LOG_F
    if LOG_F is not None:
        LOG_F.write(msg + "\n")
        LOG_F.flush()


# ---- NumPy pickle shim ----
fake_core = types.ModuleType("numpy._core")
fake_core.multiarray = np.core.multiarray
fake_core.umath = np.core.umath
sys.modules["numpy._core"] = fake_core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.umath"] = np.core.umath
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

print(">>> NumPy shim loaded.")

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 1337):
    print(">>> Setting seed.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_xy(coords_np: np.ndarray):
    if coords_np.size == 0:
        return coords_np.astype(np.float32)
    xs = coords_np[:, 0].astype(np.float32)
    ys = coords_np[:, 1].astype(np.float32)
    xs = (xs - xs.min()) / max(xs.max() - xs.min(), 1.0)
    ys = (ys - ys.min()) / max(ys.max() - ys.min(), 1.0)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def list_pt_files_recursive(root_dir: str):
    print(f">>> Recursively scanning: {root_dir}")
    pt = []
    for r, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".pt") and "DX1" in name and not name.endswith("_A.pt"):
                pt.append(os.path.join(r, name))
    print(f">>> Found {len(pt)} candidate .pt files.")
    return sorted(pt)


# ============================================================
# Dataset
# ============================================================

class MultiRegionDataset(Dataset):

    def __init__(self, class_dirs, surv_df):

        print(f">>> Loading dataset from {len(class_dirs)} region folders")
        t0 = time.time()

        self.region_dirs = class_dirs
        self.region_names = [os.path.basename(d.rstrip("/")) for d in class_dirs]

        self.samples = []
        self.labels = []

        ref_dir = class_dirs[0]

        if any(name.endswith(".pt") for name in os.listdir(ref_dir)):
            candidate = [
                name for name in os.listdir(ref_dir)
                if name.endswith(".pt") and "DX1" in name and not name.endswith("_A.pt")
            ]
        else:
            candidate = [
                os.path.basename(p) for p in list_pt_files_recursive(ref_dir)
            ]

        print(f">>> Found {len(candidate)} reference slides")

        for slide_name in candidate:

            case_id = slide_name[:12]
            row = surv_df.loc[surv_df["bcr_patient_barcode"] == case_id]

            if row.empty:
                continue

            region_paths = {}

            for region_name, region_dir in zip(self.region_names, self.region_dirs):

                p = os.path.join(region_dir, slide_name)

                if os.path.exists(p):
                    region_paths[region_name] = p
                else:
                    region_paths[region_name] = None

            t = float(row["PFI.time"].values[0])
            e = int(row["PFI"].values[0])

            self.samples.append(region_paths)
            self.labels.append((t, e))

        print(f">>> Loaded {len(self.samples)} slides with survival labels.")
        print(f">>> Dataset load time: {time.time() - t0:.2f} sec")

        if not self.samples:
            raise RuntimeError("No slides matched survival table")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        region_paths = self.samples[idx]
        time_t, event = self.labels[idx]

        return region_paths, torch.tensor(time_t), torch.tensor(event)

'''
    def __getitem__(self, idx):
        path = self.paths[idx]
        d = torch.load(path, map_location="cpu", weights_only=False)

        coords = d["coords"]
        coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords
        feats = d["features"].float()

        if idx == 0:
            print(f">>> Example slide '{os.path.basename(path)}' contains {feats.shape[0]} tiles.")

        pos = torch.from_numpy(normalize_xy(coords_np))
        time_t, event = self.labels[idx]
        return feats, pos, torch.tensor(time_t), torch.tensor(event), path
'''
'''
def collate_bags(batch):
    feats_list, pos_list, times, events = [], [], [], []
    paths = []
    for feats, pos, t, e, path in batch:
        feats_list.append(feats)
        pos_list.append(pos)
        times.append(t.item())
        events.append(int(e.item()))
        paths.append(path)

    return feats_list, pos_list, np.array(times), np.array(events), paths
'''
def collate_bags(batch):

    feats_list, pos_list = [], []
    times, events = [], []
    paths_list = []
    ntiles_list = []

    for region_paths, t, e in batch:

        feats_regions = []
        pos_regions = []
        paths_regions = []
        ntiles_regions = []

        for region_name, path in region_paths.items():

            if path is None:
                feats_regions.append(None)
                pos_regions.append(None)
                paths_regions.append(None)
                ntiles_regions.append(0)
                continue

            #d = torch.load(path, map_location="cpu", weights_only=False)
            d = load_slide_cached(path)
            coords = d["coords"]
            if isinstance(coords, torch.Tensor):
                coords_np = coords.cpu().numpy()
            else:
                coords_np = np.asarray(coords)

            feats = d["features"]
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)

            feats = feats.to(torch.float32).contiguous()
            pos = torch.from_numpy(normalize_xy(coords_np)).to(torch.float32).contiguous()

            feats_regions.append(feats)
            pos_regions.append(pos)
            paths_regions.append(path)
            ntiles_regions.append(feats.shape[0])

        feats_list.append(feats_regions)
        pos_list.append(pos_regions)
        paths_list.append(paths_regions)
        ntiles_list.append(ntiles_regions)

        times.append(t.item())
        events.append(int(e.item()))

    return feats_list, pos_list, np.array(times), np.array(events), paths_list, ntiles_list


# ============================================================
# Sparse graph (precomputed on disk)
# ============================================================

def load_precomputed_knn_graph(path: str):
    data = torch.load(path, map_location="cpu")
    A = data["A"]
    if A.is_sparse:
        A = A.coalesce()
    return A


@lru_cache(maxsize=512)
def cached_graph(path: str):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return load_precomputed_knn_graph(path)

from functools import lru_cache

#@lru_cache(maxsize=4096)
def load_slide_cached(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================
# OOM-safe runner (ADDED)
# ============================================================

def run_with_oom_split(fn, feats_list, pos_list, A_list, t=None, e=None, opt=None, device="cuda"):
    try:
        return fn(feats_list, pos_list, A_list, t, e)
    except RuntimeError as ex:
        if "out of memory" in str(ex).lower() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            if len(feats_list) == 1:
                return None
            mid = len(feats_list) // 2
            left = run_with_oom_split(
                fn,
                feats_list[:mid], pos_list[:mid], A_list[:mid],
                None if t is None else t[:mid],
                None if e is None else e[:mid],
                opt, device
            )
            right = run_with_oom_split(
                fn,
                feats_list[mid:], pos_list[mid:], A_list[mid:],
                None if t is None else t[mid:],
                None if e is None else e[mid:],
                opt, device
            )
            if left is None: return right
            if right is None: return left
            return torch.cat([left, right]) if isinstance(left, torch.Tensor) else left + right
        raise


# ============================================================
# AGT Layer
# ============================================================

class AGTLayer(nn.Module):
    def __init__(self, dim_in, nheads=6, emb_dropout=0.15):
        super().__init__()
        self.nheads = nheads
        self.dim_in = dim_in
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

        q = self.relu(q)
        k = self.relu(k)

        kv = torch.einsum("nhd,nhe->hde", k, v)
        num = torch.einsum("nhd,hde->nhe", q, kv)
        denom = torch.einsum("nhd,hd->n", q, k.sum(dim=0)) + 1e-6
        attn = torch.einsum("nhe,n->nhe", num, 1.0 / denom)

        h_new = self.linear_final(attn.reshape(N, self.dim_in))
        out = self.ln(h + self.dropout(h_new))
        return out


# ============================================================
# Model
# ============================================================

class IPGGraphFormer(nn.Module):
    def __init__(self, in_dim=1536, d_model=384, agt_heads=6, agt_layers=2,
                 gnn_hops=3, knn_k=16, hidden_mlp=256, dropout=0.2):
        super().__init__()
        print(">>> Initializing model.")
        self.gnn_hops = gnn_hops
        self.knn_k = knn_k

        num_regions = 8

        self.proj = nn.ModuleList([
            nn.Linear(in_dim, d_model) for _ in range(num_regions)
        ])

        self.pos_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(num_regions)
        ])

        self.beta = nn.ParameterList([
            nn.Parameter(torch.ones(gnn_hops + 1)) for _ in range(num_regions)
        ])

        self.agt_layers = nn.ModuleList([
            nn.ModuleList([
                AGTLayer(d_model, nheads=agt_heads, emb_dropout=dropout)
                for _ in range(agt_layers)
            ])
            for _ in range(num_regions)
        ])

        self.attn_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            ) for _ in range(num_regions)
        ])

        self.region_weights = nn.Parameter(torch.ones(num_regions))

        self.d_model = d_model

        # Cross-region self-attention: 1 layer, fully-connected over 8 region tokens.
        # Captures interactions between regions before the final weighted sum.
        # Interpretability: region_weights still tell you which region the model
        # selected; the 8x8 attention matrix tells you which regions co-interacted.
        self.cross_region_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,          # 4 heads × 96-dim per head = 384
            dropout=dropout,
            batch_first=True,
        )
        self.cross_region_norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 1)

    def sparse_diffusion(self, H0, A, region_idx):
            if A is None:
                return H0
            
            # Ensure A is coalesced and on the correct device
            if not A.is_coalesced():
                A = A.coalesce()
                
            beta = self.beta[region_idx]
            Z_prev = H0
            Z = beta[0] * Z_prev
            
            for s in range(1, len(beta)):
                # Explicitly check for potential issues before the sparse MM
                Z_prev = torch.sparse.mm(A, Z_prev)
                Z = Z + beta[s] * Z_prev
                
            return Z

    def forward_bag(self, feats, pos, A, region_idx):

            device = self.proj[region_idx].weight.device
            pos = pos.float().to(device)
            feats = feats.float().to(device)
            
            # 1. Initial Projection
            H = self.proj[region_idx](feats) + self.pos_mlp[region_idx](pos)
            
            # 2. BYPASS logic for tiny slides
            N = H.size(0)
            if N > 1:
                # Only do Graph Diffusion if A is provided
                if A is not None:
                    H = self.sparse_diffusion(H, A, region_idx)
                
                # Only do Attention if there is more than one tile to attend to
                for layer in self.agt_layers[region_idx]:
                    H = layer(H)
                
                # Pooling
                w = self.attn_pool[region_idx](H).squeeze(-1)
                w = torch.softmax(w, dim=0)
                pooled = (w.unsqueeze(1) * H).sum(dim=0)
            else:
                # N=1: No diffusion, no attention, no pooling needed
                pooled = H.squeeze(0)
            
            # 3. Final Head
            #out = self.head(pooled).squeeze()
            #return out
            return pooled

    def forward_batch(self, feats_list, pos_list, A_list, return_attn=False):

        t0 = time.time()
        print(f">>> Forward batch with {len(feats_list)} slides...")

        slide_embeddings = []
        slide_attn_weights = []   # (8, 8) per slide, collected when return_attn=True

        device = next(self.parameters()).device

        for slide_idx in range(len(feats_list)):

            region_embs = []

            for region_idx in range(len(feats_list[slide_idx])):

                f = feats_list[slide_idx][region_idx]
                p = pos_list[slide_idx][region_idx]
                A = None if A_list is None else A_list[slide_idx][region_idx]

                if f is None:
                    # Missing region: pad with zeros so all slides have 8 embeddings
                    region_embs.append(torch.zeros(self.d_model, device=device))
                    continue

                emb = self.forward_bag(f, p, A, region_idx)
                region_embs.append(emb)

            # stack region embeddings → shape (num_regions, d_model)
            region_embs = torch.stack(region_embs)  # (8, 384)

            # ---- Cross-region self-attention ----
            # Add batch dim: (1, 8, 384)
            reg_3d = region_embs.unsqueeze(0)
            attn_out, attn_w = self.cross_region_attn(
                reg_3d, reg_3d, reg_3d,
                need_weights=True,
                average_attn_weights=True,   # average over 4 heads → (1, 8, 8)
            )
            # Residual + LayerNorm
            region_embs = self.cross_region_norm(region_embs + attn_out.squeeze(0))  # (8, 384)

            if return_attn:
                # attn_w: (1, 8, 8) → (8, 8)
                slide_attn_weights.append(attn_w.squeeze(0).detach().cpu())

            # ---- Final weighted sum (interpretable region selection) ----
            # ReLU + floor (eps=0.01) + normalize: decouples weights so each
            # region can move freely, while the floor ensures no region ever
            # loses gradient signal entirely.
            w_raw = torch.relu(self.region_weights) + 0.01
            w = w_raw / w_raw.sum()
            slide_emb = (w.unsqueeze(1) * region_embs).sum(dim=0)  # (384,)

            slide_embeddings.append(slide_emb)

        H = torch.stack(slide_embeddings)

        risk = self.head(H).squeeze(-1)  # only squeeze last dim, preserves batch dim even for batch size 1

        print(f">>> forward_batch time: {time.time() - t0:.2f} sec")

        if return_attn:
            return risk, slide_attn_weights  # list of (8, 8) tensors, one per slide
        return risk

'''
    def forward_batch(self, feats_list, pos_list, A_list):
        t0 = time.time()
        print(f">>> Forward batch with {len(feats_list)} slides...")
        out = torch.stack([
            self.forward_bag(f, p, A) for f, p, A in zip(feats_list, pos_list, A_list)
        ])
        print(f">>> forward_batch time: {time.time() - t0:.2f} sec")
        return out
'''

# ============================================================
# Training + Main
# ============================================================

@torch.no_grad()
def extract_risks(model, loader, device, return_attn=False):
    """Extract per-patient risk scores. If return_attn=True, also returns
    the mean cross-region attention matrix (8x8) averaged over all patients."""
    model.eval()
    all_patient_ids = []
    all_risks = []
    all_attn = []   # list of (8, 8) tensors, one per patient

    for feats_list, pos_list, t, e, paths, ntiles in loader:
        A_list = None
        if return_attn:
            risks, batch_attn = model.forward_batch(feats_list, pos_list, A_list, return_attn=True)
            all_attn.extend(batch_attn)  # each is (8, 8)
        else:
            risks = model.forward_batch(feats_list, pos_list, A_list)
        risks = risks.detach().cpu().numpy()

        for i in range(len(paths)):
            pid = os.path.basename(paths[i][0])[:12] if paths[i][0] else f"missing_{i}"
            all_patient_ids.append(pid)
            all_risks.append(risks[i])

    if return_attn:
        # Average attention across all patients → (8, 8)
        mean_attn = torch.stack(all_attn).mean(dim=0).numpy()
        return all_patient_ids, all_risks, mean_attn
    return all_patient_ids, all_risks

def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    time = time[order]
    event = event[order]
    risk = risk[order]
    lse = torch.logcumsumexp(risk, dim=0)
    loglik = risk - lse
    return - (loglik * event).sum() / max(event.sum(), 1.0)


def fast_cindex(times, events, risks, tied_tol=1e-8):
    """
    Matches sksurv.metrics.concordance_index_censored behavior:
    - Comparable pairs: (i event) compared to (j later time) AND (j censored at same time)
    - Tied event times: include event vs censored-at-same-time comparisons
    - Tied risks: abs(diff) <= tied_tol counted as 0.5
    Assumes higher risk = worse outcome (earlier event).
    """
    import numpy as np

    times = np.asarray(times)
    events = np.asarray(events).astype(bool)
    risks = np.asarray(risks)

    n = len(times)
    if n <= 1:
        return 0.0

    if tied_tol is not None and tied_tol > 0:
        risks_q = np.round(risks / tied_tol) * tied_tol
    else:
        risks_q = risks

    order = np.argsort(-times)
    times = times[order]
    events = events[order]
    risks_q = risks_q[order]

    uniq = np.unique(risks_q)
    ranks = np.searchsorted(uniq, risks_q) + 1
    m = len(uniq)

    bit = np.zeros(m + 1, dtype=np.int64)

    def bit_add(i, v):
        while i <= m:
            bit[i] += v
            i += i & -i

    def bit_sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    ci_num = 0.0
    ci_den = 0.0

    i = 0
    while i < n:
        t = times[i]
        j = i
        while j < n and times[j] == t:
            j += 1

        block_ranks = ranks[i:j]
        block_events = events[i:j]

        total_later = bit_sum(m)
        if total_later > 0:
            for k in range(i, j):
                if not events[k]:
                    continue
                r = ranks[k]
                less = bit_sum(r - 1)
                leq = bit_sum(r)
                equal = leq - less
                ci_num += less + 0.5 * equal
                ci_den += total_later

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
                equal = cnt[r]
                ci_num += less + 0.5 * equal
                ci_den += n_cens

        for k in range(i, j):
            bit_add(ranks[k], 1)

        i = j

    return ci_num / max(ci_den, 1.0)

@torch.no_grad()
def eval_cindex_evalsurv(model, loader, device):
    t0 = time.time()
    model.eval()
    Ts, Es, Rs = [], [], []

    for feats_list, pos_list, t, e, paths, ntiles in loader:
        #feats_list = [x.to(device) for x in feats_list]
        #pos_list = [x.to(device) for x in pos_list]

        #A_list = []
        A_list = None


        def _eval_call(fl, pl, Al, *_):
            return model.forward_batch(fl, pl, Al)

        print(f"DEBUG: Batch slides: {[os.path.basename(p[0]) if p[0] else 'missing' for p in paths]}")
        risks_t = run_with_oom_split(_eval_call, feats_list, pos_list, A_list,
                                     None, None, None, device)
        if risks_t is not None:
            Rs.extend(np.atleast_1d(risks_t.detach().cpu().numpy()))
            Ts.extend(np.atleast_1d(t))
            Es.extend(np.atleast_1d(e))

        del feats_list, pos_list, A_list, risks_t

    out = fast_cindex(Ts, Es, Rs)
    print(f">>> eval_cindex time: {time.time() - t0:.2f} sec")
    return out

def train_one_round(model, train_loader, val_loader, device, round_id, epochs=8, lr=3e-5):
    print(">>> Training round start.")
    writer = SummaryWriter(f"/scratch/sorkwos/tensorboard_logs/job_{job_id}/round_{round_id}")
    t_round = time.time()
    model.to(device)
    #opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if n != "region_weights"], "weight_decay": 1e-3},
            {"params": [model.region_weights], "lr": 1e-3, "weight_decay": 0.0},
        ],
        lr=lr
    )
    print("LR:", opt.param_groups[0]["lr"], "WD:", opt.param_groups[0].get("weight_decay"))
    warmup_epochs = 3
    decay_start_epoch = 12

    steps_per_epoch = len(train_loader)

    warmup_steps = warmup_epochs * steps_per_epoch
    hold_steps = (decay_start_epoch - 1 - warmup_epochs) * steps_per_epoch
    decay_steps = (epochs - (decay_start_epoch - 1)) * steps_per_epoch

    sched_warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.33, end_factor=1.0, total_iters=warmup_steps
    )

    sched_hold = torch.optim.lr_scheduler.ConstantLR(
        opt, factor=1.0, total_iters=hold_steps
    )

    sched_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=decay_steps, eta_min=5e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[sched_warmup, sched_hold, sched_decay],
        milestones=[warmup_steps, warmup_steps + hold_steps]
    )

    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs - warmup_epochs,
        eta_min=5e-6
    )

    valC_by_epoch = []
    loss_by_epoch = []

    for ep in range(1, epochs + 1):
        print(f">>> Epoch {ep} start.")
        model.train()
        total = 0.0

        t_wait = time.time()
        for feats_list, pos_list, t, e, paths, ntiles in train_loader:
            print(f"[DL] wait_next_batch={time.time()-t_wait:.2f}s"); t_wait = time.time()
            #feats_list = [x.to(device) for x in feats_list]
            #pos_list = [x.to(device) for x in pos_list]
            t = torch.tensor(t, dtype=torch.float32, device=device)
            e = torch.tensor(e, dtype=torch.float32, device=device)

            tA0 = time.time()
            #A_paths = [resolve_adj_path(path) for path in paths]
            #print(f"[A] sample={A_paths[0]}")

            #tA1 = time.time()
            #A_cpu = [cached_graph(ap) for ap in A_paths]
            #print(f"[A] load_cpu={time.time()-tA1:.2f}s")

            A_paths = []
            for slide_paths in paths:
                slide_adj = []
                for p in slide_paths:
                    slide_adj.append(resolve_adj_path(p) if p else None)
                A_paths.append(slide_adj)

            print(f"[A] sample={A_paths[0]}")

            A_cpu = []
            for slide_adj in A_paths:
                slide_cpu = []
                for ap in slide_adj:
                    slide_cpu.append(cached_graph(ap) if ap else None)
                A_cpu.append(slide_cpu)
#new shit
            # RIGHT HERE, after:
            # A_cpu = [cached_graph(ap) for ap in A_paths]

            bad = []
            #for i, (f, ap, A) in enumerate(zip(feats_list, A_paths, A_cpu)):
            for slide_idx in range(len(feats_list)):
                for region_idx in range(len(feats_list[slide_idx])):

                    f = feats_list[slide_idx][region_idx]
                    ap = A_paths[slide_idx][region_idx]
                    A = A_cpu[slide_idx][region_idx]

                    if f is None:
                        continue
                    # 1. Check features for NaNs/Infs
                    if not torch.isfinite(f).all():
                        print(f"BAD_FEATS_NONFINITE: slide={slide_idx} region={region_idx} file={os.path.basename(paths[slide_idx][region_idx])}")
                        bad.append((slide_idx, region_idx))
                        continue                    

                    if A is None:
                        continue

                    # 2. Shape Mismatch Check
                    N = f.shape[0]
                    if A.shape[0] != N or A.shape[1] != N:
                        print(f"BAD_ADJ_SHAPE: slide={slide_idx} region={region_idx} file={os.path.basename(paths[slide_idx][region_idx])} N={N} A={tuple(A.shape)}")
                        bad.append((slide_idx, region_idx))
                        continue

                    # 3. CRITICAL: Out-of-bounds Index Check
                    if A.is_sparse:
                        A = A.coalesce()
                        Ai = A.indices()
                        if Ai.numel() > 0:
                            mx = int(Ai.max().item())
                            mn = int(Ai.min().item())
                            if mn < 0 or mx >= N:
                                print(f"!!! INDEX OVERFLOW: slide={slide_idx} region={region_idx} file={os.path.basename(paths[slide_idx][region_idx])} N={N} min_idx={mn} max_idx={mx}")
                                bad.append((slide_idx, region_idx))
                                continue

#new shit ends

            tA2 = time.time()
            # TO THIS:
            '''
            A_list = []
            for idx, A in enumerate(A_cpu):
                if A is None or ntiles[idx] < 10:
                    A_list.append(None)
                else:
                    A_list.append(A.to(device, non_blocking=True))
            '''
            A_list = []

            for slide_idx in range(len(A_cpu)):

                slide_A = []

                for region_idx in range(len(A_cpu[slide_idx])):

                    A = A_cpu[slide_idx][region_idx]

                    if A is None or ntiles[slide_idx][region_idx] < 10:
                        slide_A.append(None)
                    else:
                        slide_A.append(A.to(device, non_blocking=True))

                A_list.append(slide_A)            
            torch.cuda.synchronize()
            print(f"[A] to_gpu={time.time()-tA2:.2f}s (total {time.time()-tA0:.2f}s)")

            '''
            def _train_call(fl, pl, Al, tt, ee):
                r = model.forward_batch(fl, pl, Al)
                loss = cox_ph_loss(r, tt, ee)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return loss.item()
            '''

            def _train_call(fl, pl, Al, tt, ee):
                #print("BATCH:", [os.path.basename(p) for p in paths])
                print("BATCH:", [os.path.basename(p[0]) if p[0] else "missing" for p in paths])
                # -------------------------
                # 1) FORWARD (isolate if crash)
                # -------------------------
                try:
                    r = model.forward_batch(fl, pl, Al)
                    torch.cuda.synchronize()
                except Exception:
                    print("\nFORWARD CRASH — isolating offending slide/region via per-region forward...")

                    for si in range(len(fl)):  # slides in batch
                        for ri in range(len(fl[si])):  # regions
                            f = fl[si][ri]
                            p = pl[si][ri]
                            A = None if Al is None else Al[si][ri]

                            if f is None:
                                continue

                            try:
                                _ = model.forward_bag(f, p, A)  # returns pooled embedding (384,)
                                torch.cuda.synchronize()
                            except Exception:
                                bad_path = paths[si][ri] if paths[si][ri] else None
                                print("OFFENDING:", f"slide={si}", f"region={ri}", f"file={os.path.basename(bad_path) if bad_path else None}")

                                ap = resolve_adj_path(bad_path) if bad_path else None
                                print("OFFENDING_ADJ:", os.path.basename(ap) if ap else None)

                                N = f.shape[0]
                                print("N_tiles:", N,
                                    "A_is_none:", (A is None),
                                    "A.shape:", None if A is None else tuple(A.shape),
                                    "A.is_sparse:", None if A is None else bool(A.is_sparse),
                                    "A.dtype:", None if A is None else A.dtype,
                                    "A.device:", None if A is None else A.device)

                                if A is not None and A.is_sparse:
                                    Ai = A.coalesce().indices()
                                    if Ai.numel():
                                        mn = int(Ai.min().item())
                                        mx = int(Ai.max().item())
                                        print("ADJ_INDEX_RANGE:", "min=", mn, "max=", mx, "N=", N)

                                raise
                    raise

                # -------------------------
                # 2) LOSS + BACKWARD (isolate if crash)
                # -------------------------

                w_raw = torch.relu(model.region_weights) + 0.01
                w = w_raw / w_raw.sum()
                entropy = -(w * torch.log(w + 1e-8)).sum()
                loss = cox_ph_loss(r, tt, ee) + 1e-3 * entropy
                opt.zero_grad(set_to_none=True)

                try:
                    loss.backward()
                    torch.cuda.synchronize()
                except Exception:
                    print("\nBACKWARD CRASH — isolating offending slide/region via per-region backward...")

                    for si in range(len(fl)):
                        for ri in range(len(fl[si])):
                            f = fl[si][ri]
                            p = pl[si][ri]
                            A = None if Al is None else Al[si][ri]

                            if f is None:
                                continue

                            try:
                                opt.zero_grad(set_to_none=True)

                                # forward_bag returns pooled embedding (384,) so make a scalar
                                emb = model.forward_bag(f, p, A)
                                out = model.head(emb).squeeze()
                                torch.cuda.synchronize()

                                out.backward()
                                torch.cuda.synchronize()

                            except Exception:
                                bad_path = paths[si][ri] if paths[si][ri] else None
                                print("OFFENDING:", f"slide={si}", f"region={ri}", f"file={os.path.basename(bad_path) if bad_path else None}")
                                ap = resolve_adj_path(bad_path) if bad_path else None
                                print("OFFENDING_ADJ:", os.path.basename(ap) if ap else None)
                                print("N_tiles:", f.shape[0],
                                    "A.shape:", None if A is None else tuple(A.shape),
                                    "A.is_sparse:", None if A is None else bool(A.is_sparse))
                                raise

                    print("Per-region backward all OK (so crash is likely in Cox loss / batch interaction).")
                    raise
                    
                opt.step()
                scheduler.step()
                return loss.item()



            loss_val = run_with_oom_split(_train_call, feats_list, pos_list, A_list,
                                          t, e, opt, device)
            if loss_val is not None:
                total += float(loss_val)

            del feats_list, pos_list, A_list, t, e

        val_c = eval_cindex_evalsurv(model, val_loader, device)

        #if ep <= warmup_epochs:
        #    warmup_scheduler.step()
        #else:
        #    decay_scheduler.step()

        #warmup_scheduler.step()

        valC_by_epoch.append(float(val_c))
        ckpt_path = f"{SAVE_ROOT}/round_{round_id}_epoch_{ep}.pt"
        torch.save(model.state_dict(), ckpt_path)
        loss_by_epoch.append(float(total))

        msg = f"Epoch {ep} | loss={total:.4f} | valC={val_c:.4f}"
        print(msg)
        log_result(msg)
        writer.add_scalar("Loss/train", total, ep)
        writer.add_scalar("Cindex/val", val_c, ep)
        print("Region logits:", model.region_weights.detach().cpu())
        _w_raw = torch.relu(model.region_weights) + 0.01
        print("Region weights:", (_w_raw / _w_raw.sum()).detach().cpu())

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f">>> Training round time: {time.time() - t_round:.2f} sec")

    return valC_by_epoch



def bootstrap_rounds(dataset, rounds=30, train_frac=0.8, batch_size=16, epochs=8, seed_offset=0):
    print(">>> Starting bootstrap rounds.")

    t0 = time.time()
    N = len(dataset)
    idx = np.arange(N)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_valC = np.full((rounds, epochs), np.nan, dtype=np.float64)
    all_region_weights = []

    MIN_VAL_EVENTS = 20   # minimum events in validation set for a valid split
    MAX_SEED_RETRIES = 20  # try up to this many fallback seeds before giving up

    for r in range(1, rounds + 1):
        t_round = time.time()
        global_round = seed_offset + r
        print(f"\n=== Round {r} (global seed {global_round}) ===")

        # --- Seed retry loop: find a split with enough validation events ---
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
            if val_events >= MIN_VAL_EVENTS:
                print(f"  Seed {seed_attempt}: {val_events} validation events — OK")
                break
            retry += 1
            if retry >= MAX_SEED_RETRIES:
                print(f"  [WARN] Could not find seed with >={MIN_VAL_EVENTS} val events after {MAX_SEED_RETRIES} retries. Using seed {seed_attempt}.")
                break
            # fallback seed: encode original round + retry attempt uniquely
            seed_attempt = global_round * 1000 + retry
            print(f"  Seed {global_round * 1000 + retry - 1}: only {val_events} val events — retrying with seed {seed_attempt}")

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=24,
            prefetch_factor=1,
            collate_fn=collate_bags,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=24,
            prefetch_factor=1,
            collate_fn=collate_bags,
        )

        model = IPGGraphFormer()

        valC_by_epoch = train_one_round(
            model, train_loader, test_loader, device, epochs=epochs, round_id=r
        )

        full_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_bags, num_workers=0
        )

        for ep in range(1, epochs + 1):
            model_ep = IPGGraphFormer()
            ckpt_path = f"{SAVE_ROOT}/round_{r}_epoch_{ep}.pt"
            model_ep.load_state_dict(torch.load(ckpt_path, map_location=device))
            model_ep.to(device)

            pids, risks, mean_attn = extract_risks(model_ep, full_loader, device, return_attn=True)
            _w_raw = torch.relu(model_ep.region_weights) + 0.01
            ep_weights = (_w_raw / _w_raw.sum()).detach().cpu().numpy()

            save_dir = os.path.join(SAVE_ROOT, f"round_{r}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{ep}.npz")

            val_c_this_epoch = valC_by_epoch[ep - 1] if ep <= len(valC_by_epoch) else float("nan")
            np.savez(
                save_path,
                patient_ids=np.array(pids),
                risks=np.array(risks),
                region_names=np.array(dataset.region_names),
                val_cidx=np.float32(val_c_this_epoch),
                region_weights=ep_weights,
                mean_region_attn=mean_attn,   # (8, 8) averaged over all patients
            )
            print(f">>> Saved: {save_path}")
            os.remove(ckpt_path)

            del model_ep
            torch.cuda.empty_cache()

        all_valC[r - 1, :len(valC_by_epoch)] = np.array(valC_by_epoch, dtype=np.float64)

        msg = f"[Round {r}] valC_lastEpoch={valC_by_epoch[-1]:.4f}"
        print(msg)
        log_result(msg)


        # ---- LOG REGION WEIGHTS BEFORE DELETING MODEL ----
        _w_raw = torch.relu(model.region_weights) + 0.01
        weights = (_w_raw / _w_raw.sum()).detach().cpu().numpy()
        all_region_weights.append(weights)

        region_line = f"[Round {r}] Region_weights:\n"
        for name, w in zip(dataset.region_names, weights):
            region_line += f"  {name}: {w:.4f}\n"
        print(region_line)
        log_result(region_line)

        print(f"=== Round {r} time: {time.time() - t_round:.2f} sec ===")

        del model, train_loader, test_loader
        import gc
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ===== AGGREGATE BEST EPOCH ACROSS ROUNDS =====
    mean_by_epoch = np.nanmean(all_valC, axis=0)
    best_ep_idx = int(np.nanargmax(mean_by_epoch))
    best_ep = best_ep_idx + 1
    print(f">>> Best epoch = {best_ep}")

    all_risks = []
    all_pids = []
    all_round_attn = []   # (8, 8) per round, from best epoch

    for r in range(1, rounds + 1):
        path = os.path.join(SAVE_ROOT, f"round_{r}", f"epoch_{best_ep}.npz")
        data = np.load(path, allow_pickle=True)
        all_pids.append(data["patient_ids"])
        all_risks.append(data["risks"])
        if "mean_region_attn" in data:
            all_round_attn.append(np.asarray(data["mean_region_attn"], dtype=np.float32))

    all_risks = np.stack(all_risks)
    mean_risk = np.mean(all_risks, axis=0)
    var_risk = np.var(all_risks, axis=0)

    # Aggregate cross-region attention across rounds
    mean_attn_global = None
    std_attn_global  = None
    if all_round_attn:
        attn_stack       = np.stack(all_round_attn)   # (n_rounds, 8, 8)
        mean_attn_global = attn_stack.mean(axis=0)
        std_attn_global  = attn_stack.std(axis=0)

    save_kwargs = dict(
        patient_ids=all_pids[0],
        mean_risk=mean_risk,
        var_risk=var_risk,
        region_names=data["region_names"],
    )
    if mean_attn_global is not None:
        save_kwargs["mean_region_attn"] = mean_attn_global
        save_kwargs["std_region_attn"]  = std_attn_global

    np.savez(os.path.join(SAVE_ROOT, "final_med3pa_input.npz"), **save_kwargs)
    print(">>> FINAL MED3PA INPUT SAVED")

    mean_by_epoch = np.nanmean(all_valC, axis=0)
    best_ep_idx = int(np.nanargmax(mean_by_epoch))
    best_ep = best_ep_idx + 1
    best_mean = float(mean_by_epoch[best_ep_idx])

    log_result("")
    log_result("=== BOOTSTRAP SUMMARY (mean valC across rounds by epoch) ===")
    print("\n=== BOOTSTRAP SUMMARY (mean valC across rounds by epoch) ===")

    for ep in range(1, epochs + 1):
        line = f"[AVG Epoch {ep}] mean_valC={mean_by_epoch[ep-1]:.4f}"
        print(line)
        log_result(line)

    final_line = f"[BEST AVG] epoch={best_ep} | mean_valC={best_mean:.4f}"
    print(final_line)
    log_result(final_line)

    print("\n=== AVERAGE REGION IMPORTANCE ACROSS BOOTSTRAP ROUNDS ===")

    all_region_weights = np.array(all_region_weights)
    mean_w = all_region_weights.mean(axis=0)
    std_w = all_region_weights.std(axis=0)

    for name, m, s in zip(dataset.region_names, mean_w, std_w):
        line = f"{name}: {m:.4f} ± {s:.4f}"
        print(line)
        log_result(line)

    # ---- Cross-region attention summary ----
    if mean_attn_global is not None:
        print("\n=== AVERAGE CROSS-REGION ATTENTION ACROSS BOOTSTRAP ROUNDS ===")
        log_result("\n=== AVERAGE CROSS-REGION ATTENTION ACROSS BOOTSTRAP ROUNDS ===")
        # Short names for compact matrix display
        short = [n.split("_")[-1][:8] for n in dataset.region_names]
        header = " " * 14 + "  ".join(f"{s:>8}" for s in short)
        print(header)
        log_result(header)
        for i, rn in enumerate(dataset.region_names):
            row = f"{rn[:12]:>12}  " + "  ".join(f"{mean_attn_global[i, j]:.4f}" for j in range(len(dataset.region_names)))
            print(row)
            log_result(row)

    print(f">>> TOTAL bootstrap time: {time.time() - t0:.2f} sec")



def main():
    global LOG_F
    t_total = time.time()
    print(">>> main() started.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_dir", action="append", required=True,
                    help="Path to a region folder. Repeat this flag for multiple regions.")
    ap.add_argument("--bootstrap_rounds", type=int, default=30)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--save_root", type=str,
                    default="/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate",
                    help="Root directory to save checkpoints and outputs.")
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="Offset added to round index for reproducible parallel seeds.")
    args = ap.parse_args()

    global SAVE_ROOT
    SAVE_ROOT = args.save_root
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # args.class_dir is now a list of paths
    class_dirs = args.class_dir

    class_name = os.path.basename(class_dirs[0].rstrip("/"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{class_name.replace(' ', '_')}_ipgphormer_{timestamp}_results.txt"
    LOG_F = open(log_name, "w", buffering=1)
    print(f">>> Logging results to {log_name}")

    set_seed(1337)

    excel = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
    print(">>> Loading survival table...")
    t_surv = time.time()
    surv_df = pd.read_excel(excel)
    surv_df = surv_df[surv_df["type"].str.contains("BLCA|HNSC|LUAD|BRCA|UCEC", na=False)]
    surv_df = surv_df[["bcr_patient_barcode", "PFI", "PFI.time"]]
    surv_df["PFI.time"] = pd.to_numeric(surv_df["PFI.time"], errors="coerce")
    surv_df = surv_df.dropna(subset=["PFI.time"])
    print(f">>> Survival table load time: {time.time() - t_surv:.2f} sec")

    print(">>> Building dataset...")
    t_ds = time.time()
    ds = MultiRegionDataset(class_dirs, surv_df)
    print(f">>> Slides loaded: {len(ds)}")
    print(f">>> Dataset build time: {time.time() - t_ds:.2f} sec")
    #stage_adjacency_matrices(ds.paths,args.class_dir)
    print(f"[STAGE] SLURM_TMPDIR={os.environ.get('SLURM_TMPDIR')}")
    print(f"[STAGE] STAGED_ADJ_DIR={STAGED_ADJ_DIR}")
    if STAGED_ADJ_DIR:
        print(f"[STAGE] staged_files={len(os.listdir(STAGED_ADJ_DIR))}")


    bootstrap_rounds(
        ds,
        rounds=args.bootstrap_rounds,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed_offset=args.seed_offset,
    )

    print(f">>> TOTAL SCRIPT TIME: {time.time() - t_total:.2f} sec")

    if LOG_F is not None:
        LOG_F.close()


if __name__ == "__main__":
    main()
    cleanup_staged_adjacency()


# python interpret_graphtrans_12.py --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" --bootstrap_rounds 1 --epochs 1
# salloc --time=2:00:00 --gpus=a100_4g.20gb:1 --cpus-per-task=32 --mem=64G
# python interpret_graphtrans_12.py --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" --bootstrap_rounds 5 --epochs 10
# ssh trig-login01
#salloc -p debug --nodes=1 --gpus-per-node=1 --time=01:00:00

# python interpret_graphtrans_optimal.py --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" --bootstrap_rounds 1 --epochs 25

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)"

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)"

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" 

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)"

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" 

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)"

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)"

#"/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis"

'''
python interpret_graphtrans_late_int.py \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)" \
  --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis" \
  --bootstrap_rounds 5 \
  --epochs 1
  '''

#python interpret_graphtrans_late_int_sparse_m3save.py --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Perivesical adipose tissue (extravesical fat)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Invasive urothelial carcinoma (tumor)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Normal urothelium (benign mucosa)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Inflammatory infiltrates (immune cells)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Lamina propria (fibrovascular stroma)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Blood vessels (vasculature)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Muscularis propria (detrusor muscle)" --class_dir "/home/sorkwos/links/scratch/UNI2_classwise_embeddings/Necrosis" --bootstrap_rounds 1 --epochs 1