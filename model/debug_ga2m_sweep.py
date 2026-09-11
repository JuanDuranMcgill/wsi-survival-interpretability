#!/usr/bin/env python3
"""Given the oracle test proves center_decomposition is exact, sweep rank and
L1 strength to check the capacity-mismatch hypothesis: does training find a
recoverable decomposition when the bilinear rank is closer to the true
(rank-1-per-term) signal, or when the L1 penalty is much stronger?
"""
import os
import sys
import torch

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga2m_head import GA2MHead, region_pairs
from scipy.stats import pearsonr


def make_synthetic(n, num_regions, d_model, seed):
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(n, num_regions, d_model, generator=g)
    c = torch.randn(d_model, generator=g)
    a0 = torch.randn(d_model, generator=g)
    b1 = torch.randn(d_model, generator=g)
    a1 = torch.randn(d_model, generator=g)
    b0 = torch.randn(d_model, generator=g)
    g1_true = H[:, 0, :] @ c
    g12_true = (H[:, 0, :] @ a0) * (H[:, 1, :] @ b1) + (H[:, 1, :] @ a1) * (H[:, 0, :] @ b0)
    noise = 0.1 * torch.randn(n, generator=g)
    y = g1_true + g12_true + noise
    y = (y - y.mean()) / y.std()
    return H, y, g1_true, g12_true


def train_and_eval(rank, lam_int, num_regions=4, d_model=16, n_train=4000, n_eval=4000,
                    epochs=60, batch_size=64, lr=3e-3, weight_decay=1e-4):
    torch.manual_seed(0)
    H_train, y_train, _, _ = make_synthetic(n_train, num_regions, d_model, seed=1)
    H_eval, y_eval, g1_true_eval, g12_true_eval = make_synthetic(n_eval, num_regions, d_model, seed=2)

    head = GA2MHead(num_regions=num_regions, d_model=d_model, rank=rank, main_hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = H_train.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch, target = H_train[idx], y_train[idx]
            risk, main_c, inter_c, intercept, _ = head(batch)
            mse = ((risk - target) ** 2).mean()
            loss = mse + lam_int * head.l1_interaction_penalty(inter_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(mse.item()) * len(idx)
        if os.environ.get("GA2M_DEBUG_VERBOSE") and (ep % 10 == 0 or ep == 0):
            print(f"    [rank={rank} lam={lam_int}] epoch {ep:3d} train_mse={total/n:.5f}", flush=True)

    with torch.no_grad():
        risk, main_c, inter_c, intercept, _ = head(H_eval)
    final_mse = ((risk - y_eval) ** 2).mean().item()

    pairs = region_pairs(num_regions)
    pair01 = pairs.index((0, 1))
    r_main, _ = pearsonr(main_c[:, 0].numpy(), g1_true_eval.numpy())
    r_int, _ = pearsonr(inter_c[:, pair01].numpy(), g12_true_eval.numpy())
    return r_main, r_int, final_mse


import ga2m_head
print(f"ga2m_head module file: {ga2m_head.__file__}")

os.environ["GA2M_DEBUG_VERBOSE"] = "1"
print(f"\n{'rank':>5} {'lam_int':>9} {'mse':>10} {'r_main':>8} {'r_int':>8}")
r_main, r_int, mse = train_and_eval(rank=8, lam_int=1e-3)
print(f"{8:>5} {1e-3:>9.1e} {mse:>10.5f} {r_main:>+8.3f} {r_int:>+8.3f}")
