#!/usr/bin/env python3
"""Isolate further: train the *same* main_mlp/A/B architecture, predicting
risk = main_raw.sum(1) + raw_interaction.sum(1) directly (no centering at
all, no L1 penalty), on the same synthetic task. If this also plateaus
around the same held-out MSE (~1.7-1.8) that the full GA2MHead does, the bug
is in the base main_mlp/A/B architecture or its optimization, not in
center_decomposition or the L1 penalty. If this trains fine, the problem is
specifically introduced by centering's interaction with gradients/training.
"""
import os
import sys
import torch

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga2m_head import GA2MHead, raw_interactions_from_projections


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
    return H, y


n_train, n_eval, num_regions, d_model = 4000, 4000, 4, 16
H_train, y_train = make_synthetic(n_train, num_regions, d_model, seed=1)
H_eval, y_eval = make_synthetic(n_eval, num_regions, d_model, seed=2)


def train_eval(use_centering, lr=3e-4, weight_decay=1e-2, batch_size=64, epochs=300, patience=30):
    torch.manual_seed(0)
    head = GA2MHead(num_regions=num_regions, d_model=d_model, rank=8, main_hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = H_train.shape[0]
    best_mse, best_ep, bad = float("inf"), 0, 0
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch, target = H_train[idx], y_train[idx]
            if use_centering:
                risk, main_c, inter_c, intercept, _ = head(batch)
                loss = ((risk - target) ** 2).mean() + 1e-2 * head.l1_interaction_penalty(inter_c)
            else:
                main_raw, Ah, Bh = head.raw_projections(batch)
                inter_raw = raw_interactions_from_projections(Ah, Bh, head.pairs)
                risk = main_raw.sum(dim=1) + inter_raw.sum(dim=1)
                loss = ((risk - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            if use_centering:
                risk, *_ = head(H_eval)
            else:
                main_raw, Ah, Bh = head.raw_projections(H_eval)
                inter_raw = raw_interactions_from_projections(Ah, Bh, head.pairs)
                risk = main_raw.sum(dim=1) + inter_raw.sum(dim=1)
            eval_mse = ((risk - y_eval) ** 2).mean().item()
        if eval_mse < best_mse - 1e-5:
            best_mse, best_ep, bad = eval_mse, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_mse, best_ep, ep


print("=== WITHOUT centering (raw sum, no L1) ===")
best_mse, best_ep, stopped = train_eval(use_centering=False)
print(f"  best held-out MSE = {best_mse:.4f} @ epoch {best_ep} (stopped {stopped})")

print("\n=== WITH centering (full GA2MHead, L1=1e-2) ===")
best_mse, best_ep, stopped = train_eval(use_centering=True)
print(f"  best held-out MSE = {best_mse:.4f} @ epoch {best_ep} (stopped {stopped})")
