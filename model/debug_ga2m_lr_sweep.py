#!/usr/bin/env python3
"""Held-out MSE cannot even beat 'predict the mean' in the previous run —
that's a training-dynamics problem, not attribution. Sweep learning rate /
batch size / epochs to see whether this architecture can genuinely
generalize on the synthetic task at all, before concluding anything about
identifiability. Also fits a plain ridge regression on the same data as a
sanity floor: if ridge can't learn it either, the synthetic task itself
(noise level, sample size) is the problem, not the GA2M head.
"""
import os
import sys
import torch
import numpy as np

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga2m_head import GA2MHead


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


# --- Ridge regression floor, using the *true* engineered features (best case) ---
n_train, n_eval, num_regions, d_model = 4000, 4000, 4, 16
H_train, y_train, g1_tr, g12_tr = make_synthetic(n_train, num_regions, d_model, seed=1)
H_eval, y_eval, g1_ev, g12_ev = make_synthetic(n_eval, num_regions, d_model, seed=2)

X_tr = torch.cat([g1_tr.unsqueeze(1), g12_tr.unsqueeze(1)], dim=1).numpy()
X_ev = torch.cat([g1_ev.unsqueeze(1), g12_ev.unsqueeze(1)], dim=1).numpy()
beta, *_ = np.linalg.lstsq(np.concatenate([X_tr, np.ones((len(X_tr), 1))], axis=1), y_train.numpy(), rcond=None)
pred_ev = np.concatenate([X_ev, np.ones((len(X_ev), 1))], axis=1) @ beta
oracle_mse = float(np.mean((pred_ev - y_eval.numpy()) ** 2))
print(f"Oracle-features ridge (using true g1,g12 directly): held-out MSE = {oracle_mse:.5f}  (should be ~noise floor 0.01)")

# --- Now the actual GA2M head under different training recipes ---
def train_eval(lr, weight_decay, batch_size, epochs, lam_int=1e-2, patience=30):
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
            risk, main_c, inter_c, intercept, _ = head(batch)
            loss = ((risk - target) ** 2).mean() + lam_int * head.l1_interaction_penalty(inter_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            risk, *_ = head(H_eval)
            eval_mse = ((risk - y_eval) ** 2).mean().item()
        if eval_mse < best_mse - 1e-5:
            best_mse, best_ep, bad = eval_mse, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_mse, best_ep, ep


configs = [
    dict(lr=3e-3, weight_decay=1e-2, batch_size=64, epochs=200),
    dict(lr=1e-3, weight_decay=1e-2, batch_size=64, epochs=200),
    dict(lr=3e-4, weight_decay=1e-2, batch_size=64, epochs=400),
    dict(lr=1e-4, weight_decay=1e-2, batch_size=64, epochs=800),
    dict(lr=3e-4, weight_decay=1e-1, batch_size=256, epochs=400),
    dict(lr=3e-4, weight_decay=3e-2, batch_size=256, epochs=400),
    dict(lr=1e-4, weight_decay=1e-1, batch_size=256, epochs=800),
]
print(f"\n{'lr':>8} {'wd':>8} {'batch':>6} {'max_ep':>7} {'best_mse':>9} {'best_ep':>8} {'stopped_ep':>10}")
for cfg in configs:
    best_mse, best_ep, stopped_ep = train_eval(**cfg)
    print(f"{cfg['lr']:>8.0e} {cfg['weight_decay']:>8.0e} {cfg['batch_size']:>6} {cfg['epochs']:>7} "
          f"{best_mse:>9.4f} {best_ep:>8} {stopped_ep:>10}", flush=True)
