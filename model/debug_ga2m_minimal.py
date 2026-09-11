#!/usr/bin/env python3
"""Reduce as far as possible: does a bare 16->32->1 MLP (GA2MHead's exact
main_mlp architecture) learn a *linear* function of one region's embedding
at all, in complete isolation from everything else? If this fails too, the
bug is at the most basic level (a real bug in this repo's training loop /
layer setup, not anything specific to GA2M). If it succeeds, the problem is
specific to training main_mlps and A/B jointly (redundant-capacity
interference), not a basic bug.
"""
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

torch.manual_seed(0)
n_train, n_eval, d_model = 4000, 4000, 16

g1 = torch.Generator().manual_seed(1)
H_train = torch.randn(n_train, d_model, generator=g1)
c = torch.randn(d_model, generator=g1)
y_train = H_train @ c
y_train = y_train + 0.1 * torch.randn(n_train, generator=g1)
y_train = (y_train - y_train.mean()) / y_train.std()

g2 = torch.Generator().manual_seed(2)
H_eval = torch.randn(n_eval, d_model, generator=g2)
y_eval = H_eval @ c
y_eval = y_eval + 0.1 * torch.randn(n_eval, generator=g2)
y_eval = (y_eval - y_eval.mean()) / y_eval.std()

mlp = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=3e-4, weight_decay=1e-2)

batch_size = 64
best_mse, best_ep, bad = float("inf"), 0, 0
for ep in range(1, 301):
    mlp.train()
    perm = torch.randperm(n_train)
    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        batch, target = H_train[idx], y_train[idx]
        pred = mlp(batch).squeeze(-1)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        pred = mlp(H_eval).squeeze(-1)
        eval_mse = ((pred - y_eval) ** 2).mean().item()
    if eval_mse < best_mse - 1e-5:
        best_mse, best_ep, bad = eval_mse, ep, 0
    else:
        bad += 1
        if bad >= 30:
            break
    if ep % 20 == 0 or ep == 1:
        print(f"  epoch {ep:3d}  eval_mse={eval_mse:.4f}  (best {best_mse:.4f} @ {best_ep})", flush=True)

print(f"\nFinal: best held-out MSE = {best_mse:.4f} @ epoch {best_ep}  "
      f"(1.0 = predict mean, noise floor ~0.01)")
print("RESULT:", "OK, bare MLP learns fine" if best_mse < 0.1 else "FAIL — bug is this basic")
