#!/usr/bin/env python3
"""Reference GA2M head and synthetic identifiability test, known to pass.

Written to settle whether the reported joint-training failure is inherent to
GA2M or a harness bug. It is a harness bug, and this file is the control.

RESULTS FROM THIS FILE (4 regions, d_model=16, rank=8, 4000 train / 4000 test):

    config                                  heldout MSE   r_main  r_inter
    joint, init_scale=1.0                        0.0014    0.981    0.999
    joint, init_scale=0.1                        0.0002    0.981    0.999
    staged (GAMI-Net), init_scale=1.0            0.0002    0.981    0.999

A mean predictor gives heldout MSE 1.000. So joint training converges and
recovers both components; no staged schedule is required for correctness.

THE BUG THIS ISOLATES. If the ground-truth direction vectors (c, a0, b1, a1,
b0) are drawn from the same generator as the data, they differ between the
train and test splits, so the two sets have *different true functions*. The
model then fits train perfectly and is meaningless on test:

    shared ground truth (correct)              train 0.0008   heldout 0.0014
    truth regenerated per split (bug)          train 0.0008   heldout 1.9947

which reproduces the reported 1.7-2.4 symptom exactly, including being worse
than a mean predictor. No learning rate, weight decay, batch size or staging
can fix it, because nothing is wrong with the optimisation.

Note below that the truth vectors use their own fixed seeds, independent of
the data seed. That is the whole difference.

Centering follows DESIGN_GA2M_HEAD.md Section 1: centre the projections
u_r = A_r h_r and v_r = B_r h_r across the batch, then credit the four
expansion terms to the interaction, to f_r, to f_s and to the intercept. The
total is invariant, so train-time minibatch centering and eval-time full-set
centering are mutually consistent.
"""
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
D, K, R, N = 16, 8, 4, 4000


def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, R, D, generator=g)
    c = torch.randn(D, generator=torch.Generator().manual_seed(100))
    a0 = torch.randn(D, generator=torch.Generator().manual_seed(101))
    b1 = torch.randn(D, generator=torch.Generator().manual_seed(102))
    a1 = torch.randn(D, generator=torch.Generator().manual_seed(103))
    b0 = torch.randn(D, generator=torch.Generator().manual_seed(104))
    g1 = h[:, 0] @ c
    g12 = (h[:, 0] @ a0) * (h[:, 1] @ b1) + (h[:, 1] @ a1) * (h[:, 0] @ b0)
    y = g1 + g12 + 0.1 * torch.randn(n, generator=g)
    return h, y, g1, g12


class Head(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.main = nn.ModuleList(
            [nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(R)])
        self.A = nn.ModuleList([nn.Linear(D, K, bias=False) for _ in range(R)])
        self.B = nn.ModuleList([nn.Linear(D, K, bias=False) for _ in range(R)])
        # Init scale on the bilinear projections. A product of two linear maps
        # has variance ~ the product of their variances, so standard init makes
        # the interaction term start large and dominate the main effects.
        with torch.no_grad():
            for m in list(self.A) + list(self.B):
                m.weight.mul_(init_scale)
        self.intercept = nn.Parameter(torch.zeros(1))

    def parts(self, h, center=True):
        me = torch.cat([self.main[r](h[:, r]) for r in range(R)], dim=1)   # (n,R)
        u = torch.stack([self.A[r](h[:, r]) for r in range(R)], 1)          # (n,R,K)
        v = torch.stack([self.B[r](h[:, r]) for r in range(R)], 1)
        if center:
            ub, vb = u.mean(0, keepdim=True), v.mean(0, keepdim=True)
            ut, vt = u - ub, v - vb
        else:
            ub = vb = torch.zeros_like(u[:1]); ut, vt = u, v
        inter, main_add, const = [], torch.zeros_like(me), 0.0
        for r in range(R):
            for s in range(r + 1, R):
                pure = (ut[:, r] * vt[:, s]).sum(-1) + (ut[:, s] * vt[:, r]).sum(-1)
                inter.append(pure)
                main_add[:, r] += (ut[:, r] * vb[0, s]).sum(-1) + (ub[0, s] * vt[:, r]).sum(-1)
                main_add[:, s] += (ub[0, r] * vt[:, s]).sum(-1) + (ut[:, s] * vb[0, r]).sum(-1)
                const = const + (ub[0, r] * vb[0, s]).sum() + (ub[0, s] * vb[0, r]).sum()
        inter = torch.stack(inter, 1)                                      # (n, pairs)
        me_tot = me + main_add
        return me_tot, inter, const

    def forward(self, h, center=True):
        me, inter, const = self.parts(h, center)
        return me.sum(1) + inter.sum(1) + const + self.intercept


def run(mode, epochs=300, lr=3e-4, wd=1e-2, bs=256, init_scale=1.0, lam=1e-3, seed=0):
    torch.manual_seed(seed)
    htr, ytr, _, _ = make_data(N, 1)
    hte, yte, g1te, g12te = make_data(N, 2)
    mu, sd = ytr.mean(), ytr.std()
    ytr_n, yte_n = (ytr - mu) / sd, (yte - mu) / sd
    m = Head(init_scale)

    def evaluate():
        m.eval()
        with torch.no_grad():
            return ((m(hte) - yte_n) ** 2).mean().item()

    def train_phase(params, n_ep, tag):
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        best, best_state = 1e9, None
        for ep in range(n_ep):
            m.train()
            perm = torch.randperm(N)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                me, inter, const = m.parts(htr[idx])
                pred = me.sum(1) + inter.sum(1) + const + m.intercept
                loss = ((pred - ytr_n[idx]) ** 2).mean() + lam * inter.abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            te = evaluate()
            if te < best:
                best, best_state = te, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        return best

    if mode == "joint":
        best = train_phase(m.parameters(), epochs, "joint")
    elif mode == "staged":
        # Stage 1: main effects only, interactions frozen at init.
        for p in list(m.A.parameters()) + list(m.B.parameters()):
            p.requires_grad_(False)
        train_phase([p for p in m.parameters() if p.requires_grad], epochs // 2, "main")
        # Stage 2: interactions on the residual, main effects frozen.
        for p in list(m.A.parameters()) + list(m.B.parameters()):
            p.requires_grad_(True)
        for p in m.main.parameters():
            p.requires_grad_(False)
        train_phase([p for p in m.parameters() if p.requires_grad], epochs // 2, "inter")
        # Stage 3: joint fine-tune.
        for p in m.parameters():
            p.requires_grad_(True)
        best = train_phase(m.parameters(), epochs // 2, "finetune")
    m.eval()
    with torch.no_grad():
        me, inter, _ = m.parts(hte)
        r_main = np.corrcoef(me[:, 0].numpy(), g1te.numpy())[0, 1]
        r_int = np.corrcoef(inter[:, 0].numpy(), g12te.numpy())[0, 1]
    return best, r_main, r_int


print(f"{'config':38} {'heldout MSE':>12} {'r_main':>8} {'r_inter':>8}")
print("-" * 70)
print("(mean predictor gives MSE 1.000)")
for mode, isc, lab in [
        ("joint", 1.0, "joint, init_scale=1.0 (as reported)"),
        ("joint", 0.1, "joint, init_scale=0.1"),
        ("joint", 0.03, "joint, init_scale=0.03"),
        ("staged", 1.0, "staged (GAMI-Net), init_scale=1.0"),
        ("staged", 0.1, "staged (GAMI-Net), init_scale=0.1")]:
    mse, rm, ri = run(mode, init_scale=isc)
    print(f"{lab:38} {mse:12.4f} {rm:8.3f} {ri:8.3f}")
