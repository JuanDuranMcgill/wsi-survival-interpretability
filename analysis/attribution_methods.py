"""Gradient-based compartment attributions, for comparison with Shapley values.

Reviewers of an attribution audit will ask how exact Shapley values compare
with the gradient methods the field uses. Two standard ones are computed on the
same trained models and out-of-bag patients as the Shapley values, and
summarised per compartment as the mean over patients of each compartment's
share of the absolute attribution, the same scale as a fusion weight or an
attention share:

- gradient x input (Shrikumar et al., 2017);
- integrated gradients (Sundararajan et al., 2017), from the cohort-average
  compartment embedding for the region-fusion model and from zero tile features
  for ABMIL, with the completeness gap (sum of attributions minus the change in
  risk from the baseline) recorded as a check.

For the region-fusion model the inputs are the per-compartment embeddings that
enter cross-region attention, the same point at which the Shapley analysis
removes a compartment. For ABMIL the inputs are the tile features, summed over
each compartment's tiles.
"""
from __future__ import annotations

import numpy as np
import torch


def _fusion_head(model, X):
    """IPGGraphFormer from region embeddings (n, R, d) to risk (n,), differentiable."""
    attn, _ = model.cross_region_attn(X, X, X, need_weights=False)
    H = model.cross_region_norm(X + attn)
    w_raw = torch.relu(model.region_weights) + 0.01
    w = w_raw / w_raw.sum()
    return model.head((w.view(1, -1, 1) * H).sum(dim=1)).squeeze(-1)


def _share(A):
    """(n, R) signed attributions -> mean share of absolute attribution per compartment."""
    a = np.abs(A)
    return (a / np.maximum(a.sum(1, keepdims=True), 1e-12)).mean(0)


def graph_attributions(model, E, device, steps=32):
    """Gradient x input and integrated gradients on cached region embeddings E (n, R, d).

    The embeddings pass through a LayerNorm straight away, which makes the risk
    almost scale-invariant in them and jump near the all-zero input, so a zero
    baseline breaks the completeness property of integrated gradients. The
    baseline is therefore the cohort-average embedding of each compartment (the
    'average patient'); the completeness gap is recorded as a check.
    """
    model.eval()
    E = E.to(device).float()
    B = E.mean(0, keepdim=True).expand_as(E)
    with torch.enable_grad():
        X = E.clone().requires_grad_(True)
        _fusion_head(model, X).sum().backward()
        gxi = (X.grad * E).sum(-1).detach().cpu().numpy()
        total = torch.zeros_like(E)
        for k in range(steps):
            Xa = (B + (E - B) * ((k + 0.5) / steps)).clone().requires_grad_(True)
            _fusion_head(model, Xa).sum().backward()
            total += Xa.grad
        ig = ((E - B) * total / steps).sum(-1).detach().cpu().numpy()
    with torch.no_grad():
        delta = (_fusion_head(model, E) - _fusion_head(model, B)).cpu().numpy()
    gap = float(np.max(np.abs(ig.sum(1) - delta)) / (np.std(delta) or 1.0))
    return {"attr_grad_x_input": _share(gxi).tolist(),
            "attr_integrated_gradients": _share(ig).tolist(),
            "ig_completeness_gap_in_risk_sd": gap, "ig_steps": steps,
            "ig_baseline": "cohort-average compartment embedding"}


def abmil_attributions(model, loader, device, R, steps=16):
    """Gradient x input and integrated gradients on ABMIL tile features, summed per compartment."""
    model.eval()
    GXI, IG, gaps = [], [], []
    for feats_list, _, _, _, _, _ in loader:
        for fl in feats_list:
            sizes = [0 if f is None else f.shape[0] for f in fl]
            if sum(sizes) == 0:
                GXI.append(np.zeros(R)); IG.append(np.zeros(R))
                continue
            x = torch.cat([f for f in fl if f is not None and f.shape[0] > 0]).to(device).float()
            region = np.repeat(np.arange(R), sizes)
            with torch.enable_grad():
                xg = x.clone().requires_grad_(True)
                model.forward_bag([xg], device).backward()
                g = (xg.grad * x).sum(1).detach().cpu().numpy()
                total = torch.zeros_like(x)
                for k in range(1, steps + 1):
                    xa = (x * (k / steps)).clone().requires_grad_(True)
                    model.forward_bag([xa], device).backward()
                    total += xa.grad
                ig = (x * total / steps).sum(1).detach().cpu().numpy()
            with torch.no_grad():
                d = float(model.forward_bag([x], device) - model.forward_bag([torch.zeros_like(x)], device))
            gaps.append(abs(ig.sum() - d))
            GXI.append(np.bincount(region, weights=g, minlength=R))
            IG.append(np.bincount(region, weights=ig, minlength=R))
    return {"attr_grad_x_input": _share(np.array(GXI)).tolist(),
            "attr_integrated_gradients": _share(np.array(IG)).tolist(),
            "ig_completeness_gap_max_abs": float(max(gaps)) if gaps else None, "ig_steps": steps}
