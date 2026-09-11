"""GA2M aggregation head: region-level main effects + pairwise interactions.

Implements docs/DESIGN_GA2M_HEAD.md Section 1. Replaces
`cross_region_attn -> cross_region_norm -> region_weights weighted sum -> head`
in the existing backbone. The per-region encoder (proj/pos_mlp/beta/agt_layers/
attn_pool/forward_bag) is untouched and lives in `train_ga2m.py`.

    risk(patient) = intercept + sum_r f_r(h_r) + sum_{r<s} f_rs(h_r, h_s)

Identifiability (Section 1, "do not skip this"): an additive decomposition is
not unique without centering. The first implementation centered the *scalar*
interaction output by binning on the main-effect output and subtracting a
conditional mean; a synthetic recovery test (y = g1(h0) + g12(h0,h1) + noise,
ground truth chosen inside the model's own functional form) showed that
approach gives *negatively* correlated attributions (Pearson r ~ -0.4) despite
fitting the target almost exactly. Root cause: conditioning on the main-effect
*output* rather than the input is a many-to-one, moving-target constraint that
couples the centering to the very quantity it's supposed to identify.

The fix is exact and closed-form, no binning, no iteration. With
u_r = A_r(h_r), v_r = B_r(h_r) (the rank-k projections), the interaction is
f_rs = <u_r, v_s> + <u_s, v_r>. Centre the *projections* across the batch,
ubar_r = mean(u_r), utilde_r = u_r - ubar_r (same for v), and expand:

    <u_r, v_s> + <u_s, v_r>
      = [<utilde_r, vtilde_s> + <utilde_s, vtilde_r>]   pure interaction
      + [<utilde_r, vbar_s>   + <ubar_s,  vtilde_r>]    function of h_r alone -> add to f_r
      + [<ubar_r,  vtilde_s>  + <utilde_s, vbar_r>]     function of h_s alone -> add to f_s
      + [<ubar_r,  vbar_s>    + <ubar_s,  vbar_r>]       batch-wide constant -> add to intercept

E[f_rs | h_r] = <utilde_r, E[vtilde_s]> = 0 exactly under independence of
regions r and s. (If regions are correlated within a slide, this holds only
approximately; validate_ga2m_synthetic.py includes a correlated-region case to
check whether that residual leakage matters in practice.)
"""
from __future__ import annotations

import itertools
import torch
import torch.nn as nn


def region_pairs(num_regions: int):
    """All (r, s) with r < s, a fixed canonical ordering used everywhere."""
    return list(itertools.combinations(range(num_regions), 2))


def raw_interactions_from_projections(Ah: torch.Tensor, Bh: torch.Tensor, pairs):
    """The *uncentered* interaction f_rs = <u_r,v_s> + <u_s,v_r> for each pair,
    from the raw projections. Used only for the sum-preservation check."""
    cols = []
    for (r, s) in pairs:
        cols.append((Ah[:, r, :] * Bh[:, s, :]).sum(-1) + (Ah[:, s, :] * Bh[:, r, :]).sum(-1))
    return torch.stack(cols, dim=1)


def center_decomposition(main_raw: torch.Tensor, Ah: torch.Tensor, Bh: torch.Tensor, pairs):
    """Exact identifiability centering (see module docstring).

    Args:
        main_raw: (B, R) uncentered main effects.
        Ah, Bh:   (B, R, k) rank-k projections u_r = A_r(h_r), v_r = B_r(h_r).
        pairs: canonical `region_pairs(R)` ordering.

    Returns:
        main_centered (B, R), inter_centered (B, P), intercept (scalar tensor),
        info dict.
    """
    B, R = main_raw.shape
    P = len(pairs)

    intercept = main_raw.new_zeros(())

    # Main effects: plain batch-mean centering (unambiguous, always exact).
    mean_main = main_raw.mean(dim=0)                     # (R,)
    main = main_raw - mean_main.unsqueeze(0)
    intercept = intercept + mean_main.sum()

    # Interactions: exact algebraic split of the bilinear form (module docstring).
    u_bar = Ah.mean(dim=0)                                # (R, k)
    v_bar = Bh.mean(dim=0)                                # (R, k)
    u_tilde = Ah - u_bar.unsqueeze(0)                     # (B, R, k)
    v_tilde = Bh - v_bar.unsqueeze(0)                     # (B, R, k)

    main = main.clone()
    inter_list = []
    for (r, s) in pairs:
        pure = ((u_tilde[:, r, :] * v_tilde[:, s, :]).sum(-1)
                + (u_tilde[:, s, :] * v_tilde[:, r, :]).sum(-1))

        contrib_r = ((u_tilde[:, r, :] * v_bar[s].unsqueeze(0)).sum(-1)
                     + (v_tilde[:, r, :] * u_bar[s].unsqueeze(0)).sum(-1))
        contrib_s = ((v_tilde[:, s, :] * u_bar[r].unsqueeze(0)).sum(-1)
                     + (u_tilde[:, s, :] * v_bar[r].unsqueeze(0)).sum(-1))
        const = (u_bar[r] * v_bar[s]).sum() + (u_bar[s] * v_bar[r]).sum()

        inter_list.append(pure)
        main[:, r] = main[:, r] + contrib_r
        main[:, s] = main[:, s] + contrib_s
        intercept = intercept + const

    inter = torch.stack(inter_list, dim=1)               # (B, P)
    info = {"method": "exact_projection_centering", "n_regions": R, "n_pairs": P}
    return main, inter, intercept, info


def residualize_and_recenter(main_raw, Ah, Bh, pairs):
    """Fallback for correlated regions (E[vtilde_s | h_r] != 0 exactly).

    Linearly residualizes vtilde_s on utilde_r within the batch (per pair,
    per direction) before the exact split, and moves the explained piece into
    the corresponding main effect. Only use this if validate_ga2m_synthetic.py's
    correlated-region case shows leakage with the plain `center_decomposition`;
    it costs a per-pair least-squares solve and is not needed under
    independence.
    """
    B, R = main_raw.shape
    intercept = main_raw.new_zeros(())
    mean_main = main_raw.mean(dim=0)
    main = (main_raw - mean_main.unsqueeze(0)).clone()
    intercept = intercept + mean_main.sum()

    u_bar = Ah.mean(dim=0)
    v_bar = Bh.mean(dim=0)
    u_tilde = Ah - u_bar.unsqueeze(0)
    v_tilde = Bh - v_bar.unsqueeze(0)

    def _residualize(target, predictor):
        # target, predictor: (B, k). Least-squares beta (k,k) s.t.
        # target ~= predictor @ beta; returns (residual, explained).
        # Solved per-pair on the fly; B is small (mini-batch or one round's
        # OOB set), k is 16, so this is cheap.
        sol = torch.linalg.lstsq(predictor, target).solution  # (k, k)
        explained = predictor @ sol
        return target - explained, explained

    inter_list = []
    for (r, s) in pairs:
        # Remove the part of vtilde_s linearly explained by utilde_r (and
        # symmetrically for the other cross term) before the exact split.
        vtilde_s_resid, vtilde_s_expl = _residualize(v_tilde[:, s, :], u_tilde[:, r, :])
        utilde_s_resid, utilde_s_expl = _residualize(u_tilde[:, s, :], v_tilde[:, r, :])

        pure = ((u_tilde[:, r, :] * vtilde_s_resid).sum(-1)
                + (utilde_s_resid * v_tilde[:, r, :]).sum(-1))

        # Explained pieces are functions of h_r alone (built from utilde_r /
        # vtilde_r) -> credit to main effect r, same as the batch-mean terms.
        contrib_r = ((u_tilde[:, r, :] * v_bar[s].unsqueeze(0)).sum(-1)
                     + (v_tilde[:, r, :] * u_bar[s].unsqueeze(0)).sum(-1)
                     + (u_tilde[:, r, :] * vtilde_s_expl).sum(-1)
                     + (utilde_s_expl * v_tilde[:, r, :]).sum(-1))
        contrib_s = ((v_tilde[:, s, :] * u_bar[r].unsqueeze(0)).sum(-1)
                     + (u_tilde[:, s, :] * v_bar[r].unsqueeze(0)).sum(-1))
        const = (u_bar[r] * v_bar[s]).sum() + (u_bar[s] * v_bar[r]).sum()

        inter_list.append(pure)
        main[:, r] = main[:, r] + contrib_r
        main[:, s] = main[:, s] + contrib_s
        intercept = intercept + const

    inter = torch.stack(inter_list, dim=1)
    info = {"method": "residualized_projection_centering", "n_regions": R, "n_pairs": len(pairs)}
    return main, inter, intercept, info


def check_sum_preserved(main_raw, inter_raw, main_c, inter_c, intercept, atol=1e-4):
    """Sanity check: intercept + centered sum == raw sum, per patient.

    Centering must not change what the model predicts, only how credit is
    split between terms. Returns (ok: bool, max_abs_diff: float).
    """
    raw_total = main_raw.sum(dim=1) + inter_raw.sum(dim=1)
    centered_total = intercept + main_c.sum(dim=1) + inter_c.sum(dim=1)
    diff = (raw_total - centered_total).abs()
    return bool((diff < atol).all().item()), float(diff.max().item())


class GA2MHead(nn.Module):
    """Main effects (per-region MLP) + low-rank bilinear pairwise interactions."""

    def __init__(self, num_regions: int, d_model: int = 384, rank: int = 16,
                 main_hidden: int = 64, use_residualization: bool = False):
        super().__init__()
        self.num_regions = num_regions
        self.d_model = d_model
        self.rank = rank
        self.use_residualization = use_residualization
        self.pairs = region_pairs(num_regions)

        self.main_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, main_hidden),
                nn.GELU(),
                nn.Linear(main_hidden, 1),
            ) for _ in range(num_regions)
        ])

        # A_r, B_r in R^{k x d_model}, one pair of projections per region,
        # shared across every pair involving that region (Section 1).
        self.A = nn.ModuleList([nn.Linear(d_model, rank, bias=False) for _ in range(num_regions)])
        self.B = nn.ModuleList([nn.Linear(d_model, rank, bias=False) for _ in range(num_regions)])

    def raw_projections(self, region_embs: torch.Tensor):
        """region_embs: (B, R, d_model) -> main_raw (B,R), Ah (B,R,k), Bh (B,R,k)."""
        B, R, D = region_embs.shape
        assert R == self.num_regions

        main_raw = torch.stack(
            [self.main_mlps[r](region_embs[:, r, :]).squeeze(-1) for r in range(R)],
            dim=1,
        )  # (B, R)
        Ah = torch.stack([self.A[r](region_embs[:, r, :]) for r in range(R)], dim=1)  # (B, R, k)
        Bh = torch.stack([self.B[r](region_embs[:, r, :]) for r in range(R)], dim=1)  # (B, R, k)
        return main_raw, Ah, Bh

    def forward(self, region_embs: torch.Tensor):
        """Returns risk (B,), main_c (B,R), inter_c (B,P), intercept (scalar), info dict."""
        main_raw, Ah, Bh = self.raw_projections(region_embs)
        fn = residualize_and_recenter if self.use_residualization else center_decomposition
        main_c, inter_c, intercept, info = fn(main_raw, Ah, Bh, self.pairs)
        risk = intercept + main_c.sum(dim=1) + inter_c.sum(dim=1)
        return risk, main_c, inter_c, intercept, info

    def l1_interaction_penalty(self, inter_c: torch.Tensor) -> torch.Tensor:
        """Sparsity penalty on interaction magnitudes (Section 1)."""
        return inter_c.abs().mean()
