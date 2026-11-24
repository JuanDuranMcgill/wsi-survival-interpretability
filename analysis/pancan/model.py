"""The paper's per-tissue model with two optional switches (2026-11-24, Juan: "yes, test it").

  use_graph  the paper's graph part: each tile mixed with its k nearest tiles of the same compartment over 3 hops
             (IPGGraphFormer.sparse_diffusion, learned hop weights). The paper trained with it but scored without it;
             here it is used in both.
  use_ctx    the tissue mix around each tile (share of each class in the 5 x 5 tiles around it, bags.py
             neighbour_context) added to the tile's embedding, so a tumour tile can be seen as touching immune
             cells or stroma. Arrangement carried information no other kind had (docs/PANCAN_AUDIT_NOTES.md).
With both switches off the model is exactly the paper's (tested).
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from region_ablation import IPGGraphFormer  # noqa: E402


class PancanFormer(IPGGraphFormer):
    def __init__(self, num_regions, use_graph=False, use_ctx=False, n_ctx=9, **kw):
        super().__init__(num_regions, **kw)
        self.use_graph, self.use_ctx = use_graph, use_ctx
        if use_ctx:
            self.ctx_mlp = nn.ModuleList([nn.Sequential(nn.Linear(n_ctx, self.d_model), nn.ReLU(),
                                                        nn.Linear(self.d_model, self.d_model))
                                          for _ in range(num_regions)])

    def bag(self, feats, pos, A, ctx, ri):
        device = self.proj[ri].weight.device
        H = self.proj[ri](feats.float().to(device)) + self.pos_mlp[ri](pos.float().to(device))
        if self.use_ctx and ctx is not None:
            H = H + self.ctx_mlp[ri](ctx.float().to(device))
        if H.size(0) > 1:
            if self.use_graph and A is not None:
                H = self.sparse_diffusion(H, A.to(device), ri)
            for layer in self.agt_layers[ri]:
                H = layer(H)
            w = torch.softmax(self.attn_pool[ri](H).squeeze(-1), dim=0)
            return (w.unsqueeze(1) * H).sum(dim=0)
        return H.squeeze(0)

    def forward_batch(self, feats_list, pos_list, A_list=None, ctx_list=None):
        device = next(self.parameters()).device
        out = []
        for si in range(len(feats_list)):
            embs = []
            for ri in range(len(feats_list[si])):
                f = feats_list[si][ri]
                if f is None:
                    embs.append(torch.zeros(self.d_model, device=device))
                    continue
                A = A_list[si][ri] if A_list is not None else None
                c = ctx_list[si][ri] if ctx_list is not None else None
                embs.append(self.bag(f, pos_list[si][ri], A, c, ri))
            E = torch.stack(embs)
            attn, _ = self.cross_region_attn(E.unsqueeze(0), E.unsqueeze(0), E.unsqueeze(0), need_weights=False)
            E = self.cross_region_norm(E + attn.squeeze(0))
            w = torch.relu(self.region_weights) + 0.01
            out.append(((w / w.sum()).unsqueeze(1) * E).sum(dim=0))
        return self.head(torch.stack(out)).squeeze(-1)
