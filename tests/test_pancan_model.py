import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _inputs(R=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats, pos, A, ctx = [], [], [], []
    for _ in range(3):
        fs, ps, As, cs = [], [], [], []
        for _ in range(R):
            n = 6
            fs.append(torch.randn(n, 1536, generator=g))
            ps.append(torch.rand(n, 2, generator=g))
            e = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
            deg = torch.bincount(e[0], minlength=n).clamp_min(1).float()
            As.append(torch.sparse_coo_tensor(e, 1.0 / deg[e[0]], (n, n)).coalesce())
            cs.append(torch.rand(n, 9, generator=g))
        feats.append(fs); pos.append(ps); A.append(As); ctx.append(cs)
    return feats, pos, A, ctx


def test_switches_off_reproduce_the_papers_model():
    from analysis.pancan.model import PancanFormer
    from region_ablation import IPGGraphFormer
    torch.manual_seed(0)
    base = IPGGraphFormer(num_regions=2).eval()
    ours = PancanFormer(num_regions=2).eval()
    ours.load_state_dict(base.state_dict())
    feats, pos, A, ctx = _inputs()
    with torch.no_grad():
        assert torch.allclose(base.forward_batch(feats, pos), ours.forward_batch(feats, pos, A, ctx), atol=1e-6)


@pytest.mark.parametrize("graph,context", [(True, False), (False, True), (True, True)])
def test_graph_and_context_change_the_risk_and_learn(graph, context):
    from analysis.pancan.model import PancanFormer
    torch.manual_seed(0)
    m = PancanFormer(num_regions=2, use_graph=graph, use_ctx=context)
    feats, pos, A, ctx = _inputs()
    r = m.forward_batch(feats, pos, A, ctx)
    assert torch.isfinite(r).all()
    r.sum().backward()
    if graph:
        assert m.beta[0].grad is not None and m.beta[0].grad.abs().sum() > 0
    if context:
        assert m.ctx_mlp[0][0].weight.grad.abs().sum() > 0
