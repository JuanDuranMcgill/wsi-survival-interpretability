import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _bags(tmp_path, n=12, R=2, dim=1536):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        feats = [torch.randn(int(rng.integers(0, 6)), dim, dtype=torch.float16) for _ in range(R)]
        coords = [torch.from_numpy(rng.integers(0, 5000, (len(f), 2))) for f in feats]
        p = tmp_path / f"s{i}.pt"
        torch.save({"compartments": ["tumour", "fat"][:R], "feats": feats, "coords": coords}, p)
        rows.append({"patient": f"P{i}", "type": "A" if i % 2 else "B", "pfi_time": float(rng.integers(10, 900)),
                     "pfi_event": int(i % 3 == 0), "bag": str(p)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kind", ["ours", "abmil"])
def test_train_round_runs_on_cpu_and_writes_oob_results(tmp_path, kind):
    from analysis.pancan.train import train_round
    t = _bags(tmp_path)
    tr, oob = t.iloc[:8], t.iloc[8:]
    c = train_round(kind, 2, tr, oob, tmp_path / "out", epochs=2, device="cpu", batch_size=4, workers=0)
    o = np.load(tmp_path / "out" / "oob.npz")
    assert o["risk"].shape == (2, 4) and np.isfinite(o["risk"]).all()
    assert len(c) == 2 and (tmp_path / "out" / "epoch_02.pt").exists()
    assert o["region_weights"].shape == ((2, 2) if kind == "ours" else (0, 2))


def test_shapley_round_reproduces_the_model_and_is_efficient(tmp_path):
    import json
    from analysis.pancan.check_i import shapley_round
    from analysis.pancan.train import train_round
    t = _bags(tmp_path, n=16)
    rd = tmp_path / "run" / "round_000"
    train_round("ours", 2, t.iloc[:8], t.iloc[8:], rd, epochs=1, device="cpu", batch_size=4, workers=0)
    rec = shapley_round(rd, t, 2, tmp_path / "run" / "shapley")
    assert rec["head_check_max_abs_diff"] < 1e-3
    assert rec["efficiency_gap"] < 1e-9
    assert json.loads((tmp_path / "run" / "shapley" / "round_000.json").read_text())["round"] == 0
