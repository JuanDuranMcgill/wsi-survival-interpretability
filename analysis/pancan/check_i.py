"""Check I for the pan-carcinoma runs: exact Shapley over compartments, within-type C-index as payoff (Beast, env audit).

  python -I check_i.py <run_dir> [--device cuda]    run_dir = .../runs/<variant>/ours/<subset>

Per round (as analysis/region_shapley.py run_round, from the round's best epoch by OOB within-type C-index): the
model's per-compartment embeddings of the OOB patients, the risk of every coalition of compartments, the exact
Shapley values with the within-type C-index as payoff (global) and with each patient's risk as payoff (local).
Writes <run_dir>/shapley/round_###.json and round_###_local.npz in the paper's format, then the paper's
region_shapley.aggregate -> <run_dir>/shapley/region_shapley.json. Verdicts: analysis/pancan/verdicts.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from region_ablation import IPGGraphFormer  # noqa: E402
from region_shapley import (aggregate, cache_region_embeddings, coalition_risks,  # noqa: E402
                            interaction_from_table, shapley_from_table)

from analysis.pancan.metrics import cindex_within  # noqa: E402
from analysis.pancan.train import BagDataset, collate  # noqa: E402


def shapley_round(round_dir, table, R, out_dir, device="cpu"):
    o = np.load(round_dir / "oob.npz")
    best = int(np.nanargmax(o["cindex"])) + 1
    oob = table.set_index("patient").loc[o["patient"]].reset_index()
    types = oob["type"].values
    model = IPGGraphFormer(num_regions=R)
    model.load_state_dict(torch.load(round_dir / f"epoch_{best:02d}.pt", map_location="cpu"))
    model.to(device).eval()
    loader = DataLoader(BagDataset(oob), batch_size=16, shuffle=False, collate_fn=collate)
    E, times, events, risk_full = cache_region_embeddings(model, loader, device)
    tab = coalition_risks(model, E, device)
    full = (1 << R) - 1
    head_check = float(np.max(np.abs(tab[full] - risk_full)))
    if head_check > 1e-3:
        raise RuntimeError(f"cached-embedding head differs from the model by {head_check}")
    v = np.array([cindex_within(times, events, tab[m], types) for m in range(1 << R)])
    phi = shapley_from_table(v, R)
    phi_loc = shapley_from_table(tab, R)
    I_loc = interaction_from_table(tab, R)
    share = np.abs(phi_loc) / np.maximum(np.abs(phi_loc).sum(0, keepdims=True), 1e-12)
    risk_sd = float(np.std(risk_full)) or 1.0
    w = torch.relu(model.region_weights.detach().cpu()) + 0.01
    rnd = int(round_dir.name.split("_")[1])
    rec = {"round": rnd, "best_epoch": best, "n_oob": len(oob), "oob_events": int(events.sum()),
           "removal": "zeroed", "baseline_cindex": float(v[full]), "empty_coalition_cindex": float(v[0]),
           "head_check_max_abs_diff": head_check, "efficiency_gap": float(abs(phi.sum() - (v[full] - v[0]))),
           "fusion_weights": (w / w.sum()).numpy().tolist(), "v_perf": v.tolist(), "phi_perf": phi.tolist(),
           "interaction_perf": interaction_from_table(v, R).tolist(),
           "deletion_delta": [float(v[full & ~(1 << r)] - v[full]) for r in range(R)],
           "keep_only_gain": [float(v[1 << r] - v[0]) for r in range(R)],
           "local_share_mean": share.mean(1).tolist(), "local_share_sd_across_patients": share.std(1).tolist(),
           "local_signed_mean_in_risk_sd": (phi_loc.mean(1) / risk_sd).tolist(),
           "local_interaction_mean_in_risk_sd": (I_loc.mean(2) / risk_sd).tolist(),
           "payoff": "within-cancer-type out-of-bag concordance"}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"round_{rnd:03d}_local.npz", patient_ids=np.asarray(oob["patient"], dtype=str),
                        times=times, events=events, types=np.asarray(types, dtype=str),
                        coalition_risks=tab.astype(np.float32), phi_local=phi_loc.astype(np.float32))
    (out_dir / f"round_{rnd:03d}.json").write_text(json.dumps(rec))
    return rec


def main(argv=None):
    from analysis.pancan.train import patient_table
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    variant, subset = run.parts[-3], run.parts[-1]
    table = patient_table(variant, subset)
    comps = torch.load(table["bag"].iat[0], map_location="cpu", weights_only=False)["compartments"]
    out = run / "shapley"
    for rd in sorted(run.glob("round_*")):
        if (rd / "oob.npz").exists() and not (out / f"{rd.name}.json").exists():
            r = shapley_round(rd, table, len(comps), out, a.device)
            print(f"{rd.name}: baseline within-type C {r['baseline_cindex']:.4f}, best epoch {r['best_epoch']}")
    aggregate(str(out), comps, str(out / "region_shapley.json"), f"pancan-{variant}-{subset}")


if __name__ == "__main__":
    main()
