"""Planted-signal test for the pan-carcinoma runs (design decision 9; the paper's planted_signal.py, within type).

  python -I planted.py make <variant> <compartment> <target_c>   (Beast, env audit)
      -> /data/processed/juan/pancan/planted/<variant>_<compartment>_c<100*target>.csv (patient, pfi_time, pfi_event)
  train.py --variant <variant> --outcome <that csv> --tag planted_<compartment>_c<..>; check_i.py on the run dir;
  python -I planted.py report <run_dir>/shapley/region_shapley.json <compartment>

Outcome: z = projection of each patient's mean UNI-2 embedding of the planted compartment on a fixed random
direction, standardised within cancer type over the patients who have it (absent -> 0). Event times and censoring
are drawn by the paper's synth_outcome (hazard exp(beta z), censored share and median time of the real cohort,
beta set so the true risk reaches target_c). A measure passes if it ranks the planted compartment first.
Patient-level outcomes stay on the Beast.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

PANCAN = Path("/data/processed/juan/pancan")
MEASURES = ("fusion_weight", "shapley_cindex", "local_attribution_share")


def planted_z_within(means, types, seed):
    present = ~np.isnan(means).any(1)
    u = np.random.default_rng(seed).standard_normal(means.shape[1])
    u /= np.linalg.norm(u)
    z = np.zeros(len(means))
    proj = np.where(present, np.nan_to_num(means) @ u, np.nan)
    for t in np.unique(types):
        m = present & (np.asarray(types) == t)
        if m.sum() > 1:
            z[m] = (proj[m] - proj[m].mean()) / (proj[m].std() or 1.0)
    return z


def ranks_of(per_region, compartment):
    return {k: 1 + sum(per_region[n][k]["mean"] > per_region[compartment][k]["mean"] for n in per_region)
            for k in MEASURES}


def compartment_means(table, compartment):
    import torch
    out = []
    for b in table["bag"]:
        o = torch.load(b, map_location="cpu", weights_only=False)
        f = o["feats"][o["compartments"].index(compartment)]
        out.append(f.float().mean(0).numpy() if len(f) else np.full(1536, np.nan))
    return np.array(out, dtype=float)


def make(variant, compartment, target_c, seed=0):
    import pandas as pd
    from planted_signal import synth_outcome
    from analysis.pancan.metrics import cindex_within
    from analysis.pancan.train import patient_table
    t = patient_table(variant)
    z = planted_z_within(compartment_means(t, compartment), t["type"].values, seed)
    censor = 1 - t["pfi_event"].mean()
    times, events, beta, c_pooled = synth_outcome(z, target_c, censor, float(t["pfi_time"].median()), seed)
    c_within = cindex_within(times, events, beta * z, t["type"].values)
    out = PANCAN / "planted" / f"{variant}_{compartment}_c{int(round(100 * target_c))}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"patient": t["patient"], "pfi_time": times, "pfi_event": events}).to_csv(out, index=False)
    print(json.dumps({"file": str(out), "patients": len(t), "events": int(events.sum()), "beta": beta,
                      "oracle_c_pooled": c_pooled, "oracle_c_within_type": c_within,
                      "patients_with_compartment": int((z != 0).sum())}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    elif cmd == "report":
        agg = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps({"planted": sys.argv[3], "ranks": ranks_of(agg["per_region"], sys.argv[3])}))
