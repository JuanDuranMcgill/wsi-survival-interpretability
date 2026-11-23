"""Planted-signal test for the pan-carcinoma runs (design decision 9; the paper's planted_signal.py, within type).

  python -I planted.py make <variant> <compartment> <target_c> [--specific]   (Beast, env audit)
      -> /data/processed/juan/pancan/planted/<variant>_<compartment>_c<100*target>[_specific].csv
         (patient, pfi_time, pfi_event); --specific uses specific_z (signal only in the planted compartment)
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


def specific_z(means_by_comp, types, planted, n_pcs=50):
    """The planted compartment's own signal (2026-11-23, Juan: 'yes, add it'). The paper's z also lives in the
    other compartments of the same slide (correlation 0.58-0.68 on 920 slides), so credit spreads and no audit
    could single the planted one out. Here the planted compartment's mean embedding is first cleared, by least
    squares, of cancer-type means and of the top n_pcs principal components of every other compartment (absent ->
    0 plus an indicator); z is that residual's strongest direction (its first principal component), standardised
    within type over the patients who have the compartment (absent -> 0). On 920 slides, n_pcs=50 left the other
    compartments predicting none of z (cross-validated R2 -0.02) but stroma itself only 0.16: check
    learnability() before training."""
    types = np.asarray(types)
    M = means_by_comp[planted]
    present = ~np.isnan(M).any(1)
    cols = [(types == t).astype(float) for t in np.unique(types)]
    for c, Mc in means_by_comp.items():
        if c == planted:
            continue
        ok = ~np.isnan(Mc).any(1)
        Xc = np.where(ok[:, None], np.nan_to_num(Mc), 0.0)
        Xc[ok] -= Xc[ok].mean(0)
        cols.append((~ok).astype(float))
        k = min(n_pcs, max(int(ok.sum()) - 1, 0), Xc.shape[1])
        if k:
            V = np.linalg.svd(Xc[ok], full_matrices=False)[2][:k]
            cols += list((Xc @ V.T).T)
    A = np.column_stack(cols)[present]
    Y = M[present]
    R = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    v = np.linalg.svd(R - R.mean(0), full_matrices=False)[2][0]
    r = R @ v
    z = np.zeros(len(M))
    zp = np.zeros(len(r))
    tp = types[present]
    for t in np.unique(tp):
        m = tp == t
        if m.sum() > 1:
            zp[m] = (r[m] - r[m].mean()) / (r[m].std() or 1.0)
    z[present] = zp
    return z


def learnability(z, means_by_comp, planted, folds=5):
    """Cross-validated ridge R2 of z from the planted compartment's mean embedding and from all the others'."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    sets = {"planted_itself": np.nan_to_num(means_by_comp[planted]),
            "other_compartments": np.hstack([np.nan_to_num(m) for c, m in means_by_comp.items() if c != planted])}
    out = {}
    for k, X in sets.items():
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X, z, cv=folds)
        out[k] = float(1 - np.mean((z - p) ** 2) / np.var(z))
    return out


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


def make(variant, compartment, target_c, seed=0, specific=False):
    import pandas as pd
    from planted_signal import synth_outcome
    from analysis.pancan.metrics import cindex_within
    from analysis.pancan.train import patient_table
    t = patient_table(variant)
    comps = torch_compartments(t)
    means = {c: compartment_means(t, c) for c in comps}
    z_paper = planted_z_within(means[compartment], t["type"].values, seed)
    z = specific_z(means, t["type"].values, compartment) if specific else z_paper
    leak = {c: float(np.corrcoef(z, planted_z_within(means[c], t["type"].values, seed))[0, 1])
            for c in comps if c != compartment}
    censor = 1 - t["pfi_event"].mean()
    times, events, beta, c_pooled = synth_outcome(z, target_c, censor, float(t["pfi_time"].median()), seed)
    c_within = cindex_within(times, events, beta * z, t["type"].values)
    out = PANCAN / "planted" / (f"{variant}_{compartment}_c{int(round(100 * target_c))}"
                                + ("_specific" if specific else "") + ".csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"patient": t["patient"], "pfi_time": times, "pfi_event": events}).to_csv(out, index=False)
    print(json.dumps({"file": str(out), "patients": len(t), "events": int(events.sum()), "beta": beta,
                      "oracle_c_pooled": c_pooled, "oracle_c_within_type": c_within,
                      "patients_with_compartment": int((z != 0).sum()),
                      "corr_with_paper_z": float(np.corrcoef(z, z_paper)[0, 1]),
                      "learnability_r2": learnability(z, means, compartment),
                      "corr_with_other_compartments_projection": leak}))


def torch_compartments(table):
    import torch
    return torch.load(table["bag"].iat[0], map_location="cpu", weights_only=False)["compartments"]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2], sys.argv[3], float(sys.argv[4]), specific="--specific" in sys.argv)
    elif cmd == "report":
        agg = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps({"planted": sys.argv[3], "ranks": ranks_of(agg["per_region"], sys.argv[3])}))
