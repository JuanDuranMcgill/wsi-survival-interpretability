"""Which kind of slide information predicts survival? (Juan, 2026-11-23: "are we capturing all possible ways of
information?") Beast, env audit, CPU only. Exploratory; reported as such.

  python -I info_kinds.py features [variant]   -> /data/processed/juan/pancan/info_kinds_<variant>/<type>/<slide>.npz
  python -I info_kinds.py evaluate [variant]   -> results/pancan/info_kinds_<variant>.csv and .json (aggregates only)

Five kinds of information per slide, each a feature family:
  composition    share of each class of the label set, and the number of tissue tiles (log)
  arrangement    which classes touch which (share of a tumour/stroma/immune tile's 8 neighbours that are tumour,
                 stroma or immune), and how the tumour is split (pieces per 100 tumour tiles, largest piece's share)
  heterogeneity  spread of tile embeddings inside tumour, stroma and immune, and how much 1 mm regions of tumour
                 (16 x 16 tiles at 40x) and of all tissue differ from each other (the HIPT idea of region context)
  morphology     mean UNI-2 embedding of each compartment (what the tissue looks like)
  global         mean UNI-2 embedding of all tissue tiles (the slide as a whole)
Each family, and every combination, is scored as the paper's models are: a Cox model stratified by cancer type
(ridge penalty 0.1; morphology and global reduced to principal components fit on training patients), trained on
the hospital-held-out rounds of analysis/pancan/splits.py, scored by the within-type C-index on out-of-bag
patients. Exact Shapley values over the five families split the C-index gain among them. Clinical variables
(age, sex, stage, grade) are scored alone and with all five families, for reference.
"""
import json
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.tissue.classes import CLASSES  # noqa: E402

PANCAN = Path("/data/processed/juan/pancan")
RESULTS = ROOT / "results" / "pancan"
STEP, REGION, SAMPLE, SEED = 256, 16, 2000, 0
FAMILIES = ["composition", "arrangement", "heterogeneity", "morphology", "global"]
TRIO = ["tumour", "stroma", "immune"]
N_PCS, PENALIZER, N_ROUNDS = 16, 0.1, 20


def _grid(coords):
    g = np.asarray(coords, dtype=np.int64) // STEP
    return g - g.min(0)


def spatial_features(coords, labels):
    g = _grid(coords)
    grid = np.full(tuple(g.max(0) + 3), -1, dtype=np.int64)
    grid[g[:, 0] + 1, g[:, 1] + 1] = labels
    out = {}
    nb = np.stack([grid[g[:, 0] + 1 + dx, g[:, 1] + 1 + dy]
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy], 1)   # (n, 8)
    tissue = nb >= 0
    for a in TRIO:
        m = labels == CLASSES.index(a)
        tot = tissue[m].sum()
        for b in TRIO:
            out[f"{a}_to_{b}"] = float((nb[m] == CLASSES.index(b)).sum() / tot) if tot else np.nan
    from scipy.ndimage import label as cc
    tum = grid == CLASSES.index("tumour")
    lab, n = cc(tum, structure=np.ones((3, 3)))
    n_t = int(tum.sum())
    out["tumour_pieces_per_100"] = 100.0 * n / n_t if n_t else np.nan
    out["largest_tumour_piece_share"] = float(np.bincount(lab.ravel())[1:].max() / n_t) if n_t else np.nan
    return out


def region_dispersion(coords, feats, tiles_per_region=REGION, min_tiles=8):
    """Mean squared distance of 1 mm region means from their average (regions with >= min_tiles tiles)."""
    g = _grid(coords) // tiles_per_region
    keys, inv, counts = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    sums = np.zeros((len(keys), feats.shape[1]))
    np.add.at(sums, inv, feats)
    ok = counts >= min_tiles
    if ok.sum() < 2:
        return np.nan
    means = sums[ok] / counts[ok, None]
    return float(((means - means.mean(0)) ** 2).sum(1).mean())


def family_shapley(v, names):
    R = len(names)
    phi = {}
    for i, n in enumerate(names):
        others = [m for m in names if m != n]
        tot = 0.0
        for k in range(R):
            w = factorial(k) * factorial(R - k - 1) / factorial(R)
            for s in combinations(others, k):
                tot += w * (v[frozenset(s) | {n}] - v[frozenset(s)])
        phi[n] = tot
    return phi


def _slide(args):
    import torch
    label_path, emb_path, out, kept = args
    if out.exists():
        return "skip"
    lab = torch.load(label_path, map_location="cpu", weights_only=False)["label"].numpy().astype(int)
    f = torch.load(emb_path, map_location="cpu", weights_only=False)
    X = f["uni2"].float().numpy()
    coords = np.asarray(f["coords"])
    rng = np.random.default_rng(SEED)
    tissue = lab != CLASSES.index("other")
    feats = {f"share_{c}": float((lab == CLASSES.index(c)).mean()) for c in kept}
    feats["log_tissue_tiles"] = float(np.log1p(tissue.sum()))
    feats.update(spatial_features(coords, lab))
    for c in TRIO:
        m = np.flatnonzero(lab == CLASSES.index(c))
        if len(m) > SAMPLE:
            m = rng.choice(m, SAMPLE, replace=False)
        feats[f"spread_{c}"] = float(((X[m] - X[m].mean(0)) ** 2).sum(1).mean()) if len(m) > 1 else np.nan
    t = lab == CLASSES.index("tumour")
    feats["regions_tumour"] = region_dispersion(coords[t], X[t]) if t.sum() else np.nan
    feats["regions_tissue"] = region_dispersion(coords[tissue], X[tissue]) if tissue.sum() else np.nan
    morph = np.stack([X[lab == CLASSES.index(c)].mean(0) if (lab == CLASSES.index(c)).any()
                      else np.full(X.shape[1], np.nan) for c in kept if c != "other"])
    glob = X[tissue].mean(0) if tissue.any() else np.full(X.shape[1], np.nan)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(list(feats)), values=np.array(list(feats.values()), dtype=float),
             morphology=morph.astype(np.float32), global_=glob.astype(np.float32))
    return "done"


def features(variant="stroma", workers=8):
    from concurrent.futures import ProcessPoolExecutor
    from analysis.tissue.label_run import paths_for
    weights, labels, _ = paths_for(None if variant == "main" else variant)
    kept = ["tumour"] + [c for c in CLASSES[1:] if c in {str(x) for x in np.load(weights)["classes2"]}]
    jobs = [(f, PANCAN / "features" / f.parent.name / f.name.replace(".pt", "_emb16.pt"),
             PANCAN / f"info_kinds_{variant}" / f.parent.name / f.name.replace(".pt", ".npz"), kept)
            for f in sorted(labels.glob("*/*.pt"))]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_slide, jobs))
    print(variant, kept, {k: res.count(k) for k in set(res)})


def _load(variant):
    import pandas as pd
    c = pd.read_csv(PANCAN / "cohort.csv")
    rows, morph, glob = [], [], []
    for _, r in c.iterrows():
        p = PANCAN / f"info_kinds_{variant}" / r["type"] / f"{r['slide']}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rows.append({"patient": r["patient"], **dict(zip(z["names"], z["values"]))})
        morph.append(z["morphology"])
        glob.append(z["global_"])
    d = pd.DataFrame(rows).set_index("patient")
    return d, np.stack(morph), np.stack(glob)


def _clinical(patients):
    from analysis.pancan.check_r import clinical_features
    X = clinical_features(patients)
    return X[[c for c in X.columns if not c.startswith("type_")]]


def _design(fams, d, morph, glob, clin, tr):
    """Features of the families in fams; dense blocks reduced by PCA fit on training rows (tr)."""
    blocks, low = [], []
    if "composition" in fams:
        low += [c for c in d.columns if c.startswith("share_") or c == "log_tissue_tiles"]
    if "arrangement" in fams:
        low += [c for c in d.columns if "_to_" in c or "tumour_piece" in c]
    if "heterogeneity" in fams:
        low += [c for c in d.columns if c.startswith("spread_") or c.startswith("regions_")]
    if low:
        blocks.append(d[low].to_numpy(dtype=float))
    dense = []
    if "morphology" in fams:
        dense += [morph[:, k] for k in range(morph.shape[1])]
    if "global" in fams:
        dense.append(glob)
    for D in dense:
        miss = np.isnan(D).any(1)
        D = np.where(miss[:, None], np.nan, D)
        mu = np.nanmean(D[tr], 0)
        Dc = np.where(miss[:, None], 0.0, D - mu)
        k = N_PCS // 2 if "morphology" in fams and D is not glob else N_PCS
        V = np.linalg.svd(Dc[tr & ~miss], full_matrices=False)[2][:k]
        blocks.append(np.column_stack([Dc @ V.T, miss.astype(float)]))
    if clin is not None:
        blocks.append(clin)
    X = np.column_stack(blocks)
    med = np.nanmedian(X[tr], 0)
    X = np.where(np.isnan(X), med, X)
    sd = X[tr].std(0)
    keep = sd > 1e-9
    return (X[:, keep] - X[tr][:, keep].mean(0)) / sd[keep]


def _score(X, t, e, ty, tr):
    import pandas as pd
    from lifelines import CoxPHFitter
    from analysis.pancan.metrics import cindex_within
    cols = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols).assign(T=t, E=e, type=ty)
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df[tr], duration_col="T", event_col="E", strata=["type"])
    risk = cph.predict_partial_hazard(df[~tr]).to_numpy()
    return cindex_within(t[~tr], e[~tr], np.log(risk), ty[~tr])


def evaluate(variant="stroma", n_rounds=N_ROUNDS):
    import pandas as pd
    from analysis.pancan.splits import site_rounds
    d, morph, glob = _load(variant)
    c = pd.read_csv(PANCAN / "cohort.csv").set_index("patient").loc[d.index]
    t, e, ty = c["pfi_time"].to_numpy(float), c["pfi_event"].to_numpy(int), c["type"].to_numpy()
    clin = _clinical(d.index.values).to_numpy(dtype=float)
    rounds = site_rounds(pd.read_csv(PANCAN / "cohort.csv"), n_rounds)
    subsets = [frozenset(s) for k in range(1, len(FAMILIES) + 1) for s in combinations(FAMILIES, k)]
    rows = []
    for r in range(n_rounds):
        intr = set(rounds.loc[(rounds["round"] == r) & rounds["in_train"], "patient"])
        tr = d.index.isin(intr)
        for s in subsets:
            rows.append({"round": r, "set": "+".join(f for f in FAMILIES if f in s),
                         "cindex": _score(_design(s, d, morph, glob, None, tr), t, e, ty, tr)})
        rows.append({"round": r, "set": "clinical",
                     "cindex": _score(_design(frozenset(), d, morph, glob, clin, tr), t, e, ty, tr)})
        rows.append({"round": r, "set": "clinical+all",
                     "cindex": _score(_design(frozenset(FAMILIES), d, morph, glob, clin, tr), t, e, ty, tr)})
        print(f"round {r} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"info_kinds_{variant}_rounds.csv", index=False)
    mean = res.groupby("set")["cindex"].mean()
    v = {frozenset(): 0.5, **{s: float(mean["+".join(f for f in FAMILIES if f in s)]) for s in subsets}}
    phi = family_shapley(v, FAMILIES)
    summ = res.groupby("set")["cindex"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    summ.to_csv(RESULTS / f"info_kinds_{variant}.csv")
    out = {"variant": variant, "patients": int(len(d)), "events": int(e.sum()), "rounds": n_rounds,
           "alone": {f: v[frozenset([f])] for f in FAMILIES}, "all_five": v[frozenset(FAMILIES)],
           "clinical": float(mean["clinical"]), "clinical_plus_all": float(mean["clinical+all"]),
           "shapley_over_kinds": phi, "note": "exploratory; partial cohort; numbers are not results"}
    (RESULTS / f"info_kinds_{variant}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    var = sys.argv[2] if len(sys.argv) > 2 else "stroma"
    features(var) if cmd == "features" else evaluate(var)
