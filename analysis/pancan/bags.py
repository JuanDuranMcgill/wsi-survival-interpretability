"""Per-slide bags for training: each slide's UNI-2 tiles grouped by tissue compartment (Beast, env pilot or audit).

  python -I bags.py main|stroma [ctx]   -> /data/processed/juan/pancan/bags_<variant>[_ctx]/<type>/<slide>.pt
  (ctx adds per tile the tissue mix of its neighbourhood, 'ctx', and kNN edges within each compartment, 'knn')

A compartment is a class the label set keeps, except 'other' (left out on purpose; docs/TISSUE_CLASSIFIER_RESULTS.md).
At most CAP tiles per compartment, drawn with a seed per slide (Claude's call, so all 6,940 slides fit in memory and
on disk: about 110 GB per label set). Each file: compartments, feats (list of n_c x 1536 float16), coords (list of
n_c x 2), n_tiles (tiles per compartment before the cap), labels (sha256 of the label file's classifier), cap.
Re-runnable: slides already bagged from the same classifier and cap are skipped.
"""
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.classes import CLASSES  # noqa: E402

FEAT = Path("/data/processed/juan/pancan/features")
BAGS = Path("/data/processed/juan/pancan")
CAP, SEED = 1024, 0
STEP = 256  # tile size in level-0 pixels (the run's grid)


def compartments(weights_path):
    """Compartments of a label set: tumour plus the step-2 classes it keeps, without 'other', in CLASSES order."""
    kept = {"tumour"} | {str(c) for c in np.load(weights_path, allow_pickle=False)["classes2"]}
    return [c for c in CLASSES if c in kept and c != "other"]


def make_bag(feats, coords, label, comps, cap=CAP, seed=SEED):
    rng = np.random.default_rng(seed)
    bag = {}
    for c in comps:
        idx = np.flatnonzero(np.asarray(label) == CLASSES.index(c))
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        bag[c] = (feats[idx], coords[idx])
    return bag


def neighbour_context(coords, label, idx, radius=2):
    """For the tiles idx: share of each class among the tiles in the (2r+1) x (2r+1) window around each (itself
    excluded), on the slide's full label grid. radius 2 = about 160 um at 40x. Zero where no tile is around."""
    r = radius
    g = np.asarray(coords, dtype=np.int64) // STEP
    g = g - g.min(0)
    H, W = (g.max(0) + 1).tolist()
    K = len(CLASSES)
    S = np.zeros((H + 2 * r + 1, W + 2 * r + 1, K), dtype=np.int32)
    S[g[:, 0] + r + 1, g[:, 1] + r + 1, np.asarray(label)] = 1
    S = S.cumsum(0).cumsum(1)
    x, y = g[idx, 0], g[idx, 1]                      # window [x, x + 2r] x [y, y + 2r] in padded grid rows 1..
    cnt = S[x + 2 * r + 1, y + 2 * r + 1] - S[x, y + 2 * r + 1] - S[x + 2 * r + 1, y] + S[x, y]
    cnt = cnt.astype(np.float64)
    cnt[np.arange(len(idx)), np.asarray(label)[idx]] -= 1
    tot = cnt.sum(1, keepdims=True)
    return np.divide(cnt, tot, out=np.zeros_like(cnt), where=tot > 0)


def knn_edges(coords, k=16):
    """Symmetric k-nearest-neighbour edges (2, E) among tiles, no self loops (the paper's model used knn_k=16)."""
    from scipy.spatial import cKDTree
    c = np.asarray(coords, dtype=float)
    n = len(c)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    kk = min(k, n - 1)
    _, nb = cKDTree(c).query(c, kk + 1)
    i = np.repeat(np.arange(n), kk)
    j = nb[:, 1:].ravel()
    e = np.unique(np.concatenate([np.stack([i, j]), np.stack([j, i])], 1), axis=1)
    return e[:, e[0] != e[1]].astype(np.int64)


def _bag_one(args):
    import torch
    label_path, out, comps, cap, ctx = args
    lab = torch.load(label_path, map_location="cpu", weights_only=False)
    if out.exists():
        try:
            o = torch.load(out, map_location="cpu", weights_only=False)
            if o.get("labels") == lab["classifier"] and o.get("cap") == cap and ("ctx" in o) == ctx:
                return "skip_done"
        except Exception:  # noqa: BLE001 - a broken bag is redone
            pass
    f = torch.load(FEAT / label_path.parent.name / label_path.name.replace(".pt", "_emb16.pt"),
                   map_location="cpu", weights_only=False)
    label = lab["label"].numpy().astype(int)
    feats, coords = f["uni2"], f["coords"]
    assert len(feats) == len(label), label_path
    seed = SEED + zlib.crc32(label_path.stem.encode())
    bag = make_bag(np.arange(len(label)), np.arange(len(label)), label, comps, cap, seed)
    keep = [bag[c][0] for c in comps]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    rec = {"compartments": comps, "feats": [feats[k].clone() for k in keep],
           "coords": [torch.as_tensor(np.asarray(coords)[k]) for k in keep],
           "n_tiles": [int((label == CLASSES.index(c)).sum()) for c in comps],
           "labels": lab["classifier"], "cap": cap}
    if ctx:  # neighbourhood context and within-compartment kNN graph (2026-11-24, Juan: "yes, test it")
        cn = np.asarray(coords)
        rec["ctx"] = [torch.from_numpy(neighbour_context(cn, label, k).astype(np.float16)) for k in keep]
        rec["knn"] = [torch.from_numpy(knn_edges(cn[k])) for k in keep]
    torch.save(rec, tmp)
    tmp.replace(out)
    return "bagged"


def main(variant, workers=8, ctx=False):
    from concurrent.futures import ProcessPoolExecutor
    from analysis.tissue.label_run import paths_for
    weights, labels, _ = paths_for(None if variant == "main" else variant)
    comps = compartments(weights)
    out_dir = BAGS / (f"bags_{variant}" + ("_ctx" if ctx else ""))
    jobs = [(f, out_dir / f.parent.name / f.name, comps, CAP, ctx) for f in sorted(labels.glob("*/*.pt"))]
    counts = {}
    with ProcessPoolExecutor(workers) as ex:
        for s in ex.map(_bag_one, jobs):
            counts[s] = counts.get(s, 0) + 1
    print(variant, comps, counts)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main", ctx="ctx" in sys.argv[2:])
