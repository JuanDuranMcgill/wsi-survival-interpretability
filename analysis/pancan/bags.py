"""Per-slide bags for training: each slide's UNI-2 tiles grouped by tissue compartment (Beast, env pilot or audit).

  python -I bags.py main|stroma   -> /data/processed/juan/pancan/bags_<variant>/<type>/<slide>.pt

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


def _bag_one(args):
    import torch
    label_path, out, comps, cap = args
    lab = torch.load(label_path, map_location="cpu", weights_only=False)
    if out.exists():
        try:
            o = torch.load(out, map_location="cpu", weights_only=False)
            if o.get("labels") == lab["classifier"] and o.get("cap") == cap:
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
    torch.save({"compartments": comps, "feats": [feats[k].clone() for k in keep],
                "coords": [torch.as_tensor(np.asarray(coords)[k]) for k in keep],
                "n_tiles": [int((label == CLASSES.index(c)).sum()) for c in comps],
                "labels": lab["classifier"], "cap": cap}, tmp)
    tmp.replace(out)
    return "bagged"


def main(variant, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    from analysis.tissue.label_run import paths_for
    weights, labels, _ = paths_for(None if variant == "main" else variant)
    comps = compartments(weights)
    out_dir = BAGS / f"bags_{variant}"
    jobs = [(f, out_dir / f.parent.name / f.name, comps, CAP) for f in sorted(labels.glob("*/*.pt"))]
    counts = {}
    with ProcessPoolExecutor(workers) as ex:
        for s in ex.map(_bag_one, jobs):
            counts[s] = counts.get(s, 0) + 1
    print(variant, comps, counts)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
