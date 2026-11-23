"""Label every finished slide of the pan-carcinoma run from its saved embeddings (Beast, env pilot, CPU).

  python -I label_run.py [stroma]          -> /data/processed/juan/pancan/tissue_labels[_stroma]/<type>/<slide>.pt
  python -I label_run.py shares [stroma]   -> results/pancan/tissue_threshold_kept[_stroma].csv (share of tiles
                                              each class's confidence cut keeps, over all labelled slides)

'stroma' is the sensitivity variant that keeps stroma as its own class (Juan, 2026-11-22); without it, the
main classifier (stroma in 'other', as the fixed merge table says).
Each output: label (uint8, index into CLASSES), probs (float16, n x 9), classifier (sha256 of the weights file).
Uses the embedding the classifier names (UNI-2). Skips files without it and slides already labelled by the
same classifier.
Re-runnable: run again after the preprocessing run ends.
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FEAT = Path("/data/processed/juan/pancan/features")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "pancan"


def paths_for(variant):
    """(weights file, labels directory, kept-share csv) of the main classifier (None) or a variant."""
    sfx = f"_{variant}" if variant else ""
    return (RESULTS / f"tissue_classifier_weights{sfx}.npz",
            Path(f"/data/processed/juan/pancan/tissue_labels{sfx}"),
            RESULTS / f"tissue_threshold_kept{sfx}.csv")


def _label_one(args):
    import numpy as np
    import torch
    from analysis.tissue.train_eval import load_classifier
    f, digest, weights, labels = args
    f = Path(f)
    out = Path(labels) / f.parent.name / f.name.replace("_emb16.pt", ".pt")
    if out.exists():
        try:
            if torch.load(out, map_location="cpu", weights_only=False).get("classifier") == digest:
                return "skip_done", str(f)
        except Exception:  # noqa: BLE001 - a broken label file is redone
            pass
    o = torch.load(f, map_location="cpu", weights_only=False)
    clf = load_classifier(weights)
    if clf.feature not in o:
        return "skip_no_feature", str(f)
    P = clf(o[clf.feature].float().numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"label": torch.from_numpy(P.argmax(1).astype(np.uint8)),
                "probs": torch.from_numpy(P.astype(np.float16)), "classifier": digest}, tmp)
    os.replace(tmp, out)
    return "labelled", str(f)


def kept_counts(label, probs, thresholds):
    """Per class: tiles labelled that class, and how many of them reach its confidence cut (NaN = no cut)."""
    import numpy as np
    k = probs.shape[1]
    n_lab = np.bincount(label, minlength=k)
    p_own = probs[np.arange(len(label)), label]
    thr = np.nan_to_num(np.asarray(thresholds, dtype=float)[label], nan=-np.inf)
    n_kept = np.bincount(label[p_own >= thr], minlength=k)
    return n_lab, n_kept


def shares(variant=None):
    import numpy as np
    import pandas as pd
    import torch
    from analysis.tissue.classes import CLASSES
    weights, labels, out = paths_for(variant)
    thr = np.load(weights, allow_pickle=False)["thresholds"]
    n_lab, n_kept = np.zeros(len(CLASSES), int), np.zeros(len(CLASSES), int)
    files = sorted(labels.glob("*/*.pt"))
    for f in files:
        o = torch.load(f, map_location="cpu", weights_only=False)
        a, b = kept_counts(o["label"].numpy().astype(int), o["probs"].float().numpy(), thr)
        n_lab += a
        n_kept += b
    df = pd.DataFrame({"class": CLASSES, "threshold": thr, "tiles_labelled": n_lab, "tiles_kept": n_kept})
    df["share_kept"] = df["tiles_kept"] / df["tiles_labelled"].where(df["tiles_labelled"] > 0)
    df.to_csv(out, index=False)
    print(len(files), "slides"); print(df.to_string(index=False))


def main(variant=None, workers=8):
    weights, labels, _ = paths_for(variant)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    files = sorted(str(f) for f in FEAT.glob("*/*_emb16.pt"))
    counts, skipped = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for status, f in ex.map(_label_one, [(f, digest, str(weights), str(labels)) for f in files]):
            counts[status] = counts.get(status, 0) + 1
            if status == "skip_no_feature":
                skipped.append(f)
    print(counts)
    for f in skipped:
        print("no classifier feature:", f)


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = "stroma" if "stroma" in args else None
    shares(variant) if "shares" in args else main(variant)
