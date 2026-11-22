"""Label every finished slide of the pan-carcinoma run from its saved CONCH embeddings (Beast, env pilot, CPU).

  python -I label_run.py   -> /data/processed/juan/pancan/tissue_labels/<type>/<slide>.pt

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
LABELS = Path("/data/processed/juan/pancan/tissue_labels")
WEIGHTS = Path(__file__).resolve().parents[2] / "results" / "pancan" / "tissue_classifier_weights.npz"


def _label_one(args):
    import numpy as np
    import torch
    from analysis.tissue.train_eval import load_classifier
    f, digest = args
    f = Path(f)
    out = LABELS / f.parent.name / f.name.replace("_emb16.pt", ".pt")
    if out.exists():
        try:
            if torch.load(out, map_location="cpu", weights_only=False).get("classifier") == digest:
                return "skip_done", str(f)
        except Exception:  # noqa: BLE001 - a broken label file is redone
            pass
    o = torch.load(f, map_location="cpu", weights_only=False)
    clf = load_classifier(WEIGHTS)
    if clf.feature not in o:
        return "skip_no_feature", str(f)
    P = clf(o[clf.feature].float().numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"label": torch.from_numpy(P.argmax(1).astype(np.uint8)),
                "probs": torch.from_numpy(P.astype(np.float16)), "classifier": digest}, tmp)
    os.replace(tmp, out)
    return "labelled", str(f)


def main(workers=8):
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    files = sorted(str(f) for f in FEAT.glob("*/*_emb16.pt"))
    counts, skipped = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for status, f in ex.map(_label_one, [(f, digest) for f in files]):
            counts[status] = counts.get(status, 0) + 1
            if status == "skip_no_feature":
                skipped.append(f)
    print(counts)
    for f in skipped:
        print("no classifier feature:", f)


if __name__ == "__main__":
    main()
