"""What is in 'other' when stroma is kept? (Juan, 2026-11-23.) Beast, env pilot, CPU only.

  python -I other_check.py        -> results/pancan/tissue_other_check_by_type.csv, tissue_other_check.json
  python -I other_check.py gtex   -> results/pancan/tissue_other_check_gtex.csv (normal GTEx tiles by organ)

On the stroma-kept labels of every labelled slide:
  - 'other' share per type, and per slide (are a few slides mostly 'other'?);
  - how sure the classifier is ('other' probability) and its second choice for those tiles;
  - whether 'other' tiles sit at the tissue edge (fewer than 8 of their 8 grid neighbours are tissue tiles);
  - what a fine-class model (step 2 of round 1, before any failed-class merge, same UNI-2 data and C) calls a
    sample of them: necrosis or haemorrhage, background/mucus/secretions ('other' in training), or a tissue.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")  # before numpy; the preprocessing run needs the CPU
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FEAT = Path("/data/processed/juan/pancan/features")
LABELS = Path("/data/processed/juan/pancan/tissue_labels_stroma")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "pancan"
STEP, SAMPLE, SEED = 256, 200, 0


def edge_mask(coords, step=STEP):
    """True where fewer than all 8 grid neighbours are tiles of the same slide (tissue edge or hole)."""
    g = np.asarray(coords, dtype=np.int64) // step
    g = g - g.min(0)
    grid = np.zeros(tuple(g.max(0) + 3), dtype=bool)
    grid[g[:, 0] + 1, g[:, 1] + 1] = True
    n = np.zeros(len(g), dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                n += grid[g[:, 0] + 1 + dx, g[:, 1] + 1 + dy]
    return n < 8


def _slide(path):
    import torch
    from analysis.tissue.classes import CLASSES
    other = CLASSES.index("other")
    path = Path(path)
    o = torch.load(path, map_location="cpu", weights_only=False)
    lab = o["label"].numpy().astype(int)
    P = o["probs"].float().numpy()
    f = torch.load(FEAT / path.parent.name / path.name.replace(".pt", "_emb16.pt"), map_location="cpu",
                   weights_only=False)
    coords = f["coords"].numpy() if hasattr(f["coords"], "numpy") else np.asarray(f["coords"])
    assert len(coords) == len(lab), path
    edge = edge_mask(coords)
    oth = lab == other
    Q = P.copy()
    Q[np.arange(len(lab)), lab] = -1
    runner = Q.argmax(1)
    pick = np.flatnonzero(oth)
    if len(pick) > SAMPLE:
        pick = np.sort(np.random.default_rng(SEED).choice(pick, SAMPLE, replace=False))
    return {"type": path.parent.name, "slide": path.stem, "n": len(lab), "n_other": int(oth.sum()),
            "n_edge": int(edge.sum()), "n_other_edge": int((oth & edge).sum()),
            "runner": np.bincount(runner[oth], minlength=len(CLASSES)),
            "p_hist": np.histogram(P[oth, other], bins=np.linspace(0, 1, 11))[0],
            "emb": f["uni2"][pick].float().numpy()}


def fine_model():
    """Step 2 of round 1 (non-tumour classes, only too-small classes merged), with round 1's chosen C."""
    from sklearn.linear_model import LogisticRegression
    from analysis.tissue.bars import premerge_small
    from analysis.tissue.train_eval import MIN_TRAIN_TILES, STEP2_SOURCES, group_weights, load
    C = json.loads((RESULTS / "tissue_classifier_summary.json").read_text())["uni2"][0]["C2"]
    X, idx = load("uni2")
    nt = idx[idx["source"].isin(STEP2_SOURCES) & (idx["label"] != "tumour") & (idx["split"] == "train")]
    merged = premerge_small(nt[nt["version"] == "crop64"]["label"].value_counts().to_dict(), MIN_TRAIN_TILES)
    y = nt["label"].map(lambda c: merged.get(c, c)).values
    print("fine model: C =", C, "merged before training:", merged, "rows:", len(nt), flush=True)
    return LogisticRegression(C=C, max_iter=3000).fit(X[nt["row"]], y, sample_weight=group_weights(nt, y))


def main(workers=6):
    import pandas as pd
    from analysis.tissue.classes import CLASSES
    files = sorted(str(f) for f in LABELS.glob("*/*.pt"))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_slide, files))
    print(len(rows), "slides read", flush=True)
    d = pd.DataFrame([{k: r[k] for k in ("type", "slide", "n", "n_other", "n_edge", "n_other_edge")} for r in rows])
    d["other_share"] = d["n_other"] / d["n"]
    m = fine_model()
    emb = np.concatenate([r["emb"] for r in rows])
    fine = m.predict(emb)
    fine_type = np.concatenate([[r["type"]] * len(r["emb"]) for r in rows])
    runner = pd.DataFrame([r["runner"] for r in rows], columns=CLASSES).assign(type=d["type"])
    by = d.groupby("type").agg(slides=("slide", "size"), tiles=("n", "sum"), other=("n_other", "sum"),
                              edge=("n_edge", "sum"), other_edge=("n_other_edge", "sum"),
                              slide_median_other=("other_share", "median"),
                              slides_over_half_other=("other_share", lambda s: int((s > 0.5).sum())))
    by["other_share"] = by["other"] / by["tiles"]
    by["edge_share_all_tiles"] = by["edge"] / by["tiles"]
    by["edge_share_other_tiles"] = by["other_edge"] / by["other"]
    rs = runner.groupby("type").sum()
    for c in ("tumour", "normal_epithelium", "stroma", "immune", "fat"):
        by[f"second_{c}"] = rs[c] / rs.sum(1)
    ft = pd.crosstab(fine_type, fine, normalize="index")
    by = by.join(ft.add_prefix("fine_"))
    by.to_csv(RESULTS / "tissue_other_check_by_type.csv")
    hist = np.sum([r["p_hist"] for r in rows], 0)
    tot = {"slides": len(d), "tiles": int(d["n"].sum()), "other_tiles": int(d["n_other"].sum()),
           "other_share": float(d["n_other"].sum() / d["n"].sum()),
           "slide_other_share_quantiles": d["other_share"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict(),
           "slides_over_half_other": int((d["other_share"] > 0.5).sum()),
           "edge_share_all_tiles": float(d["n_edge"].sum() / d["n"].sum()),
           "edge_share_other_tiles": float(d["n_other_edge"].sum() / d["n_other"].sum()),
           "other_prob_histogram_0_to_1_by_0.1": hist.tolist(),
           "second_choice_share": (runner[CLASSES].sum() / runner[CLASSES].sum().sum()).round(4).to_dict(),
           "fine_class_share_of_sample": pd.Series(fine).value_counts(normalize=True).round(4).to_dict(),
           "sample_tiles": int(len(fine)),
           "top_other_slides_by_type": d.sort_values("other_share", ascending=False).head(15)[
               ["type", "n", "other_share"]].rename(columns={"n": "tiles"}).to_dict("records")}  # no slide IDs in git
    (RESULTS / "tissue_other_check.json").write_text(json.dumps(tot, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(tot, indent=2, default=float))
    print(by.round(3).to_string())


def gtex_shares():
    """Share of each class the stroma-kept classifier gives normal GTEx tiles (64 um), per organ.
    GTEx tiles carry no tissue label in step 2, so this shows where normal parenchyma of each organ lands."""
    import pandas as pd
    from analysis.tissue.classes import CLASSES
    from analysis.tissue.train_eval import load, load_classifier
    X, idx = load("uni2")
    g = idx[(idx["source"] == "gtex") & (idx["version"] == "crop64")].drop_duplicates("tile_id")
    clf = load_classifier(RESULTS / "tissue_classifier_weights_stroma.npz")
    lab = np.array(CLASSES)[clf(X[g["row"]]).argmax(1)]
    t = pd.crosstab(g["tissue"].values, lab, normalize="index")
    t.insert(0, "tiles", g.groupby("tissue").size())
    t.to_csv(RESULTS / "tissue_other_check_gtex.csv")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())


if __name__ == "__main__":
    gtex_shares() if "gtex" in sys.argv[1:] else main()
