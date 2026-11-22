"""Does averaging a tile's class probabilities with its neighbours' help at the 64 um field? (BCSS only.)

Approved by Juan 2026-11-22 (about 5 minutes of shared GPU). BCSS regions are large (about 4,000 px at
0.25 um/px), so every labelled 64 um tile has unlabelled neighbours on the same grid as our run's tiles.

  python -I smooth_test.py cut     every 256 px grid tile with tissue, native 0.25 um/px (env dltest)
  python -I smooth_test.py embed   CONCH embeddings of those tiles (env pilot, GPU)
  python -I smooth_test.py eval    single-tile vs 3x3 and 5x5 neighbour means, held-out BCSS (env pilot, CPU)
"""
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("HF_HOME", "/data/scratch/juan/cache/huggingface")  # before any HF import
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DATA = Path("/data/scratch/juan/datasets/tissue")
GRID = DATA / "tiles" / "bcss_grid"
INDEX = DATA / "bcss_grid_index.csv"
EMB = DATA / "emb_extra" / "bcss_grid.pt"
OUT = Path(__file__).resolve().parents[2] / "results" / "pancan" / "tissue_smoothing_bcss.csv"
TILE, MIN_TISSUE = 256, 0.5


def neighbour_mean(image, gx, gy, P, radius):
    """Mean of P over the tiles of the same image within `radius` grid steps (a square), itself included."""
    pos = {(i, x, y): k for k, (i, x, y) in enumerate(zip(image, gx, gy))}
    S, n = np.zeros_like(P, dtype=float), np.zeros(len(P))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for k, (i, x, y) in enumerate(zip(image, gx, gy)):
                j = pos.get((i, x + dx, y + dy))
                if j is not None:
                    S[k] += P[j]
                    n[k] += 1
    return S / n[:, None]


def _cut_shard(shard):
    import pyarrow.parquet as pq
    from PIL import Image
    from analysis.tissue.classes import BCSS_MAP
    from analysis.tissue.tiles import labelled_tiles, otsu_threshold, tissue_fraction
    rows = []
    tab = pq.read_table(shard).to_pydict()
    for im, mk, iid, pid in zip(tab["image"], tab["mask"], tab["image_id"], tab["patient_id"]):
        img = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
        mask = np.asarray(Image.open(io.BytesIO(mk["bytes"])))
        thr = otsu_threshold(img[::8, ::8].mean(-1).astype(np.uint8))
        lab = {(x, y): BCSS_MAP[c] for x, y, c in labelled_tiles(mask, TILE)}
        h, w = mask.shape[:2]
        for y in range(0, h - TILE + 1, TILE):
            for x in range(0, w - TILE + 1, TILE):
                t = img[y:y + TILE, x:x + TILE]
                label = lab.get((x, y)) or ""
                if not label and tissue_fraction(t, thr) < MIN_TISSUE:
                    continue
                p = GRID / pid / f"{iid}_{x}_{y}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(t).save(p)
                rows.append({"tile_id": f"bcssgrid|{iid}|{x}|{y}", "image_id": iid, "patient": pid,
                             "gx": x // TILE, "gy": y // TILE, "label": label, "path": str(p)})
    return rows


def cut():
    from concurrent.futures import ProcessPoolExecutor
    from analysis.tissue.splits import tss_of
    shards = sorted(glob.glob(str(DATA / "raw/MedOtter__BCSS/data/*.parquet")))
    with ProcessPoolExecutor(8) as ex:
        rows = [r for part in ex.map(_cut_shard, shards) for r in part]
    df = pd.DataFrame(rows)
    pub = pd.read_csv(DATA / "public_index.csv", keep_default_na=False)
    split_of = pub[pub["source"] == "bcss"].groupby("group")["split"].first().to_dict()
    df["split"] = df["patient"].map(tss_of).map(split_of)
    df.to_csv(INDEX, index=False)
    print(len(df), "grid tiles;", (df["label"] != "").sum(), "labelled; by split:",
          df.groupby("split").size().to_dict())


def embed():
    import torch
    from conch.open_clip_custom import create_model_from_pretrained
    from PIL import Image
    df = pd.read_csv(INDEX, keep_default_na=False)
    dev = torch.device("cuda:0")
    model, pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", device=dev)
    model.eval()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            return pre(Image.open(df["path"].iat[i]).convert("RGB")).half()

    dl = torch.utils.data.DataLoader(DS(), batch_size=256, num_workers=8, pin_memory=True)
    out = []
    with torch.inference_mode():
        for c in dl:
            c = c.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out.append(model.encode_image(c).half().cpu())
    EMB.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tile_id": df["tile_id"].tolist(), "conch": torch.cat(out)}, EMB)
    print("embedded", len(df))


def eval_():
    import torch
    from analysis.tissue.bars import premerge_small
    from analysis.tissue.classes import CLASSES
    from analysis.tissue.metrics import recall_precision
    from analysis.tissue.train_eval import MIN_TRAIN_TILES, load, predict, train
    X, idx = load("conch")
    nt = idx[idx["source"].isin(["kather100k", "bcss"]) & (idx["label"] != "tumour")
             & (idx["split"] == "train") & (idx["version"] == "crop64")]
    merged = premerge_small(nt["label"].value_counts().to_dict(), MIN_TRAIN_TILES)
    model = train(X, idx, merged)  # the fine-class model (round 1), before any failed-class merge
    df = pd.read_csv(INDEX, keep_default_na=False)
    o = torch.load(EMB, map_location="cpu", weights_only=False)
    assert o["tile_id"] == df["tile_id"].tolist()
    P = predict(model, o["conch"].float().numpy())
    img = df["image_id"].values
    rows = []
    for radius in (0, 1, 2):
        S = neighbour_mean(img, df["gx"].values, df["gy"].values, P, radius) if radius else P
        for split in ("valid", "test"):
            m = (df["split"] == split) & (df["label"] != "")
            true = df.loc[m, "label"].map(lambda c: merged.get(c, c))
            pred = pd.Series(np.array(CLASSES)[S[m.values].argmax(1)], index=true.index)
            t = recall_precision(true, pred, CLASSES)
            t = t[t["n_true"] > 0].assign(split=split, window=f"{2 * radius + 1}x{2 * radius + 1}")
            rows.append(t)
    res = pd.concat(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(res.pivot_table(index=["split", "class"], columns="window", values=["recall", "precision"]).round(3))


if __name__ == "__main__":
    cmds = {"cut": cut, "embed": embed, "eval": eval_}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
