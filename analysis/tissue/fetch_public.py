"""Fetch and tile the public labelled sets for the tissue classifier (runs on the Beast).

  python -I fetch_public.py download   TCGA-UT (external split tars), Kather NONORM, BCSS from Hugging Face (env pilot)
  python -I fetch_public.py tile       write tiles and public_index.csv (env dltest: needs pyarrow)
  python -I fetch_public.py summary    counts, split check, 20 random tiles per source and label to peek/

Versions: 'crop64' is a 64 um field (our 40x tiles), 'full' a 112-128 um field (our 20x tiles).
Data stay in /data/scratch/juan/datasets/tissue/ (not in git).
"""
import glob
import io
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.classes import BCSS_MAP, KATHER_MAP, tcgaut_type  # noqa: E402
from analysis.tissue.splits import assign_groups, tss_of  # noqa: E402
from analysis.tissue.tiles import center_crop, labelled_tiles  # noqa: E402

DATA = Path("/data/scratch/juan/datasets/tissue")
RAW = DATA / "raw"
TILES = DATA / "tiles"
INDEX = DATA / "public_index.csv"
KATHER7K = Path("/data/scratch/juan/datasets/kather_crc_val_7k/CRC-VAL-HE-7K")
HF_CACHE = Path("/data/scratch/juan/cache/huggingface")
SOURCES = {  # repo id -> files we need
    "dakomura/tcga-ut": ["README.md", "config.yaml", "data/dataset_external_*"],
    "1aurent/NCT-CRC-HE": ["README.md", "data/NCT_CRC_HE_100K_NONORM-*"],
    "MedOtter/BCSS": ["*"],
}
KATHER_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]  # parquet label order
CROP, SEED = 128, 0


def download():
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_HOME"] = str(HF_CACHE)
    from huggingface_hub import snapshot_download
    for repo, patterns in SOURCES.items():
        dest = RAW / repo.replace("/", "__")
        print("downloading", repo, "->", dest, flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                          local_dir=str(dest), max_workers=8)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print("done", repo, f"{size / 1e9:.2f} GB", flush=True)


def _save(img, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return str(path)


def _row(source, path, version, label, group, typ, patient):
    return {"tile_id": f"{source}/{Path(path).stem}", "source": source, "path": path, "version": version,
            "label": label, "group": group, "type": typ, "patient": patient}


def _tile_tcgaut_tar(tar_path):
    from PIL import Image
    rows, pending = [], {}
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile():
                continue
            stem, ext = m.name.rsplit(".", 1)
            pending.setdefault(stem, {})[ext] = t.extractfile(m).read()
            item = pending[stem]
            if "jpg" not in item or "json" not in item:
                continue
            del pending[stem]
            typ = tcgaut_type(json.loads(item["json"])["label"])
            if typ is None:
                continue
            barcode, name = stem.split("/")
            img = np.asarray(Image.open(io.BytesIO(item["jpg"])).convert("RGB"))
            base = TILES / "tcgaut" / barcode / name
            for version, arr in (("full", img), ("crop64", center_crop(img, CROP))):
                p = _save(arr, base.with_name(f"{barcode}_{name}_{version}.png"))
                rows.append(_row("tcgaut", p, version, "tumour", tss_of(barcode), typ, barcode[:12]))
    return rows


def _tile_kather_shard(shard):
    import pyarrow.parquet as pq
    from PIL import Image
    rows = []
    tab = pq.read_table(shard).to_pydict()
    for i, (im, lab) in enumerate(zip(tab["image"], tab["label"])):
        code = KATHER_CODES[lab]
        img = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
        name = Path(im["path"] or f"{Path(shard).stem}_{i}").stem
        for version, arr in (("full", img), ("crop64", center_crop(img, CROP))):
            p = _save(arr, TILES / "kather100k" / code / f"{name}_{version}.png")
            rows.append(_row("kather100k", p, version, KATHER_MAP[code], "kather", "COAD", ""))
    return rows


def _tile_kather7k(code):
    from PIL import Image
    rows = []
    for f in sorted((KATHER7K / code).glob("*.tif")):
        img = np.asarray(Image.open(f).convert("RGB"))
        for version, arr in (("full", img), ("crop64", center_crop(img, CROP))):
            p = _save(arr, TILES / "kather7k" / code / f"{f.stem}_{version}.png")
            rows.append(_row("kather7k", p, version, KATHER_MAP[code], "kather", "COAD", ""))
    return rows


def _tile_bcss_shard(shard):
    import pyarrow.parquet as pq
    from PIL import Image
    rows = []
    tab = pq.read_table(shard).to_pydict()
    for im, mk, iid, pid in zip(tab["image"], tab["mask"], tab["image_id"], tab["patient_id"]):
        img = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
        mask = np.asarray(Image.open(io.BytesIO(mk["bytes"])))
        h2, w2 = mask.shape[0] // 2, mask.shape[1] // 2
        img2 = np.asarray(Image.fromarray(img).resize((w2, h2), Image.BILINEAR))
        mask2 = np.asarray(Image.fromarray(mask).resize((w2, h2), Image.NEAREST))
        # crop64: 256 px at the native 0.25 um/px; full: 256 px after 2x downsampling (0.5 um/px)
        for version, im_v, mk_v in (("crop64", img, mask), ("full", img2, mask2)):
            for x, y, code in labelled_tiles(mk_v, 256):
                label = BCSS_MAP[code]
                if label is None:
                    continue
                p = _save(im_v[y:y + 256, x:x + 256], TILES / "bcss" / pid / f"{iid}_{x}_{y}_{version}.png")
                rows.append(_row("bcss", p, version, label, tss_of(pid), "BRCA", pid))
    return rows


def tile(workers=12):
    jobs = [(_tile_tcgaut_tar, f) for f in sorted(glob.glob(str(RAW / "dakomura__tcga-ut/data/*.tar")))]
    jobs += [(_tile_kather_shard, f) for f in sorted(glob.glob(str(RAW / "1aurent__NCT-CRC-HE/data/*.parquet")))]
    jobs += [(_tile_bcss_shard, f) for f in sorted(glob.glob(str(RAW / "MedOtter__BCSS/data/*.parquet")))]
    jobs += [(_tile_kather7k, c) for c in KATHER_CODES]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_call, jobs):
            rows += r
            print(len(rows), "tiles", flush=True)
    idx = pd.DataFrame(rows)
    tcga = idx["source"].isin(["tcgaut", "bcss"])
    site_split = assign_groups(idx.loc[tcga, "group"], idx.loc[tcga, "type"], seed=SEED)
    idx["split"] = idx["group"].map(site_split)
    idx.loc[idx["source"] == "kather100k", "split"] = "train"
    idx.loc[idx["source"] == "kather7k", "split"] = "test"
    idx.to_csv(INDEX, index=False)
    print("wrote", INDEX, len(idx), "rows")


def _call(job):
    fn, arg = job
    return fn(arg)


def summary():
    idx = pd.read_csv(INDEX, keep_default_na=False)
    pd.set_option("display.width", 250)
    print(idx.pivot_table(index=["source", "label"], columns=["split", "version"], values="tile_id",
                          aggfunc="count", fill_value=0).to_string())
    t = idx[(idx["source"] == "tcgaut") & (idx["version"] == "crop64")]
    print(t.pivot_table(index="type", columns="split", values="tile_id", aggfunc="count", fill_value=0).to_string())
    print("types with tcgaut test tiles:", t[t["split"] == "test"]["type"].nunique(), "of 17")
    tcga = idx[idx["source"].isin(["tcgaut", "bcss"])]
    print("sites in more than one split:", int((tcga.groupby("group")["split"].nunique() > 1).sum()))
    pats = tcga.groupby("patient")["split"].nunique()
    print("patients in more than one split:", int((pats > 1).sum()))
    from PIL import Image
    peek = DATA / "peek" / "public"
    peek.mkdir(parents=True, exist_ok=True)
    full = idx[idx["version"] == "full"]
    for (s, lab), g in full.groupby(["source", "label"]):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((224, 224))) for p in
               g.sample(min(20, len(g)), random_state=SEED)["path"]]
        if len(ims) >= 5:
            rows = [np.concatenate(ims[i:i + 5], 1) for i in range(0, len(ims) - len(ims) % 5, 5)]
            Image.fromarray(np.concatenate(rows, 0)).save(peek / f"{s}_{lab}.png")


if __name__ == "__main__":
    cmds = {"download": download, "tile": tile, "summary": summary}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
