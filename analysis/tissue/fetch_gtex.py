"""Fetch and tile GTEx normal tissue for the tissue classifier (runs on the Beast, env dltest).

  python -I fetch_gtex.py select     choose 20 donors per tissue (seed 0) -> gtex_slides.csv
  python -I fetch_gtex.py download   s5cmd download of the chosen slides from IDC
  python -I fetch_gtex.py tile       tissue tiles, 128 px (64 um) and 256 px (128 um) at level 0 -> gtex_index.csv
  python -I fetch_gtex.py summary    counts per tissue x split; 20 random tiles per tissue to peek/

Every GTEx tile is 'not tumour' (donors without cancer). Data stay in /data/scratch/juan/datasets/.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.classes import GTEX_TISSUES  # noqa: E402
from analysis.tissue.splits import assign_groups  # noqa: E402
from analysis.tissue.tiles import center_crop, otsu_threshold, tissue_fraction  # noqa: E402

DATA = Path("/data/scratch/juan/datasets/tissue")
RAW = Path("/data/scratch/juan/datasets/gtex")
SLIDES = DATA / "gtex_slides.csv"
INDEX = DATA / "gtex_index.csv"
DONORS_PER_TISSUE, TILES_PER_SLIDE, MIN_TISSUE, SEED = 20, 300, 0.5, 0
FULL, CROP = 256, 128  # px at level 0 (0.49 um/px): 125 um and 63 um fields


def select():
    from idc_index import IDCClient
    c = IDCClient()
    c.fetch_index("sm_index")
    q = c.sql_query("select i.SeriesInstanceUID series, i.PatientID donor, s.ContainerIdentifier specimen, "
                    "s.primaryAnatomicStructure_CodeMeaning tissue, s.min_PixelSpacing_2sf spacing_mm, "
                    "i.series_size_MB mb from sm_index s join index i on s.SeriesInstanceUID = i.SeriesInstanceUID "
                    "where i.collection_id = 'gtex'")
    tissues = sorted({t for ts in GTEX_TISSUES.values() for t in ts})
    missing = [t for t in tissues if t not in set(q["tissue"])]
    if missing:
        sys.exit(f"GTEx tissues not in IDC: {missing}")
    rng = np.random.default_rng(SEED)
    rows = []
    for t in tissues:
        g = q[q["tissue"] == t].sort_values("series").drop_duplicates("donor")
        rows.append(g.iloc[rng.permutation(len(g))[:DONORS_PER_TISSUE]])
    sel = pd.concat(rows)
    sel["types"] = sel["tissue"].map(lambda t: ";".join(k for k, v in GTEX_TISSUES.items() if t in v))
    sel.to_csv(SLIDES, index=False)
    print(sel.groupby("tissue").agg(slides=("series", "size"), gb=("mb", lambda x: x.sum() / 1e3)).round(1).to_string())
    print("total GB", round(sel["mb"].sum() / 1e3, 1))


def download():
    from idc_index import IDCClient
    sel = pd.read_csv(SLIDES)
    RAW.mkdir(parents=True, exist_ok=True)
    IDCClient().download_from_selection(seriesInstanceUID=sel["series"].tolist(), downloadDir=str(RAW),
                                        dirTemplate="%SeriesInstanceUID")


def _tile_slide(row):
    from wsidicom import WsiDicom
    out_dir = DATA / "tiles" / "gtex" / row["donor"]
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with WsiDicom.open(str(RAW / row["series"])) as wsi:
            W, H = wsi.size.width, wsi.size.height
            thumb = np.asarray(wsi.read_thumbnail((W // 32, H // 32)).convert("RGB"))
            gray = thumb.mean(-1)
            thr = otsu_threshold(gray.astype(np.uint8))
            dark = gray < thr
            cand = []
            for y in range(0, H - FULL + 1, FULL):
                for x in range(0, W - FULL + 1, FULL):
                    m = dark[y // 32:(y + FULL) // 32, x // 32:(x + FULL) // 32]
                    if m.size and m.mean() >= MIN_TISSUE:
                        cand.append((x, y))
            rng = np.random.default_rng(SEED)
            pick = [cand[i] for i in rng.permutation(len(cand))[:TILES_PER_SLIDE]]
            rows = []
            from PIL import Image
            for x, y in pick:
                full = np.asarray(wsi.read_region((x, y), 0, (FULL, FULL)).convert("RGB"))
                for version, img in (("full", full), ("crop64", center_crop(full, CROP))):
                    if tissue_fraction(img, thr) < MIN_TISSUE:
                        continue
                    p = out_dir / f"{row['series'][-12:]}_{x}_{y}_{version}.png"
                    Image.fromarray(img).save(p)
                    rows.append({"tile_id": f"gtex/{p.stem}", "source": "gtex", "path": str(p),
                                 "version": version, "label": "not_tumour", "group": row["donor"],
                                 "tissue": row["tissue"], "types": row["types"], "otsu": thr})
        return rows, None
    except Exception as e:  # noqa: BLE001 - one unreadable slide must not stop the rest
        return [], f"{row['series']} {row['tissue']}: {type(e).__name__}: {e}"


def tile(workers=8):
    sel = pd.read_csv(SLIDES)
    rows, errors = [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, err in ex.map(_tile_slide, sel.to_dict("records")):
            rows += r
            if err:
                errors.append(err)
                print("FAILED", err, flush=True)
    idx = pd.DataFrame(rows)
    split = assign_groups(sel["donor"], sel["tissue"], seed=SEED)
    idx["split"] = idx["group"].map(split)
    idx = idx.assign(type=idx["types"].str.split(";")).explode("type").drop(columns="types")
    idx.to_csv(INDEX, index=False)
    print(len(idx), "rows;", len(errors), "slides failed")


def summary():
    idx = pd.read_csv(INDEX)
    print(idx.drop_duplicates("tile_id").pivot_table(index="tissue", columns=["split", "version"],
                                                     values="tile_id", aggfunc="count").to_string())
    print("types with test tiles:", sorted(idx[idx["split"] == "test"]["type"].unique()))
    donors_in_splits = idx.groupby("group")["split"].nunique()
    print("donors in more than one split:", int((donors_in_splits > 1).sum()))
    peek = DATA / "peek" / "gtex"
    peek.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for t, g in idx[idx["version"] == "full"].drop_duplicates("tile_id").groupby("tissue"):
        ims = [np.asarray(Image.open(p)) for p in g.sample(min(20, len(g)), random_state=SEED)["path"]]
        rows = [np.concatenate(ims[i:i + 5], 1) for i in range(0, len(ims) - len(ims) % 5, 5)]
        Image.fromarray(np.concatenate(rows, 0)).save(peek / f"{t.replace(' ', '_')}.png")


if __name__ == "__main__":
    cmds = {"select": select, "download": download, "tile": tile, "summary": summary}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
