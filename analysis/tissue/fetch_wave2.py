"""Second-wave labelled tiles for the tissue classifier (Beast, env dltest; approved by Juan 2026-11-22).

  python -I fetch_wave2.py tiger     TIGER ROI masks (breast; TCGA-BRCA, RUMC, JB) -> tiger_index.csv
  python -I fetch_wave2.py brummer   Brummer ccRCC tiles (kidney; TCGA-KIRC and Helsinki) -> brummer_index.csv
  python -I fetch_wave2.py summary   counts per source, split, version and label; peek grids

Downloads: raw/tiger (Zenodo 6014422, CC BY-NC 4.0), raw/brummer (Zenodo 7898308, CC BY 4.0). TIGER's tissue-bcss folder repeats BCSS and is skipped.
TIGER is 0.5 um/px: 'full' is a 256 px tile (128 um), 'crop64' a 128 px tile (64 um, half the detail of
our 40x tiles), each labelled when one class covers 80% of its own field.
TCGA sites keep the split they already have (TCGA-UT and BCSS share one site map); new sites, and the
non-TCGA slides, are split among themselves.
"""
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.classes import BRUMMER_MAP, TIGER_MAP, brummer_group  # noqa: E402
from analysis.tissue.fetch_public import _row, _save  # noqa: E402
from analysis.tissue.splits import assign_groups, extend_splits, tss_of  # noqa: E402
from analysis.tissue.tiles import labelled_tiles, tissue_fraction  # noqa: E402

DATA = Path("/data/scratch/juan/datasets/tissue")
RAW, TILES = DATA / "raw", DATA / "tiles"
TIGER_ZIP = RAW / "tiger" / "roi-level-annotations.zip"
# A tile needs half its pixels darker than grey 220 (not near-white). Otsu is not used: inside an ROI there
# is no glass, so Otsu would split dark from pale tissue and drop pale stroma.
MIN_TISSUE, WHITE = 0.5, 220


def tcga_site_splits():
    """The existing TCGA site -> split map (TCGA-UT and BCSS)."""
    pub = pd.read_csv(DATA / "public_index.csv", keep_default_na=False)
    t = pub[pub["source"].isin(["tcgaut", "bcss"])]
    return t.groupby("group")["split"].first().to_dict()


def _tiger_part(names):
    from PIL import Image
    z = zipfile.ZipFile(TIGER_ZIP)
    rows = []
    for m in names:
        slide = m.split("/")[-1].split("_[")[0]
        img = np.asarray(Image.open(io.BytesIO(z.read(m.replace("/masks/", "/images/")))).convert("RGB"))
        mask = np.asarray(Image.open(io.BytesIO(z.read(m))))
        roi = Path(m).stem.split("_[")[1].rstrip("]").replace(", ", "-")
        is_tcga = slide.startswith("TCGA")
        group = tss_of(slide) if is_tcga else f"tiger:{slide}"
        patient = slide[:12] if is_tcga else slide
        # the 80% rule is applied to each version's own field, as for BCSS: 256 px (128 um) for 'full',
        # 128 px (64 um) for 'crop64'
        for version, size in (("full", 256), ("crop64", 128)):
            for x, y, code in labelled_tiles(mask, size):
                label = TIGER_MAP[code]
                if label is None or tissue_fraction(img[y:y + size, x:x + size], WHITE) < MIN_TISSUE:
                    continue
                p = _save(img[y:y + size, x:x + size], TILES / "tiger" / slide / f"{roi}_{x}_{y}_{version}.png")
                rows.append(_row("tiger", p, version, label, group, "BRCA", patient))
    return rows


def tiger():
    names = sorted(n for n in zipfile.ZipFile(TIGER_ZIP).namelist()
                   if "/tissue-cells/masks/" in n and n.endswith(".png"))
    parts = [names[i::12] for i in range(12)]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame([r for part in ex.map(_tiger_part, parts) for r in part])
    tc = df["group"].str.startswith("tiger:")
    split = extend_splits(tcga_site_splits(), df.loc[~tc, "group"].tolist(), df.loc[~tc, "type"].tolist())
    split.update(assign_groups(df.loc[tc, "group"].tolist(), df.loc[tc, "type"].tolist()))
    df["split"] = df["group"].map(split)
    df.to_csv(DATA / "tiger_index.csv", index=False)
    print(len(df), "rows")
    print(df.groupby(["split", "version", "label"]).size().unstack("label").fillna(0).astype(int))


def _brummer_part(names):
    from PIL import Image
    z = zipfile.ZipFile(RAW / "brummer" / "tissue_classification.zip")
    rows = []
    for n in names:
        folder, fname = n.split("/")[-2], n.split("/")[-1]
        label = BRUMMER_MAP[folder]
        if label is None:
            continue
        img = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
        if img.shape[0] != 256:  # Helsinki tiles are 300 px: keep the central 256
            o = (img.shape[0] - 256) // 2
            img = img[o:o + 256, o:o + 256]
        group, patient = brummer_group(fname)
        p = _save(img, TILES / "brummer" / folder / (Path(fname).stem.replace(" ", "_") + "_crop64.png"))
        rows.append(_row("brummer", p, "crop64", label, group, "KIRC", patient))
    return rows


def brummer():
    """Tiles look like 40x (nuclei 20-30 px; the source gives no pixel size), so they are used as they are,
    as the 64 um version only."""
    names = sorted(n for n in zipfile.ZipFile(RAW / "brummer" / "tissue_classification.zip").namelist()
                   if not n.endswith("/"))
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame([r for part in ex.map(_brummer_part, [names[i::12] for i in range(12)]) for r in part])
    tc = df["group"].str.startswith("brummer:")
    split = extend_splits(tcga_site_splits(), df.loc[~tc, "group"].tolist(), df.loc[~tc, "type"].tolist())
    split.update(assign_groups(df.loc[tc, "group"].tolist(), df.loc[tc, "type"].tolist()))
    df["split"] = df["group"].map(split)
    df.to_csv(DATA / "brummer_index.csv", index=False)
    print(len(df), "rows")
    print(df.groupby(["split", "label"]).size().unstack("label").fillna(0).astype(int))


def summary():
    from PIL import Image
    rng = np.random.default_rng(0)
    for f in ["tiger_index.csv", "brummer_index.csv", "segpath_index.csv"]:
        if not (DATA / f).exists():
            continue
        df = pd.read_csv(DATA / f, keep_default_na=False)
        print("==", f, len(df), "rows")
        print(df.groupby(["split", "version", "label"]).size().unstack("label").fillna(0).astype(int))
        for (src, lab), g in df[df["version"] == "crop64"].groupby(["source", "label"]):
            pick = g["path"].values[rng.choice(len(g), size=min(20, len(g)), replace=False)]
            ims = [np.asarray(Image.open(p).convert("RGB").resize((256, 256))) for p in pick]
            ims += [np.full((256, 256, 3), 255, np.uint8)] * (20 - len(ims))
            grid = np.vstack([np.hstack(ims[r * 5:(r + 1) * 5]) for r in range(4)])
            out = DATA / "peek" / src / f"{lab}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(grid).save(out)


if __name__ == "__main__":
    cmds = {"tiger": tiger, "brummer": brummer, "summary": summary}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
