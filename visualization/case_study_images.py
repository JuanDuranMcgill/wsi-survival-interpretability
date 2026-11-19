#!/usr/bin/env python
"""
Images for the case study (docs/HANDOFF_CASE_STUDY.md). The BRCA images were first made on
Narval (slides under $SCRATCH/openpath-data/tcga/slides/TCGA-BRCA); the BLCA slides and the
per-patient attribution are on Trillium (slides under /scratch/sorkwos/TCGA-BLCA*). Per
patient, in paper_figures/case_study/{patient}/:

- thumbnail.png: the slide, long side 1,500 px;
- tissue_map.png: the CONCH compartment of every tile the model used, same size as the
  thumbnail. Tiles are drawn from the coordinates in the class-wise embedding files
  (level-0 pixels, 256 px tiles), coloured with the paper's compartment palette
  (generate_paper_figures.REGION_COLORS; the questionnaire's colours are not on Narval);
- tiles/{compartment}_{k}.png: three 256 px tiles from each of the patient's two
  compartments with the largest attribution (results/case_study/{patient}.json,
  attribution.top2, from analysis/case_study_attribution.py), or the largest tile share
  when attribution is unavailable. Tiles are drawn with a fixed seed among tiles that
  are mostly tissue;
- images.json: sizes, scale, colours, tile coordinates and tile counts per compartment.

On Narval, in a CPU job with the TRIDENT env (OpenSlide + torch):
    source ~/openpath-agents/jobs/trident_env.sh
    $TPY visualization/case_study_images.py --cohort brca
On Trillium:
    python visualization/case_study_images.py --cohort blca --slides "/scratch/$USER/TCGA-BLCA*/**/*.svs"
    python visualization/case_study_images.py --cohort brca --slides "/scratch/$USER/TCGA-BRCA*/**/*.svs" \
        --emb /scratch/$USER/UNI2_classwise_embeddings_BRCA --tiles-only
--tiles-only redoes only the example tiles and their entries in images.json, and deletes the
tiles it replaces.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figures", "case_study")
CASES = os.path.join(ROOT, "results", "case_study")
CANDIDATES = {"brca": ["TCGA-B6-A0X1", "TCGA-B6-A0I9", "TCGA-B6-A0RG", "TCGA-D8-A1JL", "TCGA-LL-A8F5"],
              "blca": ["TCGA-ZF-AA53", "TCGA-FT-A3EE", "TCGA-2F-A9KT", "TCGA-K4-A5RI", "TCGA-ZF-A9RF"]}
DEFAULT_SLIDES = {"brca": os.path.expandvars("$SCRATCH/openpath-data/tcga/slides/TCGA-BRCA/*.svs"),
                  "blca": "/scratch/sorkwos/TCGA-BLCA*/**/*.svs"}
DEFAULT_EMB = {"brca": "/scratch/sorkwos/UNI2_classwise_embeddings_BRCA",
               "blca": "/scratch/sorkwos/UNI2_classwise_embeddings"}
LONG_SIDE, TILE = 1500, 256
# generate_paper_figures.REGION_COLORS, keyed by the compartment folder names.
KEY_BLCA = {"Perivesical adipose tissue (extravesical fat)": "adipose",
            "Invasive urothelial carcinoma (tumor)": "tumor",
            "Normal urothelium (benign mucosa)": "urothelium",
            "Inflammatory infiltrates (immune cells)": "immune",
            "Lamina propria (fibrovascular stroma)": "lamina",
            "Blood vessels (vasculature)": "vessels",
            "Muscularis propria (detrusor muscle)": "muscularis", "Necrosis": "necrosis"}
KEY_BRCA = {"Adipose tissue (fat)": "adipose", "Blood vessels (vasculature)": "vessels",
       "Ductal carcinoma in situ (DCIS)": "dcis", "Fibrous desmoplastic stroma": "stroma",
       "Invasive breast carcinoma (tumor cells)": "tumor",
       "Muscle tissue (smooth or skeletal muscle)": "muscle", "Necrosis or hemorrhage": "necrosis",
       "Normal breast glands and lobules (TDLU)": "tdlu",
       "Tumor-infiltrating lymphocytes (immune infiltrates)": "tils"}
COLORS = {'tumor': '#DC2626', 'urothelium': '#2563EB', 'adipose': '#D97706', 'vessels': '#7C3AED',
          'lamina': '#059669', 'muscularis': '#0891B2', 'immune': '#DB2777', 'necrosis': '#78716C',
          'dcis': '#EA580C', 'tils': '#9333EA', 'stroma': '#0369A1', 'tdlu': '#EC4899', 'muscle': '#65A30D'}


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def tissue_fraction(img):
    """Share of pixels that are not background (white/near-white or very dark)."""
    a = np.asarray(img.convert("RGB"), float)
    sat = a.max(2) - a.min(2)
    return float(((sat > 20) & (a.mean(2) < 225)).mean())


def pick_tiles(s, coords, folders, key, out):
    rng = np.random.default_rng(0)
    tiles = {}
    for folder in folders:
        c = coords[folder]; picked = []
        for j in rng.permutation(len(c))[:200]:
            x, y = (int(v) for v in c[j])
            img = s.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
            if tissue_fraction(img) >= 0.6:
                k = len(picked) + 1
                name = f"{key[folder]}_{k}.png"
                img.save(os.path.join(out, "tiles", name))
                picked.append({"file": f"tiles/{name}", "x": x, "y": y, "level": 0, "size": TILE})
            if len(picked) == 3:
                break
        tiles[folder] = picked
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--slides", default=None, help="glob for the .svs files (recursive ** allowed)")
    ap.add_argument("--emb", default=None, help="class-wise embedding root")
    ap.add_argument("--tiles-only", action="store_true")
    args = ap.parse_args()
    key = KEY_BLCA if args.cohort == "blca" else KEY_BRCA
    emb = args.emb or DEFAULT_EMB[args.cohort]
    all_svs = glob.glob(args.slides or DEFAULT_SLIDES[args.cohort], recursive=True)

    for pid in CANDIDATES[args.cohort]:
        by_name = {os.path.basename(p): p for p in all_svs
                   if os.path.basename(p).startswith(f"{pid}-01Z") and "DX1" in os.path.basename(p)}
        if len(by_name) != 1:      # the same slide may sit in two download folders
            print(f"{pid}: {len(by_name)} slides found, skipping"); continue
        svs = next(iter(by_name.values()))
        stem = os.path.basename(svs)[:-4]
        out = os.path.join(OUT, pid); os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
        s = openslide.OpenSlide(svs)
        W, H = s.dimensions

        coords, counts = {}, {}
        for folder in key:
            f = os.path.join(emb, folder, stem + ".pt")
            if os.path.exists(f):
                c = np.asarray(torch.load(f, map_location="cpu", weights_only=False)["coords"])
                coords[folder] = c; counts[folder] = int(len(c))
            else:
                counts[folder] = 0
        total = sum(counts.values())

        attr = json.load(open(os.path.join(CASES, f"{pid}.json")))["attribution"]
        if attr.get("available"):
            top2 = [f for f in attr["top2"] if counts.get(f, 0) > 0]
            rule = ("two compartments with the largest attribution (results/case_study, attribution.top2: "
                    "mean |local Shapley| share over out-of-bag rounds of the main Shapley run); ")
        else:
            top2 = sorted(counts, key=lambda k: -counts[k])[:2]
            rule = "two compartments with the largest tile share (attribution unavailable); "
        rule += "3 tiles each, fixed seed 0, among tiles with >= 60% tissue pixels"

        if args.tiles_only:
            meta = json.load(open(os.path.join(out, "images.json")))
            old = [t["file"] for v in meta["example_tiles"].values() for t in v]
            tiles = pick_tiles(s, coords, top2, key, out)
            new = {t["file"] for v in tiles.values() for t in v}
            for fpath in old:
                if fpath not in new and os.path.exists(os.path.join(out, fpath)):
                    os.remove(os.path.join(out, fpath))
            meta["example_tiles"] = {key[f]: v for f, v in tiles.items()}
            meta["example_tiles_rule"] = rule
            json.dump(meta, open(os.path.join(out, "images.json"), "w"), indent=2)
            print(f"{pid}: tiles redone from {[key[f] for f in top2]}, "
                  f"{[len(v) for v in tiles.values()]} tiles, removed {sorted(set(old) - new)}", flush=True)
            continue

        thumb = s.get_thumbnail((LONG_SIDE, LONG_SIDE)).convert("RGB")
        tw, th = thumb.size
        sx, sy = tw / W, th / H
        thumb.save(os.path.join(out, "thumbnail.png"))
        # tissue map: white background, one rectangle per tile
        tmap = Image.new("RGB", (tw, th), (255, 255, 255))
        dr = ImageDraw.Draw(tmap)
        for folder, c in coords.items():
            col = hex_rgb(COLORS[key[folder]])
            for x, y in c:
                x0, y0 = x * sx, y * sy
                dr.rectangle([x0, y0, max(x0 + 1, (x + TILE) * sx - 1), max(y0 + 1, (y + TILE) * sy - 1)], fill=col)
        tmap.save(os.path.join(out, "tissue_map.png"))
        Image.blend(thumb, tmap, 0.45).save(os.path.join(out, "tissue_overlay.png"))

        tiles = pick_tiles(s, coords, top2, key, out)
        meta = {"patient_id": pid, "slide": os.path.basename(svs), "level0_size": [W, H],
                "thumbnail_size": [tw, th], "scale_thumb_per_level0": [sx, sy],
                "mpp": s.properties.get("openslide.mpp-x"), "objective_power": s.properties.get("openslide.objective-power"),
                "tile_size_level0": TILE,
                "tiles_per_compartment_in_embeddings": counts, "n_tiles_total": total,
                "colours": {key[f]: COLORS[key[f]] for f in key},
                "colour_source": "visualization/generate_paper_figures.py REGION_COLORS (questionnaire colours not on the cluster)",
                "example_tiles": {key[f]: v for f, v in tiles.items()},
                "example_tiles_rule": rule}
        json.dump(meta, open(os.path.join(out, "images.json"), "w"), indent=2)
        print(f"{pid}: thumb {tw}x{th}, {total} tiles, top2 {[key[f] for f in top2]}, "
              f"tiles {[len(v) for v in tiles.values()]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
