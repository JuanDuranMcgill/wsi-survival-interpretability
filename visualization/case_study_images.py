#!/usr/bin/env python
"""
Images for the case study (docs/HANDOFF_CASE_STUDY.md), for the candidates whose slides are
on Narval (the BRCA ones, under $SCRATCH/openpath-data/tcga/slides/TCGA-BRCA). Per patient,
in paper_figures/case_study/{patient}/:

- thumbnail.png: the slide, long side 1,500 px;
- tissue_map.png: the CONCH compartment of every tile the model used, same size as the
  thumbnail. Tiles are drawn from the coordinates in the class-wise embedding files
  (level-0 pixels, 256 px tiles), coloured with the paper's compartment palette
  (generate_paper_figures.REGION_COLORS; the questionnaire's colours are not on Narval);
- tiles/{compartment}_{k}.png: three 256 px tiles from each of the patient's two
  compartments with the largest tile share (attribution is not available; see
  results/case_study/{patient}.json). Tiles are drawn with a fixed seed among tiles that
  are mostly tissue;
- images.json: sizes, scale, colours, tile coordinates and tile counts per compartment.

Run in a CPU job with the TRIDENT env (OpenSlide + torch):
    source ~/openpath-agents/jobs/trident_env.sh
    $TPY visualization/case_study_images.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES = os.path.expandvars("$SCRATCH/openpath-data/tcga/slides/TCGA-BRCA")
EMB = "/scratch/sorkwos/UNI2_classwise_embeddings_BRCA"
OUT = os.path.join(ROOT, "paper_figures", "case_study")
CANDIDATES = ["TCGA-B6-A0X1", "TCGA-B6-A0I9", "TCGA-B6-A0RG", "TCGA-D8-A1JL", "TCGA-LL-A8F5"]
LONG_SIDE, TILE = 1500, 256
# generate_paper_figures.REGION_COLORS, keyed by the BRCA compartment folder names.
KEY = {"Adipose tissue (fat)": "adipose", "Blood vessels (vasculature)": "vessels",
       "Ductal carcinoma in situ (DCIS)": "dcis", "Fibrous desmoplastic stroma": "stroma",
       "Invasive breast carcinoma (tumor cells)": "tumor",
       "Muscle tissue (smooth or skeletal muscle)": "muscle", "Necrosis or hemorrhage": "necrosis",
       "Normal breast glands and lobules (TDLU)": "tdlu",
       "Tumor-infiltrating lymphocytes (immune infiltrates)": "tils"}
COLORS = {'tumor': '#DC2626', 'adipose': '#D97706', 'vessels': '#7C3AED', 'necrosis': '#78716C',
          'dcis': '#EA580C', 'tils': '#9333EA', 'stroma': '#0369A1', 'tdlu': '#EC4899', 'muscle': '#65A30D'}


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def tissue_fraction(img):
    """Share of pixels that are not background (white/near-white or very dark)."""
    a = np.asarray(img.convert("RGB"), float)
    sat = a.max(2) - a.min(2)
    return float(((sat > 20) & (a.mean(2) < 225)).mean())


def main():
    for pid in CANDIDATES:
        svs = glob.glob(os.path.join(SLIDES, f"{pid}-01Z-*DX1*.svs"))
        if len(svs) != 1:
            print(f"{pid}: {len(svs)} slides found, skipping"); continue
        svs = svs[0]
        stem = os.path.basename(svs)[:-4]
        out = os.path.join(OUT, pid); os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
        s = openslide.OpenSlide(svs)
        W, H = s.dimensions
        thumb = s.get_thumbnail((LONG_SIDE, LONG_SIDE)).convert("RGB")
        tw, th = thumb.size
        sx, sy = tw / W, th / H
        thumb.save(os.path.join(out, "thumbnail.png"))

        coords, counts = {}, {}
        for folder, key in KEY.items():
            f = os.path.join(EMB, folder, stem + ".pt")
            if os.path.exists(f):
                c = np.asarray(torch.load(f, map_location="cpu", weights_only=False)["coords"])
                coords[folder] = c; counts[folder] = int(len(c))
            else:
                counts[folder] = 0
        # tissue map: white background, one rectangle per tile
        tmap = Image.new("RGB", (tw, th), (255, 255, 255))
        dr = ImageDraw.Draw(tmap)
        for folder, c in coords.items():
            col = hex_rgb(COLORS[KEY[folder]])
            for x, y in c:
                x0, y0 = x * sx, y * sy
                dr.rectangle([x0, y0, max(x0 + 1, (x + TILE) * sx - 1), max(y0 + 1, (y + TILE) * sy - 1)], fill=col)
        tmap.save(os.path.join(out, "tissue_map.png"))
        Image.blend(thumb, tmap, 0.45).save(os.path.join(out, "tissue_overlay.png"))

        total = sum(counts.values())
        top2 = sorted(counts, key=lambda k: -counts[k])[:2]
        rng = np.random.default_rng(0)
        tiles = {}
        for folder in top2:
            c = coords[folder]; picked = []
            for j in rng.permutation(len(c))[:200]:
                x, y = (int(v) for v in c[j])
                img = s.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
                if tissue_fraction(img) >= 0.6:
                    k = len(picked) + 1
                    name = f"{KEY[folder]}_{k}.png"
                    img.save(os.path.join(out, "tiles", name))
                    picked.append({"file": f"tiles/{name}", "x": x, "y": y, "level": 0, "size": TILE})
                if len(picked) == 3:
                    break
            tiles[folder] = picked
        meta = {"patient_id": pid, "slide": os.path.basename(svs), "level0_size": [W, H],
                "thumbnail_size": [tw, th], "scale_thumb_per_level0": [sx, sy],
                "mpp": s.properties.get("openslide.mpp-x"), "objective_power": s.properties.get("openslide.objective-power"),
                "tile_size_level0": TILE,
                "tiles_per_compartment_in_embeddings": counts, "n_tiles_total": total,
                "colours": {KEY[f]: COLORS[KEY[f]] for f in KEY},
                "colour_source": "visualization/generate_paper_figures.py REGION_COLORS (questionnaire colours not on Narval)",
                "example_tiles": {KEY[f]: v for f, v in tiles.items()},
                "example_tiles_rule": "two compartments with the largest tile share (attribution unavailable); "
                                      "3 tiles each, fixed seed 0, among tiles with >= 60% tissue pixels"}
        json.dump(meta, open(os.path.join(out, "images.json"), "w"), indent=2)
        print(f"{pid}: thumb {tw}x{th}, {total} tiles, top2 {[KEY[f] for f in top2]}, "
              f"tiles {[len(v) for v in tiles.values()]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
