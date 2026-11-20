#!/usr/bin/env python
"""
Highest-probability example tiles for one case-study patient (TCGA-B6-A0I9).

Uses the saved CONCH zero-shot output for the slide
(brca_jsons/{slide}_tiles.jsonl from preprocessing/classify_all_tiles_multi_6.py), which
records every class's softmax probability per 256 px tile. These are the labels behind the
case study's class-wise embeddings and tissue shares; nothing is recomputed.

For each label, tiles are ranked by that label's probability over all of the slide's
tiles. The saved probabilities were computed in half precision and saturate at exactly
1.0 for the most confident tiles (160 muscle and 729 adipose tiles on this slide), so
ties are broken by the smallest summed probability of the other eight classes, which is
1 - p before rounding. Tiles are read at level 0 and kept if at least 60% of pixels are tissue
(case_study_images.tissue_fraction), until 12 are found. Writes, in
paper_figures/case_study/{patient}/tiles_top/:
- {key}_NN.png, 256 px tiles in rank order;
- {key}_contact_sheet.png, a 4 x 3 grid captioned with probability, 1 - p (the other
  classes' summed probability) and level-0 x,y;
- tiles_top.json, the probability, top-1 label and coordinates of every tile.

    python visualization/case_study_top_tiles.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import openslide
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_study_images import tissue_fraction  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENT = "TCGA-B6-A0I9"
SLIDE = "TCGA-B6-A0I9-01Z-00-DX1.78E1D43A-507E-482E-B988-AF13B12B4497"
JSONL = f"/scratch/sorkwos/brca_jsons/{SLIDE}_tiles.jsonl"
SVS_GLOB = f"/scratch/sorkwos/TCGA-BRCA*/**/{SLIDE}.svs"
LABELS = {"muscle": "Muscle tissue (smooth or skeletal muscle)", "adipose": "Adipose tissue (fat)"}
N, TILE, MIN_TISSUE, CAPTION = 12, 256, 0.6, 30


def main():
    out = os.path.join(ROOT, "paper_figures", "case_study", PATIENT, "tiles_top")
    os.makedirs(out, exist_ok=True)
    recs = [json.loads(line) for line in open(JSONL)]
    svs = sorted(set(glob.glob(SVS_GLOB, recursive=True)))[0]
    s = openslide.OpenSlide(svs)
    meta = {"patient_id": PATIENT, "slide": SLIDE + ".svs",
            "source_probabilities": os.path.basename(JSONL) + " (saved CONCH zero-shot output, "
                                    "preprocessing/classify_all_tiles_multi_6.py; not recomputed)",
            "n_tiles_on_slide": len(recs), "tile_size_level0": TILE, "level": 0,
            "mpp": s.properties.get("openslide.mpp-x"),
            "rule": f"ranked by the label's softmax probability over all tiles, ties at the saturated "
                    f"1.0 broken by the smallest summed probability of the other classes; kept the "
                    f"first {N} with >= {MIN_TISSUE:.0%} tissue pixels", "labels": {}}
    for key, label in LABELS.items():
        others = lambda r: sum(v for k, v in r["all_scores"].items() if k != label)  # noqa: E731
        ranked = sorted(recs, key=lambda r: (-r["all_scores"][label], others(r)))
        kept, skipped = [], 0
        for rank, r in enumerate(ranked, 1):
            x, y = r["coords"]["x"], r["coords"]["y"]
            img = s.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
            tf = tissue_fraction(img)
            if tf < MIN_TISSUE:
                skipped += 1
                continue
            name = f"{key}_{len(kept) + 1:02d}.png"
            img.save(os.path.join(out, name))
            kept.append({"file": name, "x": x, "y": y, "prob": r["all_scores"][label],
                         "other_classes_prob_sum": others(r),
                         "top1_label": r["pred_label"], "top1_prob": r["pred_prob"],
                         "rank_by_prob_on_slide": rank, "tissue_fraction": round(tf, 3), "img": img})
            if len(kept) == N:
                break
        sheet = Image.new("RGB", (4 * TILE, 3 * (TILE + CAPTION)), "white")
        dr = ImageDraw.Draw(sheet)
        for i, t in enumerate(kept):
            cx, cy = (i % 4) * TILE, (i // 4) * (TILE + CAPTION)
            sheet.paste(t["img"], (cx, cy + CAPTION))
            dr.text((cx + 4, cy + 2), f"{t['file'][:-4]}  p={t['prob']:.4f}  1-p={t['other_classes_prob_sum']:.1e}",
                    fill="black")
            dr.text((cx + 4, cy + 15), f"x={t['x']}, y={t['y']}", fill="black")
        sheet.save(os.path.join(out, f"{key}_contact_sheet.png"))
        for t in kept:
            del t["img"]
        meta["labels"][key] = {"label": label, "n_skipped_below_tissue_threshold": skipped,
                               "n_tiles_with_prob_exactly_1": sum(r["all_scores"][label] == 1.0 for r in recs),
                               "n_tiles_with_label_as_top1": sum(r["pred_label"] == label for r in recs),
                               "contact_sheet": f"{key}_contact_sheet.png", "tiles": kept}
        print(f"{key}: probs {kept[0]['prob']:.3f}..{kept[-1]['prob']:.3f}, skipped {skipped}, "
              f"top-1 is this label for {sum(t['top1_label'] == label for t in kept)}/{len(kept)}")
    json.dump(meta, open(os.path.join(out, "tiles_top.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
