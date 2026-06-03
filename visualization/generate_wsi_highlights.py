#!/usr/bin/env python3
"""
Generate WSI thumbnails with the top-2 tissue regions (by learned fusion weight)
highlighted in distinct colours, for BLCA and BRCA.

Outputs:
  output_highlights/
    BLCA_<patient>_highlight.png   -- 3 BLCA examples
    BRCA_<patient>_highlight.png   -- 3 BRCA examples
    combined_panel.png             -- 2-row × 3-col paper-ready grid

Usage:
    cd /home/sorkwos/links/scratch
    source ~/envs/conch_env/bin/activate
    python generate_wsi_highlights.py
"""

import os
import glob
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import openslide
except ImportError:
    raise SystemExit("openslide not found — activate conch_env first")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLCA_SVS_ROOTS = [
    "/home/sorkwos/links/scratch/TCGA-BLCA-p2",
    "/home/sorkwos/links/scratch/TCGA-BLCA/WSI/output_folder1",
]
BRCA_SVS_ROOTS = [
    "/home/sorkwos/links/scratch/TCGA-BRCA-1",
    "/home/sorkwos/links/scratch/TCGA-BRCA-2",
    "/home/sorkwos/links/scratch/TCGA-BRCA-3",
    "/home/sorkwos/links/scratch/TCGA-BRCA-4",
    "/home/sorkwos/links/scratch/TCGA-BRCA-5",
    "/home/sorkwos/links/scratch/TCGA-BRCA-new",
]
BLCA_JSON_ROOT = "/home/sorkwos/links/scratch/blca_jsons"
BRCA_JSON_ROOT = "/home/sorkwos/links/scratch/brca_jsons"

OUTPUT_DIR = "/home/sorkwos/links/scratch/output_highlights"
N_SAMPLES  = 3        # examples per cohort
THUMB_SIZE = (1024, 1024)
TILE_SIZE  = 256
RANDOM_SEED = 7

# Top-2 regions per cohort (exact CONCH label strings, ordered by weight)
COHORT_CONFIG = {
    "BLCA": {
        "regions": [
            {
                "label":  "Necrosis",
                "short":  "Necrosis",
                "weight": 0.134,
                "color":  (220, 38, 38, 170),   # red
            },
            {
                "label":  "Invasive urothelial carcinoma (tumor)",
                "short":  "Invasive UC",
                "weight": 0.133,
                "color":  (37, 99, 235, 170),    # blue
            },
        ],
    },
    "BRCA": {
        "regions": [
            {
                "label":  "Necrosis or hemorrhage",
                "short":  "Necrosis",
                "weight": 0.117,
                "color":  (220, 38, 38, 170),    # red
            },
            {
                "label":  "Ductal carcinoma in situ (DCIS)",
                "short":  "DCIS",
                "weight": 0.113,
                "color":  (37, 99, 235, 170),    # blue
            },
        ],
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_matched_pairs(svs_roots, json_root, dx_only=True):
    """Return list of (svs_path, json_path) pairs."""
    all_svs = []
    for root in svs_roots:
        all_svs.extend(glob.glob(os.path.join(root, "**", "*.svs"), recursive=True))
    if dx_only:
        all_svs = [f for f in all_svs if "DX1" in os.path.basename(f)]
    pairs = []
    for svs_path in all_svs:
        stem = os.path.splitext(os.path.basename(svs_path))[0]
        json_path = os.path.join(json_root, stem + "_tiles.jsonl")
        if os.path.exists(json_path):
            pairs.append((svs_path, json_path))
    return pairs


def load_tiles_by_label(json_path, labels):
    """Return dict {label: [(x, y), ...]} for the requested labels."""
    tiles = {lbl: [] for lbl in labels}
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            lbl = t["pred_label"]
            if lbl in tiles:
                c = t["coords"]
                tiles[lbl].append((c["x"], c["y"]))
    return tiles


def draw_legend(draw, regions, box_x, box_y, font):
    """Draw a small legend at (box_x, box_y)."""
    pad = 8
    line_h = 28
    box_w  = 260

    # Background
    total_h = len(regions) * line_h + pad * 2
    draw.rectangle(
        [box_x, box_y, box_x + box_w, box_y + total_h],
        fill=(0, 0, 0, 180)
    )
    for i, r in enumerate(regions):
        cy = box_y + pad + i * line_h + line_h // 2
        cx = box_x + pad + 10
        # coloured square
        sq = 14
        draw.rectangle(
            [cx, cy - sq // 2, cx + sq, cy + sq // 2],
            fill=r["color"][:3] + (255,)
        )
        label_str = f"{r['short']}  (w={r['weight']:.3f})"
        draw.text((cx + sq + 8, cy - 9), label_str, font=font, fill=(255, 255, 255, 255))


def process_slide(svs_path, json_path, regions, cohort, out_path):
    """Load slide, overlay top-2 regions, save."""
    stem = os.path.splitext(os.path.basename(svs_path))[0]
    patient = stem[:12]   # e.g. TCGA-2F-A9KQ

    slide = openslide.OpenSlide(svs_path)
    full_w, full_h = slide.dimensions
    thumb = slide.get_thumbnail(THUMB_SIZE)
    slide.close()

    actual_w, actual_h = thumb.size
    scale_x = actual_w / full_w
    scale_y = actual_h / full_h
    thumb_tile = max(1, int(TILE_SIZE * min(scale_x, scale_y)))

    labels = [r["label"] for r in regions]
    tiles_by_label = load_tiles_by_label(json_path, labels)

    # Check we actually have tiles for at least one region
    total_tiles = sum(len(v) for v in tiles_by_label.values())
    if total_tiles == 0:
        print(f"  [{patient}] no tiles found for target regions — skipping")
        return False

    overlay = thumb.copy().convert("RGBA")

    # Draw each region
    for r in regions:
        coords = tiles_by_label[r["label"]]
        if not coords:
            continue
        mask_arr = np.zeros((actual_h, actual_w, 4), dtype=np.uint8)
        for (tx_raw, ty_raw) in coords:
            tx  = int(tx_raw * scale_x)
            ty  = int(ty_raw * scale_y)
            tx2 = min(tx + thumb_tile, actual_w)
            ty2 = min(ty + thumb_tile, actual_h)
            mask_arr[ty:ty2, tx:tx2] = r["color"]
        mask = Image.fromarray(mask_arr, "RGBA")
        overlay = Image.alpha_composite(overlay, mask)

    # Patient label only — legend is drawn once on the combined panel
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([10, 10, 220, 38], fill=(0, 0, 0, 160))
    draw.text((16, 14), f"{cohort}  {patient}", font=font, fill=(255, 255, 255, 255))

    result = overlay.convert("RGB")
    result.save(out_path)
    print(f"  saved {out_path}  (tiles: {', '.join(str(len(tiles_by_label[r['label']])) + ' ' + r['short'] for r in regions)})")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BASE = "/home/sorkwos/links/scratch"

# Curated slide pairs — selected for balanced coverage of both top-2 regions
SELECTED = {
    "BLCA": [
        # necro=28745 tumor=35958 — most tiles, well balanced
        (f"{BASE}/TCGA-BLCA-p2/291a3c7c-0a2c-4ffe-987e-a8ae70bcea5f/TCGA-E7-A7DV-01Z-00-DX1.FCE4752B-30BC-4747-A9F4-38141E09F005.svs",
         f"{BASE}/blca_jsons/TCGA-E7-A7DV-01Z-00-DX1.FCE4752B-30BC-4747-A9F4-38141E09F005_tiles.jsonl"),
        # necro=20763 tumor=23749 — balanced
        (f"{BASE}/TCGA-BLCA-p2/97627fb1-4093-4420-8f13-661b6b0c1028/TCGA-FJ-A3ZE-01Z-00-DX1.CD478617-4D80-4F36-8908-8480738367C6.svs",
         f"{BASE}/blca_jsons/TCGA-FJ-A3ZE-01Z-00-DX1.CD478617-4D80-4F36-8908-8480738367C6_tiles.jsonl"),
    ],
    "BRCA": [
        # necro=4048 dcis=4391 — most balanced
        (f"{BASE}/TCGA-BRCA-3/d8a5a25e-f7dc-4c0d-8d08-efa050713725/TCGA-OL-A6VQ-01Z-00-DX1.E3B81163-0239-47C3-B53F-064405B58685.svs",
         f"{BASE}/brca_jsons/TCGA-OL-A6VQ-01Z-00-DX1.E3B81163-0239-47C3-B53F-064405B58685_tiles.jsonl"),
        # necro=4116 dcis=3303 — good balance
        (f"{BASE}/TCGA-BRCA-new/a220b48d-fa86-4efd-9e5d-f739a8db5ef9/TCGA-UU-A93S-01Z-00-DX1.C4809779-DF5F-4F5D-A78C-B7F95F2D050F.svs",
         f"{BASE}/brca_jsons/TCGA-UU-A93S-01Z-00-DX1.C4809779-DF5F-4F5D-A78C-B7F95F2D050F_tiles.jsonl"),
    ],
}


def main():
    saved = {"BLCA": [], "BRCA": []}

    for cohort in ["BLCA", "BRCA"]:
        cfg = COHORT_CONFIG[cohort]
        print(f"\nProcessing {cohort}...")
        for svs_path, json_path in SELECTED[cohort]:
            stem    = os.path.splitext(os.path.basename(svs_path))[0]
            patient = stem[:12]
            out_path = os.path.join(OUTPUT_DIR, f"{cohort}_{patient}_highlight.png")
            print(f"  Processing {patient}...")
            ok = process_slide(svs_path, json_path, cfg["regions"], cohort, out_path)
            if ok:
                saved[cohort].append(out_path)

    # -------------------------------------------------------------------
    # Assemble combined panel  (2 rows × 2 cols + single legend strip)
    # -------------------------------------------------------------------
    all_imgs = saved["BLCA"] + saved["BRCA"]
    if len(all_imgs) == 0:
        print("\nNo images generated.")
        return

    imgs     = [Image.open(p).convert("RGB") for p in all_imgs]
    n_cols   = 2
    n_rows   = 2
    pad      = 12
    label_h  = 36
    legend_h = 70    # shared legend strip at bottom
    cell_w   = imgs[0].width
    cell_h   = imgs[0].height

    panel_w = n_cols * cell_w + (n_cols + 1) * pad
    panel_h = n_rows * (cell_h + label_h) + (n_rows + 1) * pad + legend_h

    panel = Image.new("RGB", (panel_w, panel_h), color=(245, 245, 245))
    draw  = ImageDraw.Draw(panel)

    try:
        font_lbl = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
        font_leg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font_lbl = ImageFont.load_default()
        font_leg = font_lbl

    row_labels = (["BLCA"] * len(saved["BLCA"]) + ["BRCA"] * len(saved["BRCA"]))

    for idx, (img, row_lbl) in enumerate(zip(imgs, row_labels)):
        col = idx % n_cols
        row = idx // n_cols
        x   = pad + col * (cell_w + pad)
        y   = pad + row * (cell_h + label_h + pad)
        draw.text((x + 4, y + 4), row_lbl, font=font_lbl, fill=(50, 50, 50))
        panel.paste(img, (x, y + label_h))

    # --- single shared legend at the bottom ---
    # collect all unique region definitions across both cohorts
    all_regions = (
        COHORT_CONFIG["BLCA"]["regions"] + COHORT_CONFIG["BRCA"]["regions"]
    )
    # deduplicate by short name
    seen  = set()
    unique_regions = []
    for r in all_regions:
        if r["short"] not in seen:
            seen.add(r["short"])
            unique_regions.append(r)

    legend_y  = panel_h - legend_h + 10
    sq        = 18
    item_gap  = 30
    x_cursor  = pad + 10

    for r in unique_regions:
        # coloured square
        draw.rectangle(
            [x_cursor, legend_y, x_cursor + sq, legend_y + sq],
            fill=r["color"][:3]
        )
        lbl = f"{r['short']}  (w={r['weight']:.3f})"
        draw.text((x_cursor + sq + 8, legend_y - 1), lbl,
                  font=font_leg, fill=(30, 30, 30))
        # measure text width for next item placement
        bbox = draw.textbbox((0, 0), lbl, font=font_leg)
        x_cursor += sq + 8 + (bbox[2] - bbox[0]) + item_gap

    panel_path = os.path.join(OUTPUT_DIR, "combined_panel.png")
    panel.save(panel_path, dpi=(300, 300))
    print(f"\nCombined panel saved: {panel_path}")
    print(f"Total images: BLCA={len(saved['BLCA'])}, BRCA={len(saved['BRCA'])}")


if __name__ == "__main__":
    main()
