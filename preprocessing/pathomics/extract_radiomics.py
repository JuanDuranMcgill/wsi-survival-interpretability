#!/usr/bin/env python3
# Copied unchanged on 2026-11-21 from Trillium /scratch/sorkwos/multimodality/graph_subset_generation/extract_radiomics.py (last modified 2026-04-04 18:25); only this line was added.
# extract_radiomics.py
# Per-slide radiomic feature extraction from WSI tiles.
# Usage: python extract_radiomics.py <WSI_PATH> <JSONL_PATH>

import os
import sys
import json
import time
import random
import warnings
import logging
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from openslide import OpenSlide
from skimage.color import rgb2hed
from radiomics import featureextractor

warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

# -------------------------------------------------------
# Config
# -------------------------------------------------------
TILE_SIZE            = 256
CONFIDENCE_THRESHOLD = 0.7
MIN_TILES_PER_REGION = 5
MAX_TILES_PER_REGION = 1000  # cap per region for speed; None = no cap
NUM_WORKERS          = 4
OUTPUT_DIR           = "/home/sorkwos/links/scratch/radiomics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = [
    "Invasive urothelial carcinoma (tumor)",
    "Normal urothelium (benign mucosa)",
    "Lamina propria (fibrovascular stroma)",
    "Muscularis propria (detrusor muscle)",
    "Perivesical adipose tissue (extravesical fat)",
    "Necrosis",
    "Inflammatory infiltrates (immune cells)",
    "Blood vessels (vasculature)",
]

REGION_SHORT = {
    "Invasive urothelial carcinoma (tumor)": "tumor",
    "Normal urothelium (benign mucosa)": "urothelium",
    "Lamina propria (fibrovascular stroma)": "lamina",
    "Muscularis propria (detrusor muscle)": "muscularis",
    "Perivesical adipose tissue (extravesical fat)": "adipose",
    "Necrosis": "necrosis",
    "Inflammatory infiltrates (immune cells)": "immune",
    "Blood vessels (vasculature)": "vessels",
}


# -------------------------------------------------------
# Worker pool initializer — runs ONCE per worker at startup
# -------------------------------------------------------
_worker_slide     = None
_worker_extractor = None

def worker_init(wsi_path):
    global _worker_slide, _worker_extractor

    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"

    warnings.filterwarnings("ignore")
    logging.getLogger("radiomics").setLevel(logging.ERROR)

    _worker_slide = OpenSlide(wsi_path)

    _worker_extractor = featureextractor.RadiomicsFeatureExtractor(
        force2D=True,
        force2Ddimension=0,
        binWidth=25,
        minimumROIDimensions=1,
        minimumROISize=1,
    )
    _worker_extractor.disableAllFeatures()
    _worker_extractor.enableFeatureClassByName("firstorder")
    _worker_extractor.enableFeatureClassByName("glcm")


# -------------------------------------------------------
# Stain deconvolution
# -------------------------------------------------------
def separate_he_channels(rgb_array):
    hed = rgb2hed(rgb_array)

    def rescale(ch):
        ch = ch - ch.min()
        mx = ch.max()
        if mx > 0:
            ch = ch / mx
        return (ch * 255).astype(np.uint8)

    return rescale(hed[:, :, 0]), rescale(hed[:, :, 1])


# -------------------------------------------------------
# Feature extraction from a single grayscale tile
# -------------------------------------------------------
def extract_tile_features(extractor, gray_2d):
    arr_3d  = gray_2d[np.newaxis, :, :]
    mask_3d = np.ones_like(arr_3d, dtype=np.uint8)
    im = sitk.GetImageFromArray(arr_3d.astype(np.float64))
    ma = sitk.GetImageFromArray(mask_3d)
    ma.CopyInformation(im)
    result = extractor.execute(im, ma)
    return {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}


# -------------------------------------------------------
# Worker task
# -------------------------------------------------------
def process_tile(coord):
    x, y = coord
    try:
        tile_img = _worker_slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE))
        tile_rgb = np.array(tile_img.convert("RGB"))
        h_channel, e_channel = separate_he_channels(tile_rgb)
        feats_h = extract_tile_features(_worker_extractor, h_channel)
        feats_e = extract_tile_features(_worker_extractor, e_channel)
        return (feats_h, feats_e)
    except Exception:
        return None


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_radiomics.py <WSI_PATH> <JSONL_PATH>")
        sys.exit(1)

    wsi_path   = sys.argv[1]
    jsonl_path = sys.argv[2]

    if not os.path.exists(wsi_path):
        print(f"WSI not found: {wsi_path}")
        sys.exit(1)
    if not os.path.exists(jsonl_path):
        print(f"JSONL not found: {jsonl_path}")
        sys.exit(1)

    slide_stem = os.path.splitext(os.path.basename(wsi_path))[0]
    patient_id = slide_stem[:12]

    out_path = os.path.join(OUTPUT_DIR, f"{patient_id}_radiomics.npz")
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        sys.exit(0)

    print(f"\n=== Radiomic Feature Extraction ===")
    print(f"WSI:     {wsi_path}")
    print(f"JSONL:   {jsonl_path}")
    print(f"Patient: {patient_id}")
    t0 = time.time()

    # ------ Step 1: Read JSONL ------
    print("\n[1/3] Reading JSONL and grouping tiles by region...")
    region_tiles   = {r: [] for r in REGIONS}
    total_tiles    = 0
    filtered_tiles = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_tiles += 1
            label = rec["pred_label"]
            prob  = rec["pred_prob"]
            if prob < CONFIDENCE_THRESHOLD:
                filtered_tiles += 1
                continue
            if label in region_tiles:
                region_tiles[label].append((rec["coords"]["x"], rec["coords"]["y"]))

    print(f"  Total tiles: {total_tiles}")
    print(f"  Filtered (below {CONFIDENCE_THRESHOLD}): {filtered_tiles}")
    for r in REGIONS:
        print(f"  {REGION_SHORT[r]}: {len(region_tiles[r])} tiles")

    # ------ Step 2: Spin up worker pool ------
    print(f"\n[2/3] Starting {NUM_WORKERS} workers...")

    all_feature_names  = None
    region_means       = {}
    region_stds        = {}
    region_tile_counts = {}

    with Pool(processes=NUM_WORKERS, initializer=worker_init, initargs=(wsi_path,)) as pool:

        print("[3/3] Extracting features per region...\n")

        for region in REGIONS:
            short  = REGION_SHORT[region]
            coords = region_tiles[region]
            region_tile_counts[short] = len(coords)

            if len(coords) < MIN_TILES_PER_REGION:
                print(f"  [{short}] Only {len(coords)} tiles (< {MIN_TILES_PER_REGION}), marking as missing.")
                region_means[short] = None
                region_stds[short]  = None
                continue

            if MAX_TILES_PER_REGION is not None and len(coords) > MAX_TILES_PER_REGION:
                coords = random.sample(coords, MAX_TILES_PER_REGION)

            print(f"  [{short}] Processing {len(coords)} tiles...", end=" ", flush=True)
            t_region = time.time()

            results = pool.map(process_tile, coords)

            tile_features_h = []
            tile_features_e = []
            for res in results:
                if res is not None:
                    feats_h, feats_e = res
                    tile_features_h.append(feats_h)
                    tile_features_e.append(feats_e)

            if len(tile_features_h) == 0:
                print("no valid tiles after extraction.")
                region_means[short] = None
                region_stds[short]  = None
                continue

            if all_feature_names is None:
                base_names = sorted(tile_features_h[0].keys())
                all_feature_names = (
                    [f"H_{name}" for name in base_names]
                    + [f"E_{name}" for name in base_names]
                )

            h_matrix = np.array([[f[k] for k in sorted(f.keys())] for f in tile_features_h])
            e_matrix = np.array([[f[k] for k in sorted(f.keys())] for f in tile_features_e])
            combined = np.concatenate([h_matrix, e_matrix], axis=1)

            region_means[short] = np.mean(combined, axis=0)
            region_stds[short]  = np.std(combined, axis=0)

            dt = time.time() - t_region
            print(f"done ({len(tile_features_h)} valid, {dt:.1f}s)")

    # ------ Build final feature vector ------
    if all_feature_names is None:
        print("\nNo regions had enough tiles. Nothing to save.")
        sys.exit(1)

    n_features = len(all_feature_names)

    full_feature_names = []
    for short in REGION_SHORT.values():
        for fname in all_feature_names:
            full_feature_names.append(f"{short}_{fname}_mean")
            full_feature_names.append(f"{short}_{fname}_std")

    feature_vector  = []
    missing_regions = []

    for short in REGION_SHORT.values():
        if region_means[short] is None:
            feature_vector.extend([np.nan] * (n_features * 2))
            missing_regions.append(short)
        else:
            for m, s in zip(region_means[short], region_stds[short]):
                feature_vector.append(m)
                feature_vector.append(s)

    feature_vector = np.array(feature_vector)

    np.savez(
        out_path,
        patient_id=patient_id,
        feature_vector=feature_vector,
        feature_names=np.array(full_feature_names),
        region_tile_counts=json.dumps(region_tile_counts),
        missing_regions=np.array(missing_regions),
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    elapsed = time.time() - t0
    print(f"\nSaved: {out_path}")
    print(f"Feature vector length: {len(feature_vector)}")
    print(f"Missing regions: {missing_regions if missing_regions else 'none'}")
    print(f"Elapsed: {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()