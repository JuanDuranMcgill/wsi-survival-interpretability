#!/usr/bin/env python3
# extract_uni2_embeddings.py
# OFFLINE. Threaded prefetch -> GPU batched embedding extraction.

import os, sys, time, glob, threading, shutil
import numpy as np
import cv2
import torch
from openslide import OpenSlide
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# -------------------------------
# Force Hugging Face offline + cache path
# -------------------------------
os.environ["HF_HOME"] = "/home/sorkwos/.cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------------------
# Parse input WSI path
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python extract_uni2_embeddings.py <WSI_PATH>")
    sys.exit(1)

WSI_PATH = sys.argv[1]
if not os.path.exists(WSI_PATH):
    print(f"❌ WSI file not found: {WSI_PATH}")
    sys.exit(1)

# -------------------------------
# Output setup
# -------------------------------
OUTPUT_DIR = "/home/sorkwos/links/scratch/UNI2_embeddings_BRCA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE = 256
TISSUE_THRESHOLD = 0.8
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "128"))
MAX_WORKERS  = int(os.environ.get("MAX_WORKERS", str(min(32, os.cpu_count() or 16))))
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", str(8 * MAX_WORKERS)))
LOG_EVERY    = int(os.environ.get("LOG_EVERY", "2000"))

MODEL_ID = "MahmoodLab/UNI2-h"
HF_HOME = os.environ["HF_HOME"]

avail_threads = os.cpu_count() or 1
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","CV2_NUM_THREADS"]:
    os.environ[var] = str(avail_threads)
cv2.setNumThreads(avail_threads)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print("\n=== UNI2-h Embedding Extractor (Offline, FAST) ===")
print(f"HF_HOME={HF_HOME} | HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}")
print(f"CPU detected: {avail_threads} | MAX_WORKERS: {MAX_WORKERS} | MAX_INFLIGHT: {MAX_INFLIGHT}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"WSI (source): {WSI_PATH}")

# -------------------------------
# Verify cache
# -------------------------------
def check_hf_cache(model_id: str, hf_home: str) -> bool:
    owner, repo = model_id.split("/", 1)
    pattern = os.path.join(hf_home, "hub", f"models--{owner}--{repo}")
    matches = glob.glob(pattern)
    if not matches:
        print(f"❌ Cached model not found: {pattern}")
        return False
    print(f"✅ HF cache OK for {model_id} → {matches[0]}")
    return True

if not check_hf_cache(MODEL_ID, HF_HOME):
    print("Warm the cache once on a login node, then rerun offline.")
    sys.exit(1)

# -------------------------------
# Output path
# -------------------------------
out_basename = os.path.splitext(os.path.basename(WSI_PATH))[0] + "_uni2.pt"
OUTPUT_PT = os.path.join(OUTPUT_DIR, out_basename)
print(f"\nOutput: {OUTPUT_PT}\n")

if os.path.exists(OUTPUT_PT):
    print(f"✅ Output already exists, skipping: {OUTPUT_PT}")
    sys.exit(0)

# -------------------------------
# Local copy for fast I/O
# -------------------------------
def free_bytes(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except Exception:
        return -1

src_size = os.path.getsize(WSI_PATH)
ram_root = "/dev/shm"
choices = []
if os.path.isdir(ram_root) and free_bytes(ram_root) > src_size + (1 << 30):
    choices.append(ram_root)
slurm_tmp = os.environ.get("SLURM_TMPDIR")
if slurm_tmp:
    choices.append(slurm_tmp)
choices.append("/tmp")

dest_root = None
for c in choices:
    try:
        os.makedirs(c, exist_ok=True)
        if free_bytes(c) > src_size:
            dest_root = c
            break
    except Exception:
        continue

if dest_root is None:
    print("❌ No local scratch has enough space.")
    sys.exit(1)

local_wsi = os.path.join(dest_root, os.path.basename(WSI_PATH))
copied_here = False

try:
    if not (os.path.exists(local_wsi) and os.path.getsize(local_wsi) == src_size):
        print(f"[0/6] Copying WSI to {dest_root} for fast I/O...")
        t0 = time.time()
        shutil.copy2(WSI_PATH, local_wsi)
        print(f"   → Copied in {time.time()-t0:.1f}s\n")
        copied_here = True
    else:
        print(f"[0/6] Local copy already exists.\n")

    # -------------------------------
    # Load UNI2-h model (true offline mode)
    # -------------------------------
    print("[1/6] Loading UNI2-h model (fully offline)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_ckpt = "/home/sorkwos/.cache/huggingface/hub/models--MahmoodLab--UNI2-h/snapshots/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin"

    timm_kwargs = {
        'model_name': 'vit_huge_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    model = timm.create_model(
        timm_kwargs['model_name'],
        pretrained=False,
        **{k: v for k, v in timm_kwargs.items() if k != 'model_name'}
    ).to(device)

    state_dict = torch.load(local_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    print("✅ UNI2-h model loaded completely offline.\n")

    # -------------------------------
    # Tissue mask
    # -------------------------------
    print("[2/6] Opening WSI...")
    slide = OpenSlide(local_wsi)
    W, H = slide.level_dimensions[0]
    print(f"Slide size: {W} x {H}")

    print("[3/6] Generating tissue mask (Otsu)...")
    thumb_scale = 32
    thumb = slide.get_thumbnail((W // thumb_scale, H // thumb_scale)).convert("RGB")
    thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    _, mask_small = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_small = cv2.bitwise_not(mask_small)
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    slide.close()

    # -------------------------------
    # Tile coordinates
    # -------------------------------
    print("[4/6] Scanning grid for tissue tiles...")
    coords = [(x, y) for y in range(0, H, TILE_SIZE)
                      for x in range(0, W, TILE_SIZE)
                      if np.mean(mask[y:y+TILE_SIZE, x:x+TILE_SIZE] > 0) > (1 - TISSUE_THRESHOLD)]
    num_tiles = len(coords)
    print(f"  → {num_tiles} tiles will be processed.\n")

    # -------------------------------
    # Thread-local slide reading
    # -------------------------------
    _tls = threading.local()
    def _get_slide():
        s = getattr(_tls, "slide", None)
        if s is None:
            _tls.slide = OpenSlide(local_wsi)
        return _tls.slide

    def load_and_preprocess(coord):
        x, y = coord
        try:
            sl = _get_slide()
            img = sl.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
            tensor = transform(img)
            return (x, y, tensor)
        except Exception:
            return (x, y, None)

    # -------------------------------
    # Inference loop
    # -------------------------------
    print("[5/6] Starting threaded prefetch + GPU embedding extraction...")
    t0 = time.time()
    processed = 0
    coords_out, feats_out = [], []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as tp, torch.inference_mode():
        futures, it = [], iter(coords)
        batch_imgs, batch_xy = [], []

        while len(futures) < min(MAX_INFLIGHT, num_tiles):
            try:
                coord = next(it); futures.append(tp.submit(load_and_preprocess, coord))
            except StopIteration:
                break

        while futures:
            for fut in as_completed(futures):
                futures.remove(fut)
                x, y, tensor = fut.result()
                if tensor is not None:
                    batch_imgs.append(tensor)
                    batch_xy.append((x, y))
                    processed += 1

                    if len(batch_imgs) == BATCH_SIZE:
                        imgs = torch.stack(batch_imgs).to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            emb = model(imgs).detach().cpu()
                        feats_out.append(emb)
                        coords_out.extend(batch_xy)
                        batch_imgs.clear(); batch_xy.clear()

                    if processed % LOG_EVERY == 0:
                        elapsed = time.time() - t0
                        print(f"[gpu] processed {processed}/{num_tiles} ({processed/elapsed:.1f} tiles/s)")

                try:
                    while len(futures) < MAX_INFLIGHT:
                        coord = next(it); futures.append(tp.submit(load_and_preprocess, coord))
                except StopIteration:
                    pass
                break

        if batch_imgs:
            imgs = torch.stack(batch_imgs).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                emb = model(imgs).detach().cpu()
            feats_out.append(emb)
            coords_out.extend(batch_xy)

    # -------------------------------
    # Save embeddings
    # -------------------------------
    print("[6/6] Saving embeddings...")
    all_feats = torch.cat(feats_out, dim=0)
    coords_np = np.array(coords_out, dtype=np.int32)
    torch.save({"coords": coords_np, "features": all_feats}, OUTPUT_PT)

    print(f"✅ Done! Saved {len(coords_np)} embeddings → {OUTPUT_PT}")
    print(f"⏱  Total time: {(time.time()-t0)/60:.2f} min | {len(coords_np)/(time.time()-t0):.1f} tiles/s\n")

finally:
    try:
        if copied_here and os.path.exists(local_wsi):
            os.remove(local_wsi)
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass



#python embed_all_tiles_uni2.py /home/sorkwos/scratch/TCGA-BLCA-p2/007b7ccb-0aea-446c-ad8a-a373da529f25/TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs