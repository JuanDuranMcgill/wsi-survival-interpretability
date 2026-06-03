#!/usr/bin/env python3
# classify_all_tiles_multi_6.py
# OFFLINE. Threaded prefetch -> GPU batched inference.

import os, sys, json, time, glob, threading, shutil
import numpy as np
import cv2
import torch
from openslide import OpenSlide
from PIL import Image

# -------------------------------
# Parse input WSI path
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python classify_all_tiles_multi_6.py <WSI_PATH>")
    sys.exit(1)
WSI_PATH = sys.argv[1]
if not os.path.exists(WSI_PATH):
    print(f"❌ WSI file not found: {WSI_PATH}")
    sys.exit(1)

# Where to save JSONL outputs
OUTPUT_DIR = "/home/sorkwos/links/scratch/brca_jsons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# User config
# -------------------------------
TILE_SIZE = 256
TISSUE_THRESHOLD = 0.8
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",   "160"))
MAX_WORKERS  = int(os.environ.get("MAX_WORKERS",  str(min(32, os.cpu_count() or 16))))
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", str(8 * MAX_WORKERS)))
FLUSH_EVERY  = int(os.environ.get("FLUSH_EVERY",  "5000"))
LOG_EVERY    = int(os.environ.get("LOG_EVERY",    "2000"))

MODEL_CFG = "conch_ViT-B-16"
MODEL_ID  = "MahmoodLab/conch"

# -------------------------------
# HF cache (offline)
# -------------------------------
HF_HOME = os.environ.get("HF_HOME", "/home/sorkwos/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_OFFLINE"] = "1"

# -------------------------------
# BLAS threading hygiene
# -------------------------------
avail_threads = os.cpu_count() or 1
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","CV2_NUM_THREADS"]:
    os.environ[var] = str(avail_threads)
cv2.setNumThreads(avail_threads)

# -------------------------------
# CUDA speed knobs
# -------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print("\n=== Offline Full-Slide Classifier (CONCH, FAST) ===")
print(f"HF_HOME={HF_HOME} | HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}")
print(f"CPU detected: {avail_threads} | MAX_WORKERS: {MAX_WORKERS} | MAX_INFLIGHT: {MAX_INFLIGHT}")
print(f"BATCH_SIZE: {BATCH_SIZE} | FLUSH_EVERY: {FLUSH_EVERY} | LOG_EVERY: {LOG_EVERY}")
print(f"WSI (source): {WSI_PATH}")

# -------------------------------
# Verify HF cache presence
# -------------------------------
def check_hf_cache(model_id: str, hf_home: str) -> bool:
    hub_root = os.path.join(hf_home, "hub")
    owner, repo = model_id.split("/", 1)
    pattern = os.path.join(hub_root, f"models--{owner}--{repo}")
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
# Compute output path (from SOURCE name) and SKIP BEFORE ANY COPY
# -------------------------------
out_basename = os.path.splitext(os.path.basename(WSI_PATH))[0] + "_tiles.jsonl"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, out_basename)
print(f"\nOutput: {OUTPUT_JSONL}\n")

if os.path.exists(OUTPUT_JSONL) and os.path.getsize(OUTPUT_JSONL) > 0:
    print(f"✅ Output already exists for {WSI_PATH}, skipping before any copy.")
    sys.exit(0)

# -------------------------------
# Choose fastest local target: /dev/shm (RAM-disk) > SLURM_TMPDIR > /tmp
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
for candidate in choices:
    try:
        os.makedirs(candidate, exist_ok=True)
        if free_bytes(candidate) > src_size:
            dest_root = candidate
            break
    except Exception:
        continue

if dest_root is None:
    print("❌ No local scratch has enough space. Exiting.")
    sys.exit(1)

local_wsi = os.path.join(dest_root, os.path.basename(WSI_PATH))
copied_here = False

try:
    if not (os.path.exists(local_wsi) and os.path.getsize(local_wsi) == src_size):
        label = "RAM-disk" if dest_root == "/dev/shm" else "local SSD"
        print(f"[0/7] Copying SVS to {label} ({dest_root}) for fast I/O...")
        tcopy = time.time()
        shutil.copy2(WSI_PATH, local_wsi)
        dt = time.time() - tcopy
        sz_gb = src_size / (1024**3)
        speed = sz_gb / dt if dt > 0 else float("inf")
        print(f"   → Copied {sz_gb:.2f} GB in {dt:.1f}s ({speed:.2f} GB/s)\n")
        copied_here = True
    else:
        where = "RAM-disk" if dest_root == "/dev/shm" else "local SSD"
        print(f"[0/7] {where} already has the SVS (using: {local_wsi})\n")

    # -------------------------------
    # Load CONCH model (offline)
    # -------------------------------
    from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch CUDA: {torch.cuda.is_available()} | device: {device}\n")

    print("[1/7] Loading CONCH model (offline cache)...")
    model, preprocess = create_model_from_pretrained(MODEL_CFG, "hf_hub:" + MODEL_ID, device=device)
    model.eval()
    tokenizer = get_tokenizer()
    print("✅ Model loaded.\n")

    # -------------------------------
    # Prompts and tokenization
    # -------------------------------
    CLASSES = [
        "Invasive breast carcinoma (tumor cells)",
        "Ductal carcinoma in situ (DCIS)",
        "Normal breast glands and lobules (TDLU)",
        "Fibrous desmoplastic stroma",
        "Adipose tissue (fat)",
        "Tumor-infiltrating lymphocytes (immune infiltrates)",
        "Necrosis or hemorrhage",
        "Blood vessels (vasculature)",
        "Muscle tissue (smooth or skeletal muscle)",
    ]
    PROMPTS = [f"an H&E image of {t}" for t in CLASSES]
    tokenized_prompts = tokenize(texts=PROMPTS, tokenizer=tokenizer).to(device)

    # -------------------------------
    # Open WSI + tissue mask
    # -------------------------------
    print("[2/7] Opening WSI...")
    slide = OpenSlide(local_wsi)
    W, H = slide.level_dimensions[0]
    print(f"Slide size: {W} x {H}")

    print("[3/7] Generating tissue mask (thumbnail + Otsu)...")
    thumb_scale = 32
    thumb = slide.get_thumbnail((W // thumb_scale, H // thumb_scale)).convert("RGB")
    thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    _, mask_small = cv2.threshold(thumb_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_small = cv2.bitwise_not(mask_small)  # tissue=255
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    tissue_fraction = np.mean(mask > 0)
    print(f"Tissue area fraction: {tissue_fraction:.3f}\n")

    # --- DEBUG ---
    total_tiles = sum(1 for y in range(0, H, TILE_SIZE) for x in range(0, W, TILE_SIZE))
    coords_test = []
    for y in range(0, H, TILE_SIZE):
        for x in range(0, W, TILE_SIZE):
            pm = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if pm.size == 0:
                continue
            coords_test.append(np.mean(pm > 0))
    print(f"Total grid tiles: {total_tiles}")
    print(f"Mean tissue fraction per tile: {np.mean(coords_test):.3f}")
    print(f"Min tissue fraction: {np.min(coords_test):.3f}")
    print(f"Max tissue fraction: {np.max(coords_test):.3f}")
    slide.close()  # threads open their own handle
    # -------------------------------
    # Precompute coords that pass filter
    # -------------------------------
    print("[4/7] Scanning grid for tissue tiles...")
    coords = []
    for y in range(0, H, TILE_SIZE):
        for x in range(0, W, TILE_SIZE):
            pm = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if pm.size == 0:
                continue
            if np.mean(pm > 0) > (1 - TISSUE_THRESHOLD):
                coords.append((x, y))
    num_tiles = len(coords)
    print(f"  → {num_tiles} tiles will be classified.\n")

    # -------------------------------
    # Thread-local OpenSlide
    # -------------------------------
    _tls = threading.local()
    def _get_slide() -> OpenSlide:
        s = getattr(_tls, "slide", None)
        if s is None:
            _tls.slide = OpenSlide(local_wsi)
        return _tls.slide

    def load_and_preprocess(coord):
        x, y = coord
        try:
            sl = _get_slide()
            img = sl.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
            tensor = preprocess(img)  # CHW float32 on CPU
            return (x, y, tensor)
        except Exception:
            return (x, y, None)

    # -------------------------------
    # Inference loop (threaded prefetch)
    # -------------------------------
    print("[5/7] Starting threaded prefetch + GPU loop...")
    t0 = time.time()
    processed = 0
    written = 0
    write_buf = []

    text_emb = model.encode_text(tokenized_prompts)

    def log_progress(n):
        elapsed = max(time.time() - t0, 1e-6)
        rate = n / elapsed
        eta = (num_tiles - n) / max(rate, 1e-6)
        print(f"[gpu] processed {n}/{num_tiles} ({rate:.1f} tiles/s, ETA {eta/60:.1f} min)")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as tp, open(OUTPUT_JSONL, "w") as f_out, torch.inference_mode():
        batch_imgs, batch_xy = [], []
        futures = []
        it = iter(coords)

        while len(futures) < min(MAX_INFLIGHT, num_tiles):
            try:
                coord = next(it); futures.append(tp.submit(load_and_preprocess, coord))
            except StopIteration:
                break

        while futures:
            for fut in as_completed(futures, timeout=None):
                futures.remove(fut)
                x, y, tensor = fut.result()
                if tensor is not None:
                    batch_imgs.append(tensor); batch_xy.append((x, y)); processed += 1

                    if len(batch_imgs) == BATCH_SIZE:
                        images_cpu = (
                            torch.stack(batch_imgs, dim=0)
                            .contiguous(memory_format=torch.channels_last)
                            .pin_memory()
                        )
                        images = images_cpu.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            img_emb = model.encode_image(images)
                            logits  = (img_emb @ text_emb.T) * model.logit_scale.exp()
                        probs = logits.softmax(dim=-1).cpu().numpy()

                        for (cx, cy), p in zip(batch_xy, probs):
                            top = int(np.argmax(p))
                            rec = {
                                "coords": {"x": int(cx), "y": int(cy), "size": TILE_SIZE},
                                "pred_label": CLASSES[top],
                                "pred_prob": float(p[top]),
                                "all_scores": {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}
                            }
                            write_buf.append(json.dumps(rec))

                        written += len(batch_xy)
                        if len(write_buf) >= FLUSH_EVERY:
                            f_out.write("\n".join(write_buf) + "\n")
                            write_buf.clear()

                        batch_imgs.clear(); batch_xy.clear()

                    if processed % LOG_EVERY == 0:
                        log_progress(processed)

                try:
                    while len(futures) < MAX_INFLIGHT:
                        coord = next(it); futures.append(tp.submit(load_and_preprocess, coord))
                except StopIteration:
                    pass
                break

        if batch_imgs:
            images_cpu = (
                torch.stack(batch_imgs, dim=0)
                .contiguous(memory_format=torch.channels_last)
                .pin_memory()
            )
            images = images_cpu.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                img_emb = model.encode_image(images)
                logits  = (img_emb @ text_emb.T) * model.logit_scale.exp()
            probs = logits.softmax(dim=-1).cpu().numpy()
            for (cx, cy), p in zip(batch_xy, probs):
                top = int(np.argmax(p))
                rec = {
                    "coords": {"x": int(cx), "y": int(cy), "size": TILE_SIZE},
                    "pred_label": CLASSES[top],
                    "pred_prob": float(p[top]),
                    "all_scores": {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}
                }
                write_buf.append(json.dumps(rec))
            written += len(batch_xy)

        if write_buf:
            with open(OUTPUT_JSONL, "a") as f_out2:
                f_out2.write("\n".join(write_buf) + "\n")

    elapsed = time.time() - t0
    rate = written / max(elapsed, 1e-6)
    print(f"\n[6/7] ✅ Done! Saved {written} classified tiles → {OUTPUT_JSONL}")
    print(f"[7/7] ⏱  Elapsed: {elapsed/60:.2f} min | Throughput: {rate:.1f} tiles/s\n")

finally:
    # Guaranteed cleanup of temp copy to avoid /dev/shm growth
    try:
        if copied_here and os.path.exists(local_wsi):
            os.remove(local_wsi)
            # print(f"(Cleaned: {local_wsi})")
    except Exception:
        pass
    # Lightweight memory cleanup (kept minimal—main fix is the early skip + file removal)
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
