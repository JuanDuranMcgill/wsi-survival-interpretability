#!/usr/bin/env python3
"""Pan-carcinoma preprocessing runner for the MEDomics-Beast (two services around one SQLite ledger).

  python pancan_runner.py download   GDC downloads (STREAMS threads, resume on drop, MD5 check, at most CAP
                                     unprocessed slides waiting), moves processed raw slides to long-term storage,
                                     rewrites docs/PANCAN_RUN_STATUS.md and commits it locally once a day.
  python pancan_runner.py process    GPU worker: tile once, CONCH + UNI-2 in one pass, float16 outputs.
                                     Pauses while another user's process is on the GPU.

Pipeline = the one tested in docs/PROCESSING_TEST_RESULTS.md (256 px tiles at level 0, Otsu mask on a 1/32 thumbnail,
0.8 threshold, one tile read for both models, DataLoader workers, autocast fp16, float16 outputs).
Crash safety: outputs are written to <name>.tmp and renamed when complete; every slide has a state in the ledger
(pending, downloading, downloaded, processing, done, failed_download, failed_process); on restart finished slides are
skipped and in-flight ones go back one step. Slide order: manifest_order.csv (make_order.py). Never pushes.
"""
import os, sys, csv, json, time, sqlite3, hashlib, shutil, threading, datetime, getpass, traceback

ROOT = "/data/processed/juan/pancan"
LEDGER, ORDER, FEAT, LOGS = f"{ROOT}/ledger.sqlite", f"{ROOT}/manifest_order.csv", f"{ROOT}/features", f"{ROOT}/logs"
INBOX = "/data/scratch/juan/pancan_slides/_inbox"            # fresh downloads (fast disk, read by the tile loaders)
SCR_KEEP = "/data/scratch/juan/pancan_slides"                # long-term home once /data/storage is nearly full
STO = "/data/storage/juan/pancan_slides"                     # long-term home of the raw slides (slow disk: never read in a run)
REPO = "/home/users/juan/wsi-survival-interpretability"
STATUS_DOC = f"{REPO}/docs/PANCAN_RUN_STATUS.md"
GDC = "https://api.gdc.cancer.gov/data/"
CAP, STREAMS, WORKERS, BATCH, TILE, THR = 150, 24, 16, 256, 256, 0.8
STORAGE_MIN_FREE, SCRATCH_MIN_FREE, PROCESSED_MIN_FREE = 300e9, 250e9, 60e9
MAX_DL, MAX_PROC = 5, 3
ME = getpass.getuser()
LIVE = "('downloading','downloaded','processing')"


def now(): return time.time()
def log(*a): print(datetime.datetime.now().strftime("%F %T"), *a, flush=True)


# ---------------------------------------------------------------- ledger
_tl = threading.local()
def db():
    c = getattr(_tl, "c", None)
    if c is None:
        c = sqlite3.connect(LEDGER, timeout=60, isolation_level=None); c.row_factory = sqlite3.Row
        c.execute("pragma busy_timeout=60000"); _tl.c = c
    return c

class txn:
    def __enter__(s): db().execute("begin immediate"); return db()
    def __exit__(s, t, v, tb): db().execute("commit" if t is None else "rollback")

SCHEMA = """create table if not exists slides(idx integer primary key, file_id text unique, file_name text, patient text, type text,
 size integer, md5 text, state text, raw_loc text, dl_attempts integer default 0, proc_attempts integer default 0, next_try real default 0,
 error text, dl_s real, dl_bytes integer, dl_drops integer default 0, t_download real, tiles integer, tiling_s real, embed_s real,
 pause_s real, tiles_per_s real, gpu_util real, save_s real, out_bytes integer, t_done real, app_mag text);
create table if not exists kv(k text primary key, v text);"""

def init_ledger():
    c = db(); c.execute("pragma journal_mode=wal"); c.executescript(SCHEMA)
    with txn() as c:
        for r in csv.DictReader(open(ORDER)):
            c.execute("insert or ignore into slides(idx,file_id,file_name,patient,type,size,md5,state) values (?,?,?,?,?,?,?,'pending')",
                      (int(r["order"]), r["file_id"], r["file_name"], r["patient"], r["type"], int(r["file_size"]), r["md5sum"]))

def kv_set(k, v):
    db().execute("insert into kv(k,v) values(?,?) on conflict(k) do update set v=excluded.v", (k, str(v)))
def kv_get(k, d=None):
    r = db().execute("select v from kv where k=?", (k,)).fetchone(); return r[0] if r else d
def upd(idx, **kw):
    db().execute(f"update slides set {','.join(k + '=?' for k in kw)} where idx=?", (*kw.values(), idx))
def out_path(r):
    return f"{FEAT}/{r['type']}/{r['file_name'].rsplit('.', 1)[0]}_emb16.pt"


# ---------------------------------------------------------------- download service
def rehash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h

def fetch(sess, r):
    """Download r to INBOX with Range resume; returns number of drops. Raises on failure or md5 mismatch."""
    final = f"{INBOX}/{r['file_name']}"; part = final + ".part"; size = r["size"]; drops = 0; last = None
    for attempt in range(8):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got > size: os.remove(part); got = 0
        h = rehash(part) if got else hashlib.md5()
        try:
            if got < size:
                with sess.get(GDC + r["file_id"], headers={"Range": f"bytes={got}-"} if got else {}, stream=True, timeout=(30, 60)) as resp:
                    if got and resp.status_code == 200: got = 0; h = hashlib.md5(); mode = "wb"      # Range ignored: restart
                    elif resp.status_code in (200, 206): mode = "ab" if got else "wb"
                    else: resp.raise_for_status(); raise IOError(f"HTTP {resp.status_code}")
                    with open(part, mode) as f:
                        for chunk in resp.iter_content(1 << 20): f.write(chunk); h.update(chunk); got += len(chunk)
            if got != size: raise IOError(f"size {got} != {size}")
            if h.hexdigest() != r["md5"]:
                os.remove(part); raise IOError("md5 mismatch")
            return drops
        except Exception as e:
            drops += 1; last = e; time.sleep(min(30, 2 ** attempt))
    raise IOError(f"gave up after {drops} drops: {type(last).__name__}: {last}")

def claim_download():
    if shutil.disk_usage("/data/scratch").free < SCRATCH_MIN_FREE: return None
    with txn() as c:
        if c.execute(f"select count(*) from slides where state in {LIVE}").fetchone()[0] >= CAP: return None
        r = c.execute("select * from slides where state='pending' and next_try<=? order by idx limit 1", (now(),)).fetchone()
        if r is None: return None
        c.execute("update slides set state='downloading' where idx=?", (r["idx"],)); return r

def dl_worker(k):
    import requests
    sess = requests.Session()
    while True:
        r = claim_download()
        if r is None: time.sleep(10); continue
        t0 = now(); final = f"{INBOX}/{r['file_name']}"
        try:
            drops = fetch(sess, r); os.replace(final + ".part", final)
            upd(r["idx"], state="downloaded", raw_loc=final, dl_s=now() - t0, dl_bytes=r["size"], dl_drops=drops, t_download=now(), error=None)
        except Exception as e:
            a = r["dl_attempts"] + 1
            upd(r["idx"], state="failed_download" if a >= MAX_DL else "pending", dl_attempts=a, next_try=now() + 120 * a, error=f"download: {e}"[:300])
            log("download failed", r["file_name"], e)

def move_one(r):
    src = r["raw_loc"]
    if not os.path.exists(src): upd(r["idx"], raw_loc="MISSING"); return
    base = STO if shutil.disk_usage("/data/storage").free - r["size"] > STORAGE_MIN_FREE else SCR_KEEP
    d = f"{base}/{r['type']}"; os.makedirs(d, exist_ok=True); dst = f"{d}/{r['file_name']}"
    if os.stat(src).st_dev == os.stat(d).st_dev: os.replace(src, dst)
    else:
        shutil.copyfile(src, dst + ".part")
        if os.path.getsize(dst + ".part") != os.path.getsize(src): raise IOError("copy size mismatch")
        os.replace(dst + ".part", dst); os.remove(src)
    upd(r["idx"], raw_loc=dst)

def mover():
    while True:
        try:
            rows = db().execute("select * from slides where state in ('done','failed_process') and raw_loc like ? limit 20", (INBOX + "/%",)).fetchall()
            for r in rows: move_one(r)
        except Exception as e: log("mover error", repr(e)); rows = []
        time.sleep(1 if rows else 30)

def download_service():
    os.makedirs(INBOX, exist_ok=True); init_ledger()
    with txn() as c:
        c.execute("update slides set state='pending' where state='downloading'")
        for r in c.execute("select idx,raw_loc from slides where state='downloaded'").fetchall():
            if not (r["raw_loc"] and os.path.exists(r["raw_loc"])): c.execute("update slides set state='pending' where idx=?", (r["idx"],))
    log(f"download service up: {STREAMS} streams, cap {CAP}")
    for k in range(STREAMS): threading.Thread(target=dl_worker, args=(k,), daemon=True).start(); time.sleep(0.5)
    threading.Thread(target=mover, daemon=True).start()
    last_commit = float(kv_get("last_status_commit", 0))
    while True:
        time.sleep(1200)
        try:
            due = now() - last_commit > 86400
            write_status(commit=due)
            if due: last_commit = now(); kv_set("last_status_commit", last_commit)
            n = db().execute("select count(*) from slides where state in ('pending','downloading','downloaded','processing')").fetchone()[0]
            if n == 0: log("all slides finished"); return 0
        except Exception: log("status error", traceback.format_exc())


# ---------------------------------------------------------------- status document
def write_status(commit=False):
    c = db(); t = now(); tm = lambda s: f"{s/3600:.1f} h"
    st = dict(c.execute("select state, count(*) from slides group by state").fetchall()); total = sum(st.values())
    done = st.get("done", 0)
    rows = c.execute("select type, count(*) n, sum(state='done') d, sum(state like 'failed%') f, sum(size)/1e9 gb, sum(case when state='done' then size else 0 end)/1e9 dgb, "
                     "sum(out_bytes)/1e9 ob from slides group by type order by type").fetchall()
    rec = c.execute("select * from slides where state='done' and tiles>0 order by t_done desc limit 50").fetchall()
    allr = c.execute("select * from slides where state='done' and tiles>0").fetchall()
    L = [f"# Pan-carcinoma preprocessing run: status (updated {datetime.datetime.now():%Y-%m-%d %H:%M})", "",
         "Written by `preprocessing/beast_runner/pancan_runner.py`. Cohort: `in_cohort == True`, 17 carcinoma types, "
         f"{total:,} slides. Order: `/data/processed/juan/pancan/manifest_order.csv` (stratified by type, seed 0).", "",
         "## Progress", "", f"- **Done: {done:,} of {total:,} slides ({100*done/max(total,1):.1f}%).** " + ", ".join(f"{k} {v}" for k, v in sorted(st.items()) if k != "done"), ]
    if allr:
        cs = sum(r["tiling_s"] + r["embed_s"] for r in allr); tl = sum(r["tiles"] for r in allr); sz = sum(r["size"] for r in allr)
        rc = rec; rcs = sum(r["tiling_s"] + r["embed_s"] for r in rc); rtl = sum(r["tiles"] for r in rc)
        first, lastd = min(r["t_done"] for r in allr), max(r["t_done"] for r in allr)
        rem = c.execute("select sum(size) from slides where state not in ('done','failed_download','failed_process')").fetchone()[0] or 0
        remn = total - done - st.get("failed_download", 0) - st.get("failed_process", 0)
        eta_gpu = rem * (cs / sz); eta_wall = remn * ((lastd - first) / max(len(allr) - 1, 1)) if len(allr) > 5 else None
        L += [f"- **Speed, last {len(rc)} slides:** {rtl/rcs:.0f} tiles/s ({1000*rcs/rtl:.2f} s per 1,000 tiles, tiling + both models, pauses excluded); "
              f"the processing test gave 346 tiles/s and 2.96 s per 1,000 tiles. All {len(allr)} done: {tl/cs:.0f} tiles/s, mean {tl/len(allr):,.0f} tiles per slide, "
              f"{cs/len(allr):.0f} s compute per slide.",
              f"- **Wall rate:** {3600*(len(allr)-1)/max(lastd-first,1):.1f} slides per hour since the first slide (includes pauses and waits).",
              f"- **ETA:** {tm(eta_gpu)} of GPU compute left (by raw size, {rem/1e12:.2f} TB to go)"
              + (f"; {tm(eta_wall)} at the observed wall rate" if eta_wall else "") + f". Finish about {datetime.datetime.now() + datetime.timedelta(seconds=(eta_wall or eta_gpu)):%Y-%m-%d %H:%M}."]
    dl = c.execute("select count(*) n, sum(dl_bytes) b, sum(dl_s) s, sum(dl_drops) d from slides where dl_bytes is not null").fetchone()
    if dl["n"]:
        L.append(f"- **Downloads:** {dl['n']:,} slides, {dl['b']/1e9:.0f} GB, mean {dl['b']/dl['s']/1e6:.2f} MB/s per stream, {dl['d']} dropped connections (all resumed), MD5 checked on every file. Waiting in the buffer: {st.get('downloaded',0)} (cap {CAP}).")
    ps = float(kv_get("gpu_pause_total_s", 0)); since = kv_get("gpu_paused_since", "")
    L.append(f"- **GPU pauses (another user's process):** {tm(ps)} in total" + (f"; **paused now since {datetime.datetime.fromtimestamp(float(since)):%H:%M}**" if since else "") + ".")
    free = lambda p: shutil.disk_usage(p).free / 1e12
    L += ["", "## Disk free", "", f"- /data/storage {free('/data/storage'):.2f} TB (raw slides go here until 300 GB free), /data/scratch {free('/data/scratch'):.2f} TB, /data/processed {free('/data/processed'):.2f} TB, /home/users {free('/home/users'):.2f} TB",
          "- Raw slides: " + ", ".join(f"{k} {v}" for k, v in c.execute("select case when raw_loc like '/data/storage%' then 'storage' when raw_loc like '%/_inbox/%' then 'inbox' when raw_loc like '/data/scratch%' then 'scratch (kept)' else 'none' end k, count(*) from slides where state not in ('pending') group by k").fetchall()),
          "", "## Slides done per type", "", "| Type | Slides | Done | Failed | Raw GB (done / all) | Output GB |", "|---|---|---|---|---|---|"]
    L += [f"| {r['type']} | {r['n']} | {r['d']} | {r['f']} | {r['dgb']:.0f} / {r['gb']:.0f} | {(r['ob'] or 0):.1f} |" for r in rows]
    errs = c.execute("select idx,file_name,state,dl_attempts,proc_attempts,error from slides where error is not null order by idx limit 40").fetchall()
    L += ["", "## Errors", "", f"{len(errs)} slide(s) have a recorded error (first 40 shown; a slide that later succeeded keeps no error):" if errs else "None recorded.", ""]
    L += [f"- {r['idx']} {r['file_name'][:23]} state={r['state']} dl={r['dl_attempts']} proc={r['proc_attempts']}: {r['error']}" for r in errs]
    L += ["", "## Checks", "",
          "- 20-slide check (2026-11-22 00:45): all 20 output files open with the expected shapes (`uni2` N x 1536, `conch` N x 512 with norm 1.0000, `conch_probs` N x 9, `conch_probs8` N x 8, `coords` N x 2 int32), all float16, no non-finite values. Speed 2.94 s per 1,000 tiles (processing test 2.96), 350 tiles/s on slides with at least 20,000 tiles (test 346), GPU 97.6%, 4,138 bytes per tile (test 4,122). No fix needed; the run continued.",
          "- Restart test (2026-11-22): the download service was restarted mid-run; in-flight downloads resumed from their partial files (see Errors for any that did not)."]
    L += ["", "## Choices made unattended", "",
          "- Downloads land on the fast scratch disk (`/data/scratch/juan/pancan_slides/_inbox`), where the tile loaders read them, and move to `/data/storage/juan/pancan_slides/<type>/` after processing. `/data/storage` is not read during the run, as the storage rule requires. When `/data/storage` has under 300 GB free the slide stays in `/data/scratch/juan/pancan_slides/<type>/`; the ledger column `raw_loc` says where each slide is.",
          "- `/data/storage` had 6.5 TB free at the start (df shows 6.0 TiB) against 7.15 TB of slides, and 300 GB is kept free, so about 0.9 TB of slides will stay on scratch.",
          f"- {STREAMS} download streams, buffer cap {CAP}, {WORKERS} loader workers, batch {BATCH} (the processing test's best setting).",
          "- Each output file holds `coords` (int32, level 0), `uni2` (1536), `conch` (512, L2-normalised), `conch_probs` (9 breast prompts, as in the test), `conch_probs8` (the 8 shared prompts of `analysis/pancan_tissue_check.py`), all float16, and `meta`. The probabilities are only for comparison: labels are recomputed from `conch`.",
          "- A slide is retried up to 5 times (download) or 3 times (processing), then marked failed and skipped.",
          "- The ledger is `/data/processed/juan/pancan/ledger.sqlite`; logs are in `/data/processed/juan/pancan/logs/`."]
    open(STATUS_DOC + ".tmp", "w").write("\n".join(L) + "\n"); os.replace(STATUS_DOC + ".tmp", STATUS_DOC)
    if commit:
        import subprocess
        msg = f"Pan-carcinoma run status: {done} of {total} slides done\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        subprocess.run(["git", "-C", REPO, "add", "docs/PANCAN_RUN_STATUS.md"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "-c", "user.name=juanduranmcgill", "-c", "user.email=juan.duran@mail.mcgill.ca", "commit", "-m", msg, "--", "docs/PANCAN_RUN_STATUS.md"], capture_output=True)


# ---------------------------------------------------------------- process service
def process_service():
    global torch, np, cv2, pynvml, psutil, OpenSlide, conch, te9, te8, SCALE, uni, conch_pre, uni_pre, dev, NH, PAUSED
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np, cv2, torch, psutil, pynvml
    from openslide import OpenSlide
    init_ledger(); PAUSED = False
    with txn() as c:
        for r in c.execute("select * from slides where state='processing'").fetchall():
            if os.path.exists(out_path(r)): c.execute("update slides set state='done', t_done=? where idx=?", (now(), r["idx"]))
            else: c.execute("update slides set state='downloaded' where idx=?", (r["idx"],))
    for dp, _, fs in os.walk(FEAT):
        for f in fs:
            if f.endswith(".tmp"): os.remove(os.path.join(dp, f))
    cv2.setNumThreads(os.cpu_count())
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda:0"); pynvml.nvmlInit(); NH = pynvml.nvmlDeviceGetHandleByIndex(0)
    wait_gpu()
    t0 = now()
    from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
    conch, conch_pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", device=dev); conch.eval()
    BREAST = ["Invasive breast carcinoma (tumor cells)", "Ductal carcinoma in situ (DCIS)", "Normal breast glands and lobules (TDLU)",
              "Fibrous desmoplastic stroma", "Adipose tissue (fat)", "Tumor-infiltrating lymphocytes (immune infiltrates)", "Necrosis or hemorrhage",
              "Blood vessels (vasculature)", "Muscle tissue (smooth or skeletal muscle)"]
    SHARED = ["Carcinoma (tumor cells)", "Normal epithelium and glands", "Fibrous desmoplastic stroma", "Lymphocytes (immune infiltrates)",
              "Necrosis or hemorrhage", "Adipose tissue (fat)", "Blood vessels (vasculature)", "Muscle tissue (smooth or skeletal muscle)"]
    with torch.inference_mode():
        enc = lambda L: conch.encode_text(tokenize(texts=[f"an H&E image of {t}" for t in L], tokenizer=get_tokenizer()).to(dev))
        te9, te8 = enc(BREAST), enc(SHARED); SCALE = conch.logit_scale.exp()
    import timm
    from huggingface_hub import hf_hub_download
    from torchvision import transforms
    uni = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True).to(dev)
    uni.load_state_dict(torch.load(hf_hub_download("MahmoodLab/UNI2-h", "pytorch_model.bin"), map_location="cpu"), strict=True); uni.eval()
    uni_pre = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    globals()["OpenSlide"] = OpenSlide
    log(f"process service up, models loaded in {now()-t0:.1f}s, workers {WORKERS}, batch {BATCH}")
    while True:
        if shutil.disk_usage("/data/processed").free < PROCESSED_MIN_FREE:
            log("PAUSE: /data/processed nearly full"); time.sleep(300); continue
        with txn() as c:
            r = c.execute("select * from slides where state='downloaded' order by idx limit 1").fetchone()
            if r: c.execute("update slides set state='processing' where idx=?", (r["idx"],))
        if r is None:
            if db().execute("select count(*) from slides where state in ('pending','downloading','downloaded','processing')").fetchone()[0] == 0:
                write_status(); log("all slides finished"); return 0
            time.sleep(10); continue
        try:
            process_one(r)
        except Exception as e:
            a = r["proc_attempts"] + 1; msg = f"process: {type(e).__name__}: {e}"[:300]
            upd(r["idx"], state="failed_process" if a >= MAX_PROC else "downloaded", proc_attempts=a, error=msg)
            log("process failed", r["file_name"], msg); log(traceback.format_exc())
            if "CUDA" in msg or "cuda" in msg: log("CUDA error: exiting for a clean restart"); sys.exit(3)


def others():
    out = []
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(NH):
            if p.pid == os.getpid(): continue
            try: u = psutil.Process(p.pid).username()
            except Exception: u = "?"
            if u != ME: out.append((p.pid, u, (p.usedGpuMemory or 0) // 2**20))
    except Exception: pass
    return out

def wait_gpu():
    """Block while another user's process is on the GPU. Returns seconds paused."""
    global PAUSED
    o = others()
    if not o: return 0
    t = now(); PAUSED = True; kv_set("gpu_paused_since", t); log("PAUSE: another user's process on the GPU:", o)
    while others(): time.sleep(15)
    p = now() - t; PAUSED = False; kv_set("gpu_paused_since", ""); kv_set("gpu_pause_total_s", float(kv_get("gpu_pause_total_s", 0)) + p)
    log(f"RESUME after {p:.0f}s"); return p

def tiling(path):
    t = now(); s = OpenSlide(path); W, H = s.level_dimensions[0]
    mag = s.properties.get("aperio.AppMag"); thumb = s.get_thumbnail((W // 32, H // 32)).convert("RGB"); s.close()
    g = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY); _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.resize(cv2.bitwise_not(m), (W, H), interpolation=cv2.INTER_NEAREST); coords = []
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            pm = mask[y:y + TILE, x:x + TILE]
            if pm.size and np.mean(pm > 0) > (1 - THR): coords.append((x, y))
    return np.array(coords, dtype=np.int64).reshape(-1, 2), {"tiling_s": now() - t, "width": W, "height": H, "app_mag": mag}

class Tiles:
    """Dataset (duck-typed for torch DataLoader): one read per tile, both transforms applied to that image."""
    def __init__(self, path, coords): self.path, self.coords, self.s = path, coords, None
    def __len__(self): return len(self.coords)
    def __getitem__(self, i):
        if self.s is None: self.s = OpenSlide(self.path)
        x, y = int(self.coords[i][0]), int(self.coords[i][1])
        try:
            img = self.s.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
            return conch_pre(img).half(), uni_pre(img).half(), 1
        except Exception:
            return torch.zeros(3, 448, 448, dtype=torch.half), torch.zeros(3, 224, 224, dtype=torch.half), 0

def winit(_): torch.set_num_threads(1)   # no cv2 call here: it deadlocks forked workers after the parent used cv2 threads

def run_slide(path, coords):
    import threading as th
    N = len(coords); U, C, P9, P8, XY = [], [], [], [], []; pos = 0; pause = 0.0; util = []; stop = False
    def samp():
        while not stop:
            if not PAUSED: util.append(pynvml.nvmlDeviceGetUtilizationRates(NH).gpu)
            time.sleep(0.5)
    th_ = th.Thread(target=samp, daemon=True); th_.start(); t = now()
    dl = torch.utils.data.DataLoader(Tiles(path, coords), batch_size=BATCH, num_workers=WORKERS, shuffle=False, pin_memory=True,
                                     prefetch_factor=4, worker_init_fn=winit)
    try:
        for c, u, ok in dl:
            pause += wait_gpu()
            c = c.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last); u = u.to(dev, non_blocking=True)
            ok = ok.to(dev).bool(); bs = len(ok)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ce = conch.encode_image(c); l9 = (ce @ te9.T) * SCALE; l8 = (ce @ te8.T) * SCALE; ue = uni(u)
            okc = ok.cpu().numpy()
            U.append(ue[ok].half().cpu()); C.append(ce[ok].half().cpu()); P9.append(l9.float().softmax(-1)[ok].half().cpu())
            P8.append(l8.float().softmax(-1)[ok].half().cpu()); XY.append(coords[pos:pos + bs][okc]); pos += bs
    finally:
        stop = True; th_.join()
    embed_s = now() - t - pause; n_ok = int(sum(len(x) for x in XY))
    o = {"coords": torch.from_numpy(np.concatenate(XY).astype(np.int32)), "uni2": torch.cat(U), "conch": torch.cat(C),
         "conch_probs": torch.cat(P9), "conch_probs8": torch.cat(P8)}
    return {"embed_s": embed_s, "pause_s": pause, "tiles_ok": n_ok, "gpu_util": float(np.mean(util)) if util else None}, o

def process_one(r):
    path = r["raw_loc"]; coords, tm = tiling(path)
    if len(coords) == 0: raise RuntimeError("no tissue tiles")
    with torch.inference_mode(): res, o = run_slide(path, coords)
    n = len(coords)
    if res["tiles_ok"] < 0.98 * n: raise RuntimeError(f"{n - res['tiles_ok']} of {n} tiles unreadable")
    k = res["tiles_ok"]
    assert o["coords"].shape == (k, 2) and o["uni2"].shape == (k, 1536) and o["conch"].shape == (k, 512) and o["conch_probs"].shape == (k, 9) \
        and o["conch_probs8"].shape == (k, 8) and all(o[x].dtype == torch.half for x in ("uni2", "conch", "conch_probs", "conch_probs8")), "bad output shapes or dtypes"
    assert all(bool(torch.isfinite(o[x]).all()) for x in ("uni2", "conch", "conch_probs", "conch_probs8")), "non-finite values"
    o["meta"] = {"file_id": r["file_id"], "file_name": r["file_name"], "type": r["type"], "tile": TILE, "level": 0, "threshold": THR,
                 "width": tm["width"], "height": tm["height"], "app_mag": tm["app_mag"], "n_tiles": k, "dtype": "float16"}
    final = out_path(r); os.makedirs(os.path.dirname(final), exist_ok=True); tmp = final + ".tmp"; t = now()
    with open(tmp, "wb") as f: torch.save(o, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, final); save_s = now() - t
    upd(r["idx"], state="done", tiles=k, tiling_s=tm["tiling_s"], embed_s=res["embed_s"], pause_s=res["pause_s"], tiles_per_s=n / res["embed_s"],
        gpu_util=res["gpu_util"], save_s=save_s, out_bytes=os.path.getsize(final), t_done=now(), app_mag=tm["app_mag"], error=None)
    log(f"done {r['idx']} {r['type']} {r['file_name'][:23]} tiles={k} tiling={tm['tiling_s']:.1f}s embed={res['embed_s']:.0f}s ({n/res['embed_s']:.0f} t/s) "
        f"gpu={res['gpu_util'] or 0:.0f}% pause={res['pause_s']:.0f}s out={os.path.getsize(final)/1e6:.0f}MB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download": sys.exit(download_service())
    elif cmd == "process": sys.exit(process_service())
    elif cmd == "status": init_ledger(); write_status(commit="--commit" in sys.argv)
    elif cmd == "init": init_ledger(); print("ledger ready")
    else: sys.exit("usage: pancan_runner.py download|process|status|init")
