"""Embed the public tissue tiles with exactly the run's CONCH and UNI-2 code (Beast, env pilot, GPU shared
with the preprocessing run, approved by Juan 2026-11-22).

  python -I embed_public.py check   re-embed 512 tiles of one run slide (copied to check/) and compare with
                                    the run's saved CONCH and UNI-2 embeddings (cosine >= 0.999 expected)
  python -I embed_public.py embed   embed every tile in public_index.csv and gtex_index.csv -> emb/<source>.pt

Model construction and transforms are copied from preprocessing/beast_runner/pancan_runner.py.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HOME", "/data/scratch/juan/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DATA = Path("/data/scratch/juan/datasets/tissue")
EMB = DATA / "emb"
FEAT = Path("/data/processed/juan/pancan/features")
BATCH, WORKERS, SEED = 256, 8, 0


def load_models():
    import timm
    import torch
    from conch.open_clip_custom import create_model_from_pretrained
    from huggingface_hub import hf_hub_download
    from torchvision import transforms
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda:0")
    conch, conch_pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", device=dev)
    conch.eval()
    uni = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24,
                            num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                            no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                            reg_tokens=8, dynamic_img_size=True).to(dev)
    uni.load_state_dict(torch.load(hf_hub_download("MahmoodLab/UNI2-h", "pytorch_model.bin"), map_location="cpu"),
                        strict=True)
    uni.eval()
    uni_pre = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return dev, conch, conch_pre, uni, uni_pre


class Images:
    """Duck-typed torch Dataset: image files, or regions of one slide; both transforms on the same image."""

    def __init__(self, items, conch_pre, uni_pre, slide=None, tile=256):
        self.items, self.cp, self.up, self.slide, self.tile, self.s = items, conch_pre, uni_pre, slide, tile, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        if self.slide:
            from openslide import OpenSlide
            if self.s is None:
                self.s = OpenSlide(self.slide)
            x, y = self.items[i]
            img = self.s.read_region((int(x), int(y)), 0, (self.tile, self.tile)).convert("RGB")
        else:
            img = Image.open(self.items[i]).convert("RGB")
        return self.cp(img).half(), self.up(img).half()


def run(ds, dev, conch, uni):
    import torch
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, num_workers=WORKERS, shuffle=False, pin_memory=True,
                                     prefetch_factor=4, worker_init_fn=lambda _: torch.set_num_threads(1))
    C, U = [], []
    with torch.inference_mode():
        for k, (c, u) in enumerate(dl):
            c = c.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            u = u.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ce = conch.encode_image(c)
                ue = uni(u)
            C.append(ce.half().cpu())
            U.append(ue.half().cpu())
            if k % 200 == 0:
                print(f"  batch {k} of {len(dl)}", flush=True)
    return torch.cat(C), torch.cat(U)


def check():
    import torch
    idx, typ, name = (DATA / "check" / "slide.txt").read_text().split("\t")[:3]
    out = FEAT / typ / (name.rsplit(".", 1)[0] + "_emb16.pt")
    if not out.exists():
        sys.exit(f"run output not there yet: {out}")
    o = torch.load(out, map_location="cpu", weights_only=False)
    pick = np.random.default_rng(SEED).choice(len(o["coords"]), size=min(512, len(o["coords"])), replace=False)
    dev, conch, conch_pre, uni, uni_pre = load_models()
    xy = o["coords"][pick].numpy()
    c, u = run(Images(list(map(tuple, xy)), conch_pre, uni_pre, slide=str(DATA / "check" / name)), dev, conch, uni)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)  # noqa: E731
    cc, cu = cos(c, o["conch"][pick]), cos(u, o["uni2"][pick])
    print(f"slide {name}: {len(pick)} tiles; CONCH cosine min {cc.min():.5f} mean {cc.mean():.5f}; "
          f"UNI-2 cosine min {cu.min():.5f} mean {cu.mean():.5f}; CONCH norm {c.float().norm(dim=1).mean():.4f}")
    print("PASS" if float(cc.min()) >= 0.999 and float(cu.min()) >= 0.999 else "FAIL")


def embed():
    import torch
    parts = [DATA / "public_index.csv", DATA / "gtex_index.csv"]
    idx = pd.concat([pd.read_csv(p, keep_default_na=False) for p in parts if p.exists()])
    idx = idx.drop_duplicates("tile_id")
    EMB.mkdir(parents=True, exist_ok=True)
    dev, conch, conch_pre, uni, uni_pre = load_models()
    for source, g in idx.groupby("source", sort=True):
        out = EMB / f"{source}.pt"
        if out.exists():
            print("skip", source, "(done)")
            continue
        print("embedding", source, len(g), "tiles", flush=True)
        c, u = run(Images(g["path"].tolist(), conch_pre, uni_pre), dev, conch, uni)
        tmp = out.with_suffix(".tmp")
        torch.save({"tile_id": g["tile_id"].tolist(), "conch": c, "uni2": u}, tmp)
        os.replace(tmp, out)
        print("done", source, tuple(c.shape), flush=True)


if __name__ == "__main__":
    cmds = {"check": check, "embed": embed}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
