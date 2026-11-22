"""Tissue prompt test: CONCH zero-shot accuracy of the shared tissue classes on labelled tiles
(Kather et al., CRC-VAL-HE-7K, Zenodo 1214456: 7,180 colorectal tiles, 224 px at 0.5 um/px, 9 classes).

Tiles are embedded as our pipeline does (CONCH transform, float16 input, autocast fp16, L2-normalised
image embedding). Kather classes map onto ours: TUM tumour, NORM normal epithelium, STR stroma,
LYM immune, DEB necrosis, ADI fat, MUS muscle. Vessels has no Kather class (a tile called vessels is
wrong); BACK and MUC have no class of ours and are reported separately.

Prompt sets (classification is always over our 8 classes):
  a  the 8 prompts of analysis/pancan_tissue_check.py, one each ("an H&E image of ...")
  b  several class names x CONCH's prompt templates, text embeddings averaged per class
  c  b plus colon-specific tumour and normal-epithelium names
Views: "full" (the 224 px tile, 112 um field) and "crop128" (the central 128 px, 64 um, the field of
view of our 256 px tiles at 40x), both resized by the CONCH transform.

Usage: python analysis/pancan_tissue_prompt_test.py --data DIR --cache FILE --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

OURS = ["tumour", "normal_epithelium", "stroma", "immune", "necrosis", "fat", "vessels", "muscle"]
KATHER_TO_OURS = {"TUM": "tumour", "NORM": "normal_epithelium", "STR": "stroma", "LYM": "immune",
                  "DEB": "necrosis", "ADI": "fat", "MUS": "muscle"}
UNMAPPED = ["BACK", "MUC"]

SET_A = {  # analysis/pancan_tissue_check.py, SHARED, in order
    "tumour": ["Carcinoma (tumor cells)"],
    "normal_epithelium": ["Normal epithelium and glands"],
    "stroma": ["Fibrous desmoplastic stroma"],
    "immune": ["Lymphocytes (immune infiltrates)"],
    "necrosis": ["Necrosis or hemorrhage"],
    "fat": ["Adipose tissue (fat)"],
    "vessels": ["Blood vessels (vasculature)"],
    "muscle": ["Muscle tissue (smooth or skeletal muscle)"],
}
NAMES_B = {  # generic (not site-specific) class names
    "tumour": ["carcinoma", "tumor", "tumor tissue", "invasive carcinoma", "malignant epithelium",
               "cancer cells", "carcinoma (tumor cells)"],
    "normal_epithelium": ["normal epithelium", "benign epithelium", "normal glands",
                          "non-neoplastic epithelium", "normal epithelium and glands"],
    "stroma": ["stroma", "fibrous stroma", "desmoplastic stroma", "connective tissue",
               "cancer-associated stroma"],
    "immune": ["lymphocytes", "immune cells", "inflammatory infiltrate", "lymphoid aggregate",
               "tumor-infiltrating lymphocytes"],
    "necrosis": ["necrosis", "necrotic tissue", "necrotic debris", "hemorrhage", "cellular debris"],
    "fat": ["adipose tissue", "fat", "adipocytes", "fat cells"],
    "vessels": ["blood vessels", "vasculature", "capillaries", "vessel wall"],
    "muscle": ["smooth muscle", "muscle tissue", "skeletal muscle", "muscle fibers"],
}
COLON_EXTRA = {
    "tumour": ["colorectal adenocarcinoma", "colon adenocarcinoma", "adenocarcinoma of the colon",
               "colorectal cancer epithelium"],
    "normal_epithelium": ["normal colon mucosa", "normal colonic glands", "colonic mucosa",
                          "normal colorectal epithelium"],
}
TEMPLATES = [  # CONCH zero-shot templates (CONCH repository prompts)
    "CLASSNAME.", "a photomicrograph showing CLASSNAME.", "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.", "an image showing CLASSNAME.", "an example of CLASSNAME.",
    "CLASSNAME is shown.", "this is CLASSNAME.", "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.", "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.", "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.", "presence of CLASSNAME.", "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.", "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.", "an H&E image of CLASSNAME.", "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
]


def prompt_sets():
    a = {k: [f"an H&E image of {v[0]}"] for k, v in SET_A.items()}
    b = {k: [t.replace("CLASSNAME", n) for n in NAMES_B[k] for t in TEMPLATES] for k in OURS}
    c = {k: b[k] + [t.replace("CLASSNAME", n) for n in COLON_EXTRA.get(k, []) for t in TEMPLATES] for k in OURS}
    return {"a_one_prompt": a, "b_ensemble": b, "c_ensemble_colon": c}


def load_model(dev):
    from conch.open_clip_custom import create_model_from_pretrained
    model, pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", device=dev)
    return model.eval(), pre


@torch.inference_mode()
def embed_tiles(model, pre, files, dev, view):
    from PIL import Image
    out = []
    for i in range(0, len(files), 128):
        ims = []
        for f in files[i:i + 128]:
            im = Image.open(f).convert("RGB")
            if view == "crop128":
                w, h = im.size
                im = im.crop(((w - 128) // 2, (h - 128) // 2, (w + 128) // 2, (h + 128) // 2))
            ims.append(pre(im).half())
        x = torch.stack(ims).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model.encode_image(x).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def class_text(model, ps, dev):
    from conch.open_clip_custom import get_tokenizer, tokenize
    rows = []
    for k in OURS:
        e = model.encode_text(tokenize(texts=ps[k], tokenizer=get_tokenizer()).to(dev)).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        rows.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.stack(rows).cpu()


def evaluate(img, labels, temb, scale):
    p = ((img @ temb.T) * scale).softmax(-1)
    pred = p.argmax(-1).numpy()
    lab = np.array(labels)
    res = {"per_class_accuracy": {}, "confusion": {}, "unmapped": {}}
    for kc, ours in KATHER_TO_OURS.items():
        m = lab == kc
        res["per_class_accuracy"][kc] = float((pred[m] == OURS.index(ours)).mean())
        res["confusion"][kc] = {OURS[j]: int((pred[m] == j).sum()) for j in range(len(OURS))}
    res["balanced_accuracy"] = float(np.mean(list(res["per_class_accuracy"].values())))
    # same, with vessels removed from the label set (vessels has no Kather class)
    keep = [i for i, n in enumerate(OURS) if n != "vessels"]
    pred7 = np.array(keep)[p[:, keep].argmax(-1).numpy()]
    res["balanced_accuracy_without_vessels"] = float(np.mean(
        [(pred7[lab == kc] == OURS.index(o)).mean() for kc, o in KATHER_TO_OURS.items()]))
    mapped = np.isin(lab, list(KATHER_TO_OURS))
    res["mapped_tile_accuracy"] = float(np.mean([pred[i] == OURS.index(KATHER_TO_OURS[lab[i]])
                                                 for i in np.where(mapped)[0]]))
    top = p.max(-1).values.numpy()
    corr = np.array([mapped[i] and pred[i] == OURS.index(KATHER_TO_OURS[lab[i]]) for i in range(len(lab))])
    res["median_top_prob_mapped"] = float(np.median(top[mapped]))
    res["accuracy_top_prob_ge_0.7"] = float(corr[mapped & (top >= 0.7)].mean()) if (mapped & (top >= 0.7)).any() else None
    res["share_mapped_top_prob_ge_0.7"] = float((top[mapped] >= 0.7).mean())
    for u in UNMAPPED:
        m = lab == u
        res["unmapped"][u] = {OURS[j]: float((pred[m] == j).mean()) for j in range(len(OURS))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding the 9 class folders")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join(a.data, "*", "*.tif")))
    labels = [os.path.basename(os.path.dirname(f)) for f in files]
    model, pre = load_model(dev)
    if os.path.exists(a.cache):
        emb = torch.load(a.cache)
    else:
        emb = {v: embed_tiles(model, pre, files, dev, v) for v in ("full", "crop128")}
        torch.save(emb, a.cache)
    scale = model.logit_scale.exp().float().cpu().detach()
    sets = prompt_sets()
    out = {"tiles": len(files), "per_class_tiles": {k: labels.count(k) for k in sorted(set(labels))},
           "n_prompts_per_class": {s: {k: len(v) for k, v in ps.items()} for s, ps in sets.items()},
           "results": {}}
    for s, ps in sets.items():
        temb = class_text(model, ps, dev)
        for v in ("full", "crop128"):
            out["results"][f"{s}/{v}"] = evaluate(emb[v], labels, temb, scale)
    # Reference: a linear probe on the same embeddings, 5-fold CV over tiles of the 7 mapped classes.
    # Optimistic: Kather gives no patient IDs, so tiles of one patient can be in train and test folds.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    lab = np.array(labels)
    m = np.isin(lab, list(KATHER_TO_OURS))
    out["linear_probe_5fold_tiles"] = {}
    for v in ("full", "crop128"):
        y = lab[m]
        p = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0), emb[v].numpy()[m], y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
        acc = {c: float((p[y == c] == c).mean()) for c in KATHER_TO_OURS}
        out["linear_probe_5fold_tiles"][v] = {"balanced_accuracy": float(np.mean(list(acc.values()))),
                                              "per_class_accuracy": acc}
    with open(os.path.join(a.out, "tissue_prompt_test.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, r in out["results"].items():
        print(f"{k:28s} bal_acc {r['balanced_accuracy']:.3f} (no vessels {r['balanced_accuracy_without_vessels']:.3f}) "
              + " ".join(f"{c}:{x:.2f}" for c, x in r["per_class_accuracy"].items()))


if __name__ == "__main__":
    main()
