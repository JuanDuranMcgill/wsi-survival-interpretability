"""Check the shared pan-carcinoma tissue classes on slides whose CONCH image embeddings are
saved (no slide needed). Runs on the MEDomics-Beast in the pilot env.

For each slide: zero-shot labels for the 8 shared classes from the saved CONCH embedding,
with the same prompt template as the paper ("an H&E image of ..."). Reports per cancer
type: share of tiles per class, median top-class probability (the paper reports 0.93 for
BLCA and 0.85 for BRCA slides), share of tiles at or above 0.7 (the pathomic threshold), and
how many classes reach 5 confident tiles (what the pathomic step needs). For BRCA slides it
also compares the shared labels with the paper's 9 breast classes mapped onto them.

Sanity check: the 9 breast-class labels recomputed here must match the labels the
processing test saved (`conch_probs`).

Usage: python analysis/pancan_tissue_check.py --emb-dir DIR --manifest CSV --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

TEMPLATE = "an H&E image of {}"
SHARED = [  # (short name, prompt text); wording follows the paper's breast prompts
    ("tumour", "Carcinoma (tumor cells)"),
    ("normal_epithelium", "Normal epithelium and glands"),
    ("stroma", "Fibrous desmoplastic stroma"),
    ("immune", "Lymphocytes (immune infiltrates)"),
    ("necrosis", "Necrosis or hemorrhage"),
    ("fat", "Adipose tissue (fat)"),
    ("vessels", "Blood vessels (vasculature)"),
    ("muscle", "Muscle tissue (smooth or skeletal muscle)"),
]
BREAST = [  # preprocessing/classify_all_tiles_multi_6.py, in order
    "Invasive breast carcinoma (tumor cells)", "Ductal carcinoma in situ (DCIS)",
    "Normal breast glands and lobules (TDLU)", "Fibrous desmoplastic stroma",
    "Adipose tissue (fat)", "Tumor-infiltrating lymphocytes (immune infiltrates)",
    "Necrosis or hemorrhage", "Blood vessels (vasculature)",
    "Muscle tissue (smooth or skeletal muscle)",
]
BREAST_TO_SHARED = ["tumour", "tumour", "normal_epithelium", "stroma", "fat", "immune",
                    "necrosis", "vessels", "muscle"]
CONF = 0.7
MIN_TILES = 5


def text_embeddings(names):
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
                                            device="cpu")
    model.eval()
    out = {}
    with torch.no_grad():
        for key, classes in names.items():
            tok = tokenize(texts=[TEMPLATE.format(c) for c in classes], tokenizer=get_tokenizer())
            out[key] = model.encode_text(tok).float()
        scale = model.logit_scale.exp().float()
    return out, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = pd.read_csv(a.manifest)
    type_of = dict(zip(man["file_name"].str.replace(".svs", "", regex=False), man["type"]))
    cohort = set(man.loc[man["in_cohort"], "type"])
    temb, scale = text_embeddings({"shared": [p for _, p in SHARED], "breast": BREAST})
    names = [n for n, _ in SHARED]

    rows, sanity, brca_agree = [], [], []
    for f in sorted(glob.glob(os.path.join(a.emb_dir, "*_emb16.pt"))):
        stem = os.path.basename(f).replace("_emb16.pt", "")
        t = type_of.get(stem)
        if t not in cohort:
            continue
        o = torch.load(f, map_location="cpu", weights_only=False)
        img = o["conch"].float()
        pb = ((img @ temb["breast"].T) * scale).softmax(-1)
        sanity.append(float((pb.argmax(-1) == o["conch_probs"].float().argmax(-1)).float().mean()))
        p = ((img @ temb["shared"].T) * scale).softmax(-1)
        top, lab = p.max(-1)
        conf = top >= CONF
        row = {"slide": stem, "type": t, "tiles": int(len(lab)),
               "median_top_prob": float(top.median()), "share_conf_0.7": float(conf.float().mean())}
        for i, n in enumerate(names):
            row[f"share_{n}"] = float((lab == i).float().mean())
            row[f"conf_tiles_{n}"] = int(((lab == i) & conf).sum())
        row["classes_with_5_conf_tiles"] = int(sum(row[f"conf_tiles_{n}"] >= MIN_TILES for n in names))
        rows.append(row)
        if t == "BRCA":
            mapped = torch.tensor([names.index(BREAST_TO_SHARED[i]) for i in pb.argmax(-1).tolist()])
            brca_agree.append(float((mapped == lab).float().mean()))

    per_slide = pd.DataFrame(rows)
    per_slide.to_csv(os.path.join(a.out, "tissue_check_per_slide.csv"), index=False)
    cols = ["tiles", "median_top_prob", "share_conf_0.7", "classes_with_5_conf_tiles"] + \
           [f"share_{n}" for n in names]
    by_type = per_slide.groupby("type")[cols].mean().round(3)
    by_type.insert(0, "slides", per_slide.groupby("type").size())
    by_type.to_csv(os.path.join(a.out, "tissue_check_by_type.csv"))
    summary = {
        "classes": dict(SHARED), "template": TEMPLATE, "conf_threshold": CONF,
        "slides": int(len(per_slide)), "types": sorted(per_slide["type"].unique()),
        "sanity_breast_labels_reproduced_min": min(sanity) if sanity else None,
        "brca_agreement_with_paper_classes_mapped": brca_agree,
        "median_top_prob_all_tiles_mean_over_slides": float(per_slide["median_top_prob"].mean()),
    }
    with open(os.path.join(a.out, "tissue_check_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(by_type.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
