"""Teach step 2 the normal tissue of every GTEx organ (2026-11-23; Juan: "just fix it"). Beast, env pilot, CPU.

  python -I gtex_normal.py   -> results/pancan/gtex_normal/tissue_classifier_{weights,weights_stroma}.npz,
                                tissue_classifier_tests.csv, tissue_classifier_summary.json, gtex_heldout.csv

Why: step 2 learned normal tissue only from colon, breast and kidney. Normal GTEx lung, liver, thyroid, prostate and
uterus went to 'other', stroma or (lung) immune (results/pancan/tissue_other_check_gtex.csv). Here every GTEx
training tile joins step 2 as 'normal_epithelium' (read: normal tissue of the organ). The same procedure as
train_eval.py then runs unchanged: round 1, the fixed merge table, one retrain, the stroma-kept variant, the same
bars. New: held-out GTEx donors, share of each class per organ. Decided after seeing results; the paper lists it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis.tissue.train_eval as te  # noqa: E402
from analysis.tissue.bars import keep_stroma  # noqa: E402
from analysis.tissue.classes import CLASSES  # noqa: E402

OUT = te.OUT / "gtex_normal"


def gtex_heldout(model, X, idx, name):
    g = idx[(idx["source"] == "gtex") & (idx["split"] == "test") & (idx["version"] == "crop64")]
    g = g.drop_duplicates("tile_id")
    pred = np.array(CLASSES)[te.predict(model, X[g["row"]]).argmax(1)]
    t = pd.crosstab(g["tissue"].values, pred, normalize="index")
    t.insert(0, "tiles", g.groupby("tissue").size())
    return t.assign(model=name)


def main():
    te.GTEX_NORMAL = True
    OUT.mkdir(parents=True, exist_ok=True)
    model, rounds, long_m, X, idx = te.run_feature(te.CLASSIFIER)
    thr = te.thresholds(model, X, idx)
    te.save_classifier(OUT / "tissue_classifier_weights.npz", model, thr)
    merged_s = keep_stroma(model["merged"])
    print("stroma variant, merges:", merged_s, flush=True)
    model_s = te.train(X, idx, merged_s)
    longs, ev_s = [long_m], {}
    for version in ("crop64", "full"):
        long, ev_s[version] = te.evaluate(model_s, X, idx, version, te.CLASSIFIER, "stroma")
        longs.append(long)
    cb_s, organ_s = te.bars_of(ev_s["crop64"])
    thr_s = te.thresholds(model_s, X, idx)
    te.save_classifier(OUT / "tissue_classifier_weights_stroma.npz", model_s, thr_s)
    pd.concat(longs).to_csv(OUT / "tissue_classifier_tests.csv", index=False)
    gt = pd.concat([gtex_heldout(model, X, idx, "main"), gtex_heldout(model_s, X, idx, "stroma")])
    gt.to_csv(OUT / "gtex_heldout.csv")
    summary = {"classifier": te.CLASSIFIER, "gtex_normal": True, te.CLASSIFIER: rounds,
               "pathomic_thresholds": thr,
               "stroma_variant": {"merged": merged_s, "class_bars": cb_s,
                                  "organ_bars": organ_s.reset_index(names="type").to_dict("records"),
                                  "pathomic_thresholds": thr_s}}
    (OUT / "tissue_classifier_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps({"main_final_bars": rounds[-1]["class_bars"], "main_failed": rounds[-1]["failed"],
                      "main_merged": rounds[-1]["merged"], "stroma_bars": cb_s}, indent=2, default=float))
    print(gt.round(3).to_string())


if __name__ == "__main__":
    main()
