"""Compare two label sets on the same finished slides (2026-11-23). Beast, env pilot, CPU.

  python -I compare_labels.py [a b]   (default: stroma gtexn_stroma) -> results/pancan/tissue_label_compare.csv

Per type, on the patients both label sets have labelled: mean share of each compartment, and the Spearman
correlation of tumour, stroma and immune shares with TCGA's pathologist estimates (frozen sections; see
plausibility.py). 'stroma' = current stroma-kept labels; 'gtexn_stroma' = step 2 also taught GTEx normal tissue.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(__file__).resolve().parents[2] / "results" / "pancan" / "tissue_label_compare.csv"
SHARES = ["tumour", "normal_epithelium", "stroma", "immune", "fat", "other"]
CHECKED = ["tumour", "stroma", "immune"]


def main(variants=("stroma", "gtexn_stroma")):
    from analysis.tissue.label_run import paths_for
    from analysis.tissue.plausibility import PAIRS, gdc_percentages, predicted_shares
    pred = {v: predicted_shares(paths_for(v)[1]) for v in variants}
    common = sorted(set.intersection(*(set(p.index) for p in pred.values())))
    gdc = gdc_percentages(common)
    rows = []
    for v, p in pred.items():
        d = p.loc[common].join(gdc, how="left")
        for t, g in [*d.groupby("type"), ("ALL", d)]:
            row = {"labels": v, "type": t, "patients": len(g)}
            row.update({f"share_{c}": float(g[c].mean()) for c in SHARES})
            for c in CHECKED:
                ok = g[[c, PAIRS[c]]].dropna()
                row[f"{c}_spearman"] = (ok[c].rank().corr(ok[PAIRS[c]].rank())
                                        if t != "ALL" and len(ok) >= 10 else np.nan)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    cols = ["patients", "share_other", "share_immune", "share_stroma"] + [f"{c}_spearman" for c in CHECKED]
    print(res.pivot(index="type", columns="labels", values=cols).round(2).to_string())
    for v in variants:
        r = res[(res["labels"] == v) & (res["type"] != "ALL")]
        print(v, {c: int((r[f"{c}_spearman"] > 0.05).sum()) for c in CHECKED}, "types above 0.05 of",
              int(r["tumour_spearman"].notna().sum()))


if __name__ == "__main__":
    main(tuple(sys.argv[1:3]) if len(sys.argv) >= 3 else ("stroma", "gtexn_stroma"))
