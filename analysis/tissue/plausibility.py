"""Plausibility of the tissue labels on our own slides (no ground truth; reported, not pass or fail).

  python -I plausibility.py [stroma]   -> results/pancan/tissue_plausibility_by_type[_stroma].csv

Per cohort type: Spearman correlation across patients between the predicted share of a class and TCGA's
pathologist estimate for the frozen top and bottom sections of the same tumour (GDC biospecimen slide fields;
diagnostic slides have none). Different sections of the same tumour, so a loose positive relation is expected.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.classes import CLASSES  # noqa: E402

MANIFEST = Path("/data/processed/juan/pancan/manifest_patients.csv")
OUT = Path(__file__).resolve().parents[2] / "results" / "pancan"
PAIRS = {"tumour": "percent_tumor_nuclei", "necrosis": "percent_necrosis",
         "stroma": "percent_stromal_cells", "immune": "percent_lymphocyte_infiltration"}


def gdc_percentages(patients):
    """Mean over the primary tumour's frozen top and bottom slides, per patient."""
    fields = ",".join(["submitter_id", "samples.submitter_id", "samples.sample_type",
                       "samples.portions.slides.submitter_id"] +
                      [f"samples.portions.slides.{f}" for f in PAIRS.values()])
    rows = []
    for i in range(0, len(patients), 500):
        q = {"filters": {"op": "in", "content": {"field": "submitter_id", "value": patients[i:i + 500]}},
             "fields": fields, "size": "1000", "format": "json"}
        req = urllib.request.Request("https://api.gdc.cancer.gov/cases", data=json.dumps(q).encode(),
                                     headers={"Content-Type": "application/json"})
        for h in json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]:
            for s in h.get("samples", []):
                if s.get("sample_type") != "Primary Tumor":
                    continue
                for p in s.get("portions", []):
                    for sl in p.get("slides", []):
                        if "-DX" in sl.get("submitter_id", ""):
                            continue
                        rows.append({"patient": h["submitter_id"], **{f: sl.get(f) for f in PAIRS.values()}})
    df = pd.DataFrame(rows)
    return df.groupby("patient")[list(PAIRS.values())].mean()


def predicted_shares(labels):
    import torch
    man = pd.read_csv(MANIFEST)
    man = man[man["in_cohort"]]
    stem_to = dict(zip(man["file_name"].str.replace(".svs", "", regex=False), zip(man["patient"], man["type"])))
    rows = []
    for f in labels.glob("*/*.pt"):
        pt = stem_to.get(f.stem)
        if pt is None:
            continue
        lab = torch.load(f, map_location="cpu", weights_only=False)["label"].numpy()
        shares = np.bincount(lab, minlength=len(CLASSES)) / max(len(lab), 1)
        rows.append({"patient": pt[0], "type": pt[1], **{c: shares[k] for k, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows).set_index("patient")


def main(variant=None):
    from analysis.tissue.label_run import paths_for
    pred = predicted_shares(paths_for(variant)[1])
    gdc = gdc_percentages(sorted(pred.index))
    d = pred.join(gdc, how="inner")
    out = []
    for t, g in d.groupby("type"):
        row = {"type": t, "patients": len(g)}
        for cls, field in PAIRS.items():
            ok = g[[cls, field]].dropna()
            row[f"{cls}_n"] = len(ok)
            row[f"{cls}_spearman"] = (ok[cls].rank().corr(ok[field].rank()) if len(ok) >= 10 else np.nan)
        row.update({f"mean_share_{c}": float(g[c].mean()) for c in CLASSES})
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / f"tissue_plausibility_by_type{'_' + variant if variant else ''}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["type", "patients"] + [f"{c}_spearman" for c in PAIRS]].round(2).to_string(index=False))
    bad = res.loc[~(res["tumour_spearman"] > 0), "type"].tolist()
    print("types with tumour correlation <= 0 or not computable:", bad)


if __name__ == "__main__":
    main("stroma" if "stroma" in sys.argv[1:] else None)
