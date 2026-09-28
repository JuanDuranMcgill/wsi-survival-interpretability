#!/usr/bin/env python3
"""Compare pathologist tile labels with the zero-shot CONCH labels.

Reads the key written by preprocessing/export_label_tiles.py and one returned
labelling sheet per pathologist, and reports, against each pathologist and
against their consensus (tiles where both gave the same tissue class):

- overall accuracy of the CONCH label and Cohen's kappa;
- per-class precision (of tiles CONCH called class c, the share the
  pathologist agreed were c) and recall (of tiles the pathologist called c, the
  share CONCH also called c), with Wilson 95% intervals;
- the confusion matrix, rows = pathologist, columns = CONCH;
- agreement between the two pathologists (Cohen's kappa).

"Unsure" answers are excluded; "Other / not tissue" is kept as its own row.
Precision is the paper's quantity of interest: it says how much of what the
model treats as, say, necrosis is necrosis. Because the sample is stratified by
predicted class, recall is not a population recall and is reported for
completeness only.

Usage:

    python analysis/label_agreement.py --key label_tiles_blca/key.csv \\
        --sheet pathologist_A=MS_blca.xlsx --sheet pathologist_B=GSJ_blca.xlsx \\
        --out results/label_agreement_blca.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math

UNSURE = "Unsure"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else None


def read_sheet(path):
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=True)["labels"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[0] and row[2]:
            out[str(row[0])] = str(row[2]).strip()
    return out


def evaluate(pred, truth, classes):
    ids = [i for i in truth if truth[i] != UNSURE and i in pred]
    y, p = [truth[i] for i in ids], [pred[i] for i in ids]
    rows = sorted(set(y) | set(classes))
    cm = {r: {c: 0 for c in classes} for r in rows}
    for a, b in zip(y, p):
        cm[a][b] += 1
    per = {}
    for c in classes:
        called = [a for a, b in zip(y, p) if b == c]
        true_c = [b for a, b in zip(y, p) if a == c]
        per[c] = {"n_called": len(called), "precision": (called.count(c) / len(called)) if called else None,
                  "precision_95ci": wilson(called.count(c), len(called)),
                  "n_true": len(true_c), "recall": (true_c.count(c) / len(true_c)) if true_c else None}
    return {"n_tiles": len(ids), "accuracy": sum(a == b for a, b in zip(y, p)) / len(ids) if ids else None,
            "accuracy_95ci": wilson(sum(a == b for a, b in zip(y, p)), len(ids)),
            "cohen_kappa": kappa(y, p), "per_class": per,
            "confusion_rows_pathologist_cols_conch": cm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--sheet", action="append", required=True, help="name=path.xlsx, one per pathologist")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = list(csv.DictReader(open(args.key)))
    pred = {r["id"]: r["predicted"] for r in key}
    classes = sorted(set(pred.values()))
    sheets = {s.split("=", 1)[0]: read_sheet(s.split("=", 1)[1]) for s in args.sheet}
    out = {"n_exported": len(key), "classes": classes, "vs_conch": {}}
    for name, lab in sheets.items():
        out["vs_conch"][name] = evaluate(pred, lab, classes)
    if len(sheets) == 2:
        (na, a), (nb, b) = sheets.items()
        both = [i for i in a if i in b and a[i] != UNSURE and b[i] != UNSURE]
        out["between_pathologists"] = {"n_tiles": len(both),
                                       "cohen_kappa": kappa([a[i] for i in both], [b[i] for i in both]),
                                       "raw_agreement": sum(a[i] == b[i] for i in both) / len(both) if both else None}
        consensus = {i: a[i] for i in both if a[i] == b[i]}
        out["vs_conch"]["consensus"] = evaluate(pred, consensus, classes)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for name, r in out["vs_conch"].items():
        print(f"{name}: {r['n_tiles']} tiles, CONCH accuracy {r['accuracy']:.2f}, kappa {r['cohen_kappa']:.2f}")
        for c, v in r["per_class"].items():
            if v["precision"] is not None:
                print(f"   {c[:40]:40} precision {v['precision']:.2f} (n={v['n_called']})")
    if "between_pathologists" in out:
        print(f"between pathologists: kappa {out['between_pathologists']['cohen_kappa']:.2f}")


if __name__ == "__main__":
    main()
