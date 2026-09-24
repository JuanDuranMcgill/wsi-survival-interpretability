"""Shared loading for the compartment experiments.

Used by compartment_subsets.py (retraining on subsets of compartments),
abmil_compartment_shapley.py (the audit applied to a standard attention MIL
model) and planted_signal.py (the audit on an outcome whose driver is known).

The patients, their order and their labels always come from the full
multi-region dataset, so a view restricted to some compartments has the same
out-of-bag split, round for round, as the full model and as
region_shapley.py.
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_ablation import (  # noqa: E402
    BLCA_BASE, BLCA_REGIONS, BRCA_BASE, BRCA_REGIONS, MultiRegionDataset)

CDR_DEFAULT = ("/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/"
               "TCGA-CDR-SupplementalTableS1.xlsx")


def default_class_dirs(cohort):
    base, regions = (BLCA_BASE, BLCA_REGIONS) if cohort == "blca" else (BRCA_BASE, BRCA_REGIONS)
    return [os.path.join(base, r) for r in regions]


def load_survival(cohort, cdr_xlsx=None, surv_csv=None):
    """PFI table. surv_csv (bcr_patient_barcode, PFI, PFI.time) overrides the CDR."""
    if surv_csv:
        df = pd.read_csv(surv_csv)
    else:
        df = pd.read_excel(cdr_xlsx or CDR_DEFAULT)
        df = df[df["type"].str.contains("BLCA" if cohort == "blca" else "BRCA", na=False)]
    df = df[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
    df["PFI.time"] = pd.to_numeric(df["PFI.time"], errors="coerce")
    return df.dropna(subset=["PFI.time"])


def build_dataset(cohort, class_dirs=None, cdr_xlsx=None, surv_csv=None):
    class_dirs = class_dirs or default_class_dirs(cohort)
    ds = MultiRegionDataset(class_dirs, load_survival(cohort, cdr_xlsx, surv_csv))
    return ds, list(ds.region_names)


class RegionView(Dataset):
    """The same patients and labels, restricted to a subset of compartments.

    Region order is kept. Labels are shared with the base dataset, so replacing
    base.labels (as planted_signal.py does) is seen by every view.
    """

    def __init__(self, base, keep_idx):
        self.base = base
        self.keep = sorted(keep_idx)
        names = base.region_names
        self.region_names = [names[i] for i in self.keep]
        self.samples = [{names[i]: s[names[i]] for i in self.keep} for s in base.samples]

    @property
    def labels(self):
        return self.base.labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t, e = self.labels[i]
        return self.samples[i], torch.tensor(t), torch.tensor(e)


def resolve_regions(spec, names):
    """'Necrosis+adipose' or '0+3' -> sorted indices. Each token is an index or a
    case-insensitive substring that must match exactly one region name."""
    out = set()
    for tok in [t.strip() for t in spec.split("+") if t.strip()]:
        if tok.isdigit():
            out.add(int(tok))
            continue
        hits = [i for i, n in enumerate(names) if tok.lower() in n.lower()]
        if len(hits) != 1:
            raise SystemExit(f"region token {tok!r} matches {len(hits)} regions: "
                             f"{[names[i] for i in hits]}")
        out.add(hits[0])
    return sorted(out)


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:80]
