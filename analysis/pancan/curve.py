"""Events-curve subsets (design decision 8), fixed once before the first curve point (Beast, any env).

  python -I curve.py   -> /data/processed/juan/pancan/curve/e<size>_r<k>.csv (patient lists; never in git)
                          results/pancan/curve_subsets_counts.csv (patients and events per type and subset)

Each subset keeps the same share of every type's events and of every type's censored patients (sampling by patient,
stratified by type and event), so a subset of E events has each type's events in proportion to the full cohort.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANCAN = Path("/data/processed/juan/pancan")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "pancan"
SIZES, REPEATS, SEED = (150, 300, 600, 1200), 3, 0


def curve_subsets(cohort, sizes=SIZES, repeats=REPEATS, seed=SEED):
    frac_of = {s: s / cohort["pfi_event"].sum() for s in sizes}
    out = {}
    for s in sizes:
        for k in range(repeats):
            rng = np.random.default_rng(seed * 100000 + s * 10 + k)
            keep = []
            for _, g in cohort.groupby(["type", "pfi_event"], sort=True):
                n = int(round(frac_of[s] * len(g)))
                keep += rng.choice(g["patient"].values, n, replace=False).tolist()
            out[(s, k)] = sorted(keep)
    return out


if __name__ == "__main__":
    c = pd.read_csv(PANCAN / "cohort.csv")
    subs = curve_subsets(c)
    (PANCAN / "curve").mkdir(exist_ok=True)
    rows = []
    for (s, k), pts in subs.items():
        pd.DataFrame({"patient": pts}).to_csv(PANCAN / "curve" / f"e{s}_r{k}.csv", index=False)
        g = c[c["patient"].isin(pts)].groupby("type").agg(patients=("patient", "size"), events=("pfi_event", "sum"))
        rows.append(g.assign(subset=f"e{s}_r{k}").reset_index())
    t = pd.concat(rows)
    t.to_csv(RESULTS / "curve_subsets_counts.csv", index=False)
    print(t.groupby("subset")[["patients", "events"]].sum(), file=sys.stderr)
