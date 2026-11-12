#!/usr/bin/env python
"""
Scan magnification of the slides used, per cohort and per TCGA source site.

Tiles were cut at each slide's native resolution with no resampling, so a tile
covers a different physical area at 20x and 40x. This reads the scanner metadata
(openslide.objective-power, aperio.AppMag, openslide.mpp-x) of every slide the
models use: the DX1 slides with embeddings in the first compartment directory
whose patient is in the main Shapley run (379 BLCA, 1,060 BRCA), the same rule
as MultiRegionDataset. Site is characters 6-7 of the barcode.

    python analysis/slide_magnification.py --svs-root /scratch/$USER --out-dir results

Writes results/magnification_{blca,brca}.json (counts only; no per-slide table).

    python analysis/slide_magnification.py --per-slide-csv /scratch/$USER/slide_magnification_per_slide.csv

writes one row per slide (cohort, patient_id, slide, site, mpp_x, objective_power)
instead of the JSONs, and checks it against the saved counts. Keep the CSV out of
the repo.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import openslide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_ablation import BLCA_BASE, BLCA_REGIONS, BRCA_BASE, BRCA_REGIONS  # noqa: E402


def magnification(props):
    for key in ("openslide.objective-power", "aperio.AppMag"):
        v = props.get(key)
        if v not in (None, ""):
            try:
                return f"{float(v):g}x"
            except ValueError:
                pass
    return None


def mpp_value(props):
    v = props.get("openslide.mpp-x") or props.get("aperio.MPP")
    return None if v in (None, "") else float(v)


def mpp_class(props):
    m = mpp_value(props)
    if m is None:
        return None
    return "~0.25 um/px (40x)" if m < 0.375 else "~0.5 um/px (20x)" if m < 0.75 else f"{m:.2f} um/px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs-root", default="/scratch/sorkwos")
    ap.add_argument("--shapley-root", default="/scratch/sorkwos")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--per-slide-csv", default=None,
                    help="write a per-slide table here instead of the count JSONs")
    args = ap.parse_args()
    if args.per_slide_csv and os.path.exists(args.per_slide_csv):
        raise SystemExit(f"{args.per_slide_csv} exists; refusing to overwrite")
    rows, mismatches = [], []

    svs = {}
    for p in glob.glob(os.path.join(args.svs_root, "TCGA-*", "**", "*.svs"), recursive=True):
        svs.setdefault(os.path.basename(p)[:-4], p)

    for cohort, base, regions in [("blca", BLCA_BASE, BLCA_REGIONS), ("brca", BRCA_BASE, BRCA_REGIONS)]:
        ids = set(np.concatenate([
            np.load(p)["patient_ids"].astype(str)
            for p in glob.glob(os.path.join(args.shapley_root, f"shapley_{cohort}", "round_*_local.npz"))]))
        ref = os.path.join(base, regions[0])
        slides = sorted(n[:-3] for n in os.listdir(ref)
                        if n.endswith(".pt") and "DX1" in n and not n.endswith("_A.pt") and n[:12] in ids)
        mags, mpps, by_site, mpp_mag = {}, {}, defaultdict(Counter), Counter()
        missing_file = []
        for s in slides:
            if s not in svs:
                missing_file.append(s); mag = mpp = "no slide file"; mpp_x = power = None
            else:
                props = openslide.OpenSlide(svs[s]).properties
                power, mpp_x = magnification(props), mpp_value(props)
                mag = power or "not recorded"
                mpp = mpp_class(props) or "not recorded"
            rows.append({"cohort": cohort, "patient_id": s[:12], "slide": s, "site": s[5:7],
                         "mpp_x": "" if mpp_x is None else mpp_x,
                         "objective_power": power or ""})
            mags[s], mpps[s] = mag, mpp
            by_site[s[5:7]][mag] += 1
            mpp_mag[f"{mag} | {mpp}"] += 1

        site_counts = {site: dict(c) for site, c in sorted(by_site.items())}
        recorded = lambda c: [k for k in c if k not in ("not recorded", "no slide file")]  # noqa: E731
        mixed = {site: dict(c) for site, c in by_site.items() if len(recorded(c)) > 1}
        out = {"cohort": cohort, "n_slides": len(slides), "n_patients": len({s[:12] for s in slides}),
               "magnification_counts": dict(Counter(mags.values())),
               "mpp_counts": dict(Counter(mpps.values())),
               "magnification_vs_mpp": dict(mpp_mag),
               "n_no_recorded_magnification": sum(m in ("not recorded", "no slide file") for m in mags.values()),
               "n_no_slide_file": len(missing_file),
               "n_sites": len(site_counts),
               "n_sites_mixing_magnifications": len(mixed),
               "n_slides_in_mixed_sites": int(sum(sum(c.values()) for c in mixed.values())),
               "per_site": site_counts,
               "source": "openslide.objective-power, else aperio.AppMag; mpp from openslide.mpp-x"}
        if args.per_slide_csv:
            saved = json.load(open(os.path.join(args.out_dir, f"magnification_{cohort}.json")))
            for key in ("n_slides", "magnification_counts", "mpp_counts", "magnification_vs_mpp"):
                if saved[key] != out[key]:
                    mismatches.append((cohort, key, saved[key], out[key]))
            print(f"[{cohort}] {len(slides)} slides, mpp {out['mpp_counts']}; "
                  f"matches results/magnification_{cohort}.json: "
                  f"{not any(m[0] == cohort for m in mismatches)}")
            continue
        path = os.path.join(args.out_dir, f"magnification_{cohort}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite")
        json.dump(out, open(path, "w"), indent=2)
        print(f"[{cohort}] {len(slides)} slides: {out['magnification_counts']}, mpp {out['mpp_counts']}, "
              f"{len(mixed)}/{len(site_counts)} sites mix magnifications; wrote {path}")

    if args.per_slide_csv:
        with open(args.per_slide_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.per_slide_csv} ({len(rows)} rows)")
        if mismatches:
            raise SystemExit(f"per-slide table does not reproduce the saved counts: {mismatches}")


if __name__ == "__main__":
    main()
