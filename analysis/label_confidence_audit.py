#!/usr/bin/env python3
"""How confidently was each tile assigned to its tissue compartment?

Every downstream result in the paper, the fusion weights, the ablation, the
pathomic features and the redundancy analysis, is computed over tissue
compartments defined by one zero-shot CONCH call per tile: a single prompt
template, "an H&E image of {class}", and an argmax with no confidence threshold.
The paper argues that its central comparison does not depend on those labels
being right. It never measures how right they are.

No pathologist-labelled patches exist for these cohorts, so accuracy cannot be
measured. Ambiguity can. The classifier saved its full softmax over classes for
every tile, and from that this script reports:

- how confident each assignment was: the top-1 probability, the margin between
  the first and second class, the log-ratio ln(p1/p2), and the entropy of the
  softmax normalised to [0, 1]
- all of the above per assigned compartment, so a compartment made up largely
  of near-ties is visible as such
- which compartments are confused with which, as the distribution of the
  runner-up class and as the mean softmax per assigned class, to set against
  the errors the reviewing pathologists flagged
- what a confidence threshold would do: the share of each compartment's tiles
  that survive it, and how many slides would still have enough tiles of that
  compartment to build a region graph

The last item previews the planned rebuild of the regions, where ambiguous tiles
go to an uncertain bin instead of being forced into a class.

A softmax probability is not an accuracy. A tile can be assigned confidently and
wrongly, and CONCH's temperature makes its softmax sharper than its similarity
scores might suggest. What these numbers measure is how decisively the model
chose, which bounds how much the compartment names can be trusted without
establishing it.

Reads the JSONL written by preprocessing/classify_all_tiles_multi_6.py, one file
per slide, named {slide}_tiles.jsonl. Class names are taken from the files, so
the same script serves both cohorts. Slides are processed in parallel and every
accumulator has a fixed size, so memory does not grow with the cohort.

Usage (on the cluster, where the JSONL lives):

    python analysis/label_confidence_audit.py --cohort blca \\
        --jsonl-dir /home/sorkwos/links/scratch/blca_jsons \\
        --oob-npz results/oob_risk_blca.npz \\
        --out /scratch/$USER/label_confidence_blca.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

try:                                    # an order of magnitude faster, if present
    import orjson

    def _loads(line):
        return orjson.loads(line)
except ImportError:                     # pragma: no cover
    def _loads(line):
        return json.loads(line)

N_BINS = 200                            # histogram resolution for quantiles
LR_MAX = 10.0                           # ln(p1/p2) is clipped here for binning
NEAR_TIE = 0.10                         # a margin below this counts as a near-tie


def _empty(C, T):
    return {
        "n": np.zeros(C, np.int64),
        "h_top1": np.zeros((C, N_BINS), np.int64),
        "h_margin": np.zeros((C, N_BINS), np.int64),
        "h_logratio": np.zeros((C, N_BINS), np.int64),
        "sum_entropy": np.zeros(C),
        "sum_scores": np.zeros((C, C)),
        "runner_up": np.zeros((C, C), np.int64),
        "neartie_runner_up": np.zeros((C, C), np.int64),
        "n_top1_below_half": np.zeros(C, np.int64),
        "n_near_tie": np.zeros(C, np.int64),
        "survive": np.zeros((C, T), np.int64),
    }


def _bin(v, hi=1.0):
    return np.clip((np.asarray(v) / hi * N_BINS).astype(int), 0, N_BINS - 1)


def process_slide(args):
    """Accumulate one slide. Returns (slide, case, accumulators, per-class counts)."""
    path, classes, thresholds, cap, seed = args
    C, T = len(classes), len(thresholds)
    idx = {c: i for i, c in enumerate(classes)}
    with open(path, "rb") as f:
        lines = [ln for ln in f if ln.strip()]
    if cap and len(lines) > cap:
        rng = np.random.default_rng(seed + hash(os.path.basename(path)) % 100000)
        lines = [lines[i] for i in rng.choice(len(lines), cap, replace=False)]

    P = np.zeros((len(lines), C), np.float64)
    kept = 0
    for ln in lines:
        try:
            s = _loads(ln)["all_scores"]
        except Exception:
            continue
        row = np.array([s.get(c, 0.0) for c in classes], np.float64)
        tot = row.sum()
        if tot <= 0:
            continue
        P[kept] = row / tot             # float16 inference: renormalise exactly
        kept += 1
    P = P[:kept]
    acc = _empty(C, T)
    if kept == 0:
        return path, None, acc, np.zeros((C, T), np.int64)

    order = np.argsort(-P, axis=1)
    top, second = order[:, 0], order[:, 1]
    p1 = P[np.arange(kept), top]
    p2 = P[np.arange(kept), second]
    margin = p1 - p2
    logratio = np.log(np.maximum(p1, 1e-12) / np.maximum(p2, 1e-12))
    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(axis=1) / np.log(C)

    per_slide = np.zeros((C, T), np.int64)
    for k in range(C):
        m = top == k
        nk = int(m.sum())
        if nk == 0:
            continue
        acc["n"][k] = nk
        np.add.at(acc["h_top1"][k], _bin(p1[m]), 1)
        np.add.at(acc["h_margin"][k], _bin(margin[m]), 1)
        np.add.at(acc["h_logratio"][k], _bin(np.minimum(logratio[m], LR_MAX), LR_MAX), 1)
        acc["sum_entropy"][k] = float(ent[m].sum())
        acc["sum_scores"][k] = P[m].sum(axis=0)
        np.add.at(acc["runner_up"][k], second[m], 1)
        tie = m & (margin < NEAR_TIE)
        np.add.at(acc["neartie_runner_up"][k], second[tie], 1)
        acc["n_top1_below_half"][k] = int((p1[m] < 0.5).sum())
        acc["n_near_tie"][k] = int(tie.sum())
        for j, t in enumerate(thresholds):
            c_ = int((p1[m] >= t).sum())
            acc["survive"][k, j] = c_
            per_slide[k, j] = c_
    case = os.path.basename(path)[:12]
    return path, case, acc, per_slide


def _merge(a, b):
    for k in a:
        a[k] += b[k]
    return a


def _quantiles(hist, hi=1.0, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    tot = hist.sum()
    if tot == 0:
        return {f"q{int(q*100)}": None for q in qs}
    cdf = np.cumsum(hist) / tot
    edges = np.linspace(0, hi, N_BINS + 1)
    return {f"q{int(q*100)}": float(edges[np.searchsorted(cdf, q) + 1]) for q in qs}


def _rownorm(M):
    s = M.sum(axis=1, keepdims=True)
    return np.where(s > 0, M / np.maximum(s, 1), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--pattern", default="*DX1*_tiles.jsonl",
                    help="the model used DX1 slides only; match that")
    ap.add_argument("--oob-npz", default=None,
                    help="restrict to the modelled patients listed in this file")
    ap.add_argument("--patients-file", default=None,
                    help="restrict to the patients in this text file, one TCGA "
                         "barcode per line; results/modelled_patients_{cohort}.txt "
                         "is committed for this, since the oob_risk .npz files "
                         "are gitignored and absent on the cluster")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--min-tiles", type=int, default=10,
                    help="tiles a compartment needs on a slide to count as "
                         "present; matches region_redundancy.py")
    ap.add_argument("--max-tiles-per-slide", type=int, default=0,
                    help="random subsample per slide for a fast pass; 0 reads "
                         "every tile, which the coverage figures require")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    paths = sorted(glob.glob(os.path.join(args.jsonl_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no files match {args.pattern} in {args.jsonl_dir}")

    keep = None
    if args.patients_file:
        with open(args.patients_file) as f:
            keep = {ln.strip()[:12] for ln in f if ln.strip()}
    elif args.oob_npz:
        keep = {str(p) for p in np.load(args.oob_npz, allow_pickle=True)["patient_ids"]}
    if keep is not None:
        seen, sel = set(), []
        for p in paths:                  # one slide per patient, first wins,
            c = os.path.basename(p)[:12] # as in the dataset builder
            if c in keep and c not in seen:
                seen.add(c)
                sel.append(p)
        print(f"[{args.cohort}] {len(sel)} of {len(keep)} modelled patients have "
              f"a JSONL file ({len(paths)} files matched in total)")
        paths = sel
    if not paths:
        raise SystemExit("no JSONL file matches a modelled patient")

    with open(paths[0], "rb") as f:
        classes = list(_loads(f.readline())["all_scores"].keys())
    C, T = len(classes), len(thresholds)
    print(f"[{args.cohort}] {len(paths)} slides, {C} classes, {args.workers} workers")

    total = _empty(C, T)
    coverage = np.zeros((C, T), np.int64)
    coverage_any = np.zeros(C, np.int64)
    rows = []
    jobs = [(p, classes, thresholds, args.max_tiles_per_slide, args.seed) for p in paths]
    with Pool(args.workers) as pool:
        for i, (path, case, acc, per_slide) in enumerate(
                pool.imap_unordered(process_slide, jobs, chunksize=4), 1):
            if case is None:
                continue
            _merge(total, acc)
            coverage += per_slide >= args.min_tiles
            coverage_any += acc["n"] >= args.min_tiles
            for k, c in enumerate(classes):
                if acc["n"][k]:
                    rows.append((os.path.basename(path), case, c, int(acc["n"][k]),
                                 int(acc["survive"][k, thresholds.index(0.5)])
                                 if 0.5 in thresholds else -1))
            if i % 100 == 0:
                print(f"  {i}/{len(paths)} slides")

    n_all = int(total["n"].sum())
    per_class = {}
    for k, c in enumerate(classes):
        nk = int(total["n"][k])
        per_class[c] = {
            "n_tiles": nk,
            "share_of_tiles": nk / n_all if n_all else None,
            "top1_quantiles": _quantiles(total["h_top1"][k]),
            "margin_quantiles": _quantiles(total["h_margin"][k]),
            "logratio_quantiles": _quantiles(total["h_logratio"][k], LR_MAX),
            "frac_top1_below_half": total["n_top1_below_half"][k] / nk if nk else None,
            "frac_near_tie": total["n_near_tie"][k] / nk if nk else None,
            "mean_normalised_entropy": total["sum_entropy"][k] / nk if nk else None,
            "survival_fraction": {f"{t:.2f}": (total["survive"][k, j] / nk if nk else None)
                                  for j, t in enumerate(thresholds)},
            "slides_with_min_tiles": {"no_threshold": int(coverage_any[k]),
                                      **{f"{t:.2f}": int(coverage[k, j])
                                         for j, t in enumerate(thresholds)}},
        }

    g = {key: total[key].sum(axis=0) for key in ("h_top1", "h_margin", "h_logratio")}
    rep = {
        "cohort": args.cohort,
        "jsonl_dir": args.jsonl_dir,
        "n_slides": int(len(paths)),
        "n_tiles": n_all,
        "classes": classes,
        "prompt_template": "an H&E image of {class}",
        "params": {"thresholds": thresholds, "min_tiles": args.min_tiles,
                   "near_tie_margin": NEAR_TIE,
                   "max_tiles_per_slide": args.max_tiles_per_slide},
        "global": {
            "top1_quantiles": _quantiles(g["h_top1"]),
            "margin_quantiles": _quantiles(g["h_margin"]),
            "logratio_quantiles": _quantiles(g["h_logratio"], LR_MAX),
            "frac_top1_below_half": float(total["n_top1_below_half"].sum() / n_all),
            "frac_near_tie": float(total["n_near_tie"].sum() / n_all),
            "mean_normalised_entropy": float(total["sum_entropy"].sum() / n_all),
        },
        "per_class": per_class,
        "soft_confusion": {
            "rows_are": "assigned class",
            "columns_are": "mean softmax probability given to each class",
            "matrix": (total["sum_scores"] /
                       np.maximum(total["n"][:, None], 1)).tolist(),
        },
        "runner_up": {
            "rows_are": "assigned class",
            "columns_are": "share of those tiles whose second choice was this class",
            "matrix": _rownorm(total["runner_up"]).tolist(),
        },
        "near_tie_runner_up": {
            "rows_are": "assigned class, near-ties only (margin < 0.10)",
            "matrix": _rownorm(total["neartie_runner_up"]).tolist(),
        },
        "caveat": (
            "Softmax confidence is not accuracy: a tile can be assigned "
            "decisively and wrongly, and no labelled ground truth exists for "
            "these cohorts. These figures measure how decisively the classifier "
            "chose, which bounds the trust owed to the compartment names without "
            "establishing it."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    csv_path = os.path.splitext(args.out)[0] + "_per_slide.csv"
    with open(csv_path, "w") as f:
        f.write("slide,case,compartment,n_tiles,n_tiles_top1_ge_0.50\n")
        for r in sorted(rows):
            f.write(",".join(f'"{x}"' if isinstance(x, str) and "," in x else str(x)
                             for x in r) + "\n")

    print(f"\n[{args.cohort}] {len(paths)} slides, {n_all:,} tiles")
    print(f"[{args.cohort}] global: median top-1 "
          f"{rep['global']['top1_quantiles']['q50']:.2f}, "
          f"near-ties {rep['global']['frac_near_tie']:.1%}, "
          f"top-1 below 0.5 {rep['global']['frac_top1_below_half']:.1%}")
    head = f"{'compartment':36} {'share':>6} {'med p1':>7} {'ties':>6} {'<0.5':>6}"
    head += "".join(f" {'@'+f'{t:.1f}':>6}" for t in thresholds) + f" {'slides':>7}"
    print(f"[{args.cohort}] {head}")
    for c in classes:
        v = per_class[c]
        if not v["n_tiles"]:             # never anyone's top choice; itself a finding
            print(f"[{args.cohort}] {c[:36]:36} {'0.0%':>6}   no tile assigned")
            continue
        line = (f"{c[:36]:36} {v['share_of_tiles']:6.1%} "
                f"{v['top1_quantiles']['q50']:7.2f} {v['frac_near_tie']:6.1%} "
                f"{v['frac_top1_below_half']:6.1%}")
        line += "".join(f" {v['survival_fraction'][f'{t:.2f}']:6.1%}" for t in thresholds)
        line += f" {v['slides_with_min_tiles']['no_threshold']:7d}"
        print(f"[{args.cohort}] {line}")
    print(f"[{args.cohort}] wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
