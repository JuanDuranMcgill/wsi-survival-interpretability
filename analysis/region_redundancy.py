#!/usr/bin/env python3
"""How much of each tissue compartment's content is already in the others?

Closes the caveat the paper's ablation section leaves open: a near-zero ablation
delta is consistent both with a region carrying no signal and with its signal
being available elsewhere. Those have opposite implications for the paper's
central claim, and nothing in the paper distinguishes them.

Measured on the UNI-2 tile features rather than on learned embeddings, so no
model, no GPU and no checkpoints are needed, and both cohorts are covered rather
than the one that happens to retain checkpoints. See
docs/REGION_REDUNDANCY_PLAN.md for the reasoning and for what this test can and
cannot settle.

Per region r, the reported number is the out-of-fold R2 of a ridge regression
predicting r's leading principal components from the other regions'. High means
redundant.

Everything is computed twice: on the raw pooled embeddings, and after centering
each region within TCGA source site. Institutions differ in staining protocol,
fixation and scanner, and the paper already reports that the risk score varies
sharply with source site, so between-site technical variation is a documented
confound here rather than a hypothetical one. The gap between the two numbers is
the share of apparent redundancy attributable to it.

Subtracting each patient's own across-region mean would seem the more direct
control and is wrong: it forces the centered regions to sum to zero, so every
region becomes exactly determined by the others and every R2 goes to one. A
synthetic check with one deliberately independent region catches this
immediately. Within-slide technical variation therefore remains in these
numbers, and is noted as a limitation rather than removed.

A permutation null is reported alongside: patient identity is shuffled within
each region independently, which destroys the per-patient coupling while leaving
each region's marginal distribution intact. It should sit at zero, and it is the
check that caught the first version of this script. With a fixed ridge penalty
on unstandardised components the null came back at -0.30 (BRCA) and -0.61
(BLCA), because out-of-fold R2 is pulled negative when the training fold is
small relative to the predictor count: roughly 290 patients against 112
predictor components in BLCA. That is a property of the estimator, not of the
data, and it made the absolute numbers uninterpretable. Standardising the
predictors and tuning the penalty by inner cross-validation returns the null to
zero at both cohort shapes while leaving a genuinely duplicated region at 0.90.

Usage (on the cluster, where the embeddings live):

    python analysis/region_redundancy.py --cohort blca \
        --class-dir "/path/UNI2_classwise_embeddings/Necrosis" \
        --class-dir "/path/UNI2_classwise_embeddings/..." \
        --out results/region_redundancy_blca.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Ridge penalties searched by inner cross-validation. A fixed penalty is not
# usable here: the predictors are PCA components whose variances span orders of
# magnitude, so any single alpha is far too weak for the leading ones and far
# too strong for the trailing ones.
ALPHAS = np.logspace(-1, 6, 24)


def site_of(case_id):
    """TCGA source site: the two characters after the project code."""
    parts = case_id.split("-")
    return parts[1] if len(parts) > 1 else "??"


def center_within_site(X, cases, min_site_n=5):
    """Remove each source site's mean from every region independently.

    Sites contributing fewer than min_site_n patients are pooled into one
    residual group, since a site of two gives a mean that is mostly noise.
    """
    sites = np.array([site_of(c) for c in cases])
    uniq, counts = np.unique(sites, return_counts=True)
    small = set(uniq[counts < min_site_n])
    sites = np.array(["__small__" if s in small else s for s in sites])
    out = X.copy()
    for s in np.unique(sites):
        m = sites == s
        out[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return out, sites


def pooled_embeddings(class_dirs, min_tiles):
    """One mean-pooled UNI-2 vector per patient per region.

    Streams each slide file and reduces it immediately, so peak memory is one
    slide rather than the cohort.
    """
    region_names = [os.path.basename(d.rstrip("/")) for d in class_dirs]
    per_region = {r: {} for r in region_names}
    tile_counts = {r: {} for r in region_names}
    coverage = {}          # a region absent from many slides is itself a finding

    for rname, rdir in zip(region_names, class_dirs):
        if not os.path.isdir(rdir):
            raise SystemExit(f"missing region directory: {rdir}")
        files = sorted(f for f in os.listdir(rdir) if f.endswith(".pt"))
        for fn in files:
            case = fn[:12]
            try:
                d = torch.load(os.path.join(rdir, fn), map_location="cpu",
                               weights_only=False)
                feats = d["features"]
            except Exception as exc:                       # unreadable slide
                print(f"  [skip] {rname}/{fn}: {exc}")
                continue
            f = np.asarray(feats, dtype=np.float64)
            if f.ndim != 2 or f.shape[0] < min_tiles:
                continue
            per_region[rname][case] = f.mean(axis=0)
            tile_counts[rname][case] = int(f.shape[0])
        coverage[rname] = {"slides_with_file": len(files),
                           "slides_pooled": len(per_region[rname])}
        print(f"  {rname}: {len(per_region[rname])} pooled of {len(files)} files")

    # keep only patients present in every region, so the design is balanced
    common = set.intersection(*(set(v) for v in per_region.values()))
    cases = sorted(common)
    if not cases:
        raise SystemExit("no patient has all regions present")
    X = np.stack([np.stack([per_region[r][c] for r in region_names])
                  for c in cases])                          # (n, R, D)
    counts = np.array([[tile_counts[r][c] for r in region_names] for c in cases])
    for r in region_names:
        coverage[r]["in_intersection"] = len(cases)
    return X, region_names, cases, counts, coverage


def redundancy(X, n_comp, n_folds, seed):
    """Out-of-fold R2 predicting each region's components from the others."""
    n, R, _ = X.shape
    n_comp = min(n_comp, max(2, n // (n_folds + 1)))
    out = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for r in range(R):
        num = np.zeros(n_comp)          # summed squared error, per component
        den = np.zeros(n_comp)          # summed squared deviation from the mean
        alphas_used = []
        for tr, te in kf.split(np.arange(n)):
            # PCA fitted on the training fold only, target and predictors alike
            p_t = PCA(n_components=n_comp, random_state=seed).fit(X[tr, r, :])
            y_tr, y_te = p_t.transform(X[tr, r, :]), p_t.transform(X[te, r, :])

            others = [s for s in range(R) if s != r]
            Z_tr, Z_te = [], []
            for s in others:
                p_s = PCA(n_components=n_comp, random_state=seed).fit(X[tr, s, :])
                Z_tr.append(p_s.transform(X[tr, s, :]))
                Z_te.append(p_s.transform(X[te, s, :]))
            Z_tr, Z_te = np.hstack(Z_tr), np.hstack(Z_te)

            # scale so one penalty is meaningful across components, then let
            # inner CV pick it; both are required for the null to sit at zero
            sc = StandardScaler().fit(Z_tr)
            fit = RidgeCV(alphas=ALPHAS).fit(sc.transform(Z_tr), y_tr)
            alphas_used.append(float(fit.alpha_))
            pred = fit.predict(sc.transform(Z_te))
            num += ((y_te - pred) ** 2).sum(axis=0)
            den += ((y_te - y_tr.mean(axis=0)) ** 2).sum(axis=0)

        r2 = 1.0 - num / np.maximum(den, 1e-12)
        w = den / den.sum()                       # weight by variance explained
        out[r] = {"r2_weighted": float((r2 * w).sum()),
                  "r2_per_component": [float(v) for v in r2],
                  "median_ridge_alpha": float(np.median(alphas_used))}
    return out


def pairwise_corr(X, n_comp, seed):
    """Mean |correlation| between each region pair on their leading components."""
    R = X.shape[1]
    P = [PCA(n_components=min(n_comp, X.shape[0] - 1),
             random_state=seed).fit_transform(X[:, r, :]) for r in range(R)]
    M = np.zeros((R, R))
    for a in range(R):
        for b in range(R):
            if a == b:
                M[a, b] = 1.0
                continue
            c = np.corrcoef(P[a].T, P[b].T)[:P[a].shape[1], P[a].shape[1]:]
            M[a, b] = float(np.abs(c).mean())
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--class-dir", action="append", required=True,
                    help="one per tissue region, same dirs the model trains on")
    ap.add_argument("--min-tiles", type=int, default=10,
                    help="skip a region on a slide with fewer tiles than this; "
                         "its mean would be dominated by a handful of patches")
    ap.add_argument("--n-comp", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[{args.cohort}] pooling {len(args.class_dir)} regions")
    X, names, cases, counts, coverage = pooled_embeddings(
        args.class_dir, args.min_tiles)
    print(f"[{args.cohort}] {X.shape[0]} patients x {X.shape[1]} regions "
          f"x {X.shape[2]} dims")

    Xs, sites = center_within_site(X, cases)
    n_sites = int(len(np.unique(sites)))
    print(f"[{args.cohort}] {n_sites} source-site groups after pooling small ones")

    rng = np.random.default_rng(args.seed)
    Xp = np.stack([X[rng.permutation(X.shape[0]), r, :] for r in range(X.shape[1])],
                  axis=1)                                   # permutation null

    raw = redundancy(X, args.n_comp, args.n_folds, args.seed)
    cen = redundancy(Xs, args.n_comp, args.n_folds, args.seed)
    perm = redundancy(Xp, args.n_comp, args.n_folds, args.seed)
    M_raw = pairwise_corr(X, args.n_comp, args.seed)
    M_cen = pairwise_corr(Xs, args.n_comp, args.seed)

    rep = {
        "cohort": args.cohort,
        "n_patients": int(X.shape[0]),
        "n_regions": int(X.shape[1]),
        "embedding_dim": int(X.shape[2]),
        "region_names": names,
        "params": {"n_components": args.n_comp, "n_folds": args.n_folds,
                   "ridge_alphas_searched": [float(ALPHAS[0]), float(ALPHAS[-1])],
                   "min_tiles": args.min_tiles,
                   "seed": args.seed},
        "median_tiles_per_region": {n: int(np.median(counts[:, i]))
                                    for i, n in enumerate(names)},
        "coverage_per_region": coverage,
        "coverage_note": (
            "slides_pooled counts slides with at least min_tiles tiles of that "
            "region; slides_with_file counts those with any. A region present "
            "on far fewer slides than the cohort size restricts the analysis to "
            "the patients who have it, and is worth reporting in its own right: "
            "a single global fusion weight cannot represent a compartment that "
            "is absent in a large share of patients. Note the model itself "
            "applies no minimum and will use however few tiles exist."),
        "n_source_site_groups": n_sites,
        "redundancy_raw": {names[i]: raw[i] for i in range(len(names))},
        "redundancy_site_centered": {names[i]: cen[i] for i in range(len(names))},
        "redundancy_permutation_null": {names[i]: perm[i] for i in range(len(names))},
        "pairwise_abs_corr_raw": M_raw.tolist(),
        "pairwise_abs_corr_site_centered": M_cen.tolist(),
        "reading": (
            "redundancy_* is the out-of-fold R2 predicting a region's leading "
            "principal components from the other regions'. High means the "
            "region adds little the others do not already carry, so a near-zero "
            "ablation delta for it cannot be read as absence of signal. The "
            "site-centered figures remove between-institution differences in "
            "staining, fixation and scanner; the drop from raw to centered is "
            "how much of the apparent redundancy that accounts for. "
            "Within-slide technical variation is not removed. The permutation "
            "null shuffles patient identity within each region and should sit "
            "near zero. "
            "Note this measures the input representation, so it can confirm the "
            "confound but not fully rule it out: cross-region self-attention "
            "can create redundancy the inputs did not have."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n[{args.cohort}] {'region':38} {'R2 raw':>8} {'R2 site':>9} {'R2 null':>9}")
    for i, nme in enumerate(names):
        print(f"[{args.cohort}] {nme[:38]:38} {raw[i]['r2_weighted']:8.3f} "
              f"{cen[i]['r2_weighted']:9.3f} {perm[i]['r2_weighted']:9.3f}")
    print(f"[{args.cohort}] wrote {args.out}")


if __name__ == "__main__":
    main()
