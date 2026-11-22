"""Whole-group splits: tissue-source sites (TCGA) or donors (GTEx) go to one split each."""
from collections import Counter, defaultdict

import numpy as np


def tss_of(barcode):
    """Tissue-source-site code of a TCGA barcode: 'TCGA-A2-A0T2-...' -> 'A2'."""
    return str(barcode).split("-")[1]


def assign_groups(groups, strata, fractions=(0.70, 0.15, 0.15), seed=0):
    """Map each group to 'train', 'valid' or 'test'. A group's stratum is its most common
    stratum; groups are shuffled within their stratum (sorted first, so input order does
    not matter) and cut by the valid and test fractions, rounded."""
    seen = defaultdict(Counter)
    for g, s in zip(groups, strata):
        seen[g][s] += 1
    by_stratum = defaultdict(list)
    for g, c in seen.items():
        by_stratum[sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]].append(g)
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(by_stratum):
        gs = sorted(by_stratum[s])
        gs = [gs[i] for i in rng.permutation(len(gs))]
        n_valid, n_test = round(len(gs) * fractions[1]), round(len(gs) * fractions[2])
        for i, g in enumerate(gs):
            out[g] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    return out


def extend_splits(existing, groups, strata, seed=0):
    """Known groups keep their split; new groups are split among themselves with assign_groups."""
    new = [(g, s) for g, s in zip(groups, strata) if g not in existing]
    out = {g: existing[g] for g in groups if g in existing}
    if new:
        out.update(assign_groups([g for g, _ in new], [s for _, s in new], seed=seed))
    return out
