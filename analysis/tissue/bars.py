"""Pass bars for the tissue classifier, fixed before any result (spec: Pass bars)."""
import numpy as np
import pandas as pd

from analysis.tissue.classes import MERGE

RECALL_MIN = 0.85
PRECISION_MIN = 0.80
TUMOUR_RECALL_MIN = 0.85
GTEX_FALSE_TUMOUR_MAX = 0.10
NORMAL_FROM_TUMOUR_MAX = 0.05
PATHOMIC_PRECISION = 0.90


def class_bars(table):
    """table: test_set, class, n_true, recall, precision. A class passes if, on every test set
    with true tiles of that class, recall >= RECALL_MIN and precision >= PRECISION_MIN.
    A class no test set labels does not pass."""
    out = {}
    for c, g in table.groupby("class", sort=False):
        g = g[g["n_true"] > 0]
        out[c] = bool(len(g)) and bool(((g["recall"] >= RECALL_MIN)
                                        & (g["precision"].fillna(-1) >= PRECISION_MIN)).all())
    return out


def tumour_organ_bars(recall_by_type, gtex_false_tumour):
    """Per type: TCGA-UT tumour recall and share of GTEx tiles called tumour; passes if both bars hold."""
    types = sorted(set(recall_by_type) | set(gtex_false_tumour))
    t = pd.DataFrame({"recall": [recall_by_type.get(k, np.nan) for k in types],
                      "false_tumour": [gtex_false_tumour.get(k, np.nan) for k in types]}, index=types)
    t["passes"] = (t["recall"] >= TUMOUR_RECALL_MIN) & (t["false_tumour"] <= GTEX_FALSE_TUMOUR_MAX)
    return t


def normal_epithelium_bar(passes_on_sets, tumour_called_normal):
    """Passes on its labelled test sets and under NORMAL_FROM_TUMOUR_MAX of TCGA-UT tumour tiles
    called normal epithelium in every type."""
    return bool(passes_on_sets) and all(v < NORMAL_FROM_TUMOUR_MAX for v in tumour_called_normal.values())


def merges_for(failed):
    """Fixed merge-table entries for the classes that failed."""
    return {c: MERGE[c] for c in sorted(failed) if c in MERGE}


def premerge_small(train_counts, min_tiles):
    """Merge-table entries for classes with fewer than min_tiles training tiles (merged before training)."""
    return {c: MERGE[c] for c, n in sorted(train_counts.items()) if n < min_tiles and c in MERGE}
