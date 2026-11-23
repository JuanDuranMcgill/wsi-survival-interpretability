"""Hospital-held-out rounds (design decision 5): per round, whole institutions are out of bag (OOB) for every type.

An institution goes OOB if at least one type it contributes to is still below `oob_frac` OOB and every type it
touches stays at or below `max_frac` after adding it, so a dominant hospital (OV 54%, LIHC 41%) is never the whole
OOB set, and a type whose hospitals also serve other types still gets OOB patients. Institutions are
tried in a random order per round (seed = seed * 1000 + round).
"""
import numpy as np
import pandas as pd


def site_rounds(cohort, n_rounds, oob_frac=0.2, max_frac=0.4, seed=0):
    n_type = cohort.groupby("type").size()
    by_site = cohort.groupby(["site", "type"]).size().unstack(fill_value=0)
    sites = by_site.index.to_numpy()
    out = []
    for r in range(n_rounds):
        rng = np.random.default_rng(seed * 1000 + r)
        oob_n = pd.Series(0, index=n_type.index)
        chosen = set()
        for s in rng.permutation(sites):
            add = by_site.loc[s]
            touched = add[add > 0].index
            if not ((oob_n[touched] / n_type[touched]) < oob_frac).any():
                continue
            if (((oob_n[touched] + add[touched]) / n_type[touched]) > max_frac).any():
                continue
            oob_n[touched] += add[touched]
            chosen.add(s)
        out.append(pd.DataFrame({"round": r, "patient": cohort["patient"].values,
                                 "in_train": ~cohort["site"].isin(chosen).values}))
    return pd.concat(out, ignore_index=True)


def round_summary(cohort, rounds):
    """Per round and type: OOB patients, OOB events, OOB share (types with no OOB events are flagged)."""
    m = rounds.merge(cohort[["patient", "type", "pfi_event"]], on="patient")
    oob = m[~m["in_train"]]
    s = oob.groupby(["round", "type"]).agg(oob_patients=("patient", "size"), oob_events=("pfi_event", "sum"))
    s = s.reindex(pd.MultiIndex.from_product([sorted(m["round"].unique()), sorted(m["type"].unique())],
                                             names=["round", "type"]), fill_value=0)
    s["oob_share"] = s["oob_patients"] / m.groupby(["round", "type"]).size()
    return s.reset_index()
