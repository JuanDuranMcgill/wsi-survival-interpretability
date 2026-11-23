"""Pieces of check R that need no MED3pa (testable anywhere): OOB risk per patient, the clinical feature matrix,
and the within-type C-index of a profile with its bootstrap interval."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pancan.metrics import cindex_within

GRADE = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "Low Grade": 1, "High Grade": 3, "GB": 1}  # as pancan_clinical_check


def stage_num(s):
    m = re.match(r"^Stage (IV|III|II|I)", str(s).strip())
    return {"I": 1, "II": 2, "III": 3, "IV": 4}[m.group(1)] if m else np.nan


def oob_risk(run_dir):
    """Mean OOB risk per patient over the rounds where the patient was OOB, at the epoch with the best mean
    within-type OOB C-index across rounds (as the paper's oob_risk.py). df.attrs['epoch'] is that epoch (1-based)."""
    files = sorted(Path(run_dir).glob("round_*/oob.npz"))
    C = np.array([np.load(f)["cindex"] for f in files], dtype=float)
    ep = int(np.nanargmax(np.nanmean(C, 0)))
    parts = []
    for f in files:
        o = np.load(f)
        parts.append(pd.DataFrame({"patient": o["patient"], "time": o["time"], "event": o["event"],
                                   "type": o["type"], "risk": o["risk"][ep]}))
    d = pd.concat(parts).groupby("patient").agg(time=("time", "first"), event=("event", "first"),
                                                type=("type", "first"), risk=("risk", "mean"),
                                                n_oob=("risk", "size")).reset_index()
    d.attrs["epoch"] = ep + 1
    return d


def clinical_matrix(c):
    """Age (z, median-imputed), male, stage (AJCC, else clinical/FIGO; 0 + indicator if missing), grade (0 +
    indicator if missing), cancer type one-hot. Index: patient."""
    age = pd.to_numeric(c["age"], errors="coerce")
    X = pd.DataFrame(index=c["patient"].values)
    X["age"] = ((age.fillna(age.median()) - age.median()) / (age.std() or 1.0)).values
    X["male"] = (c["sex"].values == "MALE").astype(int)
    stage = c["ajcc_pathologic_tumor_stage"].map(stage_num).fillna(c["clinical_stage"].map(stage_num))
    grade = c["histological_grade"].map(GRADE)
    for name, v in (("stage", stage), ("grade", grade)):
        X[name] = v.fillna(0).values
        X[name + "_na"] = v.isna().astype(int).values
    for t in sorted(c["type"].unique()):
        X[f"type_{t}"] = (c["type"].values == t).astype(int)
    return X


def profile_cindex(mask, t, e, r, types, n_boot=1000, seed=0):
    m = np.flatnonzero(np.asarray(mask, bool))
    c = cindex_within(t[m], e[m], r[m], types[m])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = m[rng.integers(0, len(m), len(m))]
        v = cindex_within(t[s], e[s], r[s], types[s])
        if np.isfinite(v):
            vals.append(v)
    ci = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (np.nan, np.nan)
    return c, ci


def null_risk(risk, types, seed):
    """A model with no skill: the risk scores shuffled within each cancer type (same values per type)."""
    rng = np.random.default_rng(seed)
    out = np.asarray(risk, dtype=float).copy()
    types = np.asarray(types)
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        out[idx] = out[rng.permutation(idx)]
    return out


def rf_scores(X_disc, y_disc, X_eval, names, run_idx, test_size=0.3, ref_size=0.25):
    """The error classifier of the paper's run_single_med3pa (same seed, splits and forest), without the MED3pa
    profile search: its predicted high-error probability on the evaluation set. Used for the null control."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    seed = 1000 + run_idx
    xd = pd.DataFrame(X_disc, columns=names)
    pool, _, ypool, _ = train_test_split(xd, y_disc, test_size=test_size, random_state=seed, stratify=y_disc)
    xt, _, yt, _ = train_test_split(pool, ypool, test_size=ref_size, random_state=seed, stratify=ypool)
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, class_weight="balanced", n_jobs=1)
    return clf.fit(xt, yt).predict_proba(pd.DataFrame(X_eval, columns=names))[:, 1]
