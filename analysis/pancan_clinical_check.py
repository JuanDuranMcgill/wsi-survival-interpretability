"""How much PFI signal do pan-cancer clinical variable sets keep? Cox stratified by cancer
type, 5-fold CV, C-index computed within each type on out-of-fold predictions."""
import re
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def stratified_folds(labels, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labels), dtype=int)
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
cdr = pd.read_excel(f"{ROOT}/data/pancan/raw/TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
p = pd.read_csv(f"{ROOT}/data/pancan/manifest_patients.csv")
p = p[p.included_type][["patient", "type"]].merge(
    cdr.drop(columns=["type", "Unnamed: 0"]), left_on="patient", right_on="bcr_patient_barcode")
p = p[p["PFI.time"] > 0].reset_index(drop=True)


def stage_num(s):
    m = re.match(r"^Stage (IV|III|II|I)", str(s).strip())
    return {"I": 1, "II": 2, "III": 3, "IV": 4}[m.group(1)] if m else np.nan


GRADE = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "Low Grade": 1, "High Grade": 3, "GB": 1}
d = pd.DataFrame({"type": p["type"], "T": p["PFI.time"], "E": p["PFI"].astype(int)})
age = pd.to_numeric(p["age_at_initial_pathologic_diagnosis"], errors="coerce")
d["age"] = (age.fillna(age.median()) - age.median()) / age.std()
d["male"] = (p["gender"] == "MALE").astype(int)
ajcc = p["ajcc_pathologic_tumor_stage"].map(stage_num)
anyst = ajcc.fillna(p["clinical_stage"].map(stage_num))
grade = p["histological_grade"].map(GRADE)
for name, v in (("stage_ajcc", ajcc), ("stage_any", anyst), ("grade", grade)):
    d[name] = v.fillna(0)
    d[name + "_na"] = v.isna().astype(int)

SETS = {
    "age+sex": ["age", "male"],
    "+AJCC stage": ["age", "male", "stage_ajcc", "stage_ajcc_na"],
    "+any stage": ["age", "male", "stage_any", "stage_any_na"],
    "+any stage+grade": ["age", "male", "stage_any", "stage_any_na", "grade", "grade_na"],
}

folds = stratified_folds((d["type"] + d["E"].astype(str)).values)
rows = {}
for name, cols in SETS.items():
    oof = np.zeros(len(d))
    for tr, te in folds:
        cph = CoxPHFitter(penalizer=0.01)
        sub = d.iloc[tr][cols + ["T", "E", "type"]]
        # drop columns constant in training (avoid singular fits)
        keep = [c for c in cols if sub[c].nunique() > 1]
        cph.fit(sub[keep + ["T", "E", "type"]], "T", "E", strata=["type"])
        oof[te] = (d.iloc[te][keep].values @ cph.params_[keep].values)
    per = {}
    for t, g in d.assign(r=oof).groupby("type"):
        per[t] = concordance_index(g["T"], -g["r"], g["E"])
    rows[name] = per

res = pd.DataFrame(rows)
ev = d.groupby("type")["E"].sum()
res.insert(0, "events", ev)
res.insert(1, "stage?", [("AJCC" if (p.loc[p.type == t, "ajcc_pathologic_tumor_stage"].map(stage_num).notna().mean() > .5)
                          else "FIGO" if anyst[p.type == t].notna().mean() > .5 else "none")
                         for t in res.index])
res.insert(2, "grade%", [int(round(100 * grade[p.type == t].notna().mean())) for t in res.index])
w = res["events"] / res["events"].sum()
summary = {c: float((res[c] * w).sum()) for c in SETS}
pd.set_option("display.width", 200)
print(res.round(3).sort_values("events", ascending=False).to_string())
print("event-weighted within-type C:", {k: round(v, 3) for k, v in summary.items()})
nost = res[res["stage?"] == "none"]
print("types with no stage:", list(nost.index), "patients", int(p.type.isin(nost.index).sum()),
      "events", int(nost["events"].sum()))

res.round(3).sort_values("events", ascending=False).to_csv(f"{ROOT}/results/pancan/clinical_variables_check.csv")
