#!/usr/bin/env python
"""
Data for the one-patient case study (docs/HANDOFF_CASE_STUDY.md), from saved outputs only:
no training, no MED3pa rerun.

For each of the 10 pathologist-reviewed candidates, writes results/case_study/{patient}.json:

- held-out membership in each of the 16 reliability runs. Each run's discovery/evaluation
  split is rebuilt as the arm scripts build it:
  - the feature CSV inner-joined with the run's input_error_npz, in the CSV's row order;
  - the high-error label at the run's quantile of the finite errors, unscored patients 0;
  - discovery_eval_split (fraction and seed from the JSON), stratified on that label.
  Each rebuilt split is checked against the JSON's threshold, n_eval and evaluation
  prevalence. Every saved evaluation profile is re-applied and must match its patient,
  event and pair counts and its mean error;
- risk (out-of-bag, the run's input_error_npz) and its cohort percentile, PFI, patient
  error, comparable pairs, and the high-error label at q50/q70/q80/q90 (per arm, since
  each arm thresholds its own aligned cohort);
- tissue: each compartment's share of the patient's CONCH-labelled tiles
  (results/label_confidence_{cohort}_per_slide.csv);
- attribution: left as unavailable here, because the main Shapley run's per-round
  round_###_local.npz files are gone from Narval. They are on Trillium
  (/scratch/sorkwos/shapley_{cohort}); analysis/case_study_attribution.py fills this
  field from them, so run it after this script;
- pathomic features: the 10 leading features by surrogate importance at q50, raw and as
  a cohort percentile;
- clinical variables used by any recurring clinical rule (CSV units, which are the
  model's), plus age and stage as recorded in the TCGA-CDR table;
- the recurring profiles (top_profiles_eval, run_freq_pct >= 5) the patient meets in each
  run where it is held out. Rules are applied to the raw CSV values, as
  apply_profile_to_X does; the scaler only converts thresholds for display. For each
  profile: readable rule, size, high-error prevalence, in-profile c-index and 95%
  interval, cohort value, and whether the interval excludes it.
  The error classifier's per-patient probabilities were not saved (the scripts keep only
  the evaluation metrics), so none are reported.

    python analysis/case_study.py --out-dir results/case_study
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from comparable_pairs import MED3PA_ARMS  # noqa: E402
from profile_metrics import discovery_eval_split, profile_report  # noqa: E402
import figure_data as fd  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
DATA = os.path.expanduser("~/data")
CDR_XLSX = "/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx"
CANDIDATES = {"blca": ["TCGA-ZF-AA53", "TCGA-FT-A3EE", "TCGA-2F-A9KT", "TCGA-K4-A5RI", "TCGA-ZF-A9RF"],
              "brca": ["TCGA-B6-A0X1", "TCGA-B6-A0I9", "TCGA-B6-A0RG", "TCGA-D8-A1JL", "TCGA-LL-A8F5"]}
QS = [50, 70, 80, 90]
ARM_NAME = {"clinical": "clinical", "radiomic": "pathomic"}


def pct(values, x):
    """Cohort percentile: share of patients with a value <= x (finite values only)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(100 * np.mean(v <= x)) if np.isfinite(x) else None


def fnum(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def run_dir(cohort, arm, q):
    return os.path.join(RES, f"reliability_{cohort}_{arm}_q{q}")


def rebuild(cohort, arm, q):
    """Rebuild one run's split; re-apply its saved profiles; return everything per patient."""
    d = run_dir(cohort, arm, q)
    J = json.load(open(glob.glob(os.path.join(d, "reliability_*.json"))[0]))
    csv, x_dtype, profiles_csv = MED3PA_ARMS[cohort][arm]
    feats_df = pd.read_csv(os.path.join(DATA, csv), index_col="patient_id")
    feats_df.index = feats_df.index.astype(str)
    oob = np.load(J["input_error_npz"], allow_pickle=True)
    oob_df = pd.DataFrame({k: np.asarray(oob[k], np.float64) for k in ("risk", "error", "pfi_time", "pfi_event")},
                          index=[str(p)[:12] for p in oob["patient_ids"]])
    merged = feats_df.join(oob_df, how="inner")
    feature_names = [c for c in merged.columns if c not in oob_df.columns]
    ids = merged.index.to_numpy().astype(str)
    err = merged["error"].to_numpy()
    finite = np.isfinite(err)
    thresh = float(np.quantile(err[finite], q / 100))
    label = np.zeros(len(err), dtype=int)
    label[finite] = (err[finite] >= thresh).astype(int)
    sp = J["discovery_eval_split"]
    _, ev = discovery_eval_split(ids, frac_discovery=sp["frac_discovery"], seed=sp["seed"], stratify=label)
    check = {"error_threshold": [thresh, J["error_threshold"]], "n_eval": [int(ev.sum()), sp["n_eval"]],
             "eval_prevalence_high_error": [float(label[ev].mean()),
                                            J["classifier_metrics_on_eval"]["prevalence_high_error"]]}
    split_ok = all(abs(a - b) < 1e-9 for a, b in check.values())

    conds_by_profile = pd.read_csv(os.path.join(d, profiles_csv), index_col="profile")["_conditions_median"]
    X = merged[feature_names].to_numpy(dtype=x_dtype)
    col = {f: i for i, f in enumerate(feature_names)}
    X_eval = X[ev]
    t, e, r = (merged[k].to_numpy()[ev] for k in ("pfi_time", "pfi_event", "risk"))
    eval_ids = ids[ev]
    cval = fd._cohort_cindex(cohort)
    profiles, n_match = [], 0
    for tp in J["top_profiles_eval"]:
        conds = ast.literal_eval(conds_by_profile[tp["profile"]])
        mask = np.ones(len(X_eval), dtype=bool)
        for key, th in conds.items():
            feature, direction = key.rsplit("_", 1)
            if feature in col:
                x = X_eval[:, col[feature]]
                mask &= (x > th) if direction == "UP" else (x <= th)
        pr = profile_report(mask, label[ev], err[ev], t, e, r)
        ok = (all(pr[k] == tp[k] for k in ("n_patients", "n_events_in_profile", "n_comparable_pairs_in_profile"))
              and abs(pr["mean_patient_error_in_profile"] - tp["mean_patient_error_in_profile"]) < 1e-9)
        n_match += ok
        if tp.get("run_freq_pct", 0) < 5:
            continue
        lo, hi = tp["survival_model_cindex_ci95"]
        profiles.append({"members": set(eval_ids[mask]), "info": {
            "profile": tp["profile"], "readable_rule": tp["readable_rule"],
            "run_freq_pct": tp["run_freq_pct"], "n_patients": tp["n_patients"],
            "high_error_prevalence_in_profile": tp["high_error_prevalence_in_profile"],
            "high_error_prevalence_overall": tp["high_error_prevalence_overall"],
            "cindex_in_profile": tp["survival_model_cindex_in_profile"], "cindex_ci95": [lo, hi],
            "cohort_cindex": cval, "ci_excludes_cohort": bool(hi < cval or lo > cval),
            "reproduced_from_saved_rule": bool(ok)}})
    return {"json": J, "merged": merged, "feature_names": feature_names, "ids": ids, "eval_ids": set(eval_ids),
            "label": dict(zip(ids, label)), "threshold": thresh, "split_check": check, "split_ok": split_ok,
            "profiles_reproduced": [n_match, len(J["top_profiles_eval"])], "profiles": profiles}


def cdr_rows(ids):
    df = pd.read_excel(CDR_XLSX)
    df = df[df["bcr_patient_barcode"].isin(ids)].set_index("bcr_patient_barcode")
    keep = ["type", "age_at_initial_pathologic_diagnosis", "gender", "ajcc_pathologic_tumor_stage",
            "histological_type", "PFI", "PFI.time"]
    return {p: {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in df.loc[p, keep].items()} for p in df.index}


def tissue(cohort, pid):
    t = pd.read_csv(os.path.join(RES, f"label_confidence_{cohort}_per_slide.csv"))
    t = t[t["case"] == pid]
    slides = sorted(t["slide"].unique())
    t = t[t["slide"] == slides[0]] if slides else t  # one diagnostic slide per patient
    n = t["n_tiles"].sum()
    return {"slide": slides[0].replace("_tiles.jsonl", "") if slides else None,
            "n_slides_in_table": len(slides), "n_tiles": int(n),
            "share": {row.compartment: float(row.n_tiles / n) for row in t.itertuples()} if n else {},
            "short_names": {row.compartment: fd.short_name(row.compartment) for row in t.itertuples()},
            "source": f"results/label_confidence_{cohort}_per_slide.csv (CONCH top-1 labels)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(RES, "case_study"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    checks = {}
    for cohort, cands in CANDIDATES.items():
        runs = {(arm, q): rebuild(cohort, arm, q) for arm in ("clinical", "radiomic") for q in QS}
        for (arm, q), R in runs.items():
            checks[f"{cohort}_{arm}_q{q}"] = {"split": R["split_check"], "split_ok": R["split_ok"],
                                              "profiles_reproduced": R["profiles_reproduced"]}
            print(f"[{cohort} {arm} q{q}] split ok={R['split_ok']} profiles reproduced "
                  f"{R['profiles_reproduced'][0]}/{R['profiles_reproduced'][1]}")
        npz = np.load(runs[("clinical", 50)]["json"]["input_error_npz"], allow_pickle=True)
        npz_ids = np.array([str(p)[:12] for p in npz["patient_ids"]])
        risk_all = np.asarray(npz["risk"], float)
        cdr = cdr_rows(cands)
        imp = pd.read_csv(os.path.join(run_dir(cohort, "radiomic", 50), "radiomic_importance.csv")) \
            .sort_values("mean_importance", ascending=False).head(10)
        rad = runs[("radiomic", 50)]["merged"]
        clin = runs[("clinical", 50)]["merged"]
        clin_vars = sorted({c["feature"] for (arm, q), R in runs.items() if arm == "clinical"
                            for tp in R["json"]["top_profiles_eval"] if tp.get("run_freq_pct", 0) >= 5
                            for c in tp["thresholds"].values()})
        for pid in cands:
            i = np.where(npz_ids == pid)[0]
            if len(i) != 1:
                raise SystemExit(f"{pid}: {len(i)} rows in {npz.fid if hasattr(npz, 'fid') else 'npz'}")
            i = i[0]
            risk = float(npz["risk"][i])
            out = {"patient_id": pid, "cohort": cohort,
                   "slide_reviewed_by_pathologists": True,
                   "input_error_npz": runs[("clinical", 50)]["json"]["input_error_npz"],
                   "risk": {"oob_risk": risk, "cohort_percentile": pct(risk_all, risk),
                            "n_cohort": int(np.isfinite(risk_all).sum()),
                            "note": "out-of-bag risk (mean over the bootstrap rounds in which the patient was "
                                    "out-of-bag); percentile = % of cohort patients with risk <= this one"},
                   "outcome": {"pfi_time_days": fnum(npz["pfi_time"][i]), "pfi_event": int(npz["pfi_event"][i])},
                   "error": {"patient_error": fnum(npz["error"][i]),
                             "n_comparable_pairs": int(npz["n_comparable_pairs"][i]),
                             "definition": "share of the patient's comparable pairs ordered wrongly "
                                           "(analysis/patient_error.py)",
                             "high_error_label": {
                                 ARM_NAME[arm]: {f"q{q}": {"threshold": runs[(arm, q)]["threshold"],
                                                           "label": int(runs[(arm, q)]["label"].get(pid, -1))}
                                                 for q in QS} for arm in ("clinical", "radiomic")}},
                   "cdr": cdr.get(pid),
                   "tissue": tissue(cohort, pid),
                   "attribution": {"available": False,
                                   "note": "The main Shapley run's per-round local files "
                                           "(/scratch/sorkwos/shapley_" + cohort + "/round_###_local.npz) no "
                                           "longer exist on Narval (gone by 2026-11-19, likely the scratch "
                                           "purge); run analysis/case_study_attribution.py on Trillium, where "
                                           "they are, to fill this field."},
                   "pathomic_features_top10_q50": [], "clinical_variables": {}, "runs": {}}
            for row in imp.itertuples():
                f = row.feature
                v = float(rad.loc[pid, f]) if pid in rad.index else None
                out["pathomic_features_top10_q50"].append({
                    "rank": int(row.rank), "feature": f, "compartment": row.region,
                    "mean_importance": float(row.mean_importance), "value": v,
                    "cohort_percentile": pct(rad[f].to_numpy(float), v) if v is not None else None})
            for f in clin_vars:
                v = float(clin.loc[pid, f]) if pid in clin.index else None
                out["clinical_variables"][f] = {"value_csv_units": v,
                                                "cohort_percentile": pct(clin[f].to_numpy(float), v)
                                                if v is not None else None}
            n_match, n_differ = 0, 0
            for (arm, q), R in runs.items():
                key = f"{ARM_NAME[arm]}_q{q}"
                held = pid in R["eval_ids"]
                in_cohort = pid in R["label"]
                matches = [p["info"] for p in R["profiles"] if pid in p["members"]] if held else []
                n_match += len(matches)
                n_differ += sum(m["ci_excludes_cohort"] for m in matches)
                out["runs"][key] = {"in_run_cohort": in_cohort, "held_out": held,
                                    "high_error_label": int(R["label"][pid]) if in_cohort else None,
                                    "n_recurring_profiles": len(R["profiles"]),
                                    "matching_profiles": matches,
                                    "classifier_probability": None}
            out["summary"] = {
                "held_out_in": [k for k, v in out["runs"].items() if v["held_out"]],
                "n_matching_profiles": n_match, "n_matching_profiles_that_differ_from_cohort": n_differ,
                "classifier_probability_note": "not saved by the arm scripts (only evaluation metrics are kept)"}
            out["checks"] = {k: v for k, v in checks.items() if k.startswith(cohort)}
            with open(os.path.join(args.out_dir, f"{pid}.json"), "w") as fh:
                json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            print(f"[{cohort}] {pid}: held out in {out['summary']['held_out_in']}, "
                  f"{n_match} matching profiles ({n_differ} differ), risk pct "
                  f"{out['risk']['cohort_percentile']:.0f}")
    bad = [k for k, v in checks.items() if not v["split_ok"] or v["profiles_reproduced"][0] != v["profiles_reproduced"][1]]
    print("ALL CHECKS OK" if not bad else f"CHECK FAILURES: {bad}")


if __name__ == "__main__":
    main()
