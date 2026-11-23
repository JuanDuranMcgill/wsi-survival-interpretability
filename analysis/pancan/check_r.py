"""Check R for the pan-carcinoma runs: patient error within type, then MED3pa as in the paper (Beast, env audit).

  python -I check_r.py <run_dir> [--features clinical|<csv>] [--runs 200] [--quantiles 0.5,0.7,0.8,0.9]

run_dir = .../runs/<variant>/<model>/<subset>. Steps, each as the paper's *_linear_med3pa.py unless noted:
OOB risk per patient (check_r_core.oob_risk); error = symmetric concordance error over same-type pairs (new);
high-error label at each quantile; locked 60/40 discovery/evaluation split (seed 20260908, stratified on the label);
<runs> MED3pa runs on the discovery set (the paper's run_single_med3pa and settings); AUROC of the error
classifier on the evaluation set with a permutation p; recurring profiles (>= 5% of runs, top 20) and their
in-profile C-index within type (new) against the evaluation set's within-type C-index.
Clinical arm: age, sex, stage, grade, cancer type (check_r_core.clinical_matrix). Pathomic arm: a CSV indexed by
patient (after the pathomic step). Writes <run_dir>/check_r_<arm>.json with the verdicts of analysis/pancan/verdicts.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import blca_clinical_linear_med3pa as med  # noqa: E402  (the paper's MED3pa runner, imported for its functions)

from analysis.pancan.check_r_core import clinical_matrix, oob_risk, profile_cindex  # noqa: E402
from analysis.pancan.metrics import cindex_within, patient_error_within  # noqa: E402
from analysis.pancan.verdicts import r_differs, r_predictable  # noqa: E402
from analysis.profile_metrics import classifier_metrics, discovery_eval_split  # noqa: E402

PANCAN = Path("/data/processed/juan/pancan")
CDR = PANCAN / "TCGA-CDR-SupplementalTableS1.xlsx"
MED3PA_PARAMS = {  # the paper's settings
    "uncertainty_metric": "sigmoidal_error", "ipc_type": "RandomForestRegressor",
    "ipc_params": {"n_estimators": 100}, "apc_params": {"max_depth": 4},
    "ipc_grid_params": {"n_estimators": [50, 100, 200], "max_depth": [2, 4, 6]},
    "apc_grid_params": {"min_samples_leaf": [2, 4, 6]},
    "samples_ratio_min": 0, "samples_ratio_max": 10, "samples_ratio_step": 5, "evaluate_models": True}
TEST_SIZE, REF_SIZE, DISC_FRAC, SPLIT_SEED = 0.3, 0.25, 0.6, 20260908


def clinical_features(patients):
    c = pd.read_csv(PANCAN / "cohort.csv")
    cdr = pd.read_excel(CDR, sheet_name="TCGA-CDR")[["bcr_patient_barcode", "clinical_stage", "histological_grade"]]
    c = c.merge(cdr, left_on="patient", right_on="bcr_patient_barcode", how="left")
    c = c.rename(columns={"stage": "ajcc_pathologic_tumor_stage"})
    return clinical_matrix(c).loc[patients]


def run_quantile(q, X, names, d, err, n_runs, outdir):
    finite = np.isfinite(err)
    thr = float(np.quantile(err[finite], q))
    y = np.zeros(len(err), dtype=int)
    y[finite] = (err[finite] >= thr).astype(int)
    disc, ev = discovery_eval_split(d["patient"].values, frac_discovery=DISC_FRAC, seed=SPLIT_SEED, stratify=y)
    n_test = int(round(TEST_SIZE * disc.sum()))
    results = [med.run_single_med3pa(i, X[disc], y[disc], names, X[ev], TEST_SIZE, REF_SIZE, MED3PA_PARAMS, n_test,
                                     str(outdir), False) for i in range(n_runs)]
    cm = classifier_metrics(y[ev], np.vstack([r[1] for r in results]).mean(0))
    agg, summ = med.aggregate_profiles(results, n_runs)
    t, e, r, ty = (d[k].values for k in ("time", "event", "risk", "type"))
    cohort_c = cindex_within(t[ev], e[ev], r[ev], ty[ev])
    n_done = int(summ["n_runs_completed"].iat[0])
    profiles = []
    if not agg.empty:
        for _, row in agg[agg["runs_appeared"] >= max(1, math.ceil(0.05 * n_done))].head(20).iterrows():
            mask = med.apply_profile_to_X(row["_conditions_median"], X[ev], names)
            if mask.sum() < 2:
                continue
            c, ci = profile_cindex(mask, t[ev], e[ev], r[ev], ty[ev])
            profiles.append({"profile": row["readable_rule"], "runs_appeared": int(row["runs_appeared"]),
                             "run_share": row["runs_appeared"] / max(n_done, 1), "n_eval_patients": int(mask.sum()),
                             "cindex_within": c, "ci": list(ci)})
    auroc, p = cm.get("auroc", float("nan")), cm.get("auroc_p_vs_0.5", 1.0)
    return {"quantile": q, "error_threshold": thr, "n_discovery": int(disc.sum()), "n_eval": int(ev.sum()),
            "classifier_metrics_on_eval": cm, "eval_cindex_within": cohort_c, "n_runs_completed": n_done,
            "profiles": profiles,
            "verdict_predictable": r_predictable(auroc, p) if np.isfinite(auroc) else False,
            "verdict_differs": r_differs(profiles, cohort_c)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--features", default="clinical")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--quantiles", default="0.5,0.7,0.8,0.9")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    d = oob_risk(run)
    err, n_pairs = patient_error_within(d["time"].values, d["event"].values, d["risk"].values, d["type"].values)
    if a.features == "clinical":
        F = clinical_features(d["patient"].values)
        arm = "clinical"
    else:
        F = pd.read_csv(a.features, index_col=0).loc[d["patient"].values]
        arm = Path(a.features).stem
    X, names = F.to_numpy(dtype=float), [str(c) for c in F.columns]
    out = {"run_dir": str(run), "arm": arm, "epoch": d.attrs["epoch"], "n_patients": len(d),
           "n_events": int(d["event"].sum()), "oob_cindex_within": cindex_within(
               d["time"].values, d["event"].values, d["risk"].values, d["type"].values),
           "patients_without_comparable_pair": int((n_pairs == 0).sum()), "features": names, "quantiles": []}
    for q in (float(x) for x in a.quantiles.split(",")):
        res = run_quantile(q, X, names, d, err, a.runs, run / f"med3pa_{arm}_q{int(q * 100)}")
        out["quantiles"].append(res)
        cm = res["classifier_metrics_on_eval"]
        print(f"q{q}: AUROC {cm.get('auroc', float('nan')):.3f} p {cm.get('auroc_p_vs_0.5', float('nan')):.4f} "
              f"predictable={res['verdict_predictable']} differs={res['verdict_differs']} "
              f"({len(res['profiles'])} recurring profiles)", flush=True)
    (run / f"check_r_{arm}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
