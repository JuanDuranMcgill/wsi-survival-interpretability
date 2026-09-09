#!/usr/bin/env python3
# brca_clinical_linear_med3pa.py  — revised for AIM revision
#
# Changes from the MICCAI version (per docs/RUNBOOK.md Step 2):
#  2a. Error file: reads patient_error_brca_oob.npz (OOB, censoring-fixed).
#  2b. Locked discovery/evaluation split: profiles discovered on 60%,
#      all reported numbers evaluated on the other 40%.
#  2c. Metrics: AUROC + balanced accuracy on eval set; per-profile survival c-index.
#  2d. Threshold sensitivity: run with --med3pa-error-quantile 0.5/0.7/0.8/0.9.
#  2e. Inverse-transform thresholds: StandardScaler fitted on X.
#  2f. BRCA backbone: 106 rounds from med3pa_bootstrap_intermediate_BRCA.

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from MED3pa.med3pa import Med3paExperiment
from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager

# Unwrap @checkpoint decorator to avoid ObjectHashError on BaseModelManager.
import inspect as _inspect
_raw_run = getattr(Med3paExperiment.run, '__wrapped__', None)
if _raw_run is not None:
    Med3paExperiment.run = staticmethod(_raw_run)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from analysis.profile_metrics import discovery_eval_split, classifier_metrics, profile_report

warnings.filterwarnings("ignore")


_DATA = os.path.expanduser("~/data")
CLINICAL_CSV = os.path.join(_DATA, "clinical_features_BRCA.csv")
SAVE_ROOT    = os.path.join(_DATA, "med3pa_bootstrap_intermediate_BRCA")
ERROR_NPZ    = os.path.join(_DATA, "patient_error_brca_oob.npz")


def load_npz(path):
    return dict(np.load(path, allow_pickle=True))


def load_and_align_data(clinical_csv, error_npz, save_root):
    clin_df = pd.read_csv(clinical_csv, index_col="patient_id")
    clin_df.index = clin_df.index.astype(str)

    oob = load_npz(error_npz)
    pids_oob   = np.array([str(p)[:12] for p in oob["patient_ids"]])
    risks_oob  = np.asarray(oob["risk"], dtype=np.float64)
    errors_oob = np.asarray(oob["error"], dtype=np.float64)
    pfi_times  = np.asarray(oob["pfi_time"], dtype=np.float64)
    pfi_events = np.asarray(oob["pfi_event"], dtype=np.float64)

    risk_df = pd.DataFrame(
        {"survival_target": risks_oob, "backbone_error": errors_oob,
         "pfi_time": pfi_times, "pfi_event": pfi_events},
        index=pids_oob,
    )
    risk_df.index.name = "patient_id"

    merged = clin_df.join(risk_df, how="inner")

    feature_names = [
        c for c in merged.columns
        if c not in ("survival_target", "backbone_error", "pfi_time", "pfi_event")
    ]
    X             = merged[feature_names].to_numpy(dtype=np.float64)
    y             = merged["survival_target"].to_numpy(dtype=np.float64)
    backbone_error = merged["backbone_error"].to_numpy(dtype=np.float64)
    patient_ids   = merged.index.to_numpy().astype(str)
    pfi_t         = merged["pfi_time"].to_numpy(dtype=np.float64)
    pfi_e         = merged["pfi_event"].to_numpy(dtype=np.float64)

    return X, y, backbone_error, feature_names, patient_ids, merged, pfi_t, pfi_e


def run_feature_importance(X, target, feature_names, n_trees, n_bootstrap, n_jobs,
                           max_depth, min_samples_leaf):
    boot_per_job_base = n_bootstrap // n_jobs
    boot_remainder = n_bootstrap % n_jobs
    boot_counts = [boot_per_job_base + (1 if i < boot_remainder else 0) for i in range(n_jobs)]
    boot_counts = [b for b in boot_counts if b > 0]
    trees_per_bootstrap = max(100, int(np.ceil(n_trees / n_bootstrap)))

    def _fit_bootstrap_batch(args):
        Xb, yb, n_boot, trees, seed, depth, min_leaf = args
        rng = np.random.default_rng(seed)
        n, p = Xb.shape
        imps = np.zeros((n_boot, p), dtype=np.float32)
        oob_scores = np.zeros(n_boot, dtype=np.float32)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            unique_idx = np.unique(idx)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[unique_idx] = False
            model = RandomForestRegressor(
                n_estimators=trees, max_depth=depth, min_samples_leaf=min_leaf,
                max_features="sqrt", random_state=seed + b, n_jobs=1, bootstrap=True,
            )
            model.fit(Xb[idx], yb[idx])
            imps[b] = model.feature_importances_.astype(np.float32)
            if oob_mask.sum() >= 5:
                pred = model.predict(Xb[oob_mask])
                corr = np.corrcoef(pred, yb[oob_mask])[0, 1]
                oob_scores[b] = np.float32(corr) if np.isfinite(corr) else np.nan
            else:
                oob_scores[b] = np.nan
        return imps, oob_scores

    worker_args = [
        (X, target, boot_counts[i], trees_per_bootstrap, 42 + i * 1000, max_depth, min_samples_leaf)
        for i in range(len(boot_counts))
    ]
    results = Parallel(n_jobs=len(boot_counts))(
        delayed(_fit_bootstrap_batch)(wa) for wa in worker_args
    )

    all_imps = np.concatenate([r[0] for r in results], axis=0)
    oob_scores = np.concatenate([r[1] for r in results], axis=0)
    n_boot_total = all_imps.shape[0]

    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0, ddof=1) if n_boot_total > 1 else np.zeros(X.shape[1], dtype=np.float32)
    ci_half = 1.96 * std_imp / np.sqrt(max(n_boot_total, 1))
    ci_lo, ci_hi = mean_imp - ci_half, mean_imp + ci_half
    boot_ranks = np.argsort(np.argsort(-all_imps, axis=1), axis=1) + 1
    median_rank = np.median(boot_ranks, axis=0)
    rank_q025 = np.quantile(boot_ranks, 0.025, axis=0)
    rank_q975 = np.quantile(boot_ranks, 0.975, axis=0)
    rank_order = np.argsort(-mean_imp)
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, len(feature_names) + 1)
    top20_pct = 100.0 * (boot_ranks <= 20).mean(axis=0)
    top50_pct = 100.0 * (boot_ranks <= 50).mean(axis=0)

    rows = []
    for i, feat in enumerate(feature_names):
        x = X[:, i]
        ok = np.isfinite(x) & np.isfinite(target)
        if ok.sum() >= 5 and np.nanstd(x[ok]) > 0 and np.nanstd(target[ok]) > 0:
            rho, pval = stats.spearmanr(x[ok], target[ok])
        else:
            rho, pval = np.nan, np.nan
        direction = (
            "higher feature -> higher survival_target" if np.isfinite(rho) and rho > 0
            else ("higher feature -> lower survival_target" if np.isfinite(rho)
                  else "undetermined")
        )
        rows.append({
            "feature": feat,
            "mean_importance": round(float(mean_imp[i]), 6),
            "std_importance": round(float(std_imp[i]), 6),
            "ci_lo": round(float(ci_lo[i]), 6),
            "ci_hi": round(float(ci_hi[i]), 6),
            "rank": int(ranks[i]),
            "median_rank": round(float(median_rank[i]), 1),
            "rank_q025": round(float(rank_q025[i]), 1),
            "rank_q975": round(float(rank_q975[i]), 1),
            "top20_pct": round(float(top20_pct[i]), 1),
            "top50_pct": round(float(top50_pct[i]), 1),
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "spearman_p_value": round(float(pval), 6) if np.isfinite(pval) else None,
            "direction": direction,
        })

    feat_df = pd.DataFrame(rows).sort_values(["rank", "median_rank", "feature"])
    metrics = {
        "n_bootstrap": int(n_boot_total),
        "mean_oob_corr": float(np.nanmean(oob_scores)) if np.isfinite(oob_scores).any() else None,
        "std_oob_corr": float(np.nanstd(oob_scores)) if np.isfinite(oob_scores).any() else None,
    }
    return feat_df, metrics


def fit_linear_base_model(X, y, patient_ids, alpha, n_splits, seed):
    preds = np.full_like(y, np.nan, dtype=np.float64)
    abs_err = np.full_like(y, np.nan, dtype=np.float64)
    sq_err = np.full_like(y, np.nan, dtype=np.float64)
    split_rows = []
    for split_idx in range(n_splits):
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=seed + split_idx,
        )
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx]).astype(np.float64)
        preds[te_idx] = pred
        abs_err[te_idx] = np.abs(pred - y[te_idx])
        sq_err[te_idx] = (pred - y[te_idx]) ** 2
        split_rows.append({
            "split": split_idx, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "test_r2": float(r2_score(y[te_idx], pred)) if len(np.unique(y[te_idx])) > 1 else None,
            "test_rmse": float(np.sqrt(mean_squared_error(y[te_idx], pred))),
        })
    missing = np.isnan(preds)
    if missing.any():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(X[~missing], y[~missing])
        pred = model.predict(X[missing]).astype(np.float64)
        preds[missing] = pred
        abs_err[missing] = np.abs(pred - y[missing])
        sq_err[missing] = (pred - y[missing]) ** 2

    patient_df = pd.DataFrame({
        "patient_id": patient_ids, "survival_target": y,
        "predicted_survival": preds, "patient_error_abs": abs_err, "patient_error_sq": sq_err,
    }).sort_values("patient_error_abs", ascending=False)
    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame([{
        "alpha": alpha, "n_splits": n_splits,
        "mean_test_r2": split_df["test_r2"].mean(),
        "std_test_r2": split_df["test_r2"].std(ddof=1) if len(split_df) > 1 else None,
        "mean_test_rmse": split_df["test_rmse"].mean(),
        "std_test_rmse": split_df["test_rmse"].std(ddof=1) if len(split_df) > 1 else None,
        "overall_pred_target_corr": np.corrcoef(
            patient_df["predicted_survival"], patient_df["survival_target"])[0, 1]
            if len(patient_df) > 1 else None,
    }])
    return patient_df, split_df, summary_df


def parse_condition_dir(cond_str):
    parts = cond_str.strip().split(" ")
    if len(parts) < 3:
        return None, None
    feature = parts[0]
    op = parts[1]
    try:
        threshold = float(parts[2])
    except ValueError:
        return None, None
    direction = "UP" if op in (">", ">=") else "DOWN"
    return f"{feature}_{direction}", threshold


def canonicalize_path_dir(path):
    if path == ["*"]:
        return None, {}
    conds, thresholds = [], {}
    for c in path:
        if c == "*":
            continue
        cond_key, thresh = parse_condition_dir(c)
        if cond_key is None:
            continue
        conds.append(cond_key)
        thresholds[cond_key] = thresh
    if not conds:
        return None, {}
    return " | ".join(sorted(conds)), thresholds


def extract_profiles_from_run(profiles_path, n_test_approx, min_patients=1):
    if not os.path.exists(profiles_path):
        return []
    with open(profiles_path) as f:
        profiles = json.load(f)
    try:
        nodes = profiles["0"]["100"]
    except KeyError:
        return []
    results = []
    for node in nodes:
        acc = node["metrics"].get("Accuracy")
        if acc is None:
            continue
        node_info = node.get("node information", {})
        pop_abs = node_info.get("Population", None)
        pop_pct = node_info.get("Population%", None)
        if pop_abs is not None and pop_abs > 1:
            n_patients = int(pop_abs)
        elif pop_pct is not None:
            n_patients = int(round(pop_pct * n_test_approx / 100.0))
        else:
            n_patients = 0
        if n_patients < min_patients:
            continue
        path = node["path"]
        key, thresholds = canonicalize_path_dir(path)
        if key is None:
            continue
        results.append({
            "path_key": key, "path_raw": path, "thresholds": thresholds,
            "n_patients": n_patients, "pop_pct": pop_pct if pop_pct is not None else 0,
            "accuracy": acc, "n_conditions": len([c for c in path if c != "*"]),
        })
    return results


def apply_profile_to_X(conditions_median, X_eval, feature_names):
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    mask = np.ones(len(X_eval), dtype=bool)
    for cond_key, thresh in conditions_median.items():
        parts = cond_key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        feature, direction = parts
        if feature not in feat_idx:
            continue
        col = X_eval[:, feat_idx[feature]]
        if direction == "UP":
            mask &= col > thresh
        else:
            mask &= col <= thresh
    return mask


def run_single_med3pa(run_idx, X_disc, y_disc, feature_names, X_eval,
                       test_size, ref_size, med3pa_params, n_test_approx,
                       outdir, save_runs):
    seed = 1000 + run_idx
    x_df = pd.DataFrame(X_disc, columns=feature_names)
    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x_df, y_disc, test_size=test_size, random_state=seed, stratify=y_disc
    )
    x_train, x_ref, y_train, y_ref = train_test_split(
        x_train_pool, y_train_pool, test_size=ref_size, random_state=seed,
        stratify=y_train_pool,
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=seed,
        class_weight="balanced", n_jobs=1,
    )
    clf.fit(x_train, y_train)

    x_eval_df = pd.DataFrame(X_eval, columns=feature_names)
    eval_scores = clf.predict_proba(x_eval_df)[:, 1]

    datasets = DatasetsManager()
    datasets.set_from_data("reference", x_ref.to_numpy(), y_ref, list(x_ref.columns))
    datasets.set_from_data("testing",   x_test.to_numpy(), y_test, list(x_test.columns))
    base_model_manager = BaseModelManager(model=clf)

    outdir_run = os.path.join(outdir, "runs", f"run_{run_idx:04d}")
    os.makedirs(outdir_run, exist_ok=True)

    results = Med3paExperiment.run(
        datasets_manager=datasets,
        base_model_manager=base_model_manager,
        **med3pa_params,
    )
    results.save(file_path=outdir_run)

    profiles_path = os.path.join(outdir_run, "reference", "profiles.json")
    profiles = extract_profiles_from_run(profiles_path, n_test_approx, min_patients=1)

    if not save_runs:
        import shutil
        shutil.rmtree(outdir_run, ignore_errors=True)

    return profiles, eval_scores


def aggregate_profiles(all_run_profiles, n_runs_requested):
    profile_records = defaultdict(list)
    n_completed = 0
    for run_profiles, _ in all_run_profiles:
        if run_profiles is not None:
            n_completed += 1
            for p in run_profiles:
                profile_records[p["path_key"]].append({
                    "n_patients": p["n_patients"], "pop_pct": p["pop_pct"],
                    "accuracy": p["accuracy"], "n_conditions": p["n_conditions"],
                    "thresholds": p["thresholds"], "path_raw": p["path_raw"],
                })

    rows = []
    for key, records in profile_records.items():
        n_runs_appeared = len(records)
        avg_n_patients = np.mean([r["n_patients"] for r in records])
        avg_pop_pct    = np.mean([r["pop_pct"]    for r in records])
        avg_accuracy   = np.mean([r["accuracy"]   for r in records])
        std_accuracy   = np.std([r["accuracy"]    for r in records])
        n_conditions   = records[0]["n_conditions"]

        all_thresholds = defaultdict(list)
        for r in records:
            for cond_key, thresh_val in r["thresholds"].items():
                all_thresholds[cond_key].append(thresh_val)

        threshold_summary = {}
        conditions_median = {}
        for cond_key, vals in sorted(all_thresholds.items()):
            med = float(np.median(vals))
            threshold_summary[cond_key] = {
                "median": round(med, 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
            conditions_median[cond_key] = med

        readable_parts = []
        for cond_key in sorted(all_thresholds.keys()):
            parts = cond_key.rsplit("_", 1)
            feature = parts[0]
            direction = parts[1]
            op = ">" if direction == "UP" else "<="
            median_t = threshold_summary[cond_key]["median"]
            range_str = f"[{threshold_summary[cond_key]['min']}, {threshold_summary[cond_key]['max']}]"
            readable_parts.append(f"{feature} {op} {median_t} {range_str}")

        rows.append({
            "profile": key,
            "readable_rule": " AND ".join(readable_parts),
            "n_conditions": n_conditions,
            "runs_appeared": n_runs_appeared,
            "run_freq_pct": round(100 * n_runs_appeared / max(n_completed, 1), 1),
            "avg_n_patients": round(avg_n_patients, 1),
            "avg_pop_pct": round(avg_pop_pct, 2),
            "avg_accuracy": round(avg_accuracy, 4),
            "std_accuracy": round(std_accuracy, 4),
            "_conditions_median": conditions_median,
        })

    agg_df = (
        pd.DataFrame(rows).sort_values("runs_appeared", ascending=False)
        if rows else pd.DataFrame()
    )
    summary = pd.DataFrame([{
        "n_runs_requested": n_runs_requested,
        "n_runs_completed": n_completed,
        "n_unique_profiles": len(profile_records),
    }])
    return agg_df, summary


def make_scaler_params(X, feature_names):
    sc = StandardScaler()
    sc.fit(X)
    return {f: {"mean": float(sc.mean_[i]), "scale": float(sc.scale_[i])}
            for i, f in enumerate(feature_names)}


def inverse_transform_threshold(thresh_zscore, feature, scaler_params):
    if feature not in scaler_params:
        return None
    p = scaler_params[feature]
    return float(thresh_zscore * p["scale"] + p["mean"])


def main():
    ap = argparse.ArgumentParser(
        description="BRCA clinical-only pipeline: RF importance + MED3PA reliability"
    )
    ap.add_argument("--clinical-csv", type=str, default=CLINICAL_CSV)
    ap.add_argument("--error-npz", type=str, default=ERROR_NPZ)
    ap.add_argument("--save-root", type=str, default=SAVE_ROOT)
    ap.add_argument("--round-count", type=int, default=106,
                    help="Number of bootstrap rounds BRCA backbone ran (default: 106).")
    ap.add_argument("--outdir", type=str, default="results/reliability_brca_clinical")

    ap.add_argument("--fi-n-trees", type=int, default=200000)
    ap.add_argument("--fi-n-bootstrap", type=int, default=200)
    ap.add_argument("--fi-n-jobs", type=int, default=8)
    ap.add_argument("--fi-max-depth", type=int, default=6)
    ap.add_argument("--fi-min-samples-leaf", type=int, default=5)

    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--cv-splits", type=int, default=5)

    ap.add_argument("--med3pa-n-runs", type=int, default=200)
    ap.add_argument("--med3pa-n-parallel", type=int, default=10)
    ap.add_argument("--med3pa-test-size", type=float, default=0.3)
    ap.add_argument("--med3pa-ref-size", type=float, default=0.25)
    ap.add_argument("--med3pa-error-quantile", type=float, default=0.5)
    ap.add_argument("--discovery-frac", type=float, default=0.6)
    ap.add_argument("--split-seed", type=int, default=20260908)
    ap.add_argument("--save-runs", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X, y, backbone_error, feature_names, patient_ids, merged, pfi_t, pfi_e = \
        load_and_align_data(args.clinical_csv, args.error_npz, args.save_root)
    print(f"[brca_clinical] {len(patient_ids)} patients aligned")

    feat_df, fi_metrics = run_feature_importance(
        X, y, feature_names, args.fi_n_trees, args.fi_n_bootstrap,
        args.fi_n_jobs, args.fi_max_depth, args.fi_min_samples_leaf,
    )
    feat_df.to_csv(os.path.join(args.outdir, "clinical_importance_survival.csv"), index=False)

    patient_pred_df, split_df, summary_df = fit_linear_base_model(
        X, y, patient_ids, args.ridge_alpha, args.cv_splits, 123,
    )
    patient_pred_df["backbone_error"] = backbone_error
    split_df.to_csv(os.path.join(args.outdir, "clinical_linear_model_split_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "clinical_linear_model_summary.csv"), index=False)
    patient_pred_df.to_csv(os.path.join(args.outdir, "clinical_linear_model_patient_predictions.csv"), index=False)

    scaler_params = make_scaler_params(X, feature_names)

    finite = np.isfinite(backbone_error)
    thresh = float(np.quantile(backbone_error[finite], args.med3pa_error_quantile))
    y_med3pa = np.zeros(len(backbone_error), dtype=int)
    y_med3pa[finite] = (backbone_error[finite] >= thresh).astype(int)
    prevalence_overall = float(y_med3pa.mean())
    print(f"[brca_clinical] q{args.med3pa_error_quantile:.2f} threshold={thresh:.4f}, "
          f"prevalence={prevalence_overall:.3f}")

    disc_mask, eval_mask = discovery_eval_split(
        patient_ids, frac_discovery=args.discovery_frac,
        seed=args.split_seed, stratify=y_med3pa,
    )
    X_disc, y_disc = X[disc_mask], y_med3pa[disc_mask]
    X_eval, y_eval = X[eval_mask], y_med3pa[eval_mask]
    err_eval = backbone_error[eval_mask]
    pfi_t_eval = pfi_t[eval_mask]
    pfi_e_eval = pfi_e[eval_mask]
    risks_eval = y[eval_mask]
    print(f"[brca_clinical] discovery N={disc_mask.sum()}, eval N={eval_mask.sum()}")

    med3pa_params = {
        "uncertainty_metric": "sigmoidal_error",
        "ipc_type": "RandomForestRegressor",
        "ipc_params": {"n_estimators": 100},
        "apc_params": {"max_depth": 4},
        "ipc_grid_params": {"n_estimators": [50, 100, 200], "max_depth": [2, 4, 6]},
        "apc_grid_params": {"min_samples_leaf": [2, 4, 6]},
        "samples_ratio_min": 0, "samples_ratio_max": 10, "samples_ratio_step": 5,
        "evaluate_models": True,
    }
    n_test_approx = int(round(args.med3pa_test_size * len(X_disc)))
    # Sequential: Med3paExperiment.run uses Ray internally; Loky subprocess
    # nesting causes ObjectHashError pickle failures.
    all_results = []
    print(f">>> Running {args.med3pa_n_runs} MED3PA iterations (sequential)...")
    for i in range(args.med3pa_n_runs):
        result = run_single_med3pa(
            run_idx=i, X_disc=X_disc, y_disc=y_disc, feature_names=feature_names,
            X_eval=X_eval, test_size=args.med3pa_test_size, ref_size=args.med3pa_ref_size,
            med3pa_params=med3pa_params, n_test_approx=n_test_approx,
            outdir=args.outdir, save_runs=args.save_runs,
        )
        all_results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{args.med3pa_n_runs}] runs done")

    all_eval_scores = np.vstack([r[1] for r in all_results if r is not None])
    mean_eval_scores = all_eval_scores.mean(axis=0)
    cm = classifier_metrics(y_eval, mean_eval_scores)
    print(f"[brca_clinical] eval AUROC={cm.get('auroc', 'N/A'):.3f}, "
          f"p_vs_0.5={cm.get('auroc_p_vs_0.5', 'N/A'):.4f}, "
          f"balanced_acc={cm.get('balanced_accuracy', 'N/A'):.3f}")

    agg_df, med3pa_summary = aggregate_profiles(all_results, args.med3pa_n_runs)

    profile_reports = []
    if not agg_df.empty:
        agg_df.to_csv(os.path.join(args.outdir, "all_profiles_dir.csv"), index=False)
        for min_freq in [10, 25, 50]:
            agg_df[agg_df["runs_appeared"] >= min_freq].to_csv(
                os.path.join(args.outdir, f"profiles_freq{min_freq}.csv"), index=False)

        top_profiles = agg_df[agg_df["runs_appeared"] >= 10].head(20)
        for _, row in top_profiles.iterrows():
            conds = row["_conditions_median"]
            mask_eval = apply_profile_to_X(conds, X_eval, feature_names)
            pr = profile_report(mask_eval, y_eval, err_eval, pfi_t_eval, pfi_e_eval, risks_eval)

            readable_clinical = {}
            for cond_key, thresh_z in conds.items():
                parts = cond_key.rsplit("_", 1)
                feature = parts[0] if len(parts) == 2 else cond_key
                direction = parts[1] if len(parts) == 2 else "?"
                op = ">" if direction == "UP" else "<="
                t_clinical = inverse_transform_threshold(thresh_z, feature, scaler_params)
                readable_clinical[cond_key] = {
                    "feature": feature, "op": op,
                    "threshold_in_csv_units": round(thresh_z, 4),
                    "threshold_clinical_approx": round(t_clinical, 4) if t_clinical is not None else None,
                }

            profile_reports.append({
                "profile": row["profile"],
                "readable_rule": row["readable_rule"],
                "runs_appeared": int(row["runs_appeared"]),
                "run_freq_pct": float(row["run_freq_pct"]),
                "discovery_avg_accuracy": float(row["avg_accuracy"]),
                "thresholds": readable_clinical,
                **pr,
            })

    med3pa_summary["error_quantile"] = args.med3pa_error_quantile
    med3pa_summary.to_csv(os.path.join(args.outdir, "med3pa_run_summary.csv"), index=False)

    out = {
        "cohort": "brca",
        "arm": "clinical",
        "endpoint": "PFI",
        "input_error_npz": os.path.abspath(args.error_npz),
        "input_clinical_csv": os.path.abspath(args.clinical_csv),
        "backbone_round_count": args.round_count,
        "backbone_save_root": args.save_root,
        "n_patients_aligned": int(len(patient_ids)),
        "n_events": int((pfi_e == 1).sum()),
        "n_censored": int((pfi_e == 0).sum()),
        "med3pa_error_quantile": args.med3pa_error_quantile,
        "error_threshold": float(thresh),
        "label_prevalence_overall": prevalence_overall,
        "majority_class_accuracy_overall": float(max(prevalence_overall, 1 - prevalence_overall)),
        "discovery_eval_split": {
            "frac_discovery": args.discovery_frac,
            "seed": args.split_seed,
            "n_discovery": int(disc_mask.sum()),
            "n_eval": int(eval_mask.sum()),
        },
        "med3pa_n_runs": args.med3pa_n_runs,
        "classifier_metrics_on_eval": cm,
        "scaler_params_note": (
            "StandardScaler fitted on aligned X. If the input CSV is already "
            "z-scored, threshold_clinical_approx ≈ threshold_in_csv_units."
        ),
        "top_profiles_eval": profile_reports,
        "feature_importance_summary": fi_metrics,
    }
    out_json = os.path.join(args.outdir, "reliability_brca_clinical.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[brca_clinical] wrote {out_json}")
    print(f"[brca_clinical] saved outputs to {args.outdir}")


if __name__ == "__main__":
    main()
