#!/usr/bin/env python3
# blca_radiomic_linear_med3pa.py  — revised for AIM revision
#
# Changes from the MICCAI version (per docs/RUNBOOK.md Step 2):
#  2a. Error file: reads patient_error_blca_oob.npz (OOB, censoring-fixed).
#  2b. Locked discovery/evaluation split on top-K radiomic subset.
#  2c. Metrics: AUROC + balanced accuracy on eval set; per-profile survival c-index.
#  2d. Threshold sensitivity: run with --med3pa-error-quantile 0.5/0.7/0.8/0.9.
#  2e. Inverse-transform thresholds: StandardScaler fitted on X_med3pa.
#  2f. BLCA backbone: default SAVE_ROOT is 102-round med3pa_bootstrap_intermediate.

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed
from scipy import stats

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MED3pa.med3pa import Med3paExperiment
from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))
from analysis.profile_metrics import discovery_eval_split, classifier_metrics, profile_report

warnings.filterwarnings("ignore")


_DATA = os.path.expanduser("~/data")
RADIOMICS_CSV = os.path.join(_DATA, "blca", "radiomics_pre_corr.csv")
SAVE_ROOT     = os.path.join(_DATA, "med3pa_bootstrap_intermediate")   # 102-round (2f)
ERROR_NPZ     = os.path.join(_DATA, "patient_error_blca_oob.npz")


KNOWN_REGIONS = [
    "urothelium", "stroma", "tumor", "necrosis",
    "vessels", "adipose", "muscularis", "blood_vessels",
    "immune", "lamina",
]


def parse_feature_components(feat_name):
    result = {
        "region": "other", "channel": "unknown",
        "feature_class": "unknown", "metric": feat_name, "stat": "unknown",
    }
    remainder = feat_name
    for r in sorted(KNOWN_REGIONS, key=len, reverse=True):
        if remainder.startswith(r + "_"):
            result["region"] = r
            remainder = remainder[len(r) + 1:]
            break
    for ch in ["H", "E"]:
        if remainder.startswith(ch + "_"):
            result["channel"] = ch
            remainder = remainder[len(ch) + 1:]
            break
    if remainder.startswith("original_"):
        remainder = remainder[len("original_"):]
    if remainder.startswith("firstorder_"):
        result["feature_class"] = "firstorder"
        remainder = remainder[len("firstorder_"):]
    elif remainder.startswith("glcm_"):
        result["feature_class"] = "glcm"
        remainder = remainder[len("glcm_"):]
    if remainder.endswith("_mean"):
        result["stat"] = "mean"
        result["metric"] = remainder[:-5]
    elif remainder.endswith("_std"):
        result["stat"] = "std"
        result["metric"] = remainder[:-4]
    else:
        result["metric"] = remainder
    return result


def load_and_align_data(radiomics_csv, error_npz):
    rad_df = pd.read_csv(radiomics_csv, index_col="patient_id")
    rad_df.index = rad_df.index.astype(str)

    oob = dict(np.load(error_npz, allow_pickle=True))
    pids_oob   = np.array([str(p)[:12] for p in oob["patient_ids"]])
    risks_oob  = np.asarray(oob["risk"], dtype=np.float64)
    errors_oob = np.asarray(oob["error"], dtype=np.float64)
    pfi_times  = np.asarray(oob["pfi_time"], dtype=np.float64)
    pfi_events = np.asarray(oob["pfi_event"], dtype=np.float64)

    oob_df = pd.DataFrame(
        {"survival_target": risks_oob, "backbone_error": errors_oob,
         "pfi_time": pfi_times, "pfi_event": pfi_events},
        index=pids_oob,
    )
    oob_df.index.name = "patient_id"

    merged = rad_df.join(oob_df, how="inner")

    feature_names = [
        c for c in merged.columns
        if c not in ("survival_target", "backbone_error", "pfi_time", "pfi_event")
    ]
    X              = merged[feature_names].to_numpy(dtype=np.float32)
    survival_target = merged["survival_target"].to_numpy(dtype=np.float64)
    backbone_error  = merged["backbone_error"].to_numpy(dtype=np.float64)
    patient_ids    = merged.index.to_numpy().astype(str)
    pfi_t          = merged["pfi_time"].to_numpy(dtype=np.float64)
    pfi_e          = merged["pfi_event"].to_numpy(dtype=np.float64)

    return X, survival_target, backbone_error, feature_names, patient_ids, pfi_t, pfi_e


def _fit_bootstrap_batch(args):
    X, y, n_boot, n_trees, seed, max_depth, min_leaf = args
    rng = np.random.default_rng(seed)
    n, p = X.shape
    imps = np.zeros((n_boot, p), dtype=np.float32)
    oob_scores = np.zeros(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        unique_idx = np.unique(idx)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[unique_idx] = False
        model = RandomForestRegressor(
            n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_leaf,
            max_features="sqrt", random_state=seed + b, n_jobs=1, bootstrap=True,
        )
        model.fit(X[idx], y[idx])
        imps[b] = model.feature_importances_.astype(np.float32)
        if oob_mask.sum() >= 5:
            pred = model.predict(X[oob_mask])
            corr = np.corrcoef(pred, y[oob_mask])[0, 1]
            oob_scores[b] = np.float32(corr) if np.isfinite(corr) else np.nan
        else:
            oob_scores[b] = np.nan
    return imps, oob_scores


def run_radiomic_importance_firstorder(X, survival_target, feature_names, n_trees,
                                        n_bootstrap, n_jobs, max_depth, min_samples_leaf,
                                        outdir, top_n_for_med3pa):
    os.makedirs(outdir, exist_ok=True)
    fo_mask = np.array(["firstorder" in f for f in feature_names])
    X_fo = X[:, fo_mask]
    fo_feature_names = [f for f, m in zip(feature_names, fo_mask) if m]

    print(f">>> BLCA radiomics (first-order-only): {len(fo_feature_names)} / {len(feature_names)} features")
    n_patients, n_features = X_fo.shape
    print(f"    Patients: {n_patients} | First-order features: {n_features}")

    boot_per_job_base = n_bootstrap // n_jobs
    boot_remainder = n_bootstrap % n_jobs
    boot_counts = [boot_per_job_base + (1 if i < boot_remainder else 0) for i in range(n_jobs)]
    boot_counts = [b for b in boot_counts if b > 0]
    trees_per_bootstrap = max(100, int(np.ceil(n_trees / n_bootstrap)))

    worker_args = [
        (X_fo, survival_target, boot_counts[i], trees_per_bootstrap, 42 + i * 1000, max_depth, min_samples_leaf)
        for i in range(len(boot_counts))
    ]
    print(f">>> Fitting {n_bootstrap} bootstrap forests ({n_jobs} workers)...")
    with mp.Pool(processes=len(boot_counts)) as pool:
        results = pool.map(_fit_bootstrap_batch, worker_args)

    all_imps = np.concatenate([r[0] for r in results], axis=0)
    oob_scores = np.concatenate([r[1] for r in results], axis=0)
    n_boot_total = all_imps.shape[0]
    print(f"    Bootstrap fits: {n_boot_total}, mean OOB corr: {np.nanmean(oob_scores):.4f}")

    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0, ddof=1) if n_boot_total > 1 else np.zeros(n_features, dtype=np.float32)
    ci_half = 1.96 * std_imp / np.sqrt(max(n_boot_total, 1))
    ci_lo, ci_hi = mean_imp - ci_half, mean_imp + ci_half
    boot_ranks = np.argsort(np.argsort(-all_imps, axis=1), axis=1) + 1
    median_rank = np.median(boot_ranks, axis=0)
    rank_q025 = np.quantile(boot_ranks, 0.025, axis=0)
    rank_q975 = np.quantile(boot_ranks, 0.975, axis=0)
    rank_order = np.argsort(-mean_imp)
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, n_features + 1)
    top20_pct = 100.0 * (boot_ranks <= 20).mean(axis=0)
    top50_pct = 100.0 * (boot_ranks <= 50).mean(axis=0)

    corr_rows = []
    for i in range(n_features):
        x = X_fo[:, i]
        ok = np.isfinite(x) & np.isfinite(survival_target)
        if ok.sum() >= 5 and np.nanstd(x[ok]) > 0 and np.nanstd(survival_target[ok]) > 0:
            rho, pval = stats.spearmanr(x[ok], survival_target[ok])
        else:
            rho, pval = np.nan, np.nan
        corr_rows.append((rho, pval))

    rows = []
    for i, feat in enumerate(fo_feature_names):
        comp = parse_feature_components(feat)
        rho, pval = corr_rows[i]
        direction = (
            "higher feature -> higher survival_target" if np.isfinite(rho) and rho > 0
            else ("higher feature -> lower survival_target" if np.isfinite(rho) else "undetermined")
        )
        rows.append({
            "feature": feat, "region": comp["region"], "channel": comp["channel"],
            "feature_class": comp["feature_class"], "metric": comp["metric"], "stat": comp["stat"],
            "mean_importance": round(float(mean_imp[i]), 6),
            "std_importance": round(float(std_imp[i]), 6),
            "ci_lo": round(float(ci_lo[i]), 6), "ci_hi": round(float(ci_hi[i]), 6),
            "cv": round(float(std_imp[i] / mean_imp[i]), 3) if mean_imp[i] > 0 else float("nan"),
            "rank": int(ranks[i]), "median_rank": round(float(median_rank[i]), 1),
            "rank_q025": round(float(rank_q025[i]), 1), "rank_q975": round(float(rank_q975[i]), 1),
            "top20_pct": round(float(top20_pct[i]), 1), "top50_pct": round(float(top50_pct[i]), 1),
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "spearman_p_value": round(float(pval), 6) if np.isfinite(pval) else None,
            "direction": direction,
        })
    feat_df = pd.DataFrame(rows).sort_values(["rank", "median_rank", "feature"])

    region_rows = []
    for region in sorted(feat_df["region"].unique()):
        sub = feat_df[feat_df["region"] == region].sort_values("rank")
        region_rows.append({
            "region": region, "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "best_rank": int(sub["rank"].min()),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
            "features_in_top50": int((sub["rank"] <= 50).sum()),
            "top_feature": sub.iloc[0]["feature"],
            "top_feature_importance": sub.iloc[0]["mean_importance"],
        })
    region_df = pd.DataFrame(region_rows).sort_values("sum_mean_importance", ascending=False)

    class_rows = []
    for fc in sorted(feat_df["feature_class"].unique()):
        sub = feat_df[feat_df["feature_class"] == fc]
        class_rows.append({
            "feature_class": fc, "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    class_df = pd.DataFrame(class_rows).sort_values("sum_mean_importance", ascending=False)

    channel_rows = []
    for ch in sorted(feat_df["channel"].unique()):
        sub = feat_df[feat_df["channel"] == ch]
        channel_rows.append({
            "channel": ch, "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    channel_df = pd.DataFrame(channel_rows).sort_values("sum_mean_importance", ascending=False)

    stat_rows = []
    for st in sorted(feat_df["stat"].unique()):
        sub = feat_df[feat_df["stat"] == st]
        stat_rows.append({
            "stat": st, "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    stat_df = pd.DataFrame(stat_rows).sort_values("sum_mean_importance", ascending=False)

    top_n_comp = min(30, n_features)
    top_indices_comp = np.argsort(-mean_imp)[:top_n_comp]
    comp_rows = []
    for idx in top_indices_comp:
        feat = fo_feature_names[idx]
        vals = X_fo[:, idx]
        ok = np.isfinite(vals) & np.isfinite(survival_target)
        vals_ok, tgt_ok = vals[ok], survival_target[ok]
        comp = parse_feature_components(feat)
        if len(vals_ok) >= 5 and np.nanstd(vals_ok) > 0 and np.nanstd(tgt_ok) > 0:
            slope, _ = np.polyfit(vals_ok, tgt_ok, 1), None
            slope = np.polyfit(vals_ok, tgt_ok, 1)[0]
            rho, pval = stats.spearmanr(vals_ok, tgt_ok)
        else:
            slope, rho, pval = np.nan, np.nan, np.nan
        comp_rows.append({
            "rank": int(ranks[idx]), "feature": feat, "region": comp["region"],
            "feature_mean": round(float(np.nanmean(vals_ok)), 4) if len(vals_ok) else None,
            "feature_std": round(float(np.nanstd(vals_ok)), 4) if len(vals_ok) else None,
            "survival_target_mean": round(float(np.nanmean(tgt_ok)), 4) if len(tgt_ok) else None,
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "p_value": round(float(pval), 6) if np.isfinite(pval) else None,
            "slope_linear_fit": round(float(slope), 6) if np.isfinite(slope) else None,
            "direction": "higher feature -> higher survival_target" if np.isfinite(rho) and rho > 0
                         else ("higher feature -> lower survival_target" if np.isfinite(rho) else "undetermined"),
        })
    comp_df = pd.DataFrame(comp_rows)

    feat_df.to_csv(os.path.join(outdir, "radiomic_importance.csv"), index=False)
    region_df.to_csv(os.path.join(outdir, "region_summary.csv"), index=False)
    class_df.to_csv(os.path.join(outdir, "class_summary.csv"), index=False)
    channel_df.to_csv(os.path.join(outdir, "channel_summary.csv"), index=False)
    stat_df.to_csv(os.path.join(outdir, "stat_summary.csv"), index=False)
    comp_df.to_csv(os.path.join(outdir, "radiomic_comparison_table.csv"), index=False)
    print(f"    Saved radiomic importance outputs to {outdir}")

    k = min(top_n_for_med3pa, n_features)
    top_indices_for_med3pa = np.argsort(-mean_imp)[:k]
    selected_feature_names = [fo_feature_names[i] for i in top_indices_for_med3pa]
    print(f"    Selected top {k} first-order features for MED3PA.")
    fi_metrics = {
        "n_bootstrap": int(n_boot_total),
        "mean_oob_corr": float(np.nanmean(oob_scores)) if np.isfinite(oob_scores).any() else None,
        "std_oob_corr": float(np.nanstd(oob_scores)) if np.isfinite(oob_scores).any() else None,
        "n_first_order_features": int(n_features),
        "n_selected_for_med3pa": int(k),
    }
    return X_fo, survival_target, fo_feature_names, top_indices_for_med3pa, selected_feature_names, fi_metrics


def parse_condition_dir(cond_str):
    parts = cond_str.strip().split(" ")
    if len(parts) < 3:
        return None, None
    feature, op = parts[0], parts[1]
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
        mask &= col > thresh if direction == "UP" else col <= thresh
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
    datasets.set_from_data(dataset_type="reference", observations=x_ref.to_numpy(),
                           true_labels=y_ref, column_labels=list(x_ref.columns))
    datasets.set_from_data(dataset_type="testing",   observations=x_test.to_numpy(),
                           true_labels=y_test, column_labels=list(x_test.columns))
    base_model_manager = BaseModelManager(model=clf)

    outdir_run = os.path.join(outdir, "runs", f"run_{run_idx:04d}")
    os.makedirs(outdir_run, exist_ok=True)
    results = Med3paExperiment.run(
        datasets_manager=datasets, base_model_manager=base_model_manager, **med3pa_params,
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
            feature, direction = parts[0], parts[1]
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


def inverse_transform_threshold(thresh_z, feature, scaler_params):
    if feature not in scaler_params:
        return None
    p = scaler_params[feature]
    return float(thresh_z * p["scale"] + p["mean"])


def main():
    ap = argparse.ArgumentParser(
        description="BLCA radiomic pipeline: first-order importance + MED3PA reliability"
    )
    ap.add_argument("--radiomics-csv", type=str, default=RADIOMICS_CSV)
    ap.add_argument("--error-npz", type=str, default=ERROR_NPZ)
    ap.add_argument("--save-root", type=str, default=SAVE_ROOT)
    ap.add_argument("--round-count", type=int, default=102)
    ap.add_argument("--outdir", type=str, default="results/reliability_blca_radiomic")

    ap.add_argument("--n-trees", type=int, default=10_000_000)
    ap.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--top-n-med3pa", type=int, default=30)

    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--n-parallel", type=int, default=10)
    ap.add_argument("--med3pa-error-quantile", type=float, default=0.5)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--ref-size", type=float, default=0.25)
    ap.add_argument("--discovery-frac", type=float, default=0.6)
    ap.add_argument("--split-seed", type=int, default=20260908)
    ap.add_argument("--save-runs", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(">>> Loading BLCA radiomics + OOB backbone...")
    X, survival_target, backbone_error, feature_names, patient_ids, pfi_t, pfi_e = \
        load_and_align_data(args.radiomics_csv, args.error_npz)
    print(f"    Patients: {X.shape[0]} | Features: {X.shape[1]}")

    # First-order slice for full cohort — needed to build eval features after split
    fo_mask = np.array(["firstorder" in f for f in feature_names])
    X_fo_all = X[:, fo_mask]

    # Label threshold defined on full cohort (consistent with reported metrics)
    finite = np.isfinite(backbone_error)
    thresh = float(np.quantile(backbone_error[finite], args.med3pa_error_quantile))
    y_med3pa = np.zeros(len(backbone_error), dtype=int)
    y_med3pa[finite] = (backbone_error[finite] >= thresh).astype(int)
    prevalence_overall = float(y_med3pa.mean())
    print(f"    q{args.med3pa_error_quantile:.2f} threshold={thresh:.4f}, "
          f"prevalence={prevalence_overall:.3f}")

    # Split BEFORE feature importance — top-K selection must not see eval patients
    disc_mask, eval_mask = discovery_eval_split(
        patient_ids, frac_discovery=args.discovery_frac,
        seed=args.split_seed, stratify=y_med3pa,
    )
    print(f">>> discovery N={disc_mask.sum()}, eval N={eval_mask.sum()}")

    # Feature importance and top-K selection on DISCOVERY patients only (no eval leak)
    X_fo_disc, _, fo_feature_names, top_idx_med3pa, selected_feature_names, fi_metrics = \
        run_radiomic_importance_firstorder(
            X=X[disc_mask], survival_target=survival_target[disc_mask],
            feature_names=feature_names,
            n_trees=args.n_trees, n_bootstrap=args.n_bootstrap, n_jobs=args.n_jobs,
            max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            outdir=args.outdir, top_n_for_med3pa=args.top_n_med3pa,
        )

    # Apply the same top-K column indices to both splits
    X_disc, y_disc = X_fo_disc[:, top_idx_med3pa], y_med3pa[disc_mask]
    X_eval, y_eval = X_fo_all[eval_mask][:, top_idx_med3pa], y_med3pa[eval_mask]
    err_eval = backbone_error[eval_mask]
    pfi_t_eval = pfi_t[eval_mask]
    pfi_e_eval = pfi_e[eval_mask]
    risks_eval = survival_target[eval_mask]
    print(f"    MED3PA uses {X_disc.shape[1]} first-order features (top-K from discovery only).")

    # Save label table
    label_df = pd.DataFrame({
        "patient_id": patient_ids, "backbone_error": backbone_error, "high_error_label": y_med3pa,
    }).sort_values("backbone_error", ascending=False)
    label_df.to_csv(os.path.join(args.outdir, "med3pa_backbone_error_labels_radiomic.csv"), index=False)

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
    n_test_approx = int(round(args.test_size * len(X_disc)))
    print(f">>> Running {args.n_runs} MED3PA iterations ({args.n_parallel} parallel workers)...")

    all_results = Parallel(n_jobs=args.n_parallel, verbose=5)(
        delayed(run_single_med3pa)(
            run_idx=i, X_disc=X_disc, y_disc=y_disc,
            feature_names=selected_feature_names, X_eval=X_eval,
            test_size=args.test_size, ref_size=args.ref_size,
            med3pa_params=med3pa_params, n_test_approx=n_test_approx,
            outdir=args.outdir, save_runs=args.save_runs,
        )
        for i in range(args.n_runs)
    )

    all_eval_scores = np.vstack([r[1] for r in all_results if r is not None])
    mean_eval_scores = all_eval_scores.mean(axis=0)
    cm = classifier_metrics(y_eval, mean_eval_scores)
    print(f">>> eval AUROC={cm.get('auroc', 'N/A'):.3f}, "
          f"p_vs_0.5={cm.get('auroc_p_vs_0.5', 'N/A'):.4f}, "
          f"balanced_acc={cm.get('balanced_accuracy', 'N/A'):.3f}")

    agg_df, med3pa_summary = aggregate_profiles(all_results, args.n_runs)

    scaler_params = make_scaler_params(X_disc, selected_feature_names)

    profile_reports = []
    if not agg_df.empty:
        agg_df.to_csv(os.path.join(args.outdir, "all_profiles_dir_radiomic.csv"), index=False)
        for min_freq in [10, 25, 50]:
            agg_df[agg_df["runs_appeared"] >= min_freq].to_csv(
                os.path.join(args.outdir, f"profiles_freq{min_freq}_radiomic.csv"), index=False)

        top_profiles = agg_df[agg_df["runs_appeared"] >= 10].head(20)
        for _, row in top_profiles.iterrows():
            conds = row["_conditions_median"]
            mask_eval = apply_profile_to_X(conds, X_eval, selected_feature_names)
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

        print(f"\n>>> Top radiomic profiles by bootstrap frequency:")
        pd.set_option("display.max_colwidth", 100)
        pd.set_option("display.width", 200)
        print(agg_df.head(20)[["readable_rule", "n_conditions", "runs_appeared",
                                "run_freq_pct", "avg_n_patients", "avg_accuracy"]].to_string(index=False))

    med3pa_summary["error_quantile"] = args.med3pa_error_quantile
    med3pa_summary.to_csv(os.path.join(args.outdir, "med3pa_run_summary_radiomic.csv"), index=False)

    out = {
        "cohort": "blca",
        "arm": "radiomic",
        "endpoint": "PFI",
        "input_error_npz": os.path.abspath(args.error_npz),
        "input_radiomics_csv": os.path.abspath(args.radiomics_csv),
        "backbone_round_count": args.round_count,
        "backbone_save_root": args.save_root,
        "n_patients_aligned": int(len(patient_ids)),
        "n_events": int((pfi_e == 1).sum()),
        "n_censored": int((pfi_e == 0).sum()),
        "n_first_order_features": fi_metrics["n_first_order_features"],
        "n_features_in_med3pa": fi_metrics["n_selected_for_med3pa"],
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
        "med3pa_n_runs": args.n_runs,
        "classifier_metrics_on_eval": cm,
        "scaler_note": "StandardScaler fitted on X_med3pa (top-K first-order features).",
        "top_profiles_eval": profile_reports,
        "feature_importance_summary": fi_metrics,
    }
    out_json = os.path.join(args.outdir, "reliability_blca_radiomic.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f">>> Wrote {out_json}")
    print(f">>> Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()
