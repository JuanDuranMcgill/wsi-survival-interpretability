#!/usr/bin/env python3
# blca_radiomic_linear_med3pa.py
#
# Step 1: load backbone outputs + radiomics CSV, align patients.
# Step 2: radiomic feature analysis vs survival_target:
#   - RF importance using survival_target (WSI risk score), consistent with clinical pipeline
#   - restrict to FIRST-ORDER radiomic features only
# Step 3: MED3PA on backbone error:
#   - define high-error = top error-quantile of backbone absolute error
#   - run MED3PA using a selected top-K subset of first-order radiomic features + this binary label
#   - aggregate direction-based profiles
#
# NOTE:
#   - MED3PA label is ALWAYS derived from backbone_error (not radiomic model error).
#   - Only FIRST-ORDER radiomic features are used, and only a top-K subset feeds MED3PA.

import os
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

from MED3pa.med3pa import Med3paExperiment
from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager

warnings.filterwarnings("ignore")


# -------------------------------------------------------
# Paths (BLCA radiomics + backbone)
# -------------------------------------------------------
RADIOMICS_CSV = "/home/sorkwos/links/scratch/radiomics_output/radiomics_pre_corr.csv"
SAVE_ROOT     = "/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate"
FINAL_FILE    = os.path.join(SAVE_ROOT, "final_med3pa_input.npz")
ERROR_FILE    = os.path.join(SAVE_ROOT, "med3pa_error_input.npz")


# -------------------------------------------------------
# Feature name parsing (BLCA regions)
# -------------------------------------------------------
KNOWN_REGIONS = [
    "urothelium", "stroma", "tumor", "necrosis",
    "vessels", "adipose", "muscularis", "blood_vessels",
    "immune", "lamina",
]


def parse_feature_components(feat_name):
    """
    Parse a radiomic feature name of the form:
        region_channel_original_firstorder_METRIC_mean/std
    into semantic components (region, channel, feature_class, metric, stat).
    """
    result = {
        "region": "other",
        "channel": "unknown",
        "feature_class": "unknown",
        "metric": feat_name,
        "stat": "unknown",
    }
    remainder = feat_name

    # Region prefix
    for r in sorted(KNOWN_REGIONS, key=len, reverse=True):
        if remainder.startswith(r + "_"):
            result["region"] = r
            remainder = remainder[len(r) + 1:]
            break

    # Channel (H/E)
    for ch in ["H", "E"]:
        if remainder.startswith(ch + "_"):
            result["channel"] = ch
            remainder = remainder[len(ch) + 1:]
            break

    # Drop "original_" if present
    if remainder.startswith("original_"):
        remainder = remainder[len("original_"):]

    # Feature class
    if remainder.startswith("firstorder_"):
        result["feature_class"] = "firstorder"
        remainder = remainder[len("firstorder_"):]
    elif remainder.startswith("glcm_"):
        result["feature_class"] = "glcm"
        remainder = remainder[len("glcm_"):]

    # Stat suffix
    if remainder.endswith("_mean"):
        result["stat"] = "mean"
        result["metric"] = remainder[:-5]
    elif remainder.endswith("_std"):
        result["stat"] = "std"
        result["metric"] = remainder[:-4]
    else:
        result["metric"] = remainder

    return result


# -------------------------------------------------------
# Step 1 – load and align backbone + radiomics
# -------------------------------------------------------
def load_and_align_data(radiomics_csv=RADIOMICS_CSV):
    """
    Load BLCA radiomics, backbone survival target (mean_risk),
    and backbone_error, then align on patient_id.
    """
    rad_df = pd.read_csv(radiomics_csv, index_col="patient_id")
    rad_df.index = rad_df.index.astype(str)

    final_data = np.load(FINAL_FILE, allow_pickle=True)
    pids_final = np.array([str(p)[:12] for p in final_data["patient_ids"]])
    mean_risk = np.asarray(final_data["mean_risk"], dtype=np.float32)
    risk_df = pd.DataFrame({"survival_target": mean_risk}, index=pids_final)
    risk_df.index.name = "patient_id"

    err_data = np.load(ERROR_FILE, allow_pickle=True)
    pids_err = np.array([str(p)[:12] for p in err_data["patient_ids"]])
    errors = np.asarray(err_data["error"], dtype=np.float32)
    err_df = pd.DataFrame({"backbone_error": errors}, index=pids_err)
    err_df.index.name = "patient_id"

    merged = rad_df.join(risk_df, how="inner").join(err_df, how="inner")

    feature_names = [c for c in merged.columns if c not in ("survival_target", "backbone_error")]
    X = merged[feature_names].to_numpy(dtype=np.float32)
    survival_target = merged["survival_target"].to_numpy(dtype=np.float32)
    backbone_error = merged["backbone_error"].to_numpy(dtype=np.float32)
    patient_ids = merged.index.to_numpy()

    return X, survival_target, backbone_error, feature_names, patient_ids


# -------------------------------------------------------
# Step 2 – RF importance vs survival_target
#          (first-order only)
# -------------------------------------------------------
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
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            random_state=seed + b,
            n_jobs=1,
            bootstrap=True,
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


def run_radiomic_importance_firstorder(
    X,
    survival_target,
    feature_names,
    n_trees,
    n_bootstrap,
    n_jobs,
    max_depth,
    min_samples_leaf,
    outdir,
    top_n_for_med3pa,
):
    """
    Harrell-style radiomic importance:
      - restrict to FIRST-ORDER features
      - model survival_target (WSI risk score) consistently with clinical pipeline
      - bootstrap RF to get stable ranks
      - write radiomic_importance.csv + summaries
      - return indices / names of top-N first-order features
        for MED3PA.
    """
    os.makedirs(outdir, exist_ok=True)

    # Restrict to first-order only
    fo_mask = np.array(["firstorder" in f for f in feature_names])
    X_fo = X[:, fo_mask]
    fo_feature_names = [f for f, m in zip(feature_names, fo_mask) if m]

    print(f">>> Radiomics (first-order-only) for Harrell-style importance")
    print(f"    First-order features: {len(fo_feature_names)} / {len(feature_names)} total")

    n_patients, n_features = X_fo.shape
    print(f"    Patients: {n_patients}")
    print(f"    First-order features: {n_features}")
    print(
        f"    Survival target (WSI risk score): "
        f"min={survival_target.min():.4f}, "
        f"median={np.median(survival_target):.4f}, "
        f"max={survival_target.max():.4f}"
    )
    print(f"    Bootstrap repetitions: {n_bootstrap}")
    print(f"    Total trees budget: {n_trees} | Workers: {n_jobs}")

    boot_per_job_base = n_bootstrap // n_jobs
    boot_remainder = n_bootstrap % n_jobs
    boot_counts = [boot_per_job_base + (1 if i < boot_remainder else 0) for i in range(n_jobs)]
    boot_counts = [b for b in boot_counts if b > 0]

    trees_per_bootstrap = max(100, int(np.ceil(n_trees / n_bootstrap)))

    worker_args = [
        (X_fo, survival_target, boot_counts[i], trees_per_bootstrap, 42 + i * 1000, max_depth, min_samples_leaf)
        for i in range(len(boot_counts))
    ]

    print(f"\n>>> Fitting {n_bootstrap} bootstrap forests in parallel...")
    print(f"    Trees per bootstrap fit: {trees_per_bootstrap}")
    with mp.Pool(processes=len(boot_counts)) as pool:
        results = pool.map(_fit_bootstrap_batch, worker_args)

    all_imps = np.concatenate([r[0] for r in results], axis=0)
    oob_scores = np.concatenate([r[1] for r in results], axis=0)
    n_boot_total = all_imps.shape[0]

    print(f"    Total bootstrap fits collected: {n_boot_total}")
    if np.isfinite(oob_scores).any():
        print(
            f"    Mean out-of-bootstrap prediction correlation: "
            f"{np.nanmean(oob_scores):.4f} ± {np.nanstd(oob_scores):.4f}"
        )
    else:
        print("    Mean out-of-bootstrap prediction correlation: unavailable")

    # Aggregate importances and ranking stability
    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0, ddof=1) if n_boot_total > 1 else np.zeros(n_features, dtype=np.float32)
    ci_half = 1.96 * std_imp / np.sqrt(max(n_boot_total, 1))
    ci_lo = mean_imp - ci_half
    ci_hi = mean_imp + ci_half

    boot_ranks = np.argsort(np.argsort(-all_imps, axis=1), axis=1) + 1
    median_rank = np.median(boot_ranks, axis=0)
    rank_q025 = np.quantile(boot_ranks, 0.025, axis=0)
    rank_q975 = np.quantile(boot_ranks, 0.975, axis=0)

    rank_order = np.argsort(-mean_imp)
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, n_features + 1)

    top20_pct = 100.0 * (boot_ranks <= 20).mean(axis=0)
    top50_pct = 100.0 * (boot_ranks <= 50).mean(axis=0)

    # Correlations with survival_target
    corr_rows = []
    for i in range(n_features):
        x = X_fo[:, i]
        ok = np.isfinite(x) & np.isfinite(survival_target)
        if ok.sum() >= 5 and np.nanstd(x[ok]) > 0 and np.nanstd(survival_target[ok]) > 0:
            rho, pval = stats.spearmanr(x[ok], survival_target[ok])
        else:
            rho, pval = np.nan, np.nan
        corr_rows.append((rho, pval))

    # Per-feature dataframe
    rows = []
    for i, feat in enumerate(fo_feature_names):
        comp = parse_feature_components(feat)
        rho, pval = corr_rows[i]
        direction = "higher feature -> higher survival_target" if np.isfinite(rho) and rho > 0 else "higher feature -> lower survival_target"
        if not np.isfinite(rho):
            direction = "undetermined"
        rows.append({
            "feature": feat,
            "region": comp["region"],
            "channel": comp["channel"],
            "feature_class": comp["feature_class"],
            "metric": comp["metric"],
            "stat": comp["stat"],
            "mean_importance": round(float(mean_imp[i]), 6),
            "std_importance": round(float(std_imp[i]), 6),
            "ci_lo": round(float(ci_lo[i]), 6),
            "ci_hi": round(float(ci_hi[i]), 6),
            "cv": round(float(std_imp[i] / mean_imp[i]), 3) if mean_imp[i] > 0 else float("nan"),
            "rank": int(ranks[i]),
            "median_rank": round(float(median_rank[i]), 1),
            "rank_q025": round(float(rank_q025[i]), 1),
            "rank_q975": round(float(rank_q975[i]), 1),
            "top20_pct": round(float(top20_pct[i]), 1),
            "top50_pct": round(float(top50_pct[i]), 1),
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else np.nan,
            "spearman_p_value": round(float(pval), 6) if np.isfinite(pval) else np.nan,
            "direction": direction,
        })

    feat_df = pd.DataFrame(rows).sort_values(["rank", "median_rank", "feature"])

    # Simple region / class / channel / stat summaries (same as original script)
    region_rows = []
    for region in sorted(feat_df["region"].unique()):
        sub = feat_df[feat_df["region"] == region].sort_values("rank")
        region_rows.append({
            "region": region,
            "n_features": len(sub),
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
            "feature_class": fc,
            "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    class_df = pd.DataFrame(class_rows).sort_values("sum_mean_importance", ascending=False)

    channel_rows = []
    for ch in sorted(feat_df["channel"].unique()):
        sub = feat_df[feat_df["channel"] == ch]
        channel_rows.append({
            "channel": ch,
            "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    channel_df = pd.DataFrame(channel_rows).sort_values("sum_mean_importance", ascending=False)

    stat_rows = []
    for st in sorted(feat_df["stat"].unique()):
        sub = feat_df[feat_df["stat"] == st]
        stat_rows.append({
            "stat": st,
            "n_features": len(sub),
            "sum_mean_importance": round(sub["mean_importance"].sum(), 4),
            "avg_mean_importance": round(sub["mean_importance"].mean(), 6),
            "features_in_top20": int((sub["rank"] <= 20).sum()),
        })
    stat_df = pd.DataFrame(stat_rows).sort_values("sum_mean_importance", ascending=False)

    # Comparison table: top 30 features vs survival_target
    print("\n>>> Building comparison table (top features vs survival_target)...")
    top_n_comp = min(30, n_features)
    top_indices_comp = np.argsort(-mean_imp)[:top_n_comp]
    comp_rows = []
    for idx in top_indices_comp:
        feat = fo_feature_names[idx]
        vals = X_fo[:, idx]
        ok = np.isfinite(vals) & np.isfinite(survival_target)
        vals = vals[ok]
        errs = survival_target[ok]
        comp = parse_feature_components(feat)

        if len(vals) >= 5 and np.nanstd(vals) > 0 and np.nanstd(errs) > 0:
            slope, intercept = np.polyfit(vals, errs, 1)
            rho, pval = stats.spearmanr(vals, errs)
        else:
            slope, intercept, rho, pval = np.nan, np.nan, np.nan, np.nan

        comp_rows.append({
            "rank": int(ranks[idx]),
            "feature": feat,
            "region": comp["region"],
            "feature_mean": round(float(np.nanmean(vals)), 4) if len(vals) else np.nan,
            "feature_std": round(float(np.nanstd(vals)), 4) if len(vals) else np.nan,
            "survival_target_mean": round(float(np.nanmean(errs)), 4) if len(errs) else np.nan,
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else np.nan,
            "p_value": round(float(pval), 6) if np.isfinite(pval) else np.nan,
            "slope_linear_fit": round(float(slope), 6) if np.isfinite(slope) else np.nan,
            "direction": "higher feature -> higher survival_target" if np.isfinite(rho) and rho > 0 else "higher feature -> lower survival_target" if np.isfinite(rho) else "undetermined",
        })
    comp_df = pd.DataFrame(comp_rows)

    # Save radiomic-level outputs
    feat_path = os.path.join(outdir, "radiomic_importance.csv")
    region_path = os.path.join(outdir, "region_summary.csv")
    class_path = os.path.join(outdir, "class_summary.csv")
    channel_path = os.path.join(outdir, "channel_summary.csv")
    stat_path = os.path.join(outdir, "stat_summary.csv")
    comp_path = os.path.join(outdir, "radiomic_comparison_table.csv")

    feat_df.to_csv(feat_path, index=False)
    region_df.to_csv(region_path, index=False)
    class_df.to_csv(class_path, index=False)
    channel_df.to_csv(channel_path, index=False)
    stat_df.to_csv(stat_path, index=False)
    comp_df.to_csv(comp_path, index=False)

    print(f"\n>>> Saved radiomic importance outputs to {outdir}")
    print(f"    {feat_path}")
    print(f"    {region_path}")
    print(f"    {class_path}")
    print(f"    {channel_path}")
    print(f"    {stat_path}")
    print(f"    {comp_path}")

    # Return indices/names for MED3PA
    # MED3PA will use top_n_for_med3pa first-order radiomic features as inputs.
    k = min(top_n_for_med3pa, n_features)
    top_indices_for_med3pa = np.argsort(-mean_imp)[:k]
    selected_feature_names = [fo_feature_names[i] for i in top_indices_for_med3pa]
    print(f"\n>>> Selected top {k} first-order radiomic features for MED3PA.")
    return X_fo, survival_target, fo_feature_names, top_indices_for_med3pa, selected_feature_names


# -------------------------------------------------------
# Step 3 – MED3PA on backbone error with radiomic features
# -------------------------------------------------------
def parse_condition_dir(cond_str):
    """
    'feature > 0.5'  -> ('feature_UP', 0.5)
    'feature <= 0.5' -> ('feature_DOWN', 0.5)
    """
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
    """
    Canonicalize a profile path by (feature, direction) only.
    Returns (profile_key, {condition_key: threshold}).
    """
    if path == ["*"]:
        return None, {}

    conds = []
    thresholds = {}

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

    key = " | ".join(sorted(conds))
    return key, thresholds


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
            "path_key": key,
            "path_raw": path,
            "thresholds": thresholds,
            "n_patients": n_patients,
            "pop_pct": pop_pct if pop_pct is not None else 0,
            "accuracy": acc,
            "n_conditions": len([c for c in path if c != "*"]),
        })

    return results


def run_single_med3pa(
    run_idx,
    X,
    y,
    feature_names,
    test_size,
    ref_size,
    med3pa_params,
    n_test_approx,
    outdir,
    save_runs,
):
    """Run one Med3PA experiment and return extracted profiles."""
    seed = 1000 + run_idx

    x_df = pd.DataFrame(X, columns=feature_names)

    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x_df, y, test_size=test_size, random_state=seed, stratify=y
    )
    x_train, x_ref, y_train, y_ref = train_test_split(
        x_train_pool, y_train_pool,
        test_size=ref_size, random_state=seed, stratify=y_train_pool
    )

    # n_jobs=1 per RF to avoid contention with outer parallelism
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=seed,
        class_weight="balanced",
        n_jobs=1,
    )
    clf.fit(x_train, y_train)

    datasets = DatasetsManager()
    datasets.set_from_data(
        dataset_type="reference",
        observations=x_ref.to_numpy(),
        true_labels=y_ref,
        column_labels=list(x_ref.columns),
    )
    datasets.set_from_data(
        dataset_type="testing",
        observations=x_test.to_numpy(),
        true_labels=y_test,
        column_labels=list(x_test.columns),
    )

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

    # Optionally clean up run files to save space
    if not save_runs:
        import shutil
        shutil.rmtree(outdir_run, ignore_errors=True)

    return profiles


def aggregate_profiles(all_run_profiles, n_runs_requested):
    profile_records = defaultdict(list)

    n_completed = 0
    for run_profiles in all_run_profiles:
        if run_profiles is not None:
            n_completed += 1
            for p in run_profiles:
                profile_records[p["path_key"]].append({
                    "n_patients": p["n_patients"],
                    "pop_pct": p["pop_pct"],
                    "accuracy": p["accuracy"],
                    "n_conditions": p["n_conditions"],
                    "thresholds": p["thresholds"],
                    "path_raw": p["path_raw"],
                })

    rows = []
    for key, records in profile_records.items():
        n_runs_appeared = len(records)
        avg_n_patients  = np.mean([r["n_patients"] for r in records])
        avg_pop_pct     = np.mean([r["pop_pct"] for r in records])
        avg_accuracy    = np.mean([r["accuracy"] for r in records])
        std_accuracy    = np.std([r["accuracy"] for r in records])
        n_conditions    = records[0]["n_conditions"]

        # Collect threshold statistics for each condition
        all_thresholds = defaultdict(list)
        for r in records:
            for cond_key, thresh_val in r["thresholds"].items():
                all_thresholds[cond_key].append(thresh_val)

        threshold_summary = {}
        for cond_key, vals in sorted(all_thresholds.items()):
            threshold_summary[cond_key] = {
                "median": round(float(np.median(vals)), 4),
                "min":    round(float(np.min(vals)), 4),
                "max":    round(float(np.max(vals)), 4),
            }

        readable_parts = []
        for cond_key in sorted(all_thresholds.keys()):
            # cond_key like "feature_UP" or "feature_DOWN"
            parts = cond_key.rsplit("_", 1)
            feature = parts[0]
            direction = parts[1]
            op = ">" if direction == "UP" else "<="
            median_t = threshold_summary[cond_key]["median"]
            range_str = f"[{threshold_summary[cond_key]['min']}, {threshold_summary[cond_key]['max']}]"
            readable_parts.append(f"{feature} {op} {median_t} {range_str}")

        readable_rule = " AND ".join(readable_parts)

        rows.append({
            "profile":          key,
            "readable_rule":    readable_rule,
            "n_conditions":     n_conditions,
            "runs_appeared":    n_runs_appeared,
            "run_freq_pct":     round(100 * n_runs_appeared / max(n_completed, 1), 1),
            "avg_n_patients":   round(avg_n_patients, 1),
            "avg_pop_pct":      round(avg_pop_pct, 2),
            "avg_accuracy":     round(avg_accuracy, 4),
            "std_accuracy":     round(std_accuracy, 4),
        })

    agg_df = pd.DataFrame(rows).sort_values("runs_appeared", ascending=False) if rows else pd.DataFrame()
    summary = pd.DataFrame(
        [{
            "n_runs_requested": n_runs_requested,
            "n_runs_completed": n_completed,
            "n_unique_profiles": len(profile_records),
        }]
    )
    return agg_df, summary


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "BLCA radiomics-only pipeline: "
            "first-order radiomic survival/error analysis + MED3PA on backbone error"
        )
    )

    # Radiomics / importance
    ap.add_argument("--radiomics-csv", type=str, default=RADIOMICS_CSV)
    ap.add_argument("--outdir", type=str, default="results/blca_radiomic_linear_med3pa")
    ap.add_argument("--n-trees", type=int, default=10_000_000,
                    help="Total trees budget across all bootstrap fits for radiomic importance.")
    ap.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Number of parallel processes for radiomic importance.")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--top-n-med3pa", type=int, default=30,
                    help="Number of top first-order radiomic features to feed into MED3PA.")

    # MED3PA
    ap.add_argument("--n-runs",         type=int,   default=200)
    ap.add_argument("--n-parallel",     type=int,   default=10,
                    help="Number of parallel Med3PA workers.")
    ap.add_argument("--error-quantile", type=float, default=0.8,
                    help="Quantile on backbone_error to define high-error label.")
    ap.add_argument("--test-size",      type=float, default=0.3)
    ap.add_argument("--ref-size",       type=float, default=0.25)
    ap.add_argument("--save-runs",      action="store_true",
                    help="Keep individual Med3PA run files on disk (default: delete after extraction).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Step 1 – load radiomics + backbone
    print(">>> Loading and aligning BLCA radiomics + backbone...")
    X, survival_target, backbone_error, feature_names, patient_ids = load_and_align_data(args.radiomics_csv)
    print(f"    Patients: {X.shape[0]} | Radiomic features: {X.shape[1]}")

    # Step 2 – radiomic first-order RF importance vs survival_target
    X_fo, _, fo_feature_names, top_idx_med3pa, selected_feature_names = run_radiomic_importance_firstorder(
        X=X,
        survival_target=survival_target,        # WSI-derived survival risk score
        feature_names=feature_names,
        n_trees=args.n_trees,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        outdir=args.outdir,
        top_n_for_med3pa=args.top_n_med3pa,
    )

    # Slice radiomic matrix down to selected top-K first-order features for MED3PA
    X_med3pa = X_fo[:, top_idx_med3pa]
    print(f"\n>>> MED3PA will use {X_med3pa.shape[1]} first-order radiomic features as inputs.")

    # Step 3 – MED3PA label from backbone_error (NOT from radiomic model error)
    thresh = float(np.quantile(backbone_error[np.isfinite(backbone_error)], args.error_quantile))
    y_med3pa = (backbone_error >= thresh).astype(int)
    print(
        f"    Backbone error quantile={args.error_quantile:.2f} "
        f"-> threshold={thresh:.4f} | high_error={y_med3pa.sum()}"
    )

    # Save label table
    label_df = (
        pd.DataFrame(
            {
                "patient_id": patient_ids,
                "backbone_error": backbone_error,
                "high_error_label": y_med3pa,
            }
        )
        .sort_values("backbone_error", ascending=False)
    )
    label_path = os.path.join(args.outdir, "med3pa_backbone_error_labels_radiomic.csv")
    label_df.to_csv(label_path, index=False)
    print(f"    Saved radiomic MED3PA label table: {label_path}")

    # MED3PA params (same template as clinical script)
    med3pa_params = {
        "uncertainty_metric": "sigmoidal_error",
        "ipc_type":           "RandomForestRegressor",
        "ipc_params":         {"n_estimators": 100},
        "apc_params":         {"max_depth": 4},
        "ipc_grid_params": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [2, 4, 6],
        },
        "apc_grid_params": {
            "min_samples_leaf": [2, 4, 6],
        },
        "samples_ratio_min":  0,
        "samples_ratio_max":  10,
        "samples_ratio_step": 5,
        "evaluate_models":    True,
    }

    n_test_approx = int(round(args.test_size * len(X_med3pa)))
    print(
        f"\n>>> Running {args.n_runs} Med3PA bootstrap iterations "
        f"({args.n_parallel} parallel workers)..."
    )

    all_run_profiles = Parallel(n_jobs=args.n_parallel, verbose=5)(
        delayed(run_single_med3pa)(
            run_idx=i,
            X=X_med3pa,
            y=y_med3pa,
            feature_names=selected_feature_names,
            test_size=args.test_size,
            ref_size=args.ref_size,
            med3pa_params=med3pa_params,
            n_test_approx=n_test_approx,
            outdir=args.outdir,
            save_runs=args.save_runs,
        )
        for i in range(args.n_runs)
    )

    # Aggregate MED3PA profiles (direction-based matching)
    print("\n>>> Aggregating radiomic MED3PA profiles (direction-based matching)...")
    agg_df, med3pa_summary = aggregate_profiles(all_run_profiles, args.n_runs)

    if agg_df.empty:
        print(">>> No profiles extracted. Check profiles.json structure.")
    else:
        all_path = os.path.join(args.outdir, "all_profiles_dir_radiomic.csv")
        agg_df.to_csv(all_path, index=False)
        print(f"    Saved all radiomic MED3PA profiles: {all_path}")

        # Also save filtered versions at various thresholds
        for min_freq in [10, 25, 50]:
            filtered = agg_df[agg_df["runs_appeared"] >= min_freq]
            fpath = os.path.join(args.outdir, f"profiles_freq{min_freq}_radiomic.csv")
            filtered.to_csv(fpath, index=False)
            print(f"    Saved radiomic profiles appearing in >= {min_freq} runs: {fpath}")

        print(f"\n>>> Radiomic MED3PA results summary:")
        print(f"    Total direction-based profiles: {len(agg_df)}")
        for min_freq in [10, 25, 50]:
            n_stable = len(agg_df[agg_df["runs_appeared"] >= min_freq])
            print(f"    Profiles appearing in >= {min_freq} runs: {n_stable}")

        print(f"\n>>> Top 20 radiomic profiles by bootstrap frequency:")
        pd.set_option("display.max_colwidth", 100)
        pd.set_option("display.width", 200)
        top = agg_df.head(20)
        print(
            top[[
                "readable_rule",
                "n_conditions",
                "runs_appeared",
                "run_freq_pct",
                "avg_n_patients",
                "avg_accuracy",
                "std_accuracy",
            ]].to_string(index=False)
        )

    # Save MED3PA summary
    med3pa_summary["error_quantile"] = args.error_quantile
    med3pa_summary.to_csv(
        os.path.join(args.outdir, "med3pa_run_summary_radiomic.csv"),
        index=False,
    )

    # README
    with open(os.path.join(args.outdir, "README_outputs_radiomic.txt"), "w") as f:
        f.write("BLCA radiomics-only pipeline outputs (first-order only)\n")
        f.write("- radiomic_importance.csv: RF feature importance vs survival_target (WSI risk score)\n")
        f.write("- region_summary.csv, class_summary.csv, channel_summary.csv, stat_summary.csv\n")
        f.write("- radiomic_comparison_table.csv: top radiomic features vs survival_target\n")
        f.write("- med3pa_backbone_error_labels_radiomic.csv: high/low-error label from backbone_error\n")
        f.write("- all_profiles_dir_radiomic.csv, profiles_freq*_radiomic.csv: aggregated radiomic MED3PA profiles\n")
        f.write("- med3pa_run_summary_radiomic.csv: MED3PA run counts + error quantile\n")

    print(f"\n>>> Done. Saved outputs to {args.outdir}")


if __name__ == "__main__":
    main()