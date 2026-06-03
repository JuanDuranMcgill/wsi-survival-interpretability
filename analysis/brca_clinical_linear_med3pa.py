#!/usr/bin/env python3
# brca_clinical_linear_med3pa.py
#
# Step 1: load backbone outputs + clinical CSV, align patients.
# Step 2: clinical feature analysis vs survival_target:
#   - bootstrap RF feature importance
#   - Ridge linear model with CV, per-patient predictions
# Step 3: MED3PA on backbone error:
#   - define high-error = top error-quantile of backbone absolute error
#   - run MED3PA using clinical features + this binary label
#   - aggregate direction-based profiles


import os
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


warnings.filterwarnings("ignore")



# -------------------------------------------------------
# Paths (adjust if needed)
# -------------------------------------------------------
CLINICAL_CSV = "/home/sorkwos/links/scratch/clinical_features_BRCA.csv"
SAVE_ROOT    = "/home/sorkwos/links/scratch/med3pa_bootstrap_intermediate_BRCA"
FINAL_FILE   = os.path.join(SAVE_ROOT, "final_med3pa_input.npz")
ERROR_FILE   = os.path.join(SAVE_ROOT, "med3pa_error_input.npz")



# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def load_npz(path):
    return dict(np.load(path, allow_pickle=True))



# -------------------------------------------------------
# Step 1 – load and align backbone + clinical
# -------------------------------------------------------
def load_and_align_data(clinical_csv=CLINICAL_CSV):
    clin_df = pd.read_csv(clinical_csv, index_col="patient_id")
    clin_df.index = clin_df.index.astype(str)

    final_data = load_npz(FINAL_FILE)
    pids_final = np.array([str(p)[:12] for p in final_data["patient_ids"]])
    mean_risk = np.asarray(final_data["mean_risk"], dtype=np.float32)
    risk_df = pd.DataFrame({"survival_target": mean_risk}, index=pids_final)
    risk_df.index.name = "patient_id"

    err_data = load_npz(ERROR_FILE)
    pids_err = np.array([str(p)[:12] for p in err_data["patient_ids"]])
    errors = np.asarray(err_data["error"], dtype=np.float32)
    err_df = pd.DataFrame({"backbone_error": errors}, index=pids_err)
    err_df.index.name = "patient_id"

    merged = clin_df.join(risk_df, how="inner").join(err_df, how="inner")

    feature_names = [
        c for c in merged.columns
        if c not in ("survival_target", "backbone_error")
    ]
    X = merged[feature_names].to_numpy(dtype=np.float32)
    y = merged["survival_target"].to_numpy(dtype=np.float32)
    backbone_error = merged["backbone_error"].to_numpy(dtype=np.float32)
    patient_ids = merged.index.to_numpy().astype(str)

    return X, y, backbone_error, feature_names, patient_ids, merged



# -------------------------------------------------------
# Step 2a – bootstrap RF importance vs survival_target
# -------------------------------------------------------
def run_feature_importance(
    X,
    target,
    feature_names,
    n_trees,
    n_bootstrap,
    n_jobs,
    max_depth,
    min_samples_leaf,
):
    boot_per_job_base = n_bootstrap // n_jobs
    boot_remainder = n_bootstrap % n_jobs
    boot_counts = [
        boot_per_job_base + (1 if i < boot_remainder else 0)
        for i in range(n_jobs)
    ]
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
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                max_features="sqrt",
                random_state=seed + b,
                n_jobs=1,
                bootstrap=True,
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
        (X, target, boot_counts[i], trees_per_bootstrap,
         42 + i * 1000, max_depth, min_samples_leaf)
        for i in range(len(boot_counts))
    ]
    results = Parallel(n_jobs=len(boot_counts))(
        delayed(_fit_bootstrap_batch)(wa) for wa in worker_args
    )

    all_imps = np.concatenate([r[0] for r in results], axis=0)
    oob_scores = np.concatenate([r[1] for r in results], axis=0)
    n_boot_total = all_imps.shape[0]

    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0, ddof=1) if n_boot_total > 1 else np.zeros(
        X.shape[1], dtype=np.float32
    )
    ci_half = 1.96 * std_imp / np.sqrt(max(n_boot_total, 1))
    ci_lo = mean_imp - ci_half
    ci_hi = mean_imp + ci_half

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
            "higher feature -> higher survival_target"
            if np.isfinite(rho) and rho > 0
            else "higher feature -> lower survival_target"
        )
        if not np.isfinite(rho):
            direction = "undetermined"

        rows.append(
            {
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
                "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else np.nan,
                "spearman_p_value": round(float(pval), 6)
                if np.isfinite(pval)
                else np.nan,
                "direction": direction,
            }
        )

    feat_df = pd.DataFrame(rows).sort_values(
        ["rank", "median_rank", "feature"]
    )
    metrics = {
        "n_bootstrap": int(n_boot_total),
        "mean_oob_corr": float(np.nanmean(oob_scores))
        if np.isfinite(oob_scores).any()
        else np.nan,
        "std_oob_corr": float(np.nanstd(oob_scores))
        if np.isfinite(oob_scores).any()
        else np.nan,
    }
    return feat_df, metrics



# -------------------------------------------------------
# Step 2b – clinical Ridge model vs survival_target
# -------------------------------------------------------
def fit_linear_base_model(X, y, patient_ids, alpha, n_splits, seed):
    preds = np.full_like(y, np.nan, dtype=np.float32)
    abs_err = np.full_like(y, np.nan, dtype=np.float32)
    sq_err = np.full_like(y, np.nan, dtype=np.float32)
    split_rows = []

    for split_idx in range(n_splits):
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=seed + split_idx,
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx]).astype(np.float32)
        preds[te_idx] = pred
        abs_err[te_idx] = np.abs(pred - y[te_idx])
        sq_err[te_idx] = (pred - y[te_idx]) ** 2
        split_rows.append(
            {
                "split": split_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "test_r2": float(r2_score(y[te_idx], pred))
                if len(np.unique(y[te_idx])) > 1
                else np.nan,
                "test_rmse": float(
                    np.sqrt(mean_squared_error(y[te_idx], pred))
                ),
            }
        )

    missing = np.isnan(preds)
    if missing.any():
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X[~missing], y[~missing])
        pred = model.predict(X[missing]).astype(np.float32)
        preds[missing] = pred
        abs_err[missing] = np.abs(pred - y[missing])
        sq_err[missing] = (pred - y[missing]) ** 2

    patient_df = (
        pd.DataFrame(
            {
                "patient_id": patient_ids,
                "survival_target": y,
                "predicted_survival": preds,
                "patient_error_abs": abs_err,
                "patient_error_sq": sq_err,
            }
        )
        .sort_values("patient_error_abs", ascending=False)
    )

    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame(
        [
            {
                "alpha": alpha,
                "n_splits": n_splits,
                "mean_test_r2": split_df["test_r2"].mean(),
                "std_test_r2": split_df["test_r2"].std(ddof=1)
                if len(split_df) > 1
                else np.nan,
                "mean_test_rmse": split_df["test_rmse"].mean(),
                "std_test_rmse": split_df["test_rmse"].std(ddof=1)
                if len(split_df) > 1
                else np.nan,
                "overall_pred_target_corr": np.corrcoef(
                    patient_df["predicted_survival"],
                    patient_df["survival_target"],
                )[0, 1]
                if len(patient_df) > 1
                else np.nan,
            }
        ]
    )

    return patient_df, split_df, summary_df



# -------------------------------------------------------
# Step 3 – MED3PA on backbone error
# -------------------------------------------------------
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

        results.append(
            {
                "path_key": key,
                "path_raw": path,
                "thresholds": thresholds,
                "n_patients": n_patients,
                "pop_pct": pop_pct if pop_pct is not None else 0,
                "accuracy": acc,
                "n_conditions": len([c for c in path if c != "*"]),
            }
        )
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
    seed = 1000 + run_idx
    x_df = pd.DataFrame(X, columns=feature_names)

    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x_df, y, test_size=test_size, random_state=seed, stratify=y
    )
    x_train, x_ref, y_train, y_ref = train_test_split(
        x_train_pool,
        y_train_pool,
        test_size=ref_size,
        random_state=seed,
        stratify=y_train_pool,
    )

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
        "reference",
        x_ref.to_numpy(),
        y_ref,
        list(x_ref.columns),
    )
    datasets.set_from_data(
        "testing",
        x_test.to_numpy(),
        y_test,
        list(x_test.columns),
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
    profiles = extract_profiles_from_run(
        profiles_path, n_test_approx, min_patients=1
    )

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
                profile_records[p["path_key"]].append(
                    {
                        "n_patients": p["n_patients"],
                        "pop_pct": p["pop_pct"],
                        "accuracy": p["accuracy"],
                        "n_conditions": p["n_conditions"],
                        "thresholds": p["thresholds"],
                        "path_raw": p["path_raw"],
                    }
                )

    rows = []
    for key, records in profile_records.items():
        n_runs_appeared = len(records)
        avg_n_patients = np.mean([r["n_patients"] for r in records])
        avg_pop_pct = np.mean([r["pop_pct"] for r in records])
        avg_accuracy = np.mean([r["accuracy"] for r in records])
        std_accuracy = np.std([r["accuracy"] for r in records])
        n_conditions = records[0]["n_conditions"]

        all_thresholds = defaultdict(list)
        for r in records:
            for cond_key, thresh_val in r["thresholds"].items():
                all_thresholds[cond_key].append(thresh_val)

        threshold_summary = {}
        for cond_key, vals in sorted(all_thresholds.items()):
            threshold_summary[cond_key] = {
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        readable_parts = []
        for cond_key in sorted(all_thresholds.keys()):
            parts = cond_key.rsplit("_", 1)
            feature = parts[0]
            direction = parts[1]
            op = ">" if direction == "UP" else "<="
            median_t = threshold_summary[cond_key]["median"]
            range_str = (
                f"[{threshold_summary[cond_key]['min']}, "
                f"{threshold_summary[cond_key]['max']}]"
            )
            readable_parts.append(
                f"{feature} {op} {median_t} {range_str}"
            )

        rows.append(
            {
                "profile": key,
                "readable_rule": " AND ".join(readable_parts),
                "n_conditions": n_conditions,
                "runs_appeared": n_runs_appeared,
                "run_freq_pct": round(
                    100 * n_runs_appeared / max(n_completed, 1), 1
                ),
                "avg_n_patients": round(avg_n_patients, 1),
                "avg_pop_pct": round(avg_pop_pct, 2),
                "avg_accuracy": round(avg_accuracy, 4),
                "std_accuracy": round(std_accuracy, 4),
            }
        )

    agg_df = (
        pd.DataFrame(rows).sort_values("runs_appeared", ascending=False)
        if rows
        else pd.DataFrame()
    )
    summary = pd.DataFrame(
        [
            {
                "n_runs_requested": n_runs_requested,
                "n_runs_completed": n_completed,
                "n_unique_profiles": len(profile_records),
            }
        ]
    )
    return agg_df, summary



# -------------------------------------------------------
# Main CLI
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Clinical-only pipeline: "
            "clinical survival analysis + MED3PA on backbone error"
        )
    )
    ap.add_argument("--clinical-csv", type=str, default=CLINICAL_CSV)
    ap.add_argument("--outdir", type=str, default="results/brca_clinical_linear_med3pa")

    # Feature importance
    ap.add_argument("--fi-n-trees", type=int, default=200000)
    ap.add_argument("--fi-n-bootstrap", type=int, default=200)
    ap.add_argument("--fi-n-jobs", type=int, default=8)
    ap.add_argument("--fi-max-depth", type=int, default=6)
    ap.add_argument("--fi-min-samples-leaf", type=int, default=5)

    # Clinical Ridge model
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--cv-splits", type=int, default=5)

    # MED3PA
    ap.add_argument("--med3pa-n-runs", type=int, default=200)
    ap.add_argument("--med3pa-n-parallel", type=int, default=10)
    ap.add_argument("--med3pa-test-size", type=float, default=0.3)
    ap.add_argument("--med3pa-ref-size", type=float, default=0.25)
    ap.add_argument("--med3pa-error-quantile", type=float, default=0.8)
    ap.add_argument("--save-runs", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Step 1
    X, y, backbone_error, feature_names, patient_ids, merged = load_and_align_data(
        args.clinical_csv
    )

    # Step 2a – importance vs survival_target
    feat_df, fi_metrics = run_feature_importance(
        X,
        y,
        feature_names,
        args.fi_n_trees,
        args.fi_n_bootstrap,
        args.fi_n_jobs,
        args.fi_max_depth,
        args.fi_min_samples_leaf,
    )
    feat_df.to_csv(
        os.path.join(args.outdir, "clinical_importance_survival.csv"),
        index=False,
    )
    pd.DataFrame([fi_metrics]).to_csv(
        os.path.join(args.outdir, "feature_importance_run_summary.csv"),
        index=False,
    )

    # Step 2b – clinical Ridge model
    patient_pred_df, split_df, summary_df = fit_linear_base_model(
        X, y, patient_ids, args.ridge_alpha, args.cv_splits, 123
    )
    patient_pred_df["backbone_error"] = backbone_error
    split_df.to_csv(
        os.path.join(args.outdir, "clinical_linear_model_split_metrics.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(args.outdir, "clinical_linear_model_summary.csv"),
        index=False,
    )
    patient_pred_df.to_csv(
        os.path.join(
            args.outdir, "clinical_linear_model_patient_predictions.csv"
        ),
        index=False,
    )

    # Optional: NPZ-style outputs (for completeness / reuse)
    np.savez(
        os.path.join(args.outdir, "final_med3pa_input_clinical_linear.npz"),
        patient_ids=patient_ids,
        mean_risk=patient_pred_df.sort_values("patient_id")[
            "predicted_survival"
        ].to_numpy(dtype=np.float32),
    )
    np.savez(
        os.path.join(args.outdir, "med3pa_error_input_backbone.npz"),
        patient_ids=patient_ids,
        error=backbone_error.astype(np.float32),
    )

    # Step 3 – MED3PA label from backbone error (NOT from clinical model error)
    thresh = float(
        np.quantile(
            backbone_error[np.isfinite(backbone_error)],
            args.med3pa_error_quantile,
        )
    )
    y_med3pa = (backbone_error >= thresh).astype(int)

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
    label_df.to_csv(
        os.path.join(args.outdir, "med3pa_backbone_error_labels.csv"),
        index=False,
    )

    med3pa_params = {
        "uncertainty_metric": "sigmoidal_error",
        "ipc_type": "RandomForestRegressor",
        "ipc_params": {"n_estimators": 100},
        "apc_params": {"max_depth": 4},
        "ipc_grid_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 4, 6],
        },
        "apc_grid_params": {"min_samples_leaf": [2, 4, 6]},
        "samples_ratio_min": 0,
        "samples_ratio_max": 10,
        "samples_ratio_step": 5,
        "evaluate_models": True,
    }

    n_test_approx = int(round(args.med3pa_test_size * len(X)))
    all_run_profiles = Parallel(
        n_jobs=args.med3pa_n_parallel, verbose=5
    )(
        delayed(run_single_med3pa)(
            run_idx=i,
            X=X,
            y=y_med3pa,
            feature_names=feature_names,
            test_size=args.med3pa_test_size,
            ref_size=args.med3pa_ref_size,
            med3pa_params=med3pa_params,
            n_test_approx=n_test_approx,
            outdir=args.outdir,
            save_runs=args.save_runs,
        )
        for i in range(args.med3pa_n_runs)
    )

    agg_df, med3pa_summary = aggregate_profiles(
        all_run_profiles, args.med3pa_n_runs
    )
    if not agg_df.empty:
        agg_df.to_csv(
            os.path.join(args.outdir, "all_profiles_dir.csv"), index=False
        )
        for min_freq in [10, 25, 50]:
            agg_df[agg_df["runs_appeared"] >= min_freq].to_csv(
                os.path.join(args.outdir, f"profiles_freq{min_freq}.csv"),
                index=False,
            )

    med3pa_summary["error_quantile"] = args.med3pa_error_quantile
    med3pa_summary.to_csv(
        os.path.join(args.outdir, "med3pa_run_summary.csv"), index=False
    )

    with open(
        os.path.join(args.outdir, "README_outputs.txt"), "w"
    ) as f:
        f.write("Clinical-only pipeline outputs\n")
        f.write(
            "- clinical_importance_survival.csv: RF feature importance vs survival_target\n"
        )
        f.write(
            "- clinical_linear_model_summary.csv: CV performance of clinical Ridge model\n"
        )
        f.write(
            "- clinical_linear_model_patient_predictions.csv: per-patient clinical prediction + error\n"
        )
        f.write(
            "- med3pa_error_input_backbone.npz: backbone absolute error for reuse\n"
        )
        f.write(
            "- med3pa_backbone_error_labels.csv: binary high/low-error label from backbone error\n"
        )
        f.write(
            "- all_profiles_dir.csv, profiles_freq*.csv: aggregated MED3PA profiles\n"
        )

    print(f"Saved outputs to {args.outdir}")


if __name__ == "__main__":
    main()