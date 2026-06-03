#!/usr/bin/env python3
"""
Publication-quality analysis figures for a single med3pa pipeline output folder.

Usage:
    python analyze_med3pa_results.py \
        --results-dir results/blca_clinical_med3pa_500k \
        --outdir figures/blca_clinical_med3pa_500k \
        --label "BLCA Clinical"

Generates:
  fig1_feature_importance.png        - Bootstrap RF importance + CIs, coloured by direction
  fig2_spearman_forest.png           - Spearman rho forest plot for all features
  fig3_predicted_vs_actual.png       - Ridge predicted vs actual survival, coloured by error
  fig4_error_distribution.png        - Per-patient absolute error histogram with high/low split
  fig5_med3pa_top_profiles.png       - Top MED3PA profiles by run frequency (bar, coloured by accuracy)
  fig6_accuracy_coverage_tradeoff.png- Scatter: accuracy vs population coverage, sized by freq
  fig7_feature_in_profiles.png       - How often each feature appears in stable profiles (>= freq_threshold)
  summary_stats.txt                  - Key numbers for the paper (model perf, profile counts, etc.)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE  = '#2563EB'
RED   = '#DC2626'
GREEN = '#16A34A'
GREY  = '#64748B'
LGREY = '#E2E8F0'
GOLD  = '#D97706'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        LGREY,
    'grid.linewidth':    0.6,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RENAME = {
    'pathologic_T': 'Pathologic T stage',
    'pathologic_stage': 'Pathologic stage',
    'pathologic_N': 'Pathologic N stage',
    'pathologic_M': 'Pathologic M stage',
    'age_at_diagnosis': 'Age at diagnosis',
    'weight': 'Body weight',
    'height': 'Body height',
    'number_of_lymphnodes_positive_by_he': 'LN positive (HE)',
    'lymph_node_examined_count': 'LN examined count',
    'lymphovascular_invasion_present': 'LVI present',
    'neoplasm_histologic_grade_low_grade': 'Histologic grade (low)',
    'neoplasm_histologic_grade_high_grade': 'Histologic grade (high)',
    'tobacco_smoking_history_1': 'Smoking: never',
    'tobacco_smoking_history_2': 'Smoking: current reformed >15y',
    'tobacco_smoking_history_3': 'Smoking: current smoker',
    'tobacco_smoking_history_4': 'Smoking: current reformed <15y',
    'tobacco_smoking_history_5': 'Smoking: other',
    'gender': 'Sex',
    'hist_of_non_mibc': 'Hx non-MIBC',
    'history_of_neoadjuvant_treatment': 'Neoadjuvant treatment',
    'histological_type_muscle_invasive_urothelial_carcinoma_pt2_or_above':
        'MIUC pT2+',
    # BRCA-specific
    'pathologic_t': 'Pathologic T stage',
    'pathologic_n': 'Pathologic N stage',
    'pathologic_m': 'Pathologic M stage',
    'breast_carcinoma_estrogen_receptor_status': 'ER status',
    'breast_carcinoma_progesterone_receptor_status': 'PR status',
    'lab_proc_her2_neu_immunohistochemistry_receptor_status': 'HER2 status (IHC)',
    'menopause_status_post_prior_bilateral_ovariectomy_or_12_mo_since_lmp_with_no_prior_hysterectomy': 'Menopause: post (oophorectomy/LMP)',
    'menopause_status_pre_6_months_since_lmp_and_no_prior_bilateral_ovariectomy_and_not_on_estrogen_replacement': 'Menopause: pre',
    'menopause_status_peri_612_months_since_last_menstrual_period': 'Menopause: peri',
    'menopause_status_indeterminate_neither_pre_or_postmenopausal': 'Menopause: indeterminate',
    'menopause_status_nan': 'Menopause: unknown',
    'histological_type_infiltrating_ductal_carcinoma': 'Histology: IDC',
    'histological_type_infiltrating_lobular_carcinoma': 'Histology: ILC',
    'histological_type_mucinous_carcinoma': 'Histology: mucinous',
    'histological_type_metaplastic_carcinoma': 'Histology: metaplastic',
    'histological_type_medullary_carcinoma': 'Histology: medullary',
    'histological_type_mixed_histology_please_specify': 'Histology: mixed',
    'histological_type_infiltrating_carcinoma_nos': 'Histology: NOS',
    'histological_type_other_specify': 'Histology: other',
    'histological_type_nan': 'Histology: unknown',
    # radiomic catch-all: keep original but shorten texture prefix
}


def pretty(name):
    if name in RENAME:
        return RENAME[name]
    # shorten common radiomic prefixes
    name = re.sub(r'^original_', '', name)
    name = re.sub(r'_', ' ', name)
    return name.title()


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def load_csv(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1 – Feature importance (bootstrap RF)
# ---------------------------------------------------------------------------
def fig_feature_importance(imp_df, outdir, label, top_n=15):
    df = imp_df.copy().head(top_n).iloc[::-1]   # reverse so top is at top
    n = len(df)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * n + 1.5)))

    for i, row in enumerate(df.itertuples()):
        direction_up = 'higher' in row.direction and 'higher survival' in row.direction
        color = BLUE if direction_up else RED
        err_lo = row.mean_importance - row.ci_lo
        err_hi = row.ci_hi - row.mean_importance
        ax.barh(i, row.mean_importance, color=color, alpha=0.82, height=0.65,
                xerr=[[err_lo], [err_hi]], error_kw=dict(ecolor=GREY, capsize=3, lw=1.2))
        stars = sig_stars(row.spearman_p_value)
        ax.text(row.ci_hi + 0.001, i, stars, va='center', fontsize=8,
                color=GREY if stars == 'ns' else '#1e293b')

    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty(r) for r in df.feature], fontsize=9)
    ax.set_xlabel('Mean RF importance (bootstrap 95 % CI)', fontsize=10)
    ax.set_title(f'{label} – Feature importance\n(survival target; top {top_n} of {len(imp_df)})', fontsize=11)

    pos_patch = mpatches.Patch(color=BLUE,  alpha=0.82, label='higher → ↑ survival')
    neg_patch = mpatches.Patch(color=RED,   alpha=0.82, label='higher → ↓ survival')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right', framealpha=0.9)

    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x')
    plt.tight_layout()
    out = os.path.join(outdir, 'fig1_feature_importance.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 2 – Spearman forest plot
# ---------------------------------------------------------------------------
def fig_spearman_forest(imp_df, outdir, label):
    df = imp_df.copy().sort_values('spearman_rho').reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * n + 1.5)))

    for i, row in enumerate(df.itertuples()):
        sig = row.spearman_p_value < 0.05
        color = BLUE if row.spearman_rho >= 0 else RED
        alpha = 0.9 if sig else 0.35
        ax.scatter(row.spearman_rho, i, color=color, s=55, zorder=3, alpha=alpha)
        if sig:
            ax.plot([row.spearman_rho, row.spearman_rho], [i - 0.28, i + 0.28],
                    color=color, lw=1.5, alpha=0.7)

    ax.axvline(0, color='#1e293b', lw=1.0, ls='--', alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty(r) for r in df.feature], fontsize=8.5)
    ax.set_xlabel("Spearman ρ (feature vs survival target)", fontsize=10)
    ax.set_title(f'{label} – Feature–survival correlation\n(faded = p ≥ 0.05)', fontsize=11)

    pos_p = mpatches.Patch(color=BLUE, alpha=0.85, label='positive (higher → better)')
    neg_p = mpatches.Patch(color=RED,  alpha=0.85, label='negative (higher → worse)')
    ax.legend(handles=[pos_p, neg_p], loc='lower right', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(outdir, 'fig2_spearman_forest.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 3 – Predicted vs actual scatter
# ---------------------------------------------------------------------------
def fig_predicted_vs_actual(preds_df, model_summary_df, outdir, label):
    df = preds_df.dropna(subset=['survival_target', 'predicted_survival']).copy()

    r2   = model_summary_df['mean_test_r2'].iloc[0]
    rmse = model_summary_df['mean_test_rmse'].iloc[0]
    corr = model_summary_df['overall_pred_target_corr'].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    sc = ax.scatter(df['survival_target'], df['predicted_survival'],
                    c=df['backbone_error'], cmap='RdYlGn_r',
                    alpha=0.7, s=28, linewidths=0, vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Backbone error (0 = low, 1 = high)', fontsize=9)

    lo = min(df['survival_target'].min(), df['predicted_survival'].min()) - 0.2
    hi = max(df['survival_target'].max(), df['predicted_survival'].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], '--', color=GREY, lw=1.2, label='perfect prediction')

    ax.set_xlabel('Actual survival target (z-scored log-rank)', fontsize=10)
    ax.set_ylabel('Predicted survival (Ridge CV)', fontsize=10)
    ax.set_title(
        f'{label} – Predicted vs Actual\n'
        f'CV R²={r2:.3f}  |  RMSE={rmse:.3f}  |  Pearson r={corr:.3f}',
        fontsize=11
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig3_predicted_vs_actual.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 4 – Error distribution
# ---------------------------------------------------------------------------
def fig_error_distribution(preds_df, error_labels_df, outdir, label):
    df = preds_df.dropna(subset=['patient_error_abs']).copy()

    # merge high/low label if available
    if error_labels_df is not None and 'high_error_label' in error_labels_df.columns:
        df = df.merge(error_labels_df[['patient_id', 'high_error_label', 'backbone_error']],
                      on='patient_id', how='left', suffixes=('', '_lbl'))
    else:
        df['high_error_label'] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # -- Left: histogram of absolute error, split by label
    ax = axes[0]
    hi  = df[df['high_error_label'] == 1]['patient_error_abs']
    lo  = df[df['high_error_label'] == 0]['patient_error_abs']
    unk = df[df['high_error_label'].isna()]['patient_error_abs']
    bins = np.linspace(df['patient_error_abs'].min(), df['patient_error_abs'].max(), 25)
    if len(lo):  ax.hist(lo,  bins=bins, color=GREEN, alpha=0.65, label=f'Low error  (n={len(lo)})')
    if len(hi):  ax.hist(hi,  bins=bins, color=RED,   alpha=0.65, label=f'High error (n={len(hi)})')
    if len(unk): ax.hist(unk, bins=bins, color=GREY,  alpha=0.35, label=f'Unlabelled (n={len(unk)})')
    ax.set_xlabel('|Predicted − Actual| survival', fontsize=10)
    ax.set_ylabel('Number of patients', fontsize=10)
    ax.set_title('Error distribution by label', fontsize=11)
    ax.legend()

    # -- Right: backbone error distribution
    ax2 = axes[1]
    be = df['backbone_error'].dropna()
    ax2.hist(be, bins=25, color=BLUE, alpha=0.75, edgecolor='white', linewidth=0.5)
    q80 = np.quantile(be, 0.8)
    ax2.axvline(q80, color=RED, lw=1.5, ls='--', label=f'80th pct = {q80:.2f}\n(high-error threshold)')
    ax2.set_xlabel('Backbone error (absolute, normalised)', fontsize=10)
    ax2.set_ylabel('Number of patients', fontsize=10)
    ax2.set_title('WSI backbone error distribution', fontsize=11)
    ax2.legend()

    fig.suptitle(f'{label} – Per-patient prediction error', fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig4_error_distribution.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 5 – Top MED3PA profiles by run frequency
# ---------------------------------------------------------------------------
def fig_top_profiles(profiles_df, outdir, label, top_n=20, freq_col='profiles_freq25'):
    df = profiles_df.nlargest(top_n, 'run_freq_pct').copy()
    df = df.iloc[::-1]  # top at top
    n = len(df)

    # colour bars by avg_accuracy
    norm = Normalize(vmin=0.4, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(10, max(5, 0.46 * n + 1.5)))
    for i, row in enumerate(df.itertuples()):
        color = cmap(norm(row.avg_accuracy))
        ax.barh(i, row.run_freq_pct, color=color, height=0.65)
        ax.text(row.run_freq_pct + 0.3, i,
                f'acc={row.avg_accuracy:.2f}  pop%={row.avg_pop_pct:.1f}',
                va='center', fontsize=7.5, color='#1e293b')

    ax.set_yticks(range(n))
    # Use readable_rule if available and short enough, else profile name
    ytick_labels = []
    for row in df.itertuples():
        rule = str(row.readable_rule)
        lbl = rule if len(rule) <= 58 else rule[:55] + '…'
        ytick_labels.append(lbl)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel('% of MED3PA runs profile appeared in', fontsize=10)
    ax.set_title(
        f'{label} – Top {top_n} MED3PA profiles by stability\n'
        f'(colour = classification accuracy; pop% = avg patients in profile)',
        fontsize=11
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label('Avg profile accuracy', fontsize=9)

    plt.tight_layout()
    out = os.path.join(outdir, 'fig5_med3pa_top_profiles.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 6 – Accuracy vs coverage trade-off scatter
# ---------------------------------------------------------------------------
def fig_accuracy_coverage(profiles_df, outdir, label, min_freq_pct=5.0, top_annotate=10):
    df = profiles_df[profiles_df['run_freq_pct'] >= min_freq_pct].copy()
    if df.empty:
        print('  [skip fig6 – no profiles above freq threshold]')
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors_by_cond = {1: BLUE, 2: GOLD}
    default_c = GREY

    sizes = 30 + 200 * (df['run_freq_pct'] / df['run_freq_pct'].max())

    for n_cond, grp in df.groupby('n_conditions'):
        c = colors_by_cond.get(n_cond, default_c)
        ax.scatter(grp['avg_pop_pct'], grp['avg_accuracy'],
                   s=sizes.loc[grp.index], color=c, alpha=0.7,
                   label=f'{n_cond}-condition profile', linewidths=0.5, edgecolors='white')

    # annotate the most frequent
    top = df.nlargest(top_annotate, 'run_freq_pct')
    for _, row in top.iterrows():
        rule = str(row['readable_rule'])
        short = rule[:40] + '…' if len(rule) > 40 else rule
        ax.annotate(short, (row['avg_pop_pct'], row['avg_accuracy']),
                    fontsize=6.5, alpha=0.8,
                    xytext=(5, 3), textcoords='offset points')

    ax.axhline(0.8, color=GREEN, lw=1, ls=':', alpha=0.7, label='accuracy = 0.8')
    ax.set_xlabel('Avg population coverage (%)', fontsize=10)
    ax.set_ylabel('Avg classification accuracy', fontsize=10)
    ax.set_title(
        f'{label} – MED3PA: accuracy vs coverage\n'
        f'(size ∝ stability; only profiles with run freq ≥ {min_freq_pct}%)',
        fontsize=11
    )
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    out = os.path.join(outdir, 'fig6_accuracy_coverage_tradeoff.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Figure 7 – Which features appear most in stable profiles
# ---------------------------------------------------------------------------
def fig_feature_in_profiles(profiles_df, outdir, label, min_freq_pct=5.0):
    df = profiles_df[profiles_df['run_freq_pct'] >= min_freq_pct].copy()
    if df.empty:
        print('  [skip fig7 – no profiles above freq threshold]')
        return

    # extract feature name from profile key (strip _UP / _DOWN suffix from each part)
    feature_counts = {}
    for _, row in df.iterrows():
        parts = str(row['profile']).split(' | ')
        for part in parts:
            feat = re.sub(r'_(UP|DOWN)$', '', part.strip())
            feature_counts[feat] = feature_counts.get(feat, 0) + row['run_freq_pct']

    fc = pd.Series(feature_counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(fc) + 1.5)))
    bars = ax.barh(range(len(fc)), fc.values, color=BLUE, alpha=0.78, height=0.65)
    ax.set_yticks(range(len(fc)))
    ax.set_yticklabels([pretty(f) for f in fc.index], fontsize=9)
    ax.set_xlabel('Cumulative run-frequency % across stable profiles', fontsize=10)
    ax.set_title(
        f'{label} – Feature presence in stable MED3PA profiles\n'
        f'(profiles with run freq ≥ {min_freq_pct}%; stacked across UP/DOWN splits)',
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(outdir, 'fig7_feature_in_profiles.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------
def write_summary(imp_df, model_summary_df, fi_run_df, med3pa_run_df,
                  preds_df, profiles_df, outdir, label):
    lines = [f'=== {label} – summary statistics ===', '']

    r2   = model_summary_df['mean_test_r2'].iloc[0]
    rmse = model_summary_df['mean_test_rmse'].iloc[0]
    corr = model_summary_df['overall_pred_target_corr'].iloc[0]
    alpha= model_summary_df['alpha'].iloc[0]
    nspl = int(model_summary_df['n_splits'].iloc[0])
    lines += [
        '--- Linear model (Ridge, clinical features) ---',
        f'  Alpha = {alpha}  |  CV splits = {nspl}',
        f'  CV R² = {r2:.4f} ± {model_summary_df["std_test_r2"].iloc[0]:.4f}',
        f'  CV RMSE = {rmse:.4f} ± {model_summary_df["std_test_rmse"].iloc[0]:.4f}',
        f'  Overall Pearson r (pred vs target) = {corr:.4f}',
        '',
    ]

    n_boot = fi_run_df['n_bootstrap'].iloc[0]
    oob = fi_run_df['mean_oob_corr'].iloc[0]
    oob_std = fi_run_df['std_oob_corr'].iloc[0]
    lines += [
        '--- Feature importance (RF, bootstrap) ---',
        f'  n_bootstrap = {int(n_boot)}',
        f'  OOB correlation = {oob:.4f} ± {oob_std:.4f}',
        '',
        '  Top 5 features by importance:',
    ]
    for _, r in imp_df.head(5).iterrows():
        lines.append(
            f'    [{r["rank"]:2.0f}] {pretty(r["feature"]):<40s}  '
            f'imp={r["mean_importance"]:.4f} (95%CI {r["ci_lo"]:.4f}–{r["ci_hi"]:.4f})  '
            f'ρ={r["spearman_rho"]:+.3f} p={r["spearman_p_value"]:.4g}  {r["direction"]}'
        )
    lines.append('')

    n_runs = int(med3pa_run_df['n_runs_completed'].iloc[0])
    n_prof = int(med3pa_run_df['n_unique_profiles'].iloc[0])
    q      = float(med3pa_run_df['error_quantile'].iloc[0])
    lines += [
        '--- MED3PA ---',
        f'  Runs completed = {n_runs}',
        f'  Unique profiles found = {n_prof}',
        f'  Error quantile threshold = {q:.2f}',
        '',
    ]

    for freq_thr in [50, 25, 10]:
        sub = profiles_df[profiles_df['run_freq_pct'] >= freq_thr]
        lines.append(
            f'  Profiles appearing in ≥{freq_thr}% of runs: {len(sub)}'
            + (f'  (avg accuracy range: {sub["avg_accuracy"].min():.2f}–{sub["avg_accuracy"].max():.2f})'
               if len(sub) else '')
        )
    lines.append('')

    # most stable profile
    top1 = profiles_df.iloc[0]
    lines += [
        '  Most stable profile:',
        f'    Rule: {top1["readable_rule"]}',
        f'    Appeared in {top1["run_freq_pct"]:.1f}% of runs  |  '
        f'avg pop {top1["avg_pop_pct"]:.1f}%  |  '
        f'avg accuracy {top1["avg_accuracy"]:.3f} ± {top1["std_accuracy"]:.3f}',
        '',
    ]

    # patient error stats
    be = preds_df['backbone_error'].dropna()
    lines += [
        '--- Patient-level errors ---',
        f'  n patients = {len(preds_df)}',
        f'  Backbone error: median={be.median():.3f}  mean={be.mean():.3f}  '
        f'IQR=[{be.quantile(0.25):.3f}, {be.quantile(0.75):.3f}]',
        f'  80th pct threshold = {be.quantile(0.8):.3f}  '
        f'→ {int((be >= be.quantile(0.8)).sum())} high-error patients',
    ]

    out = os.path.join(outdir, 'summary_stats.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Analyse a med3pa pipeline results folder.')
    ap.add_argument('--results-dir', required=True,
                    help='Path to the results folder (e.g. results/blca_clinical_med3pa_500k)')
    ap.add_argument('--outdir', default=None,
                    help='Where to save figures (default: <results-dir>/figures)')
    ap.add_argument('--label', default=None,
                    help='Short label for plot titles (e.g. "BLCA Clinical")')
    ap.add_argument('--top-n-features', type=int, default=15,
                    help='How many top features to show in fig1 (default 15)')
    ap.add_argument('--min-freq-pct', type=float, default=5.0,
                    help='Min run_freq_pct to include profile in figs 6/7 (default 5.0)')
    args = ap.parse_args()

    rdir = args.results_dir
    outdir = args.outdir or os.path.join(rdir, 'figures')
    os.makedirs(outdir, exist_ok=True)

    label = args.label or os.path.basename(rdir.rstrip('/'))

    print(f'\nAnalysing: {rdir}')
    print(f'Output:    {outdir}\n')

    # ---- load files ----
    imp_df          = load_csv(os.path.join(rdir, 'clinical_importance_survival.csv'))
    model_summary   = load_csv(os.path.join(rdir, 'clinical_linear_model_summary.csv'))
    preds_df        = load_csv(os.path.join(rdir, 'clinical_linear_model_patient_predictions.csv'))
    fi_run_df       = load_csv(os.path.join(rdir, 'feature_importance_run_summary.csv'))
    med3pa_run_df   = load_csv(os.path.join(rdir, 'med3pa_run_summary.csv'))
    all_profiles_df = load_csv(os.path.join(rdir, 'all_profiles_dir.csv'))

    error_labels_path = os.path.join(rdir, 'med3pa_backbone_error_labels.csv')
    error_labels_df   = load_csv(error_labels_path) if os.path.exists(error_labels_path) else None

    # prefer freq50 for top-profile figure since it's already pre-filtered & manageable
    for freq_csv in ['profiles_freq50.csv', 'profiles_freq25.csv', 'profiles_freq10.csv']:
        fpath = os.path.join(rdir, freq_csv)
        if os.path.exists(fpath):
            profiles_df = load_csv(fpath)
            break
    else:
        profiles_df = all_profiles_df

    # ---- generate figures ----
    fig_feature_importance(imp_df, outdir, label, top_n=args.top_n_features)
    fig_spearman_forest(imp_df, outdir, label)
    fig_predicted_vs_actual(preds_df, model_summary, outdir, label)
    fig_error_distribution(preds_df, error_labels_df, outdir, label)
    fig_top_profiles(all_profiles_df, outdir, label, top_n=20)
    fig_accuracy_coverage(all_profiles_df, outdir, label, min_freq_pct=args.min_freq_pct)
    fig_feature_in_profiles(all_profiles_df, outdir, label, min_freq_pct=args.min_freq_pct)
    write_summary(imp_df, model_summary, fi_run_df, med3pa_run_df,
                  preds_df, all_profiles_df, outdir, label)

    print(f'\nDone. All figures written to: {outdir}')


if __name__ == '__main__':
    main()
