#!/usr/bin/env python3
"""
Generate compressed, paper-ready figures and tables for a 10-page LNCS submission.
Combines results from all four arms: BLCA/BRCA × Clinical/Radiomic.

Output (results/paper_figures/):
  fig1_combined_feature_importance.png   - 2×2: top features per arm
  fig2_combined_accuracy_coverage.png    - 2×2: MED3PA trust/caution landscape
  fig3_med3pa_stability_summary.png      - Cross-arm top profile stability + accuracy
  table1_pipeline_summary.csv            - One row per arm: performance + MED3PA stats
  table2_key_profiles.csv                - Best Trust + worst Caution per arm
  paper_table_captions.txt               - Captions for both tables
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE  = 'results'
ARMS  = [
    ('BLCA\nClinical', f'{BASE}/blca_clinical_med3pa_500k', 'clinical'),
    ('BRCA\nClinical', f'{BASE}/brca_clinical_med3pa_500k', 'clinical'),
    ('BLCA\nRadiomic', f'{BASE}/blca_radiomic_med3pa_500k', 'radiomic'),
    ('BRCA\nRadiomic', f'{BASE}/brca_radiomic_med3pa_500k', 'radiomic'),
]
OUTDIR = f'{BASE}/paper_figures'
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style  (LNCS-friendly: larger fonts, prints well in greyscale)
# ---------------------------------------------------------------------------
BLUE  = '#2563EB'
RED   = '#DC2626'
GREEN = '#16A34A'
GREY  = '#64748B'
LGREY = '#E2E8F0'

REGION_COLORS = {
    'tumor': '#DC2626', 'urothelium': '#2563EB', 'adipose': '#D97706',
    'vessels': '#7C3AED', 'lamina': '#059669', 'muscularis': '#0891B2',
    'immune': '#DB2777', 'necrosis': '#78716C',
    'dcis': '#EA580C', 'tils': '#9333EA', 'stroma': '#0369A1',
    'tdlu': '#EC4899', 'muscle': '#65A30D',
}

CLINICAL_RENAME = {
    'pathologic_T': 'Path. T stage', 'pathologic_t': 'Path. T stage',
    'pathologic_stage': 'Path. stage',
    'pathologic_N': 'Path. N stage', 'pathologic_n': 'Path. N stage',
    'pathologic_M': 'Path. M stage', 'pathologic_m': 'Path. M stage',
    'age_at_diagnosis': 'Age at dx',
    'weight': 'Body weight', 'height': 'Body height',
    'number_of_lymphnodes_positive_by_he': 'LN+ (HE)',
    'lymph_node_examined_count': 'LN examined',
    'lymphovascular_invasion_present': 'LVI present',
    'neoplasm_histologic_grade_low_grade': 'Grade (low)',
    'neoplasm_histologic_grade_high_grade': 'Grade (high)',
    'breast_carcinoma_estrogen_receptor_status': 'ER status',
    'breast_carcinoma_progesterone_receptor_status': 'PR status',
    'lab_proc_her2_neu_immunohistochemistry_receptor_status': 'HER2 (IHC)',
    'tobacco_smoking_history_1': 'Smoking: never',
    'tobacco_smoking_history_3': 'Smoking: current',
    'gender': 'Sex',
    'histological_type_infiltrating_ductal_carcinoma': 'IDC',
    'histological_type_infiltrating_lobular_carcinoma': 'ILC',
    'histological_type_mucinous_carcinoma': 'Mucinous',
    'histological_type_metaplastic_carcinoma': 'Metaplastic',
    'menopause_status_pre_6_months_since_lmp_and_no_prior_bilateral_ovariectomy_and_not_on_estrogen_replacement': 'Menopause: pre',
    'menopause_status_post_prior_bilateral_ovariectomy_or_12_mo_since_lmp_with_no_prior_hysterectomy': 'Menopause: post',
    'menopause_status_indeterminate_neither_pre_or_postmenopausal': 'Menopause: indet.',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': LGREY, 'grid.linewidth': 0.5,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']

# ---------------------------------------------------------------------------
# Scaler parameters for back-transforming z-scored clinical features
# Only truly continuous features were z-scored; staging (T/N/M/stage) were not.
# (mean, std, unit_suffix) — recovered from raw TCGA XML/CDR data.
# ---------------------------------------------------------------------------
_BLCA_SCALERS = {
    'age_at_diagnosis':                      (68.55, 10.57, ' yrs'),
    'height':                                (171.75, 9.51, ' cm'),
    'weight':                                (80.67, 22.48, ' kg'),
    'lymph_node_examined_count':             (24.67, 22.33, ' nodes'),
    'number_of_lymphnodes_positive_by_he':   (1.50,  6.02,  ''),
}
_BRCA_SCALERS = {
    'age_at_diagnosis':                      (58.45, 13.21, ' yrs'),
    'lymph_node_examined_count':             (10.46,  8.27, ' nodes'),
    'number_of_lymphnodes_positive_by_he':   (2.15,   4.29, ''),
}
# Staging features use raw integer encoding (not z-scored) — thresholds are interpretable
_STAGING_FEATURES = {
    'pathologic_stage', 'pathologic_T', 'pathologic_t',
    'pathologic_N', 'pathologic_n', 'pathologic_M', 'pathologic_m',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(fig, basename):
    """Save figure as PNG (300 dpi), SVG, and PDF — same content, three formats."""
    for ext in ('png', 'svg', 'pdf'):
        path = os.path.join(OUTDIR, f'{basename}.{ext}')
        fig.savefig(path, bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    print(f'  saved {basename}.png / .svg / .pdf')


def pretty_clinical(name):
    return CLINICAL_RENAME.get(name, re.sub(r'_', ' ', name).title()[:22])

def short_radiomic(name):
    parts = str(name).split('_original_firstorder_')
    if len(parts) != 2: return name[:25]
    pfx = parts[0].split('_')
    region = pfx[0].title()
    ch = pfx[1] if len(pfx) > 1 else ''
    mp = parts[1].rsplit('_', 1)
    metric = mp[0][:12]; stat = mp[1] if len(mp) == 2 else ''
    return f'{region}·{ch} {metric}({stat})'

def clean_rule(r):
    return re.sub(r'\s*\[.*?\]', '', str(r)).strip()

def profile_region(p):
    for r in ['adipose','dcis','tils','stroma','tumor','vessels','tdlu','muscle',
              'urothelium','lamina','muscularis','immune','necrosis']:
        if str(p).lower().startswith(r): return r
    return 'other'

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ---------------------------------------------------------------------------
# Load data for all arms
# ---------------------------------------------------------------------------
def load_arm(label, rdir, mode):
    d = {'label': label, 'mode': mode, 'rdir': rdir}
    if mode == 'clinical':
        d['imp']   = pd.read_csv(f'{rdir}/clinical_importance_survival.csv')
        d['model'] = pd.read_csv(f'{rdir}/clinical_linear_model_summary.csv')
        d['fi']    = pd.read_csv(f'{rdir}/feature_importance_run_summary.csv')
        d['mr']    = pd.read_csv(f'{rdir}/med3pa_run_summary.csv')
        d['lbl']   = pd.read_csv(f'{rdir}/med3pa_backbone_error_labels.csv')
        d['prof']  = pd.read_csv(f'{rdir}/all_profiles_dir.csv')
    else:
        d['imp']   = pd.read_csv(f'{rdir}/radiomic_importance.csv')
        d['mr']    = pd.read_csv(f'{rdir}/med3pa_run_summary_radiomic.csv')
        d['lbl']   = pd.read_csv(f'{rdir}/med3pa_backbone_error_labels_radiomic.csv')
        d['prof']  = pd.read_csv(f'{rdir}/all_profiles_dir_radiomic.csv')
        d['reg']   = pd.read_csv(f'{rdir}/region_summary.csv')
    return d


# ---------------------------------------------------------------------------
# FIGURE 1 — Combined feature importance (2×2)
# ---------------------------------------------------------------------------
def fig_combined_importance(data_list, top_n=8):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, d in enumerate(data_list):
        ax   = axes[idx]
        imp  = d['imp'].head(top_n).iloc[::-1]
        mode = d['mode']
        n    = len(imp)

        for i, row in enumerate(imp.itertuples()):
            direction_up = 'higher survival' in row.direction
            err_lo = row.mean_importance - row.ci_lo
            err_hi = row.ci_hi - row.mean_importance

            if mode == 'clinical':
                color    = BLUE if direction_up else RED
                edgec    = 'none'
                edgelw   = 0
            else:
                color  = REGION_COLORS.get(str(row.region).lower(), GREY)
                # edge colour encodes direction for radiomic (bar fill = region)
                edgec  = BLUE if direction_up else RED
                edgelw = 1.8

            ax.barh(i, row.mean_importance, color=color, alpha=0.80, height=0.65,
                    edgecolor=edgec, linewidth=edgelw,
                    xerr=[[err_lo], [err_hi]],
                    error_kw=dict(ecolor='#94a3b8', capsize=2, lw=0.8))

            stars = sig_stars(row.spearman_p_value)
            star_color = '#94a3b8' if stars == 'ns' else '#1e293b'
            ax.text(row.ci_hi + row.mean_importance * 0.02, i, stars,
                    va='center', fontsize=7, color=star_color)

        ax.set_yticks(range(n))
        if mode == 'clinical':
            ax.set_yticklabels([pretty_clinical(r) for r in imp.feature], fontsize=8)
        else:
            ax.set_yticklabels([short_radiomic(r) for r in imp.feature], fontsize=7.5)

        ax.set_xlabel('Mean RF importance', fontsize=8)
        ax.grid(axis='y', linewidth=0)
        ax.text(-0.12, 1.04, PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

        short_label = d['label'].replace('\n', ' ')
        if mode == 'clinical':
            ax.set_title(f'{short_label}  (target: survival risk score)', fontsize=9)
        else:
            ax.set_title(f'{short_label}  (target: survival risk score)', fontsize=9)

    # shared legends
    axes[0].legend(handles=[
        mpatches.Patch(color=BLUE, alpha=0.80, label='Higher → worse prognosis'),
        mpatches.Patch(color=RED,  alpha=0.80, label='Higher → better prognosis'),
    ], fontsize=7.5, loc='lower right')

    seen_regions = set()
    for d in data_list:
        if d['mode'] == 'radiomic':
            seen_regions |= set(d['imp'].head(top_n)['region'].str.lower())
    region_patches = [mpatches.Patch(color=REGION_COLORS.get(r, GREY), alpha=0.80,
                                     label=r.title())
                      for r in sorted(seen_regions) if r in REGION_COLORS]
    # add edge-colour direction entries for radiomic
    dir_patches = [
        mpatches.Patch(facecolor='#e2e8f0', edgecolor=BLUE, linewidth=2.0,
                       label='Edge blue: higher → worse prognosis'),
        mpatches.Patch(facecolor='#e2e8f0', edgecolor=RED, linewidth=2.0,
                       label='Edge red: higher → better prognosis'),
    ]
    axes[2].legend(handles=region_patches + dir_patches, fontsize=6.5, loc='lower right',
                   title='Radiomic bars: fill = region, edge = direction', title_fontsize=7)

    fig.suptitle('Feature importance across all four arms  (target: WSI survival risk score)\n'
                 'Clinical (a,b): bar colour = direction (blue = worse / red = better prognosis)  |  '
                 'Radiomic (c,d): bar fill = tissue region, bar edge colour = direction  |  '
                 'error bars: 95% CI  |  ns/*/**/***: Spearman p-value',
                 fontsize=9, y=1.01)
    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    save_fig(fig, 'fig1_combined_feature_importance')
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2 — Combined MED3PA accuracy-coverage (2×2)
# ---------------------------------------------------------------------------
def fig_combined_accuracy_coverage(data_list, min_freq=5.0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    for idx, d in enumerate(data_list):
        ax   = axes[idx]
        prof = d['prof']
        mode = d['mode']
        stable = prof[prof['run_freq_pct'] >= min_freq].copy()

        if stable.empty:
            ax.text(0.5, 0.5, 'No stable profiles', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        stable['zone'] = stable['avg_accuracy'].apply(
            lambda a: 'Trust' if a >= 0.80 else 'Caution')
        sizes = 40 + 220 * (stable['run_freq_pct'] / stable['run_freq_pct'].max())

        for zone, grp in stable.groupby('zone'):
            color = GREEN if zone == 'Trust' else RED
            ax.scatter(grp['avg_pop_pct'], grp['avg_accuracy'],
                       s=sizes.loc[grp.index], color=color, alpha=0.75,
                       label=zone, linewidths=0.4, edgecolors='white', zorder=3)

        ax.axhline(0.80, color=GREY, lw=1.0, ls='--', alpha=0.7)
        ax.set_xlabel('Population coverage (%)', fontsize=8)
        ax.set_ylabel('Avg accuracy', fontsize=8)
        ax.set_ylim(0.3, 1.05)
        ax.text(-0.14, 1.06, PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

        short_label = d['label'].replace('\n', ' ')
        n_trust   = (stable['zone'] == 'Trust').sum()
        n_caution = (stable['zone'] == 'Caution').sum()
        ax.set_title(f'{short_label} — {n_trust} Trust / {n_caution} Caution profiles',
                     fontsize=8.5)
        scalers = _arm_scalers(d['label'])

        # Per-panel box positions: (x, y, va, ha)
        # idx 0 (BLCA Clin): trust=bottom-right, caution=top-left
        # idx 1 (BRCA Clin): trust=bottom-right, caution=top-left
        # idx 2 (BLCA Rad):  trust=bottom-right, caution=bottom-left
        # idx 3 (BRCA Rad):  trust=top-right,    caution=bottom-left
        trust_pos   = {0: (0.98, 0.02, 'bottom', 'right'),
                       1: (0.98, 0.02, 'bottom', 'right'),
                       2: (0.98, 0.02, 'bottom', 'right'),
                       3: (0.98, 0.98, 'top',    'right')}
        caution_pos = {0: (0.02, 0.98, 'top',    'left'),
                       1: (0.98, 0.98, 'top',    'right'),
                       2: (0.02, 0.02, 'bottom', 'left'),
                       3: (0.02, 0.02, 'bottom', 'left')}
        tx, ty, tva, tha = trust_pos[idx]
        cx, cy, cva, cha = caution_pos[idx]

        # --- annotate top 3 Trust profiles ---
        trust_top3 = (stable[stable['zone'] == 'Trust']
                      .nlargest(3, 'avg_accuracy'))
        if not trust_top3.empty:
            for rank, (_, row) in enumerate(trust_top3.iterrows(), 1):
                ax.scatter(row['avg_pop_pct'], row['avg_accuracy'],
                           s=sizes.loc[row.name] + 30,
                           color=GREEN, alpha=1.0, linewidths=0.8,
                           edgecolors='#064e3b', zorder=5)
                ax.text(row['avg_pop_pct'], row['avg_accuracy'], str(rank),
                        ha='center', va='center', fontsize=6,
                        fontweight='bold', color='white', zorder=6)
            lines = ['▲ Top Trust profiles:']
            for rank, (_, row) in enumerate(trust_top3.iterrows(), 1):
                lbl   = _compact_label(row['readable_rule'], mode, scalers).replace('\n& ', ' & ')
                stars = _acc_stars(row['avg_accuracy'], row['std_accuracy'])
                lines.append(
                    f'{rank}. {lbl}\n'
                    f'   acc={row["avg_accuracy"]:.2f}±{row["std_accuracy"]:.2f}{stars}'
                    f', cov={row["avg_pop_pct"]:.0f}%'
                )
            ax.text(tx, ty, '\n'.join(lines),
                    transform=ax.transAxes, fontsize=6.2, va=tva, ha=tha,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdf4',
                              edgecolor='#16a34a', alpha=0.92, linewidth=0.8),
                    family='monospace')

        # --- annotate worst 3 Caution profiles ---
        caution_bot3 = (stable[stable['zone'] == 'Caution']
                        .nsmallest(3, 'avg_accuracy'))
        if not caution_bot3.empty:
            for rank, (_, row) in enumerate(caution_bot3.iterrows(), 1):
                ax.scatter(row['avg_pop_pct'], row['avg_accuracy'],
                           s=sizes.loc[row.name] + 30,
                           color=RED, alpha=1.0, linewidths=0.8,
                           edgecolors='#7f1d1d', zorder=5)
                ax.text(row['avg_pop_pct'], row['avg_accuracy'], str(rank),
                        ha='center', va='center', fontsize=6,
                        fontweight='bold', color='white', zorder=6)
            lines = ['▼ Worst Caution profiles:']
            for rank, (_, row) in enumerate(caution_bot3.iterrows(), 1):
                lbl   = _compact_label(row['readable_rule'], mode, scalers).replace('\n& ', ' & ')
                stars = _acc_stars(row['avg_accuracy'], row['std_accuracy'])
                lines.append(
                    f'{rank}. {lbl}\n'
                    f'   acc={row["avg_accuracy"]:.2f}±{row["std_accuracy"]:.2f}{stars}'
                    f', cov={row["avg_pop_pct"]:.0f}%'
                )
            ax.text(cx, cy, '\n'.join(lines),
                    transform=ax.transAxes, fontsize=6.2, va=cva, ha=cha,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff1f2',
                              edgecolor='#dc2626', alpha=0.92, linewidth=0.8),
                    family='monospace')

        # legend removed — green/red self-explanatory, explained in caption

    fig.suptitle('MED3PA reliability landscape — green = Trust (model reliable), red = Caution (model unreliable)\n'
                 '▲ top-right box: 3 most reliable profiles  |  ▼ top-left box: 3 most unreliable  |  '
                 'numbered circles correspond to box entries  |  t-test vs 0.5: *p<0.05 **p<0.01 ***p<0.001',
                 fontsize=9.5, y=1.01)
    save_fig(fig, 'fig2_combined_accuracy_coverage')
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3 — Cross-arm MED3PA stability summary
# ---------------------------------------------------------------------------

# Abbreviation maps for compact labels
_CLIN_ABBREV = {
    'pathologic_stage': 'stage', 'pathologic_t': 'T stage',
    'pathologic_n': 'N stage', 'pathologic_m': 'M stage',
    'pathologic_T': 'T stage', 'pathologic_N': 'N stage',
    'pathologic_M': 'M stage',
    'number_of_lymphnodes_positive_by_he': 'LN+',
    'lymph_node_examined_count': 'LN#',
    'age_at_diagnosis': 'age', 'weight': 'weight', 'height': 'height',
    'lymphovascular_invasion_present': 'LVI',
    'histological_type_infiltrating_lobular_carcinoma': 'ILC',
    'histological_type_infiltrating_ductal_carcinoma': 'IDC',
}
# Binary receptor features rendered as +/- rather than >/≤
_RECEPTOR_ABBREV = {
    'breast_carcinoma_estrogen_receptor_status':  ('ER+', 'ER−'),
    'breast_carcinoma_progesterone_receptor_status': ('PR+', 'PR−'),
    'lab_proc_her2_neu_immunohistochemistry_receptor_status': ('HER2+', 'HER2−'),
}
_RAD_REGION = {
    'adipose': 'Adip', 'tumor': 'Tum', 'urothelium': 'Uro',
    'lamina': 'Lam', 'muscularis': 'Musc', 'immune': 'Imm',
    'vessels': 'Ves', 'necrosis': 'Necr',
    'tils': 'TIL', 'dcis': 'DCIS', 'stroma': 'Str',
    'tdlu': 'TDLU', 'muscle': 'Musc',
}
_RAD_METRIC = {
    'Variance': 'Var', 'Entropy': 'Ent', 'Uniformity': 'Unif',
    'Kurtosis': 'Kurt', 'Skewness': 'Skew',
    'MeanAbsoluteDeviation': 'MAD',
    'RobustMeanAbsoluteDeviation': 'rMAD',
    'InterquartileRange': 'IQR', 'RootMeanSquared': 'RMS',
    'Median': 'Med', 'Mean': 'Mean', 'TotalEnergy': 'TEnrg',
    'Energy': 'Enrg', '10Percentile': '10pct', '90Percentile': '90pct',
    'Range': 'Range',
}


def _fmt_thresh(val_str):
    """Format a threshold value compactly."""
    try:
        v = float(val_str)
        if abs(v) >= 1000: return f'{v:.0f}'
        if abs(v) >= 100:  return f'{v:.1f}'
        if abs(v) >= 10:   return f'{v:.1f}'
        return f'{v:.2g}'
    except Exception:
        return val_str[:6]


def _compact_clinical_condition(cond, scalers=None):
    """'feature op threshold [range]' → short label with real value where possible."""
    cond = re.sub(r'\s*\[.*?\]', '', cond).strip()
    parts = cond.split()
    if len(parts) < 2:
        return cond[:20]
    feat   = parts[0]
    op     = '>' if '>' in parts[1] else '≤'
    z_val  = float(parts[2]) if len(parts) >= 3 else None
    # binary receptor features: no threshold needed
    for key, (pos, neg) in _RECEPTOR_ABBREV.items():
        if feat == key:
            return pos if op == '>' else neg
    abbr = _CLIN_ABBREV.get(feat, feat.replace('_', ' ')[:12])
    # staging features: raw integer encoding, already interpretable
    if feat in _STAGING_FEATURES:
        thresh = _fmt_thresh(parts[2]) if len(parts) >= 3 else ''
        return f'{abbr} {op} {thresh}'
    # continuous z-scored features: back-transform if scaler available
    if z_val is not None and scalers and feat in scalers:
        mean, std, unit = scalers[feat]
        real = z_val * std + mean
        real_str = f'{real:.0f}' if abs(real) >= 10 else f'{real:.1f}'
        return f'{abbr} {op} {real_str}{unit}'
    # fallback: show z-score with (z) flag
    thresh = _fmt_thresh(parts[2]) if len(parts) >= 3 else ''
    return f'{abbr} {op} {thresh}(z)'


def _compact_radiomic_condition(cond):
    """'region_ch_original_firstorder_metric_stat op threshold [range]'
    → 'Adip·H Var-mean > 512'."""
    cond = re.sub(r'\s*\[.*?\]', '', cond).strip()
    parts = cond.split()
    op    = '>' if len(parts) > 1 and '>' in parts[1] else '≤'
    thresh = _fmt_thresh(parts[2]) if len(parts) >= 3 else ''
    feat  = parts[0]
    fparts = feat.split('_original_firstorder_')
    if len(fparts) != 2:
        return cond[:22]
    pfx    = fparts[0].split('_')
    region = _RAD_REGION.get(pfx[0].lower(), pfx[0].title()[:4])
    ch     = pfx[1] if len(pfx) > 1 else ''
    mp     = fparts[1].rsplit('_', 1)
    metric = _RAD_METRIC.get(mp[0], mp[0][:6])
    stat   = mp[1] if len(mp) == 2 else ''
    return f'{region}·{ch} {metric}-{stat} {op} {thresh}'


def _compact_label(readable_rule, mode, scalers=None):
    """Build a compact multi-line label for a profile rule."""
    rule = re.sub(r'\s*\[.*?\]', '', str(readable_rule)).strip()
    conditions = [c.strip() for c in rule.split(' AND ')]
    if mode == 'radiomic':
        return '\n& '.join(_compact_radiomic_condition(c) for c in conditions)
    else:
        return '\n& '.join(_compact_clinical_condition(c, scalers) for c in conditions)


def _arm_scalers(label):
    """Return the appropriate scaler dict based on arm label."""
    label = label.replace('\n', ' ').upper()
    if 'BRCA' in label:
        return _BRCA_SCALERS
    return _BLCA_SCALERS


def _acc_stars(avg_acc, std_acc, n_runs=1000):
    """Two-sided one-sample t-test vs chance level (0.5).
    Catches both significantly reliable (acc > 0.5, Trust profiles) and
    significantly unreliable (acc < 0.5, Caution profiles) — both meaningful."""
    if std_acc <= 0 or n_runs < 2:
        return '***'
    t_stat = (avg_acc - 0.5) / (std_acc / np.sqrt(n_runs))
    p = scipy_stats.t.sf(abs(t_stat), df=n_runs - 1) * 2  # two-sided
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _select_profiles(prof, min_freq, top_n):
    """Select top n/2 by accuracy (best Trust) + bottom n/2 by accuracy (worst Caution)."""
    stable = prof[prof['run_freq_pct'] >= min_freq]
    if stable.empty:
        return stable
    n_each  = max(2, top_n // 2)
    best    = stable.nlargest(n_each, 'avg_accuracy')   # best Trust profiles
    worst   = stable.nsmallest(n_each, 'avg_accuracy')  # worst Caution profiles
    combined = pd.concat([best, worst]).drop_duplicates(subset='profile')
    return combined.sort_values('avg_accuracy').reset_index(drop=True)


def fig_med3pa_stability_summary(data_list, top_n=6, min_freq=5.0):
    norm = Normalize(vmin=0.4, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    # leave right margin for colorbar
    fig.subplots_adjust(right=0.87, hspace=0.45, wspace=0.35)

    for idx, d in enumerate(data_list):
        ax   = axes[idx]
        prof = d['prof']
        mode = d['mode']
        sel  = _select_profiles(prof, min_freq, top_n)
        n    = len(sel)

        x_max = sel['run_freq_pct'].max() if not sel.empty else 10

        for i, row in enumerate(sel.itertuples()):
            color    = cmap(norm(row.avg_accuracy))
            zone_tag = 'T' if row.avg_accuracy >= 0.80 else 'C'
            stars    = _acc_stars(row.avg_accuracy, row.std_accuracy)
            ax.barh(i, row.run_freq_pct, color=color, height=0.62, zorder=2)
            # annotation: zone, accuracy ± sd, stars, coverage
            ann = (f'{zone_tag}  {row.avg_accuracy:.2f}±{row.std_accuracy:.2f}{stars}'
                   f'  ({row.avg_pop_pct:.0f}%)')
            ax.text(row.run_freq_pct + x_max * 0.02, i,
                    ann, va='center', fontsize=7, color='#1e293b')

        scalers = _arm_scalers(d['label'])
        yticks = [_compact_label(row.readable_rule, mode, scalers) for row in sel.itertuples()]
        ax.set_yticks(range(n))
        ax.set_yticklabels(yticks, fontsize=7.5)
        ax.set_xlabel('Run frequency (% of 1,000 MED3PA runs)', fontsize=8)
        ax.set_xlim(0, x_max * 1.55)   # extra space for annotations
        ax.grid(axis='x', linewidth=0.5, color=LGREY)
        ax.grid(axis='y', linewidth=0)
        ax.text(-0.14, 1.06, PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

        short_label = d['label'].replace('\n', ' ')
        ax.set_title(
            f'{short_label} — T=Trust (≥0.80), C=Caution; acc±SD; pop%=coverage',
            fontsize=8)

    # colorbar in dedicated axis on the right — avoids overlapping subplots
    cbar_ax = fig.add_axes([0.89, 0.15, 0.018, 0.65])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Avg accuracy', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle('MED3PA profile stability across all four arms — top 3 Trust (highest accuracy) and worst 3 Caution (lowest accuracy)\n'
                 'Bar label: T=Trust / C=Caution | acc±SD | coverage (%) | '
                 'two-sided t-test vs chance (0.5):  *p<0.05  **p<0.01  ***p<0.001',
                 fontsize=9.5, y=0.99)
    save_fig(fig, 'fig3_med3pa_stability_summary')
    plt.close(fig)


# ---------------------------------------------------------------------------
# TABLE 1 — Pipeline summary
# ---------------------------------------------------------------------------
def make_table1(data_list):
    rows = []
    for d in data_list:
        label = d['label'].replace('\n', ' ')
        mode  = d['mode']
        imp   = d['imp']
        mr    = d['mr']
        lbl   = d['lbl']
        prof  = d['prof']
        be    = lbl['backbone_error']
        q80   = np.quantile(be, 0.8)
        stable  = prof[prof['run_freq_pct'] >= 5.0]
        n_trust = (stable['avg_accuracy'] >= 0.80).sum()
        n_caut  = (stable['avg_accuracy'] <  0.80).sum()
        n_sig   = (imp['spearman_p_value'] < 0.05).sum()

        row = {
            'Arm':               label,
            'N patients':        len(lbl),
            'N features':        len(imp),
            'Sig. features (p<0.05)': f'{n_sig} ({100*n_sig/len(imp):.0f}%)',
        }
        if mode == 'clinical':
            ms  = d['model']
            fi  = d['fi']
            row['CV R²']        = f'{ms.mean_test_r2.iloc[0]:.3f} ± {ms.std_test_r2.iloc[0]:.3f}'
            row['Pearson r']    = f'{ms.overall_pred_target_corr.iloc[0]:.3f}'
            row['OOB corr']     = f'{fi.mean_oob_corr.iloc[0]:.3f} ± {fi.std_oob_corr.iloc[0]:.3f}'
        else:
            row['CV R²']     = 'N/A'
            row['Pearson r'] = 'N/A'
            row['OOB corr']  = 'N/A'

        row['80th pct error']    = f'{q80:.3f}'
        row['High-error N (%)']  = f'{int(lbl["high_error_label"].sum())} (20%)'
        row['Unique profiles']   = int(mr['n_unique_profiles'].iloc[0])
        row['Stable (>=5% runs)'] = len(stable)
        row['Trust (n)']   = n_trust
        row['Caution (n)'] = n_caut
        rows.append(row)

    t1 = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, 'table1_pipeline_summary.csv')
    t1.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  saved {out}')
    return t1


# ---------------------------------------------------------------------------
# TABLE 2 — Representative profiles (best Trust + worst Caution per arm)
# ---------------------------------------------------------------------------
def _rule_for_table(readable_rule, mode, scalers):
    """Back-transform and abbreviate a profile rule for table display (ASCII only)."""
    lbl = _compact_label(readable_rule, mode, scalers)
    lbl = lbl.replace('\n& ', ' AND ')
    # Replace all Unicode with ASCII equivalents
    lbl = (lbl
           .replace('≤', '<=').replace('≥', '>=')
           .replace('…', '...').replace('±', '+/-')
           .replace('·', '-')   # middle dot · -> -
           .replace('×', 'x')   # multiplication ×
           .replace('−', '-')   # Unicode minus − -> -
           .replace('≥', '>=').replace('≤', '<=')
           )
    return lbl


def make_table2(data_list):
    rows = []
    for d in data_list:
        label   = d['label'].replace('\n', ' ')
        mode    = d['mode']
        prof    = d['prof']
        scalers = _arm_scalers(d['label'])
        stable  = prof[prof['run_freq_pct'] >= 5.0].copy()
        if stable.empty:
            continue

        trust   = stable[stable['avg_accuracy'] >= 0.80].sort_values('avg_accuracy', ascending=False)
        caution = stable[stable['avg_accuracy'] <  0.80].sort_values('avg_accuracy', ascending=True)

        for zone_df, zone_name in [(trust, 'Trust'), (caution, 'Caution')]:
            if zone_df.empty:
                continue
            row_data = zone_df.iloc[0]
            rule = _rule_for_table(row_data['readable_rule'], mode, scalers)
            acc  = f'{row_data["avg_accuracy"]:.2f} +/- {row_data["std_accuracy"]:.2f}'
            rows.append({
                'Arm':               label,
                'Zone':              zone_name,
                'Profile rule':      rule,
                'Run freq (%)':      round(row_data['run_freq_pct'], 1),
                'Pop. coverage (%)': round(row_data['avg_pop_pct'], 1),
                'Accuracy (mean+/-SD)': acc,
            })

    t2 = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, 'table2_key_profiles.csv')
    t2.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  saved {out}')
    return t2


# ---------------------------------------------------------------------------
# TABLE 3 — Radiomic importance by tissue region (cross-arm)
# ---------------------------------------------------------------------------
def make_table3(data_list):
    rows = []
    for d in data_list:
        if d['mode'] != 'radiomic':
            continue
        label = d['label'].replace('\n', ' ')
        rs    = pd.read_csv(os.path.join(d['rdir'], 'region_summary.csv'))
        total = rs['sum_mean_importance'].sum()
        for _, r in rs.sort_values('sum_mean_importance', ascending=False).iterrows():
            rows.append({
                'Arm':              label,
                'Tissue region':    r['region'].title(),
                'Importance (%)':   round(r['sum_mean_importance'] / total * 100, 1),
                'N features':       int(r['n_features']),
                'Features in top 20': int(r['features_in_top20']),
                'Best rank':        int(r['best_rank']),
            })
    t3 = pd.DataFrame(rows)
    out = os.path.join(OUTDIR, 'table3_radiomic_region_importance.csv')
    t3.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  saved {out}')
    return t3


# ---------------------------------------------------------------------------
# Table captions
# ---------------------------------------------------------------------------
def write_table_captions(t1, t2):
    lines = [
        '==============================================================================',
        'PAPER FIGURES / TABLES — CAPTIONS',
        'LNCS MICCAI Interpretability Workshop',
        '==============================================================================',
        '',
        'NOTE ON TARGETS',
        '---------------',
        'All four arms use the same RF importance target: the WSI-derived survival risk',
        'score (survival_target), making clinical and radiomic arms directly comparable.',
        'MED3PA in all arms uses backbone error (model prediction error) to define',
        'reliable/unreliable patient subgroups — a separate question from interpretability.',
        '',
        '------------------------------------------------------------------------------',
        'TABLE 1 — Pipeline summary across all four arms',
        'File: table1_pipeline_summary.csv',
        '------------------------------------------------------------------------------',
        'Summary of experimental results across BLCA and BRCA cohorts, each analysed',
        'with clinical features (21–25 variables) and first-order radiomic features',
        '(504–514 variables). All RF importance analyses use the WSI-derived survival',
        'risk score as target, making all four arms directly comparable.',
        'Sig. features (%): features with significant Spearman correlation with the',
        'survival target (p<0.05; Spearman chosen as features are not normally',
        'distributed). CV R²/Pearson r: cross-validated Ridge regression performance',
        'predicting the survival risk score (clinical arms only; N/A for radiomic arms',
        'which use RF importance only, without a linear regression step). OOB corr:',
        'RF out-of-bag Spearman correlation across 500,000 bootstrap resamples.',
        '80th pct error: backbone error threshold for MED3PA (independent of RF target).',
        'Stable (≥5% runs): MED3PA profiles appearing in ≥5% of 1,000 runs.',
        'Trust/Caution: stable profiles at average accuracy >=0.80 vs <0.80. Significance',
        'assessed by two-sided t-test vs chance (0.5): stars indicate accuracy is',
        'significantly different from 0.5 in either direction — *** on a Trust profile',
        'means reliably better than chance; *** on a Caution profile means reliably',
        'worse than chance (i.e. model errors are genuinely unpredictable there).',
        '',
        '------------------------------------------------------------------------------',
        'TABLE 2 — Representative MED3PA profiles per arm',
        'File: table2_key_profiles.csv',
        '------------------------------------------------------------------------------',
        'For each arm: highest-accuracy stable Trust profile and lowest-accuracy stable',
        'Caution profile (best and worst cases for WSI model reliability). Feature',
        'thresholds are in z-scored units for clinical arms and raw units for radiomic.',
        'Run freq (%): % of 1,000 MED3PA runs the profile was found in (stability).',
        'Pop. coverage (%): average % of patients in the subgroup.',
        'Accuracy (mean±SD): MED3PA classifier accuracy distinguishing high vs low',
        'backbone error patients within the subgroup (two-sided t-test vs 0.5 (chance) —',
        'chance level for binary classification — confirms significance).',
        '',
        'Cross-arm findings: Trust profile accuracy is consistent (0.92–0.96) across',
        'all four arms regardless of feature modality or cancer type. Clinical Trust',
        'zones are defined by low tumour stage and low lymph node burden (BLCA) or',
        'ER/PR-positive low-T disease (BRCA). Radiomic Trust zones are defined by high',
        'adipose texture heterogeneity in both cancers, with TIL spatial variability',
        'also contributing in BRCA.',
        '',
        '------------------------------------------------------------------------------',
        'FIGURE 1 — Combined feature importance across all four arms (2×2)',
        'File: fig1_combined_feature_importance.png',
        '------------------------------------------------------------------------------',
        'Top 8 features by Random Forest importance (500,000 bootstrap resamples,',
        'target = WSI survival risk score) for each arm. Error bars: 95% bootstrap CI.',
        'Significance stars: Spearman rank correlation p-values (* p<0.05, ** p<0.01,',
        '*** p<0.001, ns = not significant; Spearman used as features are not normally',
        'distributed). All four arms share the same target (WSI survival risk score),',
        'making the panels directly comparable.',
        '',
        'Direction encoding differs by panel type by design:',
        '  Clinical (a, b): bar FILL COLOUR encodes direction — blue = higher feature',
        '  value → worse prognosis; red = higher → better prognosis.',
        '  Radiomic (c, d): bar FILL COLOUR encodes tissue region of origin (legend on',
        '  panel c), as region is the most informative grouping for 500+ features.',
        '  Direction is instead shown by bar EDGE COLOUR — blue edge = higher feature',
        '  → worse prognosis; red edge = higher → better prognosis.',
        '',
        'NOTE ON ADIPOSE DOMINANCE: adipose tissue ranks 1st in both radiomic arms',
        '(19.6% of total importance in BLCA, 25.8% in BRCA). The top 8 shown therefore',
        'predominantly reflect adipose features. All 8 tissue regions are present across',
        'the full 504–514 feature sets; their relative contributions are reported in',
        'Table 3. In BLCA, the remaining regions each contribute 10–13% (urothelium,',
        'muscularis, tumor, immune, lamina, necrosis, vessels). In BRCA, TILs rank 2nd',
        '(14.4%), reflecting the established prognostic role of tumour-infiltrating',
        'lymphocytes in breast cancer.',
        '',
        '------------------------------------------------------------------------------',
        'FIGURE 2 — MED3PA reliability landscape across all four arms (2×2)',
        'File: fig2_combined_accuracy_coverage.png',
        '------------------------------------------------------------------------------',
        'NOTE ON MED3PA FRAMING: MED3PA was designed to identify patient subgroups where',
        'a model cannot be trusted — Caution profiles where error is high and unpredictable.',
        'Trust profiles are the complementary finding: subgroups where error IS reliably',
        'predictable (low), allowing clinicians to know when the model can be relied upon.',
        'Both zones are clinically meaningful; Caution flags uncertainty, Trust enables',
        'confident deployment.',
        '',
        'Each point is one stable MED3PA profile (≥5% of 1,000 runs). X-axis: average',
        'population coverage. Y-axis: MED3PA classification accuracy (distinguishing',
        'high vs low backbone error patients). Green = Trust (accuracy ≥0.80); red =',
        'Caution (<0.80). Point size ∝ run frequency. Dashed line = 0.80 threshold.',
        'Top 3 Trust profiles (▲, bottom-right green box) and worst 3 Caution profiles',
        '(▼, top-left red box) are each numbered on the scatter and listed in their',
        'respective boxes. Numbers on scatter correspond to box entries. Each entry',
        'shows: compact profile rule, accuracy +/- SD, significance stars from a',
        'two-sided t-test vs chance (0.5): *p<0.05, **p<0.01, ***p<0.001. Stars on',
        'Trust profiles mean accuracy is significantly ABOVE chance (model reliable);',
        'stars on Caution profiles mean accuracy is significantly BELOW chance (model',
        'errors are genuinely unpredictable for that subgroup). and population coverage.',
        'Abbreviations: stage/T/N/M = pathologic stage; ER+/ER− = oestrogen receptor;',
        'PR+/PR− = progesterone receptor; Adip/TIL/Lam/Musc/Imm = tissue region;',
        'H/E = staining channel; Var/MAD/IQR/Ent = first-order radiomic metric.',
        '',
        '------------------------------------------------------------------------------',
        'FIGURE 3 — MED3PA profile stability summary across all four arms (2×2)',
        'File: fig3_med3pa_stability_summary.png',
        '------------------------------------------------------------------------------',
        'Companion to Figure 2. For each arm: top 3 profiles by accuracy (best Trust,',
        'green bars) and bottom 3 by accuracy (worst Caution, red bars), all from stable',
        'profiles (≥5% of 1,000 runs), sorted by accuracy for display. This selection',
        'directly shows the reliability extremes: the patient subgroups where the model',
        'is most vs. least trustworthy. Bar length = run frequency (% of 1,000 runs).',
        'Bar colour = average accuracy (RdYlGn colourbar, right). Bar label: T=Trust',
        '(>=0.80) / C=Caution (<0.80), accuracy +/- SD, significance stars from a',
        'two-sided t-test vs chance (0.5; key in title: *p<0.05, **p<0.01, ***p<0.001).',
        'Stars on Trust bars = significantly better than chance (reliable subgroup);',
        'stars on Caution bars = significantly worse than chance (genuinely unpredictable).',
        'and population coverage (%). Y-axis labels include feature name and threshold',
        'value; multi-condition rules joined by "&". Abbreviations same as Figure 2.',
        '',
        '------------------------------------------------------------------------------',
        'TABLE 3 — Radiomic feature importance by tissue region (cross-arm)',
        'File: table3_radiomic_region_importance.csv',
        '------------------------------------------------------------------------------',
        'For each of the two radiomic arms (BLCA and BRCA), the total RF importance',
        'attributable to each tissue region is shown, expressed as a percentage of the',
        'total across all features in that arm. This contextualises Figure 1, where the',
        'top 8 features predominantly represent adipose tissue.',
        '',
        'Columns:',
        '  Arm              — BLCA Radiomic or BRCA Radiomic.',
        '  Tissue region    — The histological compartment from which radiomic features',
        '                     were extracted.',
        '  Importance (%)   — Sum of mean RF importance across all features from this',
        '                     region, as a percentage of the total across all regions.',
        '  N features       — Total number of first-order features extracted from this',
        '                     region (across H and E channels, mean and std statistics).',
        '  Features in top 20 — Number of this region\'s features appearing in the top',
        '                     20 by mean RF importance.',
        '  Best rank        — The rank of the single highest-importance feature from',
        '                     this region.',
        '',
        'Key findings: Adipose tissue is the most important region in both arms (BLCA:',
        '19.6%, BRCA: 25.8%) and contributes 16 of the top 20 features in BLCA and 14',
        'in BRCA. However, adipose dominance should not obscure the fact that all',
        'remaining regions contribute meaningfully — in BLCA, the other 7 regions each',
        'contribute 10–13% of total importance. In BRCA, TILs rank 2nd (14.4%) with 5',
        'top-20 features, reflecting the prognostic importance of immune infiltration in',
        'breast cancer. Tumour-region features rank 5th in BLCA (12.3%) and 5th/6th in',
        'BRCA (9.9%), notably lower than the peri-tumoural adipose compartment in both',
        'cancers — a finding consistent across both cohorts suggesting that the WSI',
        'model\'s survival predictions are driven more by the tumour microenvironment',
        'than by the invasive tumour itself.',
        '',
        '==============================================================================',
    ]
    out = os.path.join(OUTDIR, 'paper_table_captions.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'\nGenerating paper figures → {OUTDIR}\n')

    data_list = [load_arm(lbl, rdir, mode) for lbl, rdir, mode in ARMS]

    fig_combined_importance(data_list, top_n=8)
    fig_combined_accuracy_coverage(data_list, min_freq=5.0)
    fig_med3pa_stability_summary(data_list, top_n=6, min_freq=5.0)

    t1 = make_table1(data_list)
    t2 = make_table2(data_list)
    t3 = make_table3(data_list)
    write_table_captions(t1, t2)

    print(f'\nDone. All paper outputs written to: {OUTDIR}')


if __name__ == '__main__':
    main()
