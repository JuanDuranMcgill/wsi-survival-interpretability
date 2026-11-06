"""Data panels (b) of the three zoom figures (I, E, R), from results/ via figure_data.

Writes paper_figures/panels/fig{2,3,4}b.{svg,pdf,png}. Every number comes from
figure_data, whose values are pinned to the paper by tests/test_figure_data.py.
Each panel is 3.3 in wide, drawn at final print size (text >= 6.5 pt).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figure_data as fd  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_figures", "panels")
NAVY, RED, GREEN, GREY, LGREY, INK = "#1F2A5C", "#C0392B", "#2E8B57", "#7F8C8D", "#C9CED6", "#222222"
BASE = 7.0
MIN_PT = 6.5
W = 3.3
plt.rcParams.update({
    "font.family": "Calibri", "font.size": BASE, "axes.labelsize": BASE - 0.5,
    "xtick.labelsize": BASE - 0.5, "ytick.labelsize": BASE - 0.5, "axes.edgecolor": GREY,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "xtick.color": "#333333", "ytick.color": "#333333",
    "svg.fonttype": "path", "pdf.fonttype": 42, "axes.unicode_minus": True,
})
MARK = {"yes": "✓", "no": "✗"}


def _axes(fig, top_in, height_in, left_in, right_in=0.08):
    """Axes placed in inches from the top-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left_in / fw, 1 - (top_in + height_in) / fh, (fw - left_in - right_in) / fw,
                         height_in / fh])


def _title(fig, top_in, text):
    fh = fig.get_size_inches()[1]
    fig.text(0.06 / W, 1 - top_in / fh, text, ha="left", va="top", fontsize=BASE + 0.5, fontweight="bold",
             color=NAVY)


def _clean(ax, left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def _signed(x):
    return f"{x:+.2f}".replace("-", "−")


def _rho_rows(ax, rows, xlabel):
    """rows: (label, rho, verdict), verdict in {'yes', 'no', 'weak'}; labels sit left of the axes."""
    ax.axvline(0, color=LGREY, lw=0.8, zorder=0)
    n = len(rows)
    for i, (label, rho, verdict) in enumerate(rows):
        y = n - 1 - i
        col = {"yes": GREEN, "no": RED, "weak": GREY}[verdict]
        ax.plot([0, rho], [y, y], color=col, lw=1.3, zorder=1, solid_capstyle="butt")
        ax.scatter([rho], [y], s=22, color=col, zorder=2, linewidths=0)
        right = rho >= 0
        x_txt = rho + (0.08 if right else -0.08)
        ax.text(x_txt, y, _signed(rho), va="center", ha="left" if right else "right", color=col,
                fontsize=BASE, fontweight="bold")
        x_mark = rho + (0.44 if right else -0.44)
        if verdict == "weak":
            ax.text(x_mark, y, "weak", va="center", ha="left" if right else "right", color=col, fontsize=BASE)
        else:
            ax.text(x_mark, y, MARK[verdict], va="center", ha="left" if right else "right", color=col,
                    fontsize=BASE + 1, fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(-1.08, y, label, va="center", ha="right", fontsize=BASE - 0.5, color=INK)
    ax.set_xlim(-1, 1.55)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "+1"])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=1)
    _clean(ax, left=False)


def fig2b():
    b, r = fd.interp_summary("blca"), fd.interp_summary("brca")
    fig = plt.figure(figsize=(W, 4.25))

    _title(fig, 0.02, "Faithful in both cohorts, prognostic only in BLCA")
    ax = _axes(fig, 0.24, 0.72, left_in=1.45)
    _rho_rows(ax, [
        ("BLCA  weights vs model use", fd.half_up(b["rho_weight_use"], 2), "yes"),
        ("BLCA  model use vs outcome", fd.half_up(b["rho_use_outcome"], 2), "yes"),
        ("BRCA  weights vs model use", fd.half_up(r["rho_weight_use"], 2), "yes"),
        ("BRCA  model use vs outcome", fd.half_up(r["rho_use_outcome"], 2), "no"),
    ], "Spearman ρ across compartments")

    # BRCA bump chart: rank by model use (left) vs rank by outcome-based Shapley value (right)
    _title(fig, 1.30, "BRCA: what the model uses is not what predicts outcome")
    ax = _axes(fig, 1.50, 1.30, left_in=0.06, right_in=0.06)
    use = sorted(r["use_share"].items(), key=lambda kv: -kv[1])
    out = sorted(r["shapley_pp"].items(), key=lambda kv: -kv[1])
    lr = {k: i for i, (k, _) in enumerate(use)}
    rr = {k: i for i, (k, _) in enumerate(out)}
    hi_use, hi_out = {"DCIS", "Necrosis"}, {"Adipose", "Muscle"}
    x0, x1 = 0.36, 0.64
    for k in lr:
        col = NAVY if k in hi_use else RED if k in hi_out else LGREY
        lw = 1.5 if col != LGREY else 0.7
        ax.plot([x0, x1], [-lr[k], -rr[k]], color=col, lw=lw, zorder=2 if col != LGREY else 1)
        ax.scatter([x0, x1], [-lr[k], -rr[k]], s=9, color=col, zorder=3, linewidths=0)
    for k, v in use:
        strong = k in hi_use
        ax.text(x0 - 0.02, -lr[k], f"{k}  {v:.1f}%", ha="right", va="center", fontsize=BASE - 0.5,
                color=NAVY if strong else "#555555", fontweight="bold" if strong else "normal")
    for k, v in out:
        strong = k in hi_out
        ax.text(x1 + 0.02, -rr[k], f"{_signed(v)} pp  {k}", ha="left", va="center", fontsize=BASE - 0.5,
                color=RED if strong else "#555555", fontweight="bold" if strong else "normal")
    ax.text(x0 - 0.02, 1.05, "share of what the\nmodel's score uses", ha="right", va="bottom",
            fontsize=BASE - 0.5, color=NAVY, linespacing=0.95)
    ax.text(x1 + 0.02, 1.05, "share of concordance\n(outcome-based Shapley)", ha="left", va="bottom",
            fontsize=BASE - 0.5, color=RED, linespacing=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(use) + 0.5, 2.3)
    ax.axis("off")

    _title(fig, 3.00, "A known signal planted in one compartment")
    ax = _axes(fig, 3.22, 0.74, left_in=1.10, right_in=0.30)
    h = fd.planted_hit_rates()
    order = ["Shapley", "Deletion", "Weights", "Integrated gradients", "Gradient x input", "ABMIL attention"]
    labels = ["Shapley value", "Deletion", "Fusion weight", "Integrated gradients", "Gradient × input",
              "Attention (ABMIL)"]
    vals = [round(h[k][0]) for k in order]
    cols = [NAVY] + [GREY] * 4 + [RED]
    y = list(range(len(order)))[::-1]
    ax.barh(y, vals, color=cols, height=0.66)
    for yi, v, k in zip(y, vals, order):
        ax.text(v + 1.0, yi, fd.pct_label(h[k][0]), va="center", fontsize=BASE - 0.5,
                color=RED if k == "ABMIL attention" else INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("% of rounds ranking it first", labelpad=1)
    _clean(ax)
    return fig


def fig3b():
    b, r = fd.explain_summary("blca"), fd.explain_summary("brca")
    fig = plt.figure(figsize=(W, 3.95))

    _title(fig, 0.02, "A second estimator: region order holds")
    ax = _axes(fig, 0.24, 0.72, left_in=1.30)
    _rho_rows(ax, [
        ("BLCA  region order", fd.half_up(b["region_rho"], 2), "yes"),
        ("BRCA  region order", fd.half_up(r["region_rho"], 2), "yes"),
        ("BLCA  feature order", fd.half_up(b["feature_rho"], 2), "weak"),
        ("BRCA  feature order", fd.half_up(r["feature_rho"], 2), "weak"),
    ], "Spearman ρ, impurity vs permutation importance")

    _title(fig, 1.30, "Same leading region, but its share is not robust")
    ax = _axes(fig, 1.52, 0.42, left_in=1.30, right_in=0.30)
    vals = [round(b["adipose_share_impurity"]), round(b["adipose_share_perm"])]
    ax.barh([1, 0], vals, color=[GREY, NAVY], height=0.62)
    for yi, v in zip([1, 0], vals):
        ax.text(v + 1.2, yi, f"{v}%", va="center", fontsize=BASE)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Impurity", "Permutation"])
    ax.set_xlim(0, 70)
    ax.set_xlabel("adipose share of region importance (BLCA)", labelpad=1)
    _clean(ax)

    _title(fig, 2.40, "Against outcome, no feature survives correction")
    ax = _axes(fig, 2.62, 0.98, left_in=1.30, right_in=0.30)
    ticks, y = [], 0.0
    for name, e in [("BLCA", b), ("BRCA", r)]:
        for lab, v, col in [("tested", e["n_tested"], LGREY), ("p < 0.05", e["n_nominal"], GREY),
                            ("after FDR", e["n_fdr"], RED)]:
            ax.barh(-y, v, color=col, height=0.72)
            fdr = lab == "after FDR"
            ax.text(v + 15, -y, f"{v:,}", va="center", fontsize=BASE, color=RED if fdr else INK,
                    fontweight="bold" if fdr else "normal")
            ticks.append((-y, f"{name}  {lab}"))
            y += 1
        y += 0.6
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([s for _, s in ticks])
    ax.set_xlim(0, 1000)
    ax.set_xlabel("pathomic features tested against observed PFI", labelpad=1)
    _clean(ax)
    return fig


FOREST = [  # (cohort, arm, thr, profile, plain label); the first three are the profiles that differ
    ("brca", "clinical", "q50", "pathologic_stage", "Stage I (q50)"),
    ("brca", "clinical", "q70", "pathologic_stage", "Stage I (q70)"),
    ("brca", "clinical", "q80", "age_at_diagnosis", "Middle age band"),
    ("brca", "clinical", "q70", "pathologic_stage_UP", "Stage II or higher"),
    ("brca", "clinical", "q80", "age_at_diagnosis_UP", "Older age"),
    ("brca", "clinical", "q70", "lymph_node_examined_count_UP", "More nodes examined"),
    ("brca", "pathomic", "q50", "adipose_E_original_firstorder_InterquartileRange_std_DOWN",
     "Fat staining, low spread"),
    ("brca", "pathomic", "q80", "stroma_H_original_firstorder_MeanAbsoluteDeviation_std_UP",
     "Stroma staining, high spread"),
]


def _pick(p, cohort, arm, thr, key):
    m = p[(p.cohort == cohort) & (p.arm == arm) & (p.thr == thr)]
    exact = m[m.profile == key]
    if len(exact):
        return exact.iloc[0]
    return m[m.differs & m.rule.str.contains(key)].iloc[0]  # the three that differ


def fig4b():
    fig = plt.figure(figsize=(W, 4.35))
    a = fd.auroc_by_arm()

    _title(fig, 0.02, "Is error predictable? In BRCA, yes")
    ax = _axes(fig, 0.24, 0.62, left_in=1.05, right_in=0.12)
    arms = [("blca", "clinical"), ("blca", "pathomic"), ("brca", "clinical"), ("brca", "pathomic")]
    ax.axvline(0.5, color=LGREY, lw=0.8, zorder=0)
    for i, (c, arm) in enumerate(arms):
        yy = len(arms) - 1 - i
        for _, row in a[(a.cohort == c) & (a.arm == arm)].iterrows():
            sig = row.p < 0.05 / 16
            ax.scatter(row.auroc, yy, s=16, facecolor=NAVY if sig else "white", edgecolor=NAVY, lw=0.8,
                       zorder=2)
    ax.set_yticks(range(len(arms))[::-1])
    ax.set_yticklabels([f"{c.upper()}  {arm}" for c, arm in arms])
    ax.set_xlim(0.40, 0.72)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("AUROC, 4 thresholds each (filled: survives Bonferroni)", labelpad=1)
    _clean(ax)

    p = fd.profiles()
    n_all, n_diff = len(p), int(p.differs.sum())
    _title(fig, 1.22, f"Does the model perform differently there? {n_diff} of {n_all} groups")
    ax = _axes(fig, 1.46, 1.50, left_in=1.42, right_in=0.12)
    cval = p[p.cohort == "brca"].cohort_value.iloc[0]
    rows = [(_pick(p, *spec[:4]), spec[4]) for spec in FOREST]
    for i, (row, label) in enumerate(rows):
        yy = len(rows) - 1 - i
        col = RED if row.differs else GREY
        ax.plot([row.lo, row.hi], [yy, yy], color=col, lw=1.2)
        ax.scatter([row.cindex], [yy], s=16, color=col, zorder=3, linewidths=0)
        ax.text(-0.03, yy, f"{label}  ({row.n})", ha="right", va="center", fontsize=BASE - 0.5,
                color=RED if row.differs else INK, transform=ax.get_yaxis_transform())
    ax.axvline(cval, color=NAVY, ls="--", lw=0.9)
    ax.text(cval - 0.02, len(rows) - 0.32, f"all BRCA patients {fd.half_up(cval, 2):.2f}", ha="right",
            va="bottom", fontsize=BASE - 0.5, color=NAVY)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(rows) + 0.25)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("C-index inside the group (95% interval)", labelpad=1)
    _clean(ax, left=False)

    _title(fig, 3.42, "Do per-patient 90% bounds hold by stage? (BRCA)")
    ax = _axes(fig, 3.64, 0.50, left_in=0.42, right_in=0.12)
    c = fd.coverage_by_stage("brca").sort_values("horizon_days")
    hs = sorted(c.horizon_days.unique())
    for stage, col, dx in [("I-II", GREY, -0.10), ("III-IV", RED, 0.10)]:
        s = c[c.stage == stage]
        xs = [hs.index(h) + dx for h in s.horizon_days]
        ax.errorbar(xs, s.coverage, yerr=[s.coverage - s.lo, s.hi - s.coverage], fmt="o", ms=3.0, color=col,
                    lw=0.9, capsize=1.5)
        name = "stage I–II" if stage == "I-II" else "stage III–IV"
        ax.text(xs[-1] + 0.12, list(s.coverage)[-1], name, va="center", ha="left", fontsize=BASE - 0.5,
                color=col)
    ax.axhline(0.90, color=NAVY, ls="--", lw=0.9)
    ax.text(1.5, 0.892, "target 0.90", fontsize=BASE - 0.5, color=NAVY, va="top", ha="center")
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"cutoff {int(h):,} d" for h in hs])
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(0.65, 1.04)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("coverage", labelpad=1)
    _clean(ax)
    return fig


def _check_fonts(fig, name):
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_fontsize() < MIN_PT]
    assert not small, f"{name}: text below {MIN_PT} pt: {small[:5]}"


def _check_inside(fig, name):
    """No text may extend beyond the figure edge (clipped labels)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(rend)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
            out.append(t.get_text())
    assert not out, f"{name}: text outside the figure: {out[:5]}"


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, make in [("fig2b", fig2b), ("fig3b", fig3b), ("fig4b", fig4b)]:
        fig = make()
        _check_fonts(fig, name)
        _check_inside(fig, name)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600, transparent=(ext == "svg"))
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
