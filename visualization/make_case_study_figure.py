"""One patient through the audit (TCGA-B6-A0I9, BRCA): a half-page figure.

Reads results/case_study/TCGA-B6-A0I9.json, the images in paper_figures/case_study/,
and cohort-level numbers from figure_data. Writes paper_figures/fig_case_study.{pdf,png}.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import figure_data as fd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PID = "TCGA-B6-A0I9"
IMG = os.path.join(ROOT, "paper_figures", "case_study", PID)
OUT = os.path.join(ROOT, "paper_figures", "fig_case_study")

NAVY, GREY, LGREY, INK, RED, GREEN, AMBER = "#1F2A5C", "#6B7280", "#D5D8DE", "#222222", "#C0392B", "#2E8B57", "#B9770E"
I_COL, E_COL, R_COL = "#B9770E", "#6C3483", "#117A65"
MUSCLE, ADIPOSE = "#65A30D", "#D97706"
BASE = 7.0
plt.rcParams.update({"font.family": "Calibri", "font.size": BASE, "axes.edgecolor": GREY, "axes.linewidth": 0.6,
                     "xtick.labelsize": BASE - 0.5, "ytick.labelsize": BASE - 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 0, "pdf.fonttype": 42, "svg.fonttype": "path"})
W, H = 7.0, 4.72
COLS = [0.05, 1.80, 3.55, 5.30]  # left edge of each column, inches
CW = 1.65  # column width

SHORT = {"Muscle tissue": "Muscle", "Adipose": "Adipose", "Ductal carcinoma": "DCIS", "Necrosis": "Necrosis",
         "Normal breast": "Normal glands", "Invasive": "Tumour", "Blood vessels": "Vessels", "Fibrous": "Stroma",
         "Tumor-infiltrating": "TILs"}
FEAT = {"InterquartileRange": "IQR", "RobustMeanAbsoluteDeviation": "robust MAD", "Variance": "variance",
        "MeanAbsoluteDeviation": "MAD", "RootMeanSquared": "RMS", "10Percentile": "10th pct.", "90Percentile": "90th pct.",
        "Mean": "mean"}
COMP = {"adipose": "Adipose", "dcis": "DCIS", "stroma": "Stroma", "tils": "TILs", "tdlu": "Normal glands"}


def short(name):
    return next(v for k, v in SHORT.items() if name.startswith(k))


def feat_label(f):
    comp, stain, _, _, stat, _ = f.split("_")
    return f"{COMP[comp]} {'eosin' if stain == 'E' else 'haemat.'} {FEAT[stat]}"


def rule_label(rule):
    feat, rest = rule.split(" ", 1)
    op, thr = rest.split(" ")[:2]
    comp, stain, _, _, stat, _ = feat.split("_")
    name = {"Normal glands": "Glands"}.get(COMP[comp], COMP[comp])
    return f"{name} {FEAT[stat]} ({stain}) {'≤' if op == '<=' else '>'} {float(thr):.3g}"


def ax_in(fig, left, top, width, height):
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def header(fig, x, letter, title, col):
    fig.patches.append(matplotlib.patches.Circle(((x + 0.11) / W, 1 - 0.17 / H), 0.10 / W, transform=fig.transFigure,
                                                 color=col))
    fig.text((x + 0.11) / W, 1 - 0.17 / H, letter, ha="center", va="center", color="white", fontsize=BASE + 1.5,
             fontweight="bold")
    fig.text((x + 0.27) / W, 1 - 0.17 / H, title, ha="left", va="center", color=NAVY, fontsize=BASE + 1.5,
             fontweight="bold")


def verdict(fig, x, top, lines, col, height=0.80):
    """lines: (mark, text), mark in {'yes', 'no', 'note'}."""
    fig.patches.append(FancyBboxPatch((x / W, 1 - (top + height) / H), CW / W, height / H,
                                      boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      facecolor="white", edgecolor=col, linewidth=0.9))
    fig.text((x + 0.07) / W, 1 - (top + 0.06) / H, "WHAT THE AUDIT SAYS", ha="left", va="top", fontsize=BASE - 0.8,
             fontweight="bold", color=col)
    y = top + 0.20
    for mark, text in lines:
        sym, c = {"yes": ("✓", GREEN), "no": ("✗", RED), "note": ("!", AMBER)}[mark]
        fig.text((x + 0.07) / W, 1 - y / H, sym, ha="left", va="top", fontsize=BASE + 0.5, fontweight="bold", color=c,
                 fontfamily="DejaVu Sans")
        fig.text((x + 0.22) / W, 1 - y / H, text, ha="left", va="top", fontsize=BASE - 0.5, color=INK,
                 linespacing=1.15)
        y += 0.19 * (1 + text.count("\n"))


def main():
    d = json.load(open(os.path.join(ROOT, "results", "case_study", f"{PID}.json"), encoding="utf-8"))
    interp = fd.interp_summary("brca")
    expl = fd.explain_summary("brca")
    au = fd.auroc_by_arm()
    au = au[(au.cohort == "brca") & (au.arm == "pathomic")]

    fig = plt.figure(figsize=(W, H))

    # ---- column 1: the patient ----
    x = COLS[0]
    header(fig, x, "P", "The patient", NAVY)
    thumb = np.array(Image.open(os.path.join(IMG, "thumbnail.png")).convert("RGB"))
    tmap = np.array(Image.open(os.path.join(IMG, "tissue_map.png")).convert("RGB"))
    focus = np.full_like(tmap, 255)
    tissue = (tmap != 255).any(axis=2)
    focus[tissue] = (213, 216, 222)
    for hexc in (MUSCLE, ADIPOSE):
        rgb = tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5))
        focus[(tmap == rgb).all(axis=2)] = rgb
    ih = CW * thumb.shape[0] / thumb.shape[1]
    for k, im in enumerate((thumb, focus)):
        a = ax_in(fig, x, 0.38 + k * (ih + 0.05), CW, ih)
        a.imshow(im, interpolation="lanczos")
        a.axis("off")
    ly = 0.38 + 2 * ih + 0.12
    for k, (lab, c) in enumerate((("Muscle", MUSCLE), ("Adipose", ADIPOSE), ("Other tissue", LGREY))):
        lx = x + 0.02 + k * 0.52
        fig.patches.append(matplotlib.patches.Rectangle((lx / W, 1 - (ly + 0.05) / H), 0.09 / W, 0.09 / H,
                                                        transform=fig.transFigure, color=c))
        fig.text((lx + 0.13) / W, 1 - ly / H, lab, ha="left", va="top", fontsize=BASE - 0.7, color=INK)
    cdr, risk, out = d["cdr"], d["risk"], d["outcome"]
    facts = [f"Breast cancer, {cdr['ajcc_pathologic_tumor_stage'].replace('Stage', 'stage')}, "
             f"age {int(cdr['age_at_initial_pathologic_diagnosis'])}",
             f"Progressed at {int(out['pfi_time_days'])} days",
             f"Model risk: {round(risk['cohort_percentile'])}th percentile",
             f"Ranked correctly: wrong in {round(100 * d['error']['patient_error'])}% of comparisons"]
    for k, f in enumerate(facts):
        fig.text((x + 0.02) / W, 1 - (ly + 0.20 + 0.16 * k) / H, f, ha="left", va="top", fontsize=BASE - 0.3,
                 color=INK, fontweight="bold" if k == 2 else "normal")
    verdict(fig, x, 3.44, [("note", "Tissue labels: a pathologist questioned\nseveral here, muscle among them")], NAVY,
            height=0.62)

    # ---- column 2: interpretability ----
    x = COLS[1]
    header(fig, x, "I", "Which tissue?", I_COL)
    att = d["attribution"]
    names = sorted(att["share"], key=lambda k: att["share"][k])
    a = ax_in(fig, x + 0.62, 0.45, CW - 0.66, 1.62)
    ys = np.arange(len(names))
    cols = [MUSCLE if n.startswith("Muscle") else ADIPOSE if n.startswith("Adipose") else LGREY for n in names]
    a.barh(ys, [100 * att["share"][n] for n in names], color=cols, height=0.7)
    a.scatter([100 * att["cohort_share"][n] for n in names], ys, marker="|", s=40, color=INK, lw=1.0, zorder=3)
    a.set_yticks(ys)
    a.set_yticklabels([short(n) for n in names], fontsize=BASE - 0.5)
    a.set_xlim(0, 20)
    a.set_xticks([0, 10, 20])
    a.set_xlabel("Share of her risk score (%)", fontsize=BASE - 0.5, labelpad=1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.text((x + 0.62) / W, 1 - 0.36 / H, "| = cohort average", fontsize=BASE - 1, color=GREY, va="top")
    # Two of the 12 tiles CONCH labelled muscle / adipose with the highest confidence (tiles_top/, chosen by eye)
    for k, (lab, f, c) in enumerate((("adipose", "adipose_07.png", ADIPOSE),)):
        t = ax_in(fig, x + 0.05 + k * 0.82, 2.42, 0.72, 0.72)
        t.imshow(Image.open(os.path.join(IMG, "tiles_top", f)))
        t.set_xticks([])
        t.set_yticks([])
        for s in t.spines.values():
            s.set_edgecolor(c)
            s.set_linewidth(1.5)
        fig.text((x + 0.05 + k * 0.82) / W, 1 - 3.17 / H, f"Labelled {lab}", fontsize=BASE - 1, color=GREY,
                 va="top")
    fig.text((x + 0.88) / W, 1 - 2.46 / H, "A tile CONCH labelled\nadipose with high\nconfidence on her\nslide (about 64 µm).",
             fontsize=BASE - 1, color=GREY, va="top", linespacing=1.3)
    verdict(fig, x, 3.44, [("yes", f"Weights match what the model\nuses (ρ = {interp['rho_weight_use']:+.2f})"),
                           ("no", f"What it uses does not track\noutcome (ρ = {interp['rho_use_outcome']:+.2f})".replace("-", "−") + "")],
            I_COL)

    # ---- column 3: explainability ----
    x = COLS[2]
    header(fig, x, "E", "Which features?", E_COL)
    feats = d["pathomic_features_top10_q50"][:6]
    a = ax_in(fig, x + 0.86, 0.45, CW - 0.92, 1.62)
    ys = np.arange(len(feats))[::-1]
    a.axvline(50, color=LGREY, lw=0.8, zorder=0)
    for y, f in zip(ys, feats):
        c = ADIPOSE if f["compartment"] == "adipose" else GREY
        a.plot([50, f["cohort_percentile"]], [y, y], color=c, lw=1.0)
        a.scatter([f["cohort_percentile"]], [y], s=18, color=c, zorder=3)
    a.set_yticks(ys)
    a.set_yticklabels([feat_label(f["feature"]) for f in feats], fontsize=BASE - 0.7)
    a.set_xlim(0, 100)
    a.set_xticks([0, 50, 100])
    a.set_xlabel("Her percentile in the cohort", fontsize=BASE - 0.5, labelpad=1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.text((x + 0.02) / W, 1 - 0.36 / H, "The surrogate's six leading features", fontsize=BASE - 1, color=GREY,
             va="top")
    fig.text((x + 0.02) / W, 1 - 2.42 / H,
             "The surrogate learns the model's\nrisk score from stain statistics.\nAdipose eosin spread leads;\n"
             "she sits mid-cohort on it.", fontsize=BASE - 0.5, color=INK, va="top", linespacing=1.3)
    verdict(fig, x, 3.44, [("yes", "The ranking describes what the\nmodel's score follows"),
                           ("no", f"No feature predicts outcome\nafter correction (best q = {expl['min_q']:.3f})")],
            E_COL)

    # ---- column 4: reliability ----
    x = COLS[3]
    header(fig, x, "R", "Can it be trusted?", R_COL)
    prof = []
    for run, v in d["runs"].items():
        if v["held_out"]:
            for p in v["matching_profiles"]:
                prof.append((run.split("_")[1], p))
    a = ax_in(fig, x + 0.86, 0.45, CW - 0.90, 1.95)
    cval = prof[0][1]["cohort_cindex"]
    a.axvline(cval, color=GREY, lw=0.8, ls="--", zorder=0)
    ys = np.arange(len(prof))[::-1]
    for y, (thr, p) in zip(ys, prof):
        c = R_COL if thr == "q80" else "#5FB3A2"
        lo, hi = p["cindex_ci95"]
        a.plot([lo, hi], [y, y], color=c, lw=0.9)
        a.scatter([p["cindex_in_profile"]], [y], s=12, color=c, zorder=3)
    a.set_yticks(ys)
    a.set_yticklabels([rule_label(p["readable_rule"]) for _, p in prof], fontsize=BASE - 1.6)
    a.set_xlim(0.3, 0.8)
    a.set_xticks([0.4, 0.6, 0.8])
    a.set_xlabel("C-index in her group", fontsize=BASE - 0.5, labelpad=1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.text((x + 0.02) / W, 1 - 0.36 / H, f"The {len(prof)} recurring groups she falls into",
             fontsize=BASE - 1, color=GREY, va="top")
    for k, (lab, c) in enumerate((("80th-percentile cut", R_COL), ("90th-percentile cut", "#5FB3A2"),
                                  ("- - all patients", GREY))):
        fig.text((x + 0.86) / W, 1 - (2.80 + 0.13 * k) / H, ("● " if k < 2 else "") + lab, fontsize=BASE - 1,
                 color=c, va="top")
    lo, hi = au.auroc.min(), au.auroc.max()
    verdict(fig, x, 3.44, [("yes", f"Error is predictable in BRCA\n(AUROC {lo:.2f} to {hi:.2f})"),
                           ("no", f"None of her {len(prof)} groups is shown to\ndiffer from all patients: no warning")], R_COL)

    # ---- bottom strip: what her report could say ----
    fig.patches.append(FancyBboxPatch((0.05 / W, 1 - 4.66 / H), 6.90 / W, 0.36 / H,
                                      boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      facecolor=NAVY, edgecolor=NAVY))
    fig.text(0.15 / W, 1 - 4.48 / H,
             "After the audit, a report of the model's output for her could still give the risk score "
             f"({round(risk['cohort_percentile'])}th percentile) and the tissue types the model relied on,\n"
             "described as the model's behaviour. It would present no tissue type or feature as prognostic, "
             "and it would carry no reliability warning.",
             ha="left", va="center", color="white", fontsize=BASE - 0.2, linespacing=1.3)

    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
