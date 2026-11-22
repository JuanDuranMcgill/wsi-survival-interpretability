"""Pan-cancer TCGA manifest, built from metadata only (no slide is downloaded).

Lists every TCGA diagnostic (FFPE, "DX") slide on the GDC, keeps one slide per patient
with the same rule as the current pipeline (DX1 only, keyed on the 12-character patient
barcode), and joins TCGA-CDR PFI (Liu et al., Cell 2018). The table is used to choose the
cancer types, compute storage, and later fix the site-split folds and the events-curve
subsets before any slide is processed (design doc
docs/superpowers/specs/2026-11-20-pancancer-audit-extension-design.md, decisions 1, 5, 8).

Outputs
  data/pancan/raw/gdc_dx_slides.json     GDC query result, cached (--refresh to requery)
  data/pancan/manifest_slides.csv        every DX slide              patient-level, gitignored
  data/pancan/manifest_patients.csv      one row per patient         patient-level, gitignored
  results/pancan/manifest_by_type.csv    counts per cancer type      aggregate, in git
  results/pancan/manifest_summary.json   totals and projections      aggregate, in git

Usage (from the repo root):
  python preprocessing/pancan_manifest.py [--refresh] [--min-events 50]
"""
import argparse
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pancan"
RAW = DATA / "raw"
RESULTS = ROOT / "results" / "pancan"

GDC = "https://api.gdc.cancer.gov"
# TCGA-CDR supplementary table S1 on the GDC PanCanAtlas publication page (same file the
# BLCA/BRCA analyses use; see analysis/compartment_common.py CDR_DEFAULT).
CDR_FILE_ID = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CDR_NAME = "TCGA-CDR-SupplementalTableS1.xlsx"

# Liu 2018 Table 3: PFI is not recommended for PCPG and is absent for LAML
# (docs/lit/wsi_cohorts.md). LAML is caught by the missing-PFI rule.
PFI_NOT_RECOMMENDED = {"PCPG"}

# Juan, 2026-11-21: the cohort is carcinomas only (published pan-cancer tissue sets cover
# carcinomas; see docs/HANDOFF_2026-11-21.md). Non-carcinoma types are flagged out of
# `in_cohort` even when they have enough events.
NON_CARCINOMA = {"GBM", "LGG", "SARC", "SKCM", "MESO", "UVM", "TGCT", "THYM", "PCPG",
                 "DLBC", "LAML", "UCS"}

# Pilot on the MEDomics-Beast, 98 slides (docs/PILOT_BEAST_RESULTS.md): means per slide.
PILOT_GPU_S_PER_SLIDE = 150.0
PILOT_OUTPUT_MB_PER_SLIDE = 291.0
PILOT_MEAN_SLIDE_MB = 860.0
# Linear fits on the same 98 pilot slides, against slide file size in MB (computed from
# /data/processed/juan/pilot/metrics/*.json on 2026-11-20; R^2 0.68 for both). Only LUAD
# and HNSC were timed, so other cancer types extrapolate.
PILOT_GPU_S_FIT = (43.05, 0.12365)       # intercept s, s per MB
PILOT_OUTPUT_MB_FIT = (82.76, 0.24229)   # intercept MB, MB per MB

SLIDE_RE = re.compile(r"^(TCGA-([A-Z0-9]{2})-[A-Z0-9]{4})-(\d{2})[A-Z]-\d{2}-(DX[0-9A-Z]+)\.")


# ---------------------------------------------------------------- pure rules (tested)

def parse_slide_name(file_name):
    """Patient barcode, tissue-source-site code, sample type code and DX tag."""
    m = SLIDE_RE.match(file_name)
    if not m:
        return {"patient": None, "tss_code": None, "sample_code": None, "slide_tag": None}
    return {"patient": m.group(1), "tss_code": m.group(2), "sample_code": m.group(3),
            "slide_tag": m.group(4)}


def select_one_slide_per_patient(slides):
    """DX1 slides only, one per patient: primary tumour (sample code 01) first, then by
    file name. Patients without a DX1 slide are dropped, as in the current pipeline."""
    dx1 = slides[slides["slide_tag"] == "DX1"].copy()
    dx1["_not_primary"] = dx1["sample_code"] != "01"
    dx1 = dx1.sort_values(["patient", "_not_primary", "file_name"])
    n = dx1.groupby("patient").size().rename("n_dx1_candidates")
    chosen = dx1.drop_duplicates("patient", keep="first").drop(columns="_not_primary")
    return chosen.merge(n, left_on="patient", right_index=True).reset_index(drop=True)


def add_eligibility(pat):
    """First failing rule names the exclusion; eligible means none failed."""
    pat = pat.copy()
    rules = [
        (~pat["in_cdr"].astype(bool), "not_in_cdr"),
        (pat["Redaction"].eq("Redacted"), "redacted"),
        (pat["PFI"].isna() | pat["PFI.time"].isna(), "no_pfi"),
        (pat["type"].isin(PFI_NOT_RECOMMENDED), "pfi_not_recommended"),
    ]
    pat["exclusion"] = np.select([c.to_numpy() for c, _ in rules], [r for _, r in rules],
                                 default="")
    pat["eligible"] = pat["exclusion"] == ""
    return pat


# ---------------------------------------------------------------- data access

def _get_json(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def fetch_gdc_slides(page=5000):
    filters = {"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "experimental_strategy",
                                "value": "Diagnostic Slide"}}]}
    fields = ["file_id", "file_name", "file_size", "md5sum", "access", "data_format",
              "cases.submitter_id", "cases.project.project_id"]
    hits, start, total = [], 0, None
    while total is None or start < total:
        d = _get_json(f"{GDC}/files", {"filters": filters, "fields": ",".join(fields),
                                       "format": "json", "size": page, "from": start})["data"]
        hits += d["hits"]
        total = d["pagination"]["total"]
        start += page
    if len(hits) != total:
        raise RuntimeError(f"GDC returned {len(hits)} of {total} files")
    status = _get_json(f"{GDC}/status")
    return {"queried_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "gdc_data_release": status.get("data_release"), "total": total, "hits": hits}


def load_gdc(refresh):
    cache = RAW / "gdc_dx_slides.json"
    if refresh or not cache.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(fetch_gdc_slides()))
    return json.loads(cache.read_text())


def load_cdr(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{GDC}/data/{CDR_FILE_ID}", path)
    x = pd.ExcelFile(path)
    cdr = x.parse("TCGA-CDR").drop(columns=["Unnamed: 0"], errors="ignore")
    tss = x.parse("TSS_Info", header=2, dtype=str)[["TSS Code", "Source Site"]].dropna()
    tss["TSS Code"] = tss["TSS Code"].str.strip().str.zfill(2)
    tss = tss.drop_duplicates("TSS Code").set_index("TSS Code")["Source Site"].str.strip()
    return cdr, tss


def slides_table(gdc):
    rows = []
    for h in gdc["hits"]:
        cases = h.get("cases") or [{}]
        row = {"file_id": h["file_id"], "file_name": h["file_name"],
               "file_size": int(h["file_size"]), "md5sum": h.get("md5sum"),
               "access": h.get("access"), "data_format": h.get("data_format"),
               "gdc_patient": cases[0].get("submitter_id"),
               "project": cases[0].get("project", {}).get("project_id"),
               "n_cases": len(cases)}
        row.update(parse_slide_name(h["file_name"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)


# ---------------------------------------------------------------- build

def _size_fit(df, fit):
    """Sum over slides of intercept + slope * file size in MB."""
    a, b = fit
    return float(len(df) * a + b * df["file_size"].sum() / 1e6)


def build(refresh=False, cdr_path=RAW / CDR_NAME, min_events=50):
    gdc = load_gdc(refresh)
    cdr, tss = load_cdr(cdr_path)
    slides = slides_table(gdc)

    checks = {
        "gdc_dx_slides": int(len(slides)),
        "slides_name_unparsed": int(slides["patient"].isna().sum()),
        "slides_name_patient_mismatch": int(
            (slides["patient"].notna() & (slides["patient"] != slides["gdc_patient"])).sum()),
        "slides_multi_case": int((slides["n_cases"] != 1).sum()),
        "slides_not_open_access": int((slides["access"] != "open").sum()),
        "slide_formats": slides["data_format"].value_counts().to_dict(),
    }

    chosen = select_one_slide_per_patient(slides.dropna(subset=["patient"]))
    per_patient_any = slides.dropna(subset=["patient"]).groupby("patient").agg(
        n_dx_slides=("file_name", "size"), project=("project", "first"))

    pat = per_patient_any.join(chosen.set_index("patient")[
        ["file_id", "file_name", "file_size", "md5sum", "tss_code", "sample_code",
         "n_dx1_candidates"]], how="left").reset_index()
    pat["has_dx1"] = pat["file_name"].notna()

    keep = ["bcr_patient_barcode", "type", "PFI", "PFI.time", "OS", "OS.time", "Redaction",
            "age_at_initial_pathologic_diagnosis", "gender", "ajcc_pathologic_tumor_stage",
            "histological_type"]
    pat = pat.merge(cdr[keep], left_on="patient", right_on="bcr_patient_barcode",
                    how="left").drop(columns="bcr_patient_barcode")
    pat["in_cdr"] = pat["type"].notna()
    pat["tss_institution"] = pat["tss_code"].map(tss)

    dx1 = pat[pat["has_dx1"]].copy()
    dx1 = add_eligibility(dx1)
    pat = pat.merge(dx1[["patient", "exclusion", "eligible"]], on="patient", how="left")
    pat.loc[~pat["has_dx1"], "exclusion"] = "no_dx1_slide"
    pat["eligible"] = pat["eligible"].eq(True)

    checks.update({
        "patients_with_dx": int(len(pat)),
        "patients_with_dx1": int(pat["has_dx1"].sum()),
        "patients_with_several_dx1": int((pat["n_dx1_candidates"] > 1).sum()),
        "exclusions": pat["exclusion"].replace("", "eligible").value_counts().to_dict(),
        "cdr_type_vs_gdc_project_mismatch": int(
            (pat["in_cdr"] & ("TCGA-" + pat["type"].fillna("") != pat["project"])).sum()),
        "eligible_pfi_time_le_0": int((pat["eligible"] & (pat["PFI.time"] <= 0)).sum()),
        "eligible_tss_unmapped": int((pat["eligible"] & pat["tss_institution"].isna()).sum()),
    })

    el = pat[pat["eligible"]].copy()
    el["tss_institution"] = el["tss_institution"].fillna("TSS-" + el["tss_code"])
    by = []
    for t, g in el.groupby("type"):
        ev = int(g["PFI"].sum())
        inst = g["tss_institution"].value_counts()
        by.append({
            "type": t,
            "patients_with_dx": int((pat["project"] == f"TCGA-{t}").sum()),
            "eligible_patients": int(len(g)),
            "pfi_events": ev,
            "censored": int(len(g) - ev),
            "event_rate": round(ev / len(g), 3),
            "median_pfi_days": float(g["PFI.time"].median()),
            "n_institutions": int(inst.size),
            "n_tss_codes": int(g["tss_code"].nunique()),
            "largest_institution_share": round(float(inst.iloc[0] / len(g)), 3),
            "non_primary_slide_share": round(float((g["sample_code"] != "01").mean()), 3),
            "raw_gb": round(float(g["file_size"].sum() / 1e9), 1),
            "mean_slide_mb": round(float(g["file_size"].mean() / 1e6), 0),
            "gpu_hours_size_fit": round(_size_fit(g, PILOT_GPU_S_FIT) / 3600, 1),
            "included": bool(ev >= min_events and t not in PFI_NOT_RECOMMENDED),
        })
    by = pd.DataFrame(by).sort_values("pfi_events", ascending=False).reset_index(drop=True)

    inc_types = set(by.loc[by["included"], "type"])
    pat["included_type"] = pat["eligible"] & pat["type"].isin(inc_types)
    pat["in_cohort"] = pat["included_type"] & ~pat["type"].isin(NON_CARCINOMA)
    by["carcinoma"] = ~by["type"].isin(NON_CARCINOMA)
    by["in_cohort"] = by["included"] & by["carcinoma"]
    inc = pat[pat["in_cohort"]]
    n = int(len(inc))

    def totals(df):
        k = int(len(df))
        return {"patients": k, "pfi_events": int(df["PFI"].sum()),
                "cancer_types": int(df["type"].nunique()),
                "institutions": int(df["tss_institution"].nunique()),
                "raw_tb": round(float(df["file_size"].sum() / 1e12), 2),
                "mean_slide_mb": round(float(df["file_size"].mean() / 1e6), 0)}

    summary = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gdc_queried_utc": gdc["queried_utc"],
        "gdc_data_release": gdc["gdc_data_release"],
        "cdr": f"{CDR_NAME} (GDC file {CDR_FILE_ID})",
        "rules": {
            "slides": "TCGA, data_type 'Slide Image', experimental_strategy 'Diagnostic Slide'",
            "one_slide_per_patient": "DX1 only; primary tumour (sample code 01) first, "
                                     "then file name",
            "eligible": "in TCGA-CDR, not redacted, PFI and PFI.time present, "
                        "PFI recommended for the type (not PCPG)",
            "included_type": f"at least {min_events} PFI events among eligible patients",
            "in_cohort": "included_type and a carcinoma type (Juan, 2026-11-21)",
        },
        "checks": checks,
        "all_eligible": totals(el),
        "included_22_types": totals(pat[pat["included_type"]]),
        "cohort_carcinomas": totals(inc),
        "cohort_types": sorted(set(inc["type"])),
        "projection_included_from_pilot": {
            "basis": "pilot on 98 LUAD/HNSC slides (docs/PILOT_BEAST_RESULTS.md). 'mean': "
                     f"{PILOT_GPU_S_PER_SLIDE:.0f} s GPU and {PILOT_OUTPUT_MB_PER_SLIDE:.0f} MB "
                     f"output per slide (pilot slides averaged {PILOT_MEAN_SLIDE_MB:.0f} MB). "
                     "'size_fit': linear fit on slide file size (R^2 0.68), applied to "
                     "each slide's size; extrapolates to other cancer types",
            "gpu_hours_mean": round(n * PILOT_GPU_S_PER_SLIDE / 3600, 0),
            "gpu_hours_size_fit": round(_size_fit(inc, PILOT_GPU_S_FIT) / 3600, 0),
            "gpu_days_one_gpu_mean": round(n * PILOT_GPU_S_PER_SLIDE / 86400, 1),
            "gpu_days_one_gpu_size_fit": round(_size_fit(inc, PILOT_GPU_S_FIT) / 86400, 1),
            "output_tb_float32_mean": round(n * PILOT_OUTPUT_MB_PER_SLIDE / 1e6, 2),
            "output_tb_float32_size_fit": round(_size_fit(inc, PILOT_OUTPUT_MB_FIT) / 1e6, 2),
        },
    }

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    slides.to_csv(DATA / "manifest_slides.csv", index=False)
    pat.sort_values(["type", "patient"]).to_csv(DATA / "manifest_patients.csv", index=False)
    by.to_csv(RESULTS / "manifest_by_type.csv", index=False)
    (RESULTS / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    return slides, pat, by, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="requery the GDC")
    ap.add_argument("--cdr-xlsx", type=Path, default=RAW / CDR_NAME)
    ap.add_argument("--min-events", type=int, default=50)
    a = ap.parse_args()
    _, _, by, s = build(a.refresh, a.cdr_xlsx, a.min_events)
    print(by.to_string(index=False))
    print(json.dumps({k: s[k] for k in ("checks", "all_eligible", "cohort_carcinomas",
                                        "projection_included_from_pilot")}, indent=2))


if __name__ == "__main__":
    main()
