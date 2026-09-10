#!/usr/bin/env python3
"""
Merge BRCA ablation partial JSONs produced by parallel round-range jobs into
the final region_ablation_brca.json.

Expected inputs (all must exist):
  results/region_ablation_brca_partial.json          — rounds 1-6  (from log parse)
  results/region_ablation_brca_partial_r07_12.json   — rounds 7-12
  results/region_ablation_brca_partial_r13_16.json   — rounds 13-16
  results/region_ablation_brca_partial_r17_20.json   — rounds 17-20

Output:
  results/region_ablation_brca.json

Usage:
  python analysis/merge_brca_ablation.py
"""

import json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARTIALS = [
    os.path.join(REPO, "results", "region_ablation_brca_partial.json"),
    os.path.join(REPO, "results", "region_ablation_brca_partial_r07_12.json"),
    os.path.join(REPO, "results", "region_ablation_brca_partial_r13_16.json"),
    os.path.join(REPO, "results", "region_ablation_brca_partial_r17_20.json"),
]

OUT = os.path.join(REPO, "results", "region_ablation_brca.json")

def summarise(vals):
    a = np.array(vals)
    n = len(a)
    mean = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci95 = [mean - 1.96 * se, mean + 1.96 * se]
    return {"mean": mean, "sd": float(np.std(a, ddof=1)), "se": se,
            "ci95": ci95, "n_rounds": n}

# ── Load all partials ─────────────────────────────────────────────────────────
all_baseline_rounds = []
per_region_all = {}  # ri -> list of round dicts, sorted by round number
region_names = None

for path in PARTIALS:
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        sys.exit(1)
    with open(path) as f:
        state = json.load(f)
    if region_names is None:
        region_names = state["region_names"]
    n_regions = len(region_names)
    for ri in range(n_regions):
        per_region_all.setdefault(ri, []).extend(state["per_region_rounds"][str(ri)])
    all_baseline_rounds.extend(state["baseline_rounds"])
    last = state.get("last_completed_round", state.get("completed_rounds", "?"))
    print(f"Loaded {path}  ({len(state['baseline_rounds'])} rounds, last={last})")

# Sort each region's rounds by round number and check coverage
for ri in per_region_all:
    per_region_all[ri].sort(key=lambda d: d["round"])

round_nums = sorted(d["round"] for d in per_region_all[0])
print(f"\nRounds present: {round_nums}")
missing = [r for r in range(1, 21) if r not in round_nums]
if missing:
    print(f"WARNING: missing rounds {missing}")
    sys.exit(1)
print(f"All 20 rounds present. Merging...")

# ── Build final summary ───────────────────────────────────────────────────────
per_region_summary = {}
for ri, rname in enumerate(region_names):
    deltas_no = [d["delta_no_renorm"] for d in per_region_all[ri]]
    deltas_re = [d["delta_renorm"]    for d in per_region_all[ri]]
    per_region_summary[rname] = {
        "delta_no_renorm": summarise(deltas_no),
        "delta_renorm":    summarise(deltas_re),
        "rounds":          per_region_all[ri],
    }

result = {
    "cohort":            "brca",
    "n_rounds":          20,
    "epochs_per_round":  15,
    "train_frac":        0.8,
    "num_regions":       len(region_names),
    "region_names":      region_names,
    "ablation_method": (
        "Region r's embedding is zeroed after per-region processing "
        "(projection + graph-diffusion + attention pooling) and before "
        "cross-region self-attention, so region r also does not participate "
        "in other regions' cross-region attention. After cross-region "
        "LayerNorm, position r is zeroed again. "
        "no_renorm: original normalised weights kept (w still sums to 1, "
        "region r contributes w[r]*0=0). "
        "renorm: w[r] forced to 0 and remaining weights renormalised to sum to 1."
    ),
    "evaluation":      "out-of-bag (OOB) patients only; in-sample c-index is not reported",
    "note":            "Rounds run in parallel shards (1-6, 7-12, 13-16, 17-20) and merged.",
    "baseline_cindex": summarise(all_baseline_rounds),
    "per_region":      per_region_summary,
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n>>> Saved: {OUT}")
print(f"    Baseline c-index: {result['baseline_cindex']['mean']:.4f} "
      f"± {result['baseline_cindex']['sd']:.4f}")
for rname, v in per_region_summary.items():
    d = v["delta_no_renorm"]
    print(f"    [{rname[:45]}]  Δ={d['mean']:+.4f} (sd={d['sd']:.4f})")
