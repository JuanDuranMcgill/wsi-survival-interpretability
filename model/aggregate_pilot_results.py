#!/usr/bin/env python3
"""Aggregate the 10-round GA2M pilot results per DESIGN_GA2M_HEAD.md's stop
rule: OOB c-index, in-sample c-index, wall time per round, for both cohorts."""
import glob
import json
import numpy as np

def summarize(label, paths, target_cindex):
    rows = []
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        rows.append({
            "path": p,
            "oob_cindex": float(d["oob_cindex"]),
            "train_cindex": float(d["train_cindex"]),
            "wall_time_sec": float(d["wall_time_sec"]),
            "n_oob": len(d["patient_ids"]),
            "seed": int(d["seed"]),
        })
    oob = np.array([r["oob_cindex"] for r in rows])
    train = np.array([r["train_cindex"] for r in rows])
    wt = np.array([r["wall_time_sec"] for r in rows])

    print(f"\n=== {label}: {len(rows)} rounds ===")
    for r in rows:
        print(f"  seed={r['seed']:>3}  oob_cindex={r['oob_cindex']:.4f}  "
              f"train_cindex={r['train_cindex']:.4f}  n_oob={r['n_oob']:>4}  "
              f"wall_time={r['wall_time_sec']/60:.1f} min")
    print(f"  ---")
    print(f"  mean OOB c-index   = {oob.mean():.4f} +/- {oob.std():.4f}  "
          f"(target ~{target_cindex}, within 0.02 = {abs(oob.mean()-target_cindex) <= 0.02})")
    print(f"  mean in-sample     = {train.mean():.4f} +/- {train.std():.4f}")
    print(f"  overfitting gap    = {train.mean() - oob.mean():.4f}")
    print(f"  mean wall time/rd  = {wt.mean()/60:.1f} min  (total GPU-time: {wt.sum()/3600:.2f} h)")

    return {
        "cohort": label, "n_rounds": len(rows), "rounds": rows,
        "mean_oob_cindex": float(oob.mean()), "std_oob_cindex": float(oob.std()),
        "mean_train_cindex": float(train.mean()), "std_train_cindex": float(train.std()),
        "overfitting_gap": float(train.mean() - oob.mean()),
        "mean_wall_time_sec": float(wt.mean()), "target_cindex": target_cindex,
    }


blca_paths = glob.glob("/scratch/sorkwos/ga2m_pilot/blca/round_*.npz")
brca_paths = glob.glob("/scratch/sorkwos/ga2m_pilot/brca/job_*/round_1.npz")

blca_summary = summarize("BLCA", blca_paths, target_cindex=0.62)
brca_summary = summarize("BRCA", brca_paths, target_cindex=0.61)

out = {"blca": blca_summary, "brca": brca_summary}
with open("/scratch/sorkwos/ga2m_pilot/pilot_stop_rule_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n>>> wrote /scratch/sorkwos/ga2m_pilot/pilot_stop_rule_summary.json")
