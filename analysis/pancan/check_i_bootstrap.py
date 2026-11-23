"""Check I verdicts for a pan-carcinoma run: faithful (exact permutation) and prognostic (patient bootstrap),
with comparable pairs restricted to the same cancer type (Beast, env audit).

  python -I check_i_bootstrap.py <run_dir>/shapley [--B 1000]   -> <run_dir>/shapley/check_i.json

Reuses the paper's patient_bootstrap_shapley.run_cohort unchanged, except that its comparable-pair matrix K is
multiplied by 'same type' (the payoffs check_i.py saved are within type; the run's point check confirms they match).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import patient_bootstrap_shapley as pbs  # noqa: E402

from analysis.pancan.verdicts import faithful, prognostic  # noqa: E402

_paper_load_rounds = pbs.load_rounds


def _load_rounds_within(rounds_dir, results_dir):
    rounds = _paper_load_rounds(rounds_dir, results_dir)
    for rd in rounds:
        ty = np.load(Path(rounds_dir) / f"round_{rd['round']:03d}_local.npz")["types"].astype(str)
        rd["K"] = rd["K"] * (ty[:, None] == ty[None, :]).astype(np.float32)
    return rounds


def bootstrap(shapley_dir, B=1000, seed=20261112, chunk=250):
    pbs.load_rounds = _load_rounds_within
    pbs.PAPER.setdefault("pancan", {})
    try:
        info, reps = pbs.run_cohort("pancan", str(shapley_dir), str(shapley_dir), B, seed, chunk)
    finally:
        pbs.load_rounds = _paper_load_rounds
    return info, {name: pbs.summarise(reps[name]) for name in pbs.CORRS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("shapley_dir")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args(argv)
    d = Path(a.shapley_dir)
    info, boot = bootstrap(d, a.B)
    agg = json.loads((d / "region_shapley.json").read_text())
    names = agg["region_names"]
    fw = [agg["per_region"][n]["fusion_weight"]["mean"] for n in names]
    share = [agg["per_region"][n]["local_attribution_share"]["mean"] for n in names]
    f = faithful(fw, share)
    rho = info["point"]["local_share_vs_shapley"]
    out = {**info, "bootstrap": boot, "compartments": names,
           "faithful": f, "prognostic": {"rho": rho, "ci95": boot["local_share_vs_shapley"]["ci95"],
                                         "pass": prognostic(rho, boot["local_share_vs_shapley"]["ci95"])}}
    (d / "check_i.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({"faithful": f, "prognostic": out["prognostic"]}, default=float))


if __name__ == "__main__":
    main()
