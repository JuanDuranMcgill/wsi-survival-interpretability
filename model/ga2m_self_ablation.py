#!/usr/bin/env python3
"""GA2M's own exact ablation (DESIGN_GA2M_HEAD.md Section 3/4.3, comparison 3).

For each pilot round and each region r, recompute the OOB c-index with every
term touching r dropped (main effect f_r AND every interaction f_rs for all
s), using only what's already saved in the pilot's round_*.npz files -- no
model, no GPU, pure post-hoc arithmetic on saved risk components.

    ablated_risk = risk - main_effects[:, r] - interactions[:, r, :].sum(1)

(interactions is the saved (n_oob, R, R) symmetric, zero-diagonal matrix, so
row r summed over all columns is exactly the total of every pair touching r,
each counted once.)

delta_r = baseline_cindex - ablated_cindex_r  (positive = region r helps)

This is then Spearman-correlated against GA2M's own mean |main effect| per
region -- a self-consistency test (does the architecture's own attribution
agree with the architecture's own ablation?), which is a different, and per
the request, the decisive question -- NOT the cross-model comparison against
the original architecture's ablation study done previously.

Also reports: how many interaction pairs the L1 penalty drove near zero, and
whether necrosis's signal sits in interactions rather than main effects.
"""
import glob
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr


def cindex(times, events, risks):
    """Harrell c-index, identical definition to analysis/patient_error.py's
    global_cindex, for consistency with the rest of the project."""
    t, e, r = np.asarray(times, float), np.asarray(events, float), np.asarray(risks, float)
    ti, tj, ei = t[:, None], t[None, :], e[:, None]
    earlier = (ti < tj) & (ei == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    conc = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0)
    total = earlier.sum()
    return float(conc.sum() / total) if total else float("nan")


def process_cohort(cohort, paths):
    paths = sorted(paths)
    region_names = None
    per_region_deltas = []   # list of arrays (R,), one per round
    all_main_effects = []    # list of (n_oob, R)
    all_interactions = []    # list of (n_oob, R, R)
    baseline_check = []      # (saved_oob_cindex, recomputed_cindex) per round

    for p in paths:
        d = np.load(p, allow_pickle=True)
        names = list(d["region_names"])
        if region_names is None:
            region_names = names
        assert names == region_names, f"region order mismatch in {p}"

        main = d["main_effects"]          # (n_oob, R)
        inter = d["interactions"]         # (n_oob, R, R)
        intercept = float(d["intercept"])
        risk = d["risk"]                  # (n_oob,)
        times = d["pfi_time"]
        events = d["pfi_event"]
        saved_oob_cindex = float(d["oob_cindex"])

        R = main.shape[1]

        # Sanity: recompute the full-model risk from components and confirm
        # it matches what was saved (validates the saved "risk" field is
        # exactly intercept + main.sum(1) + upper-triangle interaction sum).
        recon_risk = intercept + main.sum(axis=1) + inter.sum(axis=(1, 2)) / 2.0
        recon_diff = float(np.abs(recon_risk - risk).max())

        recomputed_baseline_c = cindex(times, events, risk)
        baseline_check.append((saved_oob_cindex, recomputed_baseline_c, recon_diff))

        deltas = np.zeros(R)
        for r in range(R):
            ablated_risk = risk - main[:, r] - inter[:, r, :].sum(axis=1)
            ablated_c = cindex(times, events, ablated_risk)
            deltas[r] = recomputed_baseline_c - ablated_c
        per_region_deltas.append(deltas)

        all_main_effects.append(main)
        all_interactions.append(inter)

    per_region_deltas = np.stack(per_region_deltas)   # (n_rounds, R)
    mean_delta = per_region_deltas.mean(axis=0)
    std_delta = per_region_deltas.std(axis=0)

    main_stack = np.concatenate(all_main_effects, axis=0)         # (total_oob, R)
    mean_abs_main = np.abs(main_stack).mean(axis=0)

    inter_stack = np.concatenate(all_interactions, axis=0)         # (total_oob, R, R)
    R = len(region_names)
    pair_mean_abs = np.zeros((R, R))
    for r in range(R):
        for s in range(R):
            if r == s:
                continue
            pair_mean_abs[r, s] = np.abs(inter_stack[:, r, s]).mean()

    rho, p_rho = spearmanr(mean_abs_main, mean_delta)
    r_pear, p_pear = pearsonr(mean_abs_main, mean_delta)

    print(f"\n{'='*84}\n=== {cohort.upper()}: sanity check (saved risk/cindex vs recomputed) ===\n{'='*84}")
    for i, (saved_c, recomp_c, recon_diff) in enumerate(baseline_check, 1):
        # npz stores some fields as float32, so ~1e-5 differences are expected
        # rounding, not a real inconsistency; only flag genuine mismatches.
        flag = "" if abs(saved_c - recomp_c) < 1e-3 and recon_diff < 1e-3 else "  <-- MISMATCH"
        print(f"  round {i:2d}: saved oob_cindex={saved_c:.4f}  recomputed={recomp_c:.4f}  "
              f"max risk recon diff={recon_diff:.2e}{flag}")

    print(f"\n=== {cohort.upper()}: GA2M's own exact-ablation delta vs its own |main effect| ===")
    print(f"{'region':50s} {'|main eff|':>11} {'own-ablation Δ':>15} {'Δ std':>8}")
    order = np.argsort(-mean_abs_main)
    for i in order:
        print(f"{region_names[i]:50s} {mean_abs_main[i]:11.4f} {mean_delta[i]:15.5f} {std_delta[i]:8.5f}")

    print(f"\n  Spearman rho = {rho:+.3f}  (p={p_rho:.3f})   [self-consistency: comparison 3]")
    print(f"  Pearson  r   = {r_pear:+.3f}  (p={p_pear:.3f})")

    # Interaction sparsity: mean |interaction| per unique pair.
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    pair_vals = np.array([pair_mean_abs[i, j] for (i, j) in pairs])
    max_pair = pair_vals.max()
    near_zero_thresh = 0.05 * max_pair
    n_near_zero = int((pair_vals < near_zero_thresh).sum())

    print(f"\n=== {cohort.upper()}: interaction sparsity (mean |interaction| per pair) ===")
    order_p = np.argsort(-pair_vals)
    for idx in order_p:
        i, j = pairs[idx]
        flag = "  <-- near-zero (<5% of max)" if pair_vals[idx] < near_zero_thresh else ""
        print(f"  {region_names[i]:40s} x {region_names[j]:40s}  {pair_vals[idx]:.4f}{flag}")
    print(f"\n  {n_near_zero} / {len(pairs)} pairs below 5% of the max pair value "
          f"({near_zero_thresh:.4f}, max={max_pair:.4f})")

    # Necrosis-specific check.
    necrosis_names = [n for n in region_names if "necrosis" in n.lower()]
    necrosis_report = None
    if necrosis_names:
        nname = necrosis_names[0]
        nidx = region_names.index(nname)
        main_rank = int((np.argsort(-mean_abs_main) == nidx).nonzero()[0][0]) + 1
        total_interaction_involvement = pair_mean_abs[nidx, :].sum()
        # rank among regions by total interaction involvement
        total_inter_per_region = pair_mean_abs.sum(axis=1)
        inter_rank = int((np.argsort(-total_inter_per_region) == nidx).nonzero()[0][0]) + 1
        print(f"\n=== {cohort.upper()}: necrosis check ({nname}) ===")
        print(f"  |main effect| = {mean_abs_main[nidx]:.4f}  (rank {main_rank}/{R} by main effect)")
        print(f"  total interaction involvement = {total_interaction_involvement:.4f}  "
              f"(rank {inter_rank}/{R} by interaction involvement)")
        in_interactions_more = inter_rank < main_rank
        print(f"  {'YES' if in_interactions_more else 'NO'}: necrosis ranks "
              f"{'higher' if in_interactions_more else 'not higher'} by interaction involvement "
              f"than by main effect")
        necrosis_report = {
            "region_name": nname, "main_effect_rank": main_rank, "interaction_rank": inter_rank,
            "mean_abs_main_effect": float(mean_abs_main[nidx]),
            "total_interaction_involvement": float(total_interaction_involvement),
            "ranks_higher_in_interactions": bool(in_interactions_more),
        }

    return {
        "cohort": cohort, "region_names": region_names, "n_rounds": len(paths),
        "mean_abs_main_effect": mean_abs_main.tolist(),
        "own_ablation_mean_delta": mean_delta.tolist(),
        "own_ablation_std_delta": std_delta.tolist(),
        "spearman_rho_self_consistency": float(rho), "spearman_p": float(p_rho),
        "pearson_r_self_consistency": float(r_pear), "pearson_p": float(p_pear),
        "pair_mean_abs_interaction": {f"{region_names[i]}__{region_names[j]}": float(pair_mean_abs[i, j])
                                       for (i, j) in pairs},
        "n_pairs_near_zero_5pct": n_near_zero, "n_pairs_total": len(pairs),
        "near_zero_threshold": float(near_zero_thresh),
        "necrosis_check": necrosis_report,
        "sanity_check_saved_vs_recomputed": [
            {"round": i + 1, "saved_oob_cindex": s, "recomputed_oob_cindex": rc, "max_risk_recon_diff": rd}
            for i, (s, rc, rd) in enumerate(baseline_check)
        ],
    }


blca_paths = glob.glob("/scratch/sorkwos/ga2m_pilot/blca/round_*.npz")
brca_paths = glob.glob("/scratch/sorkwos/ga2m_pilot/brca/job_*/round_1.npz")

out = {}
out["blca"] = process_cohort("blca", blca_paths)
out["brca"] = process_cohort("brca", brca_paths)

with open("/scratch/sorkwos/ga2m_pilot/ga2m_self_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n>>> wrote /scratch/sorkwos/ga2m_pilot/ga2m_self_ablation.json")
