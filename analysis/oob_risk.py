#!/usr/bin/env python3
"""Reconstruct out-of-bag risk scores by replaying the bootstrap seeds.

WHY THIS EXISTS
---------------
`bootstrap_rounds()` in the training scripts draws its training set WITH
replacement and validates on the out-of-bag remainder:

    train_idx = np.random.choice(idx, int(train_frac * N), replace=True)
    test_idx  = idx[mask]                                    # out-of-bag
    valC_by_epoch = train_one_round(model, train_loader, test_loader, ...)

`val_cidx` is therefore an honest out-of-bag score, and it is where the paper's
c-index of 0.600 (BLCA) and 0.632 (BRCA) comes from.

But the per-patient risks that get saved come from `full_loader`, which is the
ENTIRE cohort:

    full_loader = DataLoader(dataset, ...)
    pids, risks, mean_attn = extract_risks(model_ep, full_loader, ...)

With `train_frac = 0.8` and sampling with replacement, a given patient is
out-of-bag with probability about exp(-0.8) = 0.449, so roughly 55% of patients
are IN-SAMPLE in any round. Averaging those saved risks across rounds yields
`mean_risk`, whose c-index measures 0.90, not 0.60.

`mean_risk` is the vector the MED3pa high-error labels were built from. So the
reliability profiles describe where an overfit model fits its own training data,
not where it generalises.

`test_idx` was never saved and the checkpoints were deleted (`os.remove`), so
the only route that avoids retraining is to reconstruct out-of-bag membership
from the seeds. The split is fully deterministic:

    seed_attempt = global_round            # = seed_offset + r
    np.random.seed(seed_attempt)
    train_idx = np.random.choice(idx, int(train_frac * N), replace=True)
    # then, while the out-of-bag set has < MIN_VAL_EVENTS events:
    seed_attempt = global_round * 1000 + retry

Patient ORDER matters and is filesystem-dependent (`os.listdir` in one branch of
the dataset constructor), so we do NOT reconstruct it. Every round's npz saves
`patient_ids` in exactly the dataset order used by `np.random.choice`, so we
read the ordering from the data instead of guessing it.

SELF-VALIDATION
---------------
This is checkable. If the reconstruction is right, the out-of-bag-only mean risk
should give a c-index near the paper's 0.600 / 0.632. If it comes out near 0.90
the reconstruction is wrong and nothing downstream should be used. The script
reports both and refuses to look successful without saying which happened.

Usage:
    python oob_risk.py --cohort blca \
        --rounds-glob "$HOME/data/blca/rounds/job_*/round_*/epoch_*.npz" \
        --cdr-xlsx "$CDR_XLSX" \
        --seed-offset-per-job 17 \
        --out results/oob_risk_blca.json

Seed offsets, from the SLURM launchers:
    BLCA          : --seed-offset-per-job 17          (job_N -> 17*N)
    BRCA main     : --seed-offset-per-job 10          (job_N -> 10*N)
    BRCA top-up   : --seed-offset-base 100 --seed-offset-per-job 5
Confirm against your own submit scripts before trusting the output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.8
MIN_VAL_EVENTS = 20
MAX_SEED_RETRIES = 20


# ----------------------------------------------------------------------
def replay_split(n, events_in_dataset_order, global_round,
                 train_frac=TRAIN_FRAC,
                 min_val_events=MIN_VAL_EVENTS,
                 max_retries=MAX_SEED_RETRIES):
    """Reproduce one round's out-of-bag mask exactly as training built it.

    Mirrors the seed retry loop in `bootstrap_rounds` line for line. Returns
    (oob_mask, seed_used, n_val_events, hit_retry_limit).
    """
    idx = np.arange(n)
    seed_attempt = global_round
    retry = 0
    while True:
        np.random.seed(seed_attempt)
        train_idx = np.random.choice(idx, int(train_frac * n), replace=True)
        mask = np.ones(n, bool)
        mask[train_idx] = False
        test_idx = idx[mask]

        if len(test_idx) == 0:
            # The training code's fallback path, reproduced faithfully. It calls
            # np.random.permutation immediately after the choice above, so the
            # generator state matters and must not be reseeded here.
            perm = np.random.permutation(n)
            split = int(0.8 * n)
            test_idx = perm[split:]
            mask = np.zeros(n, bool)
            mask[test_idx] = True

        val_events = int(sum(events_in_dataset_order[i] for i in test_idx))
        if val_events >= min_val_events:
            return mask, seed_attempt, val_events, False
        retry += 1
        if retry >= max_retries:
            return mask, seed_attempt, val_events, True
        seed_attempt = global_round * 1000 + retry


# ----------------------------------------------------------------------
def collect_rounds(rounds_glob, pick="last"):
    """One (job, round, risks, patient_ids) per completed round."""
    files = sorted(glob.glob(rounds_glob))
    if not files:
        raise SystemExit(f"no files matched: {rounds_glob}")

    grouped: dict[tuple, list] = {}
    for f in files:
        parts = os.path.normpath(f).split(os.sep)
        jm = next((re.search(r"job_?(\d+)", p) for p in parts
                   if re.search(r"job_?(\d+)", p)), None)
        rm = next((re.search(r"round_?(\d+)", p) for p in parts
                   if re.search(r"round_?(\d+)", p)), None)
        if jm is None or rm is None:
            continue
        job, rnd = int(jm.group(1)), int(rm.group(1))
        ep = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        grouped.setdefault((job, rnd), []).append((ep, f))

    out = []
    for (job, rnd), entries in sorted(grouped.items()):
        ep, f = max(entries, key=lambda e: e[0]) if pick == "last" else min(entries, key=lambda e: e[0])
        d = np.load(f, allow_pickle=True)
        if "risks" not in d or "patient_ids" not in d:
            continue
        out.append({
            "job": job, "round": rnd, "epoch": ep, "file": f,
            "patient_ids": np.array([str(p)[:12] for p in d["patient_ids"]]),
            "risks": np.asarray(d["risks"], dtype=float),
            "val_cidx": float(d["val_cidx"]) if "val_cidx" in d else float("nan"),
        })
    return out


def cindex(times, events, risks, mask=None):
    t, e, r = (np.asarray(x, float) for x in (times, events, risks))
    if mask is not None:
        m = np.asarray(mask, bool)
        t, e, r = t[m], e[m], r[m]
    keep = np.isfinite(r)
    t, e, r = t[keep], e[keep], r[keep]
    if len(t) < 2:
        return float("nan"), 0
    earlier = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    np.fill_diagonal(earlier, False)
    ri, rj = r[:, None], r[None, :]
    conc = np.where(earlier, np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0)), 0.0)
    tot = int(earlier.sum())
    return (float(conc.sum() / tot) if tot else float("nan")), tot


def load_survival(cdr_xlsx, pids):
    df = pd.read_excel(cdr_xlsx)[["bcr_patient_barcode", "PFI", "PFI.time"]].copy()
    df["PFI.time"] = pd.to_numeric(df["PFI.time"], errors="coerce")
    df["PFI"] = pd.to_numeric(df["PFI"], errors="coerce")
    df = df.dropna(subset=["PFI.time", "PFI"]).drop_duplicates("bcr_patient_barcode")
    lut = df.set_index("bcr_patient_barcode")
    times = np.full(len(pids), np.nan)
    events = np.full(len(pids), np.nan)
    for k, p in enumerate(pids):
        if p in lut.index:
            times[k] = float(lut.at[p, "PFI.time"])
            events[k] = int(lut.at[p, "PFI"])
    return times, events


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["blca", "brca"])
    ap.add_argument("--rounds-glob", required=True)
    ap.add_argument("--cdr-xlsx", required=True)
    ap.add_argument("--seed-offset-per-job", type=int, default=None,
                    help="Single linear scheme: offset = base + per_job * job.")
    ap.add_argument("--seed-offset-base", type=int, default=0)
    ap.add_argument("--seed-offsets", default=None,
                    help='Explicit per-job map as JSON, e.g. \'{"0":0,"1":10}\'. '
                         "Required when a cohort used more than one scheme, as "
                         "BRCA did (main jobs job*10, top-up 100+(job-10)*5).")
    ap.add_argument("--pick", default="last", choices=["last", "first"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seed_offsets is None and args.seed_offset_per_job is None:
        raise SystemExit("give --seed-offset-per-job or --seed-offsets")
    offsets = {int(k): int(v) for k, v in json.loads(args.seed_offsets).items()} \
        if args.seed_offsets else None

    rounds = collect_rounds(args.rounds_glob, args.pick)
    print(f"[{args.cohort}] {len(rounds)} rounds found")

    pids = rounds[0]["patient_ids"]
    n = len(pids)
    for rd in rounds:
        if not np.array_equal(rd["patient_ids"], pids):
            raise SystemExit(
                f"patient order differs in {rd['file']}. Reconstruction is only "
                "valid if every round used the same dataset ordering."
            )
    print(f"[{args.cohort}] {n} patients, consistent ordering across all rounds")

    times, events = load_survival(args.cdr_xlsx, pids)
    ok = np.isfinite(times) & np.isfinite(events)
    ev_int = np.where(ok, events, 0).astype(int)

    # Accumulate risk over only the rounds where each patient was out-of-bag.
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n, dtype=int)
    insample_sum = np.zeros(n)
    insample_cnt = np.zeros(n, dtype=int)
    retry_limit_hits, seeds = 0, []

    for rd in rounds:
        if offsets is not None:
            if rd["job"] not in offsets:
                raise SystemExit(
                    f"job {rd['job']} missing from --seed-offsets; "
                    "every job dir present must have an offset"
                )
            base = offsets[rd["job"]]
        else:
            base = args.seed_offset_base + args.seed_offset_per_job * rd["job"]
        g = base + rd["round"]
        oob, seed, nev, hit = replay_split(n, ev_int, g)
        seeds.append({"job": rd["job"], "round": rd["round"], "global_round": g,
                      "seed_used": seed, "n_val_events": nev,
                      "oob_size": int(oob.sum()), "hit_retry_limit": hit,
                      "val_cidx": rd["val_cidx"]})
        retry_limit_hits += int(hit)
        r = rd["risks"]
        oob_sum[oob] += r[oob]
        oob_cnt[oob] += 1
        insample_sum[~oob] += r[~oob]
        insample_cnt[~oob] += 1

    with np.errstate(invalid="ignore"):
        oob_risk = np.where(oob_cnt > 0, oob_sum / np.maximum(oob_cnt, 1), np.nan)
        insample_risk = np.where(insample_cnt > 0, insample_sum / np.maximum(insample_cnt, 1), np.nan)
    all_risk = (oob_sum + insample_sum) / np.maximum(oob_cnt + insample_cnt, 1)

    c_oob, p_oob = cindex(times[ok], events[ok], oob_risk[ok])
    c_ins, _ = cindex(times[ok], events[ok], insample_risk[ok])
    c_all, _ = cindex(times[ok], events[ok], all_risk[ok])
    mean_val = float(np.nanmean([s["val_cidx"] for s in seeds]))

    expected = 0.600 if args.cohort == "blca" else 0.632
    verdict = ("PLAUSIBLE" if abs(c_oob - expected) < 0.06 else "SUSPECT")

    print(f"\n[{args.cohort}] mean per-round val_cidx (ground truth) = {mean_val:.4f}")
    print(f"[{args.cohort}] c-index from OOB-only mean risk         = {c_oob:.4f}   <- should be near {expected}")
    print(f"[{args.cohort}] c-index from in-sample-only mean risk   = {c_ins:.4f}")
    print(f"[{args.cohort}] c-index from all-rounds mean risk       = {c_all:.4f}   <- what the paper's labels used")
    print(f"[{args.cohort}] expected OOB fraction {np.exp(-TRAIN_FRAC):.3f}, "
          f"observed {oob_cnt.sum()/(len(rounds)*n):.3f}")
    print(f"[{args.cohort}] patients never OOB: {(oob_cnt==0).sum()}   "
          f"rounds hitting retry limit: {retry_limit_hits}")
    print(f"\n[{args.cohort}] RECONSTRUCTION VERDICT: {verdict}")
    if verdict == "SUSPECT":
        print("  The OOB c-index does not match the published value. Do NOT use "
              "this output. Likely causes: wrong seed offsets, a different "
              "train_frac, or a patient ordering that differs from training.")

    rep = {
        "cohort": args.cohort, "endpoint": "PFI",
        "n_patients": int(n), "n_rounds": len(rounds),
        "train_frac": TRAIN_FRAC,
        "seed_offset_base": args.seed_offset_base,
        "seed_offset_per_job": args.seed_offset_per_job,
        "seed_offsets_explicit": offsets,
        "epoch_selection": args.pick,
        "rounds_glob": args.rounds_glob,
        "mean_per_round_val_cidx": mean_val,
        "cindex_oob_only": c_oob,
        "cindex_insample_only": c_ins,
        "cindex_all_rounds": c_all,
        "expected_cindex_from_paper": expected,
        "reconstruction_verdict": verdict,
        "expected_oob_fraction": float(np.exp(-TRAIN_FRAC)),
        "observed_oob_fraction": float(oob_cnt.sum() / (len(rounds) * n)),
        "n_patients_never_oob": int((oob_cnt == 0).sum()),
        "n_rounds_hit_retry_limit": retry_limit_hits,
        "oob_count_quantiles": {
            f"q{q}": float(np.quantile(oob_cnt, q / 100)) for q in (0, 25, 50, 75, 100)
        },
        "n_comparable_pairs_oob": p_oob,
        "per_round": seeds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{args.cohort}] wrote {args.out}")

    # Drop-in replacement for final_med3pa_input.npz, OOB risks only.
    npz = args.out.replace(".json", ".npz")
    np.savez(npz, patient_ids=pids, mean_risk=oob_risk,
             oob_count=oob_cnt, insample_risk=insample_risk,
             pfi_time=times, pfi_event=np.where(ok, events, np.nan))
    print(f"[{args.cohort}] wrote {npz}")
    print(f"[{args.cohort}] feed that to patient_error.py via --final-npz "
          f"to rebuild the labels on OOB predictions")


if __name__ == "__main__":
    main()
