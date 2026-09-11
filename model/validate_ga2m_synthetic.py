#!/usr/bin/env python3
"""Synthetic identifiability check for the GA2M head (DESIGN_GA2M_HEAD.md, Risks).

"Validate on synthetic data first: generate y = g1(h_1) + g12(h_1,h_2) + noise
with known components and confirm recovery." Ground truth is built with the
same functional form as the model (linear main effect, rank-1 bilinear
interaction) so recovery is a fair test of the *centering*, not of whether an
MLP/bilinear form can approximate an unrelated function.

History: the first centering implementation (bin on the main-effect *output*,
subtract a conditional mean) failed this test outright — Pearson r ~ -0.4
against the true components despite the model fitting the target almost
exactly. Root cause: conditioning on a many-to-one, moving-target quantity
(the main effect's own output) rather than the input. Fixed with an exact,
closed-form centering of the bilinear projections themselves (see
ga2m_head.py's module docstring); acceptance is now r > 0.9 against both true
components — "correct total plus correct sparsity isn't sufficient", per the
spec update.

Checks, any failure -> nonzero exit code:
  1. Sum-preservation: intercept + centered main + centered interactions
     equals the raw (uncentered) sum, per patient, to numerical precision.
  2. Recovery under independent regions: centered main effect (region 0)
     correlates with true g1, centered interaction (0,1) correlates with
     true g12, both r > 0.9.
  3. Null regions: regions 2 and 3 carry no true signal and should stay
     small relative to the real signal.
  4. Recovery under correlated regions: same test, but h1 is partially
     explained by h0 (regions on the same slide are not independent in
     practice). Checked with the plain exact centering; if it leaks, the
     residualized fallback (ga2m_head.residualize_and_recenter) is tried
     too, since the spec flags this as the expected failure mode of the
     independence assumption, not a bug.

Runs on CPU in well under a minute.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ga2m_head import (  # noqa: E402
    GA2MHead, region_pairs, check_sum_preserved, raw_interactions_from_projections,
    center_decomposition, residualize_and_recenter,
)


def make_synthetic(n, num_regions, d_model, seed, rho=0.0):
    """rho > 0 makes region 1's embedding partially explained by region 0's,
    modelling that regions on the same slide are not independent.

    BUG FIXED HERE (found by the other session after my "joint training
    doesn't converge" report turned out to be wrong): the ground-truth
    direction vectors (c, a0, b1, a1, b0) used to be drawn from the *same*
    per-call generator as the data, seeded by `seed`. Since this function is
    called once with seed=1 for the training split and once with seed=2 for
    the eval split, the two splits got *different true functions* — the
    model fit train perfectly (train MSE ~0.001) and was meaningless on eval
    (MSE ~2, worse than predicting the mean), which looks exactly like a
    training-dynamics failure but has nothing to do with optimization,
    centering, rank, or regularization (all of which I swept without effect,
    because none of them were ever the problem). Fixed by drawing the truth
    vectors from fixed generators seeded independently of the data seed, so
    every split shares the same ground truth and only the sampled h's
    differ.
    """
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(n, num_regions, d_model, generator=g)
    if rho > 0.0:
        H[:, 1, :] = rho * H[:, 0, :] + (1 - rho ** 2) ** 0.5 * H[:, 1, :]

    c = torch.randn(d_model, generator=torch.Generator().manual_seed(100))
    a0 = torch.randn(d_model, generator=torch.Generator().manual_seed(101))
    b1 = torch.randn(d_model, generator=torch.Generator().manual_seed(102))
    a1 = torch.randn(d_model, generator=torch.Generator().manual_seed(103))
    b0 = torch.randn(d_model, generator=torch.Generator().manual_seed(104))

    g1_true = H[:, 0, :] @ c
    g12_true = (H[:, 0, :] @ a0) * (H[:, 1, :] @ b1) + (H[:, 1, :] @ a1) * (H[:, 0, :] @ b0)
    noise = 0.1 * torch.randn(n, generator=g)
    y = g1_true + g12_true + noise
    y = (y - y.mean()) / y.std()
    return H, y, g1_true, g12_true


def sum_preservation_check(head, H, n_trials=20):
    worst = 0.0
    ok_all = True
    rng = torch.Generator().manual_seed(0)
    for _ in range(n_trials):
        idx = torch.randint(0, H.shape[0], (64,), generator=rng)
        batch = H[idx]
        main_raw, Ah, Bh = head.raw_projections(batch)
        inter_raw = raw_interactions_from_projections(Ah, Bh, head.pairs)
        risk, main_c, inter_c, intercept, _ = head(batch)
        ok, diff = check_sum_preserved(main_raw, inter_raw, main_c, inter_c, intercept)
        ok_all = ok_all and ok
        worst = max(worst, diff)
    return ok_all, worst


def train_head(head, H_train, y_train, H_eval=None, y_eval=None, epochs=300, batch_size=64,
                lr=1e-3, lam_int=1e-2, weight_decay=1e-2, patience=15):
    """Early-stops on held-out MSE (H_eval/y_eval), not on the attribution
    correlation — using labels for model selection is standard practice and
    is not the same as leaking the true-component check into training.

    The first version of this test used lr=3e-3, weight_decay=1e-4, 60 fixed
    epochs, no held-out MSE check at all — it silently let the model
    memorize the 4000 training points (train MSE ~0.0001, held-out MSE
    ~2.4 — worse than predicting the mean) and only checked attribution
    correlation on the eval set, which is meaningless for a model that never
    generalized. Fixed by tracking held-out MSE and stopping when it stops
    improving, and by making held-out MSE part of what this script reports
    and gates on.
    """
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = H_train.shape[0]
    best_eval_mse = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch, target = H_train[idx], y_train[idx]
            risk, main_c, inter_c, intercept, _ = head(batch)
            mse = ((risk - target) ** 2).mean()
            loss = mse + lam_int * head.l1_interaction_penalty(inter_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(mse.item()) * len(idx)
        train_mse = total_loss / n

        eval_mse = train_mse
        if H_eval is not None:
            head.eval()
            with torch.no_grad():
                risk, *_ = head(H_eval)
                eval_mse = ((risk - y_eval) ** 2).mean().item()
            if eval_mse < best_eval_mse - 1e-5:
                best_eval_mse = eval_mse
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
                best_epoch = ep
                bad_epochs = 0
            else:
                bad_epochs += 1

        if ep % 20 == 0 or ep == 1:
            print(f"    epoch {ep:3d}  train_mse={train_mse:.4f}  eval_mse={eval_mse:.4f}"
                  f"  (best {best_eval_mse:.4f} @ {best_epoch})")

        if H_eval is not None and bad_epochs >= patience:
            print(f"    early stop at epoch {ep} (no improvement for {patience} epochs); "
                  f"restoring epoch {best_epoch} (eval_mse={best_eval_mse:.4f})")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_eval_mse


def recovery_check(head, H_eval, g1_true_eval, g12_true_eval, num_regions, label):
    with torch.no_grad():
        risk, main_c, inter_c, intercept, info = head(H_eval)

    pairs = region_pairs(num_regions)
    pair01 = pairs.index((0, 1))

    r_main0, p_main0 = pearsonr(main_c[:, 0].numpy(), g1_true_eval.numpy())
    rho_main0, _ = spearmanr(main_c[:, 0].numpy(), g1_true_eval.numpy())
    r_int01, p_int01 = pearsonr(inter_c[:, pair01].numpy(), g12_true_eval.numpy())
    rho_int01, _ = spearmanr(inter_c[:, pair01].numpy(), g12_true_eval.numpy())

    print(f"  [{label}] main effect region 0  vs true g1 :  Pearson r={r_main0:+.3f} (p={p_main0:.1e})  Spearman rho={rho_main0:+.3f}")
    print(f"  [{label}] interaction (0,1)     vs true g12:  Pearson r={r_int01:+.3f} (p={p_int01:.1e})  Spearman rho={rho_int01:+.3f}")

    active_main_mag = main_c[:, :2].abs().mean().item()
    null_main_mag = main_c[:, 2:].abs().mean().item()
    active_inter_mag = inter_c[:, pair01].abs().mean().item()
    null_pair_idx = [p for p in range(len(pairs)) if p != pair01 and (2 in pairs[p] or 3 in pairs[p])]
    null_inter_mag = inter_c[:, null_pair_idx].abs().mean().item()
    print(f"  [{label}] |main effect| active (0,1) = {active_main_mag:.4f}   null (2,3) = {null_main_mag:.4f}")
    print(f"  [{label}] |interaction| active (0,1) = {active_inter_mag:.4f}   null pairs = {null_inter_mag:.4f}")

    passed = (
        r_main0 > 0.9 and r_int01 > 0.9
        and null_main_mag < 0.35 * active_main_mag
        and null_inter_mag < 0.35 * active_inter_mag
    )
    return passed, dict(r_main0=r_main0, r_int01=r_int01, info=info)


def run_scenario(label, rho, num_regions=4, d_model=16, n_train=4000, n_eval=4000,
                  use_residualization=False):
    print(f"\n{'='*70}\n=== Scenario: {label} (rho={rho}, residualize={use_residualization}) ===\n{'='*70}")
    torch.manual_seed(0)
    H_train, y_train, _, _ = make_synthetic(n_train, num_regions, d_model, seed=1, rho=rho)
    H_eval, y_eval, g1_true_eval, g12_true_eval = make_synthetic(n_eval, num_regions, d_model, seed=2, rho=rho)

    head = GA2MHead(num_regions=num_regions, d_model=d_model, rank=8, main_hidden=32,
                     use_residualization=use_residualization)

    ok, worst = sum_preservation_check(head, H_train)
    print(f"  sum-preserved (untrained): {ok}  (max abs diff = {worst:.3e})")
    if not ok:
        print("FAIL: centering does not preserve the total sum (untrained head).")
        return False

    print("  training (early-stopping on held-out MSE)...")
    head, best_eval_mse = train_head(head, H_train, y_train, H_eval=H_eval, y_eval=y_eval)

    ok, worst = sum_preservation_check(head, H_train)
    print(f"  sum-preserved (trained):   {ok}  (max abs diff = {worst:.3e})")
    if not ok:
        print("FAIL: centering broke sum-preservation after training.")
        return False

    # Generalization gate: recovery correlation is meaningless if the model
    # never learned the true function in the first place. y is standardized
    # to unit variance, so MSE ~1.0 is "predict the mean"; require the model
    # to actually beat that by a wide margin before trusting attributions.
    print(f"  held-out MSE = {best_eval_mse:.4f}  (1.0 = predicting the mean; noise floor ~0.01)")
    if best_eval_mse > 0.3:
        print(f"FAIL: model did not generalize (held-out MSE {best_eval_mse:.4f}) — "
              f"attribution recovery is not a meaningful check on a model that didn't learn the function.")
        return False

    passed, details = recovery_check(head, H_eval, g1_true_eval, g12_true_eval, num_regions, label)
    print(f"  centering_info: {details['info']}")
    return passed


def main():
    results = {}

    # Scenario 1: independent regions, plain exact centering. This is the
    # core test the spec requires (r > 0.9 both components).
    results["independent"] = run_scenario("independent regions", rho=0.0)

    # Scenario 2: correlated regions (same-slide regions are not independent
    # in practice), still with the plain exact centering. Per the spec's own
    # caveat, this may leak (E[vtilde_s | h_r] != 0 exactly under
    # correlation) — that would be an expected, documented limitation, not a
    # surprise.
    results["correlated_plain"] = run_scenario("correlated regions, plain centering", rho=0.6)

    # Scenario 3: same correlated case, with the residualized fallback.
    results["correlated_residualized"] = run_scenario(
        "correlated regions, residualized centering", rho=0.6, use_residualization=True)

    print(f"\n{'='*70}\n=== SUMMARY ===\n{'='*70}")
    for k, v in results.items():
        print(f"  {k:30s}: {'PASS' if v else 'FAIL'}")

    # Gating logic: the core (independent-regions) test must pass — that is
    # the primary spec requirement. If correlated regions leak under plain
    # centering but the residualized fallback fixes it, that is a documented,
    # acceptable limitation (use_residualization=True for real data), not a
    # blocking failure.
    core_ok = results["independent"]
    correlated_ok = results["correlated_plain"] or results["correlated_residualized"]

    if not core_ok:
        print("\nFAIL: core (independent-regions) recovery test did not pass.")
        sys.exit(1)
    if not correlated_ok:
        print("\nFAIL: neither plain nor residualized centering recovers correctly "
              "under correlated regions. Real slide data has correlated regions, "
              "so this blocks moving to GPU training.")
        sys.exit(1)
    if not results["correlated_plain"] and results["correlated_residualized"]:
        print("\nNOTE: correlated regions leak under plain centering but are fixed by "
              "residualization. Train the real pilots with --use_residualization "
              "(or hardcode it on) once that flag exists in train_ga2m.py.")

    print("\n=== RESULT: PASS ===")


if __name__ == "__main__":
    main()
