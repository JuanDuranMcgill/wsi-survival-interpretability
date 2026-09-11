#!/usr/bin/env python3
"""Isolate whether the bug is in centering or in what training finds.

Hand-sets main_mlps / A / B to compute the *exact* ground-truth g1/g12
functions (no training at all), then runs the same centering and recovery
correlation check. If this doesn't recover r=1.0, the bug is in
center_decomposition itself. If it does, the bug is in what SGD finds during
training (i.e. an under-constrained optimization landscape, not the algebra).
"""
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga2m_head import region_pairs, center_decomposition, check_sum_preserved, raw_interactions_from_projections
from scipy.stats import pearsonr

torch.manual_seed(0)
n, num_regions, d_model = 2000, 4, 16

g = torch.Generator().manual_seed(1)
H = torch.randn(n, num_regions, d_model, generator=g)

c = torch.randn(d_model, generator=g)
a0 = torch.randn(d_model, generator=g)
b1 = torch.randn(d_model, generator=g)
a1 = torch.randn(d_model, generator=g)
b0 = torch.randn(d_model, generator=g)

g1_true = H[:, 0, :] @ c
g12_true = (H[:, 0, :] @ a0) * (H[:, 1, :] @ b1) + (H[:, 1, :] @ a1) * (H[:, 0, :] @ b0)

# Hand-build main_raw / Ah / Bh using the EXACT ground-truth directions.
# main_mlp for region 0 = <h0, c> exactly (region 1,2,3 = 0).
main_raw = torch.zeros(n, num_regions)
main_raw[:, 0] = g1_true

# rank=8, only first slot used for the true signal, rest zero.
rank = 8
Ah = torch.zeros(n, num_regions, rank)
Bh = torch.zeros(n, num_regions, rank)
Ah[:, 0, 0] = H[:, 0, :] @ a0   # u_0 = a0-projection in slot 0
Bh[:, 0, 0] = H[:, 0, :] @ b0   # v_0 = b0-projection in slot 0
Ah[:, 1, 0] = H[:, 1, :] @ a1   # u_1 = a1-projection in slot 0
Bh[:, 1, 0] = H[:, 1, :] @ b1   # v_1 = b1-projection in slot 0

pairs = region_pairs(num_regions)
pair01 = pairs.index((0, 1))

# Sanity: does <u_0,v_1> + <u_1,v_0> actually equal g12_true with this construction?
inter_raw = raw_interactions_from_projections(Ah, Bh, pairs)
raw_check = inter_raw[:, pair01]
r_raw, _ = pearsonr(raw_check.numpy(), g12_true.numpy())
diff_raw = (raw_check - g12_true).abs().max().item()
print(f"raw f_01 vs true g12: max abs diff = {diff_raw:.3e}, Pearson r = {r_raw:.4f}  (should be ~0 diff, r~1.0)")

main_c, inter_c, intercept, info = center_decomposition(main_raw, Ah, Bh, pairs)
ok, worst = check_sum_preserved(main_raw, inter_raw, main_c, inter_c, intercept)
print(f"sum preserved: {ok} (max diff {worst:.3e})")

r_main, p_main = pearsonr(main_c[:, 0].numpy(), g1_true.numpy())
r_int, p_int = pearsonr(inter_c[:, pair01].numpy(), g12_true.numpy())
print(f"ORACLE (hand-set exact weights, no training):")
print(f"  main_c[:,0]        vs true g1  : Pearson r = {r_main:+.4f}  (p={p_main:.1e})")
print(f"  inter_c[:,(0,1)]   vs true g12 : Pearson r = {r_int:+.4f}  (p={p_int:.1e})")

print(f"\ncentering_info: {info}")
print("\n=== RESULT:", "PASS (bug is in training dynamics)" if (r_main > 0.99 and r_int > 0.99)
      else "FAIL (bug is in center_decomposition itself)", "===")
