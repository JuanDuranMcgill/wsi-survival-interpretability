#!/usr/bin/env python3
"""Open finished feature files and check shapes, dtypes, finiteness, CONCH norm. Usage: check_outputs.py [N newest] """
import sys, glob, os, torch
fs = sorted(glob.glob("/data/processed/juan/pancan/features/*/*_emb16.pt"), key=os.path.getmtime)[-int(sys.argv[1] if len(sys.argv) > 1 else 20):]
bad = 0
for f in fs:
    o = torch.load(f, map_location="cpu", weights_only=False); k = len(o["coords"])
    ok = (o["uni2"].shape == (k, 1536) and o["conch"].shape == (k, 512) and o["conch_probs"].shape == (k, 9) and o["conch_probs8"].shape == (k, 8)
          and o["coords"].shape == (k, 2) and o["coords"].dtype == torch.int32 and all(o[x].dtype == torch.float16 for x in ("uni2", "conch", "conch_probs", "conch_probs8"))
          and all(bool(torch.isfinite(o[x]).all()) for x in ("uni2", "conch")) and abs(float(o["conch"].float().norm(dim=1).mean()) - 1) < 1e-2
          and o["meta"]["n_tiles"] == k)
    bad += not ok
    print("OK " if ok else "BAD", os.path.basename(f)[:30], k, tuple(o["uni2"].shape), tuple(o["conch"].shape), f"{os.path.getsize(f)/1e6:.0f}MB", f"norm={float(o['conch'].float().norm(dim=1).mean()):.4f}")
print(len(fs), "files,", bad, "bad")
