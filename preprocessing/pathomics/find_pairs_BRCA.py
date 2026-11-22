#!/usr/bin/env python3
# Copied unchanged on 2026-11-21 from Trillium /scratch/sorkwos/multimodality/graph_subset_generation/find_pairs_BRCA.py (last modified 2026-04-20 23:41); only this line was added.
# find_pairs_BRCA.py
# Finds matching (WSI, JSONL) pairs for BRCA and runs extraction on a batch.
#
# Usage:
#   python find_pairs_BRCA.py --dry-run                    # print all pairs
#   python find_pairs_BRCA.py --run                        # run all pairs
#   python find_pairs_BRCA.py --run --batch 1 --total 4   # run batch 1 of 4

import os
import sys
import subprocess

# -------------------------------------------------------
# Paths  ← UPDATE JSONL_DIR if yours differs
# -------------------------------------------------------
WSI_ROOTS = [
    "/home/sorkwos/links/scratch/TCGA-BRCA-new",
]
JSONL_DIR = "/home/sorkwos/links/scratch/brca_jsons"   # ← UPDATE if different
SCRIPT    = "/home/sorkwos/links/scratch/multimodality/graph_subset_generation/extract_radiomics_BRCA.py"

# -------------------------------------------------------
# Find all DX1 SVS files
# -------------------------------------------------------
print(">>> Scanning for DX1 SVS files...")
svs_files = {}
for root_dir in WSI_ROOTS:
    if not os.path.isdir(root_dir):
        print(f"  [warning] WSI root not found, skipping: {root_dir}")
        continue
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".svs") and "DX1" in fname:
                stem = os.path.splitext(fname)[0]
                svs_files[stem] = os.path.join(dirpath, fname)

print(f"  Found {len(svs_files)} DX1 SVS files.")

# -------------------------------------------------------
# Find all JSONL files
# -------------------------------------------------------
print(">>> Scanning for JSONL files...")
jsonl_files = {}
if not os.path.isdir(JSONL_DIR):
    print(f"  [error] JSONL_DIR not found: {JSONL_DIR}")
    print("  Update JSONL_DIR at the top of this script.")
    sys.exit(1)

for fname in os.listdir(JSONL_DIR):
    if fname.endswith("_tiles.jsonl"):
        stem = fname.replace("_tiles.jsonl", "")
        jsonl_files[stem] = os.path.join(JSONL_DIR, fname)

print(f"  Found {len(jsonl_files)} JSONL files.")

# -------------------------------------------------------
# Match pairs
# -------------------------------------------------------
pairs = []
for stem, svs_path in svs_files.items():
    if stem in jsonl_files:
        pairs.append((svs_path, jsonl_files[stem]))

pairs = sorted(pairs)  # deterministic order
print(f"\n  Matched pairs: {len(pairs)}")

# -------------------------------------------------------
# Parse batch args
# -------------------------------------------------------
batch_id      = None
total_batches = None

if "--batch" in sys.argv and "--total" in sys.argv:
    batch_id      = int(sys.argv[sys.argv.index("--batch") + 1])
    total_batches = int(sys.argv[sys.argv.index("--total") + 1])

    batch_size = len(pairs) // total_batches
    start = (batch_id - 1) * batch_size
    end   = start + batch_size if batch_id < total_batches else len(pairs)
    pairs = pairs[start:end]

    print(f"  Batch {batch_id}/{total_batches}: slides {start+1} to {end} ({len(pairs)} slides)")

# -------------------------------------------------------
# Dry run — just print pairs
# -------------------------------------------------------
if "--dry-run" in sys.argv:
    print("\n>>> Matched pairs:")
    for svs, jsonl in pairs:
        print(f"  SVS:   {svs}")
        print(f"  JSONL: {jsonl}")
        print()
    print(">>> Dry run only. Use --run to execute extraction.")
    sys.exit(0)

# -------------------------------------------------------
# Run extraction
# -------------------------------------------------------
if "--run" in sys.argv:
    print(f"\n>>> Running extract_radiomics_BRCA.py on {len(pairs)} pairs...\n")
    for i, (svs, jsonl) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {os.path.basename(svs)}")
        subprocess.run(["python", SCRIPT, svs, jsonl], check=False)
else:
    print(">>> Use --run to execute or --dry-run to preview.")