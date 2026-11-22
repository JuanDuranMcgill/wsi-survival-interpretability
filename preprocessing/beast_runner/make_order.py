#!/usr/bin/env python3
"""Processing order for the carcinoma cohort: in_cohort rows of the manifest, stratified by cancer type.
Each type is shuffled (fixed seed) and every slide gets the key (rank + u) / n_type, so every prefix of the
order holds all types in proportion to their size. Writes manifest_order.csv next to the manifest."""
import csv, random, collections, sys
M = "/data/processed/juan/pancan/manifest_patients.csv"; OUT = "/data/processed/juan/pancan/manifest_order.csv"; SEED = 0
rows = [r for r in csv.DictReader(open(M)) if r["in_cohort"] == "True"]
by = collections.defaultdict(list)
for r in rows: by[r["type"]].append(r)
rng = random.Random(SEED); keyed = []
for t in sorted(by):
    L = sorted(by[t], key=lambda r: r["file_id"]); rng.shuffle(L)
    keyed += [((i + rng.random()) / len(L), r) for i, r in enumerate(L)]
keyed.sort(key=lambda kr: kr[0])
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["order", "type", "patient", "file_id", "file_name", "file_size", "md5sum"])
    for i, (_, r) in enumerate(keyed):
        w.writerow([i, r["type"], r["patient"], r["file_id"], r["file_name"], int(float(r["file_size"])), r["md5sum"]])
print(len(keyed), "slides,", len(by), "types, seed", SEED, "->", OUT)
