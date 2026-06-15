#!/usr/bin/env python3
"""
Compute the theoretical GEMM (trailing rank-nc update) FLOPS fraction of the multifrontal
factorize, per panel category and aggregate. Goal: argue how much FP32 factor time *could*
be replaced by TC if WMMA accelerated the trailing GEMM.

Per panel of size fsz x fsz with nc pivot cols and uc CB rows (fsz = nc + uc):
  - LU panel factor       : nc^2 * fsz / 2     (col elimination + intra-panel rank-1)
  - U panel solve         : nc^2 * uc / 2      (back-substitution on U panel)
  - Trailing rank-nc GEMM : 2 * uc^2 * nc      (C -= L * U, multiply + subtract)
Total per panel: LU + Usolve + Trailing.

Categorize panels by dispatch path (matches src/batched/multifrontal_batched.cu):
  - small_warp : max_fsz_in_level <= 32       (warp-per-front kernel)
  - mid        : 32 < max_fsz_in_level <= 159 (shared-resident block kernel, MID_THRESH=159 for FP32)
  - big        : max_fsz_in_level > 159       (global-memory block kernel)

We don't have level info in the CSV, so approximate by per-panel fsz.

Usage: python gemm_fraction.py case8387_panels.csv
"""
import sys, csv

def analyze(path, name):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['): continue
            if line.startswith('panel'): continue
            parts = line.split(',')
            if len(parts) < 4: continue
            try:
                p, fsz, nc, uc = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                rows.append((p, fsz, nc, uc))
            except ValueError:
                continue
    n_panels = len(rows)

    # Per-panel FLOPs
    cat_stats = {'small': [0,0,0,0], 'mid': [0,0,0,0], 'big': [0,0,0,0]}
    # entries: [LU_flops, Usolve_flops, Trailing_flops, count]
    SMALL_THRESH = 32
    MID_THRESH = 159

    for p, fsz, nc, uc in rows:
        lu = nc * nc * fsz / 2.0
        us = nc * nc * uc / 2.0
        tr = 2.0 * uc * uc * nc
        if fsz <= SMALL_THRESH: cat = 'small'
        elif fsz <= MID_THRESH: cat = 'mid'
        else: cat = 'big'
        s = cat_stats[cat]
        s[0] += lu; s[1] += us; s[2] += tr; s[3] += 1

    print(f"\n=== {name}  P={n_panels} ===")
    total_lu = sum(s[0] for s in cat_stats.values())
    total_us = sum(s[1] for s in cat_stats.values())
    total_tr = sum(s[2] for s in cat_stats.values())
    grand = total_lu + total_us + total_tr

    print(f"{'Category':<10} {'#panels':>8} {'LU%':>8} {'U-solve%':>10} {'Trailing%':>11} {'cat%':>8}")
    print('-' * 65)
    for cat in ['small', 'mid', 'big']:
        s = cat_stats[cat]
        catt = s[0] + s[1] + s[2]
        if catt == 0:
            print(f"{cat:<10} {s[3]:>8} {'—':>8} {'—':>10} {'—':>11} {'0.0%':>8}")
            continue
        lu_pct = 100 * s[0] / catt
        us_pct = 100 * s[1] / catt
        tr_pct = 100 * s[2] / catt
        cat_pct = 100 * catt / grand
        print(f"{cat:<10} {s[3]:>8} {lu_pct:>7.1f}% {us_pct:>9.1f}% {tr_pct:>10.1f}% {cat_pct:>7.1f}%")
    print('-' * 65)
    glu = 100 * total_lu / grand
    gus = 100 * total_us / grand
    gtr = 100 * total_tr / grand
    print(f"{'TOTAL':<10} {n_panels:>8} {glu:>7.1f}% {gus:>9.1f}% {gtr:>10.1f}% {'100%':>8}")
    print()
    print(f"  GRAND TOTAL FLOPs = {grand:.3e}")
    print(f"  LU panel factor   : {total_lu:.3e}  ({glu:.1f}%)")
    print(f"  U panel solve     : {total_us:.3e}  ({gus:.1f}%)")
    print(f"  >>> TRAILING GEMM : {total_tr:.3e}  ({gtr:.1f}%)  <<<")
    print()
    print(f"  If TC accelerates ONLY the trailing GEMM by factor X:")
    print(f"    factor speedup upper bound = total / (LU + Usolve + Trailing/X)")
    for x in (1.5, 2, 3, 5, 10):
        new = total_lu + total_us + total_tr / x
        sp = grand / new
        pct = 100 * (1 - 1/sp)
        print(f"      X={x:>4}: speedup = {sp:.2f}x  (-{pct:.1f}% factor)")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            name = p.split('/')[-1].replace('_panels.csv', '').replace('.csv', '')
            analyze(p, name)
    else:
        analyze('/tmp/case8387_panels.csv', 'case8387')
        analyze('/tmp/usa_panels.csv', 'USA')
