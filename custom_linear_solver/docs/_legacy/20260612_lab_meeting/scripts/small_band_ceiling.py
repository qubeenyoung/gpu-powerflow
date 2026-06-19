#!/usr/bin/env python3
# Goal 1 (260612_goal.md §1.3 step 1+2): ceiling-first accounting for the small-tier
# upper sub-band fsz in [17,32]. Uses the existing per-front dumps (q,p,fsz,nc,uc,level,...).
#
# For each case we report, over ALL fronts and over the [17,32] band:
#   - front count and share of fronts
#   - trailing-GEMM FLOP = 2*uc^2*nc and the band's share of total trailing FLOP
#   - tier split (small<=32 / mid 33..159 / big>=160) for context
#   - TC tile fill: GEMM is M=N=uc, K=nc on Ampere TF32 mma tile 16x8x8.
#       tiles    = ceil(uc/16)*ceil(uc/8)*ceil(nc/8)
#       fill     = (uc*uc*nc) / (tiles*16*8*8)
#     plus the fraction of band fronts that fill >=1 full tile (uc>=16 and nc>=8).
#
# This decides whether goal 1 clears the bar: if the [17,32] band carries a tiny FLOP
# share, no TC path on it can move factorize meaningfully (ceiling-first => likely reject).

import csv, glob, math, os, statistics

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Canonical case set (dedup the *_case_* aliases that mirror the short names).
CASES = [
    ("case3012wp",   "fronts_case3012wp.csv"),
    ("case6468rte",  "fronts_case6468rte.csv"),
    ("case8387",     "fronts_case8387pegase.csv"),
    ("case13659",    "fronts_case13659pegase.csv"),
    ("ACTIVSg25k",   "fronts_case_ACTIVSg25k.csv"),
    ("ACTIVSg70k",   "fronts_case_ACTIVSg70k.csv"),
    ("SyntheticUSA", "fronts_case_SyntheticUSA.csv"),
]

SMALL_MAX = 32          # kSmallFrontMax
MID_MAX   = 159         # kFloatSharedFrontMax
BAND_LO, BAND_HI = 17, 32

def tile_fill(uc, nc):
    if uc <= 0 or nc <= 0:
        return 0.0
    tiles = math.ceil(uc/16) * math.ceil(uc/8) * math.ceil(nc/8)
    return (uc*uc*nc) / (tiles*16*8*8)

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["fsz"]), int(r["nc"]), int(r["uc"])))
    return rows

def flop(nc, uc):
    return 2 * uc * uc * nc

print(f"{'case':<14} {'fronts':>8} {'band#':>7} {'band%':>6} "
      f"{'totFLOP':>12} {'bandFLOP':>11} {'band%FL':>7} "
      f"{'sm%FL':>6} {'mid%FL':>6} {'big%FL':>6} "
      f"{'band uc med/p90':>15} {'nc med/max':>11} {'fill≥1tile%':>11} {'medfill%':>8}")

agg = {}
for name, fname in CASES:
    path = os.path.normpath(os.path.join(DATA, fname))
    rows = load(path)
    n = len(rows)
    tot_fl = sum(flop(nc, uc) for _, nc, uc in rows)
    sm_fl  = sum(flop(nc, uc) for fsz, nc, uc in rows if fsz <= SMALL_MAX)
    mid_fl = sum(flop(nc, uc) for fsz, nc, uc in rows if SMALL_MAX < fsz <= MID_MAX)
    big_fl = sum(flop(nc, uc) for fsz, nc, uc in rows if fsz > MID_MAX)

    band = [(fsz, nc, uc) for fsz, nc, uc in rows if BAND_LO <= fsz <= BAND_HI]
    bn = len(band)
    band_fl = sum(flop(nc, uc) for _, nc, uc in band)
    ucs = [uc for _, _, uc in band]
    ncs = [nc for _, nc, _ in band]
    fills = [tile_fill(uc, nc) for _, nc, uc in band]
    full_tile = sum(1 for _, nc, uc in band if uc >= 16 and nc >= 8)

    def med(x): return statistics.median(x) if x else 0
    def p90(x): return sorted(x)[int(0.9*len(x))-1] if x else 0

    print(f"{name:<14} {n:>8} {bn:>7} {100*bn/n:>5.1f}% "
          f"{tot_fl:>12} {band_fl:>11} {100*band_fl/tot_fl:>6.2f}% "
          f"{100*sm_fl/tot_fl:>5.1f}% {100*mid_fl/tot_fl:>5.1f}% {100*big_fl/tot_fl:>5.1f}% "
          f"{med(ucs):>6.0f}/{p90(ucs):<8.0f} {med(ncs):>4.0f}/{max(ncs) if ncs else 0:<6} "
          f"{100*full_tile/bn if bn else 0:>10.1f}% {100*med(fills):>7.1f}%")
    agg[name] = dict(n=n, bn=bn, tot_fl=tot_fl, band_fl=band_fl, sm_fl=sm_fl,
                     full_tile=full_tile, med_fill=med(fills))

print()
print("=== 종합 (전 케이스 합산) ===")
N  = sum(a["n"] for a in agg.values())
BN = sum(a["bn"] for a in agg.values())
TF = sum(a["tot_fl"] for a in agg.values())
BF = sum(a["band_fl"] for a in agg.values())
SF = sum(a["sm_fl"] for a in agg.values())
print(f"전체 front {N:,} 중 [17,32] band = {BN:,} ({100*BN/N:.1f}%)")
print(f"trailing FLOP 중 band 비중 = {100*BF/TF:.2f}%  (small≤32 전체 = {100*SF/TF:.1f}%)")
print(f"band 중 TC 타일 ≥1개 채우는 front(uc≥16∧nc≥8) 비중 범위 = "
      f"{min(100*a['full_tile']/a['bn'] for a in agg.values() if a['bn']):.1f}% .. "
      f"{max(100*a['full_tile']/a['bn'] for a in agg.values() if a['bn']):.1f}%")
