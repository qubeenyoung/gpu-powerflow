#!/usr/bin/env python3
"""Analyze the elimination tree shape from --analyze-info output.

Identifies:
  - spine: longest consecutive cnt=1 chain from the top (highest level)
  - K: count of panels at the level just below the spine (= number of independent subtrees)
  - per-subtree size (estimated from panel counts)
  - Phase 4 (spine fusion) potential dispatch savings
  - Phase 3 (multi-stream) K parameter

Usage:
    ./custom_linear_solver_run <case> --repeat 1 --single-precision fp64 --analyze-info 2>&1 | \
        python3 analyze_tree.py
"""

import re
import sys
from collections import namedtuple

LevelInfo = namedtuple('LevelInfo', ['L', 'cnt', 'maxfsz', 'f2'])


def parse_dump(text):
    """Parse --analyze-info output. Returns (n, P, num_levels, cap, front_total_MB, levels[]).
    Each level: L, cnt, maxfsz, f2.
    """
    head_re = re.compile(r'\[analyze\]\s+n=(\d+)\s+P=(\d+)\s+levels=(\d+)\s+cap=(\d+)\s+front_total\(MB f32\)=([\d.]+)')
    level_re = re.compile(r'^\s+L(\d+)\s+cnt=(\d+)\s+maxfsz=(\d+)\s+f2=(\d+)')

    n = P = num_levels = cap = 0
    front_total_mb = 0.0
    levels = []
    for line in text.splitlines():
        m = head_re.match(line)
        if m:
            n, P, num_levels, cap = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            front_total_mb = float(m[5])
            continue
        m = level_re.match(line)
        if m:
            L, cnt, maxfsz, f2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            levels.append(LevelInfo(L, cnt, maxfsz, f2))
    return n, P, num_levels, cap, front_total_mb, levels


def find_spine(levels):
    """Spine = longest consecutive cnt=1 chain starting from the top (highest L).
    Returns (spine_start_level, spine_end_level, spine_levels[]) where spine_start_level
    is the lowest L in the spine and spine_end_level is the highest (= root area).
    """
    if not levels:
        return None, None, []
    levels_sorted = sorted(levels, key=lambda x: x.L, reverse=True)  # highest L first
    spine = []
    for lv in levels_sorted:
        if lv.cnt == 1:
            spine.append(lv)
        else:
            break
    if not spine:
        return None, None, []
    spine.sort(key=lambda x: x.L)  # ascending by L for output
    return spine[0].L, spine[-1].L, spine


def find_subtree_roots(levels, spine_start_level):
    """K subtree roots = panels at the level just below spine_start_level."""
    if spine_start_level is None:
        return None
    candidates = [lv for lv in levels if lv.L == spine_start_level - 1]
    if not candidates:
        return None
    return candidates[0]


def analyze(text):
    n, P, num_levels, cap, front_mb, levels = parse_dump(text)
    if not levels:
        print("No [analyze] lines found in input — make sure --analyze-info was passed.")
        return

    print(f'=== Tree analysis ===')
    print(f'  n             = {n}')
    print(f'  P             = {P} panels')
    print(f'  num_levels    = {num_levels}')
    print(f'  cap           = {cap}')
    print(f'  front_total   = {front_mb:.1f} MB (FP32)')

    # Front size distribution summary
    total_f2 = sum(lv.f2 for lv in levels)
    print(f'\n  level distribution (top to bottom):')
    levels_top_down = sorted(levels, key=lambda x: x.L, reverse=True)
    for lv in levels_top_down:
        pct = 100.0 * lv.f2 / max(total_f2, 1)
        print(f'    L{lv.L:<3d}  cnt={lv.cnt:<5d}  maxfsz={lv.maxfsz:<4d}  f²={lv.f2:<9d}  ({pct:.1f}%)')

    # --- Spine identification ---
    s_lo, s_hi, spine = find_spine(levels)
    if spine:
        spine_f2 = sum(lv.f2 for lv in spine)
        spine_pct = 100.0 * spine_f2 / max(total_f2, 1)
        print(f'\n=== Spine (Phase 4 target) ===')
        print(f'  spine span    = L{s_lo} .. L{s_hi}  ({len(spine)} levels)')
        print(f'  spine f² 합   = {spine_f2}  ({spine_pct:.1f}% of total)')
        print(f'  spine fsz     = min {min(lv.maxfsz for lv in spine)}, max {max(lv.maxfsz for lv in spine)}')
        # Phase 4 ROI estimate
        # current: spine = 13 launches × ~5μs latency
        # after fusion: 1 launch
        savings_us = (len(spine) - 1) * 5  # crude
        print(f'  Phase 4 잠재 절감 (단일 배치 기준): ~{savings_us} μs (dispatch latency × {len(spine)-1} fused launches)')
    else:
        print('\n  No spine (cnt=1 chain at top) found.')

    # --- K subtree roots ---
    print(f'\n=== K subtree roots (Phase 3 target) ===')
    if s_lo is None:
        root_level = max(lv.L for lv in levels)
        K_level = [lv for lv in levels if lv.L == root_level][0]
        print(f'  no spine — root level L{root_level} has cnt={K_level.cnt}')
        K = K_level.cnt
    else:
        below = find_subtree_roots(levels, s_lo)
        if below:
            print(f'  K subtree roots at L{below.L}: cnt={below.cnt}')
            print(f'    these {below.cnt} panels each root an independent subtree')
            K = below.cnt
        else:
            print(f'  spine reaches all the way to bottom — no separate subtrees')
            K = 0
    print(f'  K (number of subtrees) = {K}')

    # --- Per-subtree size estimate (crude: assume balanced) ---
    if K > 0 and s_lo is not None:
        below_spine_panels = sum(lv.cnt for lv in levels if lv.L < s_lo)
        below_spine_f2 = sum(lv.f2 for lv in levels if lv.L < s_lo)
        print(f'\n=== Subtree size (below-spine total) ===')
        print(f'  panels        = {below_spine_panels} (of {P} total)')
        print(f'  f² 비중       = {100.0 * below_spine_f2 / total_f2:.1f}%')
        print(f'  per-subtree   = ~{below_spine_panels // K} panels (assume balanced)')
        print(f'  Phase 3 잠재 ideal speedup = {K}× on subtree section')
        print(f'  Phase 3 잠재 wall 절감 = ~{100 * (1 - 1.0/K) * (below_spine_f2 / total_f2):.0f}%')

    # --- Sibling amalgamation candidates (Phase 2) ---
    print(f'\n=== Sibling amalgamation candidates (Phase 2 target) ===')
    print(f'  narrow-mid levels (cnt in [2, 16]):')
    cand_f2 = 0
    cand_panel = 0
    for lv in sorted(levels, key=lambda x: x.L, reverse=True):
        if 2 <= lv.cnt <= 16:
            cand_f2 += lv.f2
            cand_panel += lv.cnt
            print(f'    L{lv.L}: cnt={lv.cnt} maxfsz={lv.maxfsz} f²={lv.f2}  -> 묶으면 ~ {lv.cnt * lv.maxfsz} col 의 dense block')
    cand_pct = 100.0 * cand_f2 / max(total_f2, 1)
    print(f'  total candidate f² = {cand_pct:.1f}% of total')
    print(f'  Phase 2 잠재 dispatch 절감: {cand_panel} fronts -> ~{sum(1 for lv in levels if 2 <= lv.cnt <= 16)} fused fronts')


if __name__ == '__main__':
    text = sys.stdin.read()
    analyze(text)
