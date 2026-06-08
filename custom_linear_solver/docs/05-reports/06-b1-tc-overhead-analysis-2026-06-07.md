# B=1 TC Overhead Geometry Analysis

**작성일**: 2026-06-07
**목적**: B=1에서 custom `fp16`/`tf32` factorize가 `fp32`보다 느린 구간을 front geometry와 TC 부가 비용으로 정량화한다. 이 문서는 TC 모드의 `fp32 fallback` 정책을 제안하지 않는다.

## Inputs

- Timing: `05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv`
- Front CSVs: `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/fronts/*.csv`
- Available front dumps: 11/11 cases
- Missing front dumps: none

Reproduction used two collection roots: the existing `/datasets/power_system/nr_linear_systems` cases, plus regenerated NR dumps for the four MATPOWER-only cases.

```bash
# 1) Collect already-present NR cases.
python3 custom_linear_solver/tools/analyze_b1_tc_overhead.py \
  --collect \
  --runner custom_linear_solver/build-bench/custom_linear_solver_run \
  --case-root /datasets/power_system/nr_linear_systems \
  --front-dir custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/fronts \
  --sweep custom_linear_solver/docs/05-reports/05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv \
  --out-dir custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07

# 2) Regenerate and collect the four MATPOWER-only cases.
python3 -m python.prepare.convert_m_to_mat \
  --input-root /datasets/power_system/matpower \
  --output-root /tmp/cls_missing_mat \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k

PYTHONPATH=/workspace/sparse_direct_solver/prepare_datasets/python \
python3 /workspace/sparse_direct_solver/prepare_datasets/python/prepare_nr_linear_system.py \
  --mat-root /tmp/cls_missing_mat \
  --output-root /tmp/cls_missing_nr \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k \
  --dump-iteration 2

python3 custom_linear_solver/tools/analyze_b1_tc_overhead.py \
  --collect \
  --runner custom_linear_solver/build-bench/custom_linear_solver_run \
  --case-root /tmp/cls_missing_nr \
  --front-dir custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/fronts \
  --sweep custom_linear_solver/docs/05-reports/05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv \
  --out-dir custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07
```


## B=1 Factor Timing

`>1` means TC mode is slower than custom/fp32.

| case | n | fp32_factor_ms | fp16_factor_ms | tf32_factor_ms | fp16_over_fp32 | tf32_over_fp32 |
| --- | --- | --- | --- | --- | --- | --- |
| case1197 | 2392 | 0.069279 | 0.069931 | 0.069539 | 1.01 | 1.00 |
| case_ACTIVSg2000 | 3607 | 0.274221 | 0.354832 | 0.348961 | 1.29 | 1.27 |
| case3012wp | 5725 | 0.233355 | 0.299899 | 0.299678 | 1.29 | 1.28 |
| case6468rte | 12643 | 0.273600 | 0.450811 | 0.427126 | 1.65 | 1.56 |
| case8387pegase | 14908 | 0.352687 | 0.444068 | 0.418029 | 1.26 | 1.19 |
| case9241pegase | 17036 | 0.384797 | 0.479815 | 0.431234 | 1.25 | 1.12 |
| case_ACTIVSg10k | 18544 | 0.370160 | 0.459427 | 0.430623 | 1.24 | 1.16 |
| case13659pegase | 23225 | 0.396369 | 0.490023 | 0.481769 | 1.24 | 1.22 |
| case_ACTIVSg25k | 47246 | 0.813036 | 0.816202 | 0.808227 | 1.00 | 0.99 |
| case_ACTIVSg70k | 134104 | 2.795770 | 2.072840 | 2.161060 | 0.74 | 0.77 |
| case_SyntheticUSA | 156255 | 2.621740 | 2.165690 | 2.489820 | 0.83 | 0.95 |

## Actual Front Tier Summary

`phase3_fronts_fsz_gt_48` is important because `factorize_front()` bypasses the trailing functor for `fsz <= 48`; those fronts do not execute WMMA/PTX trailing even if their level dispatched to a TC kernel.

| case | fronts | levels | small_count | mid_count | big_count | small_flop_pct | mid_flop_pct | big_flop_pct | phase3_fronts_fsz_gt_48 | mid_fsz_33_48_no_trailing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case1197 | 1166 | 11 | 1166 | 0 | 0 | 100.00 | 0.00 | 0.00 | 0 | 0 |
| case_ACTIVSg2000 | 1767 | 24 | 1701 | 66 | 0 | 19.58 | 80.42 | 0.00 | 29 | 37 |
| case3012wp | 2773 | 21 | 2759 | 14 | 0 | 68.80 | 31.20 | 0.00 | 0 | 14 |
| case6468rte | 5895 | 24 | 5847 | 48 | 0 | 50.85 | 49.15 | 0.00 | 9 | 39 |
| case8387pegase | 7408 | 28 | 7353 | 55 | 0 | 48.29 | 51.71 | 0.00 | 19 | 36 |
| case9241pegase | 8094 | 24 | 8018 | 76 | 0 | 43.05 | 56.95 | 0.00 | 19 | 57 |
| case_ACTIVSg10k | 9079 | 25 | 8947 | 132 | 0 | 27.16 | 72.84 | 0.00 | 40 | 92 |
| case13659pegase | 12392 | 25 | 12277 | 115 | 0 | 35.02 | 64.98 | 0.00 | 34 | 81 |
| case_ACTIVSg25k | 22717 | 36 | 22344 | 371 | 2 | 13.57 | 83.46 | 2.97 | 200 | 173 |
| case_ACTIVSg70k | 63546 | 42 | 62736 | 748 | 62 | 7.37 | 47.16 | 45.47 | 468 | 342 |
| case_SyntheticUSA | 74129 | 41 | 73171 | 901 | 57 | 8.83 | 52.69 | 38.48 | 542 | 416 |

## Mid Front Size

| case | mid_count | fsz_p50/p90/max | nc_p50/p90/max | uc_p50/p90/max |
| --- | --- | --- | --- | --- |
| case1197 | 0 | 0.0/0.0/0 | 0.0/0.0/0 | 0.0/0.0/0 |
| case_ACTIVSg2000 | 66 | 47.5/67.0/83 | 8.0/8.0/8 | 40.0/59.0/75 |
| case3012wp | 14 | 35.0/41.1/43 | 8.0/8.0/8 | 28.0/33.1/35 |
| case6468rte | 48 | 41.0/51.6/59 | 8.0/8.0/8 | 34.0/43.6/51 |
| case8387pegase | 55 | 41.0/66.0/74 | 8.0/8.0/8 | 35.0/58.6/66 |
| case9241pegase | 76 | 40.5/55.0/73 | 12.0/12.0/12 | 31.0/46.5/61 |
| case_ACTIVSg10k | 132 | 43.5/58.8/79 | 12.0/12.0/12 | 33.0/49.0/67 |
| case13659pegase | 115 | 40.0/61.2/84 | 12.0/12.0/12 | 32.0/49.2/72 |
| case_ACTIVSg25k | 371 | 50.0/87.0/125 | 12.0/12.0/12 | 40.0/75.0/113 |
| case_ACTIVSg70k | 748 | 51.0/90.3/128 | 14.0/20.0/20 | 38.0/71.0/113 |
| case_SyntheticUSA | 901 | 51.0/92.0/128 | 14.0/20.0/20 | 38.0/75.0/122 |

## TC Overhead Estimate

FP16 WMMA includes explicit half staging plus `Csc` fragment store/reload. TF32 PTX has no `Csc` and no explicit `__float_to_tf32`; it still pays padded staging. `padded_over_real` is computed only for fronts where Phase 3 is invoked.

| case | model | phase3_fronts | tc_scalar_gate_fronts | real_mflop | padded_mflop | padded_over_real | output_tiles | mma_count | stage_elems_m | explicit_convert_elems_m | csc_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case1197 | fp16_wmma | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case1197 | tf32_wmma | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case1197 | tf32_ptx | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case_ACTIVSg2000 | fp16_wmma | 29 | 0 | 1.259 | 3.260 | 2.59 | 398 | 398 | 0.054 | 0.054 | 0.815 |
| case_ACTIVSg2000 | tf32_wmma | 29 | 0 | 1.259 | 1.630 | 1.29 | 398 | 398 | 0.027 | 0.027 | 0.815 |
| case_ACTIVSg2000 | tf32_ptx | 29 | 0 | 1.259 | 1.630 | 1.29 | 398 | 398 | 0.027 | 0.000 | 0.000 |
| case3012wp | fp16_wmma | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case3012wp | tf32_wmma | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case3012wp | tf32_ptx | 0 | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| case6468rte | fp16_wmma | 9 | 0 | 0.301 | 0.836 | 2.77 | 102 | 102 | 0.015 | 0.015 | 0.209 |
| case6468rte | tf32_wmma | 9 | 0 | 0.301 | 0.418 | 1.39 | 102 | 102 | 0.008 | 0.008 | 0.209 |
| case6468rte | tf32_ptx | 9 | 0 | 0.301 | 0.418 | 1.39 | 102 | 102 | 0.008 | 0.000 | 0.000 |
| case8387pegase | fp16_wmma | 19 | 0 | 0.852 | 2.441 | 2.87 | 298 | 298 | 0.038 | 0.038 | 0.610 |
| case8387pegase | tf32_wmma | 19 | 0 | 0.852 | 1.221 | 1.43 | 298 | 298 | 0.019 | 0.019 | 0.610 |
| case8387pegase | tf32_ptx | 19 | 0 | 0.852 | 1.221 | 1.43 | 298 | 298 | 0.019 | 0.000 | 0.000 |
| case9241pegase | fp16_wmma | 19 | 0 | 0.849 | 1.802 | 2.12 | 220 | 220 | 0.033 | 0.033 | 0.451 |
| case9241pegase | tf32_wmma | 19 | 0 | 0.849 | 1.561 | 1.84 | 220 | 381 | 0.028 | 0.028 | 0.451 |
| case9241pegase | tf32_ptx | 19 | 0 | 0.849 | 1.176 | 1.38 | 220 | 558 | 0.021 | 0.000 | 0.000 |
| case_ACTIVSg10k | fp16_wmma | 40 | 0 | 2.020 | 3.957 | 1.96 | 483 | 483 | 0.070 | 0.070 | 0.989 |
| case_ACTIVSg10k | tf32_wmma | 40 | 0 | 2.020 | 3.789 | 1.88 | 483 | 925 | 0.067 | 0.067 | 0.989 |
| case_ACTIVSg10k | tf32_ptx | 40 | 0 | 2.020 | 2.818 | 1.39 | 483 | 1376 | 0.050 | 0.000 | 0.000 |
| case13659pegase | fp16_wmma | 34 | 0 | 1.824 | 3.416 | 1.87 | 417 | 417 | 0.060 | 0.060 | 0.854 |
| case13659pegase | tf32_wmma | 34 | 0 | 1.824 | 3.240 | 1.78 | 417 | 791 | 0.057 | 0.057 | 0.854 |
| case13659pegase | tf32_ptx | 34 | 0 | 1.824 | 2.423 | 1.33 | 417 | 1183 | 0.043 | 0.000 | 0.000 |
| case_ACTIVSg25k | fp16_wmma | 200 | 0 | 17.340 | 31.662 | 1.83 | 3865 | 3865 | 0.434 | 0.434 | 7.916 |
| case_ACTIVSg25k | tf32_wmma | 200 | 0 | 17.340 | 29.303 | 1.69 | 3865 | 7154 | 0.400 | 0.400 | 7.916 |
| case_ACTIVSg25k | tf32_ptx | 200 | 0 | 17.340 | 23.024 | 1.33 | 3865 | 10321 | 0.313 | 0.000 | 0.000 |
| case_ACTIVSg70k | fp16_wmma | 468 | 1 | 102.450 | 192.017 | 1.87 | 12937 | 23112 | 1.963 | 1.963 | 26.495 |
| case_ACTIVSg70k | tf32_wmma | 468 | 1 | 102.450 | 145.339 | 1.42 | 12937 | 34828 | 1.487 | 1.487 | 26.495 |
| case_ACTIVSg70k | tf32_ptx | 468 | 1 | 102.450 | 136.620 | 1.33 | 12937 | 43587 | 1.355 | 0.000 | 0.000 |
| case_SyntheticUSA | fp16_wmma | 542 | 0 | 102.321 | 197.263 | 1.93 | 13732 | 24080 | 2.163 | 2.163 | 28.123 |
| case_SyntheticUSA | tf32_wmma | 542 | 0 | 102.321 | 148.148 | 1.45 | 13732 | 36169 | 1.643 | 1.643 | 28.123 |
| case_SyntheticUSA | tf32_ptx | 542 | 0 | 102.321 | 139.526 | 1.36 | 13732 | 45282 | 1.509 | 0.000 | 0.000 |

## Dispatch Tier Mismatch

This table counts fronts whose actual `fsz` tier differs from the per-level dispatch tier. Mixed levels explain why small fronts can ride through mid/big kernels without implying a precision fallback policy.

| case | precision | actual_to_dispatch | fronts | front_pct |
| --- | --- | --- | --- | --- |
| case_ACTIVSg2000 | fp32 | small->mid | 105 | 5.94 |
| case_ACTIVSg2000 | fp16 | small->mid | 105 | 5.94 |
| case_ACTIVSg2000 | tf32 | small->mid | 105 | 5.94 |
| case_ACTIVSg2000 | tf32_wmma | small->mid | 105 | 5.94 |
| case3012wp | fp32 | small->mid | 65 | 2.34 |
| case3012wp | fp16 | small->mid | 65 | 2.34 |
| case3012wp | tf32 | small->mid | 65 | 2.34 |
| case3012wp | tf32_wmma | small->mid | 65 | 2.34 |
| case6468rte | fp32 | small->mid | 304 | 5.16 |
| case6468rte | fp16 | small->mid | 304 | 5.16 |
| case6468rte | tf32 | small->mid | 304 | 5.16 |
| case6468rte | tf32_wmma | small->mid | 304 | 5.16 |
| case8387pegase | fp32 | small->mid | 1752 | 23.65 |
| case8387pegase | fp16 | small->mid | 1752 | 23.65 |
| case8387pegase | tf32 | small->mid | 1752 | 23.65 |
| case8387pegase | tf32_wmma | small->mid | 1752 | 23.65 |
| case9241pegase | fp32 | small->mid | 1883 | 23.26 |
| case9241pegase | fp16 | small->mid | 1883 | 23.26 |
| case9241pegase | tf32 | small->mid | 1883 | 23.26 |
| case9241pegase | tf32_wmma | small->mid | 1883 | 23.26 |
| case_ACTIVSg10k | fp32 | small->mid | 285 | 3.14 |
| case_ACTIVSg10k | fp16 | small->mid | 285 | 3.14 |
| case_ACTIVSg10k | tf32 | small->mid | 285 | 3.14 |
| case_ACTIVSg10k | tf32_wmma | small->mid | 285 | 3.14 |
| case13659pegase | fp32 | small->mid | 1889 | 15.24 |
| case13659pegase | fp16 | small->mid | 1889 | 15.24 |
| case13659pegase | tf32 | small->mid | 1889 | 15.24 |
| case13659pegase | tf32_wmma | small->mid | 1889 | 15.24 |
| case_ACTIVSg25k | fp32 | mid->big | 24 | 0.11 |
| case_ACTIVSg25k | fp32 | small->big | 2 | 0.01 |
| case_ACTIVSg25k | fp32 | small->mid | 2494 | 10.98 |
| case_ACTIVSg25k | fp16 | mid->big | 24 | 0.11 |
| case_ACTIVSg25k | fp16 | small->big | 2 | 0.01 |
| case_ACTIVSg25k | fp16 | small->mid | 2494 | 10.98 |
| case_ACTIVSg25k | tf32 | mid->big | 24 | 0.11 |
| case_ACTIVSg25k | tf32 | small->big | 2 | 0.01 |
| case_ACTIVSg25k | tf32 | small->mid | 2494 | 10.98 |
| case_ACTIVSg25k | tf32_wmma | mid->big | 24 | 0.11 |
| case_ACTIVSg25k | tf32_wmma | small->big | 2 | 0.01 |
| case_ACTIVSg25k | tf32_wmma | small->mid | 2494 | 10.98 |
| case_ACTIVSg70k | fp32 | mid->big | 76 | 0.12 |
| case_ACTIVSg70k | fp32 | small->mid | 3878 | 6.10 |
| case_ACTIVSg70k | fp16 | mid->big | 76 | 0.12 |
| case_ACTIVSg70k | fp16 | small->mid | 3878 | 6.10 |
| case_ACTIVSg70k | tf32 | mid->big | 76 | 0.12 |
| case_ACTIVSg70k | tf32 | small->mid | 3878 | 6.10 |
| case_ACTIVSg70k | tf32_wmma | mid->big | 76 | 0.12 |
| case_ACTIVSg70k | tf32_wmma | small->mid | 3878 | 6.10 |
| case_SyntheticUSA | fp32 | mid->big | 148 | 0.20 |
| case_SyntheticUSA | fp32 | small->big | 8 | 0.01 |
| case_SyntheticUSA | fp32 | small->mid | 4644 | 6.26 |
| case_SyntheticUSA | fp16 | mid->big | 148 | 0.20 |
| case_SyntheticUSA | fp16 | small->big | 8 | 0.01 |
| case_SyntheticUSA | fp16 | small->mid | 4644 | 6.26 |
| case_SyntheticUSA | tf32 | mid->big | 148 | 0.20 |
| case_SyntheticUSA | tf32 | small->big | 8 | 0.01 |
| case_SyntheticUSA | tf32 | small->mid | 4644 | 6.26 |
| case_SyntheticUSA | tf32_wmma | mid->big | 148 | 0.20 |
| case_SyntheticUSA | tf32_wmma | small->big | 8 | 0.01 |
| case_SyntheticUSA | tf32_wmma | small->mid | 4644 | 6.26 |

## Checks

- TC scalar hard-gate fronts (`nc > 32 || uc > 256`): 1 unique front(s) across the 11 cases.
- Full TSV artifacts:
  - `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/b1_factor_times.tsv`
  - `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/front_summary.tsv`
  - `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/mid_quantiles.tsv`
  - `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/dispatch_mismatch.tsv`
  - `custom_linear_solver/docs/05-reports/06-b1-tc-overhead-analysis-2026-06-07/tc_overhead_by_tier.tsv`

## Reading Guide

- If a case has high `mid_fsz_33_48_no_trailing`, TC mode is spending mid-kernel structure on fronts that never call trailing WMMA/PTX.
- If `fp16_wmma` has large `csc_mb` and high `padded_over_real`, its B=1 loss is plausibly staging/Csc/padding overhead rather than Tensor Core math throughput.
- If `tf32_ptx` still loses with low `csc_mb=0`, the remaining suspect is small phase3 work, padded staging, level mixing, launch/synchronization, and non-trailing panel/extend-add cost.
