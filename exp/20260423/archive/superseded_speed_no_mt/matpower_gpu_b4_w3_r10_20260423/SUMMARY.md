# cuPF Benchmark `matpower_gpu_b4_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:34:05.987688+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 4
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cuda_edge | True | 0.018500 | 0.009669 | 0.008831 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.019526 | 0.010092 | 0.009434 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.019320 | 0.009924 | 0.009395 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.027800 | 0.013855 | 0.013945 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018242 | 0.009538 | 0.008704 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.031767 | 0.015487 | 0.016279 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.122007 | 0.055840 | 0.066166 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.019317 | 0.009944 | 0.009373 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.018209 | 0.009595 | 0.008614 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.019910 | 0.010122 | 0.009788 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.020148 | 0.010374 | 0.009774 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018278 | 0.009552 | 0.008726 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018280 | 0.009556 | 0.008723 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018301 | 0.009529 | 0.008771 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.018172 | 0.009522 | 0.008649 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.018407 | 0.009575 | 0.008831 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.018449 | 0.009572 | 0.008877 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.032862 | 0.016535 | 0.016327 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.018370 | 0.009599 | 0.008770 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.034248 | 0.016890 | 0.017358 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018365 | 0.009589 | 0.008776 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.039467 | 0.018494 | 0.020972 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018567 | 0.009632 | 0.008935 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.041324 | 0.019913 | 0.021410 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.042237 | 0.020073 | 0.022164 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.042528 | 0.020087 | 0.022440 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.041501 | 0.019663 | 0.021838 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.040630 | 0.020056 | 0.020574 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.042275 | 0.020213 | 0.022062 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.044135 | 0.020830 | 0.023304 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.018488 | 0.009625 | 0.008863 | n/a | 4.2 |
| case30 | cuda_edge | True | 0.018565 | 0.009680 | 0.008884 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.021961 | 0.010961 | 0.011000 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.041836 | 0.020540 | 0.021296 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.018549 | 0.009671 | 0.008877 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.018566 | 0.009688 | 0.008878 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.045262 | 0.021049 | 0.024213 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.044316 | 0.021968 | 0.022347 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018460 | 0.009632 | 0.008827 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.018587 | 0.009646 | 0.008941 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.018475 | 0.009626 | 0.008848 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.018681 | 0.009667 | 0.009013 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.018338 | 0.009684 | 0.008654 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.017915 | 0.009431 | 0.008484 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018017 | 0.009462 | 0.008554 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018053 | 0.009500 | 0.008552 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.018818 | 0.009728 | 0.009090 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.018670 | 0.009729 | 0.008940 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.022649 | 0.011587 | 0.011062 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.022601 | 0.011541 | 0.011060 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019024 | 0.009861 | 0.009163 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.019200 | 0.009803 | 0.009397 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.018604 | 0.009803 | 0.008800 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.068747 | 0.033260 | 0.035487 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.068683 | 0.033347 | 0.035335 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.065244 | 0.032210 | 0.033034 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.067192 | 0.032528 | 0.034664 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.019036 | 0.009776 | 0.009260 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018083 | 0.009492 | 0.008590 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.018879 | 0.009736 | 0.009143 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.018785 | 0.009726 | 0.009059 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.085933 | 0.041652 | 0.044281 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.019289 | 0.009892 | 0.009397 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.020326 | 0.010205 | 0.010121 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.018279 | 0.009531 | 0.008748 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.098402 | 0.045380 | 0.053022 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.019267 | 0.009887 | 0.009379 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.018270 | 0.009521 | 0.008748 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018470 | 0.009525 | 0.008945 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.097396 | 0.045164 | 0.052232 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.020225 | 0.010469 | 0.009755 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.037984 | 0.018348 | 0.019636 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.202435 | 0.094435 | 0.108000 | n/a | 5.1 |
| case_ACTIVSg500 | cuda_edge | True | 0.023215 | 0.011790 | 0.011425 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.550397 | 0.245635 | 0.304761 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.019330 | 0.009914 | 0.009416 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.725228 | 0.323525 | 0.401703 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.018426 | 0.009675 | 0.008751 | n/a | 3.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

Use the operators benchmark binary directly for profiling. Prefer `--warmup 1` to remove one-time CUDA setup from measured repeats.

```bash
nsys profile --trace=cuda,nvtx -o cupf_nsys \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
