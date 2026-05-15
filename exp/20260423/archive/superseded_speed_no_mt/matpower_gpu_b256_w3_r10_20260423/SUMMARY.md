# cuPF Benchmark `matpower_gpu_b256_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:43:49.428516+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 256
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
| case10ba | cuda_edge | True | 0.018731 | 0.009530 | 0.009201 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.026079 | 0.010854 | 0.015225 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.027040 | 0.011132 | 0.015908 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.097976 | 0.015686 | 0.082290 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018698 | 0.009545 | 0.009153 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.133747 | 0.017198 | 0.116549 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.942415 | 0.071015 | 0.871399 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.027243 | 0.011604 | 0.015639 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.018601 | 0.009581 | 0.009020 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.031799 | 0.011834 | 0.019964 | n/a | 6.3 |
| case145 | cuda_edge | True | 0.028151 | 0.011579 | 0.016571 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018800 | 0.009572 | 0.009227 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018777 | 0.009560 | 0.009217 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018830 | 0.009540 | 0.009289 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.018622 | 0.009535 | 0.009087 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.019164 | 0.009578 | 0.009586 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.019329 | 0.009582 | 0.009747 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.111527 | 0.020316 | 0.091211 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.019089 | 0.009605 | 0.009483 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.132867 | 0.021027 | 0.111840 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.019216 | 0.009594 | 0.009622 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.220043 | 0.022350 | 0.197692 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.019702 | 0.009688 | 0.010013 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.198739 | 0.024102 | 0.174637 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.227475 | 0.024292 | 0.203182 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.222612 | 0.023928 | 0.198684 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.221130 | 0.023485 | 0.197645 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.168689 | 0.024156 | 0.144533 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.216755 | 0.024330 | 0.192424 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.253119 | 0.024912 | 0.228206 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.019992 | 0.009645 | 0.010346 | n/a | 5.0 |
| case30 | cuda_edge | True | 0.019818 | 0.009696 | 0.010122 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.049443 | 0.012720 | 0.036723 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.182401 | 0.024730 | 0.157670 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.019824 | 0.009703 | 0.010121 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.019788 | 0.009693 | 0.010095 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.274785 | 0.024867 | 0.249918 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.171633 | 0.026055 | 0.145577 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.019806 | 0.009640 | 0.010166 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.020288 | 0.009656 | 0.010631 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.019930 | 0.009656 | 0.010273 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.020721 | 0.009676 | 0.011045 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.019231 | 0.009712 | 0.009519 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.018066 | 0.009438 | 0.008628 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018180 | 0.009476 | 0.008703 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018197 | 0.009482 | 0.008714 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.022399 | 0.010419 | 0.011980 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.021289 | 0.010463 | 0.010825 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.050489 | 0.013364 | 0.037125 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.050574 | 0.013366 | 0.037207 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.022345 | 0.009879 | 0.012465 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.023192 | 0.009831 | 0.013360 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.020022 | 0.009822 | 0.010199 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.352814 | 0.044714 | 0.308099 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.352313 | 0.044835 | 0.307477 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.300289 | 0.043699 | 0.256589 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.354909 | 0.043813 | 0.311095 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.023813 | 0.010439 | 0.013373 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018226 | 0.009475 | 0.008750 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.022594 | 0.010438 | 0.012154 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.023305 | 0.010739 | 0.012566 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.459968 | 0.054720 | 0.405247 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.024711 | 0.010984 | 0.013727 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.028561 | 0.010627 | 0.017933 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.018626 | 0.009519 | 0.009106 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.703188 | 0.059547 | 0.643640 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.025442 | 0.011061 | 0.014381 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.018639 | 0.009531 | 0.009108 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018936 | 0.009522 | 0.009413 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.705683 | 0.060361 | 0.645321 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.029949 | 0.012280 | 0.017669 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.182994 | 0.022916 | 0.160078 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 1.546382 | 0.118947 | 1.427434 | n/a | 5.2 |
| case_ACTIVSg500 | cuda_edge | True | 0.052980 | 0.013547 | 0.039432 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 5.516106 | 0.276661 | 5.239444 | n/a | 7.7 |
| case_RTS_GMLC | cuda_edge | True | 0.024209 | 0.009930 | 0.014279 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 6.609474 | 0.353269 | 6.256205 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.019435 | 0.009679 | 0.009755 | n/a | 3.0 |

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
