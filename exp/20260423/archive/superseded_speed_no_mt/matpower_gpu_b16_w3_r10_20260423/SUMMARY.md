# cuPF Benchmark `matpower_gpu_b16_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:35:41.627250+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 16
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
| case10ba | cuda_edge | True | 0.018325 | 0.009521 | 0.008803 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.019807 | 0.010114 | 0.009692 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.019623 | 0.009930 | 0.009692 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.030111 | 0.014205 | 0.015906 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018252 | 0.009543 | 0.008708 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.035398 | 0.015516 | 0.019881 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.154348 | 0.055805 | 0.098543 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.019579 | 0.009932 | 0.009647 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.018202 | 0.009566 | 0.008636 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.020337 | 0.010122 | 0.010214 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.020436 | 0.010364 | 0.010072 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018321 | 0.009565 | 0.008755 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018317 | 0.009564 | 0.008753 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018301 | 0.009521 | 0.008780 | n/a | 4.7 |
| case16ci | cuda_edge | True | 0.018213 | 0.009536 | 0.008676 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.018500 | 0.009562 | 0.008937 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.018525 | 0.009567 | 0.008957 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.035707 | 0.016542 | 0.019164 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.018455 | 0.009591 | 0.008863 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.038013 | 0.016900 | 0.021113 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018456 | 0.009590 | 0.008866 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.046847 | 0.018477 | 0.028369 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018681 | 0.009647 | 0.009033 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.047686 | 0.019937 | 0.027749 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.050021 | 0.020177 | 0.029844 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.049807 | 0.020095 | 0.029711 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.048958 | 0.019698 | 0.029259 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.045608 | 0.020084 | 0.025524 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.049370 | 0.020196 | 0.029173 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.052931 | 0.020871 | 0.032059 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.018624 | 0.009626 | 0.008998 | n/a | 4.6 |
| case30 | cuda_edge | True | 0.018680 | 0.009684 | 0.008995 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.023179 | 0.010982 | 0.012197 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.047319 | 0.020560 | 0.026759 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.018669 | 0.009677 | 0.008992 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.018662 | 0.009673 | 0.008988 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.054957 | 0.021057 | 0.033900 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.048814 | 0.021937 | 0.026877 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018568 | 0.009630 | 0.008937 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.018727 | 0.009644 | 0.009083 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.018608 | 0.009647 | 0.008961 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.018800 | 0.009656 | 0.009143 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.018431 | 0.009687 | 0.008743 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.017921 | 0.009428 | 0.008493 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018030 | 0.009470 | 0.008559 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018055 | 0.009488 | 0.008567 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.018999 | 0.009724 | 0.009274 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.018780 | 0.009705 | 0.009075 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.023753 | 0.011547 | 0.012206 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.023719 | 0.011565 | 0.012154 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019200 | 0.009846 | 0.009354 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.019527 | 0.009895 | 0.009631 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.018726 | 0.009812 | 0.008913 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.078733 | 0.033148 | 0.045584 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.079072 | 0.033340 | 0.045732 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.073191 | 0.032203 | 0.040987 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.077376 | 0.032345 | 0.045030 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.019170 | 0.009763 | 0.009406 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018063 | 0.009471 | 0.008591 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.018989 | 0.009737 | 0.009252 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.018989 | 0.009782 | 0.009206 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.099210 | 0.041740 | 0.057469 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.019456 | 0.009899 | 0.009556 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.020621 | 0.010204 | 0.010417 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.018282 | 0.009518 | 0.008763 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.122752 | 0.045395 | 0.077357 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.019471 | 0.009905 | 0.009566 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.018282 | 0.009519 | 0.008762 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018497 | 0.009534 | 0.008963 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.121109 | 0.045200 | 0.075909 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.020547 | 0.010467 | 0.010079 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.043401 | 0.018338 | 0.025062 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.260401 | 0.098333 | 0.162068 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.024319 | 0.011765 | 0.012554 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.775233 | 0.256874 | 0.518359 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.019511 | 0.009906 | 0.009604 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 1.021554 | 0.323629 | 0.697924 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.018529 | 0.009673 | 0.008856 | n/a | 3.0 |

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
