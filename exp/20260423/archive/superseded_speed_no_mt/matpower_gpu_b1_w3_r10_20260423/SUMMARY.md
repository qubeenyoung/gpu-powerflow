# cuPF Benchmark `matpower_gpu_b1_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:32:41.646842+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 1
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
| case10ba | cuda_edge | True | 0.010028 | 0.009541 | 0.000487 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.010618 | 0.010089 | 0.000529 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.010560 | 0.009930 | 0.000629 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.014650 | 0.013796 | 0.000853 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.009944 | 0.009538 | 0.000405 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.016973 | 0.015482 | 0.001491 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.062845 | 0.055651 | 0.007193 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.010526 | 0.009930 | 0.000596 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.009828 | 0.009541 | 0.000287 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.010924 | 0.010079 | 0.000844 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.010956 | 0.010343 | 0.000613 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.009934 | 0.009525 | 0.000408 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.009933 | 0.009526 | 0.000406 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.009895 | 0.009509 | 0.000386 | n/a | 4.0 |
| case16ci | cuda_edge | True | 0.009870 | 0.009506 | 0.000364 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.010059 | 0.009527 | 0.000532 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.010104 | 0.009578 | 0.000525 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.017432 | 0.016506 | 0.000926 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.010015 | 0.009581 | 0.000434 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.018344 | 0.016953 | 0.001390 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.009987 | 0.009562 | 0.000424 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.021049 | 0.018466 | 0.002582 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.010240 | 0.009714 | 0.000526 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.021992 | 0.019904 | 0.002087 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.022489 | 0.020064 | 0.002425 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.022598 | 0.020053 | 0.002544 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.022045 | 0.019650 | 0.002394 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.021610 | 0.020175 | 0.001435 | n/a | 3.8 |
| case2868rte | cuda_edge | True | 0.022624 | 0.020184 | 0.002440 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.023496 | 0.020826 | 0.002670 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.010103 | 0.009597 | 0.000505 | n/a | 4.6 |
| case30 | cuda_edge | True | 0.010124 | 0.009678 | 0.000445 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.012148 | 0.010956 | 0.001191 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.022142 | 0.020524 | 0.001617 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.010090 | 0.009643 | 0.000447 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.010104 | 0.009659 | 0.000445 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.024090 | 0.021049 | 0.003041 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.023239 | 0.021903 | 0.001335 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.010053 | 0.009618 | 0.000434 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.010181 | 0.009631 | 0.000548 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.010058 | 0.009619 | 0.000439 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.010221 | 0.009650 | 0.000570 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.009880 | 0.009668 | 0.000212 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.009703 | 0.009425 | 0.000278 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.009795 | 0.009460 | 0.000335 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.009797 | 0.009464 | 0.000333 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.010284 | 0.009704 | 0.000579 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.010181 | 0.009726 | 0.000455 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.012193 | 0.011520 | 0.000673 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.012202 | 0.011529 | 0.000672 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.010361 | 0.009830 | 0.000531 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.010612 | 0.009802 | 0.000810 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.010031 | 0.009809 | 0.000222 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.035484 | 0.033150 | 0.002334 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.035595 | 0.033293 | 0.002302 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.033881 | 0.032186 | 0.001694 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.034881 | 0.032421 | 0.002459 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.010368 | 0.009752 | 0.000615 | n/a | 4.9 |
| case6ww | cuda_edge | True | 0.009816 | 0.009462 | 0.000354 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.010278 | 0.009724 | 0.000553 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.010203 | 0.009715 | 0.000488 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.044744 | 0.041520 | 0.003223 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.010523 | 0.009879 | 0.000643 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.011303 | 0.010179 | 0.001123 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.009976 | 0.009503 | 0.000473 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.051220 | 0.045327 | 0.005892 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.010533 | 0.009886 | 0.000646 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.009978 | 0.009508 | 0.000470 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.010159 | 0.009500 | 0.000659 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.049463 | 0.045092 | 0.004371 | n/a | 5.6 |
| case_ACTIVSg200 | cuda_edge | True | 0.010970 | 0.010511 | 0.000459 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.020205 | 0.018309 | 0.001896 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.103166 | 0.094858 | 0.008308 | n/a | 5.1 |
| case_ACTIVSg500 | cuda_edge | True | 0.012559 | 0.011748 | 0.000811 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.272109 | 0.245226 | 0.026882 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.010578 | 0.009916 | 0.000662 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.357402 | 0.325082 | 0.032320 | n/a | 7.6 |
| case_ieee30 | cuda_edge | True | 0.009994 | 0.009657 | 0.000337 | n/a | 3.0 |

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
