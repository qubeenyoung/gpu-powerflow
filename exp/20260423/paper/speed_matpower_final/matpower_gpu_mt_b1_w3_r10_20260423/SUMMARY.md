# cuPF Benchmark `matpower_gpu_mt_b1_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:06:13.675779+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cuda_edge | True | 0.010961 | 0.010474 | 0.000487 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.010528 | 0.010002 | 0.000526 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.010531 | 0.009898 | 0.000632 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.013447 | 0.012594 | 0.000853 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.009982 | 0.009579 | 0.000401 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.014982 | 0.013479 | 0.001502 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.042276 | 0.035089 | 0.007186 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.010500 | 0.009914 | 0.000586 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.009913 | 0.009624 | 0.000289 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.010858 | 0.010022 | 0.000836 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.010867 | 0.010253 | 0.000614 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.010016 | 0.009606 | 0.000410 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.010010 | 0.009603 | 0.000407 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.009977 | 0.009590 | 0.000386 | n/a | 4.0 |
| case16ci | cuda_edge | True | 0.010801 | 0.010424 | 0.000376 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.015608 | 0.015092 | 0.000515 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.010166 | 0.009630 | 0.000535 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.014778 | 0.013849 | 0.000929 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.010097 | 0.009662 | 0.000434 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.020602 | 0.019207 | 0.001395 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.010060 | 0.009634 | 0.000425 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.017357 | 0.014769 | 0.002587 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.010965 | 0.010439 | 0.000526 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.017590 | 0.015504 | 0.002086 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.024760 | 0.022331 | 0.002429 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.018225 | 0.015677 | 0.002547 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.025400 | 0.022993 | 0.002406 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.016814 | 0.015539 | 0.001274 | n/a | 3.4 |
| case2868rte | cuda_edge | True | 0.018909 | 0.016468 | 0.002441 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.018430 | 0.015765 | 0.002664 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.010127 | 0.009657 | 0.000469 | n/a | 4.3 |
| case30 | cuda_edge | True | 0.010170 | 0.009723 | 0.000447 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.011812 | 0.010623 | 0.001188 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.017532 | 0.015916 | 0.001616 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.010172 | 0.009721 | 0.000450 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.012274 | 0.011817 | 0.000457 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.019286 | 0.016245 | 0.003040 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.017950 | 0.016610 | 0.001339 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.010113 | 0.009677 | 0.000435 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.010257 | 0.009706 | 0.000550 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.010122 | 0.009685 | 0.000436 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.010272 | 0.009700 | 0.000571 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.009944 | 0.009731 | 0.000212 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.010823 | 0.010542 | 0.000281 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.009856 | 0.009518 | 0.000337 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.009865 | 0.009532 | 0.000332 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.014889 | 0.014304 | 0.000585 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.010198 | 0.009745 | 0.000452 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.011773 | 0.011096 | 0.000676 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.011766 | 0.011093 | 0.000673 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.010402 | 0.009868 | 0.000533 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.010635 | 0.009827 | 0.000808 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.010051 | 0.009828 | 0.000222 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.024886 | 0.022553 | 0.002333 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.030618 | 0.028309 | 0.002308 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.023734 | 0.022045 | 0.001689 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.024483 | 0.022044 | 0.002439 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.017090 | 0.016474 | 0.000615 | n/a | 4.9 |
| case6ww | cuda_edge | True | 0.009949 | 0.009584 | 0.000364 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.010298 | 0.009743 | 0.000554 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.011032 | 0.010540 | 0.000491 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.031821 | 0.028595 | 0.003225 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.010524 | 0.009881 | 0.000642 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.011355 | 0.010231 | 0.001123 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.010041 | 0.009567 | 0.000473 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.035412 | 0.029521 | 0.005890 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.010520 | 0.009876 | 0.000643 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.010043 | 0.009572 | 0.000470 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.010238 | 0.009568 | 0.000669 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.033898 | 0.029291 | 0.004606 | n/a | 5.9 |
| case_ACTIVSg200 | cuda_edge | True | 0.012559 | 0.012103 | 0.000455 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.016634 | 0.014736 | 0.001897 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.063944 | 0.055839 | 0.008104 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.011909 | 0.011097 | 0.000811 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.155253 | 0.128376 | 0.026876 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.010548 | 0.009882 | 0.000665 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.194420 | 0.162935 | 0.031484 | n/a | 7.4 |
| case_ieee30 | cuda_edge | True | 0.010067 | 0.009732 | 0.000334 | n/a | 3.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
