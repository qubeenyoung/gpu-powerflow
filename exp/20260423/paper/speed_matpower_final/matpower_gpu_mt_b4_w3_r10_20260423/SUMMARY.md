# cuPF Benchmark `matpower_gpu_mt_b4_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:07:30.657366+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 4
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
| case10ba | cuda_edge | True | 0.018446 | 0.009601 | 0.008845 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.021366 | 0.011181 | 0.010184 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.022861 | 0.010852 | 0.012009 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.029746 | 0.012596 | 0.017150 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018353 | 0.009590 | 0.008763 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.030990 | 0.016779 | 0.014211 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.080601 | 0.034557 | 0.046044 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.020874 | 0.010716 | 0.010158 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.020842 | 0.011304 | 0.009537 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.019787 | 0.010047 | 0.009739 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.023935 | 0.010244 | 0.013691 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.021853 | 0.011284 | 0.010568 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018406 | 0.009625 | 0.008780 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.020429 | 0.009608 | 0.010820 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.021722 | 0.011537 | 0.010185 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.018502 | 0.009608 | 0.008893 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.019357 | 0.010413 | 0.008944 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.027636 | 0.013917 | 0.013719 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.018502 | 0.009665 | 0.008836 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.034213 | 0.014091 | 0.020121 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018486 | 0.009651 | 0.008834 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.032062 | 0.014789 | 0.017272 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018687 | 0.009695 | 0.008991 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.033010 | 0.015747 | 0.017262 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.033534 | 0.015704 | 0.017829 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.037556 | 0.019620 | 0.017936 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.039407 | 0.016646 | 0.022761 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.031625 | 0.015548 | 0.016076 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.033469 | 0.015801 | 0.017667 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.034086 | 0.015799 | 0.018287 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.018622 | 0.009679 | 0.008943 | n/a | 4.4 |
| case30 | cuda_edge | True | 0.018670 | 0.009727 | 0.008943 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.024909 | 0.014213 | 0.010695 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.036644 | 0.019971 | 0.016672 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.018691 | 0.009747 | 0.008943 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.018686 | 0.009746 | 0.008939 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.040282 | 0.016190 | 0.024092 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.033670 | 0.016596 | 0.017073 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018576 | 0.009685 | 0.008890 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.021387 | 0.011529 | 0.009858 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.019415 | 0.010513 | 0.008901 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.018765 | 0.009709 | 0.009056 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.018427 | 0.009728 | 0.008698 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.018047 | 0.009501 | 0.008546 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018188 | 0.009558 | 0.008630 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018167 | 0.009542 | 0.008625 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.021838 | 0.012698 | 0.009139 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.018726 | 0.009753 | 0.008973 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.022515 | 0.011117 | 0.011398 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.025602 | 0.014970 | 0.010632 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019085 | 0.009890 | 0.009195 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.027769 | 0.014122 | 0.013647 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.018704 | 0.009885 | 0.008819 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.046967 | 0.022258 | 0.024709 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.047837 | 0.022852 | 0.024985 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.044735 | 0.021666 | 0.023068 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.046278 | 0.021969 | 0.024309 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.019105 | 0.009825 | 0.009279 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018529 | 0.009542 | 0.008987 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.019675 | 0.009853 | 0.009822 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.019438 | 0.010337 | 0.009101 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.061719 | 0.030159 | 0.031559 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.022448 | 0.013055 | 0.009393 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.020360 | 0.010219 | 0.010140 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.019998 | 0.009579 | 0.010419 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.067513 | 0.029125 | 0.038387 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.019267 | 0.009887 | 0.009379 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.018405 | 0.009589 | 0.008816 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018604 | 0.009594 | 0.009009 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.072770 | 0.028352 | 0.044418 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.019971 | 0.010334 | 0.009637 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.033117 | 0.016254 | 0.016862 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.124488 | 0.055613 | 0.068874 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.021949 | 0.011135 | 0.010814 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.310343 | 0.123090 | 0.187252 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.019323 | 0.009923 | 0.009400 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.401829 | 0.161596 | 0.240232 | n/a | 7.9 |
| case_ieee30 | cuda_edge | True | 0.018546 | 0.009729 | 0.008816 | n/a | 3.0 |

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
