# cuPF Benchmark `matpower_gpu_mt_b256_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:16:46.227264+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 256
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
| case10ba | cuda_edge | True | 0.018871 | 0.009598 | 0.009273 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.029615 | 0.010914 | 0.018701 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.026971 | 0.011081 | 0.015890 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.106718 | 0.026328 | 0.080389 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018991 | 0.009766 | 0.009225 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.130994 | 0.015202 | 0.115792 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.909205 | 0.051991 | 0.857213 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.027234 | 0.011580 | 0.015653 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.020491 | 0.011403 | 0.009088 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.031166 | 0.011689 | 0.019477 | n/a | 6.1 |
| case145 | cuda_edge | True | 0.027870 | 0.011420 | 0.016449 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018929 | 0.009641 | 0.009287 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.019009 | 0.009726 | 0.009283 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018983 | 0.009619 | 0.009363 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.024879 | 0.012091 | 0.012788 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.019279 | 0.009638 | 0.009641 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.019461 | 0.009647 | 0.009813 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.114509 | 0.024086 | 0.090423 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.019210 | 0.009653 | 0.009557 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.128933 | 0.018930 | 0.110003 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.019363 | 0.009660 | 0.009703 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.218329 | 0.018667 | 0.199662 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.019867 | 0.009787 | 0.010080 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.191359 | 0.021163 | 0.170196 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.228594 | 0.025611 | 0.202982 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.213932 | 0.019525 | 0.194406 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.213174 | 0.019223 | 0.193950 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.166216 | 0.025678 | 0.140538 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.208949 | 0.020656 | 0.188293 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.251474 | 0.027385 | 0.224088 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.020147 | 0.009732 | 0.010415 | n/a | 5.0 |
| case30 | cuda_edge | True | 0.021520 | 0.009765 | 0.011754 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.048795 | 0.012410 | 0.036384 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.179238 | 0.025771 | 0.153467 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.019939 | 0.009760 | 0.010179 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.019937 | 0.009760 | 0.010177 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.265635 | 0.020716 | 0.244919 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.164091 | 0.020784 | 0.143306 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.020874 | 0.010626 | 0.010247 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.020410 | 0.009699 | 0.010710 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.020043 | 0.009704 | 0.010338 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.020883 | 0.009766 | 0.011117 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.019323 | 0.009764 | 0.009558 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.018581 | 0.009507 | 0.009074 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.019402 | 0.010200 | 0.009201 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018326 | 0.009536 | 0.008789 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.029325 | 0.014310 | 0.015015 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.024326 | 0.010517 | 0.013809 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.049777 | 0.012918 | 0.036858 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.049818 | 0.012939 | 0.036879 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.022427 | 0.009909 | 0.012517 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.023290 | 0.009867 | 0.013422 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.023593 | 0.013348 | 0.010244 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.331672 | 0.034193 | 0.297478 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.342198 | 0.039155 | 0.303042 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.284174 | 0.036428 | 0.247745 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.337715 | 0.037563 | 0.300151 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.023905 | 0.010480 | 0.013424 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018402 | 0.009573 | 0.008829 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.024749 | 0.012588 | 0.012160 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.023413 | 0.010798 | 0.012615 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.438457 | 0.047551 | 0.390906 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.032397 | 0.015055 | 0.017341 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.028554 | 0.010594 | 0.017960 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.018772 | 0.009603 | 0.009168 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.681450 | 0.046835 | 0.634615 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.025526 | 0.011100 | 0.014426 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.024944 | 0.013945 | 0.010998 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.020201 | 0.010735 | 0.009466 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.684341 | 0.054231 | 0.630110 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.029727 | 0.012159 | 0.017567 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.180723 | 0.018927 | 0.161795 | n/a | 4.1 |
| case_ACTIVSg25k | cuda_edge | True | 1.536234 | 0.085557 | 1.450676 | n/a | 5.6 |
| case_ACTIVSg500 | cuda_edge | True | 0.052274 | 0.012895 | 0.039378 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 5.122863 | 0.153633 | 4.969229 | n/a | 7.4 |
| case_RTS_GMLC | cuda_edge | True | 0.024076 | 0.009884 | 0.014192 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 6.309643 | 0.195001 | 6.114642 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.019610 | 0.009775 | 0.009834 | n/a | 3.0 |

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
