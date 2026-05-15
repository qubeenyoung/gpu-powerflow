# cuPF Benchmark `matpower_gpu_mt_b64_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:11:17.466804+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 64
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
| case10ba | cuda_edge | True | 0.018587 | 0.009606 | 0.008980 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.022519 | 0.011730 | 0.010789 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.020894 | 0.009958 | 0.010936 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.037602 | 0.014214 | 0.023388 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.020827 | 0.011203 | 0.009622 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.053527 | 0.020494 | 0.033033 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.279948 | 0.039523 | 0.240424 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.024362 | 0.009946 | 0.014415 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.019477 | 0.009631 | 0.009845 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.022028 | 0.010062 | 0.011966 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.025614 | 0.014386 | 0.011227 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018587 | 0.009644 | 0.008942 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018556 | 0.009619 | 0.008937 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018597 | 0.009602 | 0.008995 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.018459 | 0.009592 | 0.008866 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.018704 | 0.009613 | 0.009090 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.018808 | 0.009651 | 0.009156 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.048937 | 0.017582 | 0.031354 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.018668 | 0.009644 | 0.009023 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.050034 | 0.014861 | 0.035173 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018698 | 0.009644 | 0.009054 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.080428 | 0.015209 | 0.065219 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018954 | 0.009700 | 0.009253 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.075375 | 0.016204 | 0.059170 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.082440 | 0.016199 | 0.066240 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.082116 | 0.021519 | 0.060597 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.081752 | 0.021976 | 0.059776 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.063151 | 0.016128 | 0.047023 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.079919 | 0.021697 | 0.058221 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.085195 | 0.017091 | 0.068103 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.021618 | 0.009688 | 0.011929 | n/a | 5.0 |
| case30 | cuda_edge | True | 0.018993 | 0.009737 | 0.009255 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.027791 | 0.011186 | 0.016605 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.065366 | 0.016405 | 0.048961 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.019008 | 0.009751 | 0.009257 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.021576 | 0.012309 | 0.009267 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.095936 | 0.017466 | 0.078470 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.063209 | 0.017233 | 0.045976 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018939 | 0.009698 | 0.009241 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.022265 | 0.012016 | 0.010249 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.021038 | 0.011778 | 0.009259 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.020403 | 0.009746 | 0.010656 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.021398 | 0.012461 | 0.008937 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.018399 | 0.009828 | 0.008570 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.019877 | 0.010200 | 0.009677 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018522 | 0.009867 | 0.008654 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.019755 | 0.009869 | 0.009886 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.019268 | 0.009779 | 0.009489 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.028173 | 0.012606 | 0.015566 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.028207 | 0.012630 | 0.015576 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019936 | 0.009908 | 0.010027 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.020282 | 0.009870 | 0.010411 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.019065 | 0.009847 | 0.009217 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.117389 | 0.023931 | 0.093458 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.113075 | 0.024254 | 0.088820 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.105387 | 0.030745 | 0.074640 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.122640 | 0.028561 | 0.094079 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.020051 | 0.009810 | 0.010241 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.019021 | 0.009908 | 0.009112 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.019639 | 0.009784 | 0.009854 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.019716 | 0.009793 | 0.009922 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.145527 | 0.029534 | 0.115992 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.026105 | 0.013476 | 0.012629 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.022183 | 0.010243 | 0.011939 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.019417 | 0.009587 | 0.009829 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.212272 | 0.032675 | 0.179597 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.020461 | 0.009881 | 0.010579 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.020439 | 0.010388 | 0.010051 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018746 | 0.009601 | 0.009145 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.206592 | 0.032551 | 0.174041 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.021834 | 0.010361 | 0.011472 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.062941 | 0.018333 | 0.044607 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.452395 | 0.070832 | 0.381562 | n/a | 5.1 |
| case_ACTIVSg500 | cuda_edge | True | 0.028962 | 0.012302 | 0.016660 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 1.484422 | 0.158704 | 1.325717 | n/a | 7.4 |
| case_RTS_GMLC | cuda_edge | True | 0.023758 | 0.009920 | 0.013838 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 1.873746 | 0.195808 | 1.677937 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.019603 | 0.009746 | 0.009857 | n/a | 3.0 |

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
