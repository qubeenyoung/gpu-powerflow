# cuPF Benchmark `matpower_precision_mixed_residuals_b1_tol1e-8_maxit10_r1_20260423`

## Setup

- Created UTC: 2026-04-23T03:20:20.730285+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 0
- Repeats: 1
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
| case10ba | cuda_edge | True | 0.275379 | 0.256442 | 0.018936 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.198409 | 0.179022 | 0.019386 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.198582 | 0.178830 | 0.019752 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.202797 | 0.177649 | 0.025148 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.196627 | 0.178146 | 0.018480 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.204601 | 0.176999 | 0.027601 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.321317 | 0.194369 | 0.126948 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.193560 | 0.174069 | 0.019490 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.197540 | 0.178728 | 0.018811 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.194880 | 0.174551 | 0.020328 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.198343 | 0.178796 | 0.019546 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.198297 | 0.176504 | 0.021793 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.192570 | 0.173922 | 0.018647 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.195748 | 0.177166 | 0.018582 | n/a | 4.0 |
| case16ci | cuda_edge | True | 0.192808 | 0.174018 | 0.018790 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.193357 | 0.174302 | 0.019055 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.197946 | 0.178872 | 0.019073 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.205020 | 0.179372 | 0.025648 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.194436 | 0.175635 | 0.018801 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.209123 | 0.180475 | 0.028648 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.206773 | 0.187848 | 0.018924 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.224264 | 0.183970 | 0.040293 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.196846 | 0.177845 | 0.019001 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.214018 | 0.177253 | 0.036765 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.217202 | 0.176545 | 0.040656 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.229077 | 0.180998 | 0.048078 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.220722 | 0.180535 | 0.040187 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.229406 | 0.199964 | 0.029442 | n/a | 3.0 |
| case2868rte | cuda_edge | True | 0.218101 | 0.177090 | 0.041010 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.231842 | 0.188014 | 0.043827 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.193963 | 0.175026 | 0.018937 | n/a | 5.0 |
| case30 | cuda_edge | True | 0.210839 | 0.191883 | 0.018956 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.198749 | 0.177075 | 0.021673 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.215092 | 0.180896 | 0.034195 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.196135 | 0.177214 | 0.018920 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.193875 | 0.174880 | 0.018994 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.228429 | 0.181225 | 0.047204 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.209496 | 0.178047 | 0.031449 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.193730 | 0.174773 | 0.018956 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.192011 | 0.172798 | 0.019213 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.197228 | 0.177933 | 0.019294 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.203223 | 0.184051 | 0.019172 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.193225 | 0.174746 | 0.018478 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.198457 | 0.179830 | 0.018627 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.213372 | 0.194135 | 0.019236 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.188523 | 0.169847 | 0.018676 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.216615 | 0.197001 | 0.019614 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.198534 | 0.179651 | 0.018883 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.199855 | 0.178382 | 0.021473 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.200318 | 0.178596 | 0.021722 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.206559 | 0.187489 | 0.019069 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.196947 | 0.177368 | 0.019578 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.200263 | 0.181812 | 0.018451 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.241620 | 0.189118 | 0.052502 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.236642 | 0.184370 | 0.052271 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.230553 | 0.187124 | 0.043429 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.238667 | 0.186281 | 0.052386 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.197742 | 0.178432 | 0.019310 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.193837 | 0.175431 | 0.018406 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.196291 | 0.177005 | 0.019286 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.197701 | 0.178608 | 0.019093 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.257185 | 0.191148 | 0.066036 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.198230 | 0.178543 | 0.019686 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.195198 | 0.175066 | 0.020131 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.197154 | 0.178228 | 0.018926 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.294106 | 0.194493 | 0.099612 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.194475 | 0.174943 | 0.019532 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.194654 | 0.175695 | 0.018959 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.194383 | 0.175133 | 0.019249 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.285944 | 0.193532 | 0.092411 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.197866 | 0.178418 | 0.019447 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.205908 | 0.176990 | 0.028917 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.393728 | 0.221373 | 0.172354 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.199759 | 0.178239 | 0.021519 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.929392 | 0.300730 | 0.628662 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.200583 | 0.180325 | 0.020257 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 1.162435 | 0.326888 | 0.835546 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.196163 | 0.177450 | 0.018712 | n/a | 3.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
