# Same-Case cuDSS / cuITER FLOP Report (3K, 6K, 8K, 9K Bus)

작성일: 2026-05-12  
대상 실험: `exp/20260607-2`

## Case 선택

bus 규모별로 하나씩 골랐다. 4K~6K strict 범위에는 보유 case가 없어, 사용자가 요청한 대로 3천대 하나와 6/8/9천대 하나씩으로 재측정했다.

| band | case | buses | linear n | nnz |
|---|---|---:|---:|---:|
| 3K | `case3120sp` | 3120 | 5991 | 38329 |
| 6K | `case6468rte` | 6468 | 12643 | 87845 |
| 8K | `case8387pegase` | 8387 | 14908 | 110572 |
| 9K | `case9241pegase` | 9241 | 17036 | 129412 |

## 확인 사항

- BiCGSTAB은 선택 case의 active cuITER rows에서 **2 iterations 고정**이다. `linear_iters` unique 값은 `[2]`이다.
- `FLOPs / cuITER NR step`은 NR 반복 한 번에서 cuITER가 평균적으로 수행한 FLOPs다. BiCGSTAB 계산과 block-Jacobi apply가 합쳐진 값이며, 둘 중 하나만의 FLOPs가 아니다.
- 기본 cuITER solve FLOPs는 block-Jacobi setup을 제외한다. setup 포함 값은 `cuITER + BJ setup GFLOPs`에 별도로 둔다.
- cuDSS는 pure cuDSS NR loop 전체를 Nsight Compute로 재측정했다.

## Same-Case Summary

| band | case | buses | pure cuDSS NR iters | cuDSS calls | cuDSS NCU GFLOPs | all GPU NCU GFLOPs | cuITER NR steps | BiCGSTAB iters | cuITER solve GFLOPs | cuITER + BJ setup GFLOPs | FLOPs / cuITER NR step | FLOPs / BiCGSTAB iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3K | `case3120sp` | 3120 | 6 | 6 | `0.041494345` | `0.100707717` | 7 | 14 | `0.006318508` | `0.011759020` | `9.026440e+05` | `4.513220e+05` |
| 6K | `case6468rte` | 6468 | 3 | 3 | `0.034850190` | `0.075773110` | 2 | 4 | `0.003920584` | `0.007201480` | `1.960292e+06` | `9.801460e+05` |
| 8K | `case8387pegase` | 8387 | 3 | 3 | `0.052635726` | `0.100386086` | 2 | 4 | `0.004732048` | `0.008610960` | `2.366024e+06` | `1.183012e+06` |
| 9K | `case9241pegase` | 9241 | 6 | 6 | `0.114927287` | `0.216611222` | 5 | 10 | `0.013647800` | `0.024707000` | `2.729560e+06` | `1.364780e+06` |

합계:

- cuDSS NCU `cudss` class total: `0.243907548` GFLOPs
- cuDSS NCU `cublas` class total: `0.139189091` GFLOPs
- cuDSS NCU all GPU FP total: `0.493478135` GFLOPs
- cuITER solve-only total: `0.028618940` GFLOPs
- cuITER + block-Jacobi setup total: `0.052278460` GFLOPs
- cuITER NR steps total: `16`
- BiCGSTAB linear iterations total: `32`

## 측정 명령

각 case에 대해 다음 형태로 실행했다.

```bash
exp/20260607-2/tools/measure_gpu_flops_ncu.py \
  --output exp/20260607-2/results/flops_same_case_bus_3k_6k_8k_9k/cudss_case3120sp_ncu_flops.csv \
  -- exp/20260607-2/build/hybrid_nr_bench \
    --case case3120sp \
    --solver pure_cudss \
    --max-nr-iters 20 \
    --output exp/20260607-2/results/flops_same_case_bus_3k_6k_8k_9k/pure_cudss_case3120sp_summary.csv \
    --iter-output exp/20260607-2/results/flops_same_case_bus_3k_6k_8k_9k/pure_cudss_case3120sp_iters.csv \
    --timing-output exp/20260607-2/results/flops_same_case_bus_3k_6k_8k_9k/pure_cudss_case3120sp_timing.csv \
    --no-pure-cudss-baseline
```

## 해석

`cuDSS NCU GFLOPs`는 kernel name에 `cudss`가 포함된 GPU kernel의 dynamic FP instruction FLOPs다. `all GPU NCU GFLOPs`는 같은 pure cuDSS NR run에서 발생한 모든 GPU FP instruction 합계라 cuPF 주변 커널과 cuBLAS 호출이 포함된다.

cuITER FLOPs는 기존 `bicgstab_iter2_bs8_all78` hybrid run의 iteration CSV에서 산식으로 계산했다. 각 active cuITER NR step은 BiCGSTAB 2회 고정이며, `FLOPs / cuITER NR step`은 BiCGSTAB 2회와 block-Jacobi apply 2회/iteration, dot, update, norm check를 모두 포함한다.
