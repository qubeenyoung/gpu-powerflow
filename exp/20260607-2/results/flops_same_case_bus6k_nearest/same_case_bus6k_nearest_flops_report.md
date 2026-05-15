# Same-Case cuDSS / cuITER FLOP Report (Nearest Bus Scale)

작성일: 2026-05-12  
대상 실험: `exp/20260607-2`

## 기준 정정

요청 기준은 Jacobian/linear-system 차원 `n`이 아니라 전력계통 bus 수 기준이다. 현재 `exp/20260607-2`와 `datasets/matpower8.1/data`에는 strict `4,000 <= buses <= 6,000` case가 없다. 따라서 이번 재측정은 같은 결과셋에서 가장 가까운 상위 bus 규모인 6.4K급 case 네 개로 수행했다.

대상 case: `case6468rte`, `case6470rte`, `case6495rte`, `case6515rte`.

## 확인 사항

- BiCGSTAB은 이 결과셋에서 **2 iterations 고정**이다. 선택 case의 active cuITER rows에서 `linear_iters` unique 값은 `[2]`이다.
- `FLOPs / cuITER NR step`은 NR 반복 한 번에서 cuITER가 수행한 평균 FLOPs이며, BiCGSTAB과 block-Jacobi apply를 합친 값이다. 즉 BiCGSTAB 또는 block-Jacobi 단독 값이 아니다.
- 기본 cuITER solve FLOPs는 block-Jacobi setup을 제외한다. setup 포함 값은 별도 열로 보고한다.
- cuDSS는 API FLOP count가 없어 Nsight Compute SASS FP instruction counter로 pure cuDSS 전체 NR loop를 재측정했다.

## Same-Case Summary

| case | buses | n | nnz | pure cuDSS NR iters | cuDSS calls | cuDSS NCU GFLOPs | all GPU NCU GFLOPs | cuITER NR steps | BiCGSTAB iters | cuITER solve GFLOPs | cuITER + BJ setup GFLOPs | FLOPs / cuITER NR step | FLOPs / BiCGSTAB iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case6468rte | 6468 | 12643 | 87845 | 3 | 3 | `0.034850224` | `0.075773693` | 2 | 4 | `0.003920584` | `0.007201480` | `1.960292e+06` | `9.801460e+05` |
| case6470rte | 6470 | 12485 | 86405 | 3 | 3 | `0.034996481` | `0.075870973` | 2 | 4 | `0.003867448` | `0.007103288` | `1.933724e+06` | `9.668620e+05` |
| case6495rte | 6495 | 12495 | 86287 | 2 | 2 | `0.023344264` | `0.050626866` | 0 | 0 | `0.000000000` | `0.000000000` | `0.000000e+00` | `0.000000e+00` |
| case6515rte | 6515 | 12535 | 86561 | 3 | 3 | `0.034879708` | `0.075872324` | 2 | 4 | `0.003876104` | `0.007138568` | `1.938052e+06` | `9.690260e+05` |

합계:

- cuDSS NCU `cudss` class total: `0.128070677` GFLOPs
- cuDSS NCU `cublas` class total: `0.085522443` GFLOPs
- cuDSS NCU all GPU FP total: `0.278143856` GFLOPs
- cuITER solve-only total: `0.011664136` GFLOPs
- cuITER + block-Jacobi setup total: `0.021443336` GFLOPs
- cuITER NR steps total: `6`
- BiCGSTAB linear iterations total: `12`

## 측정 명령

각 case에 대해 다음 형태로 실행했다.

```bash
exp/20260607-2/tools/measure_gpu_flops_ncu.py \
  --output exp/20260607-2/results/flops_same_case_bus6k_nearest/cudss_case6468rte_ncu_flops.csv \
  -- exp/20260607-2/build/hybrid_nr_bench \
    --case case6468rte \
    --solver pure_cudss \
    --max-nr-iters 20 \
    --output exp/20260607-2/results/flops_same_case_bus6k_nearest/pure_cudss_case6468rte_summary.csv \
    --iter-output exp/20260607-2/results/flops_same_case_bus6k_nearest/pure_cudss_case6468rte_iters.csv \
    --timing-output exp/20260607-2/results/flops_same_case_bus6k_nearest/pure_cudss_case6468rte_timing.csv \
    --no-pure-cudss-baseline
```

## 해석

cuDSS의 `cuDSS NCU GFLOPs`는 kernel name에 `cudss`가 포함된 GPU kernel의 dynamic FP instruction FLOPs다. `all GPU NCU GFLOPs`는 같은 pure cuDSS NR run에서 발생한 모든 GPU FP instruction을 합산한 audit 값이다. 주변 cuPF norm/mismatch/update 및 cuBLAS 호출이 포함될 수 있어 cuDSS 내부 연산만 보려면 `cuDSS NCU GFLOPs` 열을 우선 사용한다.

cuITER의 `FLOPs / cuITER NR step`은 `cuiter_total_flops_est / cuiter_nr_steps`다. 각 cuITER NR step은 BiCGSTAB 2회 고정이므로, `FLOPs / BiCGSTAB iter`의 약 2배가 `FLOPs / cuITER NR step`이 된다. 여기에 block-Jacobi apply가 이미 포함되어 있다.
