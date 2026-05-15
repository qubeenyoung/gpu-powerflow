# Same-Case cuDSS / cuITER FLOP Report (n=4K-6K)

작성일: 2026-05-12  
대상 실험: `exp/20260607-2`

## 결론

이번 보고서는 같은 case 집합에서 cuDSS와 cuITER를 다시 맞췄다. 4K~6K급은 power-flow bus 수가 아니라 선형계 차원 `n` 기준으로 골랐다. FLOP 산정에 직접 들어가는 크기가 `n`과 `nnz`이기 때문이다.

선택한 case는 `case2383wp(n=4438)`, `case2869pegase(n=5227)`, `case3120sp(n=5991)`이다.

BiCGSTAB은 이 결과셋에서 **2 linear iterations 고정**이다. `bicgstab_iter2_bs8_all78_flops_iters.csv`의 active cuITER row 227개를 확인하면 `linear_iters`의 unique 값은 `[2]`뿐이다.

`FLOPs / cuITER NR step`은 NR 반복 한 번에서 cuITER가 수행한 평균 FLOPs가 맞다. 여기에는 BiCGSTAB의 SpMV, dot/reduction, vector update, norm check와 block-Jacobi apply가 합산되어 있다. 기본 solve-only 값은 block-Jacobi setup을 제외하고, setup 포함 값은 별도 열에 둔다.

## Same-Case Summary

| case | buses | n | nnz | pure cuDSS NR iters | cuDSS calls | cuDSS NCU GFLOPs | all GPU NCU GFLOPs | cuITER NR steps | BiCGSTAB iters | cuITER solve GFLOPs | cuITER + BJ setup GFLOPs | FLOPs / cuITER NR step | FLOPs / BiCGSTAB iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 2383 | 4438 | 27874 | 6 | 6 | `0.032453866` | `0.086702809` | 6 | 12 | `0.003983616` | `0.007448832` | `6.639360e+05` | `3.319680e+05` |
| case2869pegase | 2869 | 5227 | 36591 | 6 | 6 | `0.036192306` | `0.093954695` | 8 | 16 | `0.006496928` | `0.011936416` | `8.121160e+05` | `4.060580e+05` |
| case3120sp | 3120 | 5991 | 38329 | 6 | 6 | `0.041494481` | `0.100705965` | 7 | 14 | `0.006318508` | `0.011759020` | `9.026440e+05` | `4.513220e+05` |

합계:

- cuDSS NCU `cudss` class total: `0.110140653` GFLOPs
- cuDSS NCU all GPU FP total: `0.281363469` GFLOPs
- cuITER solve-only total: `0.016799052` GFLOPs
- cuITER + block-Jacobi setup total: `0.031144268` GFLOPs

## cuDSS 측정 방식

cuDSS는 sparse LU 내부 FLOP count를 API로 제공하지 않으므로 Nsight Compute SASS floating-point instruction counter로 측정했다. 실행은 같은 case에 대해 `hybrid_nr_bench --solver pure_cudss` 전체 NR loop를 프로파일했다.

예시 명령:

```bash
exp/20260607-2/tools/measure_gpu_flops_ncu.py \
  --output exp/20260607-2/results/flops_same_case_n4k_6k/cudss_case2869pegase_ncu_flops.csv \
  -- exp/20260607-2/build/hybrid_nr_bench \
    --case case2869pegase \
    --solver pure_cudss \
    --max-nr-iters 20 \
    --output exp/20260607-2/results/flops_same_case_n4k_6k/pure_cudss_case2869pegase_summary.csv \
    --iter-output exp/20260607-2/results/flops_same_case_n4k_6k/pure_cudss_case2869pegase_iters.csv \
    --timing-output exp/20260607-2/results/flops_same_case_n4k_6k/pure_cudss_case2869pegase_timing.csv \
    --no-pure-cudss-baseline
```

NCU FLOP 계산은 `add + mul + 2*fma`이며, `cudss` class는 kernel name에 `cudss`가 들어간 커널만 합산한 값이다. `all GPU`는 같은 pure cuDSS NR run에서 발생한 모든 GPU floating-point instruction을 합산한 audit 값이다. 여기에는 cuPF mismatch/norm/update 주변 커널과 cuBLAS 호출이 섞일 수 있다.

## cuITER 산정 방식

cuITER 값은 같은 case의 `bicgstab_iter2_bs8_all78` hybrid run에서 나온 NR iteration CSV를 바탕으로 산식 계산했다. 기본 solve-only 산식은 다음을 합산한다.

| 항목 | 산식 |
|---|---:|
| CSR SpMV | `2 * nnz`; BiCGSTAB 1회당 SpMV 2번이라 `4 * nnz` |
| block-Jacobi inverse GEMV apply | block별 `2*m^2 - m`; BiCGSTAB 1회당 apply 2번 |
| dot/reduction | BiCGSTAB 1회당 `8*n - 4` |
| vector update | 첫 BiCGSTAB iteration `8*n`, 이후 `12*n` |
| norm check | cuITER solve 1회당 `4*n` |

따라서 `FLOPs / cuITER NR step`은 `cuiter_total_flops_est / cuiter_nr_steps`이고, BiCGSTAB과 block-Jacobi를 따로 떼지 않은 합산 값이다.

## 주의

same-case 비교이지만 pure cuDSS와 hybrid cuITER의 nonlinear trajectory는 같지 않을 수 있다. cuDSS pure run은 모든 NR step을 cuDSS로 풀고, cuITER hybrid run은 bootstrap/fallback/polish cuDSS와 중간 BiCGSTAB step이 섞인다. 이 보고서의 목적은 같은 network case 크기에서 cuDSS 측정 FLOPs와 cuITER 중간 iterative solve FLOPs를 case 단위로 정렬하는 것이다.
