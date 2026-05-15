# cuDSS / cuITER FLOP Measurement Report

작성일: 2026-05-12  
대상 실험: `exp/20260607-2`

## 목적

cuPF의 cuDSS 직접 선형해법과 `exp/20260607-2`의 cuITER 계열
`BiCGSTAB + METIS block-Jacobi` 경로에 대해 FLOP 관점의 비교 근거를
남긴다.

특히 cuITER는 Newton-Raphson 반복 내부에서 발생한 FLOPs를

- NR 반복별 FLOPs
- case 내부 누적 FLOPs
- BiCGSTAB linear iteration당 FLOPs

로 나누어 보고한다.

## 산출물

| 파일 | 내용 |
|---|---|
| `bicgstab_iter2_bs8_all78_flops_iters.csv` | NR 반복별 cuITER FLOP 추정치와 case 내부 누적치 |
| `bicgstab_iter2_bs8_all78_flops_summary.csv` | case별 cuITER 누적 FLOP 요약 |
| `bicgstab_iter2_bs8_all78_flops.md` | cuITER solve-only FLOP 요약 보고서 |
| `bicgstab_iter2_bs8_all78_flops_with_bj_setup_summary.csv` | block-Jacobi setup 추정 포함 요약 |
| `cudss_case1197_j1_ncu_flops_summary.csv` | cuDSS `case1197/J1` Nsight Compute FLOP 측정 요약 |
| `cudss_case1197_j1_ncu_flops.csv` | cuDSS kernel별 Nsight Compute FLOP 측정 |

## cuITER FLOP 산정 범위

기본 보고서는 BiCGSTAB solve work와 block-Jacobi apply work를 포함한다.
block-Jacobi dense LU/inverse setup은 기본값에서는 제외하고,
`--count-bj-setup` 실행 결과에서 별도로 포함했다.

산식은 다음과 같다.

| 항목 | 산식 |
|---|---:|
| CSR SpMV | `2 * nnz` FLOPs |
| BiCGSTAB SpMV | linear iteration당 SpMV 2회, 즉 `4 * nnz` |
| block-Jacobi inverse GEMV apply | block별 `2*m^2 - m` |
| BiCGSTAB block-Jacobi apply | linear iteration당 apply 2회 |
| dot/reduction | linear iteration당 `8*n - 4` |
| vector update | 첫 BiCGSTAB iteration `8*n`, 이후 `12*n` |
| norm check | cuITER solve 1회당 `4*n` |
| setup 포함 옵션 | dense batched LU 및 inverse를 `num_blocks * max_block_unknowns^3` 기반 추정 |

여기서 `n`은 선형계 차원, `nnz`는 Jacobian CSR nonzero 수, `m`은
block-Jacobi 각 diagonal block의 unknown 수다.

## cuITER 전체 요약

입력:

- summary: `bicgstab_iter2_bs8_all78_summary.csv`
- iterations: `bicgstab_iter2_bs8_all78_iters.csv`
- 설정: `BiCGSTAB`, `block_size=8`, `bicgstab_iters=2`, `inverse_gemv`

| 항목 | solve-only | block-Jacobi setup 포함 |
|---|---:|---:|
| 전체 case 수 | 78 | 78 |
| cuITER가 실제 호출된 case 수 | 68 | 68 |
| cuITER NR step 수 | 227 | 227 |
| BiCGSTAB linear iteration 수 | 454 | 454 |
| 총 FLOPs | `3.52827956e8` | `6.51909124e8` |
| 총 GFLOPs | `0.352827956` | `0.651909124` |
| cuITER NR step당 평균 FLOPs | `1.554308e6` | `2.871846e6` |
| BiCGSTAB linear iteration당 평균 solve FLOPs | `7.771541e5` | `7.771541e5` |

setup 포함 열에서도 BiCGSTAB linear iteration당 값은 solve-only 기준으로
해석해야 한다. setup FLOPs는 linear iteration에 자연스럽게 귀속되지 않는
분석/전처리 성격의 비용이기 때문이다.

## cuITER 상위 case

solve-only 누적 FLOPs 기준 상위 10개 case:

| case | n | nnz | cuITER NR steps | BiCGSTAB iters | total GFLOPs | FLOPs / cuITER NR step | FLOPs / BiCGSTAB iter |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 156255 | 1052085 | 5 | 10 | `0.119689700` | `2.393794e7` | `1.196897e7` |
| case_ACTIVSg70k | 134104 | 900558 | 5 | 10 | `0.102676600` | `2.053532e7` | `1.026766e7` |
| case13659pegase | 23225 | 174703 | 6 | 12 | `0.022109736` | `3.684956e6` | `1.842478e6` |
| case_ACTIVSg25k | 47246 | 318672 | 2 | 4 | `0.014491168` | `7.245584e6` | `3.622792e6` |
| case_ACTIVSg10k | 18544 | 125174 | 5 | 10 | `0.014229480` | `2.845896e6` | `1.422948e6` |
| case9241pegase | 17036 | 129412 | 5 | 10 | `0.013647800` | `2.729560e6` | `1.364780e6` |
| case2869pegase | 5227 | 36591 | 8 | 16 | `0.006496928` | `8.121160e5` | `4.060580e5` |
| case3120sp | 5991 | 38329 | 7 | 14 | `0.006318508` | `9.026440e5` | `4.513220e5` |
| case2746wp | 5127 | 32129 | 7 | 14 | `0.005370764` | `7.672520e5` | `3.836260e5` |
| case2737sop | 5280 | 34242 | 6 | 12 | `0.004786032` | `7.976720e5` | `3.988360e5` |

## NR 반복별 보고 방식

`bicgstab_iter2_bs8_all78_flops_iters.csv`에서 핵심 컬럼은 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `case_name` | case 이름 |
| `nr_iter` | Newton-Raphson 반복 번호 |
| `solver_used` | 해당 반복에서 실제 사용된 solver |
| `linear_iters` | 해당 반복에서 수행된 BiCGSTAB iteration 수 |
| `cuiter_total_flops_est` | 해당 NR 반복의 cuITER FLOPs |
| `cuiter_cumulative_flops_est` | 동일 case 내부 누적 cuITER FLOPs |
| `cuiter_flops_per_bicgstab_iter_est` | 해당 반복의 BiCGSTAB iteration당 FLOPs |
| `fallback_used` | cuDSS fallback이 같이 사용되었는지 여부 |
| `stop_reason` | linear solve 또는 fallback 결정 이유 |

예를 들어 `case_SyntheticUSA`는 각 cuITER NR step에서
`2.393794e7` FLOPs가 추정되며, 5번의 cuITER 시도 후 누적
`1.196897e8` FLOPs가 된다.

## cuDSS FLOP 측정

cuDSS는 sparse LU 내부 알고리즘 FLOP count를 benchmark API로 직접 노출하지
않는다. 따라서 `measure_gpu_flops_ncu.py`는 Nsight Compute의 SASS floating
point instruction counter를 사용해 동적 GPU FLOPs를 측정한다.

측정 metric:

- `sm__sass_thread_inst_executed_op_dadd_pred_on`
- `sm__sass_thread_inst_executed_op_dfma_pred_on`
- `sm__sass_thread_inst_executed_op_dmul_pred_on`
- `sm__sass_thread_inst_executed_op_fadd_pred_on`
- `sm__sass_thread_inst_executed_op_ffma_pred_on`
- `sm__sass_thread_inst_executed_op_fmul_pred_on`

FMA는 2 FLOPs로 계산한다.

측정 smoke:

```bash
exp/20260607-2/tools/measure_gpu_flops_ncu.py \
  --output exp/20260607-2/results/cudss_case1197_j1_ncu_flops.csv \
  -- exp/20260607-2/build/cudss_jf_bench \
    --case case1197 \
    --matrix exp/20260607-2/raw/cupf_jf_dumps/case1197/J1.txt \
    --rhs exp/20260607-2/raw/cupf_jf_dumps/case1197/F1.txt \
    --precision fp64 \
    --repeats 1 \
    --csv
```

결과:

| class | kernel count | FP64 FLOPs | FP32 FLOPs | total FLOPs | total GFLOPs |
|---|---:|---:|---:|---:|---:|
| all | 93 | `4.456094e6` | `2.75558e5` | `4.731652e6` | `0.004731652` |
| cudss | 87 | `4.456094e6` | `2.75558e5` | `4.731652e6` | `0.004731652` |
| other | 6 | `0` | `0` | `0` | `0` |

주의: 이 cuDSS 값은 `case1197/J1` 단일 dumped linear system에서 발생한
GPU kernel floating-point instruction FLOPs다. cuDSS multi-threaded
host-side work, symbolic/numeric scheduling의 CPU 연산, 파일 I/O, host-device
copy는 포함하지 않는다.

## 해석

cuITER FLOP 추정치는 알고리즘 산식 기반이라 NR 반복별 누적 보고에 적합하다.
BiCGSTAB가 각 NR step마다 고정 2회 수행되는 설정에서는 `n`, `nnz`, block
구조가 FLOP을 거의 결정한다.

cuDSS FLOP은 내부 sparse direct solver 특성상 소스 수준 산식으로 안정적으로
복원하기 어렵다. 현재 보고서는 NCU dynamic instruction count로 GPU에서 실제
실행된 floating-point instruction을 측정한다. 따라서 cuDSS와 cuITER를 같은
표에 놓을 때는 다음처럼 구분하는 것이 안전하다.

- cuITER: algorithm-estimated FLOPs, NR iteration/case cumulative 가능
- cuDSS: NCU-measured GPU dynamic FLOPs, kernel execution 기준

최종 논문/발표용 비교에서는 cuDSS도 대표 case 전체에 대해 같은 NCU 래퍼를
반복 실행하고, cuITER도 같은 NCU 래퍼로 동적 instruction FLOPs를 보조 측정하면
산식 기반 추정치와 실측 instruction count의 차이를 함께 제시할 수 있다.
