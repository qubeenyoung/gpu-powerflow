# Bus-Scale 4K-6K Case Availability Report

작성일: 2026-05-12  
대상 실험: `exp/20260607-2`

## 결론

요청 기준은 Jacobian/linear-system 차원 `n`이 아니라 전력계통 bus 수 기준
`4,000 <= buses <= 6,000`이다.

현재 repository의 MATPOWER case 목록, `exp/20260607-2/raw/cupf_jf_dumps`,
그리고 `exp/20260607-2/results/bicgstab_iter2_bs8_all78_summary.csv`를 확인한
결과, **bus 수 4K~6K 사이 case는 0개**다.

따라서 현재 보유한 dump/result만으로는 “4K~6K bus case에서 cuDSS와 cuITER를
동일 case로 측정한 FLOP 보고서”를 만들 수 없다.

## 확인한 소스

| 소스 | 4K~6K bus case 수 |
|---|---:|
| `datasets/matpower8.1/data/*.m` | 0 |
| `exp/20260607-2/raw/cupf_jf_dumps/linear_system_dump_summary.csv` | 0 |
| `exp/20260607-2/results/bicgstab_iter2_bs8_all78_summary.csv` | 0 |

## 근접 case

현재 결과셋에서 2.5K~7K bus 범위의 case는 다음뿐이다.

| case | buses | linear n | nnz | 비고 |
|---|---:|---:|---:|---|
| `case2736sp` | 2736 | 5237 | 33715 | 4K 미만 |
| `case2737sop` | 2737 | 5280 | 34242 | 4K 미만 |
| `case2746wop` | 2746 | 5141 | 32445 | 4K 미만 |
| `case2746wp` | 2746 | 5127 | 32129 | 4K 미만 |
| `case2848rte` | 2848 | 5324 | 36002 | 4K 미만 |
| `case2868rte` | 2868 | 5321 | 36015 | 4K 미만 |
| `case2869pegase` | 2869 | 5227 | 36591 | 4K 미만 |
| `case3012wp` | 3012 | 5725 | 36263 | 4K 미만 |
| `case3120sp` | 3120 | 5991 | 38329 | 4K 미만 |
| `case3375wp` | 3374 | 6355 | 40717 | 4K 미만 |
| `case6468rte` | 6468 | 12643 | 87845 | 6K 초과 |
| `case6470rte` | 6470 | 12485 | 86405 | 6K 초과 |
| `case6495rte` | 6495 | 12495 | 86287 | 6K 초과 |
| `case6515rte` | 6515 | 12535 | 86561 | 6K 초과 |

## 이전 n=4K-6K 보고서에 대한 정정

`flops_same_case_n4k_6k/same_case_n4k_6k_flops_report.md`는 bus 수가 아니라
linear-system 차원 `n` 기준으로 작성된 보고서다. 요청 기준이 bus 수라면 해당
보고서는 본 비교의 최종 보고서로 쓰면 안 된다.

해당 보고서의 선택 case:

| case | buses | linear n |
|---|---:|---:|
| `case2383wp` | 2383 | 4438 |
| `case2869pegase` | 2869 | 5227 |
| `case3120sp` | 3120 | 5991 |

즉 모두 bus 기준 4K~6K 범위 밖이다.

## 다음 선택지

1. strict bus 4K~6K를 유지한다면 새 case/dump가 필요하다.
2. 가장 가까운 상위 bus 규모를 허용한다면 `case6468rte`, `case6470rte`,
   `case6495rte`를 같은-case 비교 대상으로 삼을 수 있다.
3. 가장 가까운 하위 bus 규모를 허용한다면 `case3012wp`, `case3120sp`,
   `case3375wp`를 사용할 수 있지만, 이는 3K급 case다.

## BiCGSTAB/FLOP 정의 확인

기존 `bicgstab_iter2_bs8_all78` 결과에서 cuITER가 실제 호출된 NR step은
모두 BiCGSTAB 2회 고정이다. active cuITER row 227개의 `linear_iters` unique
값은 `[2]`다.

`FLOPs / cuITER NR step`은 한 Newton-Raphson 반복에서 cuITER가 수행한 평균
FLOPs를 뜻한다. 이 값은 BiCGSTAB과 block-Jacobi를 분리한 값이 아니라,
다음을 합산한 값이다.

- BiCGSTAB CSR SpMV
- BiCGSTAB dot/reduction
- BiCGSTAB vector update
- residual norm check
- block-Jacobi preconditioner apply

단, 기본 solve-only 보고에서는 block-Jacobi setup FLOPs를 제외한다. setup을
포함한 값은 별도 `with_bj_setup` 결과로 보고해야 한다.
