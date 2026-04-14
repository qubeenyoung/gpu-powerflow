# Result Summary `selected_cases_20260410_gpu3_bg`

## Setup

- Created (UTC): 2026-04-09T16:02:46.970498+00:00
- Cases: 118_ieee, 793_goc, 1354_pegase, 2746wop_k, 4601_goc, 8387_pegase, 9241_pegase
- Warmup: 1
- CPU repeats: 10
- GPU repeats: 10
- Requested GPU index: 3

## Read This First

- 이번 결과는 **재측정 없이도** `analyze`와 `solve`를 나눠서 해석할 수 있다. `summary.csv`와 `aggregates.csv`에 이미 `analyze_sec`, `solve_sec`가 따로 저장돼 있다.
- 다만 현재 benchmark는 repeat마다 새 `NewtonSolver`를 만들고 `analyze()`를 다시 수행하므로, `Elapsed`는 **cold total latency**에 가깝다.
- CUDA의 `analyze`에는 Jacobian/Ybus 업로드, GPU 메모리 할당, cuSPARSE/cuDSS 객체 생성, cuDSS `ANALYSIS`와 초기 `FACTORIZATION`이 포함된다.
- 따라서 `analyze` 구간은 CPU와 CUDA에서 완전히 같은 의미가 아니다.
- 만약 보고 싶은 값이 `analyze 1회 + solve 여러 회`의 **warm solve 전용 성능**이면 그때는 재측정이 필요하다.

## Cold Total

현재 benchmark가 직접 측정한 end-to-end 1회 비용이다.  
즉, `analyze + solve`를 모두 포함한 값이다.

| case | pypower (ms) | cpu naive (ms) | cpu optimized (ms) | cuda edge (ms) | cuda vertex (ms) |
|---|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 11.169 | 1.387 | 0.547 | 13.105 | 13.123 |
| pglib_opf_case793_goc | 21.945 | 9.346 | 3.196 | 17.318 | 17.304 |
| pglib_opf_case1354_pegase | 38.989 | 19.972 | 6.493 | 20.822 | 20.816 |
| pglib_opf_case2746wop_k | 54.658 | 33.506 | 11.655 | 26.778 | 26.775 |
| pglib_opf_case4601_goc | 138.448 | 103.242 | 42.545 | 38.674 | 38.253 |
| pglib_opf_case8387_pegase | 247.531 | 176.660 | 58.166 | 56.551 | 56.608 |
| pglib_opf_case9241_pegase | 318.198 | 244.154 | 75.937 | 61.930 | 62.053 |

## Cold Total Speedup

| case | cpu naive vs pypower | cpu optimized vs pypower | cuda edge vs pypower | cuda vertex vs pypower | cuda edge vs cpu optimized | cuda vertex vs cpu optimized |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 8.05x | 20.41x | 0.85x | 0.85x | 0.04x | 0.04x |
| pglib_opf_case793_goc | 2.35x | 6.87x | 1.27x | 1.27x | 0.18x | 0.18x |
| pglib_opf_case1354_pegase | 1.95x | 6.00x | 1.87x | 1.87x | 0.31x | 0.31x |
| pglib_opf_case2746wop_k | 1.63x | 4.69x | 2.04x | 2.04x | 0.44x | 0.44x |
| pglib_opf_case4601_goc | 1.34x | 3.25x | 3.58x | 3.62x | 1.10x | 1.11x |
| pglib_opf_case8387_pegase | 1.40x | 4.26x | 4.38x | 4.37x | 1.03x | 1.03x |
| pglib_opf_case9241_pegase | 1.30x | 4.19x | 5.14x | 5.13x | 1.23x | 1.22x |

## Analyze Time

여기서 CUDA `analyze`는 setup 성격이 강하다.  
GPU 메모리 준비와 cuDSS 초기화 비용까지 함께 들어 있으므로 CPU `analyze`와 직접 대응시키면 안 된다.

| case | cpu naive analyze (ms) | cpu optimized analyze (ms) | cuda edge analyze (ms) | cuda vertex analyze (ms) |
|---|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.038 | 0.119 | 12.163 | 12.202 |
| pglib_opf_case793_goc | 0.076 | 0.753 | 15.712 | 15.749 |
| pglib_opf_case1354_pegase | 0.105 | 1.312 | 18.710 | 18.660 |
| pglib_opf_case2746wop_k | 0.155 | 3.065 | 24.483 | 24.460 |
| pglib_opf_case4601_goc | 0.282 | 6.321 | 34.592 | 34.277 |
| pglib_opf_case8387_pegase | 0.596 | 12.452 | 50.722 | 50.718 |
| pglib_opf_case9241_pegase | 0.681 | 13.817 | 54.927 | 54.965 |

## Solve Time

이 표가 현재 결과에서 가장 가까운 **steady solve phase** 비교다.  
다만 이것도 여전히 `solve()` 1회 전체이며, solve 내부 초기화 비용은 포함된다.

| case | cpu naive solve (ms) | cpu optimized solve (ms) | cuda edge solve (ms) | cuda vertex solve (ms) |
|---|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 1.348 | 0.427 | 0.941 | 0.920 |
| pglib_opf_case793_goc | 9.270 | 2.442 | 1.605 | 1.554 |
| pglib_opf_case1354_pegase | 19.866 | 5.181 | 2.111 | 2.155 |
| pglib_opf_case2746wop_k | 33.351 | 8.590 | 2.294 | 2.314 |
| pglib_opf_case4601_goc | 102.959 | 36.224 | 4.082 | 3.975 |
| pglib_opf_case8387_pegase | 176.065 | 45.713 | 5.829 | 5.889 |
| pglib_opf_case9241_pegase | 243.473 | 62.120 | 7.004 | 7.087 |

## Solve-Only Speedup

이 표는 setup-heavy한 `analyze`를 떼고, `solve` 구간만 놓고 본 상대 속도다.

| case | cpu optimized vs cpu naive | cuda edge vs cpu naive | cuda vertex vs cpu naive | cuda edge vs cpu optimized | cuda vertex vs cpu optimized |
|---|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 3.16x | 1.43x | 1.47x | 0.45x | 0.46x |
| pglib_opf_case793_goc | 3.80x | 5.78x | 5.96x | 1.52x | 1.57x |
| pglib_opf_case1354_pegase | 3.83x | 9.41x | 9.22x | 2.45x | 2.40x |
| pglib_opf_case2746wop_k | 3.88x | 14.54x | 14.41x | 3.74x | 3.71x |
| pglib_opf_case4601_goc | 2.84x | 25.23x | 25.90x | 8.87x | 9.11x |
| pglib_opf_case8387_pegase | 3.85x | 30.21x | 29.90x | 7.84x | 7.76x |
| pglib_opf_case9241_pegase | 3.92x | 34.76x | 34.35x | 8.87x | 8.76x |

## Operator Summary: cpp optimized

아래 표는 `run_01-run_00`, `run_02-run_01`, ..., `run_09-run_08`의 delta를 평균낸 값이다.  
즉, `warmup`이 섞인 `run_00`의 누적 로그를 피하고, 뒤 9개 측정 반복에서 **solve 1회당 연산자별 총 시간**을 복원한 요약이다.

| case | analyze.total (ms) | solve.init (ms) | computeMismatch sum (ms) | updateJacobian sum (ms) | solveLinearSystem sum (ms) | updateVoltage sum (ms) | downloadV (ms) | solve.total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.096 | 0.009 | 0.005 | 0.019 | 0.104 | 0.017 | 0.000 | 0.404 |
| pglib_opf_case793_goc | 0.728 | 0.053 | 0.051 | 0.141 | 1.730 | 0.120 | 0.000 | 2.422 |
| pglib_opf_case1354_pegase | 1.287 | 0.104 | 0.123 | 0.350 | 3.947 | 0.268 | 0.000 | 5.165 |
| pglib_opf_case2746wop_k | 3.029 | 0.179 | 0.173 | 0.507 | 7.017 | 0.406 | 0.001 | 8.562 |
| pglib_opf_case4601_goc | 6.291 | 0.322 | 0.372 | 1.017 | 33.497 | 0.856 | 0.002 | 36.178 |
| pglib_opf_case8387_pegase | 12.556 | 1.055 | 0.909 | 3.231 | 38.164 | 2.268 | 0.004 | 45.766 |
| pglib_opf_case9241_pegase | 13.872 | 0.744 | 1.131 | 4.339 | 51.908 | 3.819 | 0.005 | 62.110 |

## Operator Summary: cuda edge

같은 방식으로 `cpp_cuda_edge`의 반복 delta만 평균낸 값이다.  
즉, 처음 CUDA context 초기화와 warmup을 최대한 배제한 **steady repeat 기준 operator summary**다.

| case | analyze.total (ms) | solve.init (ms) | computeMismatch sum (ms) | updateJacobian sum (ms) | solveLinearSystem sum (ms) | updateVoltage sum (ms) | downloadV (ms) | solve.total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 12.144 | 0.031 | 0.179 | 0.056 | 0.504 | 0.065 | 0.008 | 0.937 |
| pglib_opf_case793_goc | 15.665 | 0.058 | 0.244 | 0.082 | 1.001 | 0.106 | 0.015 | 1.599 |
| pglib_opf_case1354_pegase | 18.658 | 0.068 | 0.240 | 0.082 | 1.497 | 0.107 | 0.020 | 2.108 |
| pglib_opf_case2746wop_k | 24.361 | 0.108 | 0.207 | 0.066 | 1.709 | 0.085 | 0.031 | 2.285 |
| pglib_opf_case4601_goc | 34.408 | 0.156 | 0.252 | 0.088 | 3.327 | 0.107 | 0.044 | 4.073 |
| pglib_opf_case8387_pegase | 50.430 | 0.247 | 0.356 | 0.120 | 4.782 | 0.132 | 0.071 | 5.819 |
| pglib_opf_case9241_pegase | 54.608 | 0.268 | 0.407 | 0.154 | 5.801 | 0.157 | 0.078 | 6.996 |

## Interpretation

- 작은 케이스에서는 CUDA의 `solve` 자체는 빠르더라도 `analyze` setup 비용이 커서 cold total은 불리하다.
- 큰 케이스에서는 `solve` 이득이 `analyze` 고정비를 넘기기 시작해서 cold total도 CPU optimized를 추월한다.
- 현재 결과만으로도 “CUDA solve는 빠르지만 analyze가 무겁다”는 결론은 충분히 읽을 수 있다.
- 다만 “analyze 1회 후 solve만 여러 번 반복”하는 운영 시나리오를 대표하는 숫자는 아니다.

## Correctness Snapshot

| case | pypower success | cpu naive success | cpu optimized success | cuda edge success | cuda vertex success | pypower iter | cpu naive iter | cpu optimized iter | cuda edge iter | cuda vertex iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | True | True | True | True | True | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| pglib_opf_case793_goc | True | True | True | True | True | 4.0 | 4.0 | 4.0 | 5.0 | 4.8 |
| pglib_opf_case1354_pegase | True | True | True | True | True | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| pglib_opf_case2746wop_k | True | True | True | True | True | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| pglib_opf_case4601_goc | True | True | True | True | True | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| pglib_opf_case8387_pegase | True | True | True | True | True | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| pglib_opf_case9241_pegase | True | True | True | True | True | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
