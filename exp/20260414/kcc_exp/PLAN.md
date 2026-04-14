# KCC 실험 계획

작성일: 2026-04-13

## 목표

이 실험은 KCC 제출용 성능 결과와 ablation 표/그림의 원천 데이터를 만드는 것을 목표로 한다.

1. PYPOWER의 연산자별 시간 소요를 측정한다.
   - 병목 분석용 pie chart의 재료로 사용한다.
   - 대상 지표: `runpf.*`, `newtonpf.init_index`, `newtonpf.mismatch`, `newtonpf.jacobian`, `newtonpf.solve`, `newtonpf.update_voltage`.
2. 전체 구현 결과를 end-to-end로 비교한다.
   - 비교 체인: `pypower -> cpp_naive -> cpp optimized -> cuda_edge`.
   - 현재 벤치마크 프로파일명 기준으로는 `pypower -> cpp_naive -> cpp -> cuda_edge`이다.
3. `cuda_edge` 기준 ablation을 연산자별로 추적한다.
   - full: `cuda_edge`
   - w/o cuDSS: CUDA edge Jacobian은 유지하고 선형 풀이만 CPU SuperLU로 대체하는 신규 benchmark-only profile
   - w/o Jacobian: cuDSS는 유지하고 Jacobian update만 CPU PyPower-like naive Jacobian으로 대체하는 신규 benchmark-only profile
   - w/o mixed precision: `cuda_fp64_edge`
4. Jacobian update strategy만 비교한다.
   - `cuda_edge` vs `cuda_vertex`
   - 필요하면 동일한 비교를 `cuda_fp64_edge` vs `cuda_fp64_vertex`로 보조 확인한다.

## 대상 데이터

데이터셋 루트:

```text
/workspace/datasets/cuPF_benchmark_dumps
```

대상 케이스:

```text
case30_ieee
case118_ieee
case793_goc
case1354_pegase
case2746wop_k
case4601_goc
case8387_pegase
case9241_pegase
```

공통 실행 조건:

- 모든 성능 측정은 `--warmup 1 --repeats 10`으로 실행한다.
  - 단, `smoke_hybrid_ablation`은 correctness 확인용이며 성능 측정에 사용하지 않는다.
- 허용 오차: `--tolerance 1e-8`
- 최대 반복: `--max-iter 50`
- CUDA 실행 GPU는 실험 시 명시한다. 예: `CUDA_VISIBLE_DEVICES=1`
- 결과 루트: `/workspace/exp/20260414/kcc_exp/results`
- 최종 `cuda_edge` 계열 측정은 cuDSS MT mode를 켜고 thread/ND 옵션을 AUTO로 둔다.
  - `--cudss-enable-mt`
  - `--cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0`
  - `--cudss-host-nthreads AUTO`
  - `--cudss-nd-nlevels AUTO`

## 현재 바로 가능한 실험

### 1. PYPOWER operator profiling

목적: PYPOWER baseline 내부 병목을 단계별로 분해한다.

```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --results-root /workspace/exp/20260414/kcc_exp/results \
  --run-name pypower_operator_profile \
  --mode operators \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --profiles pypower \
  --warmup 1 \
  --repeats 10
```

필요 산출물:

- `summary_operators.csv`
- `raw/operators/pypower/<case>/run_*.json`
- 후처리 CSV: `pypower_operator_pie.csv`

후처리 기준:

- pie chart 1: `runpf.load_case_data`, `runpf.bus_indexing`, `runpf.build_Ybus`, `runpf.build_Sbus`, `runpf.newtonpf`
- pie chart 2: `newtonpf.init_index`, `newtonpf.mismatch`, `newtonpf.jacobian`, `newtonpf.solve`, `newtonpf.update_voltage`
- 논문 본문에는 큰 케이스 평균 또는 대표 케이스(`case9241_pegase`)를 사용한다.

### 2. End-to-end 구현 결과 비교

목적: 구현 발전 단계별 전체 실행시간을 비교한다.

```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --results-root /workspace/exp/20260414/kcc_exp/results \
  --run-name end2end_main_chain \
  --mode end2end \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --profiles pypower cpp_naive cpp cuda_edge \
  --warmup 1 \
  --repeats 10 \
  --cudss-enable-mt \
  --cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0 \
  --cudss-host-nthreads AUTO \
  --cudss-nd-nlevels AUTO
```

프로파일 의미:

| 표시명 | 벤치마크 프로파일 | 의미 |
| --- | --- | --- |
| PYPOWER | `pypower` | Python PYPOWER baseline |
| C++ naive | `cpp_naive` | C++ PyPower-like Jacobian + SuperLU |
| C++ optimized | `cpp` | C++ optimized CPU path: edge-based Jacobian maps + KLU |
| CUDA edge | `cuda_edge` | CUDA mixed precision edge Jacobian + cuDSS32, cuDSS MT enabled, host threads AUTO, ND levels AUTO |

필요 산출물:

- `summary_end2end.csv`
- `aggregates_end2end.csv`
- 후처리 CSV: `end2end_main_chain.csv`

후처리 기준:

- 케이스별 평균 elapsed time
- `pypower` 대비 speedup
- `cpp` 대비 speedup
- 성공 여부, 반복 횟수, final mismatch

### 3. CUDA edge ablation

목적: `cuda_edge` full 구현에서 핵심 구성 요소를 하나씩 제거했을 때의 연산자별 영향을 확인한다.

구현 완료 항목:

- full: `cuda_edge`
- w/o mixed precision: `cuda_fp64_edge`
- w/o cuDSS: `cuda_wo_cudss`
- w/o Jacobian: `cuda_wo_jacobian`

목표 실행 명령:

```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --results-root /workspace/exp/20260414/kcc_exp/results \
  --run-name cuda_edge_ablation_operators \
  --mode operators \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --profiles cuda_edge cuda_wo_cudss cuda_wo_jacobian cuda_fp64_edge \
  --warmup 1 \
  --repeats 10 \
  --cudss-enable-mt \
  --cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0 \
  --cudss-host-nthreads AUTO \
  --cudss-nd-nlevels AUTO
```

필요 산출물:

- `summary_operators.csv`
- `raw/operators/<profile>/<case>/run_*.json`
- 후처리 CSV: `cuda_edge_ablation_operator_breakdown.csv`

후처리 기준:

- top-level: `elapsed_sec`, `analyze_sec`, `solve_sec`
- NR breakdown: `NR.solve.upload`, `NR.iteration.mismatch`, `NR.iteration.jacobian`, `NR.iteration.linear_solve`, `NR.iteration.voltage_update`, `NR.solve.download`
- cuDSS breakdown: `CUDA.solve.rhsPrepare`, `CUDA.solve.factorization32`, `CUDA.solve.refactorization32`, `CUDA.solve.solve32`
- FP64 비교에서는 `CUDA.solve.*64` 지표를 동일 카테고리로 매핑한다.

주의:

- `operators` 모드는 타이머/NVTX가 켜져 clean end-to-end와 완전히 같지는 않다.
- ablation 표에는 operator breakdown을 우선 사용하고, 필요하면 `--mode both`로 별도 clean end-to-end 값을 같이 확보한다.

### 4. Edge vs vertex Jacobian update

목적: Jacobian update strategy만 비교한다.

```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --results-root /workspace/exp/20260414/kcc_exp/results \
  --run-name jacobian_edge_vs_vertex \
  --mode operators \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --profiles cuda_edge cuda_vertex \
  --warmup 1 \
  --repeats 10 \
  --cudss-enable-mt \
  --cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0 \
  --cudss-host-nthreads AUTO \
  --cudss-nd-nlevels AUTO
```

필요 산출물:

- `summary_operators.csv`
- 후처리 CSV: `jacobian_edge_vertex.csv`

후처리 기준:

- `NR.iteration.jacobian.total_sec`
- `NR.iteration.jacobian.count`
- per-call time: `total_sec / count`
- `NR.analyze.jacobian_builder.total_sec`
- full solve time은 보조 지표로만 사용한다.

## 추가 구현 현황

추가 구현은 기존 solver core 동작을 바꾸지 않고, benchmark-only hybrid profile을 옆에 붙이는 방식으로 완료했다.

### 원칙

- 기존 프로파일의 동작을 바꾸지 않는다.
  - `pypower`, `cpp_naive`, `cpp`, `cuda_edge`, `cuda_vertex`, `cuda_fp64_edge`, `cuda_fp64_vertex`는 그대로 유지한다.
- `NewtonSolver`, `PlanBuilder`, production op 조립 로직은 건드리지 않는다.
- 새 실험 경로는 `cupf_case_benchmark` 내부 또는 benchmark-only source로 추가한다.
- 결과 CSV에는 새 프로파일이 다른 프로파일과 같은 schema로 기록되도록 한다.

### 신규 프로파일 이름

| user-facing profile | C++ internal profile | 의미 |
| --- | --- | --- |
| `cuda_wo_jacobian` | `cuda_mixed_edge_cpu_naive_jacobian` | CUDA edge full에서 Jacobian update만 CPU PyPower-like naive로 대체 |
| `cuda_wo_cudss` | `cuda_mixed_edge_cpu_superlu` | CUDA edge full에서 linear solve만 CPU SuperLU로 대체 |
| optional, 미구현 | `cuda_mixed_edge_cpu_naive_jacobian_superlu` | Jacobian과 linear solve를 모두 CPU naive로 대체. sanity/debug 전용 |

### 수정 파일

1. `/workspace/cuPF/benchmarks/run_benchmarks.py`
   - `PROFILE_SPECS`에 `cuda_wo_jacobian`, `cuda_wo_cudss` 추가.
   - `runner="cupf"`로 두고 `cupf_profile`은 C++ internal profile로 매핑한다.
   - `backend="cuda"`, `compute="mixed"`.
   - `jacobian` 값은 각각 `cpu_naive_pypower_like`, `edge_based`로 기록한다.
   - 필요하면 `implementation`에 `cuda_wo_jacobian`, `cuda_wo_cudss`를 기록해 plot label로 사용한다.

2. `/workspace/cuPF/benchmarks/cpp/cupf_case_benchmark.cpp`
   - `ProfileConfig`에 hybrid profile 플래그를 추가한다.
     - `use_cpu_naive_jacobian`
     - `use_cpu_superlu`
   - `parse_profile()`에 `cuda_mixed_edge_cpu_naive_jacobian`, `cuda_mixed_edge_cpu_superlu` 추가.
   - `run_once()`에서 hybrid profile이면 새 benchmark-only runner로 분기한다.
   - 기존 `run_core_once()`와 `run_reference_pypowerlike_once()`는 수정하지 않는다.
   - CUDA mixed storage와 CPU scratch storage를 함께 쓰는 hybrid runner를 benchmark 파일 내부에 추가한다.

3. `/workspace/cuPF/benchmarks/CMakeLists.txt`
   - 신규 파일을 만들지 않았으므로 CMake 수정은 필요 없다.
   - 기존 benchmark target이 이미 SuperLU 활성화 시 아래 reference CPU source를 포함한다.
     - `cpu_naive_jacobian_f64.cpp`
     - `cpu_superlu_solve.cpp`
   - 새 hybrid runner가 SuperLU를 쓰므로 `CUPF_SUPERLU_ENABLED`가 꺼져 있으면 새 profile은 에러 처리한다.

### w/o Jacobian 구현 상세

목표 파이프라인:

```text
CUDA mismatch -> CPU naive Jacobian -> cuDSS32 -> CUDA mixed voltage update
```

필요 bridge:

1. CUDA 전압 다운로드
   - `CudaMixedStorage::d_V_re`, `d_V_im`을 host로 내려 CPU host state `V`를 만든다.
2. CPU naive Jacobian 계산
   - 기존 `CpuNaiveJacobianOpF64`가 `CpuFp64Storage` 전용이므로, benchmark runner 내부에 `CpuFp64Storage` scratch를 함께 둔다.
   - CPU scratch storage는 analyze/upload 시점에 Ybus, Sbus, V0, pv/pq를 준비한다.
   - 매 반복마다 CPU scratch의 `V`를 CUDA에서 내려온 현재 전압으로 갱신한다.
   - `CpuNaiveJacobianOpF64::run()`으로 PyPower-like Jacobian을 만든다.
3. CPU CSC Jacobian -> CUDA CSR Jacobian 값 변환
   - CPU naive Jacobian은 Eigen CSC이고, cuDSS 입력은 CUDA CSR `d_J_values`이다.
   - `AnalyzeContext.J`의 CSR row/col 구조를 기준으로 host `h_J_values`를 채운다.
   - mixed profile이므로 `double -> float`으로 cast한 뒤 `CudaMixedStorage::d_J_values`에 업로드한다.
   - 변환 비용은 w/o Jacobian ablation에 포함한다.

새 timing label:

- `ABLATION.cuda_to_cpu_voltage`
- `ABLATION.cpu_naive_jacobian`
- 기존 CPU naive 내부 타이머: `CPU.naive.jacobian.dS_dVm`, `CPU.naive.jacobian.dS_dVa`, `CPU.naive.jacobian.assemble`
- `ABLATION.cpu_jacobian_to_cuda_csr`

### w/o cuDSS 구현 상세

목표 파이프라인:

```text
CUDA mismatch -> CUDA edge FP32 Jacobian -> CPU SuperLU -> CUDA mixed voltage update
```

필요 bridge:

1. CUDA F 다운로드
   - `CudaMixedStorage::d_F`를 host double vector로 다운로드한다.
2. CUDA CSR Jacobian 다운로드
   - `CudaMixedStorage::d_J_values`를 host float vector로 다운로드한다.
   - `AnalyzeContext.J`의 CSR row/col 구조와 함께 CPU Eigen CSC `J`를 재구성한다.
   - SuperLU는 double을 쓰므로 `float -> double`로 cast한다.
3. CPU SuperLU solve
   - 기존 `CpuLinearSolveSuperLU`가 `CpuFp64Storage` 전용이므로, CPU scratch storage의 `J`, `F`, `dx`를 채운 뒤 호출한다.
   - SuperLU는 매 iteration symbolic + numeric + solve를 수행한다. 이 특성이 w/o cuDSS ablation의 의도이다.
4. dx 업로드
   - mixed profile의 CUDA voltage update는 `float d_dx`를 읽는다.
   - CPU `dx` double을 float으로 cast해 `CudaMixedStorage::d_dx`에 업로드한다.

새 timing label:

- `ABLATION.cuda_F_to_cpu`
- `ABLATION.cuda_J_to_cpu_csc`
- 기존 SuperLU 타이머: `CPU.naive.solve.superlu`
- `ABLATION.cpu_dx_to_cuda`

### helper 분리 여부

최소 구현에서는 CPU naive op와 SuperLU op를 그대로 쓰되, hybrid runner가 `CpuFp64Storage` scratch를 관리한다.

후속 정리가 필요하면 아래 함수로 분리한다.

```text
assemble_pypower_like_jacobian(...)
solve_superlu_csc(...)
copy_cpu_csc_to_cuda_csr(...)
copy_cuda_csr_to_cpu_csc(...)
```

이번 실험 단계에서는 core 재사용성보다 ablation profile의 격리와 결과 재현성을 우선한다.

## 후처리 계획

신규 스크립트:

```text
/workspace/exp/20260414/kcc_exp/summarize_kcc_results.py
```

역할:

1. `pypower_operator_profile/summary_operators.csv`에서 pie chart용 CSV 생성
   - 출력: `tables/pypower_operator_pie.csv`
2. `end2end_main_chain/aggregates_end2end.csv`에서 구현 단계별 end-to-end 표 생성
   - 출력: `tables/end2end_main_chain.csv`
3. `cuda_edge_ablation_operators/summary_operators.csv`에서 ablation operator breakdown 생성
   - 출력: `tables/cuda_edge_ablation_operator_breakdown.csv`
4. `jacobian_edge_vs_vertex/summary_operators.csv`에서 edge vs vertex Jacobian update 표 생성
   - 출력: `tables/jacobian_edge_vertex.csv`

그림용 단위:

- table 저장은 ms 기준.
- speedup은 baseline/candidate로 계산한다.
- operator별 값은 repeat 평균과 표준편차를 같이 저장한다.

## 검증 체크리스트

1. 빌드 확인
   - `WITH_CUDA=ON`
   - `ENABLE_TIMING=ON` for operators
   - `ENABLE_TIMING=OFF` for clean end2end
   - 최종 `cuda_edge` 계열 측정은 `CUPF_CUDSS_ENABLE_MT=ON`
   - 최종 `cuda_edge` 계열 측정은 `CUPF_CUDSS_HOST_NTHREADS=AUTO`
   - 최종 `cuda_edge` 계열 측정은 `CUPF_CUDSS_ND_NLEVELS=AUTO`
   - SuperLU, KLU, cuDSS link 확인
2. correctness
   - 모든 profile에서 `success=True`
   - `final_mismatch <= 1e-8`
   - 반복 횟수가 full profile과 크게 다르면 별도 표시
3. timing sanity
   - small case는 CUDA one-time/analyze overhead가 커서 speedup 해석에서 제외 또는 보조로만 사용
   - 큰 케이스(`case4601_goc`, `case8387_pegase`, `case9241_pegase`)를 본문 중심으로 사용
4. ablation 해석
   - w/o Jacobian과 w/o cuDSS는 CPU-GPU 전송 비용이 포함된다.
   - 이는 "해당 GPU 연산자를 제거하고 CPU naive 구현으로 대체했을 때의 end-to-end 영향"으로 해석한다.
   - 순수 커널 시간만 비교하려면 별도 microbenchmark가 필요하다.

## 작업 순서

1. 완료: `run_benchmarks.py`에 신규 profile spec 추가.
2. 완료: `cupf_case_benchmark.cpp`에 hybrid profile parse와 runner dispatch 추가.
3. 완료: benchmark-only host/device bridge와 hybrid runner 구현.
4. 생략: 신규 source를 만들지 않아 `benchmarks/CMakeLists.txt` 수정은 필요 없음.
5. 완료: `case30_ieee`로 smoke test.
   - 이 단계는 correctness 확인용이며 최종 성능 측정에 포함하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --results-root /workspace/exp/20260414/kcc_exp/results \
  --run-name smoke_hybrid_ablation \
  --mode both \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case30_ieee \
  --profiles cuda_edge cuda_wo_cudss cuda_wo_jacobian cuda_fp64_edge \
  --warmup 0 \
  --repeats 1 \
  --cudss-enable-mt \
  --cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0 \
  --cudss-host-nthreads AUTO \
  --cudss-nd-nlevels AUTO
```

   - 결과: `/workspace/exp/20260414/kcc_exp/results/smoke_hybrid_ablation_mt`

6. 남음: 전체 8-case run 실행.
7. 남음: 후처리 스크립트 작성 및 표/그림용 CSV 생성.
8. 남음: `SUMMARY.md`에 최종 결과와 해석 메모 정리.
