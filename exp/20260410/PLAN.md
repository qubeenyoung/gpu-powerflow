# cuPF 5-Way Case Benchmark Plan

**위치**: `exp/20260410/`  
**날짜**: 2026-04-09  
**목적**: `cuPF/benchmarks`의 결과 포맷을 기준으로, 기본 테스트 케이스들에 대해 `pypower`, `cuPF cpu naive`, `cuPF cpu optimized`, `cuPF cuda edge`, `cuPF cuda vertex`를 각각 10회 측정하고 결과를 종합하는 실험을 설계한다.  
**범위**: 이번 단계에서는 계획만 작성한다. 코드 구현, 러너 수정, 실제 벤치마크 실행은 하지 않는다.

## 1. 참고 기준

이번 계획은 아래 기존 코드와 문서 구조를 그대로 이어받는 것을 전제로 한다.

- `cuPF/benchmarks/run_benchmarks.py`
  - 현재 결과 디렉터리 구조, `manifest.json`, `summary.csv`, `aggregates.csv`, `README.md`, `SUMMARY.md`, `raw/` 작성 방식을 이미 갖고 있다.
- `cuPF/benchmarks/cpp/cupf_case_benchmark.cpp`
  - 이미 `--backend cpu|cuda`, `--jacobian edge_based|vertex_based`, `--algorithm optimized|pypower_like`, `--warmup`, `--repeats`를 지원한다.
- `python/converters/common.py`
  - 현재 기본 벤치마크 대상 케이스 집합 `TARGET_CASES`를 정의한다.
- `exp/20260408/docs/benchmark_strategy.md`
  - CUDA 타이밍이 비동기 실행 때문에 별도 동기화 정책을 필요로 한다는 점을 이미 정리해 두었다.

## 2. 현재 상태와 차이점

현재 `cuPF/benchmarks`는 완전히 동일한 실험을 바로 수행할 수 있는 상태는 아니다.

- `run_benchmarks.py`는 현재 `pypower`, `cpp_pypowerlike`, `cpp_optimized` 3개만 실행한다.
- `run_cpp_case()`는 내부에서 `--backend cpu`를 고정한다.
- `ensure_cpp_benchmark_binary()`는 benchmark binary를 빌드할 때 `-DWITH_CUDA=OFF`를 강제로 넣는다.
- `CMakePresets.json`의 `bench-release`는 `BUILD_BENCHMARKS=ON`, `BUILD_NAIVE_CPU=ON`만 켜고, CUDA까지 함께 켜는 preset은 아직 없다.
- `cupf_case_benchmark.cpp`는 `analyze()`와 `solve()` 바깥에서 host wall-clock을 재는데, CUDA는 비동기 실행이므로 timing build 전제가 없으면 특히 `analyze_sec`가 과소 측정될 수 있다.

정리하면, 이번 실험은 기존 benchmark 파이프라인을 **5개 구현 조합으로 확장하는 계획**으로 보는 것이 맞다.

## 3. 대상 케이스

기본 대상은 `TARGET_CASES` 7개를 그대로 사용한다.

- `118_ieee`
- `793_goc`
- `1354_pegase`
- `2746wop_k`
- `4601_goc`
- `8387_pegase`
- `9241_pegase`

별도 요청이 없으면 이번 실험의 기본값은 위 7개 전체다.

## 4. 비교 대상 정의

실험에서 비교할 구현은 아래 5개로 고정한다.

| 사람용 이름 | 결과 ID | backend | algorithm | jacobian | 비고 |
|---|---|---|---|---|---|
| `pypower` | `pypower` | - | - | - | Python 기준선 |
| `cuPF cpu naive` | `cpp_pypowerlike` | `cpu` | `pypower_like` | `edge_based` | `BUILD_NAIVE_CPU=ON` 필요 |
| `cuPF cpu optimized` | `cpp_optimized` | `cpu` | `optimized` | `edge_based` | 현재 CPU 기준선 |
| `cuPF cuda edge` | `cpp_cuda_edge` | `cuda` | `optimized` | `edge_based` | `WITH_CUDA=ON` 필요 |
| `cuPF cuda vertex` | `cpp_cuda_vertex` | `cuda` | `optimized` | `vertex_based` | `WITH_CUDA=ON` 필요 |

메모:

- CUDA 경로에서는 실제 백엔드 선택이 `backend=cuda`로 결정되므로, `algorithm=optimized`는 CLI 일관성을 위한 고정값으로 취급한다.
- CPU 두 구현은 현재 benchmark와 동일하게 `edge_based`를 사용한다.
- `pypower`는 edge/vertex 개념이 없으므로 별도 Jacobian 구분 없이 하나의 기준선으로 둔다.

## 5. 측정 정책

### 5-1. 반복 횟수

- 각 구현/케이스 조합마다 **측정 10회**
- 측정 전 **warmup 1회** 권장
- 즉, 실측 기준으로는 `repeats=10`, warmup은 집계에서 제외

이유:

- 기존 `cuPF/benchmarks`가 이미 warmup + repeats 구조를 사용한다.
- 사용자가 요청한 "10번씩 실행"은 집계 대상 실측 10회로 해석하는 것이 가장 자연스럽다.

### 5-2. 수렴 조건

기본 파라미터는 현 benchmark와 동일하게 유지한다.

- `tolerance = 1e-8`
- `max_iter = 50`

### 5-3. CUDA 타이밍 규칙

CUDA 측정은 반드시 **timing build 전제**로 잡는다.

- 이유: `cupf_case_benchmark.cpp`는 `solver.analyze()`와 `solver.solve()` 바깥에서 시간을 재고,
  CUDA는 비동기이므로 별도 동기화가 없으면 `analyze_sec`, `solve_sec`가 공정하지 않을 수 있다.
- 따라서 CUDA 포함 실험에서는 `CUPF_ENABLE_TIMING`을 켠 빌드 또는 동일한 동기화 효과를 보장하는 경로를 사용한다.

## 6. 권장 빌드 전략

### 권장안: 단일 benchmark binary

가장 단순한 운영 방식은 하나의 benchmark binary가 4개 cuPF 변형을 모두 수행하게 만드는 것이다.

필요 조건:

- `BUILD_BENCHMARKS=ON`
- `BUILD_NAIVE_CPU=ON`
- `WITH_CUDA=ON`
- CUDA timing을 위한 compile flag 활성화

이 방식의 장점:

- 구현별로 binary를 나눠 관리하지 않아도 된다.
- `manifest.json`과 command log가 단순해진다.
- 한 run 안에서 5개 구현을 같은 환경 정보 아래 기록하기 쉽다.

### 대안: CPU용 / CUDA용 benchmark binary 분리

만약 단일 빌드가 환경 의존성 때문에 불편하면 아래처럼 둘로 나눌 수 있다.

- CPU binary: `pypower`, `cpp_pypowerlike`, `cpp_optimized`
- CUDA binary: `cpp_cuda_edge`, `cpp_cuda_vertex`

이 경우에도 결과 포맷은 하나로 합치되, command log나 manifest에 어떤 binary를 썼는지 남겨야 한다.

## 7. 실행 순서 계획

케이스 단위로 아래 순서를 반복하는 방식이 가장 자연스럽다.

1. `preprocess_case()`로 MAT case를 읽는다.
2. `save_cupf_dump()`로 cuPF dump를 1회 생성한다.
3. `pypower`를 warmup 1회 + 측정 10회 실행한다.
4. `cpp_pypowerlike`를 warmup 1회 + 측정 10회 실행한다.
5. `cpp_optimized`를 warmup 1회 + 측정 10회 실행한다.
6. `cpp_cuda_edge`를 warmup 1회 + 측정 10회 실행한다.
7. `cpp_cuda_vertex`를 warmup 1회 + 측정 10회 실행한다.

이 순서는 현재 `run_benchmarks.py`의 케이스별 처리 흐름과도 잘 맞는다.

## 8. 결과 저장 구조

결과 디렉터리는 `cuPF/benchmarks/results/` 대신 이번 실험 폴더 아래로 분리한다.

권장 구조:

```text
exp/20260410/results/<run_name>/
  manifest.json
  README.md
  SUMMARY.md
  summary.csv
  aggregates.csv
  cases/
    <case>.json
  raw/
    pypower/<case>/run_00.json
    cpp_pypowerlike/<case>/run_00.json
    cpp_optimized/<case>/run_00.json
    cpp_cuda_edge/<case>/run_00.json
    cpp_cuda_vertex/<case>/run_00.json
```

`<run_name>` 예시:

- `selected_cases_20260410`
- `selected_cases_20260410_rerun01`
- timestamp 기반 이름

핵심은 기존 `cuPF/benchmarks` 구조를 유지해서 후속 스크립트 재사용성을 높이는 것이다.

## 9. 수집할 필드

`summary.csv`에는 기존 필드를 최대한 유지하면서 아래 정보를 넣는다.

- `implementation`
- `case_name`
- `case_stem`
- `repeat_idx`
- `backend`
- `jacobian`
- `algorithm`
- `success`
- `iterations`
- `final_mismatch`
- `elapsed_sec`
- `analyze_sec`
- `solve_sec`
- `max_abs_v_delta_from_v0`
- `buses`
- `pv`
- `pq`

필요 시 아래 보조 필드를 추가할 수 있다.

- `binary_path` 또는 `build_label`
  - CPU/CUDA binary를 분리했을 때 재현성을 더 분명히 남기기 위함

## 10. 집계 및 요약 계획

### 10-1. CSV 집계

`aggregates.csv`는 현재 구현을 그대로 확장해서 아래 통계를 유지한다.

- `runs`
- `success_all`
- `iterations_mean`
- `final_mismatch_max`
- `elapsed_sec_mean`, `median`, `min`, `max`, `stdev`
- `analyze_sec_mean`, `median`, `min`, `max`, `stdev`
- `solve_sec_mean`, `median`, `min`, `max`, `stdev`

### 10-2. Markdown 요약

`SUMMARY.md`는 5개 구현 기준으로 표를 확장한다.

필수 표:

1. 전체 elapsed 비교
2. C++/CUDA analyze, solve breakdown
3. correctness snapshot
4. speedup 표

권장 비교 축:

- `pypower` 대비 각 cuPF 구현 speedup
- `cpp_optimized` 대비 `cpp_cuda_edge`, `cpp_cuda_vertex` speedup
- `cpp_cuda_edge` 대비 `cpp_cuda_vertex` speedup

예상 표 형태:

```text
case | pypower | cpu naive | cpu optimized | cuda edge | cuda vertex
```

```text
case | cpu optimized solve | cuda edge solve | cuda vertex solve
```

```text
case | pypower success | cpu naive success | cpu optimized success | cuda edge success | cuda vertex success
```

## 11. 구현 시 필요한 변경 포인트

이번 단계에서는 하지 않지만, 나중에 실제로 구현할 때 바뀌어야 하는 지점은 아래와 같다.

1. `run_cpp_case()` 일반화
   - `backend`, `jacobian`, `algorithm`, `binary`를 인자로 받게 바꾼다.
2. 구현 매트릭스 도입
   - 지금처럼 2번 직접 호출하지 않고, 구현 정의 리스트를 순회하게 바꾼다.
3. benchmark binary 빌드 경로 확장
   - `ensure_cpp_benchmark_binary()`가 CUDA 포함 빌드까지 지원하게 바꾼다.
4. `manifest["implementations"]` 확장
   - 3개 구현 고정값을 5개 구현으로 바꾼다.
5. `README.md`, `SUMMARY.md` 생성 로직 확장
   - 현재 3열 테이블을 5열 테이블로 변경한다.

핵심은 solver 자체를 바꾸는 것이 아니라, 기존 benchmark 러너를 5개 조합까지 일반화하는 것이다.

## 12. 검증 기준

실행 후 결과를 인정하기 위한 최소 기준은 아래와 같다.

- 모든 구현/케이스/반복에서 `success=true`
- `final_mismatch <= tolerance`
- `iterations_mean`이 구현 간에 비정상적으로 벌어지지 않을 것
- `elapsed_sec`, `analyze_sec`, `solve_sec`가 모두 유한값일 것
- CUDA 결과가 CPU optimized 대비 현저히 다른 수렴 행동을 보이면 별도 확인할 것

## 13. 한계와 주의사항

현재 결과 포맷만으로는 구현 간 최종 전압 해(`V`) 차이를 직접 비교하지는 못한다.

- 지금 수집되는 것은 `final_mismatch`, `iterations`, `max_abs_v_delta_from_v0` 중심이다.
- 즉, 이번 계획의 1차 목표는 **성능 비교 + 수렴 여부 비교**다.
- 만약 이후에 `max|V_variant - V_ref|`까지 보고 싶다면 raw payload에 최종 `V` 또는 비교용 요약값을 추가하는 별도 확장이 필요하다.

또한 naive CPU는 SuperLU, CUDA는 cuDSS가 필요하므로 환경 의존성도 manifest에 분명히 남겨야 한다.

## 14. 완료 조건

이번 실험 계획의 완료 조건은 아래와 같다.

- `exp/20260410/` 아래에 실험 계획 문서가 존재한다.
- 5개 구현 조합, 7개 기본 케이스, 10회 반복 정책이 명시되어 있다.
- 결과 저장 구조와 집계 방식이 `cuPF/benchmarks` 기준으로 정의되어 있다.
- 현재 benchmark 코드가 바로 지원하지 않는 부분과 이후 변경 포인트가 정리되어 있다.
