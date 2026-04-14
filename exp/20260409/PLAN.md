# Schur Complement Experiment Plan

**위치**: `exp/20260409/`  
**날짜**: 2026-04-09  
**목적**: cuDSS Schur complement 모드가 power-flow Jacobian 선형계 풀이에서 full solve보다 빠른지 측정한다.

---

## 1. 배경 및 현재 상태

### 1-1. 이미 있는 것

| 항목 | 위치 | 내용 |
|------|------|------|
| Full cuDSS 벤치마크 | `cudss_benchmark.cpp` | 단일 Newton step 선형계(J×=b)를 cuDSS로 풀고 ANALYSIS / FACTORIZATION / SOLVE를 별도 측정 |
| CPU NR 벤치마크 결과 | `cuPF/benchmarks/results/selected_cases_20260409/` | Python PyPower, C++ PyPowerLike, C++ Optimized 3종 비교. 7개 케이스(118→9241 buses) |
| 선형계 빌드 유틸 | `powerflow_linear_system.hpp/cpp` | V0에서 J, b = −F(V0)를 CPU로 조립. Jacobian은 FP64→FP32 변환 후 cuDSS에 투입 |
| Schur 개념 문서 | `schur_complement.md` | J 블록 구조, cuDSS Schur API, sample 동작 방식 정리 |
| cuDSS sample | `sample_schur_complement.cpp` | NVIDIA 공식 예제. Schur 인덱스 → ANALYSIS → FACTORIZATION → extract S → FWD → 외부 solve → BWD |

### 1-2. CPU 벤치마크에서 확인된 사실 (2026-04-09 기준)

| case (buses) | pypower (s) | pypowerlike C++ (s) | optimized C++ (s) |
|---|---:|---:|---:|
| case118_ieee | 0.01255 | 0.000955 | 0.000527 |
| case793_goc | 0.02181 | 0.008366 | 0.005779 |
| case1354_pegase | 0.03750 | 0.018348 | 0.012262 |
| case2746wop_k | 0.05344 | 0.034381 | 0.023155 |
| case4601_goc | 0.13494 | 0.097436 | 0.076156 |
| case8387_pegase | 0.23560 | 0.171670 | 0.113735 |
| case9241_pegase | 0.31133 | 0.230211 | 0.148902 |

CPU Optimized 기준 solve time이 전체의 대부분을 차지한다(analyze는 5~10%). GPU Schur 실험은 이 CPU solve 비용과도 비교할 수 있다.

---

## 2. 실험 목표

1. **cuDSS full solve** vs **cuDSS Schur + 외부 dense solve** 중 어느 쪽이 빠른가?
2. Schur matrix 추출 비용 (`cudssDataGet`)과 외부 cuSOLVER 비용은 얼마인가?
3. 케이스 크기(n_pq)에 따라 손익분기점이 달라지는가?
4. Schur complement는 실제로 얼마나 dense해지는가?

---

## 3. Schur 블록 선택

Power-flow Jacobian 변수 순서:
```
x = [Δθ(pv), Δθ(pq), Δ|V|(pq)]   (길이 = n_pvpq + n_pq = dimF)
```

블록 구조:
```
J = [J11  J12]   rows pvpq × cols pvpq  |  rows pvpq × cols pq
    [J21  J22]   rows pq   × cols pvpq  |  rows pq   × cols pq
```

**Schur 대상 = J22 (마지막 n_pq 행/열)**

- Schur index vector: `[0, ..., 0, 1, ..., 1]` (앞 n_pvpq개 0, 뒤 n_pq개 1)
- Schur 시스템 크기: n_pq × n_pq
- J22는 전압 크기 변수끼리의 무효전력 민감도 → 물리적으로 비교적 localized

---

## 4. 현재 벤치마크 코드 적합성 검토

### 적합한 부분

| 설계 | 이유 |
|------|------|
| 단일 Newton step 고정 (J, b 불변) | Schur variant에도 동일한 행렬을 쓰므로 공정한 비교 가능 |
| ANALYSIS를 별도 타이밍으로 분리 | Schur 모드도 ANALYSIS 한 번 → REFACTORIZATION 반복 패턴이므로 분리가 의미 있음 |
| FP32 Jacobian 사용 | NR loop 실제 경로와 일치 |
| JSON 출력 + 집계 구조 | Schur 결과도 같은 포맷으로 수집 가능 |
| `CuDssLinearSystemRunner` 클래스 캡슐화 | 새 `CuDssSchurRunner` 클래스를 병렬로 추가하기 쉬운 구조 |

### 부족한 부분 (추가 필요)

| 부족한 점 | 설명 |
|----------|------|
| `--mode` CLI 옵션 없음 | `full` vs `schur_j22` 구분 불가 |
| Schur 전용 `CuDssSchurRunner` 없음 | FWD → 외부 solve → BWD 3-phase 로직 필요 |
| cuSOLVER 의존성 없음 | `CMakeLists.txt`에 `CUDA::cusolver` 미포함 |
| Schur 전용 메트릭 없음 | `schur_dim`, `schur_density`, `schur_extract_sec`, `fwd_sec`, `schur_solve_sec`, `bwd_sec` 필요 |
| Schur index 벡터 생성 로직 없음 | n_pq 정보로 index vector를 만드는 코드 없음 |

**결론**: 현재 `cudss_benchmark`는 baseline(full solve)으로 적합하며, Schur variant를 같은 파일에 `CuDssSchurRunner` 클래스로 추가하는 방식이 가장 깔끔하다.

---

## 5. 구현 계획

### 5-1. `CuDssSchurRunner` 클래스 추가

```
cudss_benchmark.cpp 내부에 CuDssSchurRunner 추가
(CuDssLinearSystemRunner와 동일한 public interface)

  constructor:
    - CUDSS_CONFIG_SCHUR_MODE = 1 설정
    - schur_indices 벡터 생성: [0×n_pvpq, 1×n_pq]
    - cudssDataSet(CUDSS_DATA_USER_SCHUR_INDICES, ...)
    - d_schur_mat 버퍼 할당 (schur_dim × schur_dim, FP32)
    - cuSOLVER handle 생성, workspace 크기 질의 및 할당

  analyze():
    - cudssExecute(ANALYSIS)  ← Schur index 미리 설정 필수
    - cudssDataGet(SCHUR_SHAPE) → schur_dim 확인
    - cudssMatrixCreateDn(&dss_S, ...) 생성

  factorize_and_solve(repeat_idx):
    t0: cudssExecute(FACTORIZATION)          ← partial (Schur boundary까지)
    t1: cudssDataGet(SCHUR_MATRIX, &dss_S)  ← Schur 행렬 추출 (d_schur_mat 갱신)
    t2: cudssExecute(SOLVE_FWD_PERM|SOLVE_FWD)  ← L solve, d_b 변형
    t3: cuSOLVER getrf + getrs on d_schur_mat    ← Schur system solve (in-place on d_b[schur part])
    t4: cudssExecute(SOLVE_BWD|SOLVE_BWD_PERM)  ← U solve → d_x
    → 반환: SchurRepeatMetrics
```

### 5-2. CLI 옵션 추가

```
--mode full        기존 CuDssLinearSystemRunner  (FACTORIZATION + SOLVE)
--mode schur_j22   새 CuDssSchurRunner           (J22를 Schur block으로)
```

### 5-3. 메트릭 추가

`RepeatMetrics`에 추가:
```cpp
// Schur 전용 (mode = schur_j22일 때만 유효)
int32_t schur_dim = 0;
double  schur_extract_sec = 0.0;  // cudssDataGet(SCHUR_MATRIX) 시간
double  fwd_solve_sec     = 0.0;  // SOLVE_FWD_PERM | SOLVE_FWD
double  schur_solve_sec   = 0.0;  // cuSOLVER dense solve
double  bwd_solve_sec     = 0.0;  // SOLVE_BWD | SOLVE_BWD_PERM
```

JSON 출력에도 이 필드를 추가한다.

### 5-4. CMakeLists.txt 수정

```cmake
# exp/20260409/CMakeLists.txt
target_link_libraries(cudss_benchmark PRIVATE
    cupf_core
    CUDA::cusolver   # 추가
)
```

### 5-5. schur_density 측정

FACTORIZATION 이후 Schur 행렬을 host로 내려받아:
```cpp
double schur_nnz_estimated = 0;
for (int i = 0; i < schur_dim * schur_dim; i++) {
    if (std::abs(h_schur[i]) > threshold) schur_nnz_estimated++;
}
double schur_density = schur_nnz_estimated / (double)(schur_dim * schur_dim);
```

(선택적 — 측정 오버헤드가 있으니 `--measure-schur-density` 플래그로 구분)

---

## 6. 측정 지표 및 비교 축

```
case | n_pq | full_factor_sec | full_solve_sec | full_total_sec
                              | schur_factor_sec | schur_extract_sec
                              | schur_fwd_sec | schur_dense_sec | schur_bwd_sec | schur_total_sec
                              | speedup = full_total / schur_total
```

### 예상 결과 형태

| case | n_pq | full_total(ms) | schur_total(ms) | speedup |
|------|------|---------------|----------------|---------|
| case118_ieee | ~? | ? | ? | ? |
| case9241_pegase | ~? | ? | ? | ? |

Schur가 도움이 될지는 다음에 달려 있다:
- **n_pq 비율**: n_pq / dimF가 클수록 external Schur solve 비용 증가
- **Schur density**: dense에 가까울수록 cuSOLVER 비용 O(n_pq³)가 지배
- **J11 factorization 절감**: J22를 빼고 J11만 인수분해하면 실질적으로 빨라지는지

---

## 7. n_pq 규모 사전 파악

케이스별 n_pq는 현재 `powerflow_linear_system.cpp`에서 빌드 시 확인 가능.  
이미 `[system]` 로그에 `pq=<n_pq>`가 출력됨. 실험 전 아래를 먼저 확인한다:

```bash
./build/cudss_benchmark --case case118_ieee --repeats 0
./build/cudss_benchmark --case case9241_pegase --repeats 0
```

---

## 8. 실험 실행 순서

```
Step 1: CMakeLists.txt에 CUDA::cusolver 추가 후 재빌드
Step 2: CuDssSchurRunner 구현 + --mode 옵션 추가
Step 3: 수치 검증: --mode full vs --mode schur_j22 → residual_inf 동등 확인
Step 4: 성능 측정 (각 케이스, warmup=1, repeats=5)
Step 5: 결과 JSON 저장, 집계 및 비교
```

### 빌드 커맨드 (예시)

```bash
cmake -S /workspace/exp/20260409 -B /workspace/exp/20260409/build -GNinja
cmake --build /workspace/exp/20260409/build --target cudss_benchmark
```

### 실행 예시

```bash
# Full solve baseline
./build/cudss_benchmark --case case118_ieee --mode full \
  --warmup 1 --repeats 5 \
  --output-json results/case118_full.json

# Schur complement (J22 block)
./build/cudss_benchmark --case case118_ieee --mode schur_j22 \
  --warmup 1 --repeats 5 \
  --output-json results/case118_schur_j22.json
```

---

## 9. 주의사항

### cuDSS ANALYSIS는 Schur 설정 전에 호출하면 안 된다
`CUDSS_DATA_USER_SCHUR_INDICES` 설정과 `CUDSS_CONFIG_SCHUR_MODE` 활성화는 ANALYSIS 전에 완료해야 한다. (sample 주석 참조)

### FACTORIZATION은 partial factorization이다
Schur 모드에서 FACTORIZATION은 J11 부분만 인수분해하고 Schur boundary에서 멈춘다.  
따라서 SOLVE (full) 대신 SOLVE_FWD → external → SOLVE_BWD 3단계를 써야 한다.

### sample의 SOLVE_DIAG는 대칭 행렬(LDL^T)에만 필요하다
power-flow Jacobian은 비대칭(CUDSS_MTYPE_GENERAL)이므로 SOLVE_DIAG 단계는 스킵한다.  
3-phase: FWD → external → BWD.

### cuSOLVER는 Schur 행렬을 in-place로 덮어쓴다
getrf 후 Schur 행렬 버퍼는 L/U factor로 overwrite된다.  
다음 반복에서 cuDSS REFACTORIZATION 후 cudssDataGet으로 다시 가져와야 한다.

### Schur 행렬이 sparse할 수 있다
cuDSS는 dense Schur를 기본 반환한다. 만약 실제로 sparse하다면 dense cuSOLVER가 비효율적이다.  
이 경우 cuDSS `CUDSS_DATA_SCHUR_SHAPE[2]` (sparse nnz)를 확인해 판단한다.
