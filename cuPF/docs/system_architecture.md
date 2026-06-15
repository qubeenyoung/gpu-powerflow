# cuPF — 설계 문서

cuPF는 GPU 가속 Newton–Raphson(NR) 전력조류(power flow) 솔버다. 미분 가능
(differentiable)하여 adjoint(역전파)로 부하에 대한 gradient를 계산할 수 있고,
시나리오 배치(batch)를 한 번에 푼다.

이 디렉터리는 컴포넌트별 설계 문서로 구성된다. 본 문서(`README.md`)는 전체 구조와
데이터 흐름을 설명하고, 세부는 각 컴포넌트 문서로 연결한다.

| 문서 | 내용 |
|------|------|
| [newton_solver.md](newton_solver.md) | 공개 API·생애주기·옵션/결과 구조체·파이프라인 선택 |
| [core.md](core.md) | 파이프라인 variant, 컨텍스트, adjoint 드라이버, torch bridge |
| [storage.md](storage.md) | 버퍼/레이아웃 레이어(CPU·CUDA), batch-major, DeviceBuffer |
| [ops/ibus.md](ops/ibus.md) | 버스 전류 Ibus = Ybus·V |
| [ops/mismatch.md](ops/mismatch.md) | 잔차 F = V·conj(Ibus) − Sbus, 수렴 norm |
| [ops/jacobian.md](ops/jacobian.md) | Jacobian 심볼릭 분석 + GPU 조립 variant 3종 |
| [ops/linear_solve.md](ops/linear_solve.md) | cuDSS / KLU / UMFPACK 직접 솔버, forward·adjoint |
| [ops/voltage_update.md](ops/voltage_update.md) | Newton step 적용 + 전압 재구성 |
| [bindings.md](bindings.md) | Python 바인딩(_cupf), numpy/torch zero-copy |
| [utils.md](utils.md) | DeviceBuffer·스트림, 타이밍, NVTX, dump, 로깅 |
| [contribution-analysis.md](contribution-analysis.md) | claim 기반 기여 분석(전력조류 한정); 헤드라인=FP32-native가 배치 NR-PF에 충분 |
| [precision-study-report.md](precision-study-report.md) | **통합 보고서** — Mixed 정밀도 연구 전체(근거·A/B 결과·비대칭 증명·데이터 분포·결론) |
| [mixed-precision-plan.md](mixed-precision-plan.md) | Mixed의 이론적 근거(inexact Newton + 혼합정밀 IR)와 "더 FP32로/1e-3 바닥 돌파" 연구 계획 |
| [mixed-precision-findings.md](mixed-precision-findings.md) | A/B 실험 결과: Mixed=FP64 정확도(η<1 측정), full-FP32 1e-3 바닥은 state/Ybus 표현한계(보정합산 무효)→Mixed가 최적. 하니스 `tests/precision_study/` |
| [precision-error-measurement.md](precision-error-measurement.md) | 정밀도 오차의 과학적 측정(후방오차 ω·κ(J)·합산조건수 κ_sum·running-error) + FP32 비대칭 **증명**: 선형해=후방안정(ω≈u32, 자기보정)이라 가능, 잔차=상쇄로 ill-conditioned(κ_sum·u32~1e-3, 보정불가)라 불가; 데이터 분포가 근본원인 |

---

## 1. 레이어 구조

```
Python (numpy / torch)
        │
   bindings/            pybind11 모듈 _cupf  (+ torch zero-copy 확장)
        │
   core/                NewtonSolver 오케스트레이션 + 파이프라인 + adjoint
        │
   ops/                 NR 단계 연산자 (ibus·mismatch·jacobian·linear_solve·voltage_update)
        │
   storage/             정밀도 프로파일별 디바이스/호스트 버퍼 소유
        │
   utils/               DeviceBuffer, 스트림, 타이밍/프로파일/덤프
```

소스 트리:

```
cpp/inc/newton_solver/core/   공개 헤더 (newton_solver.hpp, newton_solver_types.hpp)
cpp/inc/utils/                공개 util 헤더
cpp/src/newton_solver/core/   드라이버·파이프라인·adjoint·torch bridge
cpp/src/newton_solver/ops/    {ibus, jacobian, linear_solve, mismatch, voltage_update}
cpp/src/newton_solver/storage/{cpu, cuda}
bindings/                     pybind_cupf.cpp, torch_cupf_extension.cpp
```

핵심 설계: **연산자(ops)는 계산만, 저장소(storage)는 메모리만** 소유한다. 둘을 묶는
것이 "파이프라인 프로파일"이고, `NewtonSolver`가 그 프로파일을 골라 단계를 호출한다.
추상 가상 인터페이스는 없고, 파이프라인을 `std::variant`에 담아 `std::visit`로 정적
디스패치한다(자세히는 [core.md](core.md)).

---

## 2. 데이터 흐름

```
initialize(Ybus, pv, pq)
   └─ Jacobian 심볼릭 분석(패턴+scatter map, 1회)  →  pipeline.initialize()

solve_batch(Ybus, Sbus, V0, batch_size, ...)
   ├─ upload: 입력을 디바이스로(정밀도 변환), 배치 크기로 버퍼 resize
   ├─ NR 루프 (max_iter까지 또는 수렴):
   │     ibus → mismatch → mismatch_norm → [수렴이면 종료]
   │     → jacobian → prepare_rhs → factorize → solve → voltage_update
   ├─ (옵션) prepare_adjoint_cache: 최종 상태 factorization 캐시
   └─ download: 수렴 전압 + 케이스별 통계

solve_adjoint(dL/dVa, dL/dVm, ...)      # 미분: J^T λ = dL/dx → load gradient
```

NR 루프의 단계 순서와 "수렴 체크를 jacobian/solve보다 먼저" 하는 이유는
[core.md](core.md)의 `run_iteration_stages` 참조.

---

## 3. 실행 프로파일 (backend × 정밀도)

`NewtonOptions{backend, compute, ...}`로 선택한다.

| 프로파일 | backend | compute | 상태 정밀도 | Jacobian/solve | 배치 |
|----------|---------|---------|-------------|----------------|------|
| CPU FP64 | CPU | FP64 | double | double (KLU/UMFPACK) | B=1 |
| CUDA FP64 | CUDA | FP64 | double | double (cuDSS) | ✅ |
| CUDA FP32 | CUDA | FP32 | float | float (cuDSS) | ✅ |
| CUDA Mixed | CUDA | Mixed | double | float (cuDSS) | ✅ |

- **FP64**: 정확도 최우선. **Mixed**: FP64 상태 + FP32 Jacobian으로 분해/solve를
  싸게(권장 처리량). **FP32**: 가장 싸지만 대형 ill-conditioned 계통에서 수렴 불안정.
- 배치(batch_size > 1)는 CUDA 세 프로파일 모두 지원(CPU는 단일 케이스).
- (선택) `CUPF_ENABLE_CUSTOM_SOLVER` 빌드 시 CUDA FP64에 외부 custom 직접 솔버 사용 가능.

---

## 4. 배치 모델

모든 배치 버퍼는 **batch-major + 연속(contiguous)** 이다. 케이스 `b`에 대해:

| 배열 | 위치 |
|------|------|
| per-bus (V, Va, Vm, Sbus, Ibus) | `[b*n_bus + bus]` |
| per-residual (F, dx) | `[b*dimF + row]` |
| per-J-value (J 값) | `[b*nnz_J + pos]` |

이 레이아웃이 cuDSS **uniform-batch**(`CUDSS_CONFIG_UBATCH_SIZE`)가 기대하는 형태라,
배치 전체가 하나의 cuDSS 문제로 풀린다 — B=1과 B>1이 동일 경로. Ybus 패턴/값은
배치 공통(시나리오별 admittance가 다르면 값만 케이스별로 둘 수 있음).

`dimF = n_pv + 2·n_pq` (각도 n_pvpq개 + 크기 n_pq개). 잔차/스텝 벡터 layout은
`[dVa@pv | dVa@pq | dVm@pq]`.

---

## 5. 수학 노테이션 (요약)

- 복소 전압 `V = Vm·e^{jVa}`, 버스 전류 `Ibus = Ybus·V`, 주입 전력 `S(V) = V·conj(Ibus)`.
- 잔차(mismatch) `F = S(V) − Sbus`. 실부=유효전력 P, 허부=무효전력 Q.
- NR 스텝: `J·dx = F`, `x ← x − dx` (x = [Va@pvpq, Vm@pq]).
- adjoint: 수렴 상태에서 `J^T λ = dL/dx`, load gradient = −λ (해당 버스).

블록 Jacobian과 민감도 유도는 [ops/jacobian.md](ops/jacobian.md), 잔차 식은
[ops/mismatch.md](ops/mismatch.md), adjoint 경로는 [core.md](core.md) 참조.

---

## 6. 빌드 플래그

| 플래그 | 기본 | 효과 |
|--------|------|------|
| `WITH_CUDA` | OFF | CUDA 백엔드(cuDSS) 빌드. 끄면 CPU 전용 |
| `BUILD_PYTHON_BINDINGS` | — | pybind11 `_cupf` 모듈 |
| `CUPF_WITH_TORCH` | OFF | torch zero-copy forward/backward 확장(`torch_cupf_extension.cpp`) |
| `ENABLE_TIMING` | OFF | `ScopedTimer` 단계별 wall-clock 수집(끄면 타이밍 0) |
| `CUPF_ENABLE_CUSTOM_SOLVER` | OFF | 외부 custom CUDA FP64 직접 솔버 어댑터 |
| `BUILD_EVALUATORS` / `BUILD_TESTING` | — | 평가기/테스트(`tests/`) |

cuDSS 튜닝 옵션(reordering·matching·pivot·MT)은 빌드/런타임 양쪽에서 조정 — 자세히는
[ops/linear_solve.md](ops/linear_solve.md).

예시(전형적 CUDA+torch 빌드):

```
cmake -S cuPF -B build -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_WITH_TORCH=ON -DENABLE_TIMING=ON \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
```
