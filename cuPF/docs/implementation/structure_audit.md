# 구조 적합성 진단 + 과대 파일 분리 (cycle 2 / s2)

- 작성일: 2026-05-31
- 대상: 목표 #0(구조 적합성), #2(한 파일에 과도한 로직)
- 검증: `WITH_CUDA=ON BUILD_EVALUATORS=ON` 빌드 green + `ctest`(cupf_minimal_tests) pass

## 1. 설계 원칙 vs 실제 src

`docs/overview.md`가 명시한 원칙과 실제 `cpp/src` 구조를 대조했다.

| 원칙(docs) | 실제 src | 판정 |
|---|---|---|
| inc=공개 헤더 / src=구현 | `cpp/inc`(core 인터페이스·utils), `cpp/src`(core/ops/storage/reference) | 적합 |
| Storage + Op 분리 (IStorage + 4 Op) | `storage/{cpu,cuda}`, `ops/{ibus,mismatch,jacobian,linear_solve,voltage_update}` | 적합 |
| 4-stage hot path | mismatch→jacobian→linear_solve→voltage_update 디렉터리로 1:1 대응 | 적합 |
| CPU/CUDA 백엔드별 파일 분리 | storage·ops 대부분 `cpu_*`/`cuda_*`로 분리 | 대체로 적합 |
| "한 파일 = 하나의 main kernel 또는 얇은 orchestrator" | 일부 파일이 다중 책임 보유 (아래 2절) | **부분 위반 → 본 작업에서 교정** |

문서-구현 불일치(별도 기록, 후속 cycle 대상):
- `overview.md`가 `cupf::` 네임스페이스를 전제하나 실제 코드는 전역 네임스페이스를 사용한다.
- `overview.md`의 "solver stage configuration::build", "NewtonSolver stage ownership" 등 표현이
  실제 식별자와 매칭되지 않는다(이전 일괄 치환 흔적으로 추정). → 목표 #1(명명 일관성)에서 정리.
- `core/solver_stages.cpp`는 1줄 tombstone(주석만)으로 CMake 소스 목록에도 없다. → git에서 제거 권고.

## 2. 과대/다중책임 파일 측정 및 처리

리팩토링 전 LOC 상위 .cpp/.cu와 책임 분석:

| 파일 | LOC(전) | 섞인 책임 | 조치 |
|---|---|---|---|
| `ops/linear_solve/cuda_cudss.cpp` | 620 | cuDSS forward solve + adjoint transpose + **CSR transpose 패턴 중복** | 중복 제거(아래) → 586 |
| `core/newton_solver_adjoint.cpp` | 579 | adjoint 수치 헬퍼 + 백엔드별 pipeline orchestration | **수치 헬퍼 분리** → 350 |
| `core/newton_solver_torch_bridge.cpp` | 500 | torch tensor bridge(단일 책임) | 유지(후속 검토) |
| `core/newton_solver.cpp` | 448 | NewtonSolver facade(forward+adjoint 드라이버) | 유지(후속 검토) |

### 발견: `build_transpose_pattern` / `CsrTransposePattern` 완전 중복
`cuda_cudss.cpp`와 `newton_solver_adjoint.cpp`에 동일한 익명-네임스페이스 정의가
양쪽에 복제되어 있었다(ODR상 별개 타입이지만 유지보수 위험).

### 조치 — 단일 책임 단위로 분리
신규 컴파일 단위 2개를 만들어 책임을 분리하고 중복을 제거했다(순수 코드 이동, 로직 불변).

1. `core/csr_transpose.{hpp,cpp}` — 일반 CSR→CSC(전치) 희소 패턴 유틸
   - `CsrTransposePattern`, `build_transpose_pattern`, `transpose_batched_values<T>`
   - `cuda_cudss.cpp`와 adjoint 양쪽이 공유 → 중복 제거.
2. `core/newton_solver_adjoint_math.{hpp,cpp}` — adjoint 순수 수치 헬퍼
   - `validate_adjoint_args`, `build_grad_state`, `project_load_gradients`,
     `relative_residual_norm_csr<T>`, `relative_residual_norm_csc`
   - `newton_solver_adjoint.cpp`는 백엔드별 pipeline orchestration만 보유하도록 정리.

`CMakeLists.txt`의 `CUPF_CORE_SOURCES`에 두 .cpp를 등록했다(CPU-only, CUDA 빌드에도 안전).

### 결과 LOC
- `newton_solver_adjoint.cpp`: 579 → **350** (-229)
- `cuda_cudss.cpp`: 620 → **586** (중복 34줄 제거)
- 신규: csr_transpose(hpp 44 + cpp 34), adjoint_math(hpp 99 + cpp 137)

## 3. 후속 분리 후보 (이번 cycle 미적용, 위험/범위상 분리 권고)

- `cuda_cudss.cpp`(586): forward(init/factorize/solve) vs adjoint transpose 캐시 메서드로
  추가 분리 가능. 단, `template struct CudaLinearSolveCuDSS<...>` 명시적 인스턴스화가 한 TU에
  묶여 있어 멤버 정의를 다른 TU로 옮기면 인스턴스화 전략을 함께 바꿔야 한다(ODR 주의).
- `newton_solver_torch_bridge.cpp`(500): forward/backward 분리 여지.
- `solver_stages.cpp` tombstone 제거(git rm 필요).
