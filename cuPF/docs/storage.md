# storage — 버퍼/레이아웃 레이어

storage는 한 실행 프로파일에 필요한 **메모리 버퍼**(와 CPU의 경우 솔버 핸들이 참조할
행렬)를 소유한다. 연산자(ops)는 storage(`buf`)를 받아 버퍼에 접근하며 계산만 한다.
storage는 메모리·레이아웃·호스트↔디바이스 이동만 책임진다.

**소스**: `cpp/src/newton_solver/storage/{cpu,cuda}/`.

---

## 1. 공통 생애주기 (가상 인터페이스 아님)

파이프라인이 정적으로 호출하는 멤버:

```cpp
void prepare(const InitializeContext&);          // initialize 단계 (1회): 토폴로지 업로드 + 버퍼 할당
void upload(const SolveContext&);                 // solve마다: 입력 업로드 + V/Va/Vm 시딩, 배치 resize
void download(NRResult&) const;                   // 단일 케이스(B=1) 전압 읽기
void download_batch(NRBatchResult&) const;        // batch-major 전압 + 케이스별 norm
```

backend/precision은 타입(struct/템플릿 인자)으로 구분되며 런타임 질의 메서드가 아니다.

---

## 2. CPU — `CpuFp64Storage`

CPU FP64 경로의 호스트 버퍼. 두 부분으로 이루어진다:

- **`CpuCscMatrix<T>` (+ `CpuTriplet<T>`)**: Eigen-free CSC(column-major, 항상 compressed)
  희소행렬. `Eigen::SparseMatrix`를 대체하며 파이프라인이 쓰던 API 부분집합
  (`outerIndexPtr`/`innerIndexPtr`/`valuePtr`/`setFromTriplets` 등, 중복 합산)을 흉내낸다.
- **`CpuFp64Storage`**: Ybus(CSC), J(CSC), scatter map, F, dx, Va/Vm/V, Sbus, Ibus 캐시,
  차원. KLU/UMFPACK 솔버 상태는 storage가 아니라 linear solver가 보유.

### CSR→CSC 적응 (특이사항)

Jacobian 심볼릭 *분석*(패턴+scatter map 생성)은 storage가 아니라
[`ops/jacobian`](ops/jacobian.md)에서 backend-무관하게 **CSR 순서**로 만들어진다. 그런데
CPU 백엔드의 행렬은 **CSC**라, `prepare()`가 CSR 패턴→CSC sparsity 구축 + scatter map을
CSC value 위치로 리맵하는 **레이아웃 적응**을 한다(analysis가 아님). CUDA는 J를 CSR
그대로 쓰므로 이 리맵이 없다 — 따라서 CSR→CSC 변환은 CPU 고유다.

---

## 3. CUDA — `CudaBatchedStorage<StateScalar, JacScalar>`

세 CUDA 프로파일은 **버퍼 element 타입만** 달라, 하나의 템플릿과 thin derived struct로
통합돼 있다(`cuda_batched_storage.hpp`). op 헤더들이 `struct CudaFp32Storage;`처럼 전방
선언을 하므로 `using` 별칭 대신 derived struct를 쓴다.

```cpp
struct CudaFp32Storage  : CudaBatchedStorage<float,  float>  {};
struct CudaFp64Storage  : CudaBatchedStorage<double, double> {};
struct CudaMixedStorage : CudaBatchedStorage<double, float>  {};
```

| | StateScalar | JacScalar |
|---|---|---|
| FP32 | `float` | `float` |
| FP64 | `double` | `double` |
| Mixed | `double` | `float` |

- **StateScalar**: 물리 상태와 파생값(Ybus 값, 직교 `d_V_re/im`·극형식 `d_Va/d_Vm`,
  `d_Sbus`, `d_Ibus`, 잔차 `d_F`, 케이스별 norm `d_normF`).
- **JacScalar**: cuDSS에 넘기는 선형해 객체(Jacobian 값 `d_J_values`, 스텝 `d_dx`).
  Mixed는 FP64 상태로 두되 FP32 Jacobian으로 분해/solve를 싸게 한다.
- 인덱스 버퍼(CSR 포인터/인덱스/행, scatter map, 버스타입)는 int32·정밀도 무관.

구현(`cuda_batched_storage.cpp`)은 3개 조합으로 명시적 인스턴스화. 정의가 한 곳이라
정밀도 추가/변경 시 drift가 없다. (배치 size/nnz 공용 accessor
`cuda_storage_batch_size`/`cuda_storage_nnz_j`도 여기 정의 — core/linear_solve가 공유.)

### batch-major 레이아웃

per-bus `[b*n_bus+bus]`, per-residual `[b*dimF+row]`, per-J-value `[b*nnz_J+pos]`,
norm `[b]`. Ybus 패턴은 배치 공통, 값은 `ybus_values_batched`가 아니면 공통. cuDSS
uniform-batch가 기대하는 연속 layout이라 B=1/B>1 동일 경로([system_architecture](system_architecture.md) §4).

### host↔device 변환 커널 (`storage_convert.hpp`)

`upload`/`download`의 per-element 호스트 cast/trig 루프를 디바이스 커널로 대체:
`launch_split_complex<S>`(interleaved complex→re/im), `launch_seed_state_from_v0<S>`
(V0→re/im/Va/Vm, polar 시딩), `launch_pack_complex_to_double<S>`(re/im→interleaved, D2H).
`upload`는 연속 배치면 raw FP64를 한 번에 H2D 후 디바이스 변환(fast path), strided면
호스트 변환(fallback).

---

## 4. DeviceBuffer\<T\>

CUDA device 메모리 RAII 래퍼(`utils/cuda_utils.hpp`, [utils.md](utils.md)).
`resize/assign(H2D)/copyTo(D2H)/memsetZero/data/size/empty`. 복사 불가(이동만) — 실수로
device 메모리를 복사하는 것을 방지.

---

## 5. 선택 기준

| 상황 | storage |
|------|---------|
| CPU 개발·디버깅 | `CpuFp64Storage` (B=1) |
| CUDA 정확도 최우선 | `CudaFp64Storage` (배치 지원) |
| CUDA 처리량 | `CudaMixedStorage`(권장) 또는 `CudaFp32Storage` |

처리량은 보통 Mixed < FP64(FP64 cuDSS 분해가 무거움), FP32가 가장 싸지만 대형 계통에서
수렴 불안정. Mixed는 Jacobian이 FP32라 ill-conditioned 계통에서 반복이 늘 수 있다.
