# storage — 스토리지 레이어 설계

Storage는 한 실행 프로파일에 필요한 **메모리 버퍼**와 **라이브러리 핸들**을 소유하는 레이어다.
stage Op는 Storage(`buf`)를 인자로 받아 버퍼에 접근하며 계산 로직만 구현한다.

---

## Storage 공통 메서드 (가상 인터페이스 아님)

각 프로파일은 자기 Storage struct(`CpuFp64Storage`, `CudaFp64Storage`,
`CudaFp32Storage`, `CudaMixedStorage`)를 갖는다. 공통 base class나
`op_interfaces.hpp` 같은 추상 인터페이스는 없고, pipeline이 정적으로 아래
같은 멤버를 호출한다.

```cpp
void prepare(const InitializeContext& ctx);   // initialize 단계 (한 번)
void upload(const SolveContext& ctx);          // solve 시작 시 데이터 업로드
void download(NRResult& result) const;         // 최종 결과 다운로드 (B=1)
void download_batch(NRBatchResult& result) const;  // batch 결과
```

backend/precision은 타입 자체(struct + dtype)로 구분되며 런타임 질의 메서드가 아니다.

### prepare()

`initialize()` 단계에서 **한 번만** 호출된다. 다음을 수행한다.
- 위상 메타데이터(n_bus, n_pvpq, n_pq, dimF, nnz_ybus, nnz_J) 계산
- JacobianScatterMap, JacobianPattern 복사
- 버퍼 할당 및 초기화 (single-case 크기; batch 크기는 upload에서 재조정)
- Jacobian CSC/CSR 행렬 구성 (CPU) 또는 device 업로드 (CUDA)

### upload()

`solve()` 시작 시마다 호출된다. Ybus 값, Sbus, V0를 device로 올린다.
Ybus의 희소 구조가 `initialize()` 당시와 같은지 검증한다.
기본 실행 모델은 batch-major layout이며, single-case solve는 `batch_size=1`이다.

### download() / download_batch()

NR 루프 완료 후 device의 V 벡터를 host result로 복사한다.
`download()`은 single-case(B=1) 전용, `download_batch()`은 batch-major
[batch_size, n_bus] 결과 + 케이스별 final mismatch norm을 채운다.

---

## CpuFp64Storage

**파일:** [cpu_fp64_storage.hpp](../../cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.hpp)

CPU FP64 경로의 host-side 버퍼와 (Eigen-free) CSC 행렬·KLU 상태를 관리한다.
single-case 전용(`batch_supported = false`).

### 버퍼 레이아웃

| 버퍼 | 타입 | 크기 | 설명 |
|------|------|------|------|
| `Ybus` | `CpuCscMatrix<complex<double>>` | n_bus × n_bus | Eigen-free CSC Ybus |
| `J` | `CpuCscMatrix<double>` | dimF × dimF | Eigen-free CSC Jacobian |
| `F` | `vector<double>` | [dimF] | 미스매치 벡터 |
| `dx` | `vector<double>` | [dimF] | 풀이 결과 |
| `Va, Vm` | `vector<double>` | [n_bus] | 전압 각도·크기 |
| `V` | `vector<complex<double>>` | [n_bus] | 복소 전압 |
| `Ibus` | `vector<complex<double>>` | [n_bus] | 버스 전류 (Ybus·V 캐시) |
| `Sbus` | `vector<complex<double>>` | [n_bus] | 버스 전력 주입 (지정값) |
| `Ybus_indptr/indices/data` | `vector<int32_t/complex>` | — | Ybus CSR 사본 |

### 특이사항 — CSR→CSC 리맵이 왜 여기 있나

진짜 Jacobian *분석*(패턴/scatter map 생성)은 storage가 아니라
[`ops/jacobian/jacobian_analysis`](../../cpp/src/newton_solver/ops/jacobian/jacobian_analysis.hpp)에서
백엔드 무관(backend-agnostic)하게 1회 수행되어 `InitializeContext`(`ctx.J`, `ctx.maps`)로
전달된다. 그 출력은 **CSR 순서**다.

그런데 CPU 백엔드의 `CpuCscMatrix`는 **CSC(column-major)** 다(원래 `Eigen::SparseMatrix`가
column-major였던 유산). 그래서 `prepare()`가 분석 결과를 이 행렬의 value 순서에 맞추는
**적응(adaptation)** 을 한다 — analysis가 아니라 레이아웃 변환이다:
- CSR 패턴 → CSC sparsity 구축
- `csr_to_csc[csr_pos] = csc_pos` 테이블을 만들어 `mapJ**`/`diagJ**`를 CSC value 위치로 리맵
  (`CpuJacobianOpF64`가 `J.valuePtr()[csc_pos]`에 직접 scatter하기 때문)

CUDA 백엔드는 J를 CSR 그대로 보관하므로(아래) 이 리맵이 없다. 즉 CSR→CSC 변환은
**CSC를 쓰는 CPU 백엔드 고유**라 cpu_fp64_storage에만 존재한다.

기타:
- `has_cached_Ibus`: ibus/mismatch stage가 Ibus를 계산하면 true, Jacobian이 재사용.
  voltage update가 V를 변경하면 false로 무효화.
- `has_klu_symbolic`: KLU symbolic 분석 완료 여부.

---

## CUDA storage — 단일 템플릿 (CudaBatchedStorage)

**파일:** [cuda_batched_storage.hpp](../../cpp/src/newton_solver/storage/cuda/cuda_batched_storage.hpp)
· [cuda_batched_storage.cpp](../../cpp/src/newton_solver/storage/cuda/cuda_batched_storage.cpp)

세 CUDA 프로파일(FP32/FP64/Mixed)은 **버퍼 element 타입만** 달랐기 때문에, 하나의 템플릿
`CudaBatchedStorage<StateScalar, JacScalar>`로 통합하고 각 프로파일은 thin derived struct로 둔다.

```cpp
template <typename StateScalar, typename JacScalar>
struct CudaBatchedStorage { /* 모든 멤버/메서드 */ };

struct CudaFp32Storage  : CudaBatchedStorage<float,  float>  {};
struct CudaFp64Storage  : CudaBatchedStorage<double, double> {};
struct CudaMixedStorage : CudaBatchedStorage<double, float>  {};
```

| 프로파일 | StateScalar | JacScalar |
|----------|-------------|-----------|
| `CudaFp32Storage` | `float` | `float` |
| `CudaFp64Storage` | `double` | `double` |
| `CudaMixedStorage` | `double` | `float` |

- **StateScalar** — 물리 상태와 그로부터 파생되는 모든 것의 정밀도:
  Ybus 값, 전압(직교 `d_V_re/d_V_im`, 극형식 `d_Va/d_Vm`), `Sbus`, `Ibus`,
  미스매치 `d_F`와 케이스별 norm `d_normF`.
- **JacScalar** — cuDSS에 넘기는 선형해 객체의 정밀도: Jacobian 값 `d_J_values`,
  해/스텝 `d_dx`. Mixed는 상태는 FP64로 두되 Jacobian만 FP32로 조립해
  factorize/solve를 더 싸게 한다.
- 인덱스 버퍼(CSR 포인터/인덱스, scatter map, 버스 유형 리스트)는 항상 int32, 정밀도 무관.

derived struct를 쓰는 이유: op 헤더들이 `struct CudaFp32Storage;`처럼 **전방 선언**을 하므로
`using` 별칭은 충돌한다. thin derived struct는 전방 선언·오버로드 디스패치를 그대로 유지하면서
구현만 공유한다. 구현은 `cuda_batched_storage.cpp`에서 3개 조합으로 명시적 인스턴스화된다.

### batch-major 레이아웃

batch 케이스 `b`에 대해 (연속 메모리):

| 배열 | 위치 | 버퍼 |
|------|------|------|
| per-bus | `[b*n_bus + bus]` | `d_V_*`, `d_Va`, `d_Vm`, `d_Sbus_*`, `d_Ibus_*` |
| per-residual | `[b*dimF + row]` | `d_F`, `d_dx` |
| per-J-value | `[b*nnz_J + pos]` | `d_J_values` |
| per-case norm | `[b]` | `d_normF` |

Ybus *패턴*(indptr/indices/row, scatter map)은 batch 공통, Ybus *값*은
`ybus_values_batched`가 아니면 공통. 이 연속 layout이 cuDSS uniform-batch가 기대하는 형태라
**동일 descriptor가 B=1과 B>1을 모두 구동**한다([cuda_cudss.cpp](../../cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp)).

### device 버퍼 목록 (StateScalar=S, JacScalar=J)

| 버퍼 | 타입 | 크기 | 설명 |
|------|------|------|------|
| `d_Ybus_re/im` | `S` | [nnz_ybus] | Ybus 실수·허수부 (값 공통) |
| `d_Ybus_indptr/indices/row` | `int32` | — | Ybus CSR 구조 + edge kernel용 행번호 |
| `d_J_values` | `J` | [B·nnz_J] | Jacobian 값 (cuDSS uniform batch) |
| `d_J_row_ptr/col_idx` | `int32` | — | Jacobian CSR 구조 (batch 공통) |
| `d_F` | `S` | [B·dimF] | 미스매치 벡터 |
| `d_normF` | `S` | [B] | 케이스별 L∞ norm |
| `d_dx` | `J` | [B·dimF] | 선형해(스텝) |
| `d_Va, d_Vm` | `S` | [B·n_bus] | 전압 각도·크기 (authoritative) |
| `d_V_re/im` | `S` | [B·n_bus] | 직교 전압 cache |
| `d_Sbus_re/im` | `S` | [B·n_bus] | 지정 전력 |
| `d_Ibus_re/im` | `S` | [B·n_bus] | 버스 전류 (Ybus·V 캐시) |
| `d_mapJ11/12/21/22` | `int32` | [nnz_ybus] | 오프 대각 산포 맵 (batch 공통) |
| `d_diagJ11/12/21/22` | `int32` | [n_bus] | 대각 산포 맵 |
| `d_pvpq, d_pv, d_pq` | `int32` | — | 버스 유형 인덱스 |

### 정밀도별 차이 (요약)

- **FP32** — public I/O는 FP64지만 `upload`에서 float로 down-cast, `download`에서 up-cast.
  대형 ill-conditioned 계통에서는 수렴 반복이 늘거나 발산할 수 있다.
- **FP64** — 변환 없이 전 단계 FP64. 정확도 최우선. batch도 지원.
- **Mixed** — FP64 상태 + FP32 Jacobian. Ibus 커널은 FP64 Ybus/V로 FP64 Ibus 생성,
  Jacobian 커널은 FP64 입력을 읽어 FP32 J를 채운다. 최종 전압은 FP64 `Va/Vm`에서 재구성.

업로드/다운로드의 host↔device 변환은 device 커널([storage_convert.hpp](../../cpp/src/newton_solver/storage/cuda/storage_convert.hpp))로
처리한다: `launch_split_complex<S>`(interleaved complex→re/im), `launch_seed_state_from_v0<S>`
(V0→re/im/Va/Vm), `launch_pack_complex_to_double<S>`(re/im→interleaved, D2H용). 호스트
cast/trig 루프를 대체한다.

---

## DeviceBuffer\<T\>

**파일:** [cuda_utils.hpp](../../cpp/inc/utils/cuda_utils.hpp)

CUDA device 메모리의 RAII 래퍼.

```cpp
template <typename T>
class DeviceBuffer {
    void   resize(size_t count);                  // cudaMalloc
    void   assign(const T* src, size_t count);    // cudaMemcpy (H→D)
    void   copyTo(T* dst, size_t count) const;    // cudaMemcpy (D→H)
    void   memsetZero();                          // cudaMemset 0
    T*     data() const;
    size_t size() const;
    bool   empty() const;
};
```

복사 생성자·대입 연산자가 삭제되어 있어 실수로 device 메모리를 복사하는 것을 방지한다.
이동 연산자는 지원된다.

---

## Storage 선택 기준

| 상황 | 권장 Storage | 배치 |
|------|-------------|------|
| CPU 개발·디버깅 | `CpuFp64Storage` | B=1 |
| CUDA, 정확도 최우선 | `CudaFp64Storage` | 지원 |
| CUDA, 처리량 최우선 | `CudaMixedStorage` (또는 full-FP32 `CudaFp32Storage`) | 지원 |

- 배치(batch_size > 1)는 CUDA `CudaFp32Storage`/`CudaFp64Storage`/`CudaMixedStorage`가 지원한다
  (`CpuFp64Storage`는 single-case).
- 처리량은 보통 Mixed < FP64 (FP64 cuDSS factorization이 더 무겁다). FP32는 가장 싸지만
  대형 계통에서 수렴이 불안정할 수 있다.
- Mixed 모드는 Jacobian이 FP32이므로 수치적으로 민감한 계통(ill-conditioned Y)에서 수렴에
  더 많은 반복이 필요할 수 있다.
