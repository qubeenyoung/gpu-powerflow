# storage — 스토리지 레이어 설계

Storage는 한 실행 프로파일에 필요한 **메모리 버퍼**와 **라이브러리 핸들**을 소유하는 레이어다.
Op는 `IStorage`를 통해 버퍼에 접근하며 계산 로직만 구현한다.

---

## IStorage 인터페이스

**파일:** [op_interfaces.hpp](../../cpp/inc/newton_solver/ops/op_interfaces.hpp)

```cpp
class IStorage {
    virtual BackendKind   backend() const = 0;
    virtual ComputePolicy compute()  const = 0;
    virtual void prepare(const AnalyzeContext& ctx) = 0;  // analyze 단계
    virtual void upload(const SolveContext& ctx)    = 0;  // solve 시작 시
    virtual void download_result(NRResultF64& result) const = 0;
    virtual void download_batch_result(NRBatchResultF64& result) const;
};
```

### prepare()

`analyze()` 단계에서 **한 번만** 호출된다. 다음을 수행한다.
- 위상 메타데이터(n_bus, n_pvpq, n_pq, dimF) 계산
- JacobianMaps, JacobianStructure 복사
- 버퍼 할당 및 초기화
- Jacobian CSC/CSR 행렬 구성 (CPU) 또는 device 업로드 (CUDA)

### upload()

`solve()` 시작 시마다 호출된다. Ybus 값, Sbus, V0를 device로 올린다.
Ybus의 희소 구조가 `analyze()` 당시와 같은지 검증한다.
기본 실행 모델은 batch-major layout이며, single-case solve는 `batch_size=1`이다.

### download_result() / download_batch_result()

NR 루프 완료 후 device의 V 벡터를 host result로 복사한다.
batch-aware storage는 `download_batch_result()`를 override하고, 기본 구현은 `B=1` wrapper다.

---

## CpuFp64Storage

**파일:** [cpu_fp64_storage.hpp](../../cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.hpp)

CPU FP64 경로의 host-side 버퍼와 Eigen/KLU 상태를 관리한다.

### 버퍼 레이아웃

| 버퍼 | 타입 | 크기 | 설명 |
|------|------|------|------|
| `Ybus` | `Eigen::SparseMatrix<complex<double>, ColMajor>` | n_bus × n_bus | Eigen CSC Ybus |
| `J` | `Eigen::SparseMatrix<double, ColMajor>` | dimF × dimF | Eigen CSC Jacobian |
| `lu` | `Eigen::KLU<J>` | — | KLU factorizer |
| `F` | `vector<double>` | [dimF] | 미스매치 벡터 |
| `dx` | `vector<double>` | [dimF] | 풀이 결과 |
| `Va, Vm` | `vector<double>` | [n_bus] | 전압 각도·크기 |
| `V` | `vector<complex<double>>` | [n_bus] | 복소 전압 |
| `Ibus` | `vector<complex<double>>` | [n_bus] | 버스 전류 (Ybus·V 캐시) |
| `Sbus` | `vector<complex<double>>` | [n_bus] | 버스 전력 주입 (지정값) |
| `Ybus_indptr/indices/data` | `vector<int32_t/complex>` | — | Ybus CSR 사본 |

### 특이사항

- Eigen은 내부적으로 CSC를 사용하지만 JacobianBuilder는 CSR 맵을 생성한다.
  `prepare()`에서 CSR→CSC 리맵 테이블(`csr_to_csc`)을 만들어 `mapJ**`를 CSC 위치로 변환한다.
- `has_cached_Ibus`: MismatchOp이 Ibus를 계산하면 true로 설정, JacobianOp이 재사용.
  VoltageUpdateOp이 V를 변경하면 false로 무효화.
- `has_klu_symbolic`: KLU symbolic 분석 완료 여부.

---

## CudaFp64Storage

**파일:** [cuda_fp64_storage.hpp](../../cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.hpp)

CUDA FP64 경로의 device 버퍼를 관리한다. 모든 값이 FP64.

### 정밀도 레이아웃

```
PublicScalar   = double
VoltageScalar  = double
JacobianScalar = double
SolveScalar    = double
```

### device 버퍼 목록

| 버퍼 | 타입 | 크기 | 설명 |
|------|------|------|------|
| `d_Ybus_re/im` | `DeviceBuffer<double>` | [nnz_Y] | Ybus 실수·허수부 |
| `d_Ybus_indptr` | `DeviceBuffer<int32_t>` | [n_bus+1] | Ybus CSR 행 포인터 |
| `d_Ybus_indices` | `DeviceBuffer<int32_t>` | [nnz_Y] | Ybus CSR 열 인덱스 |
| `d_Y_row` | `DeviceBuffer<int32_t>` | [nnz_Y] | Ybus CSR 행 번호 (edge kernel용) |
| `d_J_values` | `DeviceBuffer<double>` | [nnz_J] | Jacobian 값 |
| `d_J_row_ptr/col_idx` | `DeviceBuffer<int32_t>` | — | Jacobian CSR 구조 |
| `d_F` | `DeviceBuffer<double>` | [dimF] | 미스매치 벡터 |
| `d_dx` | `DeviceBuffer<double>` | [dimF] | 풀이 결과 |
| `d_Va, d_Vm` | `DeviceBuffer<double>` | [n_bus] | 전압 각도·크기 |
| `d_V_re/im` | `DeviceBuffer<double>` | [n_bus] | 복소 전압 실수·허수부 |
| `d_Sbus_re/im` | `DeviceBuffer<double>` | [n_bus] | Sbus 실수·허수부 |
| `d_mapJ11/12/21/22` | `DeviceBuffer<int32_t>` | [nnz_Y] | 오프 대각 산포 맵 |
| `d_diagJ11/12/21/22` | `DeviceBuffer<int32_t>` | [n_bus] | 대각 산포 맵 |
| `d_pvpq, d_pv, d_pq` | `DeviceBuffer<int32_t>` | [n_pvpq/n_pv/n_pq] | 버스 유형 인덱스 |

---

## CudaMixedStorage

**파일:** [cuda_mixed_storage.hpp](../../cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.hpp)

Mixed 정밀도 경로. 전압 상태(`Va/Vm`), derived cache(`V_re/V_im`), `Sbus`, `Ibus`, `F`는
FP64로 유지하고, `Ybus/J/dx`는 FP32로 둔다. Jacobian 커널은 FP64 `V/Ibus` 입력을
load한 뒤 내부 산술만 FP32로 수행한다.

### 정밀도 레이아웃

```
PublicScalar   = double
Va/Vm          = double  ← authoritative state
V cache        = double  ← derived V_re/V_im
Ybus           = float
Ibus           = double
Sbus           = double
F              = double
JacobianScalar = float
SolveScalar    = float
```

### FP64 vs FP32 버퍼 구분

| 버퍼 | 타입 | 크기 | 설명 |
|------|------|------|------|
| `d_Ybus_re/im` | `float` | [nnz_Y] 또는 [B·nnz_Y] | Ybus 값, 구조는 batch 공통 |
| `d_Sbus_re/im` | `double` | [B·n_bus] | 지정 전력 |
| `d_Ibus_re/im` | `double` | [B·n_bus] | custom CSR mismatch가 만든 재사용 전류 |
| `d_Va/Vm` | `double` | [B·n_bus] | authoritative 전압 상태 |
| `d_V_re/im` | `double` | [B·n_bus] | mismatch 입력, Jacobian FP32 내부 계산의 FP64 경계 입력 |
| `d_F` | `double` | [B·dimF] | 수렴 판정용 미스매치 |
| `d_normF` | `double` | [B] | batch별 L∞ norm |
| `d_J_values` | `float` | [B·nnz_J] | cuDSS uniform batch CSR values |
| `d_dx` | `float` | [B·dimF] | cuDSS FP32 solution |

Jacobian structure와 map 배열은 batch 공통이고, 값 버퍼만 batch-major다.
최종 `NRResultF64`/`NRBatchResultF64` 전압은 FP64 `Va/Vm`에서 재구성한다.

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

| 상황 | 권장 Storage |
|------|-------------|
| CPU 개발·디버깅 | `CpuFp64Storage` |
| CUDA, 정확도 최우선 | `CudaFp64Storage` |
| CUDA, 처리량 최우선 | `CudaMixedStorage` |

Mixed 모드는 Jacobian이 FP32이므로 수치적으로 민감한 계통(ill-conditioned Y)에서 수렴에 더 많은 반복이 필요할 수 있다.
