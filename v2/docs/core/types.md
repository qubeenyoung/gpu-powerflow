# 공용 타입

두 헤더에 나뉘어 있다.

- `inc/newton_solver/core/newton_solver_types.hpp` — 공개 API 타입
- `inc/newton_solver/core/jacobian_types.hpp` — Jacobian 내부 타입

---

## newton_solver_types.hpp

### `CSRView<T>`

```cpp
template<typename T, typename IndexType = int32_t>
struct CSRView {
    const IndexType* indptr;   // row_ptr,  size = rows + 1
    const IndexType* indices;  // col_idx,  size = nnz
    const T*         data;     // values,   size = nnz
    IndexType rows, cols, nnz;
};

using YbusView = CSRView<std::complex<double>>;
```

**용도**: 외부 메모리를 zero-copy로 참조하는 비소유 뷰. numpy/torch 포인터를 그대로 감싼다.

---

### `NRConfig`

```cpp
struct NRConfig {
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
};
```

**용도**: NR 루프의 수렴 조건과 최대 반복 횟수.

---

### `NRResult`

```cpp
struct NRResult {
    std::vector<std::complex<double>> V;  // 최종 버스 전압 (n_bus)
    int32_t iterations     = 0;
    double  final_mismatch = 0.0;
    bool    converged      = false;
};
```

**용도**: `solve()` 출력. 전압은 백엔드 정밀도와 무관하게 항상 FP64 복소수.

---

### `BackendKind` / `NewtonOptions`

```cpp
enum class BackendKind { CPU, CUDA };

struct NewtonOptions {
    BackendKind         backend  = BackendKind::CPU;
    JacobianBuilderType jacobian = JacobianBuilderType::EdgeBased;
};
```

**용도**: `NewtonSolver` 생성 시 백엔드와 Jacobian 빌더 방식을 선택.

---

## jacobian_types.hpp

### `JacobianBuilderType`

```cpp
enum class JacobianBuilderType { EdgeBased, VertexBased };
```

- `EdgeBased` — Ybus 비제로 순회 (현재 구현, v1 방식)
- `VertexBased` — 미구현 예약

---

### `JacobianStructure`

```cpp
struct JacobianStructure {
    std::vector<int32_t> row_ptr;  // size = dim + 1
    std::vector<int32_t> col_idx;  // size = nnz, 각 행 내 정렬됨
    int32_t dim = 0;
    int32_t nnz = 0;
};
```

**용도**: `JacobianBuilder`가 생성하는 CSR 희소 패턴. Eigen 타입이 없어 백엔드에 독립적.
- CPU 백엔드 → 내부에서 Eigen CSC로 변환
- CUDA 백엔드 → GPU에 직접 업로드

---

### `JacobianMaps`

```cpp
struct JacobianMaps {
    std::vector<int32_t> mapJ11, mapJ12, mapJ21, mapJ22;   // Ybus 비제로 k → J값 인덱스
    std::vector<int32_t> diagJ11, diagJ12, diagJ21, diagJ22; // 버스 i → J 대각 인덱스
    std::vector<int32_t> pvpq;   // [pv..., pq...] 연결 버스 인덱스
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
};
```

**용도**: `analyze()`에서 한 번 생성, NR 루프 전 iteration에서 재사용.

Jacobian 블록 구조:
```
J = [ J11  J12 ] = [ dP/dθ    dP/d|V| ]
    [ J21  J22 ]   [ dQ/dθ    dQ/d|V| ]
```

`mapJ11[k]` — Ybus의 k번째 비제로가 J11에서 기여하는 값 배열 인덱스 (없으면 -1).
인덱스는 CSR 위치 공간 기준. CPU 백엔드는 `analyze()`에서 CSC 위치로 리매핑한다.
