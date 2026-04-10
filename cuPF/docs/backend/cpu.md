# CPU 백엔드

**헤더**: `inc/newton_solver/backend/cpu_backend.hpp`  
**내부 구현 헤더**: `src/newton_solver/backend/cpp/cpu_backend_impl.hpp`  
**구현 파일들**:

| 파일 | 내용 |
|------|------|
| `cpp/initialize.cpp` | 생성자, `analyze()`, `initialize()` |
| `cpp/compute_mismatch.cpp` | `computeMismatch()` |
| `cpp/update_jacobian.cpp` | `updateJacobian()`, `updateVoltage()`, `downloadV()` |
| `cpp/klu_solver.cpp` | `solveLinearSystem()` |

---

## 정밀도

**FP64 일관** — Ybus, Jacobian, 전압 상태 모두 `double` / `complex<double>`.

---

## Pimpl 구조

공개 헤더(`cpu_backend.hpp`)에 Eigen 타입이 노출되지 않도록 `Impl` 구조체로 분리.

```
cpu_backend.hpp          ← Eigen 없음 (공개 헤더)
    └── Impl (cpu_backend_impl.hpp)
            ├── SpCx Ybus        (Eigen CSC complex)
            ├── JacobianMatrix J (Eigen CSC double)
            ├── JacobianMaps maps (CSC 위치로 리매핑된 버전)
            ├── ybus_indptr/indices/data (CSR 순서 순회용)
            ├── VXcd V, Sbus / VXd Vm, Va
            └── SparseLU lu
```

---

## analyze()

```
JacobianStructure(CSR) 수신
    ↓
Eigen 트리플렛으로 J(CSC) 재구성 + makeCompressed
    ↓
CSR→CSC 위치 순열 계산 (csr_to_csc[csr_k] = csc_k)
    ↓
JacobianMaps 전체 재매핑: maps.mapJ**[k] = csr_to_csc[원래값]
    ↓
ybus CSR 배열 복사 저장 (updateJacobian CSR 순회용)
    ↓
lu.analyzePattern(J)  ← 심볼릭 인수분해 (COLAMD 열 재배열)
```

**핵심**: JacobianBuilder가 CSR 위치로 인덱스를 만들지만, Eigen SparseLU는 CSC 값 배열을 요구한다. `analyze()`에서 한 번 리매핑해두면 이후 `updateJacobian()`은 추가 변환 없이 동작한다.

---

## computeMismatch()

```
Ibus = Ybus * V  (Eigen SpMV, CSC → FP64 복소수)
mis[i] = V[i] * conj(Ibus[i]) - Sbus[i]
F 패킹: [Re(pv), Re(pq), Im(pq)]
normF = max|F_i|  (std::abs, inf-norm)
```

---

## updateJacobian()

Ybus를 CSR 순서로 순회 (`ybus_indptr/indices/data`). `t`가 `JacobianMaps`의 CSR 인덱스와 일치하게 된다.

```
Ibus = Ybus * V  (SpMV)
Vnorm = V / |V|  (단위 페이저)

t = 0
for row in 0..n_bus:
  for k in ybus_indptr[row]..ybus_indptr[row+1]:
    va = -j * V[Y_i] * conj(Y[k] * V[Y_j])
    vm =  V[Y_i] * conj(Y[k] * Vnorm[Y_j])
    J_val[maps.mapJ11[t]] = va.real  ← CSC 위치로 직접 쓰기
    ...
    t++

for bus in 0..n_bus:  (대각 기여)
    J_val[maps.diagJ11[bus]] += ...
```

---

## solveLinearSystem()

```
lu.factorize(J)       ← 수치 인수분해 (심볼릭은 analyze에서 완료)
dx = lu.solve(-F)
```

---

## updateVoltage()

```
Va[pv] += dx[0:n_pv]
Va[pq] += dx[n_pv:n_pv+n_pq]
Vm[pq] += dx[n_pv+n_pq:dimF]
V[bus] = polar(Vm[bus], Va[bus])  ← 복소 전압 재구성
```
