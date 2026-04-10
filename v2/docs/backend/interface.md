# INewtonSolverBackend

**파일**: `inc/newton_solver/backend/i_backend.hpp`

모든 백엔드가 구현해야 하는 순수 가상 인터페이스. `NewtonSolver`는 이 인터페이스만 보고 NR 루프를 진행한다.

---

## 메서드 목록

| 메서드 | 호출 시점 | 역할 |
|--------|-----------|------|
| `analyze()` | solve 전 1회 | 위상 분석 결과 수신, 메모리/핸들 초기화 |
| `initialize()` | solve 시작 시 1회 | V0, Sbus 업로드 |
| `computeMismatch()` | 매 iteration | F 계산, normF 반환 |
| `updateJacobian()` | 매 iteration | J 값 채우기 |
| `solveLinearSystem()` | 매 iteration | J·dx = −F 풀기 |
| `updateVoltage()` | 매 iteration | Va, Vm 보정 및 V 재구성 |
| `downloadV()` | solve 종료 시 | 최종 전압 복사 |

---

## 시그니처

```cpp
virtual void analyze(const YbusView&          ybus,
                     const JacobianMaps&       maps,
                     const JacobianStructure&  J,
                     int32_t                   n_bus) = 0;

virtual void initialize(const YbusView&             ybus,
                        const std::complex<double>* sbus,
                        const std::complex<double>* V0) = 0;

virtual void computeMismatch(const int32_t* pv, int32_t n_pv,
                             const int32_t* pq, int32_t n_pq,
                             double* F, double& normF) = 0;

virtual void updateJacobian() = 0;

virtual void solveLinearSystem(const double* F, double* dx) = 0;

virtual void updateVoltage(const double*  dx,
                           const int32_t* pv, int32_t n_pv,
                           const int32_t* pq, int32_t n_pq) = 0;

virtual void downloadV(std::complex<double>* V_out, int32_t n_bus) = 0;
```

---

## 설계 의도

- **Eigen 타입 없음**: `analyze()`가 `JacobianStructure`(CSR, plain struct)를 받아 CUDA 헤더 없이도 인터페이스를 include 가능
- **pv/pq 반복 전달**: `computeMismatch`와 `updateVoltage`가 매번 pv/pq를 받지만, CUDA 백엔드는 `analyze()`에서 한 번 GPU에 올리고 이후 무시한다
- **새 백엔드 추가**: 이 인터페이스를 구현하는 클래스를 `src/backend/` 아래 추가하고 `make_backend()`에 등록하면 됨

---

## F 벡터 레이아웃

```
F[0       : n_pv]         = Re(mis[pv])   — PV 버스 유효전력 불일치
F[n_pv    : n_pv+n_pq]   = Re(mis[pq])   — PQ 버스 유효전력 불일치
F[n_pv+n_pq : dimF]      = Im(mis[pq])   — PQ 버스 무효전력 불일치

dimF = n_pv + 2*n_pq
```
