# JacobianBuilder

**파일**: `inc/newton_solver/core/jacobian_builder.hpp`  
**구현**: `src/newton_solver/core/jacobian_builder.cpp`

Ybus 희소 패턴을 분석해 NR 루프가 매 iteration마다 재사용할 매핑 테이블을 생성한다. `analyze()`는 `NewtonSolver::analyze()` 내부에서 한 번 호출된다.

---

## 사용 흐름

```cpp
JacobianBuilder builder(JacobianBuilderType::EdgeBased);
auto [maps, J] = builder.analyze(ybus, pv, n_pv, pq, n_pq);
// maps: JacobianMaps (매핑 테이블)
// J:    JacobianStructure (CSR 희소 패턴)
```

---

## EdgeBased 알고리즘

내부적으로 Eigen 트리플렛으로 희소 패턴을 구성한 뒤 CSR로 변환한다.

```
1. pvpq 목록 구성: pvpq = [pv..., pq...]
2. bus → Jacobian 행/열 인덱스 룩업 테이블 생성
3. Ybus 비제로 순회 → 4개 블록(J11/J12/J21/J22)에 트리플렛 추가
   (대각 포함)
4. Eigen setFromTriplets → 중복 제거, 압축
5. Eigen CSC → JacobianStructure (CSR) 변환
   - CSC를 열 순서로 순회하면 각 행의 col_idx가 자동 정렬됨
6. CSR 위치로 mapJ**, diagJ** 인덱스 계산 (이진 탐색)
```

**핵심**: 단계 6에서 생성된 인덱스는 CSR 위치 기준. 백엔드는 이를 그대로 쓰거나 (CUDA), 내부적으로 CSC로 리매핑한다 (CPU).

---

## 왜 Eigen을 내부적으로 쓰는가?

트리플렛의 중복 `(row, col)` 쌍을 병합하는 작업(대각/비대각 항이 같은 위치에 기여할 때)을 `setFromTriplets`이 자동 처리한다. 이 편의를 위해 분석 단계에서만 Eigen을 사용하고, **출력(`JacobianStructure`)은 Eigen 타입이 없다**.

---

## VertexBased

현재 `VertexBased`는 **희소 패턴/맵 생성은 EdgeBased 결과를 재사용**하고, CUDA 백엔드의 수치 채우기 단계만 row-owner(vertex-based) 커널로 전환한다.

핵심 차이:

- `mapJ**[k]`, `diagJ**[bus]` 형식은 그대로 유지
- `maps.pvpq`가 활성 bus 목록 역할을 겸함
- CUDA `updateJacobian()`은 활성 bus당 warp 1개가 Ybus CSR row를 순회하며
  off-diagonal은 direct write, diagonal은 row-local reduction 후 1회 write

즉, 현재 VertexBased는 **분석 포맷을 새로 만들기보다 실행 전략을 바꾸는 1차 구현**이다.
