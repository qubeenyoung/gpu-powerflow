# ops/ibus — 버스 전류 Ibus = Ybus · V

NR 한 반복의 첫 단계. 복소 희소 행렬-벡터 곱(SpMV)으로 버스 전류를 구한다. 결과는
mismatch(`S(V) = V·conj(Ibus)`)와 Jacobian 대각 항에 쓰인다.

**소스**: `cpp/src/newton_solver/ops/ibus/` (`compute_ibus.cu`, `compute_ibus.cpp`(CPU),
`compute_ibus.hpp`).

---

## 수학

Ybus는 복소 CSR(n_bus×n_bus). 각 버스 i에 대해

```
Ibus[i] = Σ_{(i,col)∈Ybus}  Ybus[i,col] · V[col]
```

복소 곱은 실수 산술로: `(yr+jyi)(vr+jvi) = (yr·vr − yi·vi) + j(yr·vi + yi·vr)`.

## GPU 커널 (`compute_ibus.cu`)

**scalar SpMV**: 한 스레드가 한 `(batch case, row)`를 맡아 그 행의 nonzero를 순차 누적.
전력망은 매우 희소(행당 ~2–4 nnz)라 행당 스레드 하나가 모든 스레드를 생산적으로 쓰고,
벡터화(warp-per-row) 커널의 idle lane/shuffle reduction을 피한다.

- thread→(batch,row) 매핑은 batch-major(`base = batch·n_bus`).
- 3중 정밀도 템플릿: `YbusScalar`(Ybus), `StateScalar`(V/Ibus), `AccumScalar`(누산).
  FP64/Mixed는 double 누산, FP32는 float.
- Ybus 값은 배치 공통(`y_re[k]`)이라 `ybus_values_batched`(케이스별 admittance)는 이
  커널에서 미지원 — `require_ibus_ready`가 거부한다.

세 공개 런처(`launch_compute_ibus(CudaFp64/Fp32/MixedStorage&)`)는 검증 헬퍼
`require_ibus_ready`를 공유하고 프로파일별 정밀도만 지정한다. Mixed는 상태가 FP64라
Ibus는 FP64와 동일하게 돈다.

## CPU (`compute_ibus.cpp`)

호스트에서 같은 SpMV를 수행(`CpuIbusOp`). 결과 Ibus는 캐시되어 Jacobian 대각 항
계산에 재사용된다([ops/jacobian.md](jacobian.md)).
