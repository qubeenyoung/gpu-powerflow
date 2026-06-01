# ops/voltage_update — Newton step 적용과 전압 재구성

NR 반복의 마지막 단계. 선형 solve가 낸 스텝 `dx`를 전압 상태에 적용하고, 다음 반복의
Ibus/mismatch가 읽을 직교 전압을 다시 만든다.

**소스**: `cpp/src/newton_solver/ops/voltage_update/` — `cuda_voltage_update.cu`(디스패치),
`voltage_update_kernels.hpp`(커널+런처), `cpu_voltage_update.{cpp,hpp}`.

---

## 두 단계 (디바이스)

1. **apply** (`apply_voltage_update_kernel`): Newton 업데이트 `x ← x − dx`. `J·dx = F`를
   풀었으므로 상태에서 dx를 뺀다. dx는 `dimF` 레이아웃으로 분절돼 각 성분이 갈 곳이 정해짐:
   ```
   [0, n_pv)            → Va @ pv 버스
   [n_pv, n_pv+n_pq)    → Va @ pq 버스
   [n_pv+n_pq, dimF)    → Vm @ pq 버스       (dimF = n_pv + 2·n_pq)
   ```
   dx 정밀도가 상태보다 낮을 수 있어(Mixed: float dx → double 상태) static_cast로 변환.
2. **reconstruct** (`reconstruct_voltage_kernel`): 갱신된 극형식에서 직교 전압 재구성
   `V = Vm·(cos Va + j·sin Va)` (정밀도별 `sincos`/`sincosf`).

배치는 스레드가 `[batch·dimF]`(스텝)와 `[batch·n_bus]`(버스)를 stride 처리해 B=1/B>1
동일. 디스패치(`cuda_voltage_update.cu`)는 세 프로파일이 템플릿 헬퍼
`run_voltage_update<StateScalar, DxScalar, Storage>`로 본문을 공유한다
(FP64=double/double, FP32=float/float, Mixed=double/float).

## CPU (`cpu_voltage_update.cpp`)

호스트에서 동일한 apply+reconstruct(`CpuVoltageUpdateOp`).
