# cuPF

GPU 가속 **Newton–Raphson 전력조류(power flow)** 솔버. 미분 가능(differentiable)하여
adjoint(역전파)로 부하 gradient를 얻고, 시나리오 **배치(batch)**를 한 번에 푼다.
CPU(SuiteSparse KLU/UMFPACK)와 CUDA(cuDSS) 백엔드, FP64/FP32/Mixed 정밀도 프로파일,
그리고 PyTorch zero-copy 연동을 지원한다.

## 주요 특징

- **NR 전력조류**: `Ibus = Ybus·V` → mismatch `F = V·conj(Ibus) − Sbus` → Jacobian
  `J·dx = F` → 전압 업데이트.
- **배치**: batch-major 레이아웃 + cuDSS uniform-batch로 다수 시나리오를 한 문제로 풀이
  (CUDA FP32/Mixed/FP64).
- **미분 가능**: 수렴 상태에서 `J^T λ = dL/dx`를 풀어 부하에 대한 gradient 계산
  (implicit function theorem).
- **정밀도 프로파일**: FP64(정확도), Mixed(FP64 상태 + FP32 Jacobian, 처리량 권장),
  FP32(최저 비용).
- **PyTorch 연동**: device 텐서를 호스트 복사 없이 현재 CUDA 스트림에서 forward/backward.

## 요구 사항

- C++17, CMake ≥ 3.22
- CUDA(예: 12.x) + **cuDSS** — GPU 백엔드(`WITH_CUDA=ON`)
- SuiteSparse **KLU/UMFPACK** — CPU 백엔드
- pybind11 — 파이썬 바인딩, (선택) PyTorch — `CUPF_WITH_TORCH=ON`

## 빌드

```bash
cmake -S cuPF -B build \
  -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_WITH_TORCH=ON -DENABLE_TIMING=ON \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build build -j
```

주요 플래그: `WITH_CUDA`, `BUILD_PYTHON_BINDINGS`, `CUPF_WITH_TORCH`, `ENABLE_TIMING`,
`CUPF_ENABLE_CUSTOM_SOLVER`. 자세히는 [docs/system_architecture.md](docs/system_architecture.md) §6.

## 빠른 사용 (Python)

```python
import numpy as np, _cupf as c

opts = c.NewtonOptions()
opts.backend = c.BackendKind.CUDA
opts.compute = c.ComputePolicy.Mixed          # FP64 / FP32 / Mixed

s = c.NewtonSolver(opts)
s.initialize(indptr, indices, data, n_bus, n_bus, pv, pq)   # CSR Ybus + PV/PQ

# 단일
r = s.solve(indptr, indices, data, n_bus, n_bus, sbus, v0, pv, pq)
V = r.V_numpy

# 배치 [B, n_bus]
rb = s.solve_batch(indptr, indices, data, n_bus, n_bus, sbus_B, v0_B, pv, pq)
Vb = rb.V_numpy                                # [B, n_bus]
```

PyTorch zero-copy(forward/backward)와 numpy adjoint API는
[docs/newton_solver.md](docs/newton_solver.md) 참조.

## 문서

설계 문서는 [`docs/`](docs/)에 컴포넌트별로 있다. 시작점:
[**docs/system_architecture.md**](docs/system_architecture.md) — 전체 구조·데이터 흐름·
실행 프로파일·배치 모델·빌드.

| 영역 | 문서 |
|------|------|
| 구조 개요 | [system_architecture](docs/system_architecture.md) |
| 솔버 API | [newton_solver](docs/newton_solver.md) |
| core 내부 | [core](docs/core.md) |
| storage | [storage](docs/storage.md) |
| ops | [ibus](docs/ops/ibus.md) · [mismatch](docs/ops/mismatch.md) · [jacobian](docs/ops/jacobian.md) · [linear_solve](docs/ops/linear_solve.md) · [voltage_update](docs/ops/voltage_update.md) |
| 바인딩 | [bindings](docs/bindings.md) |
| 유틸 | [utils](docs/utils.md) |

## 소스 레이아웃

```
cpp/inc/   공개 헤더 (newton_solver, utils)
cpp/src/   newton_solver/{core, ops/*, storage/{cpu,cuda}}
bindings/  pybind_cupf.cpp, torch_cupf_extension.cpp
docs/      설계 문서
tests/     테스트·평가기
```
