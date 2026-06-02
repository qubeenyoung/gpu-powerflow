# gpu-powerflow

미분 가능(differentiable) **GPU 가속 전력조류(power flow)** 솔버 **cuPF**와, 그 데이터
준비·벤치마크·튜토리얼·기준선(baseline)·실험 도구를 모은 작업 공간이다.

핵심은 [`cuPF/`](cuPF/) — 희소 `Ybus`를 입력으로 Newton–Raphson(NR) 전력조류를 GPU에서
풀고, adjoint(역전파)로 부하 gradient를 계산하며, 시나리오 **배치**를 한 번에 처리하는
C++/CUDA 솔버다. 나머지 디렉터리는 cuPF를 **둘러싼 생태계**(케이스 변환, 벤치마크, MATLAB/
pandapower 기준선, 커스텀 선형 솔버, 튜토리얼, 컨테이너)를 제공한다.

---

## 저장소 구성

```text
.
├── cuPF/                 # 핵심 솔버 (C++/CUDA + Python 바인딩 + 설계 문서)
├── python/               # 데이터 준비 · 벤치마크 하니스 · 튜토리얼 노트북
├── custom_linear_solver/ # cuDSS형 래퍼 (CUDA multifrontal 직접 솔버, 선택적 백엔드)
├── benchmark/            # 빌드/실행 하니스, 결과(results/), 분석 docs
├── matlab/               # MATPOWER runpf 기준선 실행 스크립트
├── Dockerfile, requirements.txt
```

---

## 1. cuPF — 핵심 솔버

GPU NR 전력조류 솔버. 자세한 소개·빌드·사용은 [`cuPF/README.md`](cuPF/README.md), 설계는
[`cuPF/docs/system_architecture.md`](cuPF/docs/system_architecture.md).

- **단계 파이프라인**: `Ibus = Ybus·V` → mismatch `F = V·conj(Ibus) − Sbus` → Jacobian
  `J·dx = F` → 전압 업데이트.
- **백엔드/정밀도**: CPU(SuiteSparse KLU/UMFPACK FP64), CUDA(cuDSS) FP64/FP32/Mixed.
- **배치**: batch-major + cuDSS uniform-batch로 다수 시나리오 동시 풀이(CUDA 세 프로파일).
- **미분 가능**: 수렴 상태에서 `J^T λ = dL/dx`를 풀어 부하 gradient(implicit function theorem).
- **PyTorch zero-copy**: device 텐서를 호스트 복사 없이 forward/backward.

---

## 2. Python / Torch 연동

파이썬 모듈은 `_cupf`(numpy API + 선택적 torch zero-copy 확장). 진입점 요약:

| 진입점 | 메서드 | 입력 | dtype | 배치 | 비고 |
|--------|--------|------|-------|------|------|
| numpy 단일 | `solve` | CSR Ybus + Sbus/V0 (FP64 complex) | float64 | B=1 | 단일 시나리오 |
| numpy 배치 | `solve_batch` | `[B, n_bus]` Sbus/V0 | float64 | ✅ | 다중 시나리오 |
| numpy adjoint | `solve_adjoint` | dL/dVa, dL/dVm (1D 또는 `[B,n_bus]`) | float64 | ✅ | gradient(역전파) |
| torch forward | `solve_torch` | CUDA 텐서 `[B,n_bus]`(load/out)·`[n_bus]`(base·V0) | FP64→float64, FP32·Mixed→float32 | ✅ | zero-copy, 현재 CUDA 스트림 |
| torch backward | `solve_adjoint_torch` | CUDA grad 텐서 `[B,n_bus]` | (forward와 동일) | ✅ | zero-copy autograd |

- torch 경로는 호스트 복사 없이 torch의 **현재 CUDA 스트림**에서 실행
  (`CUPF_WITH_TORCH=ON` 빌드 필요). Mixed는 비-torch의 FP64 complex 입력과 달리 **float32
  텐서**를 받는다.
- `AdjointResult.lambda`는 파이썬 예약어 → `lambda_` / `lambda_numpy`로 접근.

```python
import numpy as np, _cupf as c
opts = c.NewtonOptions(); opts.backend = c.BackendKind.CUDA; opts.compute = c.ComputePolicy.Mixed
s = c.NewtonSolver(opts)
s.initialize(indptr, indices, data, n_bus, n_bus, pv, pq)   # CSR Ybus + PV/PQ
rb = s.solve_batch(indptr, indices, data, n_bus, n_bus, sbus_B, v0_B, pv, pq)
Vb = rb.V_numpy                                             # [B, n_bus]
```

API 상세는 [`cuPF/docs/newton_solver.md`](cuPF/docs/newton_solver.md),
바인딩은 [`cuPF/docs/bindings.md`](cuPF/docs/bindings.md).

---

## 3. 성능 지표 (대표 실측)

아래는 한 환경(**NVIDIA RTX 3090, CUDA 12.8, Release, `ENABLE_TIMING=ON`**)에서의
대표값이다. 절대 수치는 하드웨어/케이스에 민감하므로, 재현 명령으로 직접 측정하는 것을
권장한다.

| 항목 | 케이스 | 대표값 |
|------|--------|--------|
| FP64 단일 solve 수렴 | case9241pegase | 7 iter, `mismatch_norm` ≈ 98 µs |
| 배치 처리량 (B=64, solve_total) | case9241pegase | Mixed ≈ 101 ms vs FP64 ≈ 150 ms |
| 정확도 `max\|V − Vref\|` (FP64/Mixed) | case1354/2869/9241 | ~1e-13 |
| FP64 배치 정확성 (batch vs 단일해) | case9241pegase | `max\|ΔV\|` ~1e-12 |
| FP64 batched adjoint (grad vs 단일해) | case2869pegase | ~1e-11 |

> 처리량은 보통 **Mixed < FP64**(FP64 cuDSS 분해가 무거움), FP32가 가장 싸지만 대형
> ill-conditioned 계통에서 수렴 불안정. 단계별 정밀 측정은 `ENABLE_TIMING=ON` 빌드에서.

재현:

```bash
# C++ 마이크로 벤치(단계별 µs): <case_dir> <fp32|mixed|fp64> <B1,B2,...> [repeats]
./build/cupf_batch_bench <case_dir> mixed 1,16,64 5
# 파이썬 엔드투엔드 벤치 + 집계
python3 -m python.tests.run_benchmark --cases case9241pegase --repeats 5
python3 -m python.tests.aggregate_results benchmark/results/<run-name>
```

---

## 4. python/ — 준비·벤치마크·튜토리얼

세 책임으로 분리(자세히는 [`python/README.md`](python/README.md)):

- **`python/prepare/`** — MATPOWER `.m` 파싱 → 기준 PF 풀이 → cuPF 덤프 작성(`prepare.py`),
  Newton 선형계 생성(`convert_linear_system.py`), `.m`→`.mat` 변환(`convert_m_to_mat.py`).
- **`python/tests/`** — 단일 벤치마크 패키지: 케이스 로딩(`matpower_data.py`),
  기준선(`run_pypower.py`, `run_matpower.py`), cuPF 러너(`run_cupf_pybind.py`,
  `run_cupf_native.py`), 오케스트레이션(`run_benchmark.py`), 집계(`aggregate_results.py`).
- **`python/tutorial/`** — 입문용 라이브 노트북 6종([README](python/tutorial/README.md)):
  ① 전력계통 기초·case9·Ybus ② MATPOWER/pandapower 기준선·병목 ③ cuPF CPU 경로
  ④ cuPF GPU 경로(cuDSS·Edge/EdgeAtomic/VertexWarp·custom) ⑤ pybind `solve_batch`·
  **Torch autograd** ⑥ 남은 병목·향후 연구.

> 사용자용 파이썬 패키지 코드는 [`cuPF/python/`](cuPF/python/)(`cupf`), 벤치마크 평가기는
> `python/tests/`에 둔다(분리 유지).

---

## 5. custom_linear_solver/

- **[`custom_linear_solver/`](custom_linear_solver/README.md)**: cuDSS와 동일한 phase
  모델(`set_data → analyze → factorize → solve`)을 따르는 CUDA multifrontal 직접 솔버
  래퍼(GPU multifrontal 커널 + 심볼릭 분석 + METIS nested-dissection 정렬 + 최소 상태).
  cuPF를 `CUPF_ENABLE_CUSTOM_SOLVER=ON`으로 빌드하면 CUDA FP64 직접 솔버로 쓸 수 있다
  ([linear_solve](cuPF/docs/ops/linear_solve.md)).

---

## 6. benchmark/ · matlab/

- **`benchmark/`**: 평가기 빌드/실행 하니스(`build_eval.bash`, `run_benchmark.bash`),
  실행 결과(`results/`, 덤프 케이스 포함), 분석 문서(`docs/`).
- **`matlab/`**: MATPOWER `runpf` 기준선 실행 경로(`login_online.bash`,
  `run_matpower_case.bash`, `run_matpower_case.m`). Python 벤치마크의
  `run_matpower.py`도 같은 `.m` 파일을 호출해 cuPF 결과를 MATPOWER와 대조.

---

## 7. 빌드 · 환경

```bash
cmake -S cuPF -B build \
  -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_WITH_TORCH=ON -DENABLE_TIMING=ON \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build build -j
```

주요 플래그: `WITH_CUDA`(CUDA/cuDSS), `BUILD_PYTHON_BINDINGS`(`_cupf`),
`CUPF_WITH_TORCH`(torch zero-copy), `ENABLE_TIMING`(단계 타이밍),
`CUPF_ENABLE_CUSTOM_SOLVER`(custom 직접 솔버). 상세는
[`cuPF/docs/system_architecture.md`](cuPF/docs/system_architecture.md) §6.

**컨테이너**: `Dockerfile`로 CUDA·cuDSS·SuiteSparse·torch·pandapower
환경 구성. 파이썬 의존성은 [`requirements.txt`](requirements.txt)(numpy/scipy/pandas,
pandapower 3.2, matpower, torch 2.9, cupy-cu12, pybind11, jupyterlab 등).

```bash
# Docker 이미지 빌드
docker build -t gpu-powerflow:latest .

# 컨테이너 실행
docker run --gpus all --rm -it gpu-powerflow:latest

# 현재 체크아웃을 컨테이너에 마운트해서 실행
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/gpu-powerflow \
  -w /workspace/gpu-powerflow \
  gpu-powerflow:latest
```

---

## 8. 시작하기

1. 컨테이너 또는 `requirements.txt`로 환경 구성.
2. cuPF 빌드(§7) → `_cupf` 모듈 생성.
3. 입문은 [`python/tutorial/`](python/tutorial/README.md) 노트북 1번부터.
4. 벤치마크는 `python -m python.tests.run_benchmark ...` → `aggregate_results`.
5. 솔버 내부·API는 [`cuPF/docs/`](cuPF/docs/system_architecture.md).
