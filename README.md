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
├── external/lin_solver/  # 외부 선형 솔버 실험 원본(custom_linear_solver는 이 부분집합)
├── benchmark/            # 빌드/실행 하니스, 결과(results/), 분석 docs
├── matlab/               # MATPOWER runpf 기준선·NR 선형솔버 스윕 스크립트
├── bash/                 # 빌드/실행 래퍼 스크립트
├── Dockerfile, docker-compose.yml, requirements.txt
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
- **Python 바인딩** `_cupf`: numpy API + (선택) torch 확장.

---

## 2. python/ — 준비·벤치마크·튜토리얼

세 책임으로 분리(자세히는 [`python/README.md`](python/README.md)):

- **`python/prepare/`** — 데이터셋/덤프 준비. MATPOWER `.m` 파싱 → 기준 PF 풀이 →
  cuPF 덤프 디렉터리 작성(`prepare.py`), Newton 선형계 생성(`convert_linear_system.py`),
  `.m`→`.mat` 변환(`convert_m_to_mat.py`).
- **`python/tests/`** — 단일 벤치마크 패키지. MATPOWER 케이스 로딩(`matpower_data.py`),
  기준선(`run_pypower.py` SciPy/pandapower NR, `run_matpower.py` MATLAB runpf),
  cuPF 러너(`run_cupf_pybind.py` `_cupf` pybind, `run_cupf_native.py` `cupf_cpp_evaluate`),
  오케스트레이션(`run_benchmark.py`), 집계(`aggregate_results.py`).
  ```bash
  python3 -m python.tests.run_benchmark --cases case9 --warmup 0 --repeats 1
  python3 -m python.tests.aggregate_results benchmark/results/<run-name>
  ```
- **`python/tutorial/`** — 입문용 스토리형 노트북 6종(라이브 실행, [README](python/tutorial/README.md)):
  1. 전력계통 기초·case9·Ybus
  2. MATPOWER/pandapower 기준선 + 파이썬 단계 병목
  3. cuPF CPU 경로(UMFPACK/KLU/Jacobian 변형 비교)
  4. cuPF GPU 경로(cuDSS·Edge/EdgeAtomic/VertexWarp·custom solver)
  5. pybind `solve_batch`·`NewtonOptions`·**Torch autograd**
  6. 남은 병목과 향후 연구(cuGraph·custom solver·multi-GPU·tensor core·mixed)

> 사용자용 파이썬 패키지 코드는 [`cuPF/python/`](cuPF/python/)(`cupf`), 벤치마크
> 평가기는 `python/tests/`에 둔다(분리 유지).

---

## 3. custom_linear_solver/ · external/lin_solver/

- **[`custom_linear_solver/`](custom_linear_solver/README.md)**: cuDSS와 동일한 phase
  모델(`set_data → analyze → factorize → solve`)을 따르는 CUDA multifrontal 직접 솔버
  래퍼. GPU multifrontal 커널 + 심볼릭 분석 + METIS nested-dissection 정렬 + 최소
  matrix/API 상태만 추린 모듈이다. cuPF를 `CUPF_ENABLE_CUSTOM_SOLVER=ON`으로 빌드하면
  CUDA FP64 경로의 직접 솔버로 쓸 수 있다([ops/linear_solve](cuPF/docs/ops/linear_solve.md)).
- **`external/lin_solver/`**: 위 원본이 되는 외부 선형 솔버 실험 일체(벤치마크 드라이버,
  matching/GPU-ND 실험, 서드파티 어댑터 등 — custom_linear_solver는 그 부분집합).

---

## 4. benchmark/ · matlab/

- **`benchmark/`**: 평가기 빌드/실행 하니스(`build_eval.bash`, `run_benchmark.bash`),
  실행 결과(`results/`, 덤프 케이스 포함), 분석 문서(`docs/`).
- **`matlab/`**: MATPOWER `runpf` 기준선과 NR 선형솔버 스윕(`run_matpower_case.*`,
  `sweep_nr_lin_solvers.*`), 100+ 케이스 선정표. cuPF 결과를 MATPOWER와 대조하는 데 쓴다.

---

## 5. 빌드

### cuPF (CMake)

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

`bash/build_cupf.bash`·`bash/run_cupf.bash`는 간단한 빌드/실행 래퍼다.

### 컨테이너

`Dockerfile` / `docker-compose.yml`로 CUDA·cuDSS·SuiteSparse·torch·pandapower 등 전체
의존성을 갖춘 환경을 만든다. 파이썬 의존성은 [`requirements.txt`](requirements.txt)
(numpy/scipy/pandas, pandapower 3.2, matpower, torch 2.9, cupy-cu12, pybind11,
jupyterlab 등).

---

## 6. 시작하기

1. 컨테이너 또는 `requirements.txt`로 환경 구성.
2. cuPF 빌드(위 §5) → `_cupf` 파이썬 모듈 생성.
3. 입문은 [`python/tutorial/`](python/tutorial/README.md) 노트북 1번부터.
4. 벤치마크는 `python -m python.tests.run_benchmark ...`, 결과 집계는 `aggregate_results`.
5. 솔버 내부·API는 [`cuPF/docs/`](cuPF/docs/system_architecture.md).
