# gpu-powerflow

미분 가능(differentiable) **GPU 가속 전력조류(power flow)** 솔버 **cuPF**와, 이를 둘러싼
데이터 준비·벤치마크·튜토리얼·기준선(baseline) 도구를 모은 작업 공간이다.

핵심은 [`cuPF/`](cuPF/) — 희소 `Ybus`를 입력으로 Newton–Raphson(NR) 전력조류를 GPU에서
풀고, adjoint(역전파)로 부하 gradient를 계산하며, 시나리오 **배치**를 한 번에 처리하는
C++/CUDA 솔버다. 백엔드는 CPU(SuiteSparse KLU/UMFPACK)와 CUDA(cuDSS / 자체 multifrontal
솔버)를 런타임에 선택하고, 정밀도는 FP64 / FP32 / Mixed / TF32를 지원한다.

---

## 1. 저장소 구성

```text
.
├── cuPF/                 # 핵심 솔버: C++/CUDA 라이브러리 + pybind/torch 바인딩 + 설계 문서
│   ├── cpp/              #   newton_solver (core, ops, storage)
│   ├── bindings/         #   pybind_cupf.cpp, torch_cupf_extension.cpp
│   ├── python/cupf/      #   torch autograd 래퍼 패키지
│   ├── tests/            #   run_cupf.py(통합 러너) · cupf_bench(C++) · unit/ · smoke/
│   └── docs/             #   컴포넌트별 설계 문서
├── custom_linear_solver/ # CUDA multifrontal 직접 선형 솔버 (cuPF의 선택적 CUDA 백엔드)
│   └── tests/            #   scripts/(build_custom·cudss·strumpack) · datasets/ · runners
├── benchmark/            # 벤치마크 일체
│   ├── pinn/             #   PINN 훈련 하니스(더미 레이어 + 5에폭) — train.py
│   ├── backends/         #   백엔드 러너: exapf/(Julia) · matlab/(MATPOWER) · pypower·pandapower·matpower
│   └── common/           #   공유 스캐폴딩: matpower_data · eval_common · run_benchmark · aggregate_results
├── prepare_datasets/     # MATPOWER .m → 기준 PF 풀이 → cuPF 덤프 / Newton 선형계 작성
├── tutorial/             # 입문용 라이브 노트북 6종
├── Dockerfile            # CUDA 12.8 + cuDSS + SuiteSparse + torch 환경
└── requirements.txt
```

핵심 진입점 한눈에:

| 목적 | 명령 |
|------|------|
| cuPF 실행/측정 | `python3 cuPF/tests/run_cupf.py …` |
| PINN 훈련 벤치 | `python3 -m benchmark.pinn.train …` |
| 기준선 교차비교 | `python3 -m benchmark.common.run_benchmark …` |
| 데이터셋 준비 | `python3 -m prepare_datasets.prepare …` |
| 선형 솔버 단독 벤치 | `custom_linear_solver/tests/scripts/build_custom.sh` |

---

## 2. 요구 사항

- **CUDA 12.x** + **cuDSS** (GPU 백엔드), C++17, **CMake ≥ 3.22**
- **SuiteSparse** KLU/UMFPACK (CPU 백엔드)
- **Python 3.10**, **PyTorch 2.9**(+cu12), pybind11, pandapower 3.2 — `requirements.txt` 참조
- (선택) GoogleTest(C++ 단위 테스트), Julia+ExaPF·MATLAB+MATPOWER(해당 기준선 백엔드)

가장 간단한 길은 **Docker**다(아래 §3).

---

## 3. 설치

### Docker (권장)

```bash
docker build -t gpu-powerflow:latest .

# 현재 체크아웃을 컨테이너에 마운트해 실행
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/gpu-powerflow -w /workspace/gpu-powerflow \
  gpu-powerflow:latest
```

`Dockerfile`은 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 위에 cuDSS·SuiteSparse·Nsight,
그리고 `requirements.txt`의 파이썬 의존성을 설치한다.

### 로컬

```bash
pip install -r requirements.txt
```

CUDA 12.x·cuDSS·SuiteSparse는 시스템에 별도 설치한다(경로는 Dockerfile 참고:
cuDSS 헤더 `/usr/include/libcudss/12`, 라이브러리 `/usr/local/cuda/lib64` 등).

---

## 4. 빌드

### cuPF (라이브러리 + 바인딩 + 벤치 + 테스트)

```bash
bash cuPF/tests/scripts/build_cupf.sh          # 단일 cmake 빌드 래퍼
# 또는 직접:
cmake -S cuPF -B cuPF/build \
  -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON -DCUPF_WITH_TORCH=ON \
  -DENABLE_TIMING=ON -DCUPF_ENABLE_CUSTOM_SOLVER=ON \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build cuPF/build -j
```

한 번의 빌드가 `_cupf`(pybind)·torch 확장·`cupf_bench`(C++)·단위/스모크 테스트를 모두 만든다.
주요 플래그: `WITH_CUDA`(CUDA/cuDSS), `BUILD_PYTHON_BINDINGS`(`_cupf`),
`CUPF_WITH_TORCH`(torch zero-copy), `ENABLE_TIMING`(단계 타이밍),
`CUPF_ENABLE_CUSTOM_SOLVER`(자체 직접 솔버). 자세히는
[`cuPF/docs/system_architecture.md`](cuPF/docs/system_architecture.md).

### custom_linear_solver (선택)

cuPF의 CUDA 백엔드로 쓰는 multifrontal 직접 솔버. 단독 벤치는 외부 라이브러리별 스크립트로
빌드한다(서로 다른 lib을 링크하므로 분리):

```bash
custom_linear_solver/tests/scripts/build_custom.sh      # 자체 솔버
custom_linear_solver/tests/scripts/build_cudss.sh       # cuDSS 기준
custom_linear_solver/tests/scripts/build_strumpack.sh   # STRUMPACK 기준
```

### 테스트 (ctest)

```bash
ctest --test-dir cuPF/build --output-on-failure   # 단위(gtest) + 파이썬 스모크
```

---

## 5. 실행

### 5.1 데이터셋 준비

MATPOWER `.m` 케이스를 풀어 cuPF 덤프 / Newton 선형계를 만든다.

```bash
# cuPF 덤프 디렉터리(Ybus/Sbus/V0/pv/pq)
python3 -m prepare_datasets.prepare --dataset-root /datasets/matpower --cases case9 case118
# custom_linear_solver용 선형계 (J.mtx / F.mtx)
python3 -m prepare_datasets.convert_linear_system --dataset-root /datasets/matpower --cases case118
```

### 5.2 cuPF 실행·측정 — `cuPF/tests/run_cupf.py`

세 실행 경로(C++ / Python pybind / PyTorch) × 세 측정 모드를 하나의 진입점에서.

```bash
# C++ 벤치: 최적화 init+solve 시간
python3 cuPF/tests/run_cupf.py --path cpp    --backend cuda-custom --precision tf32 --mode solve
# CPU(KLU/UMFPACK) 디버깅(시스템별 잔차)
python3 cuPF/tests/run_cupf.py --path python --backend cpu-klu     --precision fp64 --mode debug
# PyTorch autograd: forward+backward
python3 cuPF/tests/run_cupf.py --path torch  --backend cuda-cudss  --precision mixed --grad
```

옵션: `--path {cpp,python,torch}`, `--backend {cuda-custom,cuda-cudss,cpu-klu,cpu-umfpack}`,
`--precision {fp64,fp32,mixed,tf32}`, `--mode {solve,operators,debug}`, `--jacobian`,
`--cases --batch --repeats`. `--help` 참조.

### 5.3 PINN 훈련 벤치 — `benchmark/pinn/`

각 백엔드 앞에 더미 레이어를 두고 상태예측 PINN(손실 = 전력조류 잔차 ‖F(V)‖)을 5에폭 훈련.
`cupf-torch`는 NR을 통과하는 미분가능 경로, 나머지는 forward physics oracle.

```bash
python3 -m benchmark.pinn.train --case case118 --epochs 5
python3 -m benchmark.pinn.train --backends pypower cupf-torch --case case300
```

### 5.4 기준선 교차비교 매트릭스 — `benchmark/common/`

pypower·MATPOWER 기준선을 한 `runs.csv`로 모아 집계.

```bash
python3 -m benchmark.common.run_benchmark --cases case9 --warmup 0 --repeats 1
python3 -m benchmark.common.aggregate_results benchmark/results/<run-name>
```

### 5.5 선형 솔버 단독 벤치 — `custom_linear_solver/`

```bash
custom_linear_solver/tests/scripts/build_custom.sh
custom_linear_solver/build/custom_linear_solver_run \
  --matrix custom_linear_solver/tests/datasets/power/case118/J.mtx \
  --rhs    custom_linear_solver/tests/datasets/power/case118/F.mtx \
  --precision tf32 --batch 64
```

---

## 6. 튜토리얼

입문용 라이브 노트북은 [`tutorial/`](tutorial/README.md)에 있다(01 전력계통 기초·case9 →
02 NR·Jacobian → 03 MATPOWER/pandapower 기준선 → 04 cuPF CPU → 05 cuPF GPU →
06 배치·Torch autograd·향후 연구).

---

## 7. 문서

- 전체 설계: [`cuPF/docs/system_architecture.md`](cuPF/docs/system_architecture.md)
- 솔버 API·바인딩: [`cuPF/docs/newton_solver.md`](cuPF/docs/newton_solver.md),
  [`cuPF/docs/bindings.md`](cuPF/docs/bindings.md)
- 자체 선형 솔버: [`custom_linear_solver/README.md`](custom_linear_solver/README.md),
  [`custom_linear_solver/docs/`](custom_linear_solver/docs/)
- 벤치마크: [`benchmark/README.md`](benchmark/README.md)
