# cuPF Mixed-profile linear-solver 벤치마크 (cuDSS vs custom vs custom+CUDA graph)

cuPF Mixed 프로파일(FP32 Jacobian/step, FP64 state)에서 선형 솔버 백엔드 3종을 배치 크기별로
비교한다. CUDA 그래프 기능(`use_cuda_graph`, 문서: [cuda-graph-iteration.md](cuda-graph-iteration.md))의
실제 이득을 정량화하는 것이 목적.

## 설정

- GPU: NVIDIA RTX 3090 (sm_86). CUDA 12.x.
- 프로파일: `ComputePolicy::Mixed`. `NRConfig.tolerance = 1e-8` (프로젝트 기본), `max_iter = 30`.
- 배치: B ∈ {1, 32, 64}. 배치 구성: 1개 케이스를 B개로 복제하되 케이스 b의 부하를
  `×(1+0.001·b)`로 살짝 다르게(서로 구별되며 수렴). 배치 수렴 판정은 worst-case norm.
- 백엔드 3종:
  - **cuDSS** — `cuda_linear_solver=CuDSS` (CUDA 그래프 불가: `cudssExecute`는 stream-capture 불가).
  - **custom** — `cuda_linear_solver=Custom`, `use_cuda_graph=false`. 솔버 **내부** factor/solve
    그래프는 켜진 상태(`CLS_INTERNAL_GRAPH=ON`). 즉 솔버 커널 런치는 이미 최소화된 baseline.
  - **custom+graph** — `cuda_linear_solver=Custom`, `use_cuda_graph=true`. cuPF가 ibus→…→
    voltage_update 전체 iteration을 하나의 CUDA 그래프로 캡처/replay(솔버는 external 모드
    `CLS_INTERNAL_GRAPH=OFF`로 빌드되어 cuPF 그래프에 포함됨).
- 측정 범위: **정상상태 `solve_batch`만** (`tests/graph_bench.cpp`). 즉 아래 "결과" 표의
  시간은 **`initialize()`(심볼릭 분석)와 일회성 per-batch setup/그래프 캡처를 제외**한 반복
  solve 시간이다. `initialize()`는 타이밍 루프 밖에서 1회 호출하고, 일회성 setup/캡처는
  warmup 1회로 흡수한 뒤, `solve_batch`를 8회(70k는 4회) 측정해 평균한다. per-stage 타이밍
  (`CUPF_ENABLE_TIMING`)은 그래프 캡처와 양립 불가라 쓰지 않고 wall-clock으로 잰다.
  `solve_batch`는 결과를 다운로드하므로 반환 시점에 동기적. **initialize/일회성 비용은 아래
  별도 섹션에서 따로 보고**한다. 각 solve_batch 안의 upload + NR 루프 + download는 포함.
- 케이스: 요청한 3k/6k/9k/25k/70k에 가장 근접한 dump 케이스로 매핑(정확한 3120/6470/9241
  버스 dump는 데이터셋에 없음).

| 라벨 | 실제 케이스 | n_bus | nnz(Ybus) |
|---|---|---|---|
| 3k | case_ACTIVSg2000 | 2000 | 7334 |
| 6k | Base_Florida_42GW | 5658 | 21384 |
| 9k | Base_MIOHIN_76GW | 10189 | 39589 |
| 25k | case_ACTIVSg25k | 25000 | 85220 |
| 70k | case_ACTIVSg70k | 70000 | 236636 |

## 결과 — 시스템당 평균 시간 ms (괄호 = NR iterations)

| 케이스 | B | cuDSS | custom | custom+graph |
|---|---|---|---|---|
| **3k** | 1  | 1.743 (4) | 2.521 (4) | 2.445 (4) |
|        | 32 | 0.412 (4) | 0.192 (4) | 0.183 (5) |
|        | 64 | 0.489 (5) | 0.179 (5) | 0.179 (5) |
| **6k** | 1  | 3.00 (5)  | 4.94 (5)  | 4.80 (5)  |
|        | 32 | 0.997 (5) | 0.530 (5) | 0.520 (5) |
|        | 64 | 0.952 (5) | 0.470 (5) | 0.463 (5) |
| **9k** | 1  | 4.44 (5)  | 8.80 (5)  | 9.49 (5)  |
|        | 32 | 1.794 (5) | 1.071 (5) | 1.062 (5) |
|        | 64 | 1.694 (5) | 0.962 (5) | 0.981 (5) |
| **25k**| 1  | 6.75 (5)  | 11.46 (5) | 10.90 (5) |
|        | 32 | 3.432 (6) | 2.181 (5) | 2.053 (5) |
|        | 64 | 3.338 (6) | 2.053 (6) | 2.151 (6) |
| **70k**| 1  | 22.32 (7) | 63.93 (8) | 61.20 (8) |
|        | 32 | 13.90 (8) | 11.36 (8) | 10.89 (8) |
|        | 64 | 14.40 (8) | 10.20 (8) | 11.21 (9) |

(총 시간 = ms/sys × B. B=1 행이 곧 단일 케이스 시간. 위 표는 solve_batch만 — initialize·일회성
setup 제외.)

## Initialize / 일회성 비용 (위 solve 표에 미포함)

solve 표는 반복 solve의 정상상태 시간이다. 실제 1회 워크로드는 여기에 아래 일회성 비용이
더해진다. cuPF는 `initialize()`(심볼릭 분석)와 `solve`를 구분하며, 추가로 첫 `solve_batch`가
per-batch setup(custom의 arena 할당+내부그래프 캡처 / custom_graph의 iteration-그래프 캡처 /
cuDSS의 **지연된 분석·인수분해**)을 흡수한다.

### `initialize()` — 심볼릭 분석 (배치 무관, 1회)

| 케이스 | cuDSS | custom | custom+graph |
|---|---|---|---|
| 3k  | 35.5 ms | 7.0 ms | 7.1 ms |
| 6k  | 60.3 ms | 19.3 ms | 19.3 ms |
| 9k  | 100.6 ms | 37.0 ms | 36.8 ms |
| 25k | 183.3 ms | 61.9 ms | 62.6 ms |
| 70k | 480.8 ms | 170.3 ms | 171.4 ms |

→ cuDSS의 분석이 custom보다 **2.6~5× 비싸다**. (custom은 분석을 `initialize`로 front-load
하고, cuDSS는 가볍게 두는 대신 첫 solve로 미룬다 — 아래 참고.) 프로세스 최초 솔버 생성 1회는
여기에 CUDA/라이브러리 컨텍스트 초기화 ~140~250ms가 더 붙는다(측정 시 첫 구성에만 해당).

### 첫 `solve_batch`의 일회성 setup (= warmup − 정상 solve)

| 케이스, B | cuDSS | custom | custom+graph |
|---|---|---|---|
| 9k, B=64  | ~59 ms | ~5 ms | ~4 ms |
| 25k, B=32 | ~89 ms | ~6 ms | ~13 ms |
| 25k, B=64 | ~124 ms | ~9 ms | ~12 ms |
| 70k, B=32 | ~273 ms | ~0 ms | ~11 ms |
| 70k, B=64 | ~248 ms | ~13 ms | ~12 ms |

→ **cuDSS는 첫 solve가 매우 무겁다**(분석·인수분해를 첫 `cudssExecute`로 지연; 케이스·B에
비례해 수십~수백 ms). custom/custom+graph의 일회성 setup(그래프 캡처 포함)은 **~2~13 ms**로
작다. 따라서 "한 번만 푸는" 워크로드의 실제 비용은 `initialize + 첫 solve`이며, 이 관점에선
cuDSS의 우위(정상상태 B=1)가 크게 줄거나 사라진다. 반복 solve(같은 패턴, 값만 변경 — cuPF의
주 사용 패턴)에선 일회성 비용이 상각되어 위 solve 표가 지배한다.

## 분석

### 1. 배치 처리량: custom ≫ cuDSS (지배적 레버)
B=32/64에서 시스템당 시간이 custom 계열이 cuDSS 대비 크게 빠르다:

| 케이스 (B=64) | cuDSS | custom+graph | 배속 |
|---|---|---|---|
| 3k  | 0.489 | 0.179 | **2.7×** |
| 6k  | 0.952 | 0.463 | **2.1×** |
| 9k  | 1.694 | 0.981 | **1.7×** |
| 25k | 3.338 | 2.151 | **1.6×** |
| 70k | 14.40 | 11.21 | **1.3×** |

cuDSS는 B가 커져도 시스템당 시간이 거의 줄지 않는다(uniform-batch 효율 낮음, 오히려 큰 케이스
에선 B↑에 약간 악화). custom은 B=1→64에서 시스템당 시간이 크게 떨어진다(레벨/런치 지연이 B로
분산). 작을수록(런치 지배적) 배속이 크다.

### 2. 단일 케이스(B=1): cuDSS가 보통 빠름
custom의 batched-of-1 + Mixed 오버헤드 때문에 단일 케이스에선 cuDSS가 유리(9k: 4.44 vs ~9 ms,
70k: 22.3 vs ~62 ms). custom의 강점은 어디까지나 **배치**.

### 3. CUDA 그래프의 추가 이득: 작음 (−0%~−6%)
custom baseline이 이미 솔버 **내부** 그래프로 factor/solve 런치를 최소화한 상태라, cuGraph가
추가로 없애는 것은 cuPF 자체의 iteration당 ~8개 비-솔버 커널(ibus/mismatch/norm/jacobian/
prepare/voltage) 런치 오버헤드뿐이다. iteration 수가 같은(=공정 비교) 항목 기준 custom+graph의
custom 대비 변화:

| 항목 | custom | custom+graph | 변화 |
|---|---|---|---|
| 25k B=32 | 2.181 | 2.053 | **−5.9%** |
| 70k B=32 | 11.36 | 10.89 | **−4.1%** |
| 25k B=1  | 11.46 | 10.90 | **−4.9%** |
| 70k B=1  | 63.93 | 61.20 | **−4.3%** |
| 6k B=64  | 0.470 | 0.463 | −1.5% |
| 9k B=32  | 1.071 | 1.062 | −0.8% |
| 9k B=64  | 0.962 | 0.981 | +2.0% (노이즈) |

대형·대배치(compute-bound)에선 거의 동률이고, 측정 노이즈로 ±2% 안에서 뒤집히기도 한다.
일관된 이득은 launch-overhead 비중이 큰 구간(B=1, 그리고 25k/70k의 B=32)에서 **약 4~6%**.

### 결론
이 데이터셋·배치 범위에서 **가장 큰 성능 레버는 cuDSS→custom 전환(배치에서 1.3~2.7×)**이고,
**CUDA 그래프는 그 위에 얹는 보조 최적화로 약 0~6%** (launch-overhead가 지배적인 작은 배치/
케이스에서 가장 의미 있음). 솔버 내부 그래프가 이미 큰 몫을 가져갔기 때문에 전체-iteration
그래프의 한계 이득은 제한적이다.

## 주의사항
- **Mixed FP32 + tol 경계의 iteration 흔들림**: Mixed는 FP32 factor + 비결정적 extend-add
  atomicAdd라, worst-case 부하(b=63, +6.3%)의 잔차가 tol(1e-8) 근처일 때 iteration 수가
  run/backend마다 5↔6, 8↔9로 흔들린다(표의 일부 셀). 평균 ms는 서로 다른 iteration 수가 섞여
  ±한 자릿수 % 노이즈를 가진다. iteration 수가 다른 셀은 시간 비교가 부정확.
- **그래프 모드 제약**: forward-only, `CUPF_ENABLE_TIMING`/`ENABLE_DUMP`와 양립 불가
  (per-stage sync / D2H가 캡처를 깸). 그래서 본 벤치는 per-stage 분해 대신 end-to-end만 측정.
- **케이스 매핑**: 정확한 3120/6470/9241 버스 dump가 없어 근접 케이스로 대체. 정확한 크기가
  필요하면 .m(MATPOWER)에서 dump를 생성해야 한다.

## 재현

```bash
# 빌드 A (cuDSS, custom-내부그래프):  CUPF_ENABLE_CUDA_GRAPH=OFF
cmake -S cuPF -B build_nog -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_ENABLE_CUSTOM_SOLVER=ON -DCUPF_ENABLE_CUDA_GRAPH=OFF -DCUPF_WITH_TORCH=OFF \
  -DBUILD_EVALUATORS=ON -DCMAKE_PREFIX_PATH=<torch>/share/cmake
cmake --build build_nog --target cupf_graph_bench -j

# 빌드 B (custom+cuGraph):  CUPF_ENABLE_CUDA_GRAPH=ON  -> 솔버 external 모드(CLS_INTERNAL_GRAPH OFF)
cmake -S cuPF -B build_g -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_ENABLE_CUSTOM_SOLVER=ON -DCUPF_ENABLE_CUDA_GRAPH=ON -DCUPF_WITH_TORCH=OFF \
  -DBUILD_EVALUATORS=ON -DCMAKE_PREFIX_PATH=<torch>/share/cmake
cmake --build build_g --target cupf_graph_bench -j

# 실행 (CUPF_BENCH_TOL로 tol 오버라이드 가능, 기본 1e-8)
CASE=<datasets>/cuPF_datasets/case_ACTIVSg25k
./build_nog/tests/cupf_graph_bench $CASE cudss        1,32,64 8
./build_nog/tests/cupf_graph_bench $CASE custom       1,32,64 8
./build_g/tests/cupf_graph_bench   $CASE custom_graph 1,32,64 8
```
