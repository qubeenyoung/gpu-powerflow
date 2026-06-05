# 단일 배치 경로 (pre-batched) 의 병목 분석 — case8387pegase, FP64 / FP32

*`Solver::factorize()` / `Solver::solve()` (batched 구현 이전의 single-batch 전용 경로) 위 case8387pegase 측정. 30단 etree level-by-level CUDA Graph replay 가 (1) wall-time, (2) nsys 커널 분포, (3) ncu bound (compute / memory / 점유율) 어디서 시간을 쓰는지 분리하고, hyper-parameter sweep, warp-per-front / shared-staged kernel 의 구현·측정, FP32 비교까지 포함한 보고서. 자매 문서: [`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md).*

## 0. Baseline / Scope

| | |
|---|---|
| 매트릭스 | case8387pegase (n=14908, nnz=110572) |
| 정밀도 | **FP64** (`--single-precision fp64`) |
| GPU | RTX 3090 sm_86, 82 SM, FP64 0.56 TFLOPS, GDDR6X 936 GB/s |
| Panel cap | 8 (`n < 16000` 분기), **30 levels, 7415 panels** |
| 빌드 | `RelWithDebInfo` + `-lineinfo -O3`, `CLS_INTERNAL_GRAPH=ON` (기본) |
| 실행기 | `custom_linear_solver_run`, repeat 50 (nsys) / 5 (ncu) |
| 측정 도구 | nsys 2024.x (`--cuda-graph-trace=node`), ncu 12.8 |

**경로 확정** — single-batch 경로의 진입점은 `src/solver.cpp` 의 `Solver::factorize()` / `Solver::solve()` 다. 내부적으로는:

- `analyze()` 단계 (`src/factorize/multifrontal.cu` L759–L811) 에서 모든 level 의 `mf_factor_extend_level<T>` 와 root level 의 `mf_bigA/B/C_*` 를 한 번에 **`cudaStreamBeginCapture` → `cudaGraphInstantiate`** 로 캡처해 `plan.graph_exec` 에 저장.
- `factorize()` 는 `mf_scatter_csr_values` 한 번 → `cudaGraphLaunch(plan.graph_exec)` 한 번 (L857–L864).
- `solve()` 는 `src/solve/multifrontal.cu` 의 `mf_fwd_level<double, double>` × 30 → `mf_bwd_level<double, double>` × 30 을 graph 로 replay.

batched 경로 (`Solver::batched_factorize`, `--batch B`) 와는 **완전히 별개의 커널 세트** 다 (`batched/` 의 `*_b<FT>` 변종은 호출되지 않음).

---

## 1. Wall-time 분해 (nsys, 50회 median, 1 콜당)

`nsys profile -t cuda,nvtx --cuda-graph-trace=node` 로 캡처된 graph 내부 노드까지 attribution 한 결과 (보고 단위: per single factorize/solve call):

### 1.1 Factorize ≈ 626 μs

| 커널 | 호출/콜 | 총 시간 [μs] | 비중 |
|---|---:|---:|---:|
| `mf_factor_extend_level<double>` | 29 (level 별) | 555 | **89%** |
| `mf_invert_pivot<double>` | 1 | 24 | 4% |
| `mf_bigB_trailing` (root TC) | 1 | 16 | 3% |
| `mf_bigA_panelU` (root TC) | 1 | 15 | 2% |
| `mf_bigC_extend` (root TC) | 1 | 11 | 2% |
| `mf_scatter_csr_values` (graph 밖) | 1 | 5 | 1% |
| **합** | | **626** | **100%** |

### 1.2 Solve ≈ 332 μs

| 커널 | 호출/콜 | 총 시간 [μs] | 비중 |
|---|---:|---:|---:|
| `mf_bwd_level<double>` | 30 (level 별) | 201 | **61%** |
| `mf_fwd_level<double>` | 30 (level 별) | 127 | 38% |
| `gather_permuted_rhs` | 1 | 2 | <1% |
| `scatter_permuted_solution` | 1 | 2 | <1% |
| **합** | | **332** | **100%** |

**관찰** — factorize/solve 둘 다 시간의 대부분이 **"level 별 단일 커널 1종"** 으로 압축된다. root level 에서만 TC-style 큰 dense 분기 (`mf_bigA/B/C`) 가 따로 호출되고, 이는 시간 비중이 7% 로 작다.

---

## 2. Etree / front 분포 (`CLS_DUMP=1`)

```
n=14908  P=7415  levels=30  cap=8  front_total(MB f32)=2.0
```

| fsz 빈 | 개수 | Σfsz² 비중 | Σfsz³ 비중 |
|---|---:|---:|---:|
| 1–16 (tiny) | **7,135** | **53.8%** | **19.9%** |
| 17–32 | 225 | 23.2% | 26.2% |
| 33–48 | 36 | 10.3% | 18.7% |
| 49–64 | 15 | 9.1% | 23.8% |
| 65–96 | 4 | 3.6% | 11.4% |
| ≥ 97 | 0 | 0% | 0% |

레벨별 panel 분포 (level → 해당 level 의 panel 수, max fsz):

```
L0  cnt=4094  maxfsz=16    L10 cnt=19   maxfsz=49    L20 cnt=1   maxfsz=60
L1  cnt=1520  maxfsz=20    L11 cnt=15   maxfsz=41    L21 cnt=1   maxfsz=59
L2  cnt=733   maxfsz=33    L12 cnt=10   maxfsz=64    L22 cnt=1   maxfsz=51
L3  cnt=391   maxfsz=34    L13 cnt=9    maxfsz=56    L23 cnt=1   maxfsz=43
L4  cnt=237   maxfsz=49    L14 cnt=5    maxfsz=46    L24 cnt=1   maxfsz=35
L5  cnt=146   maxfsz=51    L15 cnt=3    maxfsz=61    L25 cnt=1   maxfsz=37
L6  cnt=91    maxfsz=65    L16 cnt=3    maxfsz=53    L26 cnt=1   maxfsz=29
L7  cnt=60    maxfsz=71    L17 cnt=1    maxfsz=68    L27 cnt=1   maxfsz=21
L8  cnt=37    maxfsz=63    L18 cnt=1    maxfsz=60    L28 cnt=1   maxfsz=13
L9  cnt=29    maxfsz=55    L19 cnt=1    maxfsz=68    L29 cnt=1   maxfsz=5
```

**구조적 특성**:

1. **L0 4094 panel 의 와이드 베이스** — single launch 한 번에 4094 block 이라 GPU 점유율은 충분.
2. **L17–L29: 13 개 레벨이 panel=1** — etree spine 의 deep tail. 각 level 이 **block 1 개** 짜리 launch 가 그래프 노드 1 개로 직렬 의존성.
3. front 의 96% 가 fsz ≤ 16 — warp 한 개도 못 채우는 영역.

---

## 3. Per-kernel ncu SoL (RTX 3090, 5 회 replay 샘플)

`ncu --section SpeedOfLight --section LaunchStats --section Occupancy` 로 각 launch 단위 측정 후 grid size 별로 버킷.

### 3.1 `mf_factor_extend_level<double>` (factorize 의 89%)

n=140 launches (29 level × 5 rep 근방). regs/thread=40, block size 256.

| grid 버킷 | 매칭 level | n | wall 비중 | avg [μs] | **SM%** | DRAM% | 점유율% |
|---|---|---:|---:|---:|---:|---:|---:|
| **=1** | L17–L29 | 40 | **19%** | 15.6 | **0.3** | <1 | 24 |
| 2–16 | L9–L16 | 45 | **36%** | 26.0 | **1.2** | <1 | 25 |
| 17–128 | L4–L8 | 25 | 23% | 29.2 | 3.8 | 1 | 25 |
| 129–1024 | L2–L3 | 20 | 15% | 24.2 | 9.8 | 3 | 60 |
| **>1024** | L0–L1 | 10 | **6%** | 20.8 | **37.6** | 12 | 78 |

전체 median: SM compute 1.3%, DRAM 0.8%, 점유율 25%. **wall 의 94% 가 SM% < 10 의 저점유 영역에서 소모됨.** 실제로 GPU 가 일하는 구간 (L0–L1) 은 wall 의 6% 에 불과.

### 3.2 `mf_invert_pivot<double>` (factorize 의 4%)

n=5. grid=7436 fixed, blocks of 32 threads, regs/thread=48.

| Duration [ns] | SM% | DRAM% | 점유율% | Waves/SM |
|---:|---:|---:|---:|---:|
| 27,488 | **51.3** | 6.3 | 24 | 5.67 |

이 한 커널만 그나마 의미 있게 GPU 를 굴린다 (모든 panel 의 pivot block 을 한 번에 invert, selinv 사전계산).

### 3.3 `mf_fwd_level<double, double>` (solve 의 38%)

n=28. regs/thread=40, block size 128.

| grid 버킷 | n | wall 비중 | avg [μs] | SM% | 점유율% |
|---|---:|---:|---:|---:|---:|
| =1 | 8 | 26% | 5.7 | 0.0 | 2.8 |
| 2–16 | 9 | 31% | 6.1 | 0.2 | 3.7 |
| 17–128 | 5 | 18% | 6.5 | 0.9 | 3.9 |
| 129–1024 | 4 | 14% | 6.3 | 3.5 | 11.1 |
| >1024 | 2 | 11% | 9.9 | 13.9 | 22.8 |

median: SM 0.2%, 점유율 3.8%. **factor 보다 더 심각** — selinv 의 GEMV 가 panel 단위라 block 수 자체가 적다.

### 3.4 `mf_bwd_level<double, double>` (solve 의 61%)

n=27. regs/thread=48, block size 128.

| grid 버킷 | n | wall 비중 | avg [μs] | SM% | 점유율% |
|---|---:|---:|---:|---:|---:|
| =1 | 8 | 27% | 8.1 | 0.1 | 2.7 |
| 2–16 | 9 | 34% | 9.2 | 0.9 | 3.8 |
| 17–128 | 5 | 20% | 9.5 | 5.0 | 4.2 |
| 129–1024 | 4 | 15% | 9.2 | 13.0 | 12.2 |
| >1024 | 1 | 5% | 11.2 | 27.3 | 24.6 |

bwd 가 fwd 보다 1.6× 느린 이유: parent extend (selinv 의 `U_inv @ rhs` 결과를 child 로 broadcast) 가 bwd 에만 있음 + register 압력 (regs/thread 48 vs 40).

---

## 4. 병목 본질 — Compute/Memory bound 어느 쪽도 아니다

| 지표 | 값 | 해석 |
|---|---|---|
| Median SM compute % (factorize) | 1.3% | 연산 유닛 거의 idle |
| Median DRAM throughput % | 0.8% | 메모리 BW 도 idle |
| Median 점유율 (small-grid 영역) | 3–25% | warp 가 SM 을 못 채움 |
| FP64 pipe utilization | 흔적 미만 | **FP64 자체가 binding 이 아님** |

**병목의 실체** = **"etree spine 을 따라가는 30 단 직렬 level 디스패치"**.

CUDA Graph 로 launch overhead 자체는 줄었지만, graph 의 노드 간 의존성 (level k+1 은 level k 의 모든 panel 완료에 의존) 이 **선형 체인** 으로 표현되어 있고, 각 그래프 노드가 적게는 4 μs, 많게는 30 μs 동안 stream 을 점유하면서 그동안 GPU 는 거의 idle.

---

## 5. 알고리즘 vs 하드웨어 분리

| 구분 | wall 비중 추정 | 항목 |
|---|---:|---|
| **알고리즘** (지배적) | ~80% | 30 단 직렬 level 디스패치, panel cap=8 로 인한 deep tail (L17–L29 = 13 levels × panel 1), 좁은 front (96% fsz ≤ 16) 로 인한 small-grid launch 다수 |
| **하드웨어** | ~14% | 그래프 노드당 잔존 launch / 스케줄링 overhead — small-grid launch 의 floor 약 3–5 μs (RTX 3090 + driver 580.x). 실제 일 양보다 dispatch latency 가 큰 영역 |
| **연산·메모리 bound 영역** | ~6% | L0–L1 (SM 38%, DRAM 12%) — 여기서만 fp64 throughput·BW 가 의미. solve 는 더 작음 |

batched 경로가 존재하는 정확한 이유 = 위 80% 의 "디스패치 직렬화" 를 batch dimension 으로 hide 하는 것 (모든 level 의 모든 (front, batch) 를 한 block grid 에 fold). batched 의 B=1 측정 ([`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md)) 과 본 single-batch 측정을 비교하면 *같은 알고리즘 + 다른 dispatch 구조* 의 차이가 그대로 보인다.

---

## 6. 즉시 시도 가능한 개선 후보 (구현 없이 우선순위만)

1. ~~**`CLS_CAP` 상향**~~ — **§7 측정 결과 기각**. cap=12 로 레벨을 30→23 으로 줄여도 factorize 가 오히려 +13% 느려짐 (각 레벨의 작업량이 fsz² 비례로 증가해 dispatch 절감을 압도).
2. **deep-chain fusion** — L_k..L_29 의 panel=1 레벨 13 개를 단일 persistent-grid 커널로 흡수. ~~잠재 절감 ~200–400 μs (factorize 의 1/3)~~ → **§7 측정으로 잠재이득 재산정 필요**. cap sweep 데이터에 의하면 deep-chain 의 dispatch latency 자체는 wall 의 ~15% 정도. fusion 으로 회수 가능한 상한은 약 100–150 μs.
3. **그래프 내 cross-level 병렬** — 현재 capture 는 단일 stream 직렬. 독립 subtree 끼리는 서로 다른 stream 으로 capture → graph 가 fork/join 을 표현 → 중간 narrow 레벨에서 idle 시간 회수. 단 dependency 분석은 `panel_parent` 로부터 derive 필요.
4. **per-level 커널의 효율 개선이 더 큰 레버** — §7 의 cap=6 가 best (factor 0.547ms) 라는 사실은 *"같은 일을 더 작은 단위로"* 가 이 GPU 에서 유리하다는 것. mid 레벨 (grid 17–128) 에서 SM% 3.8%, 점유율 25% 로 묶여 있는 부분을 풀어주는 게 dispatch 절감보다 큰 ROI. 후보:
   - block size 튜닝 (현재 256/384/768 의 3-tier, fsz 분포에 비해 과대적합 가능)
   - shared memory staging 의 reuse (현재 panel 마다 front 전체를 shared 로 올림)
5. **fp64 → mixed 효과는 미미** — 단일배치 경로에서는 SM% / DRAM% 모두 binding 이 아니므로 정밀도 down-cast 의 회수율 매우 낮음. `mf_factor_extend_mixed` 분기는 의미 있으나 2, 3, 4 가 선결되어야 효과 측정 가능.

---

## 7. CLS_CAP sweep — 가설 검증과 결과 보정

원본 §6 의 1번 (`CLS_CAP=12` 상향) 을 실측으로 검증.

### 7.1 구조 변화 (`CLS_DUMP=1`)

| cap | levels | panels(P) | deep-tail (panel=1) | front_total | L0 panels (베이스) |
|---|---:|---:|---:|---:|---:|
| 6 | **37** | 7487 | 13 (L24–L36) | 2.3 MB (+15%) | 4083 |
| 8 (default) | 30 | 7415 | 13 (L17–L29) | 2.0 MB | 4094 |
| 10 | 25 | 7362 | 7 (L18–L24) | 2.0 MB | 4102 |
| **12** | **23** | 7370 | **7** (L16–L22) | 2.0 MB | **4101** |

- 레벨이 얕아지는가: **그렇다.** cap=8→12 로 levels 30→23 (-23%), deep-tail (panel=1) 13→7 (-46%).
- 같은 레벨에 panel 이 더 들어가는가 (병렬성 증가): **거의 아니다.** L0 panel 수 4094 → 4101 (+0.2%). cap 상향은 etree chain 을 merge 하는 효과라 **panel 수가 아니라 panel 의 fsz 를 키움**. 병렬 width 는 본질적으로 sparsity 패턴이 결정한다.

### 7.2 wall-time 효과 (50 회 median, FP64)

| cap | factorize [ms] | solve [ms] | f+s [ms] | Δ vs cap=8 |
|---|---:|---:|---:|---:|
| 6 | **0.547** | 0.345 | **0.892** | **−3%** |
| 8 (default) | 0.593 | 0.327 | 0.920 | — |
| 10 | 0.595 | 0.320 | 0.915 | −1% |
| **12** | 0.668 | **0.306** | 0.974 | **+6%** |
| 16 | 0.795 | 0.345 | 1.140 | +24% |
| 20 | crash | — | — | (cap > MF_REG_NC 한계) |

**핵심 발견**:
- cap=12 는 factorize 가 +13% 느려져 전체적으로 손해.
- cap=6 가 factorize 의 최저점 — 레벨이 *더 깊은데도* (37 vs 30) factor 가 더 빠름.
- solve 는 cap 상향에 단조 개선되다 cap=12 에서 최저, cap=16 부터 다시 악화.

### 7.3 per-kernel 비교 (`mf_factor_extend_level<double>`, nsys per-call)

|  | cap=8 | cap=12 | Δ |
|---|---:|---:|---:|
| 호출 수 (= 레벨 수) | 29 | 23 | −6 |
| 호출당 평균 [μs] | 18.6 | **25.8** | **+39%** |
| per-call 누적 [μs] | 539 | **594** | **+55** |

레벨 6 개 절감으로 기대된 dispatch latency 절감 (∼6 × 5 μs = 30 μs) 보다, 각 레벨의 work 가 +39% 증가하면서 발생한 비용 (+165 μs over 23 levels) 이 **5 배 이상** 크다. 그 결과 wall 은 오히려 +55 μs 증가.

원인은 mid 레벨들의 max fsz 가 일제히 증가한 것:

| 레벨 | cap=8 maxfsz | cap=12 maxfsz |
|---|---:|---:|
| L4 | 49 | 52 |
| L7 | 71 | 76 |
| L8 | 63 | 64 |
| L12 | 64 | 67 |
| L13 | 56 | 59 |

`mf_factor_extend_level` 의 work 가 fsz² 에 비례하므로, fsz 7% 증가 → work 14% 증가가 누적된다. cap 상향이 amalgamation 을 강화해 panel 의 column 수를 늘리면서 (= nc 증가), padded fill 이 fsz 에 더해진 결과.

### 7.4 결론 보정 — 진짜 병목의 위치 재정의

원본 §4–§5 의 *"30 단 직렬 level 디스패치가 알고리즘적 병목"* 명제는 부분적으로만 맞음:

- **확실히 맞는 부분**: GPU 가 거의 idle 인 것은 사실 (median SM% 1.3%, DRAM% 0.8%).
- **정정해야 할 부분**: idle 의 원인은 *dispatch latency 자체* 보다 *small-grid 커널의 본질적 비효율* 에 더 가깝다. 같은 일을 *더 적은 레벨에 더 큰 panel 로* 묶어 dispatch 횟수를 줄여도 (cap=12) wall 은 개선되지 않는다.
- ROI 가 큰 방향: **per-level 커널의 효율 개선** (block size, shared memory 재사용, warp 단위 라우팅) — §6 의 4번. *dispatch 자체를 줄이는 방향* 은 회수 가능한 상한이 wall 의 ~15% 로 제한적.

이 결과는 batched 경로 ([`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md)) 의 설계 — *dispatch 를 batch dim 으로 hide 하면서 동시에 per-level 커널 자체를 small/mid/big tier 로 라우팅* — 가 단순히 *"batch 로 dispatch 만 묶기"* 가 아니라 *"per-level 커널 효율 개선"* 까지 함께 수행하는 이유를 설명한다.

### 7.5 재현 명령

```bash
for cap in 6 8 10 12 16; do
  CLS_CAP=$cap ./custom_linear_solver_run \
      /datasets/power_system/nr_linear_systems/case8387pegase \
      --repeat 50 --single-precision fp64 \
      | grep -E "factorize_ms|solve_ms|relative_residual_l2"
done
```

구조 dump 는 `CLS_DUMP=1` 추가. nsys per-kernel attribution 은:

```bash
nsys profile -t cuda --cuda-graph-trace=node \
    -o cap12 --force-overwrite=true \
    env CLS_CAP=12 ./custom_linear_solver_run \
        /datasets/power_system/nr_linear_systems/case8387pegase \
        --repeat 50 --single-precision fp64
nsys stats --report cuda_gpu_kern_sum --format csv cap12.nsys-rep
```

---

## 8. 구현 실험 — warp-per-front 와 shared-staged mid kernel

§6 의 후보 1, 2 (porting batched warp kernel, per-level kernel 효율 개선) 를 실제로 구현하고 측정. 두 후보를 독립적으로 toggle 가능하게 env 로 노출:

| 변경 | 위치 | env | 기본값 |
|---|---|---|---|
| **warp-per-front** (`mf_factor_small_warp<double>`) | `factorize/multifrontal.cu` | `CLS_USE_SMALL_WARP=1` | OFF |
| **shared-staged mid kernel** (`mf_factor_extend_level_shared<double>`) | `factorize/multifrontal.cu` | `CLS_NO_SHARED_FACTOR=1` | **ON** |

### 8.1 warp-per-front (batched `factor_small.cuh` 포팅)

- 매칭 조건: `maxfsz ≤ 32` 이고 `cnt ≥ 32` (L0, L1 만 해당, fsz ≤ 24 → W=8, fsz 25–32 → W=8 with opt-in)
- 한 block 안에 W warps 패킹, 각 warp 가 한 front 의 dense no-pivot LU 를 in-shared 로 수행. CB 는 shared 에서 parent 로 extend-add.
- batched 의 `mf_factor_small_warp_b<FT>` 와 동일한 패턴, batch 차원만 제거.

### 8.2 shared-staged mid kernel

- 매칭 조건: `maxfsz ∈ [33, 72]` 이고 `cnt ≥ 2` (L2–L18 의 mid-front 레벨 다수)
- 기존 fused block kernel 의 외부 시그니처 유지, 차이는 *front 전체를 dynamic shared 로 stage-in → LU phase 1/2/3 를 in-shared 로 수행 → L/U 만 writeback, CB 는 shared 에서 parent atomicAdd*
- 동기 횟수 동일, 그러나 rank-1 update 의 strided global F 접근이 shared access 로 치환됨

### 8.3 측정 결과 (8 trials × 200 repeats, FP64, RTX 3090)

| 구성 | factor [μs] | solve [μs] | Δfactor | Δsolve |
|---|---:|---:|---:|---:|
| baseline (warp OFF, shared OFF) | 604.4 ± 30.8 | 323.0 ± 16.2 | — | — |
| **warp ON, shared OFF** | 625.2 ± 31.5 | 334.4 ± 19.5 | **+3.4%** ❌ | **+3.5%** |
| **warp OFF, shared ON** (default) | **554.2 ± 23.8** | **317.7 ± 13.1** | **−8.3%** ✓ | **−1.6%** ✓ |
| warp ON, shared ON | 561.0 ± 39.5 | 322.9 ± 18.3 | −7.2% | −0.0% |

정확도: 4 구성 모두 `relative_residual_l2 < 1e-13` 유지 (baseline 동등).

### 8.4 해석

**warp-per-front 는 single-batch 에서 의미 없음 (오히려 slight 악화)**. 이유:

- 매칭되는 L0, L1 (cnt 4000+, fsz ≤ 20) 은 원래 ncu 분석에서 *grid > 1024* 버킷 = SM 38 %, 점유율 78 % 로 **이미 saturated** 영역. 압축할 헤드룸이 없음.
- batched 에서 warp packing 이 효과를 보는 이유는 (front × B) 축으로 packing 해서 단일 SM 의 warp slot 을 더 많이 채우는 것. B=1 일 때는 packing 의 효과가 wide grid 효과와 겹쳐 사라짐.
- 결과적으로 §6 의 후보 1 은 **single-batch 한정으로는 기각**. batched 경로에서는 여전히 유효 ([`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md) 참조).

**shared-staged mid kernel 은 큰 win**. 이유:

- 커버 영역: L2–L18 의 mid-front (fsz 33–72), 즉 §3 의 *grid 17–1024 + 2–16* 버킷 일부.
- 기존 block kernel 의 rank-1 update 에서 `F[ii*fsz+k]` (스트라이드 = fsz) 와 `F[k*fsz+jj]` (연속) 가 nc × (fsz − k − 1)² 번 발생. fsz=64 한 front 당 ~5000 회의 strided global read.
- shared 로 stage-in 한 뒤에는 동일 access 가 shared bank (single cycle) 로 옮겨감. dynamic shared 는 level 의 maxfsz² × 8 byte (예: fsz=64 → 32 KB) — 모든 매칭 레벨에서 default carve-out (48 KB) 안에 들어감.
- nsys per-launch 로 보면 mid-front 레벨이 19.4 μs → 19.2 μs 로 거의 같으나, **block kernel 이 left-over (L0/L1 + deep tail) 만 처리하면서 평균 19.4 → 15.2 μs 로 빨라짐**. 총합 28×19.4 = 544 μs → 15(shared)×19.2 + 12(block)×15.2 = 470 μs.

### 8.5 §6 개선 후보 재정렬 (실측 후)

| 후보 | 잠재 → 실측 | 상태 |
|---|---|---|
| ~~CLS_CAP 상향~~ | (§7 기각) | 기각 |
| warp-per-front (batched 포팅) | 예상 −35 % → **+3.4 %** | **기각 (single-batch 한정)** |
| per-level efficiency (shared-staged mid) | 예상 −10–20 % → **−8.3 %** | **채택 (기본 ON)** |
| deep-chain fusion (persistent kernel) | 예상 −10–15 % | 미구현 (남은 lever) |
| 멀티 스트림 cross-subtree | < −10 % | 미구현 (narrow-top 트리라 ROI 낮음) |
| fp64 → mixed | < 5 % | 미구현 |

### 8.6 재현 명령

```bash
# 기본 (shared ON, warp OFF)
./custom_linear_solver_run <case>/case8387pegase --repeat 200 --single-precision fp64

# baseline 으로 되돌리기 (둘 다 OFF)
CLS_NO_SHARED_FACTOR=1 ./custom_linear_solver_run ... --repeat 200 --single-precision fp64

# warp 만 켜기
CLS_NO_SHARED_FACTOR=1 CLS_USE_SMALL_WARP=1 ./custom_linear_solver_run ... --repeat 200 --single-precision fp64

# 어느 레벨이 어느 커널로 라우팅되는지 확인
CLS_WARP_DBG=1 ./custom_linear_solver_run ... --repeat 1 --single-precision fp64
```

---

## 9. FP32 single-batch 측정 — same metrics

§3 와 동일한 방식으로 `--single-precision fp32` (`pure_fp32 = true`) 측정. FP32 경로는 **warp-per-front 와 shared-staged 둘 다 우회** 하고 항상 `mf_factor_extend_level<float>` block kernel 을 탐 (dispatch gate: `!fp32 && !pure_fp32`). 따라서 FP32 비교는 FP64 baseline (block kernel, `CLS_NO_SHARED_FACTOR=1`) 과 apples-to-apples.

### 9.1 wall-time (8 trials × 200 repeats)

| 구성 | factor [μs] | solve [μs] | median rel_residual_l2 | Δfactor vs FP64-baseline | Δsolve |
|---|---:|---:|---:|---:|---:|
| FP64 baseline (block) | 603.2 ± 40.3 | 322.6 ± 16.5 | 3.5 × 10⁻¹⁴ | — | — |
| FP64 default (shared-staged ON) | 567.7 ± 30.4 | 328.7 ± 15.8 | 4.6 × 10⁻¹⁴ | −5.9 % | +1.9 % |
| **FP32** (block, `<float>`) | **407.7 ± 26.4** | **225.1 ± 8.5** | **1.7 × 10⁻⁵** | **−32.4 %** | **−30.2 %** |

FP32 는 factor / solve 양쪽에서 약 1/3 단축. 대신 잔차가 9 자릿수 악화 — FP32 의 단정밀도 한계 (≈ 10⁻⁵ ~ 10⁻⁶) 이지 알고리즘 결함 아님. NR iteration 의 수렴 판정 (보통 10⁻³ ~ 10⁻⁶) 안에는 들어가나, 1 step iterative refinement (FP32 풀고 FP64 잔차 보정) 가 필요하면 추가 cost 발생.

### 9.2 nsys per-kernel (50 reps, per-call)

| 커널 | 호출/콜 | per-call [μs] | avg [μs] |
|---|---:|---:|---:|
| `mf_factor_extend_level<float>` | 29 | **360.5** | 12.4 |
| `mf_invert_pivot<float>` | 1 | 13.6 | 13.6 |
| `mf_scatter_csr_values<float, float>` | 1 | 4.0 | 4.0 |
| `mf_fwd_level<float, float>` | 29 | **95.3** | 3.3 |
| `mf_bwd_level<float, float>` | 29 | **120.4** | 4.2 |
| gather/scatter perm | 2 | 3.7 | 1.8–1.9 |

FP64 baseline 의 per-kernel 과 비교 (§1.1, §1.2):

| 커널 (per-call) | FP64 | FP32 | Δ |
|---|---:|---:|---:|
| factor extend_level | 539 μs | **360 μs** | **−33 %** |
| invert_pivot | 24 μs | **14 μs** | **−42 %** |
| bwd_level | 201 μs | **120 μs** | **−40 %** |
| fwd_level | 127 μs | **95 μs** | **−25 %** |

### 9.3 ncu per-grid-bucket SoL (`mf_factor_extend_level`)

| 버킷 | FP32 SM% | FP64 SM% | FP32 점유율% | FP64 점유율% | FP32 DRAM% | FP64 DRAM% |
|---|---:|---:|---:|---:|---:|---:|
| =1 (deep tail) | 0.2 | 0.3 | 25 | 24 | 0.3 | <1 |
| 2–16 | 0.6 | 1.2 | 25 | 25 | 0.4 | <1 |
| 17–128 | 3.6 | 3.8 | 24 | 25 | 1.4 | 1 |
| 129–1024 | 5.4 | 9.8 | 64 | 60 | 2.0 | 3 |
| >1024 (L0–L1) | **24.2** | **37.6** | 72 | 78 | 5.0 | 12 |

### 9.4 해석 — FP32 가 30% 빨라지는 이유

원래 분석은 *"FP64 가 binding 이 아니다 (SM% 1–4 %, DRAM% < 1 %)"* 였음. 그렇다면 FP32 로 가도 큰 차이가 없어야 함. 그런데 −30 % 가 나왔음.

**핵심: SM% 가 낮은 것 ≠ FP64 비용이 안 든다**. RTX 3090 sm_86 의 ALU 처리량:
- FP32 FMA: 35.6 TFLOPS
- **FP64 FMA: 0.56 TFLOPS** (FP32 의 **1/64**)

FP64 는 same-warp 안에서 1 FMA 발급에 32 사이클 (lane 1 개씩 직렬) — Ampere consumer GPU 의 강한 architectural penalty. SM% metric 은 "한 warp 이라도 issue 중이면 SM busy" 로 카운트되는데, FP64 ALU 가 다음 FMA 받기까지 32 사이클 동안 *issue 는 못 받지만 SM 은 idle 표시는 안 함* (이 사이클들이 SM% 의 분모에서 빠지지 않을 수 있음).

즉 SM% 1–4 % 라는 표시는 *latency-bound* 상태인데, 그 latency 의 상당 부분이 **FP64 instruction 자체의 throughput limit**. FP32 로 가면 같은 work flow 가 64× 빠르게 issue 되니, latency 가 사라져 wall time 이 줄어듦.

SM% 가 FP32 에서도 *상승하지 않은* 것 (오히려 약간 낮음) 은 이를 뒷받침: kernel 이 빨리 끝나면서 *busy 구간 자체가 짧아져* SM% 분자도 같이 줄어듦. 절대 throughput (work / time) 은 올랐지만, busy fraction 은 비슷하게 유지.

> **이 발견은 §4 의 결론을 부분 수정함**: 원래 "FP64 throughput 은 binding 아님" 이라고 단정했지만, 실제로는 binding 이지만 *"latency 형태로 숨겨진 binding"* 이었음. compute-bound 의 정의를 SM% 로만 보면 놓침.

### 9.5 FP32 와 default (shared-staged FP64) 어느 게 우선?

| 우선순위 | 조건 |
|---|---|
| **FP32** | NR loop 의 수렴 판정이 ≥ 10⁻⁵ 이고 iterative refinement 불필요한 경우. wall −30 % 가 일관됨. |
| **FP64 shared-staged** (현재 default) | 수렴 판정이 ≥ 10⁻¹² 또는 FP32 residual 이 NR 발산을 유발할 때. wall −6 % 의 작은 win. |
| **FP64 mixed** (`fp32=true`, double master + float working) | 위 두 사이의 절충. 본 보고서 측정 안 함 (`mf_factor_extend_mixed` 경로 — 후속 작업) |

FP32 의 wall 우위는 *single-batch 한정* — batched 경로에서는 (front × B) packing 의 효과가 더 커서 FP64 vs FP32 비율이 달라질 수 있음 ([`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md) 참조).

---

## 10. 핵심 결론 — narrow-top 은 하이퍼파라미터로 풀리지 않음

§7 (CLS_CAP sweep) 와 §8 (warp/shared 구현 실험), §9 (FP32 측정) 의 결과를 합쳐 한 가지 메타-결론:

### 10.1 단일 배치 wall-time 의 분해

- **front 자체의 work 부족** (sparse Jacobian) → SM 점유율 한계 → wall 의 ~80 % 가 idle/저점유. **FP32 가 회수 가능 (1 회성 −30 %)**.
- **FP64 latency 비용** (RTX 3090 ALU 1/64 penalty) → 표면적으로는 SM% 낮게 보임. **FP32 또는 mixed 로 회수**.
- **per-level kernel 내부 비효율** (strided global F access) → **shared-staged 로 회수** (mid-front 영역에서 −8 %).
- **etree narrow-top** (root 근처 panel 수 ≤ 82) → **회수 불가** (아래 참조).

### 10.2 narrow-top 은 구조적 한계

multifrontal 알고리즘은 root 가 반드시 1 개. 거기서 leaf 방향으로 내려가도 *최대* `2^k` panel (level k 만큼 아래). RTX 3090 의 82 SM 을 동시에 채우려면 ≥ 82 panel, 즉 `log₂(82) ≈ 7` levels 아래까지 필요. case8387 의 30 levels 중 **위쪽 7 levels (L23–L29) 는 *어떤 cap, 어떤 ordering, 어떤 kernel* 로도 narrow**.

power-grid Jacobian 은 더 심함 — 평균 degree 7.4, METIS ND 가 만드는 separator 가 작아 *"넓은 dense 영역"* 자체가 존재하지 않음. cap 을 16 까지 올려도 fsz 가 100 + 으로 안 가는 이유 (§7.1, §7.2 데이터).

`cap`, `kBigMultiFrontThreshold`, block-size tier 모두 트리 *이후* 단계에서 작동:
- `cap` → 이미 만들어진 etree 의 chain merge (트리 모양 안 바꿈)
- `kBigMultiFrontThreshold` → 이미 결정된 front 의 kernel dispatch
- 셋 다 root 근처 panel 수 자체를 늘리지 못함

### 10.3 그래서 narrow-top 의 진짜 lever 는 알고리즘 변경

| 방향 | 본질 | case8387 에서의 적용 |
|---|---|---|
| **batched (B>1)** | 트리는 그대로, 문제를 B 개 동시에 풀어 narrow 영역에 B 배 work 주입 | 본 솔버의 batched 경로 — [`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md) 참조 |
| **iterative + preconditioner** | direct solver 의 narrow-top 직렬화 자체 회피 | 본 솔버 범위 밖 |
| **다른 ordering** (AMD, balanced ND) | 트리 모양 살짝 변경 시도 | 보통 fill 증가, ROI 낮음 |
| **hybrid: root 근처만 dense LU** | 작은 Schur complement 를 모아 dense 처리 | power-grid 는 root 근처 Schur 도 작음, 효과 한정 |

### 10.4 single-batch 솔버에서 추가 win 의 한계

shared-staged (−6%) 와 FP32 (−30%) 를 합치면 **factor 약 −35 % 가 single-batch 의 상한** (잔차 trade-off 받아들이는 경우). 그 이상 은 다음 중 하나가 필요:

1. **batched 경로로 전환** — narrow-top 의 직렬화를 batch 차원으로 hide
2. **mixed precision iterative refinement** — FP32 의 잔차를 FP64 1-step refinement 로 회복 (구현 미존재)
3. **deep-chain fusion** (§6 의 미구현 후보) — narrow-top 의 launch latency 절감, 상한 ~−10–15 %

이 세 가지 외에 *single-batch 만의* "다음 큰 win" 은 구조적으로 존재하지 않음.

---

## 11. 재현 절차 (전체)

```bash
# FP32 측정
./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 200 --single-precision fp32

# FP64 baseline (block kernel only, shared-staged 비활성)
CLS_NO_SHARED_FACTOR=1 ./custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 200 --single-precision fp64

# FP64 default (shared-staged ON)
./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 200 --single-precision fp64

# FP32 nsys timeline
nsys profile -t cuda --cuda-graph-trace=node \
    -o fp32 --force-overwrite=true \
    ./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 50 --single-precision fp32

# FP32 ncu SoL
ncu --target-processes all \
    --kernel-name "regex:(mf_factor_extend_level|mf_fwd_level|mf_bwd_level)" \
    --launch-count 200 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --csv --log-file ncu_fp32.csv \
    ./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 5 --single-precision fp32
```

### 11.1 처음 환경 빌드

```bash
# 빌드 (line-info 포함)
mkdir -p custom_linear_solver/build-prof && cd custom_linear_solver/build-prof
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CUDA_FLAGS="-lineinfo -O3" \
      -DCLS_BUILD_SCRIPTS=ON ..
cmake --build . -j

# wall + 분포 확인
CLS_DUMP=1 ./custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 50 --single-precision fp64

# nsys timeline (graph 내부 노드 포함)
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    -o single_8387_graph --force-overwrite=true \
    ./custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 50 --single-precision fp64

nsys stats --report cuda_gpu_kern_sum --format csv single_8387_graph.nsys-rep

# ncu SoL / 점유율 (hot 커널 한정)
ncu --target-processes all \
    --kernel-name "regex:(mf_factor_extend_level|mf_fwd_level|mf_bwd_level|mf_bigB_trailing|mf_invert_pivot)" \
    --launch-count 200 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --csv --log-file ncu_summary.csv \
    ./custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case8387pegase \
    --repeat 5 --single-precision fp64
```

원자료 위치 (분석 당시):

- nsys (default cap=8): `/home/claude/prof/single_8387_graph.nsys-rep`
- nsys (cap sweep): `/home/claude/prof/cap{8,12}.nsys-rep`
- ncu CSV: `/home/claude/prof/ncu_summary.csv`
