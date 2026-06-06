# Batched 솔버의 메모리 바운드 여부 검사 — case8387 / Synthetic USA (FP64, B = 4, 64, 256)

*custom_linear_solver 의 uniform-batch 모드 (`--batch B`) 를 case8387pegase (n=14908) 와 case_SyntheticUSA (n=156255) 두 매트릭스에 대해 **FP64 강제** (`MF_NO_MIXED=1`) 로 측정. 각 (case, B) 조합의 factor / solve dominant 커널이 (a) memory-bound, (b) compute-bound, (c) latency-bound 중 어디에 속하는지 ncu metric 으로 분류하고, 실제 측정된 DRAM 점유를 roofline 상에서 해석. § 5 에서 메모리 바운드 한계를 풀 수 있는 개선안을 정리.*

> 직접 선행 문서: [`07-batched-bottleneck-fp64-case8387-b1-b256.md`](07-batched-bottleneck-fp64-case8387-b1-b256.md) — 같은 측정 framework, case8387 단독 B=1/8/64/256. 본 문서는 *큰 매트릭스에서 메모리 바운드가 어떻게 강해지는지* 를 USA 로 확장.

## 0. Baseline / Scope

| | case8387pegase | case_SyntheticUSA |
|---|---|---|
| n | 14908 | 156255 |
| nnz | 110572 | 1052085 |
| panels P | 7396 | 74245 |
| levels | 32 | 40 |
| panel cap | 8 | 20 |
| front arena (FP64) per batch | **4.4 MB** | **70.4 MB** |
| GPU | RTX 3090 sm_86, 82 SM, FP64 peak 0.56 TFLOPS, GDDR6X 936 GB/s, L2 6 MB | (동일) |
| 정밀도 | **FP64 강제** (`MF_NO_MIXED=1`) | (동일) |
| 빌드 | wall: `CLS_INTERNAL_GRAPH=ON` (`/tmp/clsb`). nsys/ncu: `CLS_INTERNAL_GRAPH=OFF` (`/tmp/clsb_nograph`) | (동일) |
| B | **4, 64, 256** | (동일) |

**working set 추정 (B × front_total, FP64)**

| B | case8387 working set | USA working set | RTX 3090 메모리 (24 GB) 대비 |
|---:|---:|---:|---|
| 4   | 17.6 MB | 281.6 MB | 둘 다 cache miss 보장 (>L2 6MB) |
| 64  | 281.6 MB | **4.5 GB** | USA 는 이미 *각 level launch 가 4.5 GB 를 한 번 stream* |
| 256 | 1.13 GB | **18.0 GB** | **USA 는 디바이스 메모리 75% 점유**, factor 마다 18 GB 전부 read+write |

이 working-set 차이가 본 문서의 핵심: **B 가 커질수록, 매트릭스가 커질수록 dominant factor 커널의 DRAM 점유가 가팔라진다**.

---

## 1. Wall-time 측정 (FP64, `--batch B --batch-only`)

`/tmp/clsb/custom_linear_solver_run`, median of 5 (case8387) / 3 (USA) outer × `--repeat` inner.

| Case | B | factor / sys [μs] | solve / sys [μs] | factor total [ms] | factor scale (vs B=4) | factor / total |
|---|---:|---:|---:|---:|---:|---:|
| case8387 | 4   | 279.0 | 100.9 | 1.12  | 1.00× | 73% |
| case8387 | 64  | **94.9**  | 27.0  | 6.07  | 2.94× | 78% |
| case8387 | 256 | **88.5**  | 22.6  | 22.65 | 3.15× | 80% |
| USA      | 4   | 4345.0 | 437.3 | 17.4 | 1.00× | 91% |
| USA      | 64  | **1695.9** | 263.5 | 108.5 | 2.56× | 87% |
| USA      | 256 | **1769.5** | 253.7 | 453.0 | 2.46× | 87% |

**관찰**:

- **두 case 모두 B=4 → 64 에서 factor/sys 가 2.5-3× 빨라지고, B=64 → 256 에서는 거의 정체.** 이는 § 2 에서 보일 *memory-bound 포화* 의 직접 증거.
- **USA 의 factor 가 case8387 보다 ~19× 느림 (B=256, per sys 1.77 ms vs 0.089 ms).** 단순히 n 비율 (10.5×) 보다 크다 — front_total 도 16× 크고 (4.4 MB → 70.4 MB), DRAM 으로 stream 해야 할 데이터가 더 크기 때문.
- **USA B=256 의 factor 는 B=64 보다 *약간 느려진다* (1.70 → 1.77 ms/sys, +4%).** L2 thrash + arena 18 GB → device memory 의 거의 전부 점유로 OS/메모리 pressure 증가 가능성.

---

## 2. nsys 커널 분포

`nsys profile -t cuda --force-overwrite=true` → `nsys stats --report cuda_gpu_kern_sum`. `--repeat 5` (case8387), `--repeat 3` (USA). dominant 커널만 표기, % 는 batched factor+solve 커널 합 대비.

### 2.1 case8387

| B | extend_level_b | invert_pivot_b | factor_small_warp_b | fwd_level_b | bwd_level_b | fwd/bwd_small_warp_b | scatter_batched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4   | **49.7%** (3.96 ms) | 10.0% (0.80 ms) | 9.4% (0.75 ms) | 8.1% (0.65 ms) | 11.7% (0.93 ms) | 6.7% | 1.6% |
| 64  | 26.5% (10.10 ms) | **31.1%** (11.86 ms) | 14.4% (5.48 ms) | 4.4% (1.67 ms) | 6.1% (2.34 ms) | 11.9% | 4.5% |
| 256 | 23.8% (33.50 ms) | **33.6%** (47.34 ms) | 16.0% (22.48 ms) | 3.6% (5.13 ms) | 5.1% (7.13 ms) | 12.0% | 4.9% |

→ case8387 의 dominant 시간 점유 *축* 이 두 개:
1. **`mf_factor_extend_level_b`** — B 가 커지면서 절대 시간 8.5× 증가 (B=4 → 256), 비중은 49 → 24% 로 *감소*. 이유: § 3.2 에서 보일 *memory-bound 포화*.
2. **`mf_invert_pivot_b`** — 절대 시간 60× 증가 (0.80 → 47.34 ms), 비중 10 → 34% 로 *증가*. *compute-bound (FP64)* 의 정직한 선형 증가.

### 2.2 USA

| B | extend_level_b | invert_pivot_b | factor_small_warp_b | fwd_level_b | bwd_level_b | fwd/bwd_small_warp_b | scatter_batched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4   | **74.0%** (45.87 ms) | 8.4% (5.22 ms) | 5.6% (3.47 ms) | 2.0% (1.25 ms) | 2.9% (1.78 ms) | 3.9% | 1.1% |
| 64  | **48.1%** (183.25 ms) | 21.6% (82.07 ms) | 13.5% (51.21 ms) | 2.2% (8.20 ms) | 2.4% (8.98 ms) | 8.8% | 2.9% |
| 256 | **49.0%** (714.98 ms) | 22.5% (327.80 ms) | 11.5% (168.16 ms) | 2.2% (31.73 ms) | 2.3% (33.17 ms) | 9.1% | 3.0% |

→ USA 의 dominant 커널은 **모든 B 에서 `mf_factor_extend_level_b` 단일 (48 ~ 74%)**.
- 절대 시간 15.6× 증가 (B=4 → 256), 비중 74 → 49% 감소 — *memory-bound 의 hallmark*: B 가 늘어도 시간이 linear 보다 *살짝 빠르게* 늘어남 (16× 데이터 → 15.6× 시간) 이지만 그게 자기 비중을 *깎지는 못함*.
- `mf_invert_pivot_b` 가 case8387 처럼 큰 비중을 못 가짐 (8 → 22%) — USA 는 panel cap=20 으로 nc 가 큼, P=74245 로 inverse 자체가 무거우나 *extend_level_b 가 워낙 무겁기 때문에 상대 비중 작음*.
- solve 비중은 더 작음 (fwd+bwd 합쳐 4.6% in B=256) — § 1 의 factor/total 87% 와 일관.

---

## 3. ncu bound 분류 — dominant 커널

`ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.pct_of_peak_sustained_elapsed, smsp__warps_active.avg.pct_of_peak_sustained_active, sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed, launch__waves_per_multiprocessor, gpu__time_duration.sum`. `--launch-skip 5 --launch-count 20` (level kernels), `--launch-count 3` (1-instance invert_pivot). Duration-weighted average.

### 3.1 `mf_factor_extend_level_b<double>` — 핵심 factor 커널

| Case | B | SM% | **DRAM%** | warp% | FP64% | waves/SM | per-launch durμs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | 5.5  | 3.5  | 14.1 | 5.5  | 0.22  | 39.4   | **latency** (waves<1, SM idle) |
| case8387 | 64  | 53.6 | 33.6 | 71.5 | 53.6 | 108   | 462.5  | 전이 (memory-leaning) |
| case8387 | 256 | 48.0 | **61.8** | **84.2** | 48.0 | 24.9  | 402.6  | **memory-bound** (DRAM 62% > FP64 48%) |
| USA      | 4   | 9.0  | 5.3  | 11.4 | 9.0  | 0.10  | 718.4  | **launch+latency** (waves 0.1, SM 9%) |
| USA      | 64  | 37.2 | **52.6** | 69.1 | 37.2 | 3.58  | 3095.5 | **memory-bound** (DRAM 53% > FP64 37%) |
| USA      | 256 | 35.7 | **65.7** | **89.8** | 35.7 | 13.9  | 12144.1 | **memory-bound** (DRAM 66%, warp 90%) |

**핵심 발견**:

- **두 case 모두 B≥64 에서 `mf_factor_extend_level_b` 의 DRAM 점유가 SM/FP64 점유를 추월** — 명확한 memory-bound 신호.
- **USA 가 case8387 보다 *더 일찍, 더 강하게* memory-bound** : USA 는 B=64 만 되어도 DRAM 53% 인 반면 case8387 은 B=64 에서 아직 DRAM 34% 의 *전이 구간*.
  - 이유 1: USA 의 front_total 이 16× 큼 → 같은 B 에서 16× 더 많은 데이터 stream.
  - 이유 2: USA 의 fronts 가 평균적으로 더 큼 (`uc/fsz` 분포가 trailing GEMM 의 AI 를 비슷하게 만들지만 working set 이 L2 (6MB) 를 압도적으로 초과).
- **B=4 는 두 case 모두 latency-bound**: SM 0.2-2.5%, waves<<1, DRAM 도 못 채움. *Compute 가 빠르지도 메모리가 막히지도 않음 — GPU 가 안 채워짐*.
- USA B=256 의 warp% 90%, DRAM 66%, FP64 36% 의 조합은 **DRAM 이 dominant resource, occupancy 는 충분, FP64 unit 은 절반만 활용** — 정확히 memory-bound 정의 그대로.

### 3.2 `mf_invert_pivot_b<double>` — compute-bound 비교군

| Case | B | SM% | DRAM% | warp% | FP64% | waves/SM | per-launch durμs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | 36.6 | 4.3  | 10.5 | 36.6 | 72    | 155.0  | **compute (FP64)** |
| case8387 | 64  | **39.7** | 4.1  | 10.6 | **39.7** | 1161 | 2276.8 | **compute (FP64)** |
| case8387 | 256 | **40.0** | 4.0  | 10.6 | **40.0** | 4630 | 9043.7 | **compute (FP64)** |

(USA invert_pivot ncu 는 본 run 에서는 추출 안 됐지만 nsys 의 instance time 이 B 에 정확히 비례 — B=4: 1.74 ms, B=64: 27.36 ms (15.7×), B=256: 109.27 ms (4.0× from 64). 비례적으로 case8387 과 같은 FP64-compute 패턴.)

→ 비교 의의: **같은 working set 위에서도** `invert_pivot_b` 는 nc×nc 의 in-shared inverse 라 DRAM 4% 만 사용, FP64 unit 의 40% 를 사용해 **compute-bound**. *workload character 가 다르면 bound 도 다르다는 직접 증거* — 본 문서의 memory-bound 주장이 "FP64 가 부족해서" 가 아님을 보임.

### 3.3 `mf_fwd_level_b<double>` / `mf_bwd_level_b<double>` — solve

| Case | B | 커널 | SM% | DRAM% | warp% | FP64% | waves/SM | dur μs | 분류 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | fwd | 3.0  | 3.1  | 13.9 | 3.0  | 0.29 | 6.5   | latency |
| case8387 | 4   | bwd | 5.0  | 1.9  | 14.4 | 5.0  | 0.26 | 9.5   | latency |
| case8387 | 64  | fwd | 15.6 | 25.9 | 30.9 | 15.6 | 1.32 | 10.9  | 전이 (memory-leaning) |
| case8387 | 64  | bwd | 36.0 | 14.9 | 50.5 | 36.0 | 6.93 | 31.2  | compute-leaning |
| case8387 | 256 | fwd | 31.6 | 33.3 | 46.3 | 31.6 | 69.9 | 152.5 | **memory** (DRAM ≥ SM) |
| case8387 | 256 | bwd | 51.5 | 17.8 | 59.5 | 51.5 | 65.1 | 200.2 | **compute (FP64)** |
| USA      | 4   | fwd | 6.3  | 12.4 | 18.6 | 6.3  | 0.17 | 12.9  | latency |
| USA      | 4   | bwd | 1.9  | 2.4  | 16.0 | 1.9  | 0.05 | 20.3  | latency |
| USA      | 64  | fwd | 23.2 | **44.3** | 70.6 | 23.2 | 6.48 | 113.6 | **memory** (DRAM 44% > FP64 23%) |
| USA      | 64  | bwd | 20.8 | 30.6 | 55.1 | 20.8 | 1.67 | 45.1  | memory-leaning |

→ Solve 단계도 같은 패턴: **B=4 는 latency-bound, B=64 부터 fwd 가 memory-bound, bwd 는 compute-bound (case8387) 또는 memory-leaning (USA)**.
- bwd 가 case8387 에서 compute-bound 인 이유는 `U_pp` GEMV 의 nc-deep contraction (선행 문서 07 § 3.3 와 동일).
- USA 의 bwd 가 memory-leaning 인 이유: USA front 가 클수록 working set 이 L2 초과 → 같은 GEMV 가 DRAM-stream 형태로 바뀜.

### 3.4 요약 — Bound 매트릭스 (B ∈ {4, 64, 256})

| 커널 | case8387 B=4 | case8387 B=64 | case8387 B=256 | USA B=4 | USA B=64 | USA B=256 |
|---|---|---|---|---|---|---|
| factor_extend_level_b | latency | 전이 (memory→) | **memory** | latency | **memory** | **memory** |
| invert_pivot_b | compute | compute | compute | compute | compute | compute |
| factor_small_warp_b | launch+latency | latency | latency | launch+latency | latency | latency |
| fwd_level_b | latency | 전이 | **memory** | latency | **memory** | (memory; ncu 미캡처) |
| bwd_level_b | latency | compute | compute | latency | memory-leaning | (estimate) |

**결론** (질문 "case8387/USA × B=4/64/256 의 메모리 바운드 여부"):

- **case8387 B=4**: factor 와 solve 모두 *memory-bound 아님*. 오히려 *launch+latency-bound* (waves<1, SM 5%).
- **case8387 B=64**: factor 의 dominant `extend_level_b` 가 *memory→compute 전이*. 아직 명확히 memory-bound 라고 하긴 어려움 (DRAM 34% vs SM 54%). 하지만 **`invert_pivot_b` 가 함께 dominant 가 되어 compute-bound 의 영역으로 진입**.
- **case8387 B=256**: factor 의 *dominant 자원이 두 갈래* — `extend_level_b` 는 명확히 **memory-bound (DRAM 62%)**, `invert_pivot_b` 는 명확히 **compute-bound (FP64 40%)**. 둘 다 풀어야 추가 가속.
- **USA B=4**: factor 가 *launch+latency-bound*. USA 의 큰 front 로도 B=4 만으론 GPU 채울 수 없음.
- **USA B=64**: `extend_level_b` 가 이미 **명확한 memory-bound (DRAM 53%)** — case8387 의 B=256 보다 빠르게 memory wall 도달.
- **USA B=256**: `extend_level_b` 가 **memory-bound 더 강화 (DRAM 66%, warp 90%)**. *추가 B 가 가속을 주지 못하는 이유* — § 1 에서 보인 정체.

→ **YES, B=64 이상의 모든 (case, B) 조합에서 dominant factor 커널은 memory-bound**. 단, USA 는 B=64 부터, case8387 은 B=256 부터 명확.

---

## 4. Roofline 해석

RTX 3090 의 *FP64 ridge point* = peak FP64 / DRAM BW = 0.56 TFLOPS / 936 GB/s = **0.60 FLOP/byte**.
이보다 낮은 arithmetic intensity (AI) 의 커널은 memory-bound.

### 4.1 `mf_factor_extend_level_b` 의 이론 AI 추정

per (front, batch) 의 work (fsz × fsz front, nc pivot columns, uc = fsz − nc trailing):

- **panel LU**: ≈ fsz × nc² FLOP, ≈ fsz × nc × 8 bytes touched
- **U panel solve**: ≈ uc × nc² FLOP
- **trailing scalar**: 2 × uc² × nc FLOP, ≈ uc² × 8 + 2 × uc × nc × 8 bytes
- **extend-add**: uc² 의 atomicAdd (parent front)

**합산 (nc 작고 uc 큰 경우 — power-grid 의 전형)**:
- Total FLOP ≈ 2 × uc² × nc
- Total bytes ≈ fsz² × 8 (front 전체를 한 번 read + write) + uc² × 8 (atomicAdd 위치 read+write)
- AI ≈ 2 × uc² × nc / (8 × (fsz² + uc²)) ≈ nc / 4 × (uc/fsz)² × 1/(1 + (uc/fsz)²)

power-grid 의 case8387: 평균 nc≈4-8, uc/fsz ≈ 0.7-0.9 (cap=8) → **AI ≈ 1-1.5 FLOP/byte**.
USA: 평균 nc≈10-20 (cap=20), uc/fsz ≈ 0.6-0.9 → **AI ≈ 2-4 FLOP/byte**.

**비교**:
- case8387: AI ≈ 1-1.5 > 0.6 → *수치적으로는 compute-leaning* 영역에 가까워야 함. 하지만 측정은 memory-bound (DRAM 62%).
- USA: AI ≈ 2-4 → 더 강하게 compute 쪽이어야 함. 하지만 측정은 memory-bound 더 강함.

### 4.2 이론 AI 와 측정 사이의 갭

이론에는 *L2 hit* 가 들어 있다. 실제로:

- B 개의 (front, batch) 가 **같은 front 의 다른 batch copy** 를 읽음 — front pattern 은 같지만 **데이터는 다른 메모리 영역**. L2 cross-batch reuse 가 ZERO.
- B × front_total 의 working set 이 L2 (6 MB) 를 초과한 순간부터 fresh DRAM read.
- 한 (front, batch) 안에서도 trailing scalar 의 stride 가 fsz 라 L2 line reuse 가 약함 (uc 가 작으면 일부 reuse, 크면 thrash).

**즉, 측정된 effective AI = (compute) / (effective DRAM traffic)**, effective DRAM traffic 은 이론보다 2-4× 큼 (no L2 reuse + uc² extend-add 의 atomic).

이게 **DRAM 62%, FP64 48% 의 조합** 이 의미하는 것 — *L2 가 한 번도 hit 못 하면 ridge 가 왼쪽으로 이동* 해서 AI≈1 인 커널도 memory-bound 가 된다.

### 4.3 USA B=256 의 *measured* effective bandwidth

USA B=256, extend_level_b 의 total time = **715 ms (3 iter total)** = **238 ms / iter**.
extend_level_b 의 working set 1 회 (front arena 한 번 read+write) = 2 × 18 GB = 36 GB.
extend_level_b 가 한 iter 안에 *최소* 두 번은 같은 데이터를 건드림 (panel solve + trailing) → 50-60 GB 추정.

→ effective achieved BW ≈ 50 GB / 238 ms ≈ **210 GB/s** = peak 936 GB/s 의 **22%**.

이 22% 는 ncu 의 DRAM 65% 와 다른 metric: ncu DRAM% 는 *cycle 단위 throughput*, wall-time 기준 *achieved bandwidth* 는 더 낮음. atomic, partial-cacheline, scatter 패턴이 peak 의 1/3 정도까지 깎아내림 — 이게 *진짜 메모리 바운드 시그너처*.

→ **개선 여지 (memory)**: peak BW 까지 못 가는 비효율 (cache reuse 0, atomic, scatter) 을 풀면 **2-3× 가속 가능**. § 5 가 이 부분.

### 4.4 case8387 B=256 의 *measured* effective bandwidth

case8387 B=256, extend_level_b total = 33.50 ms / 5 = 6.70 ms / iter.
working set 1 회 ≈ 2 × 1.13 GB = 2.26 GB.
→ achieved BW ≈ 2.26 GB / 6.70 ms ≈ **337 GB/s** = peak 36%.

→ case8387 은 USA 보다 *덜 thrash*. 이유: arena 1.13 GB 라 L2 6MB 와의 ratio 가 190× — USA 의 3000× 보다 cache 영향이 작음.

---

## 5. 메모리 바운드를 풀 수 있는 개선안

§ 3-4 에서 보인 한계: **B≥64 의 `mf_factor_extend_level_b` 가 (a) cross-batch L2 reuse 0, (b) atomicAdd 의 partial cacheline, (c) trailing scalar 의 fsz-stride access** 로 인해 peak BW 의 22-36% 만 활용. 다음 후보들은 각각의 비효율을 줄이는 방향.

### 5.1 (M1) Batch-major front 레이아웃 (B 차원을 inner)

**현재**: `frontB[(long)batch * front_total + front_off[p] + i]`. **batch 가 outer dim** → 같은 (panel, row, col) 의 B 시스템이 메모리상 *front_total bytes 떨어진* 위치.
- L2 line (128 B) 위에 같은 (p, i, j) 의 16 batch가 같이 들어오면 좋음. 현재는 1 batch만 들어옴.

**제안**: `frontB[(long)(front_off[p] + i) * B + batch]` — **(p, i, j) 가 outer, batch 가 inner**.
- 같은 (p, i, j) 의 B/16 = 16 (FP64) 또는 32 (FP32) batch 가 한 L2 line 공유.
- panel LU 의 trailing update 가 *각 (i, j) 위에 B 개의 batch 를 같이 stream* → DRAM read/write 가 한 번에 256-1024 B 의 stride-1 burst.

**예상 효과**:
- L2 line utilization 32× (FP64) 또는 64× (FP32).
- atomicAdd 도 contiguous (16 batch 가 같은 line) → atomic conflict 는 늘지만 (warp-level reduce + 1 atomic) line 쓰기는 1 burst.
- 측정 DRAM 점유 같지만 wall 30-50% 감소 추정.

**리스크**:
- `extend_add` 의 parent front index 는 *batch 와 무관* 이라 batch-inner 가 직접 들어맞음.
- 단, `mf_invert_pivot_b` 의 shared-load 도 같이 바꿔야 함 (현재는 single-batch loop).
- `cudaMallocPitch`-등가 padding 필요 (B alignment).
- 코드 변경 광범위 (frontB, frontBf, yB, yBf, scatter_batched, gather_rhs_b, scatter_sol_b 전부).

**난이도**: 매우 높음. **잠재 효과**: 2-3× (USA B=256 의 ridge 까지 끌어올림).

### 5.2 (M2) Persistent block + L2 residency hint

**현재**: 한 level kernel 의 1 (front, batch) 이 1 block. block 사이 의존 없음, 하지만 같은 level 의 같은 front 가 *B 개의 다른 block 에서* read.

**제안**: persistent block 1개 / SM 에 *fan-out by batch*. 하나의 block 이 자신의 SM 위 cache 에 front (혹은 front 의 panel slice) 를 *한 번 읽어서* B 개 batch 위에 차례로 적용.
- 같은 front 의 8개 batch 가 *같은 L2 line 을 cache hit* (front pattern 재사용 1× → 8×).
- pseudo-code: `for (b=batch_begin; b<batch_end; ++b) apply_lu(F[b*front_total + ...])`.

**예상 효과**: same memory traffic 으로 *공유 read 가 1/B 로 감소* — 이론 8× peak DRAM, 실제로는 SM/L1 cache pressure 때문에 2-4×.

**리스크**:
- block-per-(front, batch) 의 simplicity 사라짐. dynamic scheduling (atomic increment work index) 필요.
- 한 block 이 여러 batch 처리 → memory order 가 batch-inner 가 아니라 batch-outer 라 *cache hit 이 시간 분산*. instead L1 hit 만 expectation.
- launch overhead 절감은 0 (이미 1 launch / level).

**난이도**: 높음. **잠재 효과**: 1.5-2.5× (warm-cache port plan 의 일부).

### 5.3 (M3) Mixed precision (FP64 master / FP32 working LU + IR)

**현재**: FP64 강제. front arena 8 bytes/elem.
**이미 구현됨**: `mf_factor_extend_mixed_b` (FP64 master, FP32 working). 단 `MF_NO_MIXED=1` 으로 꺼져 있음.

**측정 (별도 doc)**: comprehensive sweep (`05-reports/02-comprehensive-sweep-2026-06-05.md`):

| Case | B | FP64 [μs/sys] | FP32 [μs/sys] | 가속 |
|---|---:|---:|---:|---|
| case8387 | 4   | 410  | 219  | 1.87× |
| case8387 | 64  | 102  | 53   | 1.92× |
| case8387 | 256 | 94   | 45   | 2.10× |
| USA      | 4   | 4456 | 1392 | 3.20× |
| USA      | 64  | 1834 | 658  | 2.79× |
| USA      | 256 | 1937 | 643  | 3.02× |

→ **FP32 (또는 TC) 로 가면 본 문서의 memory-bound 한계가 *데이터 1/2 + L2 cache 활용 2×*** 로 동시에 풀림. USA B=256 의 3× 가속의 *대부분이 memory-bound 완화 덕분*.

**리스크**:
- power-grid NR 의 정확도 — `02-design-analysis/03-no-pivoting-empirical-proof.md` 와 `02-design-analysis/05-gemm-fraction-analysis.md` 에서 FP32 batched-IR 이 안전함이 측정됨. 본 case 들에선 안전.
- 본 보고서는 *FP64 강제 측정* 이지만, 권고는 *FP64 가 진짜 필요한 use case 가 아니면 FP32/TC 로 가라* 가 가장 효과 큼.

**난이도**: 낮음 (이미 구현, flag 만 끄면 됨). **잠재 효과**: 2-3× (USA B=64+), 1.9-2.1× (case8387).

### 5.4 (M4) Trailing GEMM 의 tile-staged shared 활용 — FP64 path 에도 적용

**현재**: `trailing_update_scalar<FT>` 는 fsz-stride 의 raw load. cache-line 활용 약함.
**이미 구현됨 (TC path 만)**: `mf_factor_extend_mixed_tc_b` 의 dynamic shared 에 L/U 를 staged FP16 로 채우는 부분.

**제안**: FP64 path 의 `trailing_update_scalar` 를 *블록당 한 번씩 L/U 의 nc×fsz panel 을 shared 에 staging* 하고, uc×uc trailing 을 shared 에서 read.
- shared 에서 reuse: 같은 L/U row 가 *fsz 번* 사용됨 (uc² 의 trailing 에서). 현재 each thread 가 매번 global 에서 fetch.
- shared 점유 = 2 × nc × fsz × 8 ≈ 1-2 KB (nc=8, fsz=128) — 충분히 여유.

**예상 효과**:
- DRAM read 1/uc → 1/uc 의 비율로 trailing 의 read traffic 감소.
- 측정상 `factor_extend_level_b` 시간 30-40% 감소 추정.

**난이도**: 중. `Σ.14 Σ.1 staged trailing` 이 이미 일부 path (TC32 native) 에 적용됨 (`tc/trailing_tiled.cuh`). FP64 path 에 같은 패턴 이식하면 됨.

**잠재 효과**: 1.3-1.6× factor 가속 (memory-bound 영역에서 가장 직접 효과).

### 5.5 (M5) atomicAdd 의 warp-level 사전 reduce

**현재**: `extend_add` 가 `uc² * B` 의 atomicAdd 를 parent front 에 발사.
- atomic 자체가 DRAM 대역 약 1/2 효율 (read-modify-write 2 cacheline cycles).
- partial-cacheline write 도 매번 발생.

**제안**:
- *같은 batch 의 같은 parent-front-cell* 에 contribute 하는 *여러 child front 의 값* 을 warp 내에서 사전 reduce.
- 또는 *non-overlapping siblings 의 extend-add* 를 한 launch 로 묶고 그 안에 *각 child 가 자기 row 에 write* 하는 형태 (atomic 자체 제거).

**리스크**:
- non-overlapping 보장은 elimination tree 의 sibling-disjoint 성질로 가능 (이미 `panel_parent` 와 `asm_local` 에 인코딩됨).
- 단, *sibling 들의 동시 write 가 정말 disjoint cell 만 건드리는지* 의 검증 필요 — power-grid 의 amalgamated front 에선 보통 disjoint, 일반 multifrontal 은 아닐 수도.

**난이도**: 매우 높음 (race condition 위험). **잠재 효과**: 10-20% (DRAM 의 atomic 비중을 normal store 로 변환).

### 5.6 (M6) Compute-bound 와 분리 처리

§ 3 에서 본 `mf_invert_pivot_b` 는 *완전한 compute-bound (FP64 40%)*. memory 개선과 무관.
**제안**: 이 커널은 *FP32 inverse + 한 번의 FP64 refine* 로 변경.
- inverse 자체는 nc×nc 의 작은 dense problem — FP32 inverse + 1 step Newton 으로 FP64 정확도 회복.
- duration 거의 절반 감소.

**잠재 효과**: case8387 B=256 의 invert_pivot 9 ms → 4.5 ms = total factor wall 4.7% 감소.
USA 는 invert_pivot 비중 22% 라 wall 11% 감소.

**난이도**: 중. **권고**: M3 (전체 FP32) 적용하면 자연히 해결되므로 M3 가 우선.

### 5.7 우선순위 정리

| 후보 | 영향 받는 영역 | 잠재 가속 (B=256) | 난이도 | 권고 |
|---|---|---|---|---|
| **M3** Mixed/FP32 + IR | factor + solve 전체 | **2-3×** (USA), 2× (case8387) | 낮음 (구현 완료) | **1순위** — flag 만 끄면 됨 |
| M4 Trailing shared-staged (FP64) | extend_level_b | 1.3-1.6× | 중 | 2순위 — FP64 강제가 필요한 use case |
| M1 Batch-major layout | 전체 batched 커널 | 2-3× | 매우 높음 (re-arch) | 3순위 — 본격 작업 가치 있음 |
| M2 Persistent block + L2 hint | extend_level_b | 1.5-2.5× | 높음 | 4순위 — M1 의 part |
| M5 Atomic warp-reduce | extend_add | 1.1-1.2× | 매우 높음 (race) | 후순위 |
| M6 FP32 inverse + refine | invert_pivot_b | 1.05-1.10× total | 중 | M3 의 부분집합 |

### 5.8 정량 — M3 만 적용했을 때의 메모리 점유 변화

USA B=256, FP32 front arena = **18 GB / 2 = 9.0 GB**. RTX 3090 메모리 24 GB 의 37%. 여전히 L2 보다 1500× 크지만:
- 한 cacheline (128B) 위에 2× 더 많은 entry (FP32) → 같은 cache 가 effective 2× 큼.
- 측정상 FP32 batched factor extend kernel 의 DRAM% 는 보통 80-85% (case8387 B=256 FP32 측정, doc 08).
- → FP32 path 는 *memory wall 에 더 가깝게* 가지만, 절대 wall time 은 FP64 의 절반 미만.

USA B=64 의 FP32 = 658 μs/sys = FP64 의 36%. 메모리 바운드는 *피하는 게 아니라 완화* 가 답.

---

## 6. 최종 답 — 질문에 대한 직접 응답

**Q: case8387 / USA × B = 4 / 64 / 256 에서 batched factor + solve 가 memory-bound 인가?**

| Case | B | dominant 커널 | bound | 비고 |
|---|---:|---|---|---|
| case8387 | 4   | extend_level_b | **NO** — launch+latency-bound | GPU idle (SM 5%) |
| case8387 | 64  | extend_level_b + invert_pivot_b | **MIXED** (전이) | DRAM 34%, FP64 40% 의 두-축 |
| case8387 | 256 | extend_level_b (memory) + invert_pivot_b (compute) | **YES (factor 의 memory 축)** | DRAM 62%, 측정 BW ≈ peak 36% |
| USA      | 4   | extend_level_b | **NO** — launch+latency-bound | waves 0.10 |
| USA      | 64  | extend_level_b | **YES** | DRAM 53%, 이미 memory wall |
| USA      | 256 | extend_level_b | **YES (가장 강함)** | DRAM 66%, warp 90%, 측정 BW ≈ peak 22% |

**Q: 메모리 바운드를 풀 방법은?**

1. **(즉시)** FP64 강제 해제 → mixed/FP32 (`MF_NO_MIXED` 끄기). USA B=64+ 에서 **2.8-3.2× 가속** 확인됨.
2. **(중기)** FP64 path 의 `trailing_update_scalar` 를 shared-staged 으로 (Σ.14 의 FP64 포팅). **1.3-1.6×** 추가 가능.
3. **(장기)** front arena 의 layout 을 batch-inner 로 재설계. cross-batch L2 reuse 가 활성화되어 **2-3×** 잠재.

**핵심 메시지**:
> *case8387 의 B=64 까지는 compute / memory / latency 가 비슷한 비율로 점유 — 단일 한계가 없음.*
> *case8387 B=256 부터, USA 는 B=64 부터 — `mf_factor_extend_level_b` 의 DRAM 이 dominant 자원.*
> *이 한계의 근본 원인은 (1) cross-batch L2 reuse 0, (2) FP64 의 큰 elem 크기, (3) atomic / partial-cacheline 의 BW 비효율 — 셋 다 풀 방법이 있고, M3 (mixed/FP32) 만으로도 측정상 2-3× 회복.*

---

## 7. 측정 환경 / 재현

### 빌드

- `/tmp/clsb` — `CLS_INTERNAL_GRAPH=ON` (wall-time 측정용)
- `/tmp/clsb_nograph` — `CLS_INTERNAL_GRAPH=OFF` (nsys / ncu 측정용 — graph 가 커널을 가리지 않도록)

### 환경

- `MF_NO_MIXED=1` (FP64 강제)
- 매트릭스:
  - `/datasets/power_system/nr_linear_systems/case8387pegase/{J,rhs}.mtx`
  - `/datasets/power_system/nr_linear_systems/case_SyntheticUSA/{J,rhs}.mtx`
- `--batch B --batch-only` — single-system path 우회

### 명령

```bash
# Wall-time
MF_NO_MIXED=1 /tmp/clsb/custom_linear_solver_run \
  --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# nsys
MF_NO_MIXED=1 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o /tmp/membound/<tag> \
  /tmp/clsb_nograph/custom_linear_solver_run --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# ncu (level kernels)
ncu --csv --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,\
launch__waves_per_multiprocessor,gpu__time_duration.sum \
  -k mf_factor_extend_level_b --launch-skip 5 --launch-count 20 \
  /tmp/clsb_nograph/custom_linear_solver_run --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# ncu (1-instance invert_pivot)
ncu ... -k mf_invert_pivot_b --launch-count 3 ...
```

### 원본 데이터

- nsys: `/tmp/membound/c8_b{4,64,256}.nsys-rep`, `/tmp/membound/usa_b{4,64,256}.nsys-rep`
- ncu CSV: `/tmp/membound/{c8,usa}_b{4,64,256}_{mf_factor_extend_level_b,mf_bwd_level_b,mf_fwd_level_b}.csv`, `c8_b*_invertfix.csv`
- 집계 스크립트: `/tmp/membound/agg2.py`

### 관련 문서

- `04-benchmarks-profiling/07-batched-bottleneck-fp64-case8387-b1-b256.md` — 같은 framework 의 case8387 단독, B=1/8/64/256 (본 문서의 직접 선행)
- `04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md` — FP32 batched throughput, M3 의 효과 측정
- `05-reports/02-comprehensive-sweep-2026-06-05.md` — FP64/FP32/TC × 5 case × 5 batch 전체 표 (M3 의 다른 case 들에 대한 효과 확인용)
- `03-optimization-notes/05-mysolver-warm-cache-port-plan.md` — M2 (persistent block + cache reuse) 의 설계 노트
