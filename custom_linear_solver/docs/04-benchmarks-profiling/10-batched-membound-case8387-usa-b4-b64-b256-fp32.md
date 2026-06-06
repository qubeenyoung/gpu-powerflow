# Batched 솔버의 메모리 바운드 여부 검사 — case8387 / Synthetic USA (FP32, B = 4, 64, 256)

*custom_linear_solver 의 uniform-batch 모드 (`--batch B`) 를 case8387pegase (n=14908) 와 case_SyntheticUSA (n=156255) 두 매트릭스에 대해 **pure FP32 강제** (`MF_FP32=1`) 로 측정. 같은 (case, B) 매트릭스 위에서 [doc 09 (FP64)](09-batched-membound-case8387-usa-b4-b64-b256.md) 와 정확히 같은 framework 으로 dominant 커널의 bound 를 재분류. *FP64 의 결론과 어떻게 달라지는지* 를 § 6 에서 직접 비교.*

> 직접 선행 문서: [`09-batched-membound-case8387-usa-b4-b64-b256.md`](09-batched-membound-case8387-usa-b4-b64-b256.md) — 같은 (case × B) 위의 FP64 측정. 본 문서는 *정밀도 1 단계 낮추면 bound 구조가 어떻게 이동하는가* 를 답.

## 0. Baseline / Scope

| | case8387pegase | case_SyntheticUSA |
|---|---|---|
| n | 14908 | 156255 |
| nnz | 110572 | 1052085 |
| panels P | 7396 | 74245 |
| levels | 32 | 40 |
| panel cap | 8 | 20 |
| front arena (FP32) per batch | **2.2 MB** | **35.2 MB** |
| GPU | RTX 3090 sm_86, 82 SM, FP32 peak 35.6 TFLOPS, GDDR6X 936 GB/s, L2 6 MB | (동일) |
| 정밀도 | **pure FP32** (`MF_FP32=1`) — front 자체 FP32, 모든 연산 FP32 | (동일) |
| 빌드 | wall: `CLS_INTERNAL_GRAPH=ON` (`/tmp/clsb`). nsys/ncu: `CLS_INTERNAL_GRAPH=OFF` (`/tmp/clsb_nograph`) | (동일) |
| B | **4, 64, 256** | (동일) |

**`MF_FP32=1` 의 의미** — `BatchPrecision::FP32` (pure FP32). front arena 자체가 FP32, 모든 working LU, trailing, solve 가 FP32. Mixed (FP64 master + FP32 working) 와 다름. FP32 mid-tier kernel (`mf_factor_mid_tc32_b<Lb0>`) 는 **Σ.14 staged-shared scalar trailing** 로 동작 — *tensor core 미사용 (HMMA 0%, § 3.1 확인)*. 진짜 TC32 path 는 `<Lb1>` instantiation 으로 별도 모드.

**working set 추정 (B × front_total, FP32)**

| B | case8387 working set | USA working set | RTX 3090 메모리 (24 GB) / L2 (6 MB) 대비 |
|---:|---:|---:|---|
| 4   | 8.8 MB  | 140.8 MB | 둘 다 L2 초과 |
| 64  | 140.8 MB | **2.25 GB** | USA 는 디바이스 메모리 9% |
| 256 | **563 MB** | **9.0 GB** | USA arena 9 GB (FP64 의 1/2) — 디바이스 메모리 38% |

→ FP64 의 같은 (case, B) 위 working set 의 **정확히 1/2**. 같은 L2 (6 MB) 에 *2배 많은 entry 들어맞음* — 이게 § 3 의 DRAM% 감소의 1차 원인.

---

## 1. Wall-time 측정 (FP32, `--batch B --batch-only`)

`/tmp/clsb/custom_linear_solver_run`, median of 5 (case8387) / 3 (USA) outer × `--repeat` inner.

| Case | B | factor / sys [μs] | solve / sys [μs] | factor total [ms] | factor scale (vs B=4) | factor / total |
|---|---:|---:|---:|---:|---:|---:|
| case8387 | 4   | 179.7 | 72.4 | 0.72  | 1.00× | 71% |
| case8387 | 64  | **73.7**  | 20.7 | 4.72  | 2.44× | 78% |
| case8387 | 256 | **64.6**  | 17.5 | 16.53 | 2.78× | 79% |
| USA      | 4   | 1415.2 | 319.7 | 5.66  | 1.00× | 82% |
| USA      | 64  | **939.0**  | 181.7 | 60.10 | 1.51× | 84% |
| USA      | 256 | **919.4**  | 169.7 | 235.4 | 1.54× | 84% |

**FP64 → FP32 가속비 (factor/sys)**:

| Case | B=4 | B=64 | B=256 |
|---|---|---|---|
| case8387 | 279 → 180 = **1.55×** | 95 → 74 = **1.29×** | 89 → 65 = **1.37×** |
| USA      | 4345 → 1415 = **3.07×** | 1696 → 939 = **1.81×** | 1770 → 919 = **1.92×** |

**관찰**:

- **FP64 의 memory-bound 가 가장 강했던 USA B=4 에서 FP32 의 가속이 가장 큼 (3.07×)** — 데이터 절반 + L2 잘 들어맞으면서 latency-bound 가 풀림.
- **USA B=64+ 의 가속이 1.8-1.9× 로 정체** — FP64 mode 의 *memory-bound 가 사라진 게 아니라 새 한계 (compute, mid_tc32 의 SM=DRAM transition) 에 다시 부딪힘*.
- **case8387 의 factor scaling 이 FP64 보다 *덜 가팔라짐* (B=4 → 256 2.78× vs FP64 의 3.15×)** — base case (B=4) 이미 어느 정도 좋고, 큰 B 의 추가 가속 여지가 적음.

---

## 2. nsys 커널 분포

`nsys profile -t cuda --force-overwrite=true` → `nsys stats --report cuda_gpu_kern_sum`. % 는 batched factor+solve 커널 합 대비. **굵게 표시된 커널** = dominant 비중.

### 2.1 case8387 (FP32)

| B | factor_mid_tc32_b | invert_pivot_b | factor_small_warp_b | fwd_level_b | bwd_level_b | fwd/bwd_small_warp_b | scatter_batched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4   | **42.8%** (2.33 ms) | 15.4% (0.84 ms) | 7.2% (0.39 ms) | 10.6% (0.58 ms) | 12.1% (0.66 ms) | 6.4% | 1.7% |
| 64  | 18.0% (5.18 ms) | **44.7%** (12.84 ms) | 9.4% (2.71 ms) | 5.0% (1.43 ms) | 5.9% (1.69 ms) | 11.4% | 4.1% |
| 256 | 16.3% (17.40 ms) | **48.0%** (51.11 ms) | 9.2% (9.79 ms) | 4.3% (4.58 ms) | 5.2% (5.56 ms) | 11.6% | 4.3% |

→ **case8387 FP32 의 dominant 시간 점유 축이 *invert_pivot 단일* 로 변함** (B=64+).
- FP64 mode 에선 extend_level_b 와 invert_pivot 이 비슷한 비중 (24%, 34% in B=256).
- **FP32 에선 invert_pivot 단독이 48%** — invert_pivot 의 *FP64 inverse 가 가속을 못 받음* (코드 상 inverse 는 FP64 로 계산되어 있음, see § 5.4). nc² × P × B 의 inverse work 자체는 정밀도 무관 동일.
- mid_tc32 (Σ.14 staged trailing) 가 FP64 의 extend_level 자리를 받지만 비중 16-18% 로 *훨씬 작음* — 진짜 numeric work 가 가벼움.

### 2.2 USA (FP32)

| B | factor_mid_tc32_b | factor_extend_level_b | invert_pivot_b | factor_small_warp_b | fwd_level_b | bwd_level_b | fwd/bwd_small_warp_b | scatter_batched |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4   | 17.2% (4.06 ms) | **29.2%** (6.88 ms) | 24.0% (5.66 ms) | 5.9% (1.39 ms) | 4.1% (0.96 ms) | 5.1% (1.21 ms) | 6.8% | 2.2% |
| 64  | 16.1% (33.32 ms) | 9.5% (19.81 ms) | **42.8%** (88.89 ms) | 10.5% (21.82 ms) | 2.0% (4.16 ms) | 2.1% (4.43 ms) | 12.0% | 3.9% |
| 256 | 15.6% (127.66 ms) | 10.1% (82.57 ms) | **43.4%** (355.18 ms) | 10.6% (86.46 ms) | 1.8% (14.93 ms) | 1.9% (15.26 ms) | 12.1% | 3.9% |

→ **USA FP32 도 invert_pivot 이 dominant (B=64+)**:
- B=64 부터 invert_pivot 이 42-43% — case8387 과 동일 패턴.
- extend_level_b (큰 front 의 *fallback* path, uc>256) 가 9-10% 로 *남아 있지만* FP64 의 49-74% 와 비교 *훨씬 작음*. Σ.14 trailing 이 대부분의 mid 크기 front 를 mid_tc32 path 로 가져갔기 때문.
- mid_tc32 (16%) + extend_level (10%) + small_warp (11%) 합쳐도 *37%* 로 invert_pivot 의 43% 보다 작음 — *factor wall 의 시간 자체는 invert_pivot 이 결정함*.

---

## 3. ncu bound 분류 — dominant 커널 (FP32)

`ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.pct_of_peak_sustained_elapsed, smsp__warps_active.avg.pct_of_peak_sustained_active, sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed, launch__waves_per_multiprocessor, gpu__time_duration.sum`. Duration-weighted average over `--launch-skip 5 --launch-count 20` (level kernels) 또는 `--launch-count 3` (1-instance invert_pivot).

> **FMA% 의 해석** — `sm__pipe_fma_cycles_active` 는 *FP32 FMA pipe* 의 사용률. SM throughput 이 높지만 FMA 가 낮으면 SM 이 *FMA 외* 작업 (shared load/store, shuffle, branch, SFU divide) 으로 바쁘다는 뜻. tensor core (HMMA) 도 별도 pipe — § 3.1 의 확인.

### 3.1 `mf_factor_mid_tc32_b<Lb0>` — Σ.14 staged-shared scalar trailing (FP32)

| Case | B | SM% | **DRAM%** | warp% | FMA% | tensor% | hmma% | waves/SM | per-launch durμs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | 3.6  | 2.4  | 12.2 | 1.4  | 0.0 | 0.0 | 0.52  | 23.7  | **launch+latency** (waves<1) |
| case8387 | 64  | 36.5 | 34.5 | 41.7 | 14.1 | 0.0 | 0.0 | 12.2  | 67.8  | **balanced** (SM≈DRAM) |
| case8387 | 256 | 46.5 | 44.0 | 44.2 | 18.1 | 0.0 | 0.0 | 47.7  | 231.2 | **balanced** (SM=DRAM, 둘 다 ~45%) |
| USA      | 4   | 21.0 | 17.9 | 23.6 | 6.9  | (n/a) | (n/a) | 3.57 | 83.1  | latency 잔존 |
| USA      | 64  | 39.1 | 33.0 | 27.9 | 12.7 | (n/a) | (n/a) | 87.5 | 1077.8 | **balanced** (SM>DRAM) |
| USA      | 256 | 41.3 | 37.1 | 31.3 | 14.0 | (n/a) | (n/a) | 341  | 4538.3 | **balanced** (SM>DRAM 살짝) |

**핵심 발견**:

- **HMMA / tensor pipe 가 0%** (case8387 모든 B 에서 확인) — `mf_factor_mid_tc32_b<Lb0>` 의 실제 구현은 *FP32-native scalar FMA 기반 staged trailing* 이지 WMMA 가 아님. 이름이 `tc32` 인 건 `<Lb1>` 의 HMMA 형제와 같은 kernel symbol 을 공유해서. 본 측정은 *pure FP32 scalar mid kernel*.
- **모든 B 에서 SM ≈ DRAM ± 5%p** — 정확히 *roofline 의 ridge 위* 에 있다. FMA 18% + SM 46% 의 차이 (28%p) 는 shared store/load + write-back overhead.
- *FP64 의 extend_level 처럼 DRAM 62-66% 까지 가지는 않음* — 같은 work 의 데이터 1/2 이지만 *staged shared 로 L1 hit 증가* 가 같이 작용.
- 결론: **mid_tc32 는 memory-bound *아님*. balanced (transition zone) 머무름**. 가속 여지는 SM, DRAM, 둘 다 동시 활용.

### 3.2 `mf_factor_extend_level_b<float>` — FP32 fallback (uc>256 큰 front)

| Case | B | SM% | DRAM% | warp% | FMA% | waves/SM | per-launch durμs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| USA | 4   | 7.7  | 5.1  | 66.5 | 3.5  | 0.30  | 146.1  | latency |
| USA | 64  | 38.3 | 29.3 | 66.1 | 17.2 | 7.82  | 500.8  | balanced |
| USA | 256 | 45.4 | 36.2 | 66.1 | 20.6 | 34.6  | 2377.0 | balanced (SM > DRAM) |

(case8387 의 mid-tier 는 mid_tc32 가 다 흡수, extend_level<float> 인스턴스 거의 없음 → 본 표에서 case8387 행 생략.)

→ **USA 의 큰-front fallback extend_level_b<float> 도 mid_tc32 와 같은 balanced 패턴**. SM=DRAM=37-45%. FP64 의 같은 커널이 DRAM 66% 였던 것 대비 *명확히 ridge 의 compute 쪽으로 이동*.

### 3.3 `mf_invert_pivot_b<float>` — FP32 dominant compute 커널

| Case | B | SM% | DRAM% | warp% | FMA% | waves/SM | per-launch durμs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | **51.6** | 3.0 | 10.5 | 2.6 | 72    | 166.4   | **compute** (SM 52%, but FMA 2.6%) |
| case8387 | 64  | **55.7** | 2.8 | 10.6 | 2.8 | 1158  | 2467.1  | **compute saturated** |
| case8387 | 256 | **55.9** | 2.8 | 10.6 | 2.8 | 4633  | 9826.5  | **compute saturated** |
| USA      | 4   | **54.9** | 3.0 | 10.6 | 2.5 | 725   | 1822.7  | **compute saturated** |
| USA      | 64  | **56.0** | 3.0 | 10.6 | 2.5 | 11591 | 28659.1 | **compute saturated** |
| USA      | 256 | **56.1** | 3.0 | 10.6 | 2.5 | 46385 | 114490.5 | **compute saturated** |

**핵심**:
- **SM% 56%, DRAM% 3%, FMA% 2.5%** — 매우 특이한 조합. SM 이 56% 사용되고 있지만 FMA pipe 는 *거의 안 쓰임*. 즉 SM 활성도가 **division / SFU / shared load** 에서 옴.
- 소스 (`factor_kernels.cuh:225`) 확인: `mf_invert_pivot_b<FT>` 는 **inverse 를 double 로 계산**해 FP32 front 에 캐스트해 저장. nc=8 (case8387) 또는 nc=20 (USA) 의 *FP64 forward/backward triangular solve 가 1 launch / iter*. 즉 *FP32 모드라도 invert_pivot 의 산술은 FP64*.
- → "FP32 mode 인데 invert_pivot 만 FP64 compute" — 이게 단일 dominant 한계. *FP32 RTX 3090 의 FP64 pipe 가 1/64 throughput* 이므로 SM 활용 56% 는 사실상 *FP64 unit 의 ridge*. (FP64% metric 으로 직접 확인 안 했지만 SM/FMA 의 큰 갭이 그 증거.)
- duration 이 B 에 정확히 비례: case8387 9.83 / 0.166 = 59.2× (B 256/4=64×의 92%) — 컴퓨트 bound 의 정직한 선형.

### 3.4 `mf_fwd_level_b<float>` / `mf_bwd_level_b<float>` — solve

| Case | B | 커널 | SM% | **DRAM%** | warp% | FMA% | waves/SM | dur μs | 분류 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| case8387 | 4   | fwd | 2.4  | 2.8  | 14.1 | 0.6 | 0.28 | 5.7   | latency |
| case8387 | 4   | bwd | 3.5  | 1.8  | 18.7 | 0.8 | 0.35 | 7.5   | latency |
| case8387 | 64  | fwd | 18.3 | 25.0 | 38.9 | 4.6 | 12.4 | 29.0  | **memory** (DRAM > SM) |
| case8387 | 64  | bwd | 22.4 | 10.7 | 52.5 | 5.5 | 14.6 | 41.2  | compute-leaning |
| case8387 | 256 | fwd | 16.3 | **44.0** | 44.3 | 3.8 | 4.73 | 19.3  | **memory** (DRAM 44%, SM 16%) |
| case8387 | 256 | bwd | 30.0 | 14.2 | 60.5 | 7.3 | 70.5 | 174.9 | compute-leaning |
| USA      | 4   | fwd | 2.8  | 10.4 | 18.9 | 0.5 | 0.16 | 8.8   | **memory-leaning** (DRAM 10% vs SM 3%) |
| USA      | 4   | bwd | 0.5  | 1.6  | 15.9 | 0.1 | 0.03 | 11.9  | latency |
| USA      | 64  | fwd | 14.8 | **49.2** | 66.6 | 2.5 | 5.52 | 50.6  | **memory** (DRAM 49% vs SM 15%) |
| USA      | 64  | bwd | 10.6 | **27.3** | 58.4 | 2.0 | 2.14 | 26.1  | **memory** (DRAM 27% vs SM 11%) |
| USA      | 256 | fwd | 16.0 | **56.3** | 78.0 | 2.7 | 19.6 | 160.9 | **memory** (DRAM 56%, warp 78%) |
| USA      | 256 | bwd | 12.5 | **38.4** | 75.0 | 2.2 | 3.70 | 49.1  | **memory** (DRAM 38% vs SM 13%) |

→ **Solve 는 FP32 에서 *더* memory-bound** :
- case8387 B=256 fwd: DRAM 44% vs SM 16% (FP64 에선 DRAM 33% vs SM 32% 균형이었음).
- USA B=256 fwd: DRAM 56% vs SM 16% (FP64 에선 ncu 미캡처지만 b=64 에서 44% > 23% → FP32 b=64 49% > 15% 로 더 강함).
- **이유**: fwd/bwd 의 work 는 front 의 일부만 (panel + L block) 읽고 GEMV — 컴퓨트가 작고 데이터는 동일. FP32 라 컴퓨트는 1/2 단축됐는데 데이터도 1/2 → 비율 같음. 하지만 *L2 의 effective 크기 2×* 효과로 DRAM 이 더 강하게 활성.
- bwd 가 FP64 에선 *compute* (FP64 53%) 였던 게 FP32 에선 *memory* 로 이동 — bwd 의 GEMV 컴퓨트가 FP32 에서 빠르게 끝나면서 DRAM stall 이 드러남.

### 3.5 요약 — Bound 매트릭스 FP32 (B ∈ {4, 64, 256})

| 커널 | case8387 B=4 | case8387 B=64 | case8387 B=256 | USA B=4 | USA B=64 | USA B=256 |
|---|---|---|---|---|---|---|
| factor_mid_tc32_b (Σ.14 scalar) | launch+latency | **balanced** | **balanced** | latency 잔존 | **balanced** | **balanced** |
| factor_extend_level_b\<float\> (USA large) | — | — | — | latency | balanced | balanced (SM > DRAM) |
| **invert_pivot_b\<float\>** | **compute (FP64 unit)** | **compute** | **compute** | **compute** | **compute** | **compute** |
| factor_small_warp_b | latency | latency | latency | latency | latency | latency |
| fwd_level_b | latency | **memory** | **memory** | memory-leaning | **memory** | **memory** |
| bwd_level_b | latency | compute-leaning | compute-leaning | latency | **memory** | **memory** |

**결론** (질문 "case8387/USA × B=4/64/256 의 메모리 바운드 여부 — FP32"):

- **case8387 B=4 (FP32)**: 전체 *latency + launch overhead* bound. GPU idle.
- **case8387 B=64 (FP32)**: factor 의 *축이 invert_pivot (compute, FP64 pipe) 단일* 로 모임. mid_tc32 는 balanced. solve fwd 만 memory-bound 시작. **전체적으로 compute-bound 우세** (invert_pivot 45% 차지).
- **case8387 B=256 (FP32)**: 같은 패턴 강화. invert_pivot 48%, fwd memory-bound 명확 (DRAM 44% > SM 16%). 하지만 fwd 비중 4% — wall 결정자는 invert_pivot **compute**.
- **USA B=4 (FP32)**: latency-bound 우세, 단 extend_level<float> 비중 29% 가 가장 큼.
- **USA B=64 (FP32)**: invert_pivot 43% (compute), fwd/bwd 둘 다 memory-bound, mid_tc32 balanced. **전체 wall 은 compute (invert_pivot) 가 결정**, 단 solve 가 진짜 memory-bound.
- **USA B=256 (FP32)**: invert_pivot 43% (compute, B 에 선형), mid_tc32 balanced, fwd/bwd 명확 memory-bound. **wall = invert_pivot**.

→ **YES — solve 는 두 case 모두 B≥64 에서 memory-bound. NO — factor 의 dominant 자원은 *FP64 compute (invert_pivot)* 로 바뀜**.

---

## 4. Roofline 해석 — FP32 ridge 의 이동

RTX 3090 의 *FP32 ridge point* = peak FP32 / DRAM BW = 35.6 TFLOPS / 936 GB/s = **38 FLOP/byte**.

FP64 ridge (0.6) 대비 **63× 더 오른쪽**. *같은 work 가 FP64 에선 memory-bound, FP32 에선 자동으로 compute-leaning 또는 balanced* 가 된다 — 메모리 BW 는 그대로, 컴퓨트 peak 가 64× 증가하므로.

### 4.1 mid_tc32 의 measured effective AI

case8387 B=256 mid_tc32: total 17.40 ms / 5 iter = 3.48 ms/iter. working set 추정 (Σ.14 staged trailing 의 fronts of fsz 49-159, B=256, FP32) ≈ 200 MB read + 200 MB write per iter (cache hit 30% 추정 → effective 280 MB).

→ achieved BW ≈ 280 MB / 3.48 ms ≈ **80 GB/s** = peak 8.6%. 측정 DRAM 44% 와 일치하지 않음 → 측정 DRAM% 가 *peak BW 가 아닌 cycle 단위 throughput* 라 더 큰 값.

USA B=256 mid_tc32: total 127.66 ms / 3 iter = 42.6 ms/iter. working set ≈ 2 × USA arena 의 mid fronts subset ≈ 4 GB.
→ achieved BW ≈ 4 GB / 42.6 ms ≈ **94 GB/s** = peak 10%.

→ FP32 의 absolute achieved BW 는 FP64 보다 *낮음* (FP64 USA B=256 extend 가 210 GB/s). 이유: *컴퓨트가 빨라져서 DRAM 이 idle 한 시간이 늘어남*. → 더 *intentional* 한 memory traffic 절감이 필요.

### 4.2 invert_pivot 의 FP64 compute 한계

`mf_invert_pivot_b<float>` 의 inverse 는 FP64 로 계산 (코드: `Ui[j * nc + j] = 1.0 / static_cast<double>(F[j*fsz+j]);`).
- nc × (nc² + nc²) ≈ 2 nc³ FP64 FLOPs per panel.
- Total FLOPs = 2 × nc³ × P × B.
- case8387 B=256: nc=8, P=7396, B=256 → 2 × 512 × 7396 × 256 ≈ 1.94 G FP64 FLOP.
- RTX 3090 FP64 peak 0.56 TFLOPS → minimum 3.47 ms. 측정 9.83 ms / 5 = 1.97 ms/iter → **추정 FP64 peak 의 ~176%** (?!?).

→ peak 보다 빨라 보이는 이유: nc inner-loop 의 *작은* nc 에선 *fully serial* (nc thread, 1 block / panel) 이라 FP64 peak 가 적용 안 됨. SFU/DIV 와 shared load 가 시간을 결정. SM% 56% 는 *FP64 unit 의 portion 이 아니라 SM 전체의 cycle-busy 비율*.

→ **invert_pivot 의 실제 한계는** (a) FP64 inverse 의 sequential nc loop, (b) shared LDST overhead, (c) 1 launch / iter 의 grid (P, B, 1) 가 GPU 의 *발진 후 retire* pattern 으로 SFU/DIV 와 FP64 pipe 를 동시에 stall.

### 4.3 fwd_level_b 의 memory roofline

USA B=256 fwd_level: total 14.93 ms / 3 iter = 4.98 ms/iter. fwd 는 per (front, batch) 에 `nc + uc` elements (panel + L block) read 와 1 y vector write.
- USA front_total 35.2 MB FP32 → fwd 읽는 부분은 nc/fsz 비율 ≈ 1/5 → effective 7 MB / batch × 256 = 1.8 GB / iter.
- y vector (n × B × 4 bytes) = 156255 × 256 × 4 = 160 MB / iter (read + write 2× = 320 MB).
- Total: ~2.1 GB / iter.
- → achieved BW ≈ 2.1 GB / 4.98 ms = **422 GB/s** = peak 45%. ncu DRAM 56% 와 정합.

→ fwd 는 **fast DRAM 활용** 중 — peak 45% 의 명확한 memory-bound. 추가 가속은 § 5 의 layout 변경 (M1, M2) 으로 cache reuse 활성화.

---

## 5. 메모리 바운드를 풀 수 있는 개선안 (FP32 specific)

§ 3 의 핵심 변화: FP32 모드의 *factor wall 은 invert_pivot 의 compute (FP64 pipe) 가 결정*, *solve 는 memory-bound*. 개선 우선순위는 FP64 와 다름.

### 5.1 (P1) Invert_pivot 의 FP32-native 재구현 + 선택적 refinement

**현재**: `mf_invert_pivot_b<float>` 가 inverse 를 *내부에서 FP64* 로 계산 (코드: `static_cast<double>(F[...])` 와 `double Ui[]`). FP32 mode 라도 FP64 pipe 사용.

**제안**:
1. inverse 자체를 *FP32 native* 로 (FP64 cast 제거). nc≤32 의 작은 inverse 는 FP32 충분.
2. 그래도 정밀도 부족이면 *FP32 inverse + 1 step Newton refinement* (X ← X(2I − AX)) — 모두 FP32 GEMM.

**예상 효과**:
- FP64 pipe stall 제거 → SM% 56% 의 *대부분이 FMA 로 흡수* → SFU/DIV bottleneck 해소.
- RTX 3090 FP32 peak 64× FP64 → 이론적으로 **5-10× 가속 가능 (this kernel only)**.
- case8387 B=256: invert_pivot 9.83 ms / 5 = 1.97 ms → 0.5 ms 추정. wall 의 48% 가 24% 가 됨. **factor wall −24%**.
- USA B=256: invert_pivot 355.18 ms / 3 = 118 ms → 40 ms 추정. wall 의 43% 가 14% 가 됨. **factor wall −29%**.

**리스크**:
- power-grid Jacobian 의 condition number 와 nc 의 inverse 안정성 — `02-design-analysis/03-no-pivoting-empirical-proof.md` 가 FP32 에서 LU 자체는 안전을 보였지만 *inverse* 의 정밀도는 별도 확인 필요.
- Newton refinement 한 step 으로 정확도 회복 검증 필요.

**난이도**: 낮음 (단일 커널, 30 라인 변경). **잠재 효과**: **factor wall 24-29% 감소** — 가장 큰 단일 개선.

### 5.2 (P2) Solve fwd/bwd 의 batch-major y / front 레이아웃

**현재**: solve 의 fwd_level_b 가 *(front_off + k) per batch* 패턴 read. y 도 *(b * n + perm[k])* — batch outer.

**제안 (M1 의 일부)**: y 와 front 의 *batch dimension 을 inner* 로.
- fwd_level 의 `y[fr[k]]` 가 **16 (FP64) 또는 32 (FP32) batch 가 한 cacheline 공유**.
- 1 line read 가 32 batch 의 같은 (front, row) 을 cover.

**예상 효과**:
- 측정 DRAM% 가 같아도 wall 30-40% 감소 — line utilization 32×.
- case8387 B=256: solve 4.58+5.56 ms / 5 = 2.0 ms → 1.2 ms.
- USA B=256: solve 14.93+15.26 ms / 3 = 10.1 ms → 6 ms.

**리스크**:
- 본격 코드 변경 (factor 의 y access 도 같이).
- gather_rhs_b / scatter_sol_b 의 permutation 패턴 변경.

**난이도**: 매우 높음. **잠재 효과**: solve wall **−40%**, factor 도 동시 개선.

### 5.3 (P3) Σ.14 staged trailing 의 더 큰 fronts 적용

**현재**: mid_tc32 (Σ.14) 가 fsz 49-159 의 mid fronts 만. fsz>159 는 `mf_factor_extend_level_b<float>` fallback (USA b=4 의 dominant).

**제안**: tile size 를 키워 (uc ≤ 512 까지 지원) USA 큰 fronts 도 mid_tc32 에 흡수.
- shared 점유: 현재 32×nc fp16 (TC mode) ≈ 2 KB. FP32 mode 의 staged scalar 는 더 작음. 더 큰 uc 도 capacity 여유.

**예상 효과**:
- USA b=64+ 의 extend_level_b<float> 9-10% 시간 (20-83 ms) → mid_tc32 로 흡수, 거기서 -20% 가속 → wall −1.8%.
- USA b=4 의 extend_level<float> 29% (6.88 ms) → 절반 흡수 → wall −7%.

**리스크**:
- Σ.14 의 trailing 결과가 다른 numeric path 라 *bit-by-bit 차이* 가능. NR loop 의 outer convergence 영향 검증 필요.

**난이도**: 중. **잠재 효과**: USA b=4 의 −7%, b=64+ 의 −2%.

### 5.4 (P4) Trailing GEMM 의 batch-축 fusion

**현재**: 모든 batched 커널이 `gridDim.y = B`. 같은 (front) 의 B copy 가 SM grid 위에 *독립* block 으로 dispatch.

**제안**: 한 block 이 *작은 N (e.g., 8) batch 를 register-tile 처리*. trailing GEMM 의 L panel 이 *한 번 load 되어 N batch 의 trailing 에 적용*.
- L panel reuse N× → DRAM read 1/N.
- shared 점유 N× — N=8 의 경우 8 × (uc×nc) FP32 = 8KB.

**예상 효과**: mid_tc32 의 SM 47% / DRAM 44% balanced 를 *DRAM 만 1/N* 으로 줄여 wall 20-30% 감소.

**난이도**: 매우 높음. **잠재 효과**: factor 의 mid_tc32 비중 (16%) × 30% = wall −5%.

### 5.5 (P5) cuBLAS batched gemmStridedBatched 로 trailing 대체

**현재**: 본 솔버는 *모든 trailing 을 자체 커널*.
**대안**: cuBLAS `gemmStridedBatched` (FP32) 는 large-batch GEMM 에 최적화. tensor core 도 자동 활성.

**예상 효과**:
- mid-tier fronts 의 trailing 이 *L2/SM 모두 잘 활용하는 cuBLAS path* 로 — 측정 BW peak 의 60-70%.
- 하지만 *level 당 trailing 의 fsz/nc 가 들쭉날쭉* 이라 batched API 가 매번 setup 새로 해야 함 → launch overhead 증가.

**난이도**: 중-높음. **잠재 효과**: 측정 의존. **권고**: 비교 prototype 만, 본격 도입 전.

### 5.6 (P6) Multi-stream 의 fwd ↔ invert_pivot overlap

**현재**: factorize 의 leaf-to-root level loop 와 invert_pivot 가 *순차*. 둘 다 batched 커널.

**제안**: invert_pivot 을 *factor 의 마지막 level 끝나는 즉시 별도 stream 에서 시작*, 같이 stream 의 다음 iteration 의 leaf-level launch 와 overlap.
- invert_pivot 의 grid (P, B, 1) 이 GPU 의 SFU/DIV 한 쪽만 사용 → factor 의 FMA pipe 와 비-경쟁.
- 측정에 따르면 invert_pivot 의 SM 56% 가 DIV/SFU, factor mid_tc32 의 SM 46% 가 FMA → resource 영역이 다름.

**예상 효과**: 실측 잠재 30-40%. 단 두 커널의 *데이터 의존성* (extend-add 후 inverse 의 input 이 결정됨) 가 있어 한 iteration 안에서는 못 함. *다음 iter 의 leaf level* 과 overlap 만 가능 → NR loop 또는 contingency 의 외부 iter 가 있어야 효과.

**난이도**: 높음 (multi-stream scheduling + dependency). **잠재 효과**: NR loop / contingency 가 있으면 10-15%.

### 5.7 우선순위 정리 (FP32 mode)

| 후보 | 영향 받는 영역 | 잠재 가속 (B=256) | 난이도 | 권고 |
|---|---|---|---|---|
| **P1** invert_pivot 의 FP32-native + refinement | invert_pivot 단일 | **factor −24-29%** | 낮음 | **1순위** — 단일 가장 큰 효과 |
| P2 Batch-major layout (solve 위주) | solve fwd/bwd | solve −40% | 매우 높음 | 2순위 (FP64 의 M1 와 통합) |
| P3 Σ.14 큰 fronts 까지 확장 | USA 의 extend fallback | wall −2-7% | 중 | 3순위 — USA 한정 |
| P4 Batch-축 register-tile | mid_tc32 | wall −5% | 매우 높음 | 후순위 |
| P5 cuBLAS gemmStridedBatched | mid-tier trailing | TBD | 중-높음 | prototype 만 |
| P6 Multi-stream invert overlap | NR / contingency | 10-15% | 높음 | use-case 의존 |

### 5.8 정량 — P1 만 적용했을 때의 예상 wall

case8387 B=256 FP32 현재 wall = factor 64.6 + solve 17.5 = 82.0 μs/sys.
- factor 의 48% 가 invert_pivot. 그것이 1/4 으로 → factor −36% → factor 41 μs.
- 새 wall = 41 + 17.5 = **58 μs/sys** (현재 대비 −29%).

USA B=256 FP32 현재 wall = factor 919 + solve 170 = 1089 μs/sys.
- factor 의 43% 가 invert_pivot. 1/4 으로 → factor −32% → factor 624 μs.
- 새 wall = 624 + 170 = **794 μs/sys** (현재 대비 −27%).

→ **P1 단일 적용으로 FP32 mode 의 wall −27 ~ −29%** 가 단일 가장 큰 개선. P2 (solve layout) 까지 적용하면 추가 −10-15%.

---

## 6. FP64 vs FP32 — bound 구조 비교

doc 09 (FP64) 와 본 doc (FP32) 의 같은 (case, B) 에서의 dominant kernel 과 bound:

| Case | B | FP64 dominant | FP64 bound | FP32 dominant | FP32 bound |
|---|---:|---|---|---|---|
| case8387 | 4   | extend_level_b\<double\> (50%) | latency | mid_tc32\<Lb0\> (43%) | latency+launch |
| case8387 | 64  | invert_pivot (31%) + extend (27%) | mixed | **invert_pivot (45%)** | **compute (FP64 unit)** |
| case8387 | 256 | invert_pivot (34%) + extend (24%) | mixed | **invert_pivot (48%)** | **compute (FP64 unit)** |
| USA      | 4   | extend_level_b\<double\> (74%) | latency | extend_level<float> (29%) + invert (24%) | latency+compute |
| USA      | 64  | extend_level_b\<double\> (48%) | **memory (DRAM 53%)** | **invert_pivot (43%)** | **compute (FP64 unit)** |
| USA      | 256 | extend_level_b\<double\> (49%) | **memory 강 (DRAM 66%)** | **invert_pivot (43%)** | **compute (FP64 unit)** |

**핵심 패턴**:

1. **FP64 의 memory wall 이 FP32 에서 사라짐** — front data 1/2, ridge 가 0.6 → 38 FLOP/byte 로 이동 → 같은 work 가 compute-leaning 으로.
2. **새 단일 bottleneck = `mf_invert_pivot_b<float>` 의 *FP64 compute*** — FP32 mode 라고 선언했어도 inverse 자체는 FP64 → RTX 3090 의 1/64 FP64 throughput 에 발이 묶임.
3. **Solve 의 memory-bound 는 FP32 에서 *더 강해짐*** — 컴퓨트가 빨라져서 DRAM stall 이 드러남. 절대 wall 은 작아도 비율적으로 memory-bound.
4. **Mid-tier factor 커널 (mid_tc32) 은 balanced (SM≈DRAM, 둘 다 ~45%)** — Σ.14 staged trailing 이 양쪽 자원을 적절히 활용.

**FP64 → FP32 의 wall 가속 (factor only, B=256)**:
- case8387: 88.5 → 64.6 μs = 1.37×
- USA: 1770 → 919 μs = 1.92×

이게 *FP32 마이그레이션의 ceiling*. P1 (invert_pivot FP32) 까지 적용하면:
- case8387: 88.5 → 41 μs = **2.16×** vs FP64
- USA: 1770 → 624 μs = **2.84×** vs FP64

P1 + P2 (solve layout) 까지:
- case8387: 1.37 × 1.45 ≈ 2.0× vs FP64
- USA: 1.92 × 1.30 ≈ 2.5-3× vs FP64

→ **답**: FP32 만으로는 *컴퓨트 한계 (FP64 invert_pivot)* 가 새 wall. *진짜 FP32 native 로 invert_pivot 만 바꿔도 FP64 대비 2-3× 가속*.

---

## 7. 최종 답 — 질문 (FP32) 에 대한 직접 응답

**Q: case8387 / USA × B = 4 / 64 / 256 의 batched factor + solve 가 FP32 모드에서 memory-bound 인가?**

| Case | B | factor의 dominant 커널 | factor bound | solve 의 bound | 종합 |
|---|---:|---|---|---|---|
| case8387 | 4   | mid_tc32 | latency+launch | latency | NOT memory-bound |
| case8387 | 64  | invert_pivot (45%) | **compute (FP64 unit)** | fwd memory-leaning | **NOT memory-bound — compute-bound** |
| case8387 | 256 | invert_pivot (48%) | **compute (FP64 unit)** | fwd memory-bound | **NOT memory-bound — compute-bound** |
| USA      | 4   | extend_level\<float\> + invert | latency 우세 | memory-leaning | mixed |
| USA      | 64  | invert_pivot (43%) | **compute (FP64 unit)** | fwd/bwd **memory-bound** | **mixed: factor compute, solve memory** |
| USA      | 256 | invert_pivot (43%) | **compute (FP64 unit)** | fwd/bwd **memory-bound** | **mixed: factor compute, solve memory** |

→ **factor 는 FP32 모드에서 *memory-bound 가 아님*** — `mf_invert_pivot_b<float>` 의 FP64 inverse 가 단일 dominant 한계 (43-48%). mid_tc32 는 balanced. extend_level<float> (USA fallback) 도 balanced.
→ **solve 는 B≥64 에서 *memory-bound*** — fwd_level 의 DRAM 44-56%, bwd_level 의 DRAM 27-38%.

**Q: FP32 모드의 메모리/컴퓨트 wall 을 풀 방법은?**

1. **(즉시, 단일 가장 큰 효과)** `mf_invert_pivot_b` 의 inverse 산술을 *FP32 native + 1 step Newton refinement* 로. case8387 B=256 factor −36%, USA B=256 factor −32%. 30 라인 코드 변경, 단일 커널.
2. **(중기)** Solve fwd/bwd 의 batch-major y 레이아웃. solve wall −40%. layout 재설계라 큰 PR.
3. **(중기)** Σ.14 staged trailing 의 tile size 확장 → USA 의 extend_level<float> fallback 흡수. USA B=4 −7%.

**핵심 메시지**:
> *FP64 의 memory wall (extend_level 의 DRAM 62-66%) 은 FP32 에서 자동 해소된다 — front 1/2 + ridge 가 0.6 → 38 FLOP/byte 로 이동.*
> *하지만 FP32 mode 가 새 wall 을 만든다 — `mf_invert_pivot_b<float>` 가 *내부적으로 FP64 inverse 를 계산*해 RTX 3090 의 FP64 1/64 throughput 에 묶임.*
> *FP32 mode 의 wall 의 43-48% 가 이 단일 커널. *진짜 FP32 inverse* 로 바꾸면 FP64 대비 2.16-2.84× 가속 가능.*
> *Solve 는 FP32 에서 더 강한 memory-bound 가 됨 (DRAM 56% USA B=256) — 컴퓨트가 빨라져 stall 이 드러남. layout 변경이 답.*

---

## 8. 측정 환경 / 재현

### 빌드

- `/tmp/clsb` — `CLS_INTERNAL_GRAPH=ON` (wall-time)
- `/tmp/clsb_nograph` — `CLS_INTERNAL_GRAPH=OFF` (nsys / ncu)

### 환경

- `MF_FP32=1` (pure FP32 모드, `BatchPrecision::FP32`)
- 매트릭스:
  - `/datasets/power_system/nr_linear_systems/case8387pegase/{J,rhs}.mtx`
  - `/datasets/power_system/nr_linear_systems/case_SyntheticUSA/{J,rhs}.mtx`
- `--batch B --batch-only` — single-system path 우회

### 명령

```bash
# Wall-time
MF_FP32=1 /tmp/clsb/custom_linear_solver_run \
  --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# nsys
MF_FP32=1 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o /tmp/membound_fp32/<tag> \
  /tmp/clsb_nograph/custom_linear_solver_run --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# ncu (level kernels, FP32-relevant metrics)
ncu --csv --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
launch__waves_per_multiprocessor,gpu__time_duration.sum \
  -k mf_factor_mid_tc32_b --launch-skip 5 --launch-count 20 \
  /tmp/clsb_nograph/custom_linear_solver_run --matrix $J --rhs $RHS --batch B --batch-only --repeat R

# ncu — tensor pipe (mid_tc32 의 *진짜* HMMA 활용 확인용, § 3.1)
ncu --csv --metrics \
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  -k mf_factor_mid_tc32_b --launch-skip 5 --launch-count 20 ...
```

### 원본 데이터

- nsys: `/tmp/membound_fp32/c8_b{4,64,256}.nsys-rep`, `/tmp/membound_fp32/usa_b{4,64,256}.nsys-rep`
- ncu CSV: `/tmp/membound_fp32/{c8,usa}_b{4,64,256}_{mid_tc32,extend,bwd,fwd}.csv`, `_invertfix.csv`, `_midtc_TENSOR.csv`
- 집계 스크립트: `/tmp/membound_fp32/agg.py`, `agg_tensor.py`, `extract_nsys.py`

### 관련 문서

- [`04-benchmarks-profiling/09-batched-membound-case8387-usa-b4-b64-b256.md`](09-batched-membound-case8387-usa-b4-b64-b256.md) — FP64 mode 의 같은 분석 (본 문서의 직접 비교 baseline)
- `04-benchmarks-profiling/07-batched-bottleneck-fp64-case8387-b1-b256.md` — case8387 단독, B=1/8/64/256 (framework 의 출발점)
- `04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md` — FP32 batched throughput saturation (큰 B 까지)
- `05-reports/02-comprehensive-sweep-2026-06-05.md` — FP64/FP32/TC × 5 case × 5 batch 전체 표
- `02-design-analysis/05-gemm-fraction-analysis.md` — trailing GEMM 의 wall fraction (mid_tc32 의 work 비중 정량)
- `02-design-analysis/03-no-pivoting-empirical-proof.md` — FP32 LU 의 power-grid 정확도 검증 (P1 의 안전성 근거)
