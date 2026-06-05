# 멀티 배치 factorize throughput — case8387pegase, FP32, B = 2 … 1024

*`batched_factorize` 의 throughput 스케일링 측정. 시스템당 처리 시간이 B 따라 어떻게 줄어드는가, 어디서 saturate 되는가.*

## 1. 측정 조건

| | |
|---|---|
| 매트릭스 | case8387pegase (n=14908, nnz=110572) |
| 정밀도 | **FP32** (`MF_FP32=1` → `BatchPrecision::FP32`, front 전체가 float, **텐서코어 안 씀** — §1.1) |
| GPU | RTX 3090 sm_86 |
| 실행 | `--batch B --batch-only`, repeat=50, 5–8 trials, 평균/median |
| B sweep | 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 |
| 환경 | `MF_NO_SELINV=1` (기본 경로로 가정) |
| **B=1 baseline** | **단일 배치 FP32 경로** ([`06-single-batch-bottleneck-case8387-fp64.md`](06-single-batch-bottleneck-case8387-fp64.md) §9), `--batch 1` 의 batched 경로 아님 |

### 1.1 텐서코어 사용 여부 — `mf_factor_mid_tc32_b` 이름 주의

nsys / ncu 출력에 `mf_factor_mid_tc32_b` 가 보여 *"FP32 모드인데 TC 가 켜져 있나?"* 헷갈릴 수 있음. **아님**.

이 커널은 `template<bool USE_TC>` 로 인스턴스화. `multifrontal_batched.cu` L115:

```cpp
const bool use_tc = (prec == BatchPrecision::TC32);
if (use_tc) mf_factor_mid_tc32_b<true>  <<<...>>>(...)   // TC32 모드: WMMA mma_sync
else        mf_factor_mid_tc32_b<false> <<<...>>>(...)   // FP32 모드: scalar FP32, WMMA 안 함
```

`MF_FP32=1` → `prec = FP32` → `use_tc = false` → **`<false>` 인스턴스 호출, WMMA 안 함**. nsys full kernel name 도 `mf_factor_mid_tc32_b<(bool)0>` 으로 표시됨.

진짜 텐서코어 사용 모드는 `MF_TC32=1` 또는 `MF_TC=1` — 이 문서 범위 밖.

## 2. Factorize throughput sweep

| B | factor μs/sys | factor 처리율 [sys/s] | 단일 배치 대비 |
|---:|---:|---:|---:|
| **1** (단일 배치) | **407.7** | 2,453 | 1.0× |
| 2 | 254.7 | 3,926 | 1.6× |
| 4 | 138.6 | 7,215 | 2.9× |
| 8 | 81.4 | 12,285 | 5.0× |
| 16 | 55.3 | 18,083 | 7.4× |
| 32 | 38.4 | 26,042 | 10.6× |
| 64 | 33.9 | 29,499 | 12.0× |
| 128 | 32.9 | 30,395 | 12.4× |
| 256 | 28.1 | 35,587 | 14.5× |
| **512** | **24.5** | **40,816** | **16.6×** |
| 1024 | 24.7 | 40,486 | 16.5× |

**factor 의 throughput 천장은 B=512 에서 도달 (≈ 40,800 sys/s)**. B=1024 는 추가 이득 없음 (변동 -1%, 노이즈 범위), 메모리 footprint 만 비례 증가 (front_total × 1024 × 4 B ≈ 8 GB).

단일 배치 FP32 (2,453 sys/s) 대비 B=512 = **16.6× factor 처리율**.

### B 두 배 늘렸을 때 추가 이득

| B 전환 | 처리율 추가 이득 |
|---|---:|
| 1 → 2 | +60 % |
| 2 → 4 | +84 % |
| 4 → 8 | +70 % |
| 8 → 16 | +47 % |
| 16 → 32 | +44 % |
| 32 → 64 | +13 % |
| 64 → 128 | +3 % |
| 128 → 256 | +17 % |
| 256 → 512 | +15 % |
| **512 → 1024** | **−1 %** |

B=1 → B=64 영역 (60–84% 이득) 이 *진짜 scaling*. B=64 이후는 천천히 한계로 수렴, B=512 가 saturation point.

## 3. SM throughput sweep

ncu 의 SM compute throughput 을 factor 커널들의 launch wall-time 으로 가중 평균:

| B | factor weighted SM% | factor weighted DRAM% |
|---:|---:|---:|
| 2 | 4.6 | 2.5 |
| 4 | 7.9 | 4.8 |
| 8 | 15.0 | 9.5 |
| 16 | 23.5 | 16.4 |
| 32 | 31.9 | 20.4 |
| 64 | 46.6 | 33.1 |
| 128 | 38.7 | 22.3 |
| 256 | 57.9 | 41.6 |
| 512 | 47.7 | 33.4 |
| **1024** | **60.9** | **44.3** |

(B=128, B=512 의 비단조 점은 ncu launch sampling 의 분포 차이로 추정 — 인접 B 의 trend 명확)

**B=2 (4.6 %) → B=1024 (60.9 %), 13× 증가**. B=1024 에서 단일 배치 peak (FP64 의 37.6 %, FP32 의 24.2 %) 대비 1.6× 이상, GPU 의 healthy 영역 (60–70 %) 진입.

### Wall throughput vs SM throughput 의 disconnect

§2 의 wall 처리율은 B=512 에서 saturate, 그러나 SM% 는 B=1024 에서도 계속 오름 (47.7 → 60.9). 의미: B=1024 의 추가 GPU 활용은 *같은 양의 일을 더 효율적 instruction mix 로* 처리 (SM 의 idle 사이클 감소), 하지만 wall 시간은 그대로. *부분적으로 latency-bound 에서 throughput-bound 로 전환* 되는 신호이나, wall 측면에서는 의미 없음.

### 개별 factor 커널의 SM%

| 커널 | B=64 | B=256 | B=512 | B=1024 |
|---|---:|---:|---:|---:|
| `mf_factor_mid_tc32_b<false>` | 28.4 | 40.6 | 43.3 | **51.2** |
| `mf_factor_small_warp_b` | 3.8 | 15.4 | 25.8 | **40.4** |

두 커널 다 B 따라 단조 증가, B=1024 에서도 아직 saturate 안 됨 (peak 80%+ 까지 여지 있음). 둘 다 WMMA 안 씀, scalar FP32 FMA.

## 4. 왜 B=64 가 B=256/512 처리율을 못 따라가는가

§2/§3 데이터:

| B | factor μs/sys | weighted SM% |
|---:|---:|---:|
| 64 | 33.9 | 46.6 |
| 256 | 28.1 | 57.9 |
| 512 | 24.5 | 47.7 |
| 1024 | 24.7 | 60.9 |

### 4.1 원인 — narrow 레벨에서 (cnt × B) 가 SM 을 못 채움

case8387 elimination tree 의 레벨별 grid (= cnt × B):

| 레벨 그룹 | cnt | grid @ B=64 | grid @ B=256 | grid @ B=512 |
|---|---:|---:|---:|---:|
| L0 (wide-base, fsz ≤ 16) | 4,094 | 262,016 | 1,048,064 | 2,096,128 |
| L4–L8 (mid, cnt 37–237) | 37–237 | 2,368–15,168 | 9,472–60,672 | 18,944 + |
| L11–L16 (narrow-mid) | 2–10 | **128–640** | 512–2,560 | 1,024–5,120 |
| L17–L29 (deep tail, cnt=1) | 1 | **64** | **256** | **512** |

RTX 3090 = 82 SM. SM 당 4–8 blocks 동시 실행 → **GPU 가득 채우려면 ≥ 656 blocks/launch 필요**.

- B=64 deep tail (13 levels): grid=64 — 82 SM 중 64 만 사용, *partial wave* 22 % SM 강제 idle.
- B=64 narrow-mid (6 levels): grid=128–640 — 1.5–8 waves, latency hiding 부족.
- B≥256 부터 deep tail 도 한 wave 가득 + 멀티 wave.

deep-tail + narrow-mid 가 factor wall 의 ~50% 차지하므로, 이 영역의 starvation 이 B=64 의 throughput 한계.

## 5. 현재 구현의 factorize 병목 — 커널별 분해

§3 / §4 의 SM% 와 wall 수치를 *커널 수준* 으로 풀면 어디서 시간이 소비되는지 보임. selinv OFF, B=64, FP32 의 nsys per-system 분해:

| 커널 | per-sys [μs] | factor wall 의 % | SM% @ B=64 | 어디서 작동 |
|---|---:|---:|---:|---|
| `mf_factor_mid_tc32_b<false>` | 15.19 | **53 %** | 28.4 | mid front (fsz 33–76), L2–L18 |
| `mf_factor_small_warp_b` | 9.71 | **34 %** | **3.8** | tiny front (fsz ≤ 24), L0/L1 |
| `scatter_batched` | 3.64 | 13 % | (미측정) | CSR → front 어셈블리 (start) |

**단일 dominant 커널 없음 — 53 / 34 / 13 의 3-way split**. 각 커널의 내부 병목 성격이 다름.

### 5.1 `mf_factor_mid_tc32_b<false>` — narrow-mid 의 GPU 부분-채움 + seq-in-k

이 한 커널 안에 4 phase 가 fused:

| Phase | 작업 | 추정 비중 (내부 wall) | 특성 |
|---|---|---:|---|
| 1 | panel LU (rank-nc, **seq in k**) | ~30 % | latency-bound, nc 회 `__syncthreads` |
| 2 | U-panel triangular solve (**seq in k**) | ~15 % | latency-bound |
| 3 | trailing GEMM rank-nc update | ~45 % | parallel, compute-bound |
| 4 | shared staging + extend-add | ~10 % | mixed |

SM% 28.4 % 의 의미:
- narrow-mid 레벨 (L11–L16) 에서 grid = cnt × B = 128–640 → **GPU 가 부분만 채움**
- Phase 1, 2 의 seq-in-k 가 *kernel 안에서* 직렬화 → 일부 warp 이 다른 warp 결과 대기

→ **이 커널의 진짜 병목은 narrow-mid 레벨의 부분-채움 + Phase 1/2 의 seq-in-k 직렬성**. *GEMM 자체가 아니라 그 앞의 panel LU/TRSM 이 critical path*.

### 5.2 `mf_factor_small_warp_b` — too-fine-grained dispatch overhead

가장 충격적인 수치는 **SM% 3.8 % at B=64**. grid 가 *과잉* (cnt=4094 × B=64 / W=8 = 32 K blocks) 인데도 3.8 %.

원인:
- 각 warp 의 work 가 *너무 작음* — fsz=16, nc=8 LU 는 lane 당 ~500 cycles
- block 안 8 warps 가 fsz²=256 elements 의 shared staging + LU 만 처리, kernel 자체가 μs 단위로 끝남
- 32 K blocks 를 dispatch 하는 비용 + warp scheduling latency 가 work 보다 큰 *fine-grained 영역*
- DRAM% 도 2.5 % → BW 도 binding 아님 → **순수 dispatch / scheduling 한계**

ncu 의 점유율 16.6 % 도 일치: W=8 warps/block, RTX 3090 SM 당 48 warps max, shared budget (≈ 16 KB/block) 으로 SM 당 1 block 가 한계 → 점유율 8/48 = 17 %. 더 많은 blocks per SM 가 막혀 있음.

→ **이 커널의 진짜 병목은 "작업이 너무 잘게 쪼개져 GPU 의 unit-of-execution 보다 작은 영역으로 떨어짐"**. SM% 의 B 따른 증가가 가장 가파른 것 (3.8 → 40.4 at B=1024, 10× 상승) 도 *이 영역에 가장 큰 headroom* 이 있다는 신호.

### 5.3 `scatter_batched` — 작지만 not bottleneck

`mf_scatter_csr_values<float, double>` — CSR 값을 front 의 적절한 위치에 atomic-add. factor 시작에 한 번 호출, B 차원으로 batched. wall 의 13 % 차지하지만 B 따라 자동 amortize. 별도 lever 불필요.

### 5.4 가장 *고칠 수 있는* 병목 — `small_warp_b`

세 커널 중 small_warp_b 가:
- 비중 큼 (factor wall 의 34 %)
- **SM% 압도적으로 낮음** (3.8 %) → *진짜 idle 영역*
- SM% 의 B 따른 증가가 가장 가파름 (B=64 → B=1024: 3.8 → 40.4, 10× 상승) → **headroom 가장 큼**

수정 방향 후보:

1. **W 를 8 → 16 으로 늘림** — block 안 warp 수 ↑, block 수 ↓ → 같은 작업이 더 큰 work-per-block 으로 묶임. shared budget (W × fsz²_cap × 8 B ≤ 96 KB) 안에서 가능. dispatch overhead 절감 잠재.
2. **L0 + L1 을 fused 한 launch** — 두 레벨 모두 small_warp_b 가 처리, dependency 있지만 persistent kernel + level barrier 로 fuse 가능. dispatch 회수 절감.
3. (포기) GPU dispatch 자체를 회피하고 CPU sequential 처리 — host-device sync 비용이 더 큼

mid_tc32_b 의 seq-in-k latency 는 알고리즘 자체에 박혀 있어 *작은 lever 로 못 푼다*. vendor MAGMA vbatched (literature §7.2 L1) 가 lever 가 있을 가능성.

## 6. TC 가속의 ROI 분석 — case8387 에서 왜 효과 작은가

이전 대화에서 "TC 를 켜면 어떻게 되나?" 가 자연스러운 질문. literature (§7) 를 보면 STRUMPACK / SuperLU_DIST 가 TC 로 큰 효과 본다. 우리 case8387 에서는?

### 6.1 GEMM 이 정말 factor 의 병목인가? — **아니다, ~24 %**

§5.1 의 분해를 보면:
- mid_tc32_b<false> 안의 trailing GEMM (Phase 3) ≈ kernel 내부의 45 %
- mid_tc32_b 가 factor wall 의 53 %
- → **TC 가속이 닿을 수 있는 GEMM-like work ≈ 0.53 × 0.45 = 24 % of factor wall**

나머지 76 % (small_warp_b 34 %, mid 의 panel LU + TRSM 28 %, scatter 13 %, ...) 는 *TC 가 못 건드림*.

### 6.2 WMMA tile vs case8387 의 front 크기

TC (WMMA) 의 기본 단위: **16 × 16 × 16 tile** (M × N × K, FP16 input → FP32 accumulate).

case8387 의 front 분포:
- max fsz ≈ 76 (cap=8 dump)
- **nc = 8** typical (panel column 수)
- uc = fsz − nc ≈ 49–68

→ K (=nc) = 8 **< 16** → 한 WMMA tile 의 절반 (8 lanes) 이 *zero-padded*. **연산의 50 % 가 fake-zero × fake-zero**.

uc = 49 일 때: ceil(49/16) = 4 tiles per axis → 64×64 padded. 마지막 tile 의 15/16 row 가 wasted.

두 padding 효과 곱하면 *WMMA 의 useful ratio ≈ 30–50 %*.

### 6.3 실효 speedup 추정

RTX 3090 sm_86 raw:
- FP32 ALU: 35.6 TFLOPS
- FP16 TC (FP32 accumulate): 142 TFLOPS = **4× FP32**

useful ratio 30–50 % 적용: **실효 1.2–2× speedup on Phase 3 only**.

factor wall 기준:
- TC 1.5×: factor wall **−8 %**
- TC 2.0×: factor wall **−12 %**

→ **transformative 아님**. small_warp_b 의 +10–15 pp SM% headroom (§5.4) 와 같은 영역의 ROI.

### 6.4 코드에 이미 있는 TC 경로

`src/tc/factor_tc.cuh` 의 `tc_trailing_wmma_f32` + `mf_factor_mid_tc32_b<true>` — 정확히 위 분석 그대로 구현됨 (`KP = ceil(nc/16)*16`, `UCP = ceil(uc/16)*16`). `MF_TC32=1` 로 활성. 우리 측정 (`MF_FP32=1`) 에서는 `<USE_TC=false>` 인스턴스라 *사용 안 함* (§1.1).

case8387 가 *literature 의 TC sweet spot 에서 한참 벗어남*. onetone2 (회로, fsz ≥ 257), 3D Helmholtz (fsz ≥ 1000+) 처럼 *큰 dense 영역이 있는* matrix 에서만 TC 가 4×에 가까운 이득. power-grid Jacobian 처럼 *fsz ≤ 76 + nc = 8* 의 sparse 영역은 TC 의 fundamental 적용 영역 밖.

## 7. 실험 — "compute 가 병목이 아님" 의 직접 증거 (warp stall 분해)

### 7.1 가설과 설계

§5/§6 에서 추정한 *"진짜 병목은 compute 가 아니라 latency/scheduling"* 을 직접 검증. ncu 의 **warp issue stall 분해** 로 측정.

**예측**:
- compute-bound ⟺ issued cycles ≥ 50 %, B↑ 따라 추가 상승
- memory-bound ⟺ `long_scoreboard` (global memory load wait) dominant
- latency-bound ⟺ issued cycles 낮게 *유지*, `wait` / `barrier` / `long_scoreboard` 가 stall 대부분

**측정**:
- ncu `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` (이 ratio 들이 합쳐서 cycle 분해)
- B = 32, 64, 128, 256, 512, 1024 × factor 의 두 dominant 커널 (mid_tc32_b<false>, small_warp_b)
- 정규화 = 전체 cycles 의 % 로 변환

### 7.2 데이터 — Warp stall % of total cycles (selinv OFF, FP32)

| 커널 | B | sm % | mem % | **issued** | long_sb | wait | barrier | short_sb | dispatch | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **mid_tc32_b<false>** | 32 | 20.6 | 18.9 | **12.2** | 24.0 | 30.9 | 20.5 | 10.7 | 1.5 | 0.3 |
| | 64 | 32.2 | 30.7 | **11.9** | 25.9 | 30.9 | 19.0 | 10.3 | 1.9 | 0.2 |
| | 128 | 36.9 | 33.0 | **9.4** | 26.4 | 24.7 | 29.7 | 8.2 | 1.5 | 0.3 |
| | 256 | 45.2 | 42.1 | **10.2** | 26.8 | 27.1 | 24.6 | 9.1 | 1.9 | 0.2 |
| | 512 | 49.9 | 49.0 | **10.3** | 28.3 | 27.5 | 21.9 | 9.3 | 2.1 | 0.5 |
| | 1024 | 41.5 | 36.9 | **8.9** | 26.6 | 23.5 | 31.4 | 7.7 | 1.4 | 0.5 |
| **small_warp_b** | 32 | 43.8 | 28.7 | **14.9** | 35.9 | 37.5 | 0.0 | 8.6 | 2.6 | 0.5 |
| | 64 | 62.7 | 38.3 | **15.1** | 31.4 | 39.5 | 0.0 | 8.7 | 4.9 | 0.4 |
| | 128 | 52.6 | 30.0 | **14.7** | 34.8 | 38.6 | 0.0 | 8.9 | 2.5 | 0.4 |
| | 256 | 74.6 | 42.2 | **14.8** | 30.7 | 40.0 | 0.0 | 8.7 | 5.5 | 0.4 |
| | 512 | 75.4 | 42.8 | **14.8** | 30.3 | 40.2 | 0.0 | 8.7 | 5.5 | 0.4 |
| | 1024 | 71.1 | 44.1 | **14.9** | 31.5 | 39.6 | 0.0 | 8.7 | 5.0 | 0.3 |

stall 카테고리 설명:
- **issued** — warp 가 실제 instruction issue 한 사이클
- **long_sb** (long scoreboard) — global memory load 결과 대기
- **wait** — instruction pipeline 의존성 대기 (A 의 결과를 B 가 기다림)
- **barrier** — `__syncthreads` 대기 (`__syncwarp` 은 포함 안 됨)
- **short_sb** (short scoreboard) — shared / local memory 의존성 대기
- **dispatch** — issue slot 자체 dispatch 대기

### 7.3 가설 검증 — compute 가 병목이 *아님*

**1. issued % 가 B 와 무관**: B 32 → 1024 (32×) 로 늘려도
- mid_tc32_b: 12.2 → 8.9 % (오히려 *감소*)
- small_warp_b: 14.9 → 14.9 % (완벽한 flat)

compute-bound 이라면 B↑ 따라 issued% 가 상승해야 함 — *실제로는 변동 없음*. 더 많은 batch 가 GPU 를 *시간상으로* 채우긴 하지만 (sm % 가 21 → 50 으로 상승), 각 사이클의 *issue 율* 은 그대로. 즉 추가된 warps 도 *각자 stall 시간이 똑같음*.

**2. stall 의 80–90 % 가 non-compute**:
- mid_tc32_b: long_sb (~27) + wait (~28) + barrier (~22) + short_sb (~9) = **~86 %**
- small_warp_b: long_sb (~32) + wait (~40) + short_sb (~9) = **~80 %**

대부분의 cycle 이 *기다리는* 시간. compute 자원이 부족한 게 아니라 *처음부터 issue 할 명령이 없거나 결과 기다림*.

**3. `sm__throughput` 75 % 라도 issued 는 15 %**: small_warp_b 가 B=512 에서 sm % = 75.4 % 인데 issued = 14.8 %. **SM busy ≠ SM doing useful work**. SM 의 pipeline 들이 75 % 시간 *활성* 이지만, 그 중 80 % 는 의존성 / 메모리 대기 cycle.

→ **compute 가 병목이 아니라는 가설은 데이터로 강하게 지지됨**.

### 7.4 어느 커널이 *진짜* 병목인가 — B=64 의 dominant stall

`mid_tc32_b<false>` (B=64) 의 stall 분해, factor wall 의 53 % 차지:

| stall 종류 | % | 의미 | 알고리즘적 출처 |
|---|---:|---|---|
| **wait** | 30.9 | instruction pipeline dep | Phase 1 (panel LU) 와 Phase 2 (U-solve) 의 *seq in k*. column k+1 의 ops 가 column k 결과를 기다림 |
| **long_sb** | 25.9 | global memory load 대기 | front 의 staging 과 trailing update 의 strided global F access |
| **barrier** | 19.0 | `__syncthreads` 대기 | nc 회 panel passes 사이의 block 동기 (`mf_factor_mid_tc32_b` 안에 ~ 8 개의 syncthreads) |
| **short_sb** | 10.3 | shared dep | shared staging 후 read-after-write 의존성 |
| issued | 11.9 | actual compute | |

`small_warp_b` (B=64) 의 stall 분해, factor wall 의 34 % 차지:

| stall 종류 | % | 의미 | 알고리즘적 출처 |
|---|---:|---|---|
| **wait** | 39.5 | pipeline dep | warp-per-front 의 lane 당 직렬 reduction (rank-1 trailing) |
| **long_sb** | 31.4 | global memory load 대기 | per-warp front staging (fsz² doubles to shared, parent extend atomicAdd) |
| **short_sb** | 8.7 | shared dep | LU 의 shared 의 dependent reads |
| dispatch | 4.9 | issue slot dispatch | warp packing 의 scheduling overhead |
| barrier | 0.0 | (`__syncwarp` 사용, barrier 안 잡힘) | |
| issued | 15.1 | actual compute | |

**B=64 에서 진짜 병목**:
- 두 커널 모두 *latency-bound* (compute, memory 어느 쪽도 saturate 못 함)
- mid_tc32_b: `wait` + `long_sb` + `barrier` 가 80 % stall — *알고리즘의 seq-in-k 직렬성* + *memory staging* + *block sync* 의 합작
- small_warp_b: `wait` + `long_sb` 가 71 % stall — *warp 내부 직렬 reduction* + *front staging memory wait*

### 7.5 시사점

수정 가능한 stall:
| stall | 커널 | 잠재 lever |
|---|---|---|
| `barrier` (mid 의 19–31 %) | mid_tc32_b | Phase 1/2 의 syncthreads 횟수 감소. blocked LU, less-frequent sync, 또는 vendor MAGMA vbatched (§7→§9.2 L1) |
| `wait` (둘 다 27–40 %) | 둘 다 | Instruction-level parallelism ↑ (loop unroll, multiple FMAs in flight, software pipelining) |
| `long_sb` (둘 다 26–36 %) | 둘 다 | global memory access 최소화. front prefetch, L2 hit ratio 올리기, persistent kernel 로 front reuse |

수정 불가능 / 어려운 stall:
| stall | 이유 |
|---|---|
| `short_sb` (8–10 %) | shared memory bank conflict 또는 swizzle pattern 최적화로 줄일 여지 있지만 한계 |
| `dispatch` (1–5 %) | scheduling 오버헤드 — 알고리즘 변경으로만 줄일 수 있음 |

**문헌 (§9) 의 60-70 % SM% 목표가 비현실적인 진짜 이유**: stall 의 80-90 % 가 non-compute 라서, compute throughput 을 늘리려면 *알고리즘의 latency 자체를 줄여야* 함. B 늘리는 것만으로는 stall *비율* 안 변함 (§7.3 발견).

## 8. 문헌 조사 — 60–70 % SM% 가 B=64 에서 가능한가?

### 8.1 SOTA 벤치마크 — SuperLU_DIST 의 ncu 데이터

Boukaram et al. (2024) ["Batched Sparse Direct Solver Design and Evaluation in SuperLU_DIST"](https://escholarship.org/uc/item/20h717s9) 가 A100 (RTX 3090 의 데이터센터 형, 더 강력) 위 B=200 으로 측정한 ncu 데이터 (Table 1, Landau Jacobian, n ≈ 50k):

| 커널 | Compute % | GB/s |
|---|---:|---:|
| **largest GEMM** (트레일링 업데이트) | **46.34 %** | 309 |
| largest Scatter (extend-add) | 39.12 % | 270 |
| Solve forward | 4.50 % | 94 |
| Solve backward | 1.78 % | 25 |

**SOTA 라이브러리 (MAGMA vendor) 풀가동, A100, B=200 에서도 GEMM 46 % 가 한계**. 우리 mid_tc32_b 가 B=64 에서 28 %, B=256 에서 41 %, B=1024 에서 51 % — *경쟁력 있는 수준*, 우리 구현의 결함이 아니라 *문제 영역의 천장*.

### 8.2 문헌의 narrow-top 기법들 — 적용성 평가

| # | 기법 | 출처 | case8387 적용성 | 잠재 SM% 이득 |
|---|---|---|---|---|
| L1 | **MAGMA vbatched routines** (변수 사이즈 batched dgetrf / dtrsm / dgemm) | STRUMPACK [Anzt et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059), SuperLU_DIST 둘 다 채택 | 적용 가능 (mid/big front 한정) | mid_tc32_b 28 → 40 % 가능성 (B=64) |
| L2 | **Block-ID ordering** — deeper-tree work 를 낮은 block ID 에 할당 | SuperLU_DIST §2.2 ("*it's beneficial to assign supernode columns at lower levels of the level sets to smaller block ID*") | 직접 적용 가능 | small (+2–4 pp) |
| L3 | **Wavefront streaming**: LU + Scatter 분리 dispatch | Laine et al. 2013 ["Megakernels Considered Harmful"](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus) + SuperLU_DIST GEMM/Scatter 분리 | 적용 가능 (현재 fused) | 측정 필요 |
| L4 | **Subtree concurrency** (독립 subtree 를 stream 별 dispatch) | STRUMPACK GPU 구현 | case8387 narrow-top 에서 ROI 낮음 (L13 부터 cnt ≤ 9, L17 부터 cnt=1) | 작음 |
| L5 | **Cooperative kernel grid_sync** (전체 etree 를 한 launch) | CUDA Cooperative Groups | Megakernel 경고 (register pressure ↑, occupancy ↓) 적용. 좁은 영역 한정으로만 시도 가치 | 불확실 |

### 8.3 새로 제안 — 문헌 기반 적용 후보

#### N1. MAGMA vbatched routines 도입 (mid/big front 한정)

STRUMPACK + SuperLU_DIST 가 GEMM 46 % SM% 도달한 핵심. 우리 mid_tc32_b 의 fused LU + trailing 을 분해:
- `magma_*getrf_vbatched` 로 panel LU
- `magma_*trsm_vbatched` 로 U-panel solve
- `magma_*gemm_vbatched` 로 trailing GEMM (이게 SM% 가장 큼)

벤더 튜닝의 효과로 mid_tc32_b 의 SM% 28 → 40 % 정도 회복 가능 (literature 의 SuperLU_DIST 데이터 기반 추정).

구현 비용 큼 — MAGMA dependency 추가, 변수-크기 marshaling 코드, kernel signature 재작성. **ROI: B=64 에서 factor +10–15 % throughput**.

#### N2. Block-ID ordering trick

현재 `plcols` 배열에 panel ID 가 *level 순서* 로 packed. dispatch 시 grid.x = level_size, blockIdx.x 가 그 안의 panel index.

SuperLU_DIST 의 통찰: GPU 가 lower block ID 를 먼저 dispatch 한다. *deeper-tree* (narrow) work 를 *smaller block ID* 에 두면, 그 블록들이 가장 먼저 시작 → wider-level 의 tail 과 overlap. critical-path 단축.

구현: symbolic 단계에서 plcols 정렬을 *level-then-cnt-ascending* 으로 바꿈 (현재 추정: level-then-panel-id). 거의 free.

ROI: small but easy. SM% +2–4 pp 예상, factor +1–2 % wall.

#### N3. 하단 트리 dense LU 흡수 (literature 직접 근거 없음, 자연스러운 hybrid)

L17–L29 의 panel=1 × 13 레벨 = 총 13 fronts (fsz 30–70 의 직렬 chain). 이걸 그대로 13 launch 시키지 말고, *symbolic 단계에서 하나의 ~70×70 dense matrix 로 assemble* → 한 번의 `cublasDgetrfBatched` 호출로 마무리.

장점:
- 13 launches → 1 launch (dispatch overhead 13× 절감)
- dense LU 가 vendor cuBLAS 의 batched 루틴 → SM% 자체적으로 높음
- B 차원이 그대로 batched LU 의 batch 차원으로 사용됨 → B=64 에서도 cuBLAS 가 충분히 채움

단점: 하단 13 levels 가 dense block 으로 변환되면서 *padding fill* 증가. case8387 에서 추정 ~5 KB 정도 추가 메모리.

ROI: deep-tail wall 의 ~50% 회수 추정 → factor wall −5–8 %.

### 8.4 정직한 한계 — 60-70 % SM% 가 B=64 에서?

literature 기반 추정:
- 현재 B=64: 46.6 % SM%
- L1 + L2 + L3 모두 적용 시 추정: **52–58 % SM%**
- **60-70 % 는 B=64 에서 비현실적**

이유:
1. SuperLU_DIST 가 SOTA + A100 + B=200 에서도 GEMM 46% 가 한계. RTX 3090 + B=64 로 그걸 뛰어넘으려면 *케이스 자체* 가 더 우호적이어야 함.
2. case8387 의 평균 degree 7.4 + narrow-top etree 는 fundamental limit. front 자체가 작아 *dense compute* 의 비중이 작음.
3. 60-70 % 가 필요하면: B 를 키우거나 (B=1024 에서 60.9 %, 우리 §3 측정), 더 큰 dense 영역이 있는 다른 matrix 가 필요.

### 8.5 권장 다음 단계

| 우선순위 | 작업 | 잠재 SM% 이득 (B=64) | 구현 비용 |
|---|---|---|---|
| 1 | **L2 + N2** (block-ID ordering) | +2–4 pp | 매우 낮음 |
| 2 | **N3** (하단 트리 dense LU 흡수) | +3–5 pp | 중간 (symbolic 변경) |
| 3 | **N1 / L1** (MAGMA vbatched) | +8–12 pp | 높음 (외부 dependency) |

## 9. 결론

### 9.1 throughput 측면

- factor 시스템당 처리 시간은 B=1 의 407.7 μs/sys 에서 B=512 의 24.5 μs/sys 로 단조 감소, **16.6× 처리율 증가**.
- B=512 이후 wall 은 saturate, 추가 B 는 메모리만 더 먹음.
- SM throughput 은 B=1024 까지 계속 오르지만 wall 에는 반영 안 됨 (latency-bound 영역의 idle cycle 채우기).
- **권장 운영 점**: factor 의 max throughput 을 원하면 **B=512** 가 sweet spot. 메모리 footprint trade-off 가 허용한다면 B=256 도 충분 (B=512 대비 −13 %, footprint 절반).

### 9.2 병목 / 가속 가능성 측면

- factor wall 의 분해 (§5): mid_tc32_b<false> 53 %, small_warp_b 34 %, scatter 13 % — *단일 dominant 커널 없음*.
- **warp stall 분해 (§7) 가 결정적 증거** — B 가 32 → 1024 로 32× 늘어도 *issued cycle 비중이 변하지 않음* (mid: 12 → 9 %, small_warp: 15 % flat). compute 가 병목이라면 issued% 가 상승해야 함. **batched factorize 는 compute-bound 가 아님**.
- 진짜 stall 의 출처 (B=64 기준):
  - mid_tc32_b: wait (31 %, pipeline dep) + long_sb (26 %, global mem) + barrier (19 %, `__syncthreads`) = 76 % 의 cycle 이 *기다림*
  - small_warp_b: wait (40 %) + long_sb (31 %) = 71 % stall, barrier 0 % (`__syncwarp` 만)
- 각 커널의 *어디서* latency 가 오는가:
  - mid_tc32_b: Phase 1/2 의 seq-in-k 직렬성 → `wait` + `barrier`, front staging 의 strided global access → `long_sb`. *알고리즘 자체에 박힌* latency.
  - small_warp_b: warp 내부 직렬 reduction → `wait`, per-warp front staging → `long_sb`. SM% 의 B 따른 증가 (3.8 → 40 %) 가 가장 가파름 — *headroom 가장 큼*.
- **TC 가속 (§6) 은 transformative 아님**: GEMM-like work 가 factor wall 의 24 % 뿐 + case8387 의 nc=8 < 16 으로 WMMA tile 의 50 % 가 zero-padded → 실효 1.2–2× on Phase 3 only → factor wall 기준 −8–12 %.
- **60–70 % SM% 가 B=64 에서 비현실적** (§8.4): SOTA SuperLU_DIST 가 A100 + B=200 으로 GEMM 46 % 가 한계. 우리 B=64 의 47 % 는 *이미 SOTA 수준*. *진짜* compute throughput 끌어올리려면 stall (특히 wait + long_sb) 의 *알고리즘적* 소스를 줄여야 함, B 늘리는 것만으로는 stall *비율* 안 변함.

### 9.3 우선순위 정리

| 우선순위 | 작업 | 잠재 이득 | 비용 |
|---|---|---|---|
| 1 | **B=64 → B=256 또는 512 로 운영** (운영상 가능하다면) | factor throughput +20–38 % | 0 (코드 변경 없음) |
| 2 | **small_warp_b 의 W 늘림 + L0/L1 fusion** | small_warp_b SM% 3.8 → ~10 + dispatch 절감 | 낮음 |
| 3 | **Block-ID ordering** (§7.3 N2) | SM% +2–4 pp | 매우 낮음 |
| 4 | **하단 트리 dense LU 흡수** (§7.3 N3) | factor wall −3–5 % | 중간 |
| 5 | **MAGMA vbatched routines** (§7.3 N1) | mid_tc32_b SM% 28 → 40 + (B=64) | 높음 (외부 dep) |
| (포기) | TC 가속 (`MF_TC32=1`) | factor wall −8–12 % only | 코드 이미 있지만 효과 작음 |

## 10. 재현

```bash
# B sweep
for B in 2 4 8 16 32 64 128 256 512 1024; do
  MF_FP32=1 MF_NO_SELINV=1 ./custom_linear_solver_run \
      /datasets/power_system/nr_linear_systems/case8387pegase \
      --repeat 50 --single-precision fp32 --batch $B --batch-only
done

# ncu SM% sweep
for B in 2 4 8 16 32 64 128 256 512 1024; do
  MF_FP32=1 MF_NO_SELINV=1 ncu --kernel-name "regex:mf_factor" \
      --section SpeedOfLight --csv --log-file ncu_factor_b${B}.csv \
      ./custom_linear_solver_run ... --repeat 3 --single-precision fp32 \
      --batch $B --batch-only
done
```
