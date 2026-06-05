# Batched 솔버의 병목 분석 — Factor / Solve 분리 (FP64, B=1, 8, 64, 256)

*custom_linear_solver 의 uniform-batch 모드 (`--batch B`) 위 case8387pegase, **FP64 강제** (`MF_NO_MIXED=1`) 측정. Factorize 와 Solve 를 독립 phase 로 분리해 각각 (1) wall-time scaling, (2) nsys 커널 분포, (3) ncu bound (compute / memory / latency), (4) B=1 개선 후보를 정리.*

## 0. Baseline / Scope

| | |
|---|---|
| 매트릭스 | case8387pegase (n=14908, nnz=110572) |
| 정밀도 | **FP64 강제** (`MF_NO_MIXED=1`) |
| GPU | RTX 3090 sm_86, 82 SM, FP64 0.56 TFLOPS, GDDR6X 936 GB/s |
| Panel cap | 8, **31 levels, 7408 panels** |
| 빌드 | wall-time: `CLS_INTERNAL_GRAPH=ON` (`/tmp/clsb`). nsys/ncu: `CLS_INTERNAL_GRAPH=OFF` (`/tmp/clsb_nograph`) |
| B | 1, 8, 64, 256 |

**노트** — `mf_invert_pivot_b` 는 *factorize phase* 안에서 launch 됨 (`batched_factorize` 가 마지막에 호출). 본질은 *solve 가속을 위한 사전계산* (selinv 의 pivot block inverse) 이지만, wall-time 은 factor 에 적립됨. 본 보고서는 *wall-time 적립처* 기준으로 factor 에 분류.

---

## 1. 커널 역할 (요약)

### 1.1 Factorize 단계 커널

| 커널 | 역할 | 단위 |
|---|---|---|
| `mf_factor_extend_level_b<FT>` | 한 level 의 모든 (front, batch) 에 대해 panel LU (rank-nc) + trailing update + parent extend-add (atomicAdd) 까지 fused. 본 측정의 dominant factor 커널. | 1 block / (front × batch), blockDim 128/384/768 (level max fsz 별) |
| `mf_factor_small_warp_b<FT>` | 작은 leaf fronts (fsz ≤ ~26) 전용. **한 warp 당 (front, batch) 한 개**, 한 block 에 여러 warp packing. front 전체를 shared 에 staging. block-sync overhead 회피. | 1 warp / (front × batch) |
| `mf_invert_pivot_b<FT>` | 각 panel 의 nc×nc pivot block 의 L_inv / U_inv 사전 계산. selinv 모드의 solve 가 sequential trsv 를 parallel GEMV 로 바꿀 수 있게 함. **1 launch / iter**, grid (P, 1), 32 threads. | grid (P, 1), 32 threads |

### 1.2 Solve 단계 커널

| 커널 | 역할 | 단위 |
|---|---|---|
| `mf_fwd_level_b<FT, YT>` | forward solve `Ly = b` per level. selinv 켜진 경우 `sh_piv[]` 에 parallel GEMV 결과 채워 trsv 대체. bottom-up. | 1 block / (front × batch) |
| `mf_bwd_level_b<FT, YT>` | backward solve `Ux = y` per level. selinv 모드 = `U_inv @ rhs` parallel GEMV. top-down. | 1 block / (front × batch) |
| `mf_fwd_small_warp_b` / `mf_bwd_small_warp_b` | 작은 leaf-level solve 전용. 같은 warp-packing 패턴 (1 warp / (front, batch)). bottom level 이 solve 의 ~절반 차지. | 1 warp / (front × batch) |

(소스: `src/batched/{multifrontal_batched.cu, factor_kernels.cuh, factor_small.cuh, solve_kernels.cuh, solve_small.cuh}`)

---

## 2. Factorize 분석

### 2.1 Wall-time scaling

`--batch B --repeat 5 --batch-only` median, graph ON, FP64:

| B | factor / sys [ms] | factor total / call [ms] | scale (vs B=1) | kernel sum [ms]† | kernel / wall ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.910 | 0.91 | 1× | 0.71 | 78% |
| 8 | 0.168 | 1.34 | **5.4×** | 1.21 | 90% |
| 64 | 0.097 | 6.21 | **9.4×** | 5.03 | 81% |
| 256 | 0.089 | 22.78 | **10.2×** | 19.11 | 84% |

†nsys 기준 factor 커널 (extend_level_b + small_warp_b + invert_pivot_b) 의 5 iter 총합을 5 로 나눈 per-iter kernel time.

**관찰**:
- **B=1 → 8 에서 5.4× 가속** — 가장 큰 단일 점프. SM 들이 일을 받기 시작.
- **B=64 → 256 추가 가속 ~1.06×**. *포화*.
- **wall-time 의 84-90% 가 kernel time**. 나머지 ~10-20% 는 batched setup 커널 (`scatter_batched`, gather/scatter B×n 데이터 이동). B=256 의 wall-time 22.8 ms 중 19.1 ms 가 numeric factor, 3.7 ms 가 setup — *batch 가 커질수록 setup 비중 상대적으로 커짐*.

### 2.2 nsys 커널 분포 (factor only, B 별)

%는 *batched factor 커널 합* 대비:

| B | factor_extend_level_b | factor_small_warp_b | invert_pivot_b |
|---:|---:|---:|---:|
| 1 | 65% | 27% | 8% (49 μs) |
| 8 | 47% | 23% | 18% (307 μs) |
| 64 | 36% | 9% | **45%** (2.28 ms) |
| 256 | 30% | 13% | **47%** (9.04 ms) |

→ B 가 커지면서 **`mf_invert_pivot_b` 가 factor 의 절반을 차지**. 이 한 커널 (1 launch/iter) 의 duration 이 B 에 비례해 늘어남 (49 → 9044 μs, B 256× 증가에 184× 시간 증가) → § 2.3 에서 *compute-bound* 임을 확인.

### 2.3 ncu bound 분류 — factor 커널

`ncu --metrics sm__throughput, dram__throughput, smsp__warps_active, sm__pipe_fp64_cycles_active, launch__waves_per_multiprocessor`, mid-tree 30 launches duration-가중.

| 커널 | B | SM% | DRAM% | warp% | FP64% | waves/SM | dur μs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **factor_extend_level_b** | 1 | 0.9 | 0.5 | 8.3 | 0.9 | 0.01 | 41.5 | launch-overhead + 극단 latency |
| | 8 | 7.4 | 5.3 | 11.3 | 7.4 | 0.11 | 39.9 | latency |
| | 64 | 35.5 | 41.3 | 51.7 | 35.5 | 2.27 | 69.3 | 전이 (DRAM 41% > FP64 35%) |
| | 256 | 37.7 | **62.3** | 83.6 | 37.7 | 6.50 | 259.8 | **memory-bound** |
| **factor_small_warp_b** | 1 | 0.1 | 0.2 | 9.6 | 0.1 | 0.01 | 18.4 | launch-overhead |
| | 8 | 0.6 | 0.3 | 16.5 | 0.6 | 0.00 | 10.3 | launch-overhead |
| | 64 | 3.2 | 0.6 | 16.7 | 3.2 | 0.02 | 7.0 | latency (work 본질이 작음) |
| | 256 | 6.5 | 0.8 | 16.8 | 6.5 | 0.10 | 5.0 | latency 잔존 |
| **invert_pivot_b** | 1 | **29.0** | 3.2 | 10.4 | **29.0** | 18.0 | 49.0 | **compute (FP64)** |
| | 8 | **38.3** | 4.2 | 10.6 | **38.3** | 144.1 | 296.7 | **compute (FP64)** |
| | 64 | **39.7** | 4.1 | 10.6 | **39.7** | 1158.1 | 2278.7 | **compute (FP64)** |
| | 256 | **39.9** | 4.0 | 10.6 | **39.9** | 4632.4 | 9060.3 | **compute (FP64)** |

(FP64% 와 SM% 가 같은 값으로 보이는 이유: FP64 가 SM 활성시간의 dominant 점유라 둘이 일치)

**Bound 이동 정리 (factor)**:

| 커널 | B=1 | B=8 | B=64 | B=256 |
|---|---|---|---|---|
| factor_extend_level_b | launch+latency | latency | 전이 | **memory** (62%) |
| factor_small_warp_b | launch | launch | latency | latency |
| invert_pivot_b | **compute (29%)** | **compute (38%)** | **compute (40%)** | **compute (40%)** |

### 2.4 종합 — Factor 의 B=256 한계

| 자원 | dominant 커널 | 사용률 |
|---|---|---:|
| DRAM bandwidth | factor_extend_level_b | 62% |
| FP64 throughput | invert_pivot_b | 40% |
| Launch overhead | (다 amortize됨) | — |
| Latency 잔존 | factor_small_warp_b | warp 17% (본질적 작음) |

→ **factor 의 추가 가속은 *둘 다* 짜내야 함**:
- factor_extend_level_b 의 DRAM 62% 를 낮추는 것 (예: column-major arena 의 더 친화적 access pattern)
- invert_pivot_b 의 compute 부담 자체를 줄이는 것 (예: pivot 작은 panel 만 invert, 또는 FP32 inverse + FP64 refine)

이 두 한계 사이에서 추가 1.5× 정도가 알고리즘 변경 없는 한도. 그 이상은 *알고리즘 변경* 또는 *GPU 변경* (A100/H100 의 FP64 + HBM).

---

## 3. Solve 분석

### 3.1 Wall-time scaling

`--batch B --repeat 5 --batch-only` median, graph ON, FP64:

| B | solve / sys [ms] | solve total / call [ms] | scale (vs B=1) | kernel sum [ms]† | kernel / wall ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.285 | 0.285 | 1× | 0.292 | 102% |
| 8 | 0.059 | 0.472 | **4.8×** | 0.486 | 103% |
| 64 | 0.027 | 1.728 | **10.6×** | 1.994 | 115% |
| 256 | 0.023 | 5.888 | **12.4×** | 5.865 | 100% |

†nsys 기준 solve 커널 (fwd/bwd × level/small_warp) 의 5 iter 총합을 5 로 나눈 per-iter kernel time.

**관찰**:
- solve scaling 이 factor scaling 보다 **약간 더 가팔라짐** (12.4× vs 10.2×). 이유: solve 는 invert_pivot 같은 *compute-bound 커널 없음* → batch 가 더 깨끗이 amortize.
- **kernel / wall ratio ≈ 100%** — solve 의 wall-time 은 사실상 전부가 kernel time. setup 오버헤드 거의 없음 (gather_rhs_b / scatter_sol_b 합쳐도 작음).
- solve 의 **B=256 절대 wall-time 은 5.9 ms** — factor (22.8 ms) 의 1/4. NR 한 iter 의 ~80% 는 factor.

### 3.2 nsys 커널 분포 (solve only, B 별)

%는 *batched solve 커널 합* 대비:

| B | fwd_level_b | bwd_level_b | fwd_small_warp_b | bwd_small_warp_b |
|---:|---:|---:|---:|---:|
| 1 | 27% | 36% | 14% | 23% |
| 8 | 25% | 38% | 15% | 22% |
| 64 | 35% | 47% | 5% | 13% |
| 256 | 31% | 44% | 9% | 16% |

→ **bwd 가 항상 fwd 보다 큼** (about 1.4×) — selinv 모드의 backward GEMV 가 forward 보다 행렬 크기 (uc × nc 의 전치 곱) 가 약간 더 크고 FP64 compute 가 dominant 라 그럼.

→ small_warp 비중은 B=1 의 ~37% 에서 B=64 의 ~18% 로 감소 — *큰 B 에선 leaf level 도 큰 grid 가 되어 일반 level_b 커널이 효율적*.

### 3.3 ncu bound 분류 — solve 커널

| 커널 | B | SM% | DRAM% | warp% | FP64% | waves/SM | dur μs | 분류 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **fwd_level_b** | 1 | 0.3 | 0.8 | 8.3 | 0.3 | 0.01 | 6.1 | 극단 latency |
| | 8 | 2.2 | 3.7 | 10.0 | 2.2 | 0.07 | 6.1 | latency |
| | 64 | 14.4 | 25.0 | 30.2 | 14.4 | 0.98 | 8.6 | 전이 (memory-leaning) |
| | 256 | 25.2 | **41.1** | 44.5 | 25.2 | 4.95 | 16.3 | **memory** (DRAM 41% > FP64 25%) |
| **bwd_level_b** | 1 | 1.6 | 0.8 | 11.0 | 1.6 | 0.08 | 9.0 | latency |
| | 8 | 11.4 | 4.0 | 25.6 | 11.4 | 0.89 | 10.7 | latency |
| | 64 | **41.8** | 15.6 | 51.1 | **41.8** | 14.1 | 26.3 | **compute (FP64)** |
| | 256 | **53.3** | 19.5 | 59.9 | **53.3** | 64.0 | 68.2 | **compute (FP64)** |
| **fwd_small_warp_b** | 1 | 0.0 | 0.7 | 14.5 | 0.0 | 0.00 | 3.4 | 극단 launch-overhead |
| | 8 | 0.1 | 0.5 | 16.4 | 0.1 | 0.00 | 4.7 | launch-overhead |
| | 64 | 0.6 | 0.8 | 16.8 | 0.6 | 0.02 | 4.1 | latency 잔존 |
| **bwd_small_warp_b** | 1 | **23.7** | 6.3 | 47.0 | **23.7** | 1.17 | 9.5 | compute 시작 |
| | 8 | **49.4** | 12.6 | 58.6 | **49.4** | 9.90 | 36.5 | **compute (FP64)** |
| | 64 | **58.9** | 12.5 | 59.2 | **58.9** | 99.9 | 340.4 | **compute (FP64)** |
| | 256 | **59.6** | 12.2 | 59.2 | **59.6** | 401.2 | 1356.1 | **compute (FP64)** |

**Bound 이동 정리 (solve)**:

| 커널 | B=1 | B=8 | B=64 | B=256 |
|---|---|---|---|---|
| fwd_level_b | 극단 latency | latency | 전이 | **memory** (41%) |
| bwd_level_b | latency | latency | **compute (42%)** | **compute (53%)** |
| fwd_small_warp_b | launch-overhead | launch-overhead | latency 잔존 | (작은 work 본질) |
| bwd_small_warp_b | compute (24%) | **compute (49%)** | **compute (59%)** | **compute (60%)** |

### 3.4 종합 — Solve 의 B=256 한계

| 자원 | dominant 커널 | 사용률 |
|---|---|---:|
| FP64 throughput | bwd_level_b, bwd_small_warp_b | 53%, 60% |
| DRAM bandwidth | fwd_level_b | 41% |
| Launch overhead | (다 amortize됨) | — |
| Latency | fwd_small_warp_b (작은 work 본질) | warp 17% |

→ **solve 의 1차 한계는 FP64 compute** (bwd 가 dominant 인데 compute-bound). fwd 는 memory-leaning 이지만 비중 작음.

→ solve 추가 가속의 후보:
- bwd_level_b 의 GEMV → **multi-RHS 환경의 GEMM 으로 fuse** 하면 tensor core (FP16/BF16/TF32) 활용 가능. 본 솔버는 selinv 의 inverse 가 미리 계산된 후 *각 batch 의 RHS 마다 한 번씩 GEMV* — RHS batch 차원을 K 로 압축해 GEMM 으로 묶을 수 있음.
- fwd_level_b 의 memory 패턴 개선

---

## 4. B=1 의 연산 집약도 / 병렬성 개선 후보

§ 2.3, § 3.3 에서 **B=1 의 거의 모든 level kernel 이 SM 0.3–1.6%** 로 GPU 가 사실상 idle. 원인을 정량화:

### 4.1 Per-level front 수 분포 (case8387)

CLS_DUMP, 31 levels:

| Level | fronts | max fsz | blockDim 128 의 추정 SM 점유 (wave) |
|---:|---:|---:|---:|
| L0 (leaf) | **4100** | 18 | ~50× saturation |
| L1 | 1493 | 18 | 18× |
| L2 | 739 | 35 | 9× |
| L3 | 396 | 43 | 4.8× |
| L4 | 243 | 49 | 3.0× |
| L5 | 145 | 55 | 1.77× |
| L6 | 88 | 53 | 1.07× (경계) |
| L7 | 52 | 76 | 0.63 (underutilization 시작) |
| L8 | 36 | 68 | 0.44 |
| L9–L14 | 28→5 | | 0.34 → 0.06 |
| L15–L20 | 5→2 | | < 0.1 |
| **L21–L30** | **1 each** | 71→6 | **~0.012** (1 block / 82 SM) |

→ **L7 이후 underutilization 시작, L21–L30 은 사실상 1 SM 만 사용** (81 SM idle).

이 분포가 *factor* 와 *solve* 양쪽에 동일하게 영향. 단, 영향의 *결* 은 다름:
- **Factor**: level 당 work 크기는 fsz³ 에 비례 → 상위 level 의 작은 fronts 가 work 자체 적음 → wall-time 비중 작음 (단, launch overhead 누적은 동일)
- **Solve**: level 당 work fsz² 또는 nc² 에 비례 → 상위 level 비중 더 작음 → solve 가 더 *launch-overhead-dominated*

### 4.2 Factor 의 B=1 개선 후보

| 후보 | 영향 받는 부분 | 잠재 효과 (factor 0.91 ms) | 난이도 | 권고 |
|---|---|---:|---|---|
| (F1) 상위 chain fusion (L21–L30 같은 1-front chain 을 single block 이 처리) | extend_level_b launch overhead 절감 | ~20 μs (~2%) | 중 (analyze 변경) | 시도 가치 |
| (F2) 중간 level multi-block-per-front (`mf_bigA/B/C` 의 threshold 낮춤, L5–L11) | extend_level_b 의 SM 점유 0.4 → 1.0 | ~50 μs (~5%) | 중-고 | 시도 가치 |
| (F3) Persistent kernel (upper levels only) | launch overhead 30+ × 2 μs | ~30 μs (~3%) | 매우 높음 | 폐기 (효과 작음 + 복잡) |
| (F4) invert_pivot 의 FP32 inverse + FP64 refine | invert_pivot duration 절반 | factor 의 5% → 2.5% | 중 | B=1 에선 작지만 큰 B 에선 큼 |
| (F5) invert_pivot 의 nc<=4 panel 만 invert (rest 는 sequential trsv) | invert_pivot 길이 절반+ | factor 의 5% → 2% | 중 | 유효, solve trade-off 확인 필요 |
| (F6) Multi-stream 독립 subtree (L5-L15 의 평행 분기) | extend_level_b 의 SM 점유 향상 | 잠재 ~15% | 매우 높음 (re-arch) | 후순위 |

**Factor 의 B=1 한계**:
- 본 솔버는 *level-batched + 한 level = 1 launch* 가 이미 최적. 더 줄일 수 있는 launch 수는 *상위 chain 의 30 levels* 뿐 — 효과 작음.
- invert_pivot 은 1 launch 인데 *작업량 자체* 가 P 에 비례라 launch 수가 아니라 *compute* 한계.
- **결론**: (F1)+(F2) 합쳐 ~7% (factor 0.91 → 0.85 ms) 정도가 알고리즘 변경 없는 한도.

### 4.3 Solve 의 B=1 개선 후보

| 후보 | 영향 받는 부분 | 잠재 효과 (solve 0.29 ms) | 난이도 | 권고 |
|---|---|---:|---|---|
| (S1) **CUDA Graph (이미 ON)** | fwd/bwd level 의 launch overhead | 이미 적용 (~10% 효과 측정됨) | — | 적용됨 |
| (S2) 상위 chain fusion (factor 의 F1 과 같은 chain) — fwd/bwd 양쪽에 동일 적용 | 31 × 2 = 62 launches → ~50 launches | ~25 μs (~9%) | 중 | **시도 가치 — solve 의 launch overhead 비중이 더 큼** |
| (S3) selinv off → 단일 trsv | fwd/bwd 의 GEMV 가 sequential trsv 로 | NEG: solve 1.5–2× 느려짐 | 낮음 | **반대 방향**. 단일 solve 가 아닌 한 손해 |
| (S4) fwd + bwd 의 single kernel fusion (한 launch 가 두 phase 다 처리) | fwd→bwd 간 launch + sync 30회 절감 | ~30 μs (~10%) | 매우 높음 (선형 의존성 깨야 함) | 폐기 (의존성 fundamentaly 차이) |
| (S5) Multi-RHS 의 GEMM 변환 (B 자체 외에 RHS 차원 활용) | bwd_level_b 가 tensor core 가능 | bwd 절반 (~30 μs, ~10%) | 높음 | use case 가 multi-RHS 일 때 유효 |
| (S6) Multi-stream 독립 subtree (F6 과 같은) | fwd/bwd level launch overlap | ~20% 가능 | 매우 높음 | 후순위 |

**Solve 의 B=1 한계**:
- solve 의 *kernel/wall ratio 가 거의 100%* 라 setup overhead 짤 게 없음.
- (S2) 가 factor 보다 *더 효과적* — solve 의 한 level kernel 이 더 짧아서 launch overhead 비중이 큼.
- **결론**: (S2) ~9% + (S5) RHS-batch 가 있다면 ~10% 추가. 합쳐 ~15-20% (solve 0.29 → 0.23 ms) 정도.

### 4.4 정리

**Factor 와 Solve 양쪽의 B=1 개선 합치면**:
- Factor: 0.91 ms → ~0.85 ms (~7%)
- Solve: 0.29 ms → ~0.23 ms (~20%, multi-RHS use case)
- 전체 wall-time: 1.20 → ~1.08 ms (~10% 통합 개선)

이게 *알고리즘을 안 바꾼* 한도. 그 이상의 가속은:
- (a) **B > 1 로 가는 것** — § 2-3 이 보인 5–12× 가속
- (b) **선형 방정식 자체를 바꾸는 것** (예: iterative refine 으로 정확도 부족분 보상하고 FP32 only 로 가서 메모리 1/2)
- (c) **GPU 의 FP64 + HBM bandwidth 자체를 키우는 것** — A100/H100

본 솔버의 핵심 use case 가 *NR loop 의 multi-iter, 또는 contingency analysis 의 N-1* 같은 *반복적 풀이* 라면 B>1 이 정답. 단일 시스템 1회 풀이 (B=1, 1-shot) 가 절대적으로 필요한 use case 만 (F1)+(F2)+(S2) 의 ~10% 가치가 있음.

---

## 5. 측정 환경 / 재현

- 빌드:
  - `CLS_INTERNAL_GRAPH=ON` (`/tmp/clsb`) — wall-time 측정
  - `CLS_INTERNAL_GRAPH=OFF` (`/tmp/clsb_nograph`) — nsys / ncu 측정 (graph 가 커널을 가리지 않도록)
- 정밀도: `MF_NO_MIXED=1` (FP64 강제)
- 드라이버: `/tmp/clsb*/custom_linear_solver_run --matrix J.mtx --rhs rhs.mtx --batch B --repeat R [--batch-only]`
- 매트릭스: `/datasets/power_system/nr_linear_systems/case8387pegase/{J,rhs}.mtx`
- nsys: `nsys profile --trace=cuda,nvtx` → `nsys stats --report cuda_gpu_kern_sum`
- ncu metrics: `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `smsp__warps_active.avg.pct_of_peak_sustained_active`, `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed`, `launch__waves_per_multiprocessor`, `gpu__time_duration.sum`
- ncu 샘플: `--launch-skip 5 --launch-count 30` (level kernels), `--launch-count 5` (1-instance invert_pivot)
- 원본 데이터:
  - nsys: `/tmp/bench/nsys_batch/nograph_B{1,8,64,256}.nsys-rep`
  - ncu: `/tmp/bench/ncu_batch_fp64/B{1,8,64,256}_{kernel}.csv`

### 관련 문서
- `02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md` — single-system level batching (B=1 의 base path)
- `03-optimization-notes/01-fp32-batched-kernel-optimization.md` — mixed-precision batched factor 설계
- `04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md` — 단일 시스템 (B=1 등가) nsys
