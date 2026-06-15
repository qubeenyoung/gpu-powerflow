# Kernel Engineering — substrate 미세최적화·병목진단·결정로그

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 연구 novelty 가 아닌 GPU 엔지니어링 레버(tier 커널·동기화·디스패치·staging)의 측정·결정 종합.

이 문서는 substrate 축이다. tensor-core 의 연구 novelty 스토리(thin-K trailing·TF32·TC 가능성)는
별도 문서 [`02-tf32-trailing-gemm.md`](02-tf32-trailing-gemm.md),
[`03-tensor-core-investigation.md`](03-tensor-core-investigation.md),
[`../02-design-analysis/04-gemm-fraction-tc-ceiling.md`](../02-design-analysis/04-gemm-fraction-tc-ceiling.md) 으로 분리되어 있고,
전체 흐름은 [`../storyline.md`](../storyline.md) 가 묶는다. tier 임계값 근거는
[`../02-design-analysis/05-tier-thresholds.md`](../02-design-analysis/05-tier-thresholds.md).

대상 HW 는 전부 RTX 3090 (GA102, sm_86), CUDA 12.8. 핵심 워크로드는 동일 sparsity 를 공유하는
B 개 power-grid Newton Jacobian 의 multi-batch factor/solve (한 번 analyze, B 번 numeric).

---

## 1. 3단 tier 커널 + analyze 최적화

power-grid multifrontal 의 front 분포는 극단적으로 비대칭이다: `fsz ≤ 16` leaf 가 front 수의
95–96% 인데 flop 의 5% 만 차지하고, flop 대부분은 mid front 에, 소수의 big separator front 가 깊은
순차 의존을 만든다. 단일 block-per-front 커널은 이 분포에서 ¾ idle thread + full block barrier 로
latency/occupancy bound 가 된다. → **front size 에 따라 3개 커널로 분기**.

### 1.1 small = warp-packed (fsz ≤ 32)

1 warp = 1 (front, batch), `SMALL_WARPS=8` warps/block (256 thread). front 를 per-warp shared 로
COALESCED staging, block barrier 대신 `__syncwarp()`, L/U-only write-back. 기존 128-thread block-per-front
(¾ idle + block barrier)을 대체. 지배 bottom level 이 **2.47 → 1.12 ms** 로 떨어지고 compute-bound(76%)로
전환. solve 도 동형(`mf_{fwd,bwd}_small_warp_b`, 8 warps/block) — solve −30~41%.

### 1.2 mid = shared-resident (32 < fsz ≤ 159)

front 전체를 dynamic shared 로 staging(level 의 max fsz² 로 sizing, sm_86 의 99 KB opt-in cap), panel
LU / U-solve / trailing / extend-add 를 매 ~nc 패스마다 global 재독 없이 shared 에서 수행. write-back 은
**L/U 만**(uc×uc CB 는 shared 에 남겨 extend-add 에 재사용 → DRAM-bound level 의 write traffic 대부분 절감).
block thread 는 front size 에 따라 64/128/256.

### 1.3 big = 1024-thread (fsz > 159)

shared 예산 초과라 global 커널 유지, 단 **1024 thread/block**. top level 은 front 가 9–25개뿐이라 occupancy 가
near-zero — 많은 warp 를 한 block 에 packing 해 긴 순차 의존을 hide. 70k factor **0.87 → 0.77**.

### 1.4 analyze: GPU graph build + 데드패스 제거

- **GPU symmetric adjacency graph build** (`matrix::build_symmetric_graph_device`): A+Aᵀ 패턴을
  `thrust::sort`/`unique` 로 GPU 에서 빌드. CPU serial adjacency build (n<32768 에서 metis_nd 의 30–46% 차지,
  9241: 40ms 중 19ms)를 대체 → 전 size 에서 ~0.6–4.4 ms.
- **데드 Sx emit 경로 제거**: `build_symmetric_filled`/`emit_map`/`d_Sx` 는 현 solve 가 dense front 직독이라
  소비 안 됨. 제거로 `case_SyntheticUSA` analyze 에서 ~100 ms.
- **METIS graph reuse**: ordering 후 `symmetric_pattern` 재빌드 대신 기존 METIS graph 를 perm/iperm 으로 relabel.
- **No-sort device CSR→CSC**, **device ordered-CSC build**, **A-entry map 절반화**(owner=panel_of[min(i,j)] 가
  pivot 열이라 한쪽 lookup 생략).
- **parallel root induce**(`induce_par`): root 의 serial subgraph 추출이 임계경로 → 70k −17ms, 25k −3ms.

#### 시핑된 핵심 speedup

| case | n | analyze | factorize | solve | relres |
|---|---:|---|---|---|---|
| case3120sp | 5,991 | 18.17 → 14.16 (−22%) | 0.317 → 0.255 (−20%) | ~0 | 9.3e-5 |
| case6470rte | 12,485 | 37.43 → 26.34 (−30%) | 0.437 → 0.346 (−21%) | ~0 | 3.3e-5 |
| case9241pegase | 17,036 | 54.36 → 35.82 (−34%) | 0.744 → 0.517 (−31%) | 0 | 2.6e-6 |
| case_ACTIVSg25k | 47,246 | 100.4 → 86.5 (−14%) | 1.351 → 1.265 (−6%) | 0.639 → 0.619 (−3%) | 3.4e-13 |
| case_ACTIVSg70k | 134,104 | 276.8 → 237.4 (−14%) | 2.849 → 2.764 (−3%) | 1.173 → 1.122 (−4%) | 3.9e-11 |

(단위: ms, 단일 시스템). 대-case analyze 가 ~14% 에 막히는 이유는 단일스레드 METIS vertex separator
(70k root separator 만 41ms) — multilevel coarsening 비용이라 tunable 아님. 진짜 lever 는 batching:
B=128 per-system factor/solve −52~91% (작은 case 일수록 큼), saturation 까지 compute-bound 로 전환.

---

## 2. 병목 진단

### 2.1 tier 별 B-saturation point (factor per-sys, tf32, --repeat 32)

per-sys 감소율 ≈ 1.0 (추가 batch 가 throughput 게인 0)이 saturation.

| case (dominant tier) | saturation B | ncu 1순위 stall (B=64) | 진단 |
|---|---|---|---|
| case8387 (small) | B=256 도 미포화 (~17% 여지) | barrier **0%**, long_scoreboard 203% | memory-latency |
| ACTIVSg25k (mid) | **B=64** | barrier **94%**, scoreboard 257%, warps_active 31% | sync + memory |
| USA (big) | **B=16** | barrier **1907%**, DRAM 23%, inst/cycle 0.26 | **극단 sync** |

big 의 inst/cycle 0.26 = Ampere peak(4)의 6.5%. barrier 1907% = 32 warp 중 평균 19 warp 가 sync 대기.
small tier 가 가장 늦게 포화(작은 work, latency dominant), big tier 가 가장 빨리 포화하면서 wall delta 가
가장 안 남는다(panel LU 의 1024-thread sync 체인 동안 thread 활용 nc/1024 ≈ 1–2%).

### 2.2 big 커널 phase breakdown (70K fp32 B=1, MB 경로로 분리)

| phase | time | 비중 |
|---|---|---|
| **factor_big_panel** (Phase 1 LU + Phase 2 U-solve) | 30.9 ms | **61.4%** |
| factor_big_trailing (Phase 3) | 9.8 ms | 19.5% |
| factor_big_extend (Phase 4) | 9.6 ms | 19.2% |

(USA 도 동일 61/19/19). **panel(LU+U-solve)이 압도적.** TC 가 가속하는 trailing 은 19.5% 뿐 — TC 가 안
보이는 직접 이유. panel 은 memory 아닌 **barrier-bound**: ncu `factor_big_panel<float>` 에서
stall_barrier 12–13, lts hit **92%**(L2 상주), DRAM 1.7%, warps_active 66%. 순차 pivot 의 1024-thread
`__syncthreads` 체인이 임계경로 — 각 pivot 의 rank-1 update(uc·(nc−k)/1024)는 극소수인데 barrier 가 그걸 압도.

### 2.3 trailing thin-K TC underfill

trailing = **C[uc×uc] ← C − L[uc×nc]·U[nc×uc]**, 즉 M=N=uc, K=nc.

| case | big front 수 | M=N=uc (med/max) | **K=nc** | 시스템당 trailing FLOP |
|---|---|---|---|---|
| 25K | 4 | 119 / 123 | 12 | 1.4 MFLOP |
| USA | 62 | 137 / 205 | 20 | 51 MFLOP |
| 70K | 53 | 141 / 242 | 20 | 47 MFLOP |

전형적 GEMM = ~140×140, **K≈20**. mma m16n8k8 기준 K=20 = **2.5 K-tile 뿐** → load 를 amortize 할 K-loop 이
없어 memory-bound. ncu (70K B=64 `factor_big_tf32_ptx`): tensor pipe **0.2%**, isolated trailing 도 0.4%
(분리해도 TC pipe 안 오름). trailing 시간 분해(CLS_TRAIL_NO_MMA/NO_STAGE): staging 1–4%, mma 자체 ~0%,
"mma+drain" 35–39% 의 정체는 mma 가 아니라 **C drain**(uc²≈140² 결과를 global write, 검산 84ms ≈ 측정 92ms).
amalgamation 으로 front 를 키워도 uc(M,N)만 커지고 K=nc(panel width)는 거의 안 커서 TC 를 근본적으로 못 살린다.

### 2.4 mid latency-bound staging stall

factor_mid 는 B=64 factor 의 ~37% (지배 커널). ncu (25K fp32 B=64): SM throughput 22%, DRAM 16%,
warps_active 73%(점유율 충분)인데 **long_scoreboard 8.4/issue** 잔존. `stage_in_async` 가 cp.async 로
front 를 shared 에 올린 뒤 `__pipeline_wait_prior(0)` 로 staging 지연을 동기 barrier 로 노출 — 모든 block 이
커널 진입 직후 **같은 위상**으로 메모리를 기다려 occupancy 가 이 구간을 못 채운다(다들 staging 중이라 채울 연산
없음). 따라서 "독립 warp 추가"(naive packing/occupancy) 레버는 소진. 진짜 lever 는 software-pipelined staging
(front A factorize ∥ front B prefetch)이며, falsifiable 예측은 "pipelined → long_scoreboard 감소, naive →
무효". L2 hit 42%(load 58% cold), store sectors/req 6.84(extend_add scatter 비합착)도 부수 lever.

---

## 3. 동기화 / 디스패치 레버

### 3.1 T4.x 결과 (sync 병목 해소 라운드)

`factor_small` 은 이미 `__syncwarp` + warp-per-front 라 barrier 미관측. T4 는 mid/big 의 FP32 path 대상.
factor_mid 의 per-front sync ≈ 3·nc+5 (case8387 nc=8: 23회, USA mid nc≤20: 65회).

| 단계 | 변경 | ncu 효과 | wall 효과 | 결정 |
|---|---|---|---|---|
| **T4.3** cp.async stage-in | `__pipeline_memcpy_async` 로 global→shared, register round-trip 제거 | USA B=1 factor_mid dur −11% (43.2→38.6μs), scoreboard 31.7→28.2% | USA B=1 −4%, case8387 B=64 −4% | **기본 ON** (sm_80+) |
| **T4.2.A** row-fused panel LU | divide+rank-1 update 를 per-row register-resident `lik` 로 fuse, sync 2·nc→nc | case8387 barrier −2.7pt | case8387 B=1 −7.8%, 그 외 ~0 | **기본 ON** (nc≤12 gate) |
| **T4.1** warp-per-front mid | small 패턴을 mid (32<fsz≤thresh)로 확장 | barrier **41%→0%** 인데 occupancy **46%→12%** 추락 | case8387 +10~+136% 회귀, USA thresh=48 −4% | **롤백** (env opt-in, deprecated/mid_warp) |

T4.1 롤백 원인: 한 block 의 8 warp 가 서로 다른 fsz front 처리 → block runtime 이 max-fsz front 에 묶여
load imbalance. small 이 같은 패턴인데 잘 도는 이유는 small level 의 fsz uniformity 덕분(본질적 우월 아님).
variance gate(`CLS_MID_WARP_VAR_GATE=8`)로 case8387 회귀는 제거되나 이득 noise 수준. nc>12 에서 row-fused 는
thread 당 nc serial FMA 가 sync 절감 압도 → USA +13~20% 회귀라 gate 필수. factor_small + cp.async 는 USA B=1
+12% 회귀(fsz=4 → 1 entry/lane, commit/wait overhead > saved latency)로 **미시핑**.

새 커널 도입 시 `cudaFuncSetAttribute(...,99K)` 누락이 silent corruption(48KB 경계 초과 시 spill→garbage)을
만든다는 교훈 — 같은 디렉토리 커널의 attribute opt-in 전부 미러링 필수.

### 3.2 panel-LU reciprocal-multiply (P1)

`F /= piv` → `inv_piv = 1/piv; F *= inv_piv`. Ampere FDIV 22cyc → RCP+FMUL 6cyc (이론 4×). 실측은 작음
(컴파일러가 일부 자동 hoist, divide 가 panel wall 의 일부): case8387 B=64 −0.7%, USA B=1 −2.0%. **기본 ON**
(`CLS_NO_RECIP_PIV` toggle). P2(phase fusion)/P3(parallel U-solve)/P4(bank pad)/P5(warp-spec)는 deferred;
P4 는 후속 라운드에서 **+85% 회귀**(stage-in integer div overhead)로 폐기.

### 3.3 big `__launch_bounds__(512, 2)` (EXP-B)

bigT 1024→512 + `__launch_bounds__(512,2)` 로 register/thread 48→64(spill 허용, 2 block/SM 강제):
barrier stall **1801%→1340% (−26%)**. wall: USA B=64 −2.7%, B=256 −3.4%, ACTIVSg25k B=256 −3.6%,
case8387 B=1 −6.3%. V9h(mid trailing PTX)와 직교 결합 시 USA B=64 **−5.7%**. LB(256,4)는 barrier
−52% 더 줄였으나 warps_active 66→49% 추락 + **USA B=1 +41% catastrophic** → 폐기. EXP-A(full warp-spec
look-ahead pipeline, 300+줄, BG1 예측 20–40%)는 BG3 의 over-estimate(예측 5–15% → 실측 3%) 와 prior art 부재
(Rennich-Davis 2016 도 sparse multifrontal small-front barrier 를 unsolved 로 인정)로 **미실행**, 현실 추정 5–10%.

> **2026-06-10 갱신 — `factor_big_tf32_ptx` 의 `__launch_bounds__` 삭제 (fused 와의 상호작용).**
> `(512,2)` 는 register 를 **64** 로 cap 해 2 block/SM 을 강제하는데, fused trail+extend(§3.6)를 켜면
> 커널이 **~106 register** 를 원해 64 cap 이 **24-byte spill(STACK:24)** 을 강제 → batch 회귀. cuobjdump
> 증거:
>
> | factor_big_tf32_ptx | REG | STACK |
> |---|---:|---:|
> | non-fused, LB(512,2) | 64 | 0 |
> | non-fused, **no-LB** | 80 | 0 |
> | fused, LB(512,2) | 64 | **24 (spill)** |
> | fused, **no-LB** | 90 | 0 |
>
> 결정: 이 커널의 `__launch_bounds__` 를 **삭제**. 실측(USA tf32, serial-ND seed 1588, factor/sys):
> non-fused 기본 path 는 LB 유무 차 **±0.5% (8387/25K/USA, 노이즈)** — EXP-B 의 −2.7~3.8% 이득은 현
> V9h 커널에서 더 이상 재현 안 됨(80 reg/1 block 으로도 big front warp 충분). 대신 fused 의 spill 이
> 사라져 **USA B=256 +4.3%(spill) → −2.2%(no-LB)**, B=1 −8.7%. → LB 의 최적값은 fused 여부에 갈리고,
> 삭제가 양쪽에 안전. (`factor_big_fp16_ptx`·`factor_big_shared_tf32_blocked` 의 LB 는 유지 — fp16 은
> reg 여유로 fused 에도 spill 없음.) 재현: [`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md) §10.

### 3.4 fsz-band-split dispatch

문제: dispatch 는 launch 단위로 자원(shared/fsz_cap/tier kernel)을 결정하는데 일감(front)은 panel 단위로
이질적 → mixed level 의 fsz=20 panel 이 max_fsz=120 의 tier 에 징발. 해법: analyze 단계에서 panel 을 band
(`≤32/≤48/≤64/≤96/≤128/>128`)로 stable 재정렬, dispatch 가 band-contiguous sub-range 마다 worst-fit tier/fsz_cap 로
sub-launch. kernel/plan/etree 무수정 — launch SHAPE 만 동질화.

| Case | B | full factor Δ | direct non-GEMM Δ |
|---|---|---|---|
| case13659pegase | 256 | −7.4% | **−30.1%** (유일한 직접 30% 점, thin margin) |
| ACTIVSg25k | 64 | −10.4% | −8.1% |
| ACTIVSg25k | 256 | −11.8% | −12.4% |
| ACTIVSg70k | 64 | −11.4% | −18.4% |
| ACTIVSg70k | 256 | −12.7% | −16.3% |
| case_SyntheticUSA | 64 | **−15.1%** | −13.9% |
| case_SyntheticUSA | 256 | −13.1% | −16.3% |

(70K/USA −11~15%, 8387 계열 약칭 −22% 는 small 비중 dominant case 의 tier 재라우팅 효과). B=1 은 gate 차단.
14건의 미세변형(fine band split, scatter 512-block, mid block-size, `__launch_bounds__(256,2)` on small/mid,
`__ldg` metadata, 70K big 512-thread)이 모두 회귀 → 6-bucket coarse band 가 이 layer 의 local optimum.

### 3.5 big-trailing L/U staging

big tier 의 scalar(FP64/FP32) trailing 은 front 가 global-resident 라 L/U 원소를 출력당 ~uc배(USA 145×,
25K 120×) 중복 global read (산술강도 1/16 FLOP/byte, memory-bound). `factor_big_staged<T>`: L/U 패널만
shared staging(`2·level_max_nc·level_max_uc ≤ 96KB` 일 때 라우팅, 초과시 scalar fallback).

| | factor_big 커널 | end-to-end factor |
|---|---|---|
| USA fp64 | **−23%** (909→700ms) | −11% |
| 25K fp64 | | −5.7% |
| case8387 fp64 | | −5.9% |
| fp32 (USA/25K) | | −1.6~1.7% |

정확성 불변(25K fp64 e-13), compute-sanitizer 전부 clean. `CLS_NO_BIG_STAGED` toggle.

### 3.6 fused trail+extend (scalar) — CB 왕복 제거

§3.5 staging 후에도 trailing 의 진짜 병목은 **CB(uc×uc Schur block)의 global write→read 왕복**이다
(note 30/53: uncoalesced C-drain store 2.5×). non-fused 경로는 trailing 이 CB 를 front 에 쓰고, 이후
`extend_add` 가 그 CB 를 도로 읽어 부모에 atomicAdd 한다. **fused** 는 trailing 누산기를 `F[off] − acc`
로 계산해 **부모에 직접 atomicAdd**, CB 의 global write+read 를 통째로 제거(이후 `extend_add` 스킵).

구현: `trailing_update_staged<T, true>` (이미 존재하던 `FuseExtend` 분기) 를 `factor_big_staged`·
`factor_mid` 의 `fsz > 48` front 에 배선(`fsz ≤ 48` 은 `lu_small_front` 가 Phase 1+3 을 융합해 trailing
functor 를 호출하지 않으므로 제외 — 누락 시 CB 유실 버그). `CLS_FUSE_SCALAR_TRAIL_EXTEND` (기본 OFF).

**실측** (fp32, serial-ND seed 1588, 확정 cap, factor/sys, repeat=61):

| case | n | B=1 | B=256 |
|---|---:|---:|---:|
| case1197 / 3012wp / 8387 | ≤15K | ~0% (노이즈) | ~0% |
| case_ACTIVSg25k | 47,246 | **−4.0%** | **−1.9%** |
| case_SyntheticUSA | 156,255 | **−7.8%** | **−7.4%** |

→ **large-case 전용 +4~8% lever, B=1·B256 둘 다 유효.** 작은/low-fill 은 big front 가 적어 CB 트래픽
자체가 미미 → 무효. tf32 fused(note 55: B=1 만 이득, 배치 손해)와 **반대** — scalar 는 trailing 이 느려
C-drain 이 wall 에 노출돼 있고 register spill 도 없어, 제거 이득이 배치에서도 그대로 남는다. relres 는
잘-conditioned 케이스 base 와 동일; USA 만 atomicAdd 누적 순서로 floor(~1e-2) 내 소폭 변동. 재현:
[`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md) §9.

### 3.7 mid → shared-blocked TC + B-적응 thread

`factor_big_shared_tf32_blocked`(big-low 129~159 용 front-resident shared + blocked TC)를 TF32 **mid
tier 전체**에 적용(`CLS_MID_AS_BIG_SHARED`). mid-dominant 케이스에서 이득이 큰데, **thread 수가
효과를 가른다**(커널은 `nt` 무관 — `CLS_MID_AS_BIG_THREADS`):

| 25K tf32 (vs base factor_mid) | 512 thr | 256 thr | 128 thr |
|---|---:|---:|---:|
| B=1 | **+17.0%** | +16.7% | +5.3% |
| B=256 | −21.4% | −0.9% | **+2.7%** |

- thread 최적이 **B 에 정반대**(B=1↑ many, B=256↑ few) → **B-적응(B=1→512, B=256→128)** 이면 25K 가
  양쪽-B 승. 감속은 메모리 아님(연산 전부 shared) — **occupancy**(작은 front 에 512 thread 과잉).
- **B=1 win 의 본체는 mma**: 같은 blocked 구조에서 inner trailing 만 mma↔scalar 교체 시 mma 가 scalar
  보다 **+21%(512 thr)~+49%(128 thr)**. blocked-scalar(0.727)≈base(0.726) → 구조 자체는 ~0, **전부
  mma**. blocked **BK=8** 가 K-tile(mma K=8)을 풀로 채워 **thin-K underfill 우회**
  ([`03-tensor-core-investigation.md`](03-tensor-core-investigation.md) §8). 정확도 무손실(Ozaki).

`CLS_MID_AS_BIG_SHARED`/`CLS_MID_AS_BIG_THREADS`/`CLS_BLOCKED_SCALAR_UPDATE`(mma 분리 진단). 재현:
[`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md) §11.

**운영 정책 `CLS_MID_TC_POLICY`** (기본 OFF): 7케이스×B 스윕에서 thread 최적은 occupancy 로 결정 —
`blocks=level_size*B < num_SMs ? 512 : (<4*num_SMs ? 256 : 128)`. `dispatch_factor_mid` 가 **B≤4 에서만**
mid 를 shared-blocked TC 로 라우팅(base mid 는 B<64 에서 scalar)하고 occupancy thread 적용. **정책 vs
base: B=1 +2.2~14.5%, B=4 +0.1~10.2%, B≥16 ~0(무회귀)**; 단일-512 가 만들던 B=256 −20~33% 회귀가 gate 로
제거. fp32 대비 분해상 *저배치 이득의 본체는 TC 정밀도가 아니라 이 라우팅*이다(§12). 재현:
[`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md) §12.

---

## 4. 핵심 메타-교훈

### 4.1 "sync 절감 ≠ wall 단축" (변환률 ~1/10)

이 codebase 의 일관된 발견: ncu barrier stall 을 크게 줄여도 wall 은 그 일부만 따라온다.

| 변경 | sync(barrier) 감소 | wall 단축 |
|---|---|---|
| T4.1 warp-per-front mid | 41% → **0%** | net 회귀 (load imbalance) |
| P1+P2 fusion (별도 kernel) | sync 64%↓ | −1~4% (B≥16) |
| LB(512,2) (EXP-B) | barrier 26%↓ (1801→1340%) | −2.7~3.8% |

→ **변환률 ≈ 1/10 ~ 1/16.** 이유: (1) ncu 의 `% of issue cycles` 는 wall fraction 이 아님 — 한 stall 을
줄이면 다른 stall(long_scoreboard, wait, mio)이 그 자리를 차지. (2) wave 단위 latency hiding 이 barrier wait 의
상당량을 이미 흡수(1907% 는 "동시 대기 warp 수"지 wall fraction 아님). (3) occupancy 의 가치가 sync 절감보다 큼.
직접 함의: warp-spec/persistent 류 "큰 sync 잠재" 레버는 같은 ceiling 에 걸릴 위험이 크므로 ROI 를 의심하고,
단순한 occupancy lever(launch_bounds)나 dispatch SHAPE(band-split) 가 더 안전한 wall 회수처다.

### 4.2 USA fp32 relres 는 데이터셋 conditioning floor (~1e-3)

USA(~80K) fp32 B=64 의 `batch_relres` 가 run 마다 ~1e-3 ↔ 0.02–0.06 으로 bimodal. cuDSS 0.7.1 레퍼런스
대조(J.mtx/rhs.mtx): cuDSS fp64 2.0e-12, **cuDSS fp32 8.9e-4 / 1.1e-3 (B=1/B=64)**. 즉 USA Jacobian 은
fp32(7자리)로는 ~1e-3 이 한계인 ill-conditioned 행렬 — 레퍼런스가 확인. 우리 "good" run(~1e-3)은 cuDSS 와
일치하고, 변동(0.02–0.06)은 multi-threaded METIS-ND 의 비결정적 ordering 이 fp32 오차를 증폭시킨 것(baseline
동일, staging 무관). **교훈**: flaky 벤치마크는 (1) 안정 케이스(fp64 e-13급) + (2) baseline 동일조건 대조
없이 correctness 를 판단하지 말 것 — 유효한 −11% 최적화를 단일 flaky 케이스의 절대 relres 로 revert 직전까지 갔다.

### 4.3 parent-update fan-in 은 8387 의 missing lever 아님

`--dump-fronts` 의 fan-in/update-size 프로파일: 8387 은 update elem 의 25%가 `≤16`, fan-in 9+ 가 21% 뿐
(uc>32 가 11.7%) — high-conflict reduction 아닌 tiny traffic. 25K/USA 는 `>256` update 가 56–64% 라
parent 재설계는 거기서 시작해야. `CLS_DISABLE_EXTEND_ADD` upper-bound: extend 전부 제거해도 8387 의 안정
1.2× TF32/FP32 ratio 를 못 만든다 → parent-update 제거는 8387 TC-enabler 아님(절대 FP32 만 개선, ratio 미개선).

---

## 5. 미채택 미세실험

- **solve spine multistream** (21): spine fusion(cnt=1 root chain persistent kernel)은 −3~8.6% net positive
  로 채택, 그러나 cnt>1 narrow band persistent(+19~38%)·multi-stream subtree(B=1 marginal)·shared staging
  (+9~18%)은 회귀 또는 무효 — solve 가 compute-bound 아니라(SM throughput 0.05–2%) precision/staging 레버 무관.
- **subwarp small tiling** (24): 1 warp 를 SG-lane sub-group(fsz≤8→8, ≤16→16)으로 쪼개 32/SG front 동시 처리,
  lane 100% 활용 + MLP. case8387 FP64 B=64 F+S **−14~16%** (solve −22%), occupancy gate 로 B=1 회귀 차단 →
  **채택**(default-on). 교훈: small tier 의 진짜 lever 는 resident warp 수가 아니라 per-warp lane 효율.
- **multiblock big trailing B=1** (25): underfill level(`level_size×B < num_SMs`)에서 trailing 을 element-tile
  multi-block 으로 분리 → B=1 factor USA FP64 **−58%**, 25K −25%, 8387 −12%, USA FP32 −23% → **채택**(per-level
  gate, batched 무영향). 단 proper TC multi-block(M-strip + full U 중복 staging)은 scalar-MB 보다 19% 느려
  dead-end — severe-underfill 은 연산속도 아닌 fanout+오버헤드 게임.

---

## 6. 결정 로그 표

| 레버 | 기본 | 측정 효과 | 토글 |
|---|---|---|---|
| small warp-packed factor/solve | ON | bottom level 2.47→1.12ms, solve −30~41% | — |
| mid shared-resident factor | ON | L/U-only write-back, DRAM write 대부분 절감 | — |
| big 1024-thread factor | ON | 70k factor 0.87→0.77 | — |
| analyze GPU graph build + Sx 제거 | ON | adjacency ~0.6–4.4ms, USA analyze −100ms | — |
| T4.3 cp.async stage-in (mid) | **ON** sm_80+ | USA B=1 factor_mid −11%(ncu), wall −3~4% | `CLS_NO_CP_ASYNC` |
| T4.2.A row-fused panel LU (nc≤12) | **ON** | case8387 B=1 −7.8%, nc>12 회귀 차단 | `CLS_NO_ROWFUSED_LU` |
| T4.1 warp-per-front mid | **OFF** | barrier 41→0% 인데 occ 46→12% net loss | `CLS_MID_WARP_THRESH` |
| factor_small cp.async | **OFF** | USA B=1 +12% 회귀 | (미시핑) |
| P1 reciprocal-multiply | **ON** | case8387 B=64 −0.7%, USA B=1 −2.0% | `CLS_NO_RECIP_PIV` |
| P4 bank-conflict pad | **OFF** | +85% 회귀 | (폐기) |
| big `__launch_bounds__(512,2)` | opt-in | barrier −26%, USA B=64 −2.7~3.8%, +V9h −5.7% | `CLS_TF32_BIG_LB_512_2` |
| big `__launch_bounds__(256,4)` | **OFF** | USA B=1 +41% catastrophic | (폐기) |
| fsz-band-split dispatch | ON (gate) | 70K/USA −11~15%, 8387 −22%, B=1 차단 | (band order gate) |
| big-trailing L/U staging | **ON** | USA fp64 factor_big −23% / end-to-end −11% | `CLS_NO_BIG_STAGED` |
| solve spine fusion (cnt=1) | ON | 9241 −8.6%, 70K −3.2% (평균 −5.8%) | — |
| subwarp small tiling (SG-lane) | **ON** | case8387 FP64 B=64 F+S −14~16% | `CLS_SMALL_SG32_ONLY` |
| multiblock big trailing (underfill) | **ON** (gate) | B=1 USA FP64 −58%, FP32 −23% | `CLS_NO_BIG_MB` |
| EXP-A warp-spec panel LU | 미실행 | 추정 5–10% (BG1 예측 20–40% 과대) | — |

### 관련 문서

- [`../storyline.md`](../storyline.md) — 전체 흐름
- [`02-tf32-trailing-gemm.md`](02-tf32-trailing-gemm.md), [`03-tensor-core-investigation.md`](03-tensor-core-investigation.md) — TC novelty 축
- [`../02-design-analysis/04-gemm-fraction-tc-ceiling.md`](../02-design-analysis/04-gemm-fraction-tc-ceiling.md) — GEMM 비중 / TC ceiling
- [`../02-design-analysis/05-tier-thresholds.md`](../02-design-analysis/05-tier-thresholds.md) — tier 임계값 근거
