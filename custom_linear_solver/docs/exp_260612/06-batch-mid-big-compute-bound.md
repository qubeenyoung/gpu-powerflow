# 멀티배치 factorize 를 compute-bound 로? — 진단·시도·물리적 벽

> **상태**: 측정·프로토타입 완료   **날짜**: 2026-06-12   **GPU**: RTX 3090
> **목표**(260612_batch_factorize_goal.md): B=16/64 fp32 factorize 를 compute-bound 로 전환해 1.2–1.4×.
> **한 줄**: 지배 커널 `factor_mid` 는 **memory bound**(arithmetic intensity AI≈2 FLOP/byte ≪ fp32 roofline ridge ~10) 이며, 그것도 **bandwidth 가 아니라 latency** 에 막혀 있다(DRAM 5–8%, long_scoreboard 40). 원인은 B=1 과 동일한 **thin-K front**(nc median 15) — load fsz²·writeback fsz²·uc² atomics 대비 compute 는 fsz²·nc 뿐. **AI≈2 인 연산은 compute-bound 로 만들 수 없다.** 실측 천장: tf32(blocked-LU) **1.15×(25K)/1.09×(70K)**, true-fp32 register-block trailing **+4%**. 1.2–1.4× 미달.

## 1. 베이스라인 커널 분해 (nsys, B=64, fp32)

| case | factor_mid | factor_small | factor_big | assemble | 지배 |
|---|---:|---:|---:|---:|---|
| 25K | **44%** | 24% | 0 | 8% | mid |
| 70K | **32%** | 27% | **16%** | 6% | mid+big 48% |

→ 목표대로 mid(+big)가 지배. (25K mid front 는 309개뿐이나 개당 크기가 커 시간 지배.)

## 2. 진단 — 왜 compute-bound 가 아닌가 (ncu, B=64)

| 커널 | warps | **FMA** | DRAM | 지배 stall | 병목 |
|---|---:|---:|---:|---|---|
| mid (25K) | 25–57% | **5%** | 5–8% | **long_scoreboard 40** | global-load **latency** |
| big (70K) | 50–69% | 20–25% | **40%** | long_scoreboard 4.5 | DRAM **bandwidth** |

- **TC pipe 0–1.1%** (tf32 에서도): front 가 작아 16×8×8 mma 가 안 채워짐. tf32 의 이득은 TC 가 아니라 **blocked-LU 구조**.
- **mid AI 계산**: compute ≈ 2·uc²·nc, traffic ≈ (load fsz² + writeback `nc·(fsz+uc)` + atomic uc²)·4B (writeback 는 이미 L|U 패널만, uc² CB 제외 — `writeback_factored`). fsz=80·nc=15·uc=65 → compute≈127K, traffic≈(6400+1950+4225)·4 → AI ≈ **3.5 FLOP/byte**. roofline ridge(fp32, Ampere) ≈ 10 → **구조적 memory-bound**(compute-bound 불가).
- 단 mid DRAM 은 5–8% 로 **bandwidth roofline(~60%)에 한참 못 미침 = latency bound**. 이론상 roofline 까지 가면 ~5–8× 여지가 있으나, occupancy 가 **whole-front shared-residency** 로 묶임(fsz55 → smem 28.8KB → 3 block/SM; fsz↑ → 더 적음) → latency 은닉 부족. **이 occupancy 벽을 깨는 게 유일한 큰 레버이나, 그것(non-resident/tiled·systems-per-block)은 회귀**(§3).

## 3. 시도한 레버와 결과

| 레버 | 결과 | 해석 |
|---|---|---|
| **register-block trailing** (CLS_TRAIL_RB, L 재사용·U 연속) | ✅ **+4%**, 정확 | 3-phase trailing 의 FMA 밀도↑. 한계: trailing 만, fused 경로 미적용 |
| mid thread 128→256 | ◑ +5% | shared-limited level 의 warp slot 회복 |
| **systems-per-block 더블버퍼** (CLS_MID_SYSBLK, load(g+1)↔compute(g) overlap) | ✗ **0.48–0.79× 회귀** | block 수 G배 감소 → inter-block latency 은닉 손실 > intra-block overlap 이득. 베이스라인의 many-small-blocks 가 이미 더 나음 |
| amalgamation (panel width 8→48) | ✗ 무효 | nc 가 안 늘어 AI 불변 |
| big-mid front → big tier (CLS_MID_FSZ_MAX) | ✗ 회귀(0.094→0.13) | big 의 global-residency 재읽기 traffic > 낮은 occupancy 손실. **shared-residency 가 traffic 상 우월** |
| **blocked-fp32** (CLS_MID_BLOCKED_FP32, tf32 의 BK=8 구조 + scalar trailing) | ✗ **1.00×(neutral)** | tf32 의 1.15× 가 blocking 구조가 아니라 **TC trailing(block_update_tf32_tc)** 에서 옴을 증명. 정확(relres 일치) |
| **Ozaki 3×TF32 TC trailing** (CLS_TF32_OZAKI_TC2, fp32-정확도 TC) | ◑ **1.06–1.08×(25K)/1.01–1.02×(70K)** | **fp32 정확도 유지**(relres 1.46e-4→1.61e-4). 3-split 이 mma 3배라 plain tf32(1.15×)보다 느림. fp32-정확 TC 의 천장 |
| **tf32 (TC trailing, precision trade)** | ◑ **1.15×(25K)/1.09×(70K)** | 이득은 TC trailing(Ozaki 가 fp32-정확 버전=1.07×). **정확도 손실**(relres 1.5e-4→5.7e-2) |

## 4. 결론 — compute-bound 전환은 thin-K AI 벽에 막힘

- mid 커널 AI≈2 → **어떤 커널 재설계로도 compute-bound 불가**(연산이 byte 당 너무 적음). B=1 의 under-fill 벽과 **같은 뿌리(thin-K front)**, batch 차원에서 재현.
- 가능한 건 **latency→bandwidth roofline 접근**(occupancy↑로 5–8% DRAM 을 끌어올림)이나, whole-front shared-residency 가 occupancy 를 묶고, 그걸 깨는 non-resident/tiled·systems-per-block 은 회귀.
- 실측 천장: **fp32 RB +4%**, **tf32 blocked 1.09–1.15%**(정확도 trade). **1.2–1.4× 미달.**
- **정직성**: tf32 의 1.15× 를 "fp32 가속"으로 쓰는 건 정밀도 변경. true-fp32 compute-bound 1.2×+ 는 thin-K 구조상 불가.

## 4b. ★ fused-assembly — 검증된 1.2× 천장 (다음 구현 타깃)

factorize wall 의 **입력 assembly(memset+scatter)가 17%**(25K B=64) 임을 측정(`CLS_SKIP_ASSEMBLE` ceiling probe — assembly 생략 시 timing):

| case | B | assembly share | **천장(=max 가속)** |
|---|---:|---:|---:|
| 25K | 16/64 | 17–18% | **1.20–1.22×** |
| 70K | 16/64 | 14–15% | 1.16–1.18× |

분해(25K B=64): **memset 11% + scatter 6%**.
- **scatter**(`assemble_front_values`, nnz atomicAdd): 작은 부분, factor 커널의 matrix-gather 로 흡수 가능(per-front nnz bucket 필요).
- **memset**(front_total×B ≈ 596MB 0-fill): 큰 부분. 자식이 부모 front 에 **atomicAdd(extend_add)** 하므로 zeroed buffer 필수 → 단순 제거 불가.

**설계 — gather-based(left-looking) assembly 로 전환**(memset+scatter+extend 동시 제거):
1. 부모 factor 커널: shared 를 zero → matrix 값 gather → **각 자식의 front CB 영역을 직접 read** 해 `asm_local`(기존 child→parent 맵을 *gather* 로 사용)으로 shared 에 더함 → factor → L/U **및 CB** 를 global front 에 write.
2. memset 제거(부모가 shared zero), scatter 커널 제거(부모가 matrix gather), extend_add atomic 제거(부모가 gather, 자식은 CB 를 자기 front 에 regular write).
3. traffic: memset(Σfsz²) 제거 + atomic→regular. CB read/write 는 기존 load/extend 와 동급.
- **분석상 ~17% 회수 → RB(+4%)와 합쳐 25K ~1.22–1.25× (목표 달성권)** 으로 기대.
- 비용: analyze(per-front nnz bucket + per-parent child CB gather list) + 전 tier factor 커널 재구성. multi-file·high-risk(memset 은 전 tier 공통이라 mid 만 부분 적용 불가 — small/big 도 gather 로).

### ✗ 구현 결과 (2026-06-12) — 정확하나 **회귀(0.87–0.91×)**

gather-based assembly 를 **전 tier(small/mid/big) + analyze(nnz bucket·child list) + 드라이버·CUDA graph 까지 완전 구현**(`CLS_GATHER_ASM`, default off). 디버깅으로 두 버그 수정: (a) mid 의 trailing 후 `__syncthreads` 누락(writeback_full race), (b) **setup-time 캡처된 factor CUDA graph 가 legacy 커널을 담고 있어** gather 시 scatter 만 빠지고 그래프는 legacy 재생 → NaN; gather 전용 graph 를 value-ptr keyed 로 lazy 캡처해 해결.

**정확성**: single-system relres 가 default 와 일치(case30~USA 전부, FMA 재정렬 오차 수준). ✓

**성능 (per-system factorize_ms, batch fp32)**:

| case | B | default | gather | |
|---|---:|---:|---:|---:|
| 25K | 16 | 0.112 | 0.126 | **0.89×** |
| 25K | 64 | 0.095 | 0.105 | **0.91×** |
| 70K | 16 | 0.433 | 0.490 | **0.88×** |
| 70K | 64 | 0.395 | 0.455 | **0.87×** |

→ **memset(11%)+scatter(6%) 제거 이득을 gather 고유 비용이 상쇄**. 원인: legacy 는 scatter+extend 로 front 를 **연속(coalesced)** 조립해 두고 factor 가 한 번에 contiguous read(stage_in) 하지만, gather 는 **자식 CB 를 흩어진(scattered) global 위치에서 읽고**(Σuc²≈fsz², 비-coalesced) + shared atomicAdd(matrix·children) + writeback_full(fsz²). read 총량은 같으나 coalescing 손실 + atomic 이 memset 절감을 초과. **17% ceiling 은 "memset/scatter 를 공짜로 없앤다"는 가정이었고, gather 는 그 일을 다른 형태로 되갚는다.**

**per-kernel 결정적 증거**(nsys, B=64): gather 가 assemble 커널(5.7ms)+memset 을 없애지만 **커널 자체가 느려짐** — factor_mid 31→49ms(**+55%**), factor_small +35%. 즉 분리된 assemble+memset 보다 **커널 내부 on-the-fly 조립(scattered child read+atomic+writeback_full)이 더 비쌈**. (ncu 상 일부 leaf level mid 는 long_scoreboard 40→7 로 개선되나, child-gather 가 있는 deep level 이 상쇄.)

**coalesced CB-buffer 변형도 구현·검증 → 더 나쁨(0.80–0.86×)**: 자식 CB 를 parent-grouped 전용 buffer 에 써서(`d_cb_batch_f`, cb_pos) 부모가 coalesced 로 읽게 함. 정확(relres 일치)하나 더 느림 — write_cb 별도 pass + writeback_factored(L/U panel, strided) 가 scattered 변형의 단일 contiguous writeback_full 보다 비쌌고, coalescing 이득이 별도 buffer write 비용에 묻힘. (root 가 clist 에 없어 cb_pos=0 으로 OOB 나던 버그 + single→batch 시 cb buffer 재할당 버그 수정.)

**결론**: 검증된 1.2× ceiling 은 *상한*일 뿐, gather 는 **두 변형(scattered / coalesced-CB-buffer) 모두 정확하나 net-negative**(0.80–0.91×) — tiling·systems-per-block 과 같은 부류. memset/scatter 제거 이득(17%)을 gather 메커니즘이 어떤 형태로든 되갚는다.

### ✗✗ 제3변형 — output-centric, atomic-free gather (2026-06-13) — 정확하나 더 나쁨

"gather 를 비효율적으로(atomic scatter) 구현해서 실패한 것" 이라는 비판에 대응해 **제대로 된** gather 를 구현: **output-centric, atomic-free, 레이아웃 정합, 완전한 메타데이터 + A/B 프레임워크**(`CLS_ASM_MODE = scatter(default) | gather(atomic) | gather_oc`).
- **메타데이터**: analyze 에서 per-front "output-centric assembly CSR"(`d_gasm_off/pos/src_off/src`) — 각 front 의 occupied position(front 내 정렬)별로 거기에 합산될 기여(matrix slot 또는 child-CB offset) 목록. 커널은 **위치당 1 thread → register 합산 → atomic 없는 단일 write**. CB 는 parent-grouped 버퍼(coalesced).
- **정확성**: ✓ (single-system relres 가 scatter 와 일치, case30~USA).
- **성능**: ✗ **0.77–0.88×** (scatter 대비), atomic gather 와 거의 동일 — **atomic 이 병목이 아니었음**.
- **결정적 원인 (ncu + nsys)**: gather_oc 의 `factor_mid` = **53.7ms vs scatter 30.5ms (+76%)**. ncu: long_scoreboard 40, DRAM 4–12%(=bandwidth 아닌 **latency** 바운드). 근본: **output-centric index 자체가 데이터만큼 큼** — gasm 총 4.7M entry(25K) = **~25MB/system ≈ front_total(9MB) 의 ~2–3×**. element 단위 assembly map(child CB 마다 1 entry = Σuc²)을 system 마다 **흩어서 읽는** 비용이, legacy 의 "batched memset+scatter 로 dense 조립 → **coalesced** stage_in" 을 초과.
- **확정된 근본 진실**: legacy 의 *materialize-densely-then-coalesced-read* 가 *in-kernel scattered gather* 를 이긴다. scattered read ≫ coalesced read 이고, child→parent CB 전송량(Σuc²≈fsz²)은 어느 방식이든 동일하므로, 조립을 **per-front 커널 안으로** 옮기면(atomic이든 index든) 항상 손해. **세 변형 모두 net-negative 로 수렴 → gather 방향 종결.**

세 변형 모두 `CLS_ASM_MODE`(default scatter)로 보존 — 정확하고 A/B 재현 가능. default(scatter) 솔버 불변·검증.

## 5. 미회수 경로 (큰 재설계, 미착수)

- **fused assembly+factorize**: 현재 assemble(global write)→mid(global load) 의 front 왕복(2 pass)을 제거 → AI 를 2→~4 로. 가장 큰 traffic 절감이나 데이터플로 전면 재구성.
- **big 커널 tiled-shared trailing**: bandwidth-bound big(DRAM 40%)의 global 재읽기를 shared 타일로 축소.
- **mixed-precision(Ozaki 3×TF32) shared-residency**: shared 를 half 로 → 2× block/SM → latency 은닉(정확도 보존). 별도 과제.

## 적용된 변경
- `front_ops.cuh`: `trailing_update_rb` + `trailing_update` 디스패치 (CLS_TRAIL_RB, default 4 = +4%, 정확).
- `mid.cuh`: RB 배선 + `factor_mid_blocked_sysblk` 프로토타입(CLS_MID_SYSBLK, default 0=off, 회귀로 미채택).

## 재현
```bash
BIN=build/custom_linear_solver_run; C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
CLS_TRAIL_RB=0 $BIN $C --precision fp32 --batch 64 --repeat 15 --warmup 5 --serial-nd --metis-seed 7  # 0.097 (scalar)
CLS_TRAIL_RB=4 $BIN $C --precision fp32 --batch 64 --repeat 15 --warmup 5 --serial-nd --metis-seed 7  # 0.0936 (+4%)
$BIN $C --precision tf32 --batch 64 --repeat 15 --warmup 5 --serial-nd --metis-seed 7                 # 0.0843 (1.15x, relres↑)
# ncu AI/stall: build-prof(graph off), factor_mid_blocked<float>, long_scoreboard/dram/fma
```
