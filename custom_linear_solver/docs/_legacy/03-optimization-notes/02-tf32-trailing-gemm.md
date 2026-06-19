# TF32/FP16 PTX Trailing GEMM — V9h 스택과 영구 교훈

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: trailing GEMM 의 TF32(V0 WMMA → V9h PTX) / FP16(WMMA → PTX) 변형 탐색과 채택·폐기·영구 교훈.

대상 커널: `trailing_update_*` (factor_mid / factor_big 의 Phase 3 trailing GEMM `C ← C − L·U`).
환경: RTX 3090 (sm_86), CUDA 12.8, panel_cap=8. 케이스: case8387pegase / case_SyntheticUSA / case_ACTIVSg25k / case_ACTIVSg10k.
원본 실험 로그: 구 노트 15·22 (현재는 본 문서로 통합·압축).

## 1. 현재 상태 — 무엇이 빌드 가능한가

TF32 trailing GEMM 탐색은 V1–V9h 의 10개 변형을 거쳤으나, **현재 재현 가능한 것은 V0 와 V9h 두 가지뿐**이다. 실패한 변형의 build flag 는 코드에서 삭제됐다 (§4).

| 모드 | CLI / flag | 의미 |
|---|---|---|
| **V0 (default)** | `--precision tf32_wmma` | WMMA m16n16k8 TF32 + Csc fragment readback. safe baseline. |
| **V9h (opt-in/default)** | `--precision tf32` | TF32 PTX 스택 (아래 flag stack). regression 0, USA/ACTIVSg25k 게인. |

> docs/15 시점에는 V0 가 default 였고 V9h 는 opt-in. 이후 (docs/22 대칭화) `--precision tf32 = V9h PTX`, `--precision tf32_wmma = V0 legacy` 로 정착. V0 vs V9h 의 **WMMA m16n16k8 ↔ PTX 스택** 구분은 유지된다.

### V9h flag stack

```
-DCLS_TF32_BIG_PTX=1 \
-DCLS_TF32_MMA_AREUSE=1 \
-DCLS_TF32_SKIP_CONVERT=1 \
-DCLS_TF32_MID_PTX=1 \
-DCLS_TF32_MID_K4_HYBRID=1 \
-DCLS_TF32_BIG_LB_512_2=1
```

| flag | 역할 | 출처 변형 |
|---|---|---|
| `CLS_TF32_BIG_PTX` | big tier 를 PTX `mma.m16n8k8` 로, Csc smem round-trip 제거 | V2 |
| `CLS_TF32_MMA_AREUSE` | loop reorder (ti, kc, tj8) — A 레지스터 hoist 로 redundant load 제거 | V6 |
| `CLS_TF32_SKIP_CONVERT` | stage-in 의 명시적 `__float_to_tf32` 제거 (mma 가 implicit truncate) | V7a |
| `CLS_TF32_MID_PTX` | mid tier 의 k8 side path 를 PTX 로 (V0 wmma 와 다름) | V9 base |
| `CLS_TF32_MID_K4_HYBRID` | mid tier 의 per-level k4/k8 동적 선택 (§3) | V9h 핵심 |
| `CLS_TF32_BIG_LB_512_2` | big kernel `__launch_bounds__(512,2)` | (occupancy) |

### V9h 측정 게인 (10-trial median, --repeat 50, V0 대비)

| case | B | V0 wall | V9h wall | delta |
|---|---|---|---|---|
| case8387 | 1 | 0.451 | 0.448 | −0.7% (noise) |
| case8387 | 64 | 0.0275 | 0.0274 | −0.6% (noise) |
| **USA** | **1** | 2.297 | **2.068** | **−10.0%** |
| USA | 64 | 0.492 | 0.479 | **−2.7%** |
| **ACTIVSg25k** | **1** | 0.800 | **0.759** | **−5.0%** |
| ACTIVSg25k | 64 | 0.114 | 0.109 | **−4.1%** |

**V9h 는 모든 case × B 조합에서 V0 를 dominate or tie** — 단일 winner, regression case 0. USA −10.0% / ACTIVSg25k −5.0% 가 시리즈 최대 게인. case8387 은 big tier 미경유 + mid 가 k8 path 로 라우팅돼 무영향. default 유지 판단의 근거: 게인이 noise floor (§6) 와 자릿수가 비슷한 영역도 있으나 regression 이 없어 power-grid 류에 안전.

## 2. 채택 경로 요약 (V0 → V9h)

10개 변형 중 V9h 스택을 구성하는 성공 변형만 압축한다 (실패 변형은 §4).

| 변형 | flag | 변화 | 측정 |
|---|---|---|---|
| **V0** | (default) | WMMA m16n16k8 = HW mma.m16n8k8 × 2, Csc smem scratch readback | baseline |
| **V2** | `CLS_TF32_BIG_PTX` | big tier 만 PTX m16n8k8 직접 호출, Csc 우회 (각 lane 이 owned element 를 F 에 직접 subtract) | USA B=1 −7.1% wall; factor_big_tf32 kernel **−14.4%**; case8387 회귀 없음 (big 미경유) |
| **V6** | `+CLS_TF32_MMA_AREUSE` | loop inversion (ti, kc, tj8) — A(Ltf row strip) 가 tj8 무관 → register hoist 로 ntj8× redundant load 제거 | USA B=1 −8.9% (V2 위 −3.7%p); reg 48→64, occupancy 66.2% 불변 |
| **V7a** | `+CLS_TF32_SKIP_CONVERT` | stage-in 의 `wmma::__float_to_tf32` 제거 | USA B=1 **−12.6%** (V6 위 −3.7%p, 시리즈 최대 단일 누적); 정확도 TF32 noise 안 |
| **V9 (MK4)** | `CLS_TF32_MID_K4` (삭제됨) | mid tier 에 PTX m16n8k4 (K=4) always-k4 | ACTIVSg25k −3.3%, USA −8.2%; **case8387 mid kernel +27%** (nc=8 dominant) → 폐기 |
| **V9h** | `CLS_TF32_MID_K4_HYBRID` | mid 의 k4/k8 **per-level 동적 선택** (§3) | §1 표 — 모든 case dominate, V9 의 case8387 약점 해소 |

각 PTX 전환 게인의 출처:
- **V2 (smem 우회)**: big front 는 uc ≥ 16 이라 per-lane bound-check divergence 가 작아 Csc round-trip 제거 게인이 그대로 실현. mid 의 small front 는 V0 의 vectorizable per-lane loop 가 더 효율적이라 V1(all-tier) 에서 case8387 mid +10% 회귀가 났던 부분 (→ V2 가 tier 분리로 회피).
- **V6 (A-reuse)**: 데이터 의존성 분석으로 A 가 tj8-loop 무관임을 인식, ntj8(≈4–8)× 의 중복 로드 제거. register +16 이 occupancy binding 을 깨지 않음.
- **V7a (conv skip)**: mma `.tf32` ABI 의 implicit truncate 인식. 정확도 영향 없는 free win.

panel_cap sweep (V8, `--panel-cap N`, lib 변경 없음): case 별 optimal 다름 (case8387 cap=12 → −5%, USA cap=4 → −5.6%) 이나 **default 8 이 합리적 절충**, 일반 권장 안 함. B=64 는 모든 case 1% 이내 noise.

## 3. K4_HYBRID — per-level k4/k8 heuristic

V9h 의 핵심. mid tier 의 typical nc(panel 폭) 는 1–20. m16n8k8 은 `KP = round_up(nc, 8)` 로 K-dim 패딩이 낭비됨. m16n8k4 (`mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`) 는 `KP = round_up(nc, 4)` 로 패딩을 줄이나, K=4 라 K-loop iter 가 2배 — 패딩 차이가 0 인 경우 (nc=8, 16) issue overhead 만 증가해 손해.

dispatch 단계의 단순 heuristic:

```
use_k4 := (KP_k4 < KP_k8)      # ↔ level_max_nc % 8 ∈ {1, 2, 3, 4}
```

| level_max_nc | KP_k4 | KP_k8 | use_k4 | 효과 |
|---|---|---|---|---|
| 1, 2, 3, 4 | 4 | 8 | true | 50% 절약 |
| 5, 6, 7, 8 | 8 | 8 | false | 같음 → k8 (issue overhead 회피) |
| 9–12 | 12 | 16 | true | 25% 절약 |
| 13–16 | 16 | 16 | false | k8 |
| 17–20 | 20 | 24 | true | 17% 절약 |
| 21–24 | 24 | 24 | false | k8 |
| 25–28 | 28 | 32 | true | 12.5% 절약 |
| 29–32 | 32 | 32 | false | k8 |

mid tier nc 분포 (실측, CLS_DUMP_FRONTS) 와 use_k4 경향:

| case | mid front | nc 분포 | use_k4 경향 | 패딩 절약 가중평균 |
|---|---|---|---|---|
| case8387 | 66 | **nc=8 이 77%** | 대다수 false (k4 회피) | ≈7% |
| USA | 886 | nc=20 36%, nc=12 8%, nc=10 8%, … | ≈50% | ≈14% |
| ACTIVSg25k | 377 | **nc=12 가 59%** | 대다수 true | ≈22% |

→ ACTIVSg25k 가 가장 유리, case8387 가 가장 불리. always-k4 (V9) 는 case8387 의 nc=8 fronts 에 k4 의 2× issue overhead 만 얹어 mid_tf32 kernel **+27%** (nsys) 회귀를 냈고, hybrid 가 이 fronts 를 k8 path 로 라우팅해 해소했다. nsys per-kernel (factor_mid_tf32, MK4 vs V0): USA B=64 **−11.5%**, ACTIVSg25k B=64 **−11.3%**, case8387 B=64 **+27%**.

dispatch 구현 골자:

```cpp
#if defined(CLS_TF32_MID_K4_HYBRID)
    const int kp_k4 = round_up_int(level_nc_clipped, 4);
    const int kp_k8 = round_up_int(level_nc_clipped, 8);
    const bool use_k4 = (kp_k4 < kp_k8);
    const int kp_max = use_k4 ? kp_k4 : kp_k8;
#endif
if (use_k4) factor_mid_tf32_k4<<<grid, mblk, shb, stream>>>(...);
else        factor_mid_tf32   <<<grid, mblk, shb, stream>>>(...);
```

`factor_mid_tf32_k4` 는 always-k4 wrapper kernel (별도 `__global__`), smem 레이아웃은 factor_mid_tf32 와 동일 (unused Csc gap 도 reserve — dispatcher 의 shb 와 일치).

## 4. 삭제된 실패 매크로 (재현 불가, git history 참조)

다음 변형들은 회귀로 채택되지 못했고 build flag 가 코드에서 삭제됐다.

| 변형 / 매크로 | 시도 | 회귀 / 실패 원인 |
|---|---|---|
| **V1** `CLS_TF32_PTX_VARIANT` | all-tier PTX (mid 포함) | **case8387 mid +10% 회귀** — PTX 의 per-lane nested bound-check 가 V0 의 batched stride-32 per-lane loop 보다 branch divergence 큼. 작은 uc front 多인 case8387 mid 에서 두드러짐. (big-only 인 V2 로 분리하며 폐기) |
| **V3** `LB_256_4` (Csc dispatch skip) | big launch 의 Csc 영역 (32 KB) skip 으로 occupancy 확보 시도 | warps_active 66.24%→66.28% (**<0.1% 변화**). sm_86 의 binding 이 smem 이 아니라 **thread cap (1536/SM)**. bigT=1024 라 1 block/SM 고정 |
| **V4** `BIG_T_512` | bigT 1024→512 로 thread cap 해소 시도 | **USA B=1 +4.1% 회귀**. occupancy 오히려 ↓ (66.24%→65.59%) — register cap (64K/SM) 이 새 binding, 3 block 불가·fractional balance 로 평균 occupancy 하락 |
| **V9** `CLS_TF32_MID_K4` (always-k4) | mid 전체에 k4 | **case8387 mid_tf32 kernel +27%** (nsys) — nc=8 (77%) 에 k4 의 2× issue overhead. V9h hybrid 가 superseded |
| **SMALL_WARPS_16** `CLS_SMALL_WARPS_16` | (docs/18 EXP-D) small tier warp 수 변경 | 게인 없음 |

공통 실패 패턴: occupancy 의 단일 metric (smem 사용량 또는 block size) 만 보고 결정 — multi-constraint 시스템에서는 가장 tight 한 constraint 가 dominate (§5.4). all-tier PTX 는 mid 의 small-front divergence 를 무시.

## 5. 영구 학습 (코드/문서를 떠나 잔존하는 5가지)

### 5.1 m16n8k8 TF32 A 레지스터 매핑은 PTX 문서 외삽과 다름

PTX ISA 의 일반 `.row` 패턴을 외삽하면 inner stride 가 K(column) 일 것 같지만, **실제 inner stride 는 M block (row+8)**. self-probe (A 한 원소만 1 세팅) 로 역추론한 검증 매핑:

```
a0 = A[laneR + 0, laneC + 0]    (M_top, K_even)
a1 = A[laneR + 8, laneC + 0]    (M_bot, K_even)   <-- swap (외삽은 K_odd 로 예측)
a2 = A[laneR + 0, laneC + 1]    (M_top, K_odd)
a3 = A[laneR + 8, laneC + 1]    (M_bot, K_odd)
```

외삽 매핑으로 빌드하면 정확도 완전 붕괴 (case8387 batch_relres 0.02 → 58.7). cutlass `SM80_16x8x8_F32TF32TF32F32_TN` 와 일치. **PTX MMA 직접 호출 전 self-probe 필수.**

### 5.2 inline asm `"+f"(arr[var])` 의 register-binding gotcha

CUDA inline asm 의 register operand 는 **compile-time 알려진 register slot 에 binding** 돼야 한다. runtime index `c[tj8][i]` 는 compiler 가 unroll 못 해 local memory dynamic 주소로 바뀌고 operand binding 이 깨져 SEGFAULT (V6 첫 구현에서 USA/ACTIVSg25k 만 터지고 case8387 통과). 강제 unroll 패턴:

```cpp
#pragma unroll
for (int i = 0; i < MAX_CONST; ++i) {  // 상수 상한
    if (i >= runtime_n) break;          // early break
    asm("..." : "+f"(arr[i]));          // i 가 compile-time 상수 → register slot 매핑
}
```

### 5.3 명시적 `__float_to_tf32` 는 redundant

mma `.tf32` ABI 의 입력은 `.b32` 이고 **HW 가 low 13 bit 를 자동 무시**. stage-in 의 `wmma::__float_to_tf32` 는 round-to-nearest 마스킹만 하는데 어차피 mma 가 truncate-toward-zero 로 다시 자른다 (두 round mode 차이는 ULP 안). 제거 시 element 당 bitwise op 1회 절약, 정확도 영향 없음 (V7a). USA B=1 −12.6% 누적의 일부.

### 5.4 m16n8k4 의 A 매핑은 단순 (각 lane 이 1 K col 만 보유)

§5.1 학습 후 m16n8k4 는 미리 self-probe. **모든 K 가 활성화** (m16n8k8 의 K_even/K_odd split 없음), K stride 가 단순 1. 검증 매핑:

```
a0 = A[laneR + 0, laneC]   (M_top, K = laneC)     2 .b32/lane
a1 = A[laneR + 8, laneC]   (M_bot, K = laneC)
b0 = B[laneC, laneR]       (K = laneC, N = laneR)  1 .b32/lane
c0..c3 = D[laneR + {0,8}, laneC*2 + {0,1}]         4 .f32/lane
```

각 lane 이 1 K col + 2 N col 담당 (m16n8k8 은 2 K + 1 N). probe 한 번에 성공.

(보조: sm_86 occupancy 의 binding 후보 — smem/block, registers/block, threads/block, threads/SM=1536, blocks/SM=16 모두 동시 만족 필요. binding 추적 시 ncu `launch__registers_per_thread` + `sm__warps_active.avg.pct...` 함께 봐야. V3/V4 가 이 multi-constraint 를 놓침.)

### 5.5 새 `__global__` kernel 의 dynamic smem cap 등록 필수 (graph-capture segfault 방지)

sm_86 의 per-block dynamic shared memory default = **48 KB**. 그 이상을 launch shb 로 요청하려면 setup 에서:

```cpp
cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
```

누락 시 launch 가 `cudaErrorInvalidValue` 로 fail. **internal graph capture mode 에서는 capture invalidate (`cudaErrorStreamCaptureInvalidated`) → 다음 launch 도 실패 → segfault**; nograph mode 에서는 garbage (`batch_relres=9.4e+27`) 로 표면화 (case 별 다른 증상 → debug 헷갈림). V9h 의 `factor_mid_tf32_k4` 첫 빌드에서 case8387 (use_k4=false 라 미호출) 만 통과하고 USA/ACTIVSg25k 만 터진 것이 이 누락. 기존 kernel 등록 옆에 추가하는 것을 패턴화.

## 6. measurement noise floor

power-grid solve 의 multi-stream + atomicAdd 비결정성으로 V0 vs V0 (동일 코드) 도 wall 차이 1.3–2.5% 발생. **wall 게인 < 3% 는 보고하기 어렵다**. nsys per-kernel 도 ±10% jitter. §1 의 case8387 ±1% 는 모두 이 noise 안.

## 7. FP16 PTX default 전환 (docs/22)

TF32 의 PTX 디폴트화와 대칭으로, FP16 trailing GEMM 도 WMMA → PTX 로 전환. WMMA 경로는 `fp16_wmma` 로 비교용 보존.

| CLI | enum | 의미 |
|---|---|---|
| `--precision fp16` | `Precision::FP16` | **FP16 PTX `mma.m16n8k16`, FP32 accumulate, no Csc (default)** |
| `--precision fp16_wmma` | `Precision::FP16_WMMA` | FP16 WMMA m16n16k16 + Csc readback (legacy) |
| `--precision tf32` | `Precision::TF32` | TF32 PTX V9h (default) |
| `--precision tf32_wmma` | `Precision::TF32_WMMA` | TF32 WMMA V0 (legacy) |

추가 device code (`src/factorize/phases.cuh`):
- `pack_half2_bits(__half lo, __half hi) → unsigned`
- `trailing_update_mma_fp16_ptx(...)` — inline PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, A-reuse loop (NTJ8_MAX=8 N-tile chunks), 4 per-lane FP32 accumulator 를 `F` 에 직접 빼기 (Csc 없음).

kernel (`src/factorize/kernels.cuh`):
- `factor_mid_fp16_ptx(...)` — mid tier, 256-thread 블록, KP=round_up(nc, 16).
- `factor_big_fp16_ptx(...)` — `__launch_bounds__(512, 2)`, tf32 PTX big 과 동일 launch shape.

dispatch: `Precision::FP16` → `*_fp16_ptx`, `Precision::FP16_WMMA` → 기존 `factor_{mid,big}_tc` (legacy). 10pct 의 `fp16_b1_scalar_mid` / `fp16_scalar_spine` 분기는 두 FP16 모드 모두에 적용. `src/multifrontal.cu` 에서 `factor_mid_fp16_ptx` / `factor_big_fp16_ptx` 에 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 99 KB)` 등록 (§5.5 학습).

### 측정 — ACTIVSg10k B=64 (median of 3 × 100 rep, per-sys ms)

| Mode | per-sys ms |
|---|---|
| `fp16` (PTX default) | **0.0321** |
| `fp16_wmma` (legacy) | 0.0348 |

→ **PTX 가 fp16_wmma legacy 대비 −7.7%**. B=1 단일 케이스도 동급 또는 약간 빠름. residual 양쪽 모두 ~1e-4 FP16 정밀도 수준.

원본 stash (codex) 에는 `Precision::TC` 하이브리드 (mid TF32 PTX + big FP16 PTX) 도 있었으나 사용자 결정으로 **FP16_PTX 만 수령, TC 모드 폐기**.

## 8. 후속 가능성

1. **k4 path 에 V6 A-reuse hoist 적용** — K-loop iter 2배라 register pressure 비대칭이나, A-load 가 잦은 만큼 hoist 효과도 클 가능성.
2. **N 확장 (m16n16k8 / m16n32k8)** — TF32 mma 의 K 는 4,8 만 (사양). N 확장으로 dispatch ntj16/ntj8 loop 단순화 여지.
3. **mid k8 PTX 의 bound-check divergence 완화** — uc 가 16 배수 아닌 마지막 strip 만 V0 fallback. 복잡도 ↑ 대비 게인 불확실.

연관 문서: [03-tensor-core-investigation.md](03-tensor-core-investigation.md), [01-kernel-engineering.md](01-kernel-engineering.md), [01-kernel-engineering.md](01-kernel-engineering.md).
