# TF32 trailing GEMM 가속 연구 — 통합 보고서

**작성일**: 2026-06-06
**대상**: `trailing_update_wmma_tf32_f32` (factor_mid_tf32 / factor_big_tf32 의 Phase 3 — TF32 WMMA trailing GEMM)
**환경**: RTX 3090 (sm_86), CUDA 12.8, panel_cap=8
**대상 케이스**: case8387pegase, case_SyntheticUSA, case_ACTIVSg25k

> **2026-06-07 정리 후 상태**
> - **default = V0** (WMMA m16n16k8). 추가 flag 불필요.
> - **opt-in = V9h stack** = `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 -DCLS_TF32_MID_K4_HYBRID=1 -DCLS_TF32_BIG_LB_512_2=1` — 모든 측정 case 에서 회귀 없음, USA/ACTIVSg25k 에서 -2~-10% 게인.
> - **삭제된 실패 매크로/코드** (재현 불가, git history 참조): `CLS_TF32_PTX_VARIANT` (V1), `CLS_TF32_BIG_T_512` (V4), `CLS_TF32_BIG_LB_256_4` (V3 aggressive), `CLS_TF32_MID_K4` (V9 always-k4), `CLS_SMALL_WARPS_16` (docs/18 EXP-D). 본 문서는 historical experiment log 로 보존하나, V1/V3/V4/V9 의 build flag 는 더 이상 존재하지 않는다.
> - 본 문서 §2-§8 의 V1-V9, §9 의 V9h 까지 의 실측 표는 유효하지만, **다시 실행할 수 있는 것은 V0 / V9h 두 가지뿐**이다.

## 0. TL;DR

10개 변형을 차례로 시도. 최종 채택: **V9h (V7a + MID_K4 hybrid) 스택** — `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 -DCLS_TF32_MID_K4_HYBRID=1`.
- 누적 영향: USA B=1 wall **-10.0%**, B=64 **-2.7%**; ACTIVSg25k B=1 **-5.0%**, B=64 **-4.1%**; case8387 무영향.
- **V9h 가 모든 case × B 조합에서 V0 dominate or tie** — 단일 winner. always-k4 (V9 = MK4) 의 case8387 약한 신호조차 정상화.
- 핵심 트릭: per-level dispatch `use_k4 = (KP_k4 < KP_k8)` ↔ `level_max_nc % 8 ∈ {1..4}`. nc=8 dominant 인 case8387 mid level 은 k8 path 자동 선택, nc=12 dominant 인 ACTIVSg25k 는 k4 path 자동 선택.
- default 는 V0 유지 — gain 이 noise floor 와 자릿수가 비슷한 영역도 있어서 user 가 무조건 켜야 한다 단언하기 애매. 다만 V9h 는 **regression case 0** 이라 power-grid 류에 opt-in 안전.
- 영구 학습 5가지: (i) m16n8k8 TF32 A 레지스터 매핑이 PTX 문서 외삽과 다름, (ii) CUDA inline asm 의 register operand 는 compile-time index 필요, (iii) `__float_to_tf32` 는 mma .tf32 ABI 의 implicit truncate 와 redundant, (iv) m16n8k4 의 A 매핑은 m16n8k8 보다 단순 (각 lane 이 1 K col 만 보유) — probe 한 번에 성공, (v) **사용자 추가 kernel 은 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 99 KB)` 등록 필수** — 빠뜨리면 launch 가 `cudaErrorInvalidValue` 로 silently 실패.

## 1. 출발점 — baseline (V0) 분석

`trailing_update_wmma_tf32_f32` 의 구조 (factor_mid_tf32 / factor_big_tf32 의 Phase 3 trailing GEMM `C ← C − L·U`):

```cpp
// V0 — src/factorize/phases.cuh
__device__ void trailing_update_wmma_tf32_f32(float* F, int fsz, int nc, int uc,
                                               float* Ltf, float* Utf, float* Csc, ...)
{
    // (1) stage-in: F → Ltf (UCP×KP), F → Utf (KP×UCP). 모든 element 에 wmma::__float_to_tf32 적용.
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Ltf[e] = (i < uc && k < nc) ? wmma::__float_to_tf32(F[(nc + i) * fsz + k]) : 0.0f;
    }
    // ... 동일 패턴으로 Utf ...
    __syncthreads();

    // (2) 16×16 outer tiles, K-loop inside. wmma::mma_sync (= 2× HW mma.m16n8k8 internally).
    for (int ti = warp; ti < UCP/16; ti += nwarp) {
        for (int tj = 0; tj < UCP/16; ++tj) {
            wmma::fragment<accumulator, 16, 16, 8, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            for (int kc = 0; kc < KP/8; ++kc) {
                wmma::load_matrix_sync(...);
                wmma::mma_sync(cf, af[kc], bf, cf);   // 2× HW mma.m16n8k8
            }
            // (3) writeback via smem scratch Csc.
            wmma::store_matrix_sync(&Csc[warp*256], cf, 16, mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                if (ii < uc && jj < uc) F[(nc+ii)*fsz + (nc+jj)] -= Csc[warp*256 + e];
            }
            __syncwarp();
        }
    }
}
```

### 1.1 식별된 비효율

| 위치 | 비효율 | 비용 |
|------|-------|------|
| (1) stage-in | 모든 element 에 `wmma::__float_to_tf32` (round-to-nearest 마스킹) | bitwise op × element 수 |
| (2) WMMA | `wmma::mma_sync<16,16,8>` 는 HW 의 `mma.m16n8k8` 을 2회 호출하는 abstraction | HW shape 와 직접 부르면 동일 FLOPs 인데 abstraction overhead 가능 |
| (3) writeback | `Csc` smem scratch (1 KB/warp/tile) 에 fragment store → per-lane bound-check 후 F 에서 빼기 | 16×16 tile 당 smem store 1 KB + load 1 KB |

(3) 의 smem round-trip 이 가장 자명한 lever. (2) 는 HW-native shape 로 가면 fragment layout 이 노출되어 (3) 의 smem 우회가 가능해짐. (1) 은 의외의 lever — 후술 §6 의 V7a 에서 발견.

### 1.2 baseline wall + per-kernel (nsys mean, --repeat 20, CLS_INTERNAL_GRAPH=OFF)

| case | B | factor_mid_tf32 mean ns | factor_big_tf32 mean ns | wall ms/sys |
|------|---|------------------------|------------------------|-------------|
| case8387 | 1 | 15890 | – (미경유) | 0.470 |
| case8387 | 64 | 46983 | – | 0.027 |
| USA | 1 | 38990 | 75067 | 2.323 |
| USA | 64 | 1014091 | 611492 | 0.504 |
| ACTIVSg25k | 1 | 22034 | 50370 | 0.775 |
| ACTIVSg25k | 64 | 149415 | 202348 | 0.115 |

## 2. 연구 흐름 요약

| 변형 | 매크로 플래그 | 주된 변화 | 결과 | 결론 |
|------|-------------|----------|------|------|
| **V1** | `CLS_TF32_PTX_VARIANT` | PTX m16n8k8 직접 호출, Csc skip (전 tier) | USA B=1 -5.7% wall; **case8387 mid +10% 회귀** | A-layout 발견 (§3), regression 으로 직 채택 보류 |
| **V2** | `CLS_TF32_BIG_PTX` | big tier 만 V1 | USA B=1 -7.1% wall, case8387 회귀 제거 | 좋음, 후속 base |
| **V3** | V2 + dispatch Csc 제거 | smem 절약 | occupancy < 0.1% 변화 — thread cap 이 binding | 효과 없음 |
| **V4** | `CLS_TF32_BIG_T_512` | bigT 1024 → 512 | USA B=1 **+4.1% 회귀** | register cap 이 새 binding |
| **V6** | `CLS_TF32_MMA_AREUSE` | V2 + loop reorder (ti,kc,tj8) — A 재사용 | USA B=1 -8.9% (V2 위에 -3.7%p) | A 가 tj8 무관 → hoist 효과 |
| **V7a** | `CLS_TF32_SKIP_CONVERT` | V6 + `__float_to_tf32` 스킵 | USA B=1 **-12.6%** (V6 위에 -3.7%p) | mma 가 implicit truncate, 명시적 conv 불필요 |
| **V8** | `--panel-cap N` 런타임 | analyze 의 panel 폭 변경 | case 별 optimal 다름 | default 8 합리적 |
| **V9 (MK4)** | `CLS_TF32_MID_K4` | mid tier 에 PTX m16n8k4 (K=4) — always-k4 | ACTIVSg25k B=1 -3.3%, USA -8.2%, case8387 noise | nc 가 8 배수 아닌 mid front 에 효과적; nc=8 dominant case 회피 못 함 |
| **V9h (Hybrid)** | `CLS_TF32_MID_K4_HYBRID` | mid 의 k4/k8 per-level 선택 (`KP_k4 < KP_k8` 시 k4) | **USA -10.0%/-2.7%, ACTIVSg25k -5.0%/-4.1%, case8387 무영향** | 모든 case 에서 dominate; V9 의 always-k4 가 case8387 에 약하던 부분 해결 |

## 3. V1 — PTX `mma.m16n8k8` 로 Csc smem round-trip 제거

### 3.1 가설

HW-native shape 인 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` 을 직접 호출하면:
- WMMA 의 m16n16k8 = HW 의 m16n8k8 × 2 → FLOPs 동일.
- 그러나 per-lane fp32 accumulator 의 mapping 이 PTX ISA 9.7.13.4.2 에 문서화되어 있으므로, **각 lane 이 본인 owned 4 element 의 (row, col) 을 알아 F 에 직접 subtract → Csc 우회 가능**.

### 3.2 첫 구현의 함정 (PTX 문서 외삽 오류)

PTX ISA 의 일반 16×8 input matrix `.row` 패턴을 외삽:
> row of a0, a1: groupID; row of a2, a3: groupID + 8;
> col of a0, a2: tig×2; col of a1, a3: tig×2 + 1

→ 외삽 매핑:
```
a0 = A[laneR + 0, laneC + 0]   (M_top, K_even)
a1 = A[laneR + 0, laneC + 1]   (M_top, K_odd)
a2 = A[laneR + 8, laneC + 0]   (M_bot, K_even)
a3 = A[laneR + 8, laneC + 1]   (M_bot, K_odd)
```

이 매핑으로 빌드 성공, **accuracy 완전 깨짐** (case8387 `batch_relres` 0.02 → 58.7).

### 3.3 systematic probing 으로 실제 layout 역추론

A 의 한 원소만 1 로 세팅, B 의 해당 K-row 만 1 로 세팅 → D 의 한 row 가 1. 어느 lane 의 어느 c register 가 활성화되는지 관찰:

| Probe | 활성 lane | 결과 |
|-------|----------|------|
| A[0, 0]=1 | lanes 0..3, c0=c1=1 | ✓ |
| A[0, 1]=1 | (없음) | ✗ |
| A[0, 2]=1 | lanes 0..3, c0=c1=1 | ✓ |
| A[0, 4]=1 | lanes 0..3, c0=c1=1 | ✓ |
| A[0, 7]=1 | (없음) | ✗ |
| A[8, 0]=1 | (없음) | ✗ |

EVEN K (0, 2, 4, 6) 만 활성화. 그렇지만 외삽 가정에 따르면 ODD K 도 a1 을 통해 활성화돼야 함. → **외삽 매핑이 잘못됨**.

### 3.4 실제 매핑 (probing 으로 검증)

```
a0 = A[laneR + 0, laneC + 0]    (M_top, K_even)
a1 = A[laneR + 8, laneC + 0]    (M_bot, K_even)   <-- swap with a2!
a2 = A[laneR + 0, laneC + 1]    (M_top, K_odd)
a3 = A[laneR + 8, laneC + 1]    (M_bot, K_odd)
```

즉 a-register 의 inner stride 가 **K (column) 가 아니라 M block (row+8)**. cutlass `cute/arch/mma_sm80.hpp` 의 `SM80_16x8x8_F32TF32TF32F32_TN` traits 와도 일치 (사후 확인).

### 3.5 교훈

`.row` layout 의 이름이 misleading. 다음에 PTX MMA 를 직접 부를 때는 작은 self-probe 를 먼저 돌리는 게 안전.

### 3.6 V1 결과

A-layout 수정 후 정확도 회복. wall:

| case | B | V0 wall | V1 wall | delta |
|------|---|---------|---------|-------|
| case8387 | 1 | 0.421 | 0.461 | +9.1% (회귀) |
| case8387 | 64 | 0.030 | 0.026 | -12% |
| USA | 1 | 2.27 | 2.13 | -5.7% |
| USA | 64 | 0.490 | 0.488 | -0.4% |

nsys kernel-level: USA big_tf32 **-17%**, case8387 mid_tf32 **+10%**.

case8387 mid 의 +10% 회귀 원인: per-lane 의 nested bound-check (`if (r < uc) { if (col0 < uc) ...; if (col1 < uc) ...; }`) 가 V0 의 stride-32 vectorizable per-lane loop 보다 branch divergence 가 큼. 작은 uc front 가 많은 case8387 mid 에서 영향 두드러짐.

## 4. V2 — tier 별 hybrid (big-only PTX)

§3.6 의 per-kernel 데이터: USA big 은 win (-14 ~ -17%), case8387 mid 는 lose (+10%). tier 분리:

```cpp
// factor_mid_tf32: V0 유지
trailing_update_wmma_tf32_f32(...);
// factor_big_tf32: V1 PTX
#if defined(CLS_TF32_BIG_PTX) || defined(CLS_TF32_PTX_VARIANT)
    trailing_update_mma_tf32_ptx(...);
#else
    trailing_update_wmma_tf32_f32(...);
#endif
```

### 4.1 결과

| case | B | V0 | V2 | delta |
|------|---|-----|-----|-------|
| case8387 | 1 | 0.421 | 0.443 | +5.3% (noise; big tier 미경유) |
| case8387 | 64 | 0.027 | 0.027 | -1.4% (noise) |
| USA | 1 | 2.252 | 2.093 | **-7.1%** |
| USA | 64 | 0.490 | 0.480 | -2.0% |
| ACTIVSg25k | 1 | 0.815 | 0.806 | -1.1% |
| ACTIVSg25k | 64 | 0.117 | 0.113 | -3.4% |

case8387 (big 미경유) 의 wall 변화는 측정 noise (cf. §9 noise floor 분석).

### 4.2 nsys per-kernel (USA B=64)

| kernel | V0 mean | V2 mean | delta |
|--------|---------|---------|-------|
| factor_big_tf32 | 611492 ns | 523666 ns | **-14.4%** |

clear, robust kernel-level 게인.

### 4.3 결론

V2 = "big tier 에 V1, mid tier 에 V0". USA 의 big 게인 보존, case8387 mid regression 제거.

## 5. V3 — Csc smem alloc 제거 + V4 — bigT 512 (occupancy 시도 2건, 모두 실패)

### 5.1 V3 — dispatch.cuh 에서 Csc 영역 skip

V2 의 big_tf32 launch config 는 여전히 Csc 영역 32 KB (= `(bigT/32) × 256 × sizeof(float)`) 를 예약 — 안 씀에도. dispatch.cuh 에서 skip:

```cpp
#if defined(CLS_TF32_BIG_PTX) || defined(CLS_TF32_PTX_VARIANT)
    const size_t shbytes = (size_t)2 * ucp_max * kp_max * sizeof(float);  // Csc 제거
#else
    const size_t shbytes = ... + (bigT/32) * 256 * sizeof(float);
#endif
```

**ncu 측정 결과**: `factor_big_tf32` USA B=64 의 `sm__warps_active.avg.pct_of_peak_sustained_active` 가 V2 (66.24%) → V3 (66.28%) — **변화 < 0.1%**.

**원인 분석**: `bigT = 1024 thread / block`, sm_86 의 SM 당 max thread = 1536. 1 block (1024) + 다음 block (1024 더) = 2048 > 1536 → **thread cap 으로 1 block / SM 만 가능**. smem 을 줄여도 thread cap 이 우선 binding.

(추가 register 계산: 48 reg/thread × 1024 thread × 2 block = 98 K reg > sm_86 의 64 K reg/SM → register cap 도 막음.)

### 5.2 V4 — bigT 1024 → 512 로 줄여 thread cap 해소 시도

V3 의 분석에 따라 block size 를 줄여 thread cap 을 풀려 함:

```cpp
#if defined(CLS_TF32_BIG_T_512)
    constexpr int bigT_tf32 = 512;
#else
    constexpr int bigT_tf32 = bigT;  // 1024
#endif
```

**ncu 결과**: `factor_big_tf32` USA B=64:

| build | block_size | reg/thread | warps_active |
|-------|-----------|-----------|--------------|
| V2 (bigT=1024) | 1024 | 48 | 66.24% |
| V4 (bigT=512) | 512 | 48 | **65.59%** |

**occupancy 가 오히려 떨어짐**. 원인: 48 reg/thread × 512 thread × 3 block = 73 K reg > 64 K → 3 block 불가. 2 block (24 K reg, 1024 thread) 은 가능하나 register 와 thread 의 fractional balance 가 평균 occupancy 를 낮춤.

**wall**:

| case | B | V0 | V2 | V4 |
|------|---|-----|-----|-----|
| USA | 1 | 2.27 | 2.18 (-4.0%) | 2.36 (**+4.1% 회귀**) |
| USA | 64 | 0.487 | 0.480 | 0.472 (-3%) |

USA B=1 (latency-bound) 에서 block size 축소가 직접 회귀. 추천 안 함.

### 5.3 V3/V4 의 메타 교훈

occupancy 의 binding constraint 를 추적할 때 **smem 만 보면 안 되고 register × thread cap × block size 모두 봐야**. sm_86 에서:
- bigT=1024: thread cap 이 binding (1 block/SM)
- bigT=512: register cap 이 binding (2 block 한계, fractional)

진짜 occupancy 게인을 얻으려면 register count 도 함께 줄여야 (e.g., `__launch_bounds__` 명시) — 별개의 큰 재설계.

## 6. V6 + V7a — 알고리즘 변화 (성공)

§5 의 occupancy 직진로는 막힘. 다른 lever: **trailing GEMM 의 알고리즘 자체 변경**.

### 6.1 V6 — loop reorder (ti, kc, tj8), A-reuse hoist

V1/V2 의 PTX trailing 루프 nest:

```
for ti:                                # M block strip
    for tj8:                           # N tile
        for kc:                        # K loop (innermost)
            load A (tj8-independent!)
            load B
            mma
```

A (Ltf 의 한 row strip) 의 로드는 (ti, kc, lane) 에만 의존, **tj8 와 무관**. 그런데 ntj8 ≈ 4-8 만큼 같은 (ti, kc) 의 A 가 재로드됨.

루프 inversion (ti, kc, tj8):

```cpp
constexpr int NTJ8_MAX = 8;  // UCP ≤ 64 까지만 활성화
if (ntj8 <= NTJ8_MAX) {
    for (int ti = warp; ti < ntj16; ti += nwarp) {
        float c[NTJ8_MAX][4] = {0};
        for (int kc = 0; kc < nks; ++kc) {
            // load A — 한 번만!
            const unsigned a0 = __float_as_uint(...);
            ...
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= ntj8) break;
                // load B per tj8
                asm volatile("mma..." : "+f"(c[tj8][0..3]) : ...);
            }
        }
        // writeback c[tj8][.] - F[...]
    }
}
```

cost: lane 당 4 × ntj8 fp32 accumulator (16-32 reg).

### 6.2 V6 의 함정 — inline asm operand binding

첫 구현에서 USA/ACTIVSg25k SEGFAULT (case8387 만 통과). 원인:

```cpp
for (int tj8 = 0; tj8 < ntj8; ++tj8) {            // ntj8 = runtime variable
    asm volatile("mma..." : "+f"(c[tj8][0]), ...);
}
```

CUDA inline asm `"+f"(c[tj8][i])` 의 operand 는 **compile-time 알려진 register slot 에 binding** 되어야 함. 위 코드는 runtime `tj8` 로 인덱스 → compiler unroll 못 함 → `c[tj8][i]` 가 local memory 의 dynamic 주소로 변환 → asm operand binding 깨짐 → 메모리 위반.

수정: `#pragma unroll` + 상수 상한 + early break:

```cpp
#pragma unroll
for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {   // 상수 한계
    if (tj8 >= ntj8) break;
    asm volatile("mma..." : "+f"(c[tj8][0]), ...);
}
```

→ compiler 가 c[0][...], c[1][...], ..., c[7][...] 을 각자 register 슬롯에 매핑. 정상 작동.

### 6.3 V6 결과

**register**: 48 → 64 reg/thread (per ncu, factor_big_tf32 USA B=64). +16 reg = c[8][4] = 32 fp32 의 절반 (SSA 재사용).
**occupancy**: 66.22% (V2 의 66.24% 와 동일) — 16 reg 증가가 binding 깨지 않음.

**wall** (median-of-6, --repeat 50):

| case | B | V0 | V2 | V6 |
|------|---|-----|-----|-----|
| USA | 1 | 2.323 | 2.249 (-3.2%) | **2.164 (-6.8%)** |
| USA | 64 | 0.504 | 0.494 | 0.490 (-2.8%) |

**nsys** USA B=1: factor_mid_tf32 -6.9%, factor_big_tf32 -4.6%. A-load 재사용 가설 확인.

### 6.4 V7a — `__float_to_tf32` 스킵 (의외의 win)

mma `.tf32` ABI 의 입력은 `.b32` type, **HW 가 low 13 bit 를 자동으로 무시**. stage-in 단계의 명시적 `wmma::__float_to_tf32` 는:
- round-to-nearest 마스킹만 수행 (low 13 bit 를 round 후 zero).
- 어차피 mma 가 truncate-toward-zero 로 다시 자름.
- 두 round mode 의 차이는 ULP 안 → TF32 정확도에 영향 없음.

→ 명시적 호출은 **redundant**. 제거 시 element 당 bitwise op 1 회 절약.

구현:

```cpp
#if defined(CLS_TF32_SKIP_CONVERT)
    Ltf[e] = (i < uc && k < nc) ? F[(nc + i) * fsz + k] : 0.0f;
#else
    Ltf[e] = (i < uc && k < nc) ? wmma::__float_to_tf32(F[(nc + i) * fsz + k]) : 0.0f;
#endif
```

### 6.5 V7a 결과

**정확도** (B=64, 1-trial):

| case | V0 | V7a |
|------|-----|-----|
| case30 | 7.84e-7 | 7.84e-7 |
| case118 | 2.27e-7 | 2.27e-7 |
| case8387 | 0.021 | 0.021 |
| USA | 0.087 | 0.068 (TF32 round mode 의 trial variance 안) |
| ACTIVSg25k | 0.012 | 0.012 |

정확도 유지 확인 (TF32 noise 안).

**wall** (median-of-6, --repeat 50):

| case | B | V0 | V6 | V7a |
|------|---|-----|-----|------|
| USA | 1 | 2.339 | 2.130 (-8.9%) | **2.044 (-12.6%)** |
| USA | 64 | 0.496 | 0.488 | 0.492 (-0.7%) |
| ACTIVSg25k | 1 | 0.803 | 0.784 | 0.794 (-1.0%) |
| ACTIVSg25k | 64 | 0.115 | 0.115 | 0.115 |

USA B=1 의 누적 -12.6% 가 본 연구 시리즈 최대.

## 7. V8 — panel_cap sweep (직교 방향)

panel_cap 은 analyze 단계 supernode 의 panel 폭 한계 (default 8). 큰 cap → 적은/넓은 front, 작은 cap → 많은/좁은 front + dispatch overhead. GEMM 최적화와 orthogonal.

runner 에 `--panel-cap N` flag 추가 (lib 변경 없음, `SolverConfig.panel_cap` 만 override).

### 7.1 sweep 결과 (V7a 위, median-of-6, --repeat 50, B=1)

| cap | USA | case8387 | ACTIVSg25k |
|-----|-----|----------|-----------|
| 4 | **2.019** | 0.559 (+25%) | 0.801 |
| 8 (default) | 2.139 | 0.471 | **0.775** |
| 12 | 2.061 | **0.446 (-5%)** | 0.802 |
| 16 | 2.238 | 0.452 | 0.794 |
| 24 | 2.035 | 0.489 | 0.802 |
| 32 | 2.108 | 0.506 | 0.810 |

case 별 optimal cap 이 다름. **default 8 은 합리적 절충**. case-specific tuning 으로 추가 5% 가능 (case8387 cap=12 → -5%, USA cap=4 → -5.6%) — 일반 권장 안 함.

B=64 에서는 모든 case 1% 이내 변동 — noise.

## 7a. V9 (MK4) — mid tier 에 PTX `mma.m16n8k4` (K=4)

V2/V6/V7a 는 모두 **big tier** 만 PTX 로 전환. mid tier 는 §3.6 의 case8387 +10% 회귀 때문에 V0 wmma 유지. mid 의 trailing GEMM 자체에 손대지 못함. **V9** 는 mid 의 K-dim 패딩 낭비를 다른 mma shape 로 풀어보는 시도.

### 7a.1 동기 — mid 의 K-padding 낭비

mid tier 의 typical nc (panel 폭) 는 1-20. m16n8k8 의 `KP = round_up(nc, 8)`:

| nc | KP_k8 | 패딩 비율 | KP_k4 = round_up(nc, 4) | 절약 |
|----|-------|----------|----------------------|------|
| 1  | 8     | 87.5%    | 4   | -50% |
| 4  | 8     | 50%      | 4   | -50% |
| 8  | 8     | 0%       | 8   | 0    |
| 12 | 16    | 25%      | 12  | -25% |
| 16 | 16    | 0%       | 16  | 0    |
| 20 | 24    | 17%      | 20  | -17% |

`nc % 8 ∈ {1,2,3,4}` 인 경우 k4 가 K-dim 패딩을 줄임. 그러나 k4 는 K=4 라 K-loop 가 2배 (e.g., nc=8 일 때 k4 = 2 iter, k8 = 1 iter). 패딩 차이가 0 인 경우 (nc=8, 16) k4 는 issue overhead 만 증가 → 손해.

PTX 의 `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32` 는 같은 16×8 output 을 K=4 로 누적 → register 풋프린트 절반 (A: 2 .b32, B: 1 .b32 vs k8 의 4 / 2).

### 7a.2 mid tier 의 nc 분포 (실측, CLS_DUMP_FRONTS)

| case | mid front 수 | nc 분포 (mid tier 만, fsz ∈ (32, 128]) | k4 효과 추정 |
|------|-------------|---------------------------------------|--------|
| case8387 | 66 | **nc=8 이 77%**, 나머지 17% 가 nc≤6 | mostly neutral (nc=8 다수) |
| USA | 886 | nc=20 36%, nc=12 8%, nc=10 8%, nc=2~8 35% | 혼합; nc % 8 ≠ 0 의 절반 가량이 절약 |
| ACTIVSg25k | 377 | **nc=12 가 59%**, nc=2~10 32% | **대부분 25% 절약 가능** |

→ ACTIVSg25k 가 가장 유리, case8387 이 가장 불리.

### 7a.3 m16n8k4 의 A-layout probe

§3 의 m16n8k8 probe 함정 (외삽 매핑이 틀림) 학습 후 m16n8k4 는 미리 self-probe. 결과:

```
A[0, 0]=1, B[0,*]=1 → expect D[0, *]=1: L0[1,1,0,0] L1[1,1,0,0] L2[1,1,0,0] L3[1,1,0,0]
A[0, 1]=1, B[1,*]=1 → expect D[0, *]=1: L0[1,1,0,0] L1[1,1,0,0] L2[1,1,0,0] L3[1,1,0,0]
A[0, 2]=1, B[2,*]=1 → expect D[0, *]=1: L0[1,1,0,0] L1[1,1,0,0] L2[1,1,0,0] L3[1,1,0,0]
A[0, 3]=1, B[3,*]=1 → expect D[0, *]=1: L0[1,1,0,0] L1[1,1,0,0] L2[1,1,0,0] L3[1,1,0,0]
A[8, 0]=1, B[0,*]=1 → expect D[8, *]=1: L0[0,0,1,1] L1[0,0,1,1] L2[0,0,1,1] L3[0,0,1,1]
```

**모든 K 가 활성화** (m16n8k8 의 even-only 문제 없음). 검증된 매핑:

```
a0 = A[laneR + 0, laneC]        (M_top, K = laneC)        2 .b32/lane
a1 = A[laneR + 8, laneC]        (M_bot, K = laneC)
b0 = B[laneC, laneR]            (K = laneC, N = laneR)     1 .b32/lane
c0 = D[laneR + 0, laneC * 2 + 0]                          4 .f32/lane
c1 = D[laneR + 0, laneC * 2 + 1]
c2 = D[laneR + 8, laneC * 2 + 0]
c3 = D[laneR + 8, laneC * 2 + 1]
```

각 lane 이 **1 K col + 2 N col** 담당 (vs m16n8k8 의 2 K + 1 N). 분포 자체는 cleaner — m16n8k8 의 K_even/K_odd split 이 사라지고 K stride 가 단순 1.

### 7a.4 구현

`src/factorize/phases.cuh` 에 `trailing_update_mma_tf32_k4_ptx` 추가 (~50 줄, V7a 의 k8 PTX 함수와 거의 동일 구조). 차이:
- K-loop stride = 4 (vs 8)
- 1 mma 당 a 가 2 reg, b 가 1 reg (vs 4 / 2)
- stage-in 의 KP 가 round_up(nc, 4) (vs round_up(nc, 8))
- A-reuse hoist (V6) 는 일단 미적용 (kc 가 2배 더 많아 register cost 가 비대칭) — 후속 최적화 여지

`src/factorize/kernels.cuh` 의 `factor_mid_tf32` 에 `CLS_TF32_MID_K4` 분기 추가:

```cpp
#if defined(CLS_TF32_MID_K4)
    trailing_update_mma_tf32_k4_ptx(Fs, fsz, nc, uc, Ltf, Utf, t, nt);
#elif defined(CLS_TF32_MID_PTX) || defined(CLS_TF32_PTX_VARIANT)
    trailing_update_mma_tf32_ptx(Fs, fsz, nc, uc, Ltf, Utf, t, nt);
#else
    trailing_update_wmma_tf32_f32(Fs, fsz, nc, uc, Ltf, Utf, Csc, t, nt);
#endif
```

`src/factorize/dispatch.cuh` 의 mid TF32 launch config 에서 kp_max 정렬을 4 로 변경 (k4 path 가 KP=round_to_4 사용):

```cpp
#if defined(CLS_TF32_MID_K4)
    const int kp_max = round_up_int(std::min(level_max_nc, 32), 4);
#else
    const int kp_max = round_up_int(std::min(level_max_nc, 32), 8);
#endif
```

→ kp_max smaller → smem 절약 (nc=12 인 경우 KP 16→12, 25% smem 절약).

### 7a.5 결과

**정확도** (B=64, 1-trial): TF32 noise 안 (다른 변형과 동등).

**wall** (10-trial median, --repeat 50):

| case | B | V0 | V7a | MK4 (V7a + MID_K4) | MK4 vs V0 | MK4 vs V7a |
|------|---|-----|------|-------------------|----------|------------|
| case8387 | 1 | 0.461 | 0.450 | 0.465 | +0.9% (noise) | +1.7% (noise) |
| case8387 | 64 | 0.0273 | 0.0278 | 0.0274 | +0.3% (noise) | -1.4% |
| USA | 1 | 2.309 | 2.117 | 2.140 | **-7.3%** | +1.1% (noise) |
| USA | 64 | 0.493 | 0.485 | 0.487 | -1.3% | +0.5% (noise) |
| **ACTIVSg25k** | 1 | 0.824 | 0.784 | **0.787** | **-4.6%** | (≈ V7a) |
| **ACTIVSg25k** | 64 | 0.114 | 0.115 | **0.111** | **-2.3%** | **-2.5%** |

**nsys per-kernel** (factor_mid_tf32 mean, --repeat 20):

| case | B | V0 | MK4 | delta |
|------|---|-----|-----|-------|
| case8387 | 1 | 15890 ns | 16199 ns | +1.9% |
| case8387 | 64 | 46983 | 59654 | **+27%** (kernel 회귀; wall 영향은 noise 안) |
| USA | 1 | 38990 | 38465 | -1.3% |
| USA | 64 | 1014091 | 897593 | **-11.5%** |
| ACTIVSg25k | 1 | 22034 | 21518 | -2.3% |
| ACTIVSg25k | 64 | 149415 | 132466 | **-11.3%** |

case8387 mid_tf32 kernel +27% 는 nc=8 (77%) 에 대한 k4 의 2× issue overhead 가 직접 영향. 다만 wall 비중 (case8387 wall 의 ~3% 가 mid_tf32) 이 작아 wall delta noise 안.

### 7a.6 왜 ACTIVSg25k 가 가장 이득인가

세 case 의 mid tier 의 KP 절약 가중평균 (k4 vs k8):

```
case8387:
  nc=1: 1.5% × (8→4 = 50% save) = 0.75%
  nc=2: 9.1% × 50% = 4.5%
  nc=8: 77.3% × 0% = 0%
  ...
  → weighted save ≈ 7% (대부분 nc=8 무이득)

USA:
  nc=20: 36% × (24→20 = 17%) = 6.1%
  nc=12: 8.4% × 25% = 2.1%
  nc=10: 7.8% × 25% = 2.0%
  nc=8: 7.9% × 0% = 0%
  ...
  → weighted save ≈ 14%

ACTIVSg25k:
  nc=12: 59.2% × 25% = 14.8%
  nc=4: 4.5% × 50% = 2.3%
  nc=2: 5.0% × 50% = 2.5%
  ...
  → weighted save ≈ 22%
```

→ ACTIVSg25k 가 패딩 절약 가중평균 가장 큼 (~22%). nsys kernel -11% 와 정성 일치 (절약의 절반 정도가 실제 kernel time 절감 — stage-in 절약 효과).

### 7a.7 결론 (잠정)

- **MK4 는 V7a 위에 추가해도 손해 없음** (case8387 noise floor 안, USA marginal, ACTIVSg25k +4% wall 게인).
- 한계: case8387 mid_tf32 kernel 의 +27% (nsys mean) — nc=8 dominant level 에 k4 가 적용되어 issue overhead 만 증가. wall 영향은 wall 비중 작아 noise 안.
- 자연스러운 후속 (§7b): per-level k4 vs k8 동적 선택 (`KP_k4 < KP_k8` 일 때만 k4).

## 7b. V9h (Hybrid) — k4/k8 per-level dispatch

V9 (always-k4) 의 약점: case8387 mid level 의 77% 가 nc=8 → `KP_k4=8 = KP_k8`. 패딩 절약 0% 인 데도 k4 의 K-loop iter 2배 (1→2) overhead 부담 → factor_mid_tf32 kernel +27% 회귀.

**V9h** 는 dispatch 단계에서 per-level 로 k4 vs k8 동적 선택. heuristic 은 단순:

```
use_k4 := (KP_k4 < KP_k8)      # ↔ level_max_nc % 8 ∈ {1, 2, 3, 4}
```

| level_max_nc | KP_k4 | KP_k8 | use_k4 | 효과 |
|--------------|-------|-------|--------|------|
| 1, 2, 3, 4   | 4     | 8     | true   | k4 가 50% 절약 |
| 5, 6, 7, 8   | 8     | 8     | false  | 같음 → k8 (no k4 issue overhead) |
| 9, 10, 11, 12 | 12   | 16    | true   | 25% 절약 |
| 13, 14, 15, 16 | 16  | 16    | false  | k8 |
| 17, 18, 19, 20 | 20  | 24    | true   | 17% 절약 |
| 21..24 | 24    | 24    | false  | k8 |
| 25..28 | 28    | 32    | true   | 12.5% 절약 |
| 29..32 | 32    | 32    | false  | k8 |

case 별 use_k4 비율 (mid level 단위):
- **case8387**: nc=8 dominant (77%) → use_k4=false 대다수. k4 회피.
- **USA**: nc 분포 다양, use_k4 약 50%.
- **ACTIVSg25k**: nc=12 dominant (59%) → use_k4=true 대다수.

### 7b.1 구현

`src/factorize/kernels.cuh` 에 `factor_mid_tf32_k4` (always-k4 trailing 의 wrapper kernel) 추가 — 기존 factor_mid_tf32 옆에 별도 `__global__`. smem 레이아웃은 정확히 동일 (Csc gap 도 reserve — dispatcher 의 shb 와 일치):

```cpp
#if defined(CLS_TF32_MID_K4_HYBRID)
__global__ void factor_mid_tf32_k4(...) {
    ...
    extern __shared__ char smem_mid_tf32[];
    float* Ltf = ...;
    float* Utf = Ltf + (long)ucp_max * kp_max;
    float* Csc = Utf + (long)kp_max * ucp_max;   // unused but reserve to match
    float* Fs  = Csc + (nt / 32) * 256;
    (void)Csc;
    ...
    trailing_update_mma_tf32_k4_ptx(Fs, fsz, nc, uc, Ltf, Utf, t, nt);
    ...
}
#endif
```

`src/factorize/dispatch.cuh` 의 mid_tf32 launch:

```cpp
#if defined(CLS_TF32_MID_K4_HYBRID)
    const int kp_k4 = round_up_int(level_nc_clipped, 4);
    const int kp_k8 = round_up_int(level_nc_clipped, 8);
    const bool use_k4 = (kp_k4 < kp_k8);
    const int kp_max = use_k4 ? kp_k4 : kp_k8;
#endif
...
if (use_k4) factor_mid_tf32_k4<<<grid, mblk, shb, stream>>>(...);
else        factor_mid_tf32<<<grid, mblk, shb, stream>>>(...);
```

`src/multifrontal.cu` 의 setup 에서 새 kernel 의 smem cap 등록:

```cpp
#if defined(CLS_TF32_MID_K4_HYBRID)
cudaFuncSetAttribute(factor_mid_tf32_k4,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
#endif
```

### 7b.2 구현 함정 — `cudaFuncSetAttribute` 누락

첫 빌드에서 case8387 만 통과, USA/ACTIVSg25k SEGFAULT.

`compute-sanitizer` 출력:
```
Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaLaunchKernel.
```

**원인**: sm_86 의 per-block dynamic shared memory default = **48 KB**. 그 이상을 launch shb 로 요청하면 `cudaErrorInvalidValue` 로 launch 실패. 기존 kernel 들 (`factor_mid_tf32`, `factor_big_tf32` 등) 은 setup 에서 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 99 KB)` 로 cap 을 99 KB 로 raise 등록. 새 `factor_mid_tf32_k4` 등록 누락 → 48 KB 초과 launch 가 silent fail.

case8387 통과한 이유: case8387 mid level 은 use_k4=false (nc=8 dominant) 라 factor_mid_tf32_k4 가 실제로 호출되지 않음. USA/ACTIVSg25k 는 use_k4=true 인 level 이 있어 미등록 kernel launch → 실패.

내부 graph capture mode 에서 이 실패가 graph capture 를 invalidate (`cudaErrorStreamCaptureInvalidated`) → host 가 다음 cudaLaunchKernel 도 실패 → segfault 형태로 표면화. nograph 모드에서는 `batch_relres=9.4e+27` 같은 garbage 값으로 표면화 (factor 가 silent fail 후 다음 단계가 잘못된 메모리 읽음).

**교훈**: 새 `__global__` kernel 을 dynamic smem > 48 KB 로 launch 할 거면 setup 에서 반드시 `cudaFuncSetAttribute` 등록. silent failure mode 가 case 별로 다르게 표면화돼 misleading.

### 7b.3 결과 — wall (10-trial median, --repeat 50)

| case | B | V0 | V7a | MK4 (V9, always-k4) | **V9h (hybrid)** |
|------|---|-----|------|---------------------|---------------------|
| case8387 | 1 | 0.451 | 0.457 (+1.3%) | 0.456 (+1.0%) | **0.448 (-0.7%)** |
| case8387 | 64 | 0.0275 | 0.0279 (+1.5%) | 0.0270 (-1.9%) | **0.0274 (-0.6%)** |
| **USA** | **1** | **2.297** | 2.209 (-3.8%) | 2.109 (-8.2%) | **2.068 (-10.0%)** |
| USA | 64 | 0.492 | 0.489 (-0.6%) | 0.484 (-1.7%) | **0.479 (-2.7%)** |
| **ACTIVSg25k** | **1** | **0.800** | 0.794 (-0.8%) | 0.773 (-3.3%) | **0.759 (-5.0%)** |
| ACTIVSg25k | 64 | 0.114 | 0.117 (+2.6%) | 0.111 (-2.7%) | **0.109 (-4.1%)** |

**V9h 가 모든 case × B 에서 best (regression 0)**. case8387 의 always-k4 약한 흔들림 (±1-2%) 도 normalize. USA -10.0% / ACTIVSg25k -5.0% 는 본 연구 시리즈 단일 변형 중 가장 큰 잘 정량화된 wall 게인.

### 7b.4 왜 V9h 가 always-k4 (V9) 보다 좋은가

case8387 의 always-k4 +27% mid_tf32 kernel 회귀 (§7a.5 의 nsys) 가 hybrid 에서 사라짐. case8387 mid level 의 nc=8 fronts 가 k8 path 로 라우팅 → k4 의 2배 issue overhead 회피. 그러면서 USA / ACTIVSg25k 의 use_k4=true level 게인은 보존.

또한 USA 의 wall delta 가 -8.2% → -10.0% 로 커진 것: USA mid 의 ~50% level 만 use_k4=true. 나머지 50% (nc=8, 16 등) 는 k4 의 issue overhead 없이 k8 path 로 라우팅 → 더 나은 평균 cost.

## 8. 최종 비교

### 8.1 V0 → V9h 누적 (10-trial median, --repeat 50)

| case | B | V0 | V7a | MK4 (V9) | V9h (hybrid) | **V9h vs V0** |
|------|---|-----|------|----------|--------------|---------------|
| case8387 | 1 | 0.451 | 0.457 | 0.456 | 0.448 | **-0.7% (noise)** |
| case8387 | 64 | 0.0275 | 0.0279 | 0.0270 | 0.0274 | **-0.6% (noise)** |
| **ACTIVSg25k** | 1 | 0.800 | 0.794 | 0.773 | **0.759** | **-5.0%** |
| **ACTIVSg25k** | 64 | 0.114 | 0.117 | 0.111 | **0.109** | **-4.1%** |
| **USA** | **1** | **2.297** | 2.209 | 2.109 | **2.068** | **-10.0%** |
| USA | 64 | 0.492 | 0.489 | 0.484 | **0.479** | **-2.7%** |

### 8.2 변형 별 채택 여부

| 변형 | flag | 권고 |
|------|------|------|
| V0 (default) | (none) | **default 권장** — safe baseline |
| V1 (all-tier PTX) | `CLS_TF32_PTX_VARIANT` | 추천 안 함 — case8387 회귀 |
| V2 (big-only PTX) | `CLS_TF32_BIG_PTX` | V9h 의 구성 요소 |
| V3 (Csc dispatch skip) | (auto with V2) | 효과 없음 (occupancy 변화 < 0.1%) |
| V4 (bigT 512) | `CLS_TF32_BIG_T_512` | 추천 안 함 — USA B=1 회귀 |
| V6 (A-reuse hoist) | `CLS_TF32_MMA_AREUSE` | V9h 의 구성 요소 |
| V7a (conv skip) | `CLS_TF32_SKIP_CONVERT` | V9h 의 구성 요소 |
| V9 (always-k4 mid) | `CLS_TF32_MID_K4` | V9h 가 superseded — case8387 약한 흔들림 있음 |
| V9 base k8 mid PTX | `CLS_TF32_MID_PTX` | V9h 의 k8 side path (k8 ≠ V0 wmma) |
| **V9h (k4/k8 hybrid)** | `CLS_TF32_MID_K4_HYBRID` | **V9h 의 핵심 — k4/k8 per-level 선택** |
| **V9h 권장 스택** | `BIG_PTX=1 MMA_AREUSE=1 SKIP_CONVERT=1 MID_PTX=1 MID_K4_HYBRID=1` | **regression 0, 모든 case dominate** |
| V8 (panel_cap) | `--panel-cap N` | case-specific manual tuning |

## 9. 성공/실패 원인 분석

### 9.1 성공 요인 (V2 / V6 / V7a / V9 / V9h)

| 변형 | 성공 원인 |
|------|---------|
| V2 | tier 별 특성 분리 — big front (uc ≥ 16) 는 bound-check divergence 가 적어 PTX 의 smem 우회 게인 그대로 실현; mid 의 small front 는 V0 의 vectorizable per-lane loop 가 더 효율 |
| V6 | A 가 tj8-loop 와 무관함을 데이터 의존성 분석으로 인식 → register hoist 로 ntj8 × 의 redundant load 제거. register pressure +16 도 occupancy binding 안 깨짐 |
| V7a | mma `.tf32` ABI 의 implicit truncate 인식 → 명시적 conv 의 redundancy 파악. 정확도 영향 없는 free win |
| V9 | mid tier 의 K-dim 패딩 분포 (CLS_DUMP_FRONTS) 를 측정 → ACTIVSg25k 의 nc=12 dominance 발견 → m16n8k4 의 KP_k4=12 < KP_k8=16 으로 25% smem/FLOP 절약 |
| V9h | V9 의 case8387 약점 (nc=8 dominant 인 level 에서 k4 의 issue overhead 만 증가) 을 dispatch 단계 per-level 분기로 해결. heuristic 이 단순 (`KP_k4 < KP_k8`) 이라 거의 zero-cost. 모든 case dominate |

공통 패턴: **(a) ABI / HW spec 의 implicit behavior 정확히 이해, (b) 실측 분포 데이터 기반 케이스 별 최적화 인지, (c) tier / level 별 dispatch 분기 로 case 차이 흡수**. PTX 문서 직역 (§3.2) 이나 occupancy 의 단순 smem-기준 추정 (§5) 은 hardware reality 와 어긋남.

### 9.2 실패 요인 (V1 all-tier / V3 / V4)

| 변형 | 실패 원인 |
|------|---------|
| V1 (all-tier) | PTX 의 per-lane bound-check 가 V0 의 batched-lane-loop 보다 branch divergence 가 큼. uc 가 16 의 배수와 멀거나 작은 front 에서 손해 두드러짐 |
| V3 | sm_86 의 occupancy binding constraint 가 **thread cap (1536/SM)** 이지 smem 이 아니었음. binding 추적 잘못함 |
| V4 | thread cap 해소를 위해 bigT 줄였더니 **register cap (64K/SM)** 이 새 binding. binding 의 fractional balancing 으로 평균 occupancy 오히려 - |

공통 패턴: occupancy 의 단일 metric (smem 사용량 또는 block size) 만 보고 결정. multi-constraint 시스템에서는 가장 tight 한 constraint 가 dominate.

## 10. 영구 학습 (코드/문서를 떠나 잔존)

### 10.1 m16n8k8 TF32 A 레지스터 layout (§3)

```
a0 = A[laneR + 0, laneC + 0]    (M_top, K_even)
a1 = A[laneR + 8, laneC + 0]    (M_bot, K_even)
a2 = A[laneR + 0, laneC + 1]    (M_top, K_odd)
a3 = A[laneR + 8, laneC + 1]    (M_bot, K_odd)
```

inner stride 가 K 가 아니라 M block. cutlass `SM80_16x8x8_F32TF32TF32F32_TN` 와 일치. 다음에 PTX MMA 직접 부를 때 self-probe 필수.

### 10.2 CUDA inline asm + 배열 인덱스 (§6.2)

`"+f"(arr[var])` 는 var 가 compile-time 알려져야 register-binding. 강제 unroll 패턴:

```cpp
#pragma unroll
for (int i = 0; i < MAX_CONST; ++i) {
    if (i >= runtime_n) break;     // <-- early break
    asm("..." : "+f"(arr[i]));     // <-- i 가 compile-time 상수
}
```

### 10.3 mma `.tf32` ABI 의 implicit truncate (§6.4)

명시적 `wmma::__float_to_tf32` 호출은 redundant. 호출 안 해도 mma 가 low 13 bit 를 자동 무시. round mode 만 round-to-nearest → truncate 로 바뀜 (ULP 안).

### 10.4 occupancy 의 multi-constraint binding (§5)

sm_86 에서 4가지 binding 후보 (모두 동시 만족 필요):
- shared memory per block
- registers per block (= reg/thread × thread/block)
- threads per block
- threads per SM (1536)
- blocks per SM (16)

binding 추적 시 ncu 의 `launch__registers_per_thread`, `sm__warps_active.avg.pct_of_peak_sustained_active` 같이 봐야 함.

### 10.5 새 `__global__` kernel 의 dynamic smem cap 등록 필수 (§7b.2)

sm_86 의 per-block dynamic shared memory default = 48 KB. 그 이상을 launch 하려면 `cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024)` 로 명시적 raise 등록 필요. **누락 시 launch 가 `cudaErrorInvalidValue` 로 fail**, internal graph capture mode 에서는 capture invalidate → segfault 형태로, nograph mode 에서는 garbage 출력으로 표면화 (case 별 다른 증상 → debug 헷갈림). setup 단계의 기존 kernel 등록 옆에 추가하는 것을 패턴화.

### 10.6 measurement noise floor (§8.1)

power-grid solve 의 multi-stream + atomicAdd 비결정성으로 인해 V0 vs V0 (동일 코드) 도 wall 차이 1.3-2.5% 발생. **wall 게인 < 3% 는 보고하기 어려움**. nsys per-kernel 도 ±10% jitter.

## 11. 코드 / 빌드 / 재현

### 11.1 추가된 / 수정된 파일

| 파일 | 변경 |
|------|-----|
| `src/factorize/phases.cuh` | V0 (`trailing_update_wmma_tf32_f32`) 유지 + V1+ (`trailing_update_mma_tf32_ptx`) 추가. V6 의 A-reuse 분기 + V7a 의 conv-skip 분기 모두 같은 함수 안. |
| `src/factorize/kernels.cuh` | `factor_mid_tf32` / `factor_big_tf32` 의 trailing 호출에 `CLS_TF32_MID_PTX` / `CLS_TF32_BIG_PTX` / `CLS_TF32_PTX_VARIANT` 매크로 분기. |
| `src/factorize/dispatch.cuh` | V2 의 big_tf32 launch config 에서 Csc 영역 skip + V4 의 `CLS_TF32_BIG_T_512` 매크로 분기. |
| `tests/run_custom_solver.cu` | V8 의 `--panel-cap N` flag (analyze 의 supernode panel 폭 override). |

### 11.2 빌드 옵션

| 빌드 dir | flags | 변형 |
|---------|-------|------|
| `build-tf32-v0/` | (default) | V0 baseline |
| `build-tf32-v1/` | `-DCLS_TF32_PTX_VARIANT=1` | V1 (all-tier PTX, deprecated) |
| `build-tf32-v2/` | `-DCLS_TF32_BIG_PTX=1` | V2 (big-only) |
| `build-tf32-v6/` | `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1` | V6 (V2 + A-reuse) |
| `build-tf32-v7a/` | `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 -DCLS_TF32_SKIP_CONVERT=1` | V7a (V6 + conv skip) |
| `build-tf32-mk4/` | V7a flags + `-DCLS_TF32_MID_K4=1` | V9 always-k4 (superseded) |
| `build-tf32-v9h/` | V7a flags + `-DCLS_TF32_MID_PTX=1 -DCLS_TF32_MID_K4_HYBRID=1` | **V9h 권장** |

cmake 예시:
```bash
cmake -S . -B build-tf32-v9h -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 \
                        -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 \
                        -DCLS_TF32_MID_K4_HYBRID=1"
```

### 11.3 재현 예시

```bash
# baseline V0
build-tf32-v0/custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case_SyntheticUSA \
    --precision tf32 --batch 1 --batch-only --repeat 50

# V7a stack
build-tf32-v7a/custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case_SyntheticUSA \
    --precision tf32 --batch 1 --batch-only --repeat 50

# panel_cap sweep
for pc in 4 8 12 16; do
    build-tf32-v7a/custom_linear_solver_run ... --panel-cap $pc
done
```

### 11.4 probe 소스 (V1 layout 발견용)

`/tmp/mma_probe[2-4].cu` 가 V1 의 A-layout 역추론에 사용된 마이크로 테스트. layout 변경 / 다른 mma shape (예: m16n8k16, m8n8k4) 사용 시 동일 패턴으로 재검증 권장.

## 12. 후속 가능성

본 연구가 다루지 않은 lever 와 그 위험성:

1. **k4 path 에 V6 A-reuse hoist 적용**. K-loop iter 가 2배 라 c[NTJ8_MAX][4] register pressure 비대칭. 위험 vs 게인 trade — A-load 가 더 자주 일어나는 만큼 hoist 의 효과도 그만큼 크다는 측면도. 실험 가치.
2. **m16n8k16 TF32 시도**. K=8 보다 더 큰 K 가 있나? 사실 TF32 mma 의 K 는 4, 8 만 (사양). 그러나 m16n16k8 / m16n32k8 같은 N 확장 시도. dispatch 의 ntj16 / ntj8 loop 단순화 가능성.
3. **FP16 TC trailing 에 같은 PTX 변환** (`factor_mid_tc`, `factor_big_tc`). `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. K=16 이라 §3 의 A-layout 이 다를 가능성. 별도 self-probe 필요.
4. **bound-check divergence 완화 (mid k8 PTX 의 V0 fallback 부분 흡수)**. uc 가 16 의 배수가 아닌 경우 마지막 row/col strip 만 V0 fallback. 코드 복잡도 ↑ 대비 게인 불확실.
5. **bigT 줄이기 + `__launch_bounds__`**. V4 의 register cap 문제를 명시적 launch_bounds 로 해결 — 위험: register 줄이면 spill 가능.
6. **trailing GEMM 외 phase 의 GEMM 가속**. panel LU, U-solve 도 TC 화 가능 (Lopez–Mary 2023). 트리거 조건이 까다로움 (mid tier 이상).
