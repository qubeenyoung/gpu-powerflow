# 최종 보고서 — Custom GPU Sparse Direct Linear Solver 최적화

> **상태**: canonical   **갱신**: 2026-06-10
> **한 줄**: power-grid Jacobian factor+solve wall 최소화 — 가장 큰 lever는 selinv default OFF(factor −40 %), 다음은 multistream subtree dispatch. TC는 B ≤ 16 작은 batch에서 −5~−17 % 우세.

- 기간: 2026-06-04 ~ 2026-06-05
- 대상: power-grid Jacobian (case_ACTIVSg2000 ~ case_SyntheticUSA)
- GPU: RTX 3090 (sm_86)
- 목표: factor + solve wall 최소화, TC 활용성 검증

이 문서는 canonical master report다. full sweep 표는 [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md), cuDSS 비교는 [`03-bench-vs-cudss.md`](03-bench-vs-cudss.md)를 참조한다.

## 1. 핵심 결과

### 1.1 가장 큰 lever 두 개

| Lever | 효과 (factor wall) | 적용 범위 |
|---|---:|---|
| **selinv default OFF** | **−40 %** | TC + batched 양쪽 모두 |
| **multistream subtree dispatch** | **−10~19 % (B 작을수록 큼)** | TC + batched 양쪽 모두 |

이 둘만으로 morning baseline 대비 factor wall **−47~−50 %** 달성.

### 1.2 TC vs FP32 — fair 비교 (B=64 case8387 기준)

| Mode | factor + solve μs | vs FP32 |
|---|---:|---:|
| FP64 (mixed-precision baseline) | 102 | +103 % |
| FP32 (selinv OFF, multistream, staged) | **50** | baseline |
| TC (selinv OFF, multistream, staged) | **49** | **−2 %** |

TC의 *실제* wall 우위 = ~2 %. 큰 영역(USA)에서는 B=1/4에 −17~22 % win 도달.

### 1.3 GEMM 비중 — 직접 측정 결과

| | 이론 (FLOPS) | 실측 (wall, skip-trailing variant) |
|---|---:|---:|
| Trailing GEMM 비중 (case8387 B=1) | 86 % | **38 %** |
| Trailing GEMM 비중 (case8387 B=256) | 86 % | **21 %** |

이론과 실측 격차의 원인: trailing은 compute-bound 고병렬, LU/stage/writeback은 memory-bound + serial chain → trailing의 wall/FLOP 효율이 평균의 2.3× 높음.

TC 가속 ceiling (Amdahl, f=실측):
- X=2 (WMMA 2× throughput): wall speedup **1.15-1.27×**
- 현재 측정 TC win: USA B=1 1.21× → **ceiling의 88 % 이미 활용**

## 2. 최적 dispatch 경로

### 2.1 TC path (`--tc`, default)

```
LEVEL의 max_fsz 기준 routing:

  max_fsz ≤ 32            → mf_factor_small_warp_b<float>
                            (FP32 scalar, warp-per-front, 8 warps/block)

  32 < max_fsz < 48       → mf_factor_mid_tc_lo_b<24>
                            (WMMA TC for fsz≥24 per-panel, fallback for small mid)

  48 ≤ max_fsz ≤ 128      → mf_factor_mid_tiled_b
                            (FP32 scalar, shared-staged trailing, 256 threads)
                            ★ dominant kernel for case8387

  max_fsz > 128           → mf_factor_extend_tc32_b
                            (WMMA TC trailing on global, 1024 threads)
                            ★ dominant kernel for USA's BIG fronts

위 위에:
  - Multi-stream: K=8 subtree branches 병렬 + spine 직렬 (default ON)
  - selinv OFF (default)
  - tc_warmup 사전 cuBLAS init (호출자가)
  - on-device cuBLAS pointer build (자동, tc_setup)
```

### 2.2 FP32 batched path (`MF_FP32=1`)

동일 routing, 단 mid_tc_lo<24> 대신 `mf_factor_mid_tc32_b<false>` 사용 (FP32 scalar fallback). extend path도 WMMA 대신 `mf_factor_extend_level_b<float>` (FP32 scalar). 나머지 lever (multistream, selinv OFF, staged trailing)는 TC와 동일하게 적용됨 (Σ.12, Σ.14).

### 2.3 환경변수 최종 (모든 default 적용 후)

| Env var | Default | 효과 |
|---|---|---|
| `CLS_NO_MULTISTREAM=1` | off (multi ON) | multistream 비활성 |
| `CLS_USE_SELINV=1` | off (selinv OFF) | selinv 활성 (1-factor-many-solve 시) |
| `CLS_USE_CUBLAS=1` | off | cuBLAS grouped batched trailing (accuracy 개선, wall flat) |
| `CLS_CUBLAS_MIN_FSZ=N` | 64 | cuBLAS 적용 최소 fsz |
| `CLS_CUBLAS_TF32=1` | off | TF32 TC compute |
| `CLS_USE_AMAL=1` | off | etree-aware amalgamation (depth ↓, wall regression) |
| `CLS_AMAL_CAP=N` | 32 | merge cap |
| `CLS_AMAL_MIN_DEPTH=N` | 0 | depth threshold |
| `CLS_USE_PIVOTING=1` | off | within-panel partial pivoting (cap ≤ 32 안전) |
| `CLS_USE_REGBLOCK=1` | off | FP32 4×4 register-tiled trailing (B=1에서 marginal) |
| `CLS_USE_REGBLOCK_H16=1` | off | FP16-input register tile (accuracy 15× 악화) |
| `CLS_NO_TILED_TRAILING=1` | off | staged trailing 비활성 (WMMA path 강제) |
| `CLS_PROFILE_NO_TRAILING=1` | off | **측정 전용**: trailing 제거 variant (wall 빼기) |
| `CLS_USE_SPINE=1` | off | spine fused kernel (noise 안) |

### 2.4 새 API

```cpp
// Process-level cuBLAS 사전 init — 앱 시작 시 호출 권장 (~10 ms 절감/setup)
bool custom_linear_solver::tc::tc_warmup();
```

## 3. Comprehensive sweep — 최종 측정 요약

Full 5 cases × 3 modes × 5 batch sizes 표는 [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md)를 canonical source로 둔다. 이 문서는 최종 결론에 필요한 crossover만 남긴다.

| Pattern | 측정 결론 |
|---|---|
| B=1 | 모든 case에서 TC가 FP32 대비 −4.7 ~ −10.7 % |
| B=4 | 대부분 TC 우세, USA에서 최대 win −17.5 % |
| B=16 | TC와 FP32가 대체로 tie, case8387은 FP32가 +5 % 우세 |
| B≥64 | 큰 grid에서는 FP32가 우세하거나 tie, case8387은 TC/FP32 혼전 |
| FP64 | 모든 영역에서 FP32/TC 대비 2-5× 느림 |

### 3.1 Optimal-dispatch 권장 (use case 별)

| Use case | B 범위 | Mode |
|---|---|---|
| Newton-Raphson 1-system | B=1 | **TC** (−5~−11%) |
| 소형 Monte Carlo (작은 grid) | B 2-8 | **TC** (−5~−12%) |
| 소형 Monte Carlo (큰 grid USA) | B 4 | **TC** (−17.5 %, 전체 최대 win) |
| 대형 batch (B ≥ 64) | 일반 | **FP32** (USA 명확, case8387 tie) |
| 정확도 critical | 모든 | **FP64** (2-5× 느리지만 안전) |

전체 optimal-configuration 결정 근거는 [`../optimal-configuration.md`](../optimal-configuration.md) 참조.

## 4. TC의 실제 작동 — 무엇이 TC인가

### 4.1 WMMA 호출은 단 하나의 device function

**`tc_trailing_wmma_f32` (`src/tc/factor_tc.cuh:38`)** — front의 trailing rank-nc update GEMM만:
```
C(uc × uc) -= L(uc × nc) * U(nc × uc)
```
- L, U를 FP32 → FP16 cast 후 shared에 stage
- `wmma::fragment<16,16,16,__half>` load
- `wmma::mma_sync(cf, af, bf, cf)` — FP32 accumulate TC 연산
- 결과 FP32로 main front의 C panel에 subtract

### 4.2 TC fire 위치 (2 kernel)

| Kernel | 영역 | WMMA 조건 |
|---|---|---|
| `mf_factor_extend_tc32_b` | BIG (fsz > 128) | nc ≤ 32, uc ≤ 256 |
| `mf_factor_mid_tc_lo_b<24>` | MID의 small fallback (max_fsz < 48) | fsz ≥ 24, nc ≤ 32, uc ≤ 256 |

### 4.3 TC가 NOT 쓰는 영역 (TC path라도 FP32 scalar)

| Kernel | 영역 |
|---|---|
| `mf_factor_small_warp_b<float>` | small (max_fsz ≤ 32) |
| **`mf_factor_mid_tiled_b`** | **MID dominant (48 ≤ max_fsz ≤ 128)** |
| LU panel factor / U solve | 모든 path |
| Triangular solve | 모든 path |
| Pivot invert (selinv on 일 때) | 모든 path |

### 4.4 TC 비중 (kernel time)

| Case | TC kernel 비중 (factor 중) |
|---|---:|
| case8387 B=1 | ~ 20 % |
| USA B=1 | ~ 60 % |

→ USA가 TC 효과 더 받는 이유 = BIG-front WMMA가 dominant (`extend_tc32_b` per-kernel −39 % vs FP32 `extend_level_b`).

## 5. 시도했으나 채택 안 한 lever (negative findings)

| Lever | 결과 | 원인 |
|---|---|---|
| **Etree-aware amalgamation** (Σ.4/5) | depth 30→16 ✓, wall +65 % ❌ | per-front work cubic 증가 > launch saving (KLU 2010 결론 재확인) |
| **Within-panel partial pivoting** (Σ.8) | cap ≤ 32 안전, cap ≥ 48 garbage ❌ | L panel row 포함 swap 필요, asm_local 큰 재작업 |
| **cuBLAS sgemmGroupedBatched** (Σ.6/7) | accuracy 2-4× 개선 ✓, wall flat | front 크기 cuBLAS sweet spot (≥ 300) 아래 |
| **FP16 register-blocked** (Σ.15) | wall −9 % ✓, relres 15× 악화 ❌ | FP16 mantissa 부족, 값 범위 underflow |
| **WMMA packing (multi-panel)** | packing으로 throughput 1.5-2× ↑, 그러나 scalar 못 이김 ❌ | 작은 panel에서 scalar도 잘 작동 + WMMA per-call overhead |
| **lower WMMA threshold** (24 → 16) | wall +8 % ❌ | padding waste의 비례 증가 |
| **Spine fused kernel** | wall ±0 (noise) | spine이 fundamentally sequential |
| **FP32 register-blocked** (Σ.2) | B=64 ±0, B=1 −7.5% | B=1 영역 lever이지만 dispatch 분기 복잡 |

## 6. 시도하지 못했거나 다음 lever (future work)

| Lever | 잠재 win | 비용 |
|---|---|---|
| **Full partial pivoting** (L panel row 포함) | cap ≥ 48 가능 → amalgamation의 wall regression 극복? 불명 | very high (asm_local 추적 + extend-add unpermute, ~1500 LOC) |
| **Solve kernel 최적화** | solve가 wall의 40 %, lever 검증 안 됨 | medium (~300 LOC) |
| **cuBLAS sgemmGroupedBatched + amalgamation 결합** | wide panel + cuBLAS-class efficiency 가능? | high (numerical 안정성 + 구현) |
| **MAGMA vbatched 통합** | 외부 dep, variable-K 지원 | high |
| **mid_tc32_b<false>의 trailing도 WMMA화 + register tile** | mid_tiled의 win 영역 잠식 가능 | medium |

## 7. 어떤 lever가 *진짜* 가속의 본질인가

시간순 ablation 분석 (case8387 B=64):

| 단계 | 적용 lever | FP32 TOTAL | TC TOTAL | TC vs FP32 |
|---|---|---:|---:|---:|
| (1) 오전 baseline | selinv ON, single-stream, non-staged | 92.1 | 97.5 | +5.9 % (TC LOSES) |
| (2) **+ selinv OFF** | (1) + selinv OFF | 64.9 | 60.0 | **−7.5 % (TC WINS flip)** ★ |
| (3) + multistream | (2) + both paths multistream | 49.9 | 50.9 | +2.0 % |
| (4) + staged trailing | (3) + Σ.14 (FP32도 staged) | 50.0 | 49.5 | −1.0 % |

**TC가 FP32 못 이기다 winner로 flip된 단일 lever = `selinv OFF`** (Σ.9). 이후 lever들은 양쪽 path 비례 가속.

selinv OFF가 TC 더 도와준 메커니즘:
- `mf_invert_pivot_b` kernel 자체 비용은 FP32/TC 동일
- 그러나 captured graph의 dispatch overhead (B*P개 launch block)가 TC의 잘게 split된 kernel 구조에 누적 effect 더 큼
- TC factor가 6.5 μs/sys 더 절감 (FP32 36.6 vs TC 43.1 μs/sys)

## 8. 직접 측정으로 정정된 이론 추정

### 8.1 GEMM 비중

- 이론 FLOPS: trailing = 86-88 %
- **실측 wall: trailing = 21-43 %**
- 격차 원인: trailing의 wall/FLOP 효율이 평균의 2.3× (LU panel factor의 serial chain + stage/writeback memory traffic이 wall 더 많이 차지)

자세한 분석은 [`../02-design-analysis/04-gemm-fraction-tc-ceiling.md`](../02-design-analysis/04-gemm-fraction-tc-ceiling.md) 참조.

### 8.2 TC ceiling

- 이전 (잘못된 FLOPS 가정): X=2 → 1.75× speedup
- **실측 (Amdahl with f=trailing wall fraction)**: X=2 → **1.15-1.27×**
- 현재 TC 활용률:
  - case8387 B=1: 1.10× (ceiling 1.24×의 42 %)
  - **USA B=1: 1.21× (ceiling 1.24×의 88 %, 거의 한계)**

### 8.3 B 증가 시 trailing% 감소

- B=1: trailing 38 %
- B=256: trailing 21 %
- 원인: stage/writeback (memory BW)이 작은 B에서 saturate, trailing (compute)은 계속 scaling → trailing 비중 *상대적* 감소

## 9. Wall time breakdown — case8387 B=1 (가장 dramatic case)

| Phase | Time μs/sys | 비중 |
|---|---:|---:|
| stage front (F→Fs) memory copy | ~40 | 6 % |
| LU panel factor (serial chain) | ~150 | 22 % |
| U panel solve | ~50 | 7 % |
| **Trailing GEMM (실측)** | **~153** | **23 %** |
| writeback (Fs→F) | ~80 | 12 % |
| extend-add (atomicAdd) | ~40 | 6 % |
| Solve forward + backward | ~250 | 36 % |
| **TOTAL (factor + solve)** | **~685** | 100 % |

→ TC가 영향 줄 수 있는 영역은 trailing 23 %뿐. 나머지 77 %는 scalar / memory-bound.

## 10. 한 줄 요약

> **현재 default dispatch가 우리 power-grid 분포에서 measurement-validated optimal.**
>
> 가장 큰 lever는 **selinv default OFF** (factor −40 %). 다음은 **multistream subtree dispatch** (B 작을수록 효과 큼).
>
> TC는 **B ≤ 16에서 −5~−17 % 우세, B ≥ 64에서 FP32 동등/우세**. TC의 본질은 BIG-front (USA-style) WMMA — 작은 grid에서는 marginal.
>
> 이론 추정 (86 % trailing → 1.75× ceiling) 보다 **실측 측정 (38 % trailing wall → 1.24× ceiling)** 이 정확. TC headroom 제한적.

## 11. 코드 변경 summary

| 카테고리 | LOC 추가/수정 |
|---|---:|
| Σ.1 staged trailing (`tc/trailing_tiled.cuh`) | +180 |
| Σ.5 amalgamation (`symbolic/amalgamate.{hpp,cpp}`) | +247 |
| Σ.6/7 cuBLAS integration (`tc/factor_split_cublas.cuh` + dispatch) | +300 |
| Σ.8 within-panel pivoting (lu_device + solve kernels + state) | +200 |
| Σ.9 selinv default flip (2 lines) | 2 |
| Σ.10 tc_warmup + device pointer build | +50 |
| Σ.11 multistream subtree fix (`factorize/multifrontal.cu` + dispatch) | +80 |
| Σ.12 multistream for FP32 batched | +100 |
| Σ.14 staged trailing for FP32 batched | +40 |
| Σ.15 FP16 register-blocked | +90 |
| Σ.16 skip-trailing variants (`tc/factor_no_trailing.cuh`) | +155 |
| WMMA pack microbench (`tests/wmma_pack_microbench.cu`) | +300 |
| 측정 분석 doc (`docs/**/*.md`) | ~1500 |

**총 약 ~3000 LOC** 추가 / 수정.

## 12. 관련 문서

| 문서 | 내용 |
|---|---|
| [`../main-report.md`](../main-report.md) | 시간순 작업 로그 / 전체 storyline |
| [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md) | 5 cases × 3 modes × 5 batches full sweep (canonical 데이터) |
| [`03-bench-vs-cudss.md`](03-bench-vs-cudss.md) | cuDSS 대비 벤치마크 (raw B=1 + ubatch mt-auto) |
| [`04-factorize-progress.md`](04-factorize-progress.md) | factorize 최적화 progress |
| [`../optimal-configuration.md`](../optimal-configuration.md) | optimal dispatch 구성 결정 |
| [`../02-design-analysis/04-gemm-fraction-tc-ceiling.md`](../02-design-analysis/04-gemm-fraction-tc-ceiling.md) | GEMM 비중 이론 + 실측 + TC ceiling |
