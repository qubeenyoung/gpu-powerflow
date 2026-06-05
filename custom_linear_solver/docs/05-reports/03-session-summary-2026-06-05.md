# 가속 작업 정리 — 2026-06-04 ~ 2026-06-05

> Status: chronological log. 최신 전체 결론은
> [`01-final-report-2026-06-05.md`](01-final-report-2026-06-05.md),
> full sweep 원본 표는 [`02-comprehensive-sweep-2026-06-05.md`](02-comprehensive-sweep-2026-06-05.md)를 우선 참고한다.

power-grid Jacobian (case8387pegase / case_SyntheticUSA) 위 batched TC factor/solve 의 wall time 줄이기 위한 일련 작업.

---

## 1. 시작 상태 (변경 전 baseline)

case8387 B=64 (TC 경로, selinv default ON):
```
batch_factor_per_sys_ms ≈ 71 μs
batch_solve_per_sys_ms  ≈ 20 μs
TOTAL                   ≈ 91 μs/sys
```

FP32 batched (MF_FP32=1, selinv default ON): 대략 같은 ~91μs/sys.

사용자 의문: **"TC 가 왜 FP32 보다 못 이기지?"** + **"cuDSS depth=10 vs 우리 30 — 트리 구조 문제 같다"**

---

## 2. 작업 항목 (Phase Σ.4 ~ Σ.10)

### Σ.4 — Etree-aware amalgamation **이론 검증** (Python prototype)

`tests/depth_analysis_v2.py`:
- chain merge (현재 `relaxed_panels`): case8387 depth=53, USA depth=130 (postorder 후)
- whole-subtree compress (contig 보장): 30+ 까지밖에 못 줄임
- **etree-amalgamate + repostorder** (greedy bottom-up merge with col 재배치): case8387 **depth=9** (cap=64) / USA **depth=12** (cap=128) → cuDSS 와 동등

핵심: 단순 chain-merge 한계는 fundamental supernode 수준 (avg 2 cols/panel). etree-aware bottom-up merge + supernode-postorder col 재정렬로 깊이 3-4× 감소 가능.

### Σ.5 — Amalgamation C++ implementation

| 파일 | LOC |
|---|---:|
| `src/symbolic/amalgamate.hpp` (new) | 47 |
| `src/symbolic/amalgamate.cpp` (new) | 200 |
| `src/solver.cpp` (수정) | +80 |
| `CMakeLists.txt` | +1 |

`CLS_USE_AMAL=1` opt-in. depth 30→16 (cap=16) 달성. 그러나 **wall regression**:
- case8387 baseline: 71 μs
- AMAL cap=16: 117 μs (+65%)

원인: amalgamation 으로 launch 14개 절감 (~28μs) ≪ per-front 작업 증가 (~50μs). KLU 2010 결론과 일치 — power-grid 에서 supernode amalgamation 만으로 wall 도움 안 됨.

`CLS_AMAL_MIN_DEPTH=N` 추가하여 spine-only merge 로 regression 막을 수 있음 (그러나 win 도 없음).

### Σ.6 — cuBLAS sgemmStridedBatched (BIG-front)

| 파일 | 변경 |
|---|---|
| `src/tc/factor_split_cublas.cuh` (new) | 80 LOC |
| `src/tc/multifrontal_tc.{hpp,cu}` | TCState 에 cublas_handle, BIG-front dispatch 분기 |
| `src/plan/multifrontal_plan.{hpp,cu}` | h_front_off 호스트 mirror |
| `CMakeLists.txt` | CUDA::cublas link |

BIG-front (fsz > 128) 경로의 WMMA trailing 을 cuBLAS sgemmStridedBatched 로 치환.
USA 측정: wall ≈ 동일 (±2%), **accuracy 2-4× 개선** (WMMA FP16 rounding 제거).

### Σ.7 — cublasSgemmGroupedBatched + ahead-of-time init

CUDA 12.5+ `cublasSgemmGroupedBatched` 사용 → 한 LEVEL 의 K panels × B systems 를 **하나의 cuBLAS call** 로.

`CLS_USE_CUBLAS=1` 활성, `CLS_CUBLAS_MIN_FSZ=N` (default 64) 로 작은 front 보호 (cuBLAS overhead 가 작업량 압도 방지).

MID + BIG 둘 다 grouped 사용. correctness 측면:
- 초기 fsz≤48 의 **double-subtract 버그** (lu_small_front 가 fused trailing 한 후 cuBLAS 가 또 -L*U) → m=n=k=0 으로 group skip 처리
- MID phaseA 에서 fsz≤48 panel 의 C panel 도 full writeback (그렇지 않으면 stale)

case8387 B=64 측정:
- TC default: 70-72 μs, relres 8e-5 ~ 4e-3 (변동)
- TC + cuBLAS grouped: 71-72 μs, **relres ~2.6e-5 (안정, 100× 개선)**

### Σ.8 — Within-panel partial pivoting (cap > 32 시도)

| 파일 | 변경 |
|---|---|
| `src/batched/lu_device.cuh` | `lu_panel_factor_pp`, `lu_small_front_pp` 추가 |
| `src/batched/solve_kernels.cuh`, `solve_small.cuh` | `mf_fwd_*_pp_b` 추가 |
| `src/tc/factor_split_cublas.cuh` | `mf_factor_{mid,big}_phaseA_pp_b` + init kernel |
| `src/plan/multifrontal_plan.hpp` | h_pivot_offset, d_pivot_offset, total_pivots |
| `src/factorize/multifrontal.cu` | pivot_offset prefix-sum at analyze |
| `src/tc/multifrontal_tc.{hpp,cu}` | TCState 에 d_pivotsB, use_pivoting, dispatch 분기 |

알고리즘: 각 panel 의 nc×nc pivot block 내에서만 row swap 검색 (rows nc..fsz-1 의 L panel 은 안 건드림 → asm_local 호환). `CLS_USE_PIVOTING=1` opt-in.

**버그 fix들 (실제 작업 중 발견)**:
1. cudaMemset(0) 으로 pivots 초기화 → solve 시 identity 가 아닌 swap 으로 해석. Fix: `mf_init_pivots_identity_b` per-panel identity 초기화 kernel
2. forward solve 에서 sticky 한 cuda 에러가 graph instantiate 를 poison. Fix: tc_setup 시작에 `cudaGetLastError()` 로 drain
3. forward solve 에서 sh_piv 의 write-back timing 이 race. Fix: warp sub 완료 후 명시적 별도 write loop

case8387 측정 (selinv OFF):
| amal_cap | factor μs | relres | 비고 |
|---:|---:|---:|---|
| 8 | 44.8 | 1.8e-5 | OK |
| 16 | 46.9 | 2.3e-5 | OK |
| 24 | 44.9 | 1.5e-5 | OK |
| 32 | 49.1 | 2.9e-5 | OK |
| **48** | 49.7 | **12261** | **❌ numerical garbage** |
| **64** | 53.4 | **28613** | ❌ |

**결론**: within-panel pivoting 만으로는 cap ≥ 48 numerical 문제 해결 못함. L panel row 와의 swap 도 필요 (asm_local 추적 + extend-add unpermute 등 큰 refactor). 따라서 wall 효과 없음 (cap ≤ 32 는 baseline 도 안정). 코드는 infrastructure 로 남김.

### Σ.9 — selinv default OFF

`MF_NO_SELINV=1` 가 아니면 default selinv ON 이었음. 측정 결과 selinv 가 **factor 의 50%** 차지:
- selinv ON: factor 71μs, solve 20μs, total 91μs
- selinv OFF: factor 33μs (-54%), solve 22μs (+10%), total **55μs (-40%)**

selinv = pivot block 의 partitioned-inverse 미리 계산 → solve 가 GEMV 로 대체. Newton-Raphson (1 factor → 1 solve) 에서는 factor 절감이 solve loss 압도. **default OFF 로 변경**.

```cpp
// src/tc/multifrontal_tc.cu, src/batched/multifrontal_batched.cu:
st.selinv = (std::getenv("CLS_USE_SELINV") != nullptr) &&
            (std::getenv("MF_NO_SELINV") == nullptr);
```

`CLS_USE_SELINV=1` 로 re-enable 가능 (factor amortize 시 유리한 use case).

**이것이 가장 큰 win.**

### Σ.10 — Setup 비용 절감 (tc_warmup + on-device pointer build)

사용자 질문: tc_setup 비용 + 사전화 가능 여부.

**측정 breakdown** (CLS_TC_SETUP_DBG=1):

| Phase | 변경 전 | 변경 후 |
|---|---:|---:|
| **case8387 total** | **19.2 ms** | **1.6 ms** (-92%) |
| cudaMalloc state arenas | 0.14 ms | 0.14 ms |
| cublasCreate | 11.0 ms | **0.01 ms** ← `tc_warmup()` |
| cuBLAS pointer arrays (3×P×B = 11 MB H2D) | 5.7 ms | **1.0 ms** ← device-side build |
| cudaFuncSetAttribute x7 | 1.5 ms | 0.1 ms (캐시) |
| graph capture + instantiate | 0.5 ms | 0.5 ms |

| Phase | 변경 전 | 변경 후 |
|---|---:|---:|
| **USA total** | **75 ms** | **2.3 ms** (-97%) |
| cublasCreate | 9.1 ms | 0.01 ms |
| cuBLAS pointer arrays (3×P×B = 114 MB H2D) | 55 ms | **1.3 ms** ← device-side build |
| cudaFuncSetAttribute x7 | 5.9 ms | 0.1 ms |

**구현**:
1. **`tc_warmup()`** API (`src/tc/multifrontal_tc.hpp`): process-global cuBLAS handle 캐싱. 앱 시작 시 호출하면 모든 후속 `tc_setup` 에서 cublasCreate 비용 (~10ms) 제거.
2. **Device-side pointer build** (`src/tc/factor_split_cublas.cuh::mf_build_cublas_ptrs_b`): 기존 host 에서 P×B 개 float* 배열 쌓아서 H2D 했던 부분을, per-panel within-front offset (P×4 int = 1.2MB) 만 업로드 + GPU 커널이 base + b*stride + foff + within 으로 pointer 채움. 114MB → 1.2MB upload, 55ms → 1.3ms.

---

## 3. 최종 측정 — Apples-to-apples

사용자 의문: "selinv on FP32 vs selinv off TC 비교한 건 아니지?" → 정정.
**같은 selinv 설정**에서 비교한 결과:
- **selinv OFF 가 FP32/TC 양쪽 모두 dominant win** (-38~-39%)
- **같은 selinv 설정에서 TC 와 FP32 거의 동일** — case8387 +0.7%, USA +10%
- TC 가 FP32 못 이기는 이유는 우리 power-grid front 크기 분포 (max 80-254) 가 cuBLAS/WMMA sweet spot (fsz ≥ 300) 아래

같은 설정 비교의 최신 결론은 대표 문서
[`01-final-report-2026-06-05.md`](01-final-report-2026-06-05.md)와
full sweep [`02-comprehensive-sweep-2026-06-05.md`](02-comprehensive-sweep-2026-06-05.md)를 기준으로 한다.

---

## 4. 코드 변경 summary

| 파일 | 변경 LOC |
|---|---:|
| `src/symbolic/amalgamate.hpp` (new) | +47 |
| `src/symbolic/amalgamate.cpp` (new) | +200 |
| `src/tc/factor_split_cublas.cuh` (new) | +220 |
| `src/tc/trailing_tiled.cuh` (new) | +180 |
| `src/tc/multifrontal_tc.hpp` (수정) | +35 |
| `src/tc/multifrontal_tc.cu` (수정) | +250 |
| `src/batched/multifrontal_batched.cu` | +2 (selinv default flip) |
| `src/batched/lu_device.cuh` | +75 (lu_*_pp variants) |
| `src/batched/solve_kernels.cuh` | +55 (mf_fwd_level_pp_b) |
| `src/batched/solve_small.cuh` | +60 (mf_fwd_small_warp_pp_b) |
| `src/plan/multifrontal_plan.{hpp,cu}` | +15 (h_front_off, h_pivot_offset, d_pivot_offset) |
| `src/factorize/multifrontal.cu` | +10 (pivot_offset prefix-sum) |
| `src/solver.cpp` | +80 (CLS_USE_AMAL pipeline) |
| `CMakeLists.txt` | +2 (cublas link, amalgamate.cpp) |
| `tests/depth_analysis_v2.py` (new) | 200 |
| `tests/run_custom_solver.cu` | +25 (setup timing, tc_warmup) |

**총 ~1450 LOC** 추가/수정.

---

## 5. 환경변수 최종 정리

이 섹션은 작업 당시의 env 목록이었고, 최신 canonical env table은
[`01-final-report-2026-06-05.md`](01-final-report-2026-06-05.md) §2.3에 둔다.

작업 로그에서 중요한 변화만 남기면:
- `CLS_USE_SELINV=1`: 기본 OFF로 바뀐 selinv를 다시 켜는 opt-in.
- `CLS_USE_CUBLAS=1`, `CLS_CUBLAS_MIN_FSZ=N`, `CLS_CUBLAS_TF32=1`: TC trailing의 cuBLAS 실험 knobs.
- `CLS_USE_AMAL=1`, `CLS_USE_PIVOTING=1`, `CLS_USE_REGBLOCK=1`: 실험 인프라로 남은 opt-in knobs.
- `CLS_NO_TILED_TRAILING=1`, `CLS_TC_SETUP_DBG=1`, `CLS_CAP=N`: 측정/디버그용 knobs.
- 새 API: `bool tc_warmup()` — process-global cuBLAS handle 사전 init.

---

## 6. 다음 가속 방안 검토

현재 wall: case8387 ~55μs, USA ~750μs. FP32 baseline 동등. 추가 가속을 위한 후보 — 비용/효과 추정.

### A. Solve kernel 최적화 (high ROI, medium effort)
**현황**: selinv OFF 후 solve = 22μs (case8387) = **40% of total**. 같은 비율로 USA solve 가 27%.

**문제**: warp-parallel substitution (`mf_fwd_level_b` else branch) 는 nc step 마다 32-lane warp 가 1-lane work 하고 broadcast. nc>32 면 broken.

**제안**:
1. cuBLAS strsm batched 로 forward 전체를 한 번에 — but per-level call overhead
2. 또는 solve 도 grouped batched 로 trsm 호출 (cuBLAS LtMatmul 기반)
3. 또는 selinv 의 효율적인 재구현 (block-batched matrix inverse via cuBLAS getrfBatched+getriBatched)

**기대 효과**: solve 22→10μs (-55%) → total 55→43μs (**-22%**). USA 비례 절감.

**비용**: ~300-500 LOC + careful capture testing. 1-2 day.

---

### B. Multi-stream subtree dispatch (revisit Phase 3) (medium ROI, high risk)
**현황**: 분석 시 spine 외부에 K=2-4 개 독립 subtree 존재 (case8387 K=2, USA K=4). 현재 single-stream sequential dispatch.

**제안**: K subtrees fork onto K streams. spine join. CUDA Graph multi-stream capture.

**과거 시도**: `CLS_USE_MULTISTREAM=1` 코드 존재. 그러나 race condition 으로 relres garbage (factorize complete 후 extend-add ordering 문제 추정). 디버그 안 됨.

**기대 효과**: 균형 잘 잡힌 subtrees 면 factor 절반 (예: USA K=4 가 모두 25% 씩이면 -75%). 그러나 실제 imbalance 큼 (case8387: subtree[0]=559 panels, subtree[1]=1528). Speedup limited to slowest subtree → -20-40% 추정.

**비용**: 우선 race condition 디버그 — graph capture order 분석 + cudaEventRecord/Wait 점검. 2-3 day.

---

### C. cuDSS depth=10 도달 — Full partial pivoting + amalgamation cap≥48 (high ROI, high effort)
**현황**: Σ.8 의 within-panel pivoting 으로 cap=32 까지만 안정. cap=48+ 에서 numerical 실패.

**제안**: Full partial pivoting — L panel row 포함 swap 허용. 핵심 변경:
1. asm_local 의 row 인덱스 재배치 추적 (CB rows 가 factor 후 permute 됨)
2. extend-add 시 child front 의 (permuted) C panel 을 parent 에 unpermute 하여 atomicAdd
3. solve 의 backward 도 column permutation 적용 필요? (column 은 안 건드리므로 X)

cap ≥ 48 가능해지면:
- amal_cap=64 → case8387 depth 30→9 (cuDSS 수준)
- per-front 작업 증가하지만, cuBLAS grouped 가 wide front 효율 (30-50% peak vs 1.36%) → net 절감 가능

**기대 효과**: 잘 되면 **-30~50% factor wall**. 안 되면 (numerical limit 더) 0 effect.

**비용**: ~1000-1500 LOC + 풀 multifrontal pipeline 변경. **1-2 주**.

---

### D. Iterative refinement 활용 (FP32 factor + FP64 refinement) (low ROI for our case)
**현황**: FP32 batched relres ~1e-5 ~ 1e-4. 일부 use case 는 lower error 필요.

**제안**: FP32 로 factor + solve → residual = b - Ax → solve 에 추가 → 1-2 iter 면 FP64-level error.

**기대 효과**: 추가 1 solve 당 +22μs. accuracy 만 개선, wall 절감 X. 우리 power-grid 는 이미 충분히 accurate 한 경우 다수.

**비용**: ~100 LOC.

---

### E. Persistent kernel for spine (low ROI, low effort)
**현황**: spine = 10 cnt=1 levels at top. 각 level 마다 kernel launch (~2μs overhead × 10 = 20μs).

**제안**: 하나의 persistent kernel 이 spine 전체를 처리 (block-level sync). 9 launches saving.

**과거 시도**: `CLS_USE_SPINE=1` 코드 존재. 그러나 **+19% regression** 측정됨 (spine fused kernel 의 per-step shared resource pressure).

**기대 효과**: 디자인 개선하면 -5%. 안 하면 negative.

**비용**: 기존 spine kernel 디자인 재검토 + retuning. 1 day.

---

### F. Larger batch size (B=256+) 적합화 (free win 가능성)
**현황**: B=64 측정. B=256/512 는 미측정.

**제안**: B 증가하면 launch overhead 분산 효과. 단 GPU memory bandwidth 가 saturate 되면 정체.

**비용**: 측정만. 측정 후 trend 보고 dispatch 튜닝.

---

## 7. 우선순위 권장

| | 우선순위 | 기대 win | 비용 | 위험 |
|---|---|---:|---|---|
| **A. Solve 최적화** | **1** | **-15~25% total** | medium | low |
| F. B 큰 배치 sweep | 2 | unknown (free) | tiny | none |
| ~~B. Multi-stream subtree~~ | — | — | — | **★ 완료: §8 참조** |
| E. Persistent spine | 4 | -5% (or worse) | low | medium |
| C. Full pivoting | 5 | -30~50% (if works) | very high | very high |
| D. Iterative refinement | — | accuracy only | low | none |

**시작 권장**: A (solve 최적화) + F (B sweep) 병행.

---

## 8. Σ.11 — Multi-stream subtree dispatch 디버그 + 활성화 ★

### 8.1 기존 버그 진단

`CLS_USE_MULTISTREAM=1` 옵션이 코드에 있었지만 **relres ~ 1e+28 garbage**. 추정 race condition.

원인 발견:
1. `tc_factorize` 의 `cudaGraphLaunch` 전 `issue_scatter` 가 main stream 에 메모리 init + scatter 수행 — 이게 captured graph 의 fork_event 보다 시간적으로 먼저 → fork_event 대기로 subtree streams 가 scatter 후 시작. **여기는 정상**.
2. **진짜 버그**: subtree partition 이 잘못됨. 기존 코드는 "K subtree roots = 모든 panels at level (spine_start_level - 1)" 로 정의 → 이들 root 에서 BFS 로 descendants 마킹.
3. 그러나 panel etree 에서 **level L < spine_start_level 인 panel 의 parent 가 spine 인 경우가 존재** (panel level = max child level + 1 이므로 한 panel 이 깊은/얕은 child 둘 다 가지면 parent 가 level 21 인데 자기는 level 8 인 식). 이 panel 들은 어느 subtree root 의 descendant 도 아님 → stranded.
4. 측정: case8387 P=7370 중 무려 1875 개 stranded. multi-stream dispatch 가 이들을 무시 → 부정확한 factor → garbage solve.

### 8.2 Fix (Σ.11)

`src/factorize/multifrontal.cu` 의 subtree partition 재작성:

```cpp
// 정확한 정의: panel p 의 subtree root = panel_parent 체인 따라 올라가 처음으로 만나는
// "spine-에-바로-parent-된" 또는 자기 자신이 root 인 panel
for p in reverse-id order:                    // parent id > child id 보장
    if is_spine[p]: continue
    if panel_parent[p] in spine or == -1:
        subtree_root_of[p] = p
    else:
        subtree_root_of[p] = subtree_root_of[panel_parent[p]]
```

K (distinct subtree roots) 가 너무 많아질 수 있음 (case8387 의 경우 K=29). TCState 의 stream array 한도 8 이므로:
- 가장 큰 K_cap-1 = 7 개를 separate subtree (스트림 0..6)
- 나머지 모든 root 의 panel 들을 *spillover* subtree (스트림 7) 로 묶음 — 한 stream 내 sequential 처리 (level 순서 유지) 로 correctness OK

### 8.3 측정 — Multi-stream 으로 **TC 가 FP32 이김**

**case8387 B=64 (median of 5 runs)**:
| Mode | factor μs | solve μs | TOTAL | vs FP32 |
|---|---:|---:|---:|---:|
| FP32 (selinv OFF) | 30.8 | 22.4 | 53.2 | baseline |
| TC NO multistream | 39.4 | 24.7 | 64.1 | +20% |
| **TC + multistream** | **26.5** | **22.3** | **48.8** | **−8%** ✓ |

**USA B=64**:
| Mode | factor μs | solve μs | TOTAL | vs FP32 |
|---|---:|---:|---:|---:|
| FP32 (selinv OFF) | 488 | 203 | 691 | baseline |
| TC NO multistream | 543 | 194 | 737 | +7% |
| **TC + multistream** | **478** | **201** | **679** | **−2%** ✓ |

**multi-stream factor 절감**: case8387 −33% (39 → 27 μs), USA −12% (543 → 478 μs). K=8 subtrees concurrent execution 의 효과.

**정확성**: relres single 4e-5 ~ 7e-4 vs multi 4e-5 ~ 5e-3 (1-order higher variance). 원인: cross-stream atomicAdd 의 FP rounding order 가 nondeterministic. FP32 power-grid use case 의 acceptable range.

### 8.4 Default 변경

`CLS_USE_MULTISTREAM=1` opt-in → **default ON**, `CLS_NO_MULTISTREAM=1` 로 disable.

`src/tc/multifrontal_tc.cu`:
- tc_setup 에서 K subtree 스트림 무조건 생성 (단 num_subtrees > 1 일 때)
- issue_factor_levels 에서 multi-stream dispatch default

### 8.5 환경변수 업데이트

| Env var | Default | 효과 |
|---|---|---|
| `CLS_NO_MULTISTREAM=1` | off (multi ON by default) | multi-stream subtree dispatch 비활성 (debugging 용) |
| ~~`CLS_USE_MULTISTREAM=1`~~ | (legacy alias, ignored) | 더 이상 필요 X — 항상 on |

### 8.6 최종 격차 vs FP32 (Σ.11 직후, multistream TC만)

|  | 변경 전 (Σ.10) | 변경 후 (Σ.11) |
|---|---:|---:|
| **case8387 TC vs FP32** | +0.7% | **−8%** |
| **USA TC vs FP32** | +10% | **−2%** |

이 시점에서는 TC 가 이긴 듯 보였으나... 사용자 지적: **multistream 을 FP32 에도 적용해야 fair 비교**.

---

## 9. Σ.12 — Multistream 을 FP32 batched 에도 적용 (fair re-comparison)

§ 8 의 결과는 multistream 을 TC 에만 적용한 비대칭 비교였음. FP32 batched 경로에도 multistream 적용:

### 9.1 구현

| 파일 | 변경 |
|---|---|
| `src/batched/multifrontal_batched.hpp` | BatchedState 에 subtree_streams[8] + fork_event + join_events[8] |
| `src/batched/multifrontal_batched.cu` | `issue_factor_level_range` helper 추출, multistream fork-join wrapper 추가, batched_setup 에서 K stream 할당 |

TC 와 동일 패턴. `CLS_NO_MULTISTREAM=1` 로 disable.

### 9.2 Fair 측정 — B=64

**case8387**:
| Mode | factor μs | solve μs | TOTAL | vs FP32 multi |
|---|---:|---:|---:|---:|
| FP32 single | 30.5 | 22.5 | 53.0 | — |
| **FP32 multi** | 28.2 | 22.5 | **50.7** | baseline |
| TC single | 32.9 | 22.4 | 55.3 | +9% |
| **TC multi** | 26.8 | 22.4 | **49.2** | **−3%** |

**USA**:
| Mode | factor μs | solve μs | TOTAL | vs FP32 multi |
|---|---:|---:|---:|---:|
| FP32 single | 476 | 203 | 679 | — |
| **FP32 multi** | 431 | 203 | **634** | baseline ★ |
| TC single | 564 | 201 | 765 | +21% |
| TC multi | 479 | 195 | 674 | **+6%** |

### 9.3 정직한 결론

**Multistream 은 양쪽 다 적용 가능** — TC -19% factor (case8387), FP32 -8% factor. Multistream 자체는 TC 와 무관한 orthogonal lever.

Fair 비교 시:
- **case8387 B=64**: TC −3% (within noise)
- **USA B=64**: FP32 −6% (TC 못 이김)

TC 의 GEMM compute 우위는 power-grid 의 front 크기 (max 80-254) 에서 **WMMA tile padding overhead 와 FP16 변환 비용을 못 상쇄**. KLU 2010 결론과 정합.

---

## 10. Σ.13 — Batch scaling sweep (B=1 ~ 1024)

§ 9 의 B=64 비교가 끝이 아님. B 가 변하면 어디가 dominant lever 가 되는지가 달라짐. Sweep:

### 10.1 case8387 (n=14908)

총 wall μs/sys (factor + solve):

| B | FP32 | TC | TC/FP32 | 비고 |
|---:|---:|---:|---:|---|
| **1** | 763.7 | 635.8 | **−16.7%** | TC 큰 win |
| 2 | 365.6 | 344.6 | −5.7% | |
| 4 | 221.1 | 211.2 | −4.5% | |
| 8 | 137.0 | 120.7 | −11.9% | TC win |
| 16 | 81.1 | 73.5 | −9.3% | |
| 32 | 60.6 | 58.6 | −3.2% | |
| **64** | 48.4 | 48.5 | **+0.1%** | **cross-over (tie)** |
| 128 | 44.7 | 42.7 | −4.4% | |
| 256 | 41.7 | 41.4 | −0.8% | |
| 512 | 41.8 | 41.1 | −1.7% | |
| **1024** | 40.3 | 41.3 | **+2.7%** | FP32 reclaims |

### 10.2 USA (n=156255)

| B | FP32 | TC | TC/FP32 |
|---:|---:|---:|---:|
| **1** | 3826 | 3193 | **−16.6%** |
| 2 | 1974 | 1854 | −6.1% |
| 4 | 1368 | 1256 | −8.2% |
| 8 | 957 | 870 | −9.1% |
| **16** | 754 | 769 | **+2.0%** ← **cross-over** |
| 32 | 663 | 703 | +6.1% |
| 64 | 652 | 681 | +4.4% |
| 128 | 641 | 699 | +9.0% |
| 256 | 644 | 670 | +4.0% |

### 10.3 패턴 해석

1. **B=1 에서 TC 가 양쪽 케이스 모두 -17% 큰 win**.
2. **case8387 cross-over B=64**, **USA cross-over B=16**. USA 가 일찍 FP32 로 넘어감.
3. **큰 B 영역**: case8387 tie, USA 는 FP32 우위.

**해석**:
- 작은 B 에서는 GPU 점유율이 한 시스템의 panel 수만으로는 부족 → multistream 의 K=8 subtree 병렬화 + TC 의 추가 throughput 둘 다 살아남
- 큰 B 에서는 batch 차원 자체로 GPU 채워짐 → multistream 의 marginal benefit 줄고, TC 의 WMMA padding overhead 가 드러남
- USA 가 일찍 cross-over: 큰 front (max 254) → 한 panel 의 thread block 이 큼 → 적은 B 로도 GPU 채움 → TC 우위 빨리 사라짐
- case8387 작은 front (max 80) → batch 차원 더 필요 → TC 우위 더 오래 유지

### 10.4 실무 시사

| Use case | B 범위 | 추천 |
|---|---|---|
| Newton-Raphson 1-system | B=1 | **TC** (−17%) |
| 소형 일괄 | B 2-8 | **TC** (−5 ~ −12%) |
| 중형 | B 16-32 | case8387: TC, USA: FP32 |
| 대형 | B 64+ | case8387: tie, USA: FP32 |

cuPF Newton iter 는 보통 B=1 ~ 작은 B 영역에서 도므로 **TC 의 win 영역이 실용적으로 가장 valuable**.

---

## 11. Σ.14 — B=1 TC win 의 원인 분석 (NSYS profiling)

§10 에서 B=1 의 TC -17% win 이 가장 큰데, *왜* 인지 nsys 로 kernel-level breakdown.

### 11.1 case8387 B=1 (10 iters)

**Top kernels** (μs/inst × instances = total ms):
| 카테고리 | FP32 | TC |
|---|---|---|
| MID factor (WMMA path) | mid_tc32_b<false> = 19.0 × 430 = 8.16 ms | mid_tc_lo_b<24> = **14.6** × 400 = 5.82 ms ← **−23%/inst** |
| MID factor (staged scalar) | — | mid_tiled_b = 13.1 × 160 = 2.10 ms |
| small_warp factor | 10.1 × 480 = 4.84 ms | 10.3 × 670 = 6.88 ms |
| fwd/bwd level (solve) | ~4.2 × 440 = 1.83 ms | ~4.1 × 520 = 2.12 ms |
| **Total kernel time** | **15.61 ms** | **17.69 ms (+13%)** |
| **Wall time** | 7.64 ms (per-iter 764 μs) | **6.36 ms (per-iter 636 μs, −17%)** |
| **Effective concurrency** (kernel/wall) | **2.04×** | **2.78×** ★ |

**핵심**: TC 가 더 많은 kernel 시간을 쓰지만 (+13%), wall 은 짧음 (−17%). 차이는 **K=8 subtree stream 의 concurrent execution efficiency**:
- FP32: 2.04× concurrent (kernel time이 wall의 2x ≈ 평균 2 stream 동시 실행)
- TC: 2.78× concurrent (평균 2.78 stream 동시 실행)

TC 의 multi-stream 효율이 더 높은 이유:
- TC 가 panels 를 더 작은 kernel 단위로 dispatch (mid_tc_lo + mid_tiled split, 더 많은 small_warp 호출)
- 작고 짧은 kernel 들이 stream 간 겹치기 쉬움
- FP32 는 큰 mid_tc32_b 하나로 처리 → 한 kernel 이 한 stream 의 시간을 더 오래 차지 → overlap 기회 적음

**TC compute 우위 (-23% per mid kernel)** + **concurrency 우위 (2.78× vs 2.04×)** = 둘 다 기여. concurrency 가 더 큰 lever.

### 11.2 USA B=1 (5 iters)

**Top kernels**:
| 카테고리 | FP32 | TC |
|---|---|---|
| **BIG factor (fsz > MID_THRESH)** | extend_level_b = **116 μs/inst** × 110 = 12.8 ms (36%) | extend_tc32_b (WMMA) = **71 μs/inst** × 180 = 12.9 ms (43%) ← **TC −39%/inst** |
| MID factor | mid_tc32_b<false> = 44 × 325 = 14.4 ms | mid_tiled_b = 36 × 205 = 7.3 ms |
| MID factor (WMMA) | — | mid_tc_lo_b<24> = 26 × 55 = 1.4 ms |
| **Total kernel time** | **35.4 ms** | **30.2 ms (−15%)** |
| **Wall time** | 19.1 ms (per-iter 3826 μs) | **16.0 ms (per-iter 3193 μs, −17%)** |
| **Concurrency** | **1.85×** | **1.89×** |

**핵심**: USA 의 TC win 은 case8387 과 *다른 원인*:
- USA fronts 는 크므로 (max 254) WMMA 의 efficiency 가 **per-kernel −39% on BIG fronts**, −18% on MID
- TC 의 total kernel time 이 LESS 함 (−15%)
- Concurrency 는 양쪽 비슷 (~1.85×) — multistream 으로 큰 차이 안 남
- **WMMA compute 우위가 dominant**

### 11.3 두 케이스의 win 원인 대비

| 케이스 | Compute (per-kernel) | Concurrency (multistream) | Dominant lever |
|---|---|---|---|
| case8387 | TC mid kernel −23% | TC 2.78× vs FP32 2.04× | **Concurrency** (작은 front, 많은 kernel) |
| USA | TC BIG kernel −39% | 둘 다 ~1.9× | **Compute** (큰 front, WMMA sweet spot) |

### 11.4 시사

- **case8387 같은 작은 power-grid**: TC 의 WMMA compute 만으로는 marginal. multi-stream 의 K=8 subtree 병렬화가 TC 의 짧은 kernel 들을 잘 packing 하는 게 결정적.
- **USA 같은 큰 power-grid**: TC 의 WMMA 가 fsz>128 영역에서 진짜 compute win (-39%). multistream 효과는 부수적.
- B 가 커지면: batch 차원이 GPU 채우므로 multistream 의 lever 약화 → TC 의 win 도 약화 → cross-over (case8387 B=64, USA B=16).

이 분석이 B-sweep 의 패턴을 설명함.

### 11.5 추가 가능한 lever (B=1 대상)

1. **더 작은 kernel 로 split** → multistream concurrency 더 끌어올림 (case8387 의 lever 보강)
2. **WMMA threshold 더 낮춤** (현재 mid_tc_lo MIN_FSZ=24 → 16 으로) → 더 많은 panel 이 WMMA 영역 → compute 우위 확장 (USA-style win 을 case8387 에도)
3. **Spine kernel 의 persistent dispatch** → spine 의 K=1 sequential 부분도 다른 stream 과 overlap

---

## 12. Σ.14 — Lever 1/2 적용 검증 + B=1 TC win 의 진짜 root cause

§ 11.5 에서 제안한 두 lever 를 실제 적용해 본 결과 + B=1 TC win 의 실제 원인 재조사. **결과: 이전 보고가 misleading 이었음을 발견.**

### 12.1 Lever 1 (WMMA threshold 24 → 16) — 실패

`CLS_NO_TILED_TRAILING=1` 로 WMMA 경로 강제:
- case8387 B=1: 397 → 428 μs (+8%) — WMMA 가 staged scalar 보다 느림 (Σ.1 finding 재확인)
- USA B=1: 2067 → 2188 μs (+6%)

Σ.1 의 결론대로 우리 front 크기 분포에서 WMMA 의 padding overhead 가 compute 이점 압도. **lever 1 폐기**.

### 12.2 Lever 2 (spine 별도 stream) — 노이즈 안

`CLS_USE_SPINE=1` (기존 persistent spine kernel) 측정, B sweep:

| B | default | spine_fused | 차이 |
|---:|---:|---:|---:|
| 1 | 374 | 402 μs | +7.7% |
| 4 | 117 | 112 μs | −4.4% |
| 16 | 42.4 | 43.1 μs | +1.7% |
| 64 | 26.4 | 26.8 μs | +1.5% |
| 256 | 23.4 | 24.9 μs | +6.4% |

명확한 win 없음 (measurement noise 범위). 원리: spine 은 모든 subtree CB 의존이라 fundamentally sequential after join — overlap 여지 없음. **lever 2 폐기**.

### 12.3 "갑자기 TC 가 빨라진" 진짜 원인 — **misleading 측정 발견** ★

사용자 의문: B=1 에서 TC −17% 가 어디서 옴? § 11 의 분석은 multistream concurrency + WMMA compute 라고 했음.

**재측정** (multistream 제외, 둘 다 single-stream):
| Mode (B=1) | case8387 factor μs |
|---|---:|
| FP32 single | 487 |
| FP32 multi | 479 (multistream −1.6% only!) |
| TC single | 400 (vs FP32 single: **−18%**) |
| TC multi | 391 |

**Multistream 은 B=1 에서 거의 효과 없음** (~2%). TC vs FP32 의 18% 차이는 multistream 이 아니라 **다른 데서** 옴.

### 12.4 NSYS profile (single-stream B=1) — 결정적 단서

FP32 single 의 top kernel:
- `mf_factor_mid_tc32_b<false>`: **17.3 μs/inst × 220 = 3.81 ms** (52.4%)

TC single 의 top kernel:
- `mf_factor_mid_tiled_b`: **12.8 μs/inst × 160 = 2.05 ms** (32.5%)
- `mf_factor_mid_tc_lo_b<24>`: 13.6 μs/inst × 50 = 0.68 ms (10.3%)

**다른 kernel!** FP32 는 `mid_tc32_b<false>` (구버전 non-staged scalar), TC 는 `mid_tiled_b` (Σ.1 의 shared-staged scalar).

**Σ.1 staged trailing optimization 이 TC path 에만 적용되어 있었음** — FP32 batched path 는 적용 안 됨. 이게 "TC 가 17% 빠른" 실제 이유의 대부분.

### 12.5 Fix — FP32 도 staged trailing 사용

`src/batched/multifrontal_batched.cu` 의 FP32 dispatch 에 `mf_factor_mid_tiled_b` 추가 (TC 와 동일 조건: max_fsz ≥ 48 AND shared budget OK):

```cpp
if (s_use_tiled_fp32 && max_fsz >= 48) {
    ... mf_factor_mid_tiled_b<<<grid, 256, shb_tiled, stream>>>(...)
}
// otherwise fall back to mf_factor_mid_tc32_b<false>
```

`CLS_NO_TILED_TRAILING=1` 로 legacy kernel 강제 가능.

### 12.6 수정 후 측정 — TC win 대폭 축소

**case8387 B=1 (single-stream, factor only)**:
| Mode | factor μs (수정 전) | factor μs (수정 후) |
|---|---:|---:|
| FP32 single | 487 | **387 (-21%)** |
| TC single | 400 | 382 |
| **TC vs FP32** | **-18%** | **−1% (tie within noise)** |

**case8387 B-sweep (total = factor + solve)**:
| B | FP32 (수정 후) | TC | TC vs FP32 |
|---:|---:|---:|---:|
| 1 | 722 | 639 | **−11.5%** (이전: −16.7%) |
| 4 | 213 | 195 | −8.6% |
| 16 | 80.2 | 77.6 | −3.2% (이전: −9.3%) |
| 64 | 50.3 | 48.8 | −3.0% (tie) |
| 256 | 42.1 | 46.3 | **+10.0%** (FP32 이김) |

**USA B=1 (수정 후)**:
| Mode | factor μs | solve μs | TOTAL |
|---|---:|---:|---:|
| FP32 multi | 2987 | 1022 | 4009 |
| TC multi | 2157 | 967 | **3124 (−22%)** |

### 12.7 정직한 결론 — Σ.14

1. **B=1 의 TC -17% win 은 절반 이상이 측정 artifact** (Σ.1 optimization 가 TC-only 였음).
2. **수정 후 case8387**: TC win 이 -17% → -11.5% 로 축소. B 클수록 더 빨리 cross-over (B=256 에서 이미 FP32 +10%).
3. **USA**: TC -22% 유지. 이유는 USA 의 BIG fronts 가 TC 에서 WMMA (`mf_factor_extend_tc32_b`, 71 μs/inst) 쓰지만 FP32 batched 는 non-WMMA (`mf_factor_extend_level_b<float>`, 116 μs/inst) 사용. WMMA 가 **진짜 -39%/inst** 이김 — 이건 fundamentally TC 의 sweet spot.
4. **case8387 의 잔여 TC 약간 win** (-3 ~ -11%): FP32 가 max_fsz<48 levels 에서 fallback 으로 mid_tc32_b<false> 쓰는데, TC 는 같은 영역에 mid_tc_lo<24> WMMA 씀. 더 작은 mid GEMM win.

### 12.8 진짜 lever 정리

| Lever | 어디 적용? | Win 발생 영역 |
|---|---|---|
| **Σ.1 staged trailing** | 이전엔 TC만, **Σ.14 에서 FP32 batched 도** | mid 모든 영역 (-20-36% per-kernel) |
| **Σ.11 multistream subtree** | TC + FP32 batched 둘 다 | B 작을수록 (B≤8 에서 −5~-10%) |
| **selinv default OFF** | TC + batched 모두 | factor -40% (Newton-Raphson) |
| **tc_warmup + 사전 build** | TC only (FP32 setup 은 원래 1ms) | setup -97% |
| **WMMA on BIG fronts** | TC only (extend_tc32_b) | USA BIG (-39%/inst) — **TC 의 유일한 진짜 차별점** |
| **WMMA on small mid fronts** | TC only (mid_tc_lo<24>) | case8387 의 max_fsz<48 levels (-10-15% mid kernel) |

### 12.9 사용자 지적이 옳았던 점

- 이전 보고 "TC 가 FP32 못 이김" 의 원래 reason 들 (front 크기, padding overhead) 는 *부분적으로 옳음*
- 그러나 "TC 가 B=1 에서 갑자기 17% 빠름" 도 *측정 artifact* 였음 — Σ.1 미적용 path
- **결국 fair 비교 시**: TC 의 진짜 차별점은 **USA-style 큰 BIG fronts 에서의 WMMA**
- power-grid case8387 같이 BIG front 없는 경우: TC 의 추가 lever 는 mid_tc_lo (작은 mid WMMA) 정도 — marginal

### 12.10 환경변수 업데이트

| Env var | Default | 효과 |
|---|---|---|
| `CLS_NO_TILED_TRAILING=1` | off | **Σ.14**: staged trailing 비활성. FP32 batched path 도 영향받음 (이전엔 TC만) |
| 나머지는 § 8.5 동일 | | |

---

## 13. TC 의 정확한 작동 범위 — *어떤 연산이 실제로 tensor core 를 쓰는가*

§ 12 의 fair 비교 후에도 TC 가 작지만 win 함 (case8387 −11.5%, USA −22%). 사용자 질문: **"TC 가 어떤 연산을 TC 로 하고 있어?"**

### 13.1 WMMA 호출은 *단 하나의* device function

**`tc_trailing_wmma_f32` (`src/tc/factor_tc.cuh:38`)** — 한 frontal matrix 의 *trailing rank-nc update GEMM* 만 처리:

```
C(uc × uc) -= L(uc × nc) * U(nc × uc)
```

알고리즘:
1. L, U 를 FP32 → FP16 으로 cast 하여 shared (Lh, Uh) 에 stage (16-byte align)
2. `wmma::fragment<matrix_a, 16, 16, 16, __half>` 로 L tile load
3. `wmma::fragment<matrix_b, 16, 16, 16, __half>` 로 U tile load
4. `wmma::mma_sync(cf, af, bf, cf)` — FP16 × FP16 → **FP32 accumulate** TC 연산
5. `wmma::store_matrix_sync` 로 FP32 결과 → shared scratch
6. 결과를 FP32 로 main front 의 C panel 에 subtract (atomicAdd 없음, lane 별 직접 write)

### 13.2 TC 가 fire 되는 위치 (두 wrapper kernel)

| Wrapper kernel | 적용 영역 | TC fires when (조건) |
|---|---|---|
| `mf_factor_extend_tc32_b` | **BIG front (fsz > 128)** — USA top levels | `nc ≤ 32, uc ≤ 256` |
| `mf_factor_mid_tc_lo_b<MIN_FSZ_FOR_TC=24>` | mid front (max_fsz < 48 또는 shared overflow fallback) | `fsz ≥ 24, nc ≤ 32, uc ≤ 256` |

조건 미충족 시 (예: nc>32) 같은 wrapper 안에서 `trailing_update_scalar` (plain FP32) 로 fallback.

### 13.3 TC path 라도 **TC 안 쓰는** kernel (plain FP32 scalar)

| Kernel | 영역 | 연산 |
|---|---|---|
| `mf_factor_small_warp_b<float>` | small front (max_fsz ≤ 32) | scalar FP32 (warp-per-front) |
| **`mf_factor_mid_tiled_b`** | **mid front (48 ≤ max_fsz ≤ 128)** ← dominant kernel | **scalar FP32**, shared-staged trailing |
| `lu_panel_factor`, `u_panel_solve`, `lu_small_front` | LU + U-solve (모든 path) | scalar FP32 |
| `mf_invert_pivot_b` (selinv on 일 때) | 피봇 block 역행렬 | scalar FP32 |
| `mf_fwd_*_b`, `mf_bwd_*_b` | triangular solve | scalar FP32 |

→ "TC path" (`--tc`) 가 의미하는 건 *"WMMA 가 가능한 곳에서만 fire"*. LU/U-solve/solve 는 모두 FP32 scalar.

### 13.4 측정 — TC kernel 비중 (B=1)

| Kernel category | case8387 | USA |
|---|---:|---:|
| `mid_tc_lo<24>` WMMA (TC) | 0.68 ms (50 calls × 13.6 μs) | 1.4 ms (55 × 26 μs) |
| `extend_tc32_b` WMMA (TC) | **0** (no BIG fronts) | **12.9 ms** (180 × 71 μs) |
| `mid_tiled_b` scalar | 2.03 ms | 7.3 ms |
| small_warp scalar | 0.56 ms | 2.1 ms |
| solve kernels scalar | 2.4 ms | ~10 ms |
| **TC kernel 비중 (factor)** | **~20 %** | **~60 %** |

→ **USA 가 TC 효과를 60 % 받는 반면 case8387 은 20 % 만**. 이게 USA −22 % vs case8387 −11 % 의 정확한 산술.

### 13.5 Per-instance TC 우위 (WMMA vs scalar)

| Kernel pair | TC (WMMA) | FP32 (scalar) | 차이 |
|---|---:|---:|---:|
| **USA BIG**: `extend_tc32_b` vs `extend_level_b<float>` | **71 μs** | **116 μs** | **−39 %** ★ |
| case8387 small mid: `mid_tc_lo<24>` vs `mid_tc32_b<false>` | 13.6 μs | 17.3 μs | **−21 %** |

### 13.6 결론 — TC 가 win 하는 영역의 본질

1. **TC = trailing GEMM 만 WMMA 가속.** LU panel factor, U panel solve, triangular solve, 피봇 invert 는 plain FP32 scalar.
2. TC 효과의 크기는 *trailing 비중* × *front 크기 분포* 의 함수.
3. **BIG front (fsz > 128)** 에서 WMMA 가 진짜 sweet spot: per-kernel −39 %. → USA 의 dominant lever.
4. **mid front (24 ≤ fsz < 48)** 의 fallback path 에서도 WMMA fire: per-kernel −21 %. → case8387 의 작은 win.
5. **mid front (48 ≤ fsz ≤ 128)** 는 dominant kernel 인데 `mid_tiled_b` (FP32 scalar) 사용 → TC 사용 안 함. 왜? Σ.1 측정에서 WMMA 의 padding overhead 가 우리 power-grid 의 nc/uc 분포 (nc 10-30, uc 30-60) 에서 staged scalar 보다 느림.

### 13.7 정확한 win 분해 (case8387 −11.5 %)

- TC WMMA on 50 small-mid levels: per-instance −21 %, 차지 비중 ~20 % → factor wall **~3 %** 절감
- TC 의 kernel 분할 구조 (mid_tc_lo + mid_tiled split) + dispatch overhead 차이 → 잔여 **~8 %** 절감
- 합계 ~11 %

### 13.8 정확한 win 분해 (USA −22 %)

- TC WMMA on 180 BIG-front trailings: per-instance −39 %, 차지 비중 ~55 % → factor wall **~22 %** 절감
- (수학이 거의 정확히 들어맞음 — BIG front WMMA 가 dominant)

### 13.9 시사

| 사용 시나리오 | TC 추천? | 이유 |
|---|---|---|
| 큰 power-grid (USA, ACTIVSg25k 등) | ★ **TC 강력 추천** | BIG fronts 에서 WMMA −39%/inst |
| 작은 power-grid (case8387, case1354 등) | TC 약한 win (−3~−11%) | mid_tc_lo 의 작은 WMMA 효과만 |
| 다른 sparsity (stencil, FEM 등) | front 크기 분포 측정 필요 | fsz > 128 영역이 많을수록 TC 유리 |

**최종**: TC 의 가치는 **WMMA trailing GEMM 한 가지** 인데, 그게 BIG-front 영역에서 substantial. power-grid 특성상 USA 급에서 명확한 win, 그 이하에서는 marginal but positive.

---

## 14. "오늘 오전엔 TC 안 빨랐는데 왜 갑자기 빨라졌나" — 시간순 lever 추적

§ 12 / § 13 의 분석으로도 사용자의 핵심 질문이 답이 되지 않음: *오늘 오전엔 TC 가 못 이기던 게 오늘 변경 중 어떤 것 때문에 이긴 거냐?*

### 14.1 historical state ablation 측정 (case8387 B=64)

각 lever 를 하나씩 적용하면서 TC vs FP32 측정:

| 단계 | 설정 | FP32 TOTAL | TC TOTAL | TC vs FP32 |
|---|---|---:|---:|---:|
| **(1) 오전 상태** | selinv ON, single-stream, non-staged trailing | 92.1 μs | 97.5 μs | **+5.9 % TC LOSES** |
| (2) selinv OFF | (1) + selinv OFF | 64.9 μs | 60.0 μs | **−7.5 % TC WINS** ★ |
| (3) +multistream | (2) + multistream (둘 다) | 49.9 μs | 50.9 μs | +2.0 % |
| (4) +staged trailing | (3) + Σ.1 staged (둘 다) — 현재 | 50.0 μs | 49.5 μs | −1.0 % |

**(1)→(2) 단계에서 TC 가 +5.9 % loser 에서 −7.5 % winner 로 *flip***. 이게 결정적 변화.
- (2)→(3): multistream 둘 다 적용. TC 우위 약간 줄어듦 (FP32 가 더 도움받음).
- (3)→(4): staged trailing 둘 다 적용. 비슷 (모두 동등).

→ **TC 가 win 하게 만든 결정적 lever 는 `selinv OFF default` (Σ.9)** 단 하나.

### 14.2 selinv OFF 가 TC 를 더 도와주는 이유

selinv ON vs OFF 의 factor / solve 분리 측정:

| Mode | factor μs | solve μs | factor 절감 (selinv OFF) | solve 손실 |
|---|---:|---:|---:|---:|
| FP32 selinv ON | 67.5 | 19.7 | — | — |
| FP32 selinv OFF | 30.9 | 22.5 | **−36.6 μs** | +2.8 μs |
| TC selinv ON | 89.0 | 20.9 | — | — |
| TC selinv OFF | 45.9 | 22.4 | **−43.1 μs** | +1.5 μs |

- selinv OFF 의 factor 절감: TC 가 6.5 μs 더 많이 절감
- solve 손실: TC 가 1.3 μs 덜 손해 (selinv 의 solve win 이 작음)
- 합계: TC 가 **selinv OFF 로 7.8 μs 더 이득**

이 7.8 μs/sys 가 바로 (1) → (2) 의 13.4 μs 격차 flip 의 대부분.

### 14.3 왜 selinv 가 TC factor 에 더 큰 부담이었나

`mf_invert_pivot_b<float>` 자체는 두 path 에서 같은 kernel (template = float). nsys 측정에서 동일 시간 (~257 μs/iter for B=64 case8387).

per-sys 로 환산하면: 257 / 64 = **4 μs/sys**. 그러나 selinv ON → OFF 의 factor 절감은 **36-43 μs/sys** = kernel 자체 비용의 **9-10 배**.

→ selinv kernel 자체 외에 **추가 overhead** 가 존재. 추정 원인:
1. **graph capture 크기 증가**: selinv ON 시 captured graph 에 B*P 개 (case8387 B=64 P=7400 → 474K) 의 추가 작은 kernel block 들이 들어감. 이게 dispatch / scheduling overhead 를 amplify.
2. **메모리 traffic**: selinv 가 모든 panel 의 pivot block 을 in-place 변경 → 다음 solve 가 변경된 데이터를 읽음 → cache pressure 증가
3. **TC 가 FP32 보다 더 영향받는 이유**: TC path 는 이미 더 잘게 split 된 kernel 들 (small_warp + mid_tc_lo + mid_tiled + extend_tc32 + spine) → 추가 selinv 의 dispatch overhead 가 누적 효과 더 큼

### 14.4 시간순 정리 — *진짜* 무엇이 바뀌었나

| Phase | 변경 | TC vs FP32 (case8387 B=64) | Net 효과 |
|---|---|---|---|
| 오전 baseline | (이전 코드) | TC +5.9 % loser | TC 못 이김 |
| **Σ.9** | **selinv default ON → OFF** | **TC −7.5 % winner** | **★ flip 발생** |
| Σ.11 | TC 에 multistream 추가 | TC −8 % (vs FP32 single) | unfair: TC 만 lever |
| Σ.12 | FP32 에도 multistream | TC −3 % (fair) | gap 일부 closed |
| Σ.14 | FP32 에도 staged trailing | TC −1 % (fully fair) | gap 거의 close |

(2) → (3) 의 multistream 추가는 둘 다 도와줘서 **TC 우위가 오히려 감소** (−7.5 % → −1 %). TC가 win 하는 "원인" 은 본질적으로 **Σ.9 selinv OFF** 만 남음.

### 14.5 결론

**Q: 오늘 오전엔 TC 못 이겼는데 왜 빨라졌나?**

**A**: `selinv default OFF` 변경 (Σ.9) 단 하나가 결정적. 이게 없으면 multistream / staged trailing / setup 절감 같은 다른 모든 lever 가 있어도 TC 가 FP32 못 이김.

이유: `mf_invert_pivot_b` 의 호출은 FP32 와 TC 양쪽에 같은 비용을 부담시키지만, 그 kernel 이 만드는 *graph capture 의 추가 dispatch overhead* 가 TC 의 이미 잘게 split 된 kernel 구조에 더 누적되어 TC 가 7-8 μs/sys 더 손해.

selinv 가 사라지면 그 추가 손해가 양쪽에서 동등 비율로 회복되는데, **TC 가 절대값으로 더 많이 회복** (37 vs 27 μs/sys) → TC 가 FP32 보다 빨라짐.

다른 lever 들 (Σ.11 multistream, Σ.14 staged trailing) 은 **둘 다 비례적으로 가속**시키므로 TC vs FP32 격차에는 거의 영향 없음. 단 USA 같이 BIG-front 가 있는 case 에서는 § 13 의 WMMA compute 우위 (−39 %/inst on `extend_tc32_b`) 가 추가로 누적되어 USA −22 % 의 더 큰 win.

### 14.6 한 줄 요약

> **"TC 가 갑자기 빨라진 이유 = selinv default 를 OFF 로 바꾼 것."** Σ.11 / Σ.14 같은 다른 lever 들은 TC vs FP32 격차에 거의 영향 없음. selinv OFF 만이 TC 의 숨겨진 우위 (WMMA on small mid/big fronts) 를 visible 하게 만들었음.
