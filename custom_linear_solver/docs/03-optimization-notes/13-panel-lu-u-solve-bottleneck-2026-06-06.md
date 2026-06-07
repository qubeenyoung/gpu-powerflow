# mid / big tier의 panel LU + U-solve 병목 분석 + 개선안

**작성일**: 2026-06-06
**대상**: refactored `custom_linear_solver` (T4.2.A row-fused panel LU + cp.async stage-in 적용 후)
**관련 소스**: `src/factorize/phases.cuh` 의 `lu_panel_factor`, `u_panel_solve` (옛 `primitives.cuh` 위치; 2026-06-06 리팩토링으로 통합됨)
**선행**:
- [`docs/12 §9, §10`](../04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md): tier 구조, ncu 병목
- [`docs/10 §8-§9`](10-t4.1-t4.3-results-2026-06-06.md): T4.2.A row-fused panel LU 결과, sync→wall 메타-결론
- [`docs/11 §6.2`](../04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md): 비-GEMM wall 분해

## 0. TL;DR

- **Phase 1 (panel LU)**: case8387 nc=8 (T4.2.A row-fused) → 8 syncs/front, USA nc=20 (split form) → **40 syncs/front**.
- **Phase 2 (U-solve)**: 모든 케이스 nc-1 syncs/front (case8387 7, USA 19).
- **B=64 mid 의 wait/barrier stall 30~45%** 이 panel LU + U-solve의 syncs + thread under-utilization 에서 비롯됨 (docs/12 §10.2).
- **5개 개선안 (P1-P5)**, **P1 (reciprocal multiply)** 만 실측, **USA B=1 −2%** small win.
- P2-P5 (phase fusion, parallel U-solve, bank-conflict pad, warp-spec)은 deferred — 예상 효과 vs 구현 부담 trade-off 평가.

## 1. Phase 정의와 코드 분석

### 1.1 lu_panel_factor (Phase 1, panel LU)

```cpp
// T4.2.A row-fused (nc ≤ 12): 1 sync/k
for (k = 0; k < nc; ++k) {
    piv = F[k*fsz + k];
    for (i = k+1+t; i < fsz; i += nt) {
        lik = F[i*fsz + k] / piv;
        F[i*fsz + k] = lik;
        for (jj = k+1; jj < nc; ++jj) F[i*fsz + jj] -= lik * F[k*fsz + jj];
    }
    __syncthreads();
}
// split form (nc > 12): 2 syncs/k
for (k = 0; k < nc; ++k) {
    piv = F[k*fsz + k];
    for (i = k+1+t; i < fsz; i += nt) F[i*fsz + k] /= piv;     // divide
    __syncthreads();                                            // sync #1
    pc = nc - 1 - k;
    for (e = t; e < (fsz-k-1)*pc; e += nt) {
        ii = k+1+e/pc; jj = k+1+e%pc;
        F[ii*fsz + jj] -= F[ii*fsz + k] * F[k*fsz + jj];        // rank-1 update
    }
    if (pc > 0) __syncthreads();                                // sync #2
}
```

**Sync 수**:
- row-fused (nc ≤ 12): **nc syncs**
- split form (nc > 12): **2·nc syncs**

case8387 (mid, nc=8): row-fused → 8 syncs.
USA (mid, nc=20): split → 40 syncs.

### 1.2 u_panel_solve (Phase 2, U triangular solve)

```cpp
for (k = 1; k < nc; ++k) {
    for (e = t; e < uc; e += nt) {
        jj = nc + e;
        v = F[k*fsz + jj];
        for (i = 0; i < k; ++i) v -= F[k*fsz + i] * F[i*fsz + jj];  // 순차 의존
        F[k*fsz + jj] = v;
    }
    __syncthreads();
}
```

**Sync 수**: **nc-1 syncs** (case8387 mid: 7, USA mid: 19).

**Thread 활용**: 매 k 단계에서 **uc threads만 활성** (256 thread 중 uc개). USA mid uc ≤ 108 → 108개 active, 148개 idle. case8387 mid uc ≤ 71 → 71개 active, 185개 idle.

### 1.3 종합 sync count + 작업량

| 케이스 | nc | uc max | Phase1 syncs | Phase2 syncs | trailing syncs | **합** |
|--------|---:|-------:|-------------:|-------------:|---------------:|------:|
| case8387 mid | 8 | 71 | 8 | 7 | 2 | **17** |
| USA mid | 20 | 108 | 40 | 19 | 2 | **61** |
| USA big | 20 | 215 | 40 | 19 | 2+ | **61+** |

이론적 FLOPs (per front):

| 케이스 | Phase1 (~nc·fsz²/2) | Phase2 (~uc·nc²/2) | Trailing (~uc²·nc) |
|--------|-------------------:|-------------------:|-------------------:|
| case8387 mid (fsz=43, nc=8) | 7.4 k | 2.3 k | 11.8 k |
| USA mid (fsz=51, nc=14) | 18 k | 5.6 k | 20 k |
| USA big (fsz=155, nc=20) | 240 k | 26 k | 376 k |

→ **Phase 1 (panel LU)** 이 작업량 측면에서 Phase 2 보다 3-10x 큼.
→ Phase 2 는 작업량 작지만 sync 수는 Phase 1 의 절반 (split form에선 1/4) — **sync 밀도 가장 높음**.

## 2. ncu 기반 병목 측정

docs/12 §10.2 의 ncu (factor_mid<float>, B=64):

| 케이스 | SOL_SM | Occ% | 1순위 stall | 2순위 |
|--------|-------:|----:|------------|------|
| case8387 mid B=64 | 47% | 70% | **barrier 41%** | wait |
| USA mid B=64 | 43% | 45% | **wait 31%** | barrier 26% |

**병목 분류**:
- case8387 (nc=8): **barrier-bound** — 17 syncs 중 panel LU 의 8 + U-solve 의 7 = **15 / 17 syncs (88%) 가 Phase 1+2 에서 발생**
- USA (nc=20): **wait-bound (FMA latency)** — 61 syncs 중 panel LU 40 + U-solve 19 = **59 / 61 syncs (97%) 가 Phase 1+2**. wait stall 은 U-solve 의 thread under-utilization 으로 인한 instruction-level latency 노출도 포함

→ **Panel LU + U-solve = factor_mid 의 sync 비용 88-97% 의 출처**

## 3. 코드 레벨 비효율 항목

### 3.1 divide 비용 (Phase 1)

```cpp
F[i*fsz + k] /= piv;
```

Ampere FDIV cycle cost:
- `__fdiv_rn(x, y)`: **22 cyc** (실측 throughput, sm_86)
- `__frcp_rn(y) * x`: **5 + 1 = 6 cyc** (RCP + FMUL)

→ 4x potential speedup per divide. nc divides × (fsz-k) entries × nc fronts per level → real wall.

**미스 이유**: 컴파일러가 shared memory 의 `piv` 를 register hoist 못 할 수 있음 (aliasing 가능성 보수). 명시적 `inv_piv = 1/piv` 가 안전.

### 3.2 thread under-utilization (Phase 2)

```cpp
for (e = t; e < uc; e += nt) { ... }  // uc < nt 라면 idle threads
```

USA mid k=1 단계 (uc=108, nt=256): 108 thread active, 148 idle = **42% utilization**. case8387 mid (uc=71, nt=256): 28% utilization.

매 k 단계에서 idle thread 가 wait stall 증가 (warp scheduler가 다른 warp 의 active work 발견 못 함).

### 3.3 bank conflict 가능성 (Phase 1)

`F[i*fsz + k] /= piv`: lane i (i=k+1+t, t∈[0,32)) 가 stride `fsz` 로 column k 쓰기.

- fsz = 32 or 64: bank = (i·fsz + k) mod 32 = k mod 32 → **32-way bank conflict**
- fsz = 33, 50, 79 등 (case8387 mid): bank = (k + i·fsz) mod 32 = lane-varying → no conflict

→ **fsz=32 또는 64 인 mid 레벨 에서 column writes 가 32-way bank conflict**. case8387 mid: fsz max 79 → 영향 작음. USA mid 분포: fsz=32, 40, 48, 56, 64, ... 균등 분포라 ~12% 의 mid front 가 fsz=64 에서 영향.

### 3.4 sync 순서 제약 (전체)

Phase 1 → Phase 2 → Phase 3 (trailing) 사이의 sync 2번 (writeback 시 추가). Phase 2 종료 후 즉시 trailing 시작 가능하지만, 현 코드 는 각 Phase 별로 완전 분리 → sync 의존성으로 Phase 간 overlap 불가.

### 3.5 reduction in U-solve inner loop

```cpp
v = F[k*fsz + jj];
for (i = 0; i < k; ++i) v -= F[k*fsz + i] * F[i*fsz + jj];
```

각 thread 가 k번 직렬 FMA. k=19 (USA mid 마지막 step) 면 19 직렬 FMA = 19 cyc latency. wait stall 의 직접 원인.

## 4. 개선안 (P1 ~ P5)

### P1 — Reciprocal multiply in panel LU divide ⭐ 실험됨

**변경**: `F /= piv` → `inv_piv = 1/piv; F *= inv_piv`. 컴파일러가 자동 hoist 못 하는 경우 명시적 hoist.

**구현**: 7 줄 변경. CLS_NO_RECIP_PIV macro 로 toggle.

**기대**: per-divide 22 cyc → 6 cyc, 4x 단축. Phase 1 wall ~20% 단축, factor_mid 전체 wall ~5-10% 단축.

**실측** (graph mode, 7-run median):

| case | B | OFF | ON | delta |
|------|--:|----:|---:|------:|
| case30, 118 | – | – | – | no-op |
| case8387 | 1 | 0.346 | 0.347 | noise |
| case8387 | 64 | 0.0268 | 0.0266 | **−0.7%** |
| USA | 1 | 2.72 | 2.66 | **−2.0%** |
| USA | 64 | 0.469 | 0.466 | −0.6% |

**해석**: 기대보다 작음. 이유:
- 컴파일러가 일부 케이스에서 이미 자동 hoist 중 (특히 register 압력 낮은 경우)
- divide 가 panel LU wall 의 일부만 차지 (sync 가 더 dominant)
- 그래도 USA B=1 **−2%** 일관 단축 → **default ON으로 ship**.

### P2 — Phase 1 + Phase 2 fusion ⭐⭐ 큰 잠재

**아이디어**: 각 k 반복 안에서 panel LU step k 완료 직후 U-solve step k 도 함께 수행. 두 phase 간 sync 제거.

**기여 의존성**:
- Phase 1 step k 가 수정하는 영역: rows [k+1, fsz), cols [0, nc) (L panel + 일부 U panel)
- Phase 2 step k 가 수정하는 영역: row k, cols [nc, fsz) (U trailing row)
- **두 영역 disjoint** → 같은 k iter 안에서 동시 수행 가능

**제안 구조**:
```cpp
for (k = 0; k < nc; ++k) {
    // Phase 1 step k: divide + panel update
    ...
    __syncthreads();
    // Phase 2 step k (only if k > 0): U-solve row k
    if (k > 0) for (e = t; e < uc; e += nt) F[k*fsz + nc+e] -= ...
    __syncthreads();
}
```

→ sync 수: split form (nc>12) 의 경우 2·nc + (nc-1) → **2·nc + (nc-1)** 그대로지만, **2개 phase 동기화 비용을 1개 sync 에 통합**.

더 적극적: panel LU + U-solve 모두 하나의 통합 k-loop 로:
```cpp
for (k = 0; k < nc; ++k) {
    // 1) divide col k          (thread group A)
    // 2) U-solve row k         (thread group B, only k>0)
    // 3) panel update          (thread group C, on all threads)
    __syncthreads();  // single sync per k
}
```

이러면 sync 수 = **nc 만**. USA nc=20: 61 → **20 syncs (−67%)**.

**기대**: USA mid wall **−10~−15%** (barrier+wait stall 의 큰 폭 감소).

**위험**: medium. thread group 분리 시 occupancy 손해 가능. 정확성 검증 필요.

**구현 부담**: 50-100 줄. 다양한 nc 케이스 처리.

→ **deferred** (큰 잠재 but 위험). 우선순위 2.

### P3 — Parallelize U-solve over (i, jj) ⭐ 중간 잠재

**문제**: u_panel_solve 의 outer e-loop 가 uc threads만 활성 (nt=256 중 uc<108 active).

**아이디어**: inner i-loop 도 thread-parallel 화. step k 의 work = uc × k → uc·k threads 분배.

```cpp
for (k = 1; k < nc; ++k) {
    // Stage A: accumulate partial sums in shared
    __shared__ T tmp[uc][warps_per_block];
    for (e = t; e < uc * k; e += nt) {
        jj = e / k; i = e % k;
        atomicAdd(&tmp[jj][warp], F[k*fsz + i] * F[i*fsz + (nc+jj)]);
    }
    __syncthreads();
    // Stage B: write back
    for (e = t; e < uc; e += nt) {
        jj = e;
        T v = F[k*fsz + nc+jj];
        for (w = 0; w < warps_per_block; ++w) v -= tmp[jj][w];
        F[k*fsz + nc+jj] = v;
    }
    __syncthreads();
}
```

→ **2 sync/k 로 늘어남** (현 1/k). 그러나 thread utilization 100% 가까이.

trade-off: utilization 100% 이득 vs sync 2x. step k 의 work = uc·k FMAs. nt=256 일 때:
- 현: uc thread × k FMA 직렬 → 약 k cyc latency
- 제안: 256 thread × (uc·k/256) FMA → 약 max(1, uc·k/256) cyc latency

uc=108, k=19 (USA 최악): 현 19 cyc, 제안 max(1, 108·19/256)=8 cyc. 2.4x speedup per step.

그러나 sync 비용 2x. 8 cyc · 19 step + 19 syncs · 8cyc = ~300 cyc vs 19·19 + 19·8 = ~510 cyc. 약 1.7x 개선.

**기대**: USA mid U-solve wall ~30% 단축 → factor_mid wall ~3-5% 단축.

**구현 부담**: 100-200 줄. shared scratch 추가 ~uc·warps·4 byte.

→ deferred. 우선순위 3.

### P4 — Bank-conflict avoidance via shared padding

**문제**: §3.3 의 fsz=32 또는 64 인 mid level 의 column writes 32-way bank conflict.

**아이디어**: shared 의 leading dimension 을 fsz + padding 으로. 예: `shared_ld = fsz + (fsz % 32 == 0 ? 1 : 0)`. address 계산 시 `Fs[i*shared_ld + j]`.

**기대**: 영향 받는 mid level (USA 의 fsz=32, 64 약 25%) 에서 column write 32x 가속 → 해당 level wall ~10% 단축. USA mid 전체 wall ~3% 단축.

**위험**: 낮음. 표준 CUDA 최적화 패턴.

**구현 부담**: 30-50 줄. 모든 shared address 계산 변경. dynamic shared size 도 padding 반영.

→ ROI 작아 deferred. 우선순위 4.

### P5 — Warp-specialized panel LU (lookahead)

**아이디어**: nc 의 직렬 의존성을 lookahead 로 완화. warp 0 가 step k 처리하는 동안 warp 1 이 step k+1 의 piv pre-read + 의존성 없는 work 시작.

**기대**: 직렬 latency 의 일부 hide. 그러나 step 간 의존성 (k+1 의 piv 가 step k 의 panel update 결과) 이 강해 lookahead 가능 영역 작음.

**구현 부담**: 큼 (300+ 줄). docs/10 §9 메타-결론에 따르면 warp specialization 시도 시 occupancy 손실 위험. 

→ docs/10 의 negative result (warp-per-front mid 가 occupancy 추락) 와 같은 함정 위험. 우선순위 5 (최후).

## 5. 우선순위와 ROI 매트릭스

| 제안 | 코드 변경 | 위험 | 예상 wall 효과 | 실측 | 결정 |
|------|----------|------|----------------|------|------|
| **P1 reciprocal mul** | 7 줄 | 낮음 | -5~-10% Phase 1 | **−0.6~−2%** factor wall | **ship default ON** |
| P2 Phase 1+2 fusion | 50-100 줄 | 중간 | -10~-15% factor wall | – | deferred, R&D |
| P3 parallel U-solve | 100-200 줄 | 중간 | -3~-5% factor wall | – | deferred, ROI 평가 |
| P4 bank conflict pad | 30-50 줄 | 낮음 | -2~-3% factor wall | – | deferred, 부수 ROI |
| P5 warp-spec panel LU | 300+ 줄 | 큼 | -5~-10% (uncertain) | – | last resort |

## 6. 메타-결론 — 본 분석의 가치

docs/10 §9 메타-결론에 따르면 sync 절감이 wall에 비례 변환되지 않음. 본 분석의 P1-P5 도 같은 ceiling 영향:
- P1 (divide 4x 가속): wall 단축 ~2% — 컴파일러가 일부 자동 처리
- P2 (sync 60% 절감): 예상 wall −10~15% but unverified — 같은 ceiling 직면 가능
- P3 (utilization 100%): wall 단축 expected 3-5%, occupancy ↑ 효과는 docs/13 multi-stream 와 trade-off

가장 큰 잠재 P2 (phase fusion) 가 docs/10 의 row-fused panel LU 의 자연 후속. row-fused 가 panel LU 단계 안에서 sync 를 nc→nc/2 줄였다면, P2 는 panel LU + U-solve 사이의 phase boundary 까지 제거.

**그러나 docs/10 §9 의 학습**: 알고리즘 수준 sync 줄이기보다 **dispatch 정확도 (docs/12 T-split), multi-stream concurrent execution (docs/13)** 이 wall ROI 더 큼. 본 P1-P5 는 mid/big tier 정밀 튜닝 용이지 메인 lever 아님.

## 7. 권장 다음 단계

1. **P1 ship** (default ON, CLS_NO_RECIP_PIV=1 로 toggle 가능). 작은 안정 win.
2. **P4 (bank conflict pad)**: 작은 코드 변경 — 차후 별도 PR.
3. **P2 (phase fusion)**: 별도 R&D 사이클. 구현 + 5+ case 검증 (정확성, wall).
4. **다른 lever 우선**: TF32 trailing GEMM (이미 ship), T-split (docs/12, opt-in), multi-stream (docs/13, default ON).

본 솔버의 mid/big tier 최적화 ROI 가 작아진 시점 — 추가 wall 단축 lever 는 GEMM (TF32, grouped) 또는 solve/scatter phase 로 이동 권장.

## 8. 측정 재현

```bash
# P1 A/B (CLS_NO_RECIP_PIV macro)
cmake -B build_recip -DCMAKE_CUDA_FLAGS="" ...        # P1 ON (default)
cmake -B build_norecip -DCMAKE_CUDA_FLAGS="-DCLS_NO_RECIP_PIV=1" ...
# 7-run median per case (case8387/USA × B=1/64) → docs §4 P1 표
```

원본 데이터:
- `/home/claude/build_cls_refactor/` (P1 ON), `/home/claude/build_cls_norecip/` (P1 OFF)
- `/home/claude/prof/` — ncu, nsys raw (docs/12, docs/13 와 공유)

## 9. 관련 문서

- panel LU sync 줄이기 history: [`docs/10 §8`](10-t4.1-t4.3-results-2026-06-06.md) T4.2.A row-fused
- factor_mid 전체 병목: [`docs/12 §9-§10`](../04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md)
- mid/big dispatch 임계 근거: [`docs/02-design-analysis/06`](../02-design-analysis/06-tier-threshold-rationale-2026-06-06.md)
- T-split (dispatch 정확도): [`archive/12-tier-split`](archive/12-tier-split-experiment-2026-06-06.deprecated.md)
- multi-stream 메커니즘: [`docs/13`](../04-benchmarks-profiling/13-multistream-tier-impact-2026-06-06.md)
