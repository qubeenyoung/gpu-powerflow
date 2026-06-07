# factor_mid_opt 실험 — P1+P2(+P4) 결합 별도 커널

**작성일**: 2026-06-06
**상태**: 실험 완료, **별도 kernel rollback (2026-06-06)**. P1 (reciprocal multiply) 만 default kernel 에 흡수 유지.
**선행 분석**: [`docs/13`](13-panel-lu-u-solve-bottleneck-2026-06-06.md) P1-P5 제안
**옛 커널 위치**: `src/factorize/mid_opt.cuh` → **`deprecated/mid_opt/mid_opt.cuh`**
**옛 Toggle**: `CLS_USE_OPT_MID=1` (이제 dispatch 안 됨)
**Rollback 사유**: USA B≥16 −1~−4% 만으로는 별도 코드 path 유지 비용을 정당화 못함. 본 문서는 "sync −64% vs wall −1~−4%" 변환률을 보여주는 evidence 로 가치 보존.

## 0. TL;DR

`docs/13`의 P1+P2+P4 (reciprocal multiply, Phase 1+2 fusion, shared padding) 를 결합한 새 커널 `factor_mid_opt<float>` 작성. 측정 결과:

- **P4 (padded LD)**: 별도 stage-in 형태 필요 → integer division 오버헤드가 P4 이득 상쇄, **+85% 회귀**로 폐기
- **P1+P2 (no P4)**: USA B=16/128 **−4%**, B=64 −1.4%, case8387 noise/+5% 회귀
- **P3**: P2 fusion에 부분 흡수 (work granularity 확대로 thread utilization 개선)
- **메타**: docs/10 §9의 "sync 절감 ≠ wall 비례 단축" 재확인. ROI 작음.

## 1. 설계

### 1.1 적용한 4개 개선안

| 제안 | 적용 여부 | 구현 위치 (mid_opt.cuh) |
|------|----------|------------------------|
| **P1** reciprocal multiply | ✓ | `inv_piv = 1/piv` hoist, `*= inv_piv` 사용 |
| **P2** Phase 1+2 fusion | ✓ | 단일 per-k loop에서 panel update + U-solve step k 함께 |
| **P3** parallel U-solve | 부분 (fusion에 흡수) | P2의 work pool 확대로 thread utilization ↑ |
| **P4** shared padding | ✗ (revert) | 처음 LD=fsz_cap+1로 시도했으나 stage-in 비용으로 폐기 |

### 1.2 핵심 코드 (P1+P2 fusion)

```cpp
for (int k = 0; k < nc; ++k) {
    T piv = Fs[k*LD + k];
    if (piv == 0) { ... }
    const T inv_piv = T(1) / piv;             // P1: hoist once
    
    // Divide column k
    for (int i = k+1+t; i < fsz; i += nt)
        Fs[i*LD + k] *= inv_piv;
    __syncthreads();
    
    // P2: fused panel-update + U-solve step k in single pass
    const long panel_work = (fsz-k-1) * (nc-k-1);
    const long usolve_work = (k > 0) ? uc : 0;
    for (long e = t; e < panel_work + usolve_work; e += nt) {
        if (e < panel_work) {
            // panel update at (i, jj) for jj < nc
            ii = k+1+e/pc; jj = k+1+e%pc;
            Fs[ii*LD+jj] -= Fs[ii*LD+k] * Fs[k*LD+jj];
        } else {
            // U-solve step k: row k, col jj = nc + (e-panel_work)
            jj = nc + (e - panel_work);
            T v = Fs[k*LD + jj];
            for (int i = 0; i < k; ++i) v -= Fs[k*LD+i] * Fs[i*LD+jj];
            Fs[k*LD + jj] = v;
        }
    }
    __syncthreads();
}
```

### 1.3 sync 수 분석

| 구성 | Phase 1 syncs | Phase 2 syncs | Phase 3 syncs | **합** |
|------|-------------:|-------------:|--------------:|------:|
| 기존 factor_mid (split form, USA nc=20) | 2·nc = 40 | nc-1 = 19 | 2 | **61** |
| 기존 factor_mid (row-fused, case8387 nc=8) | nc = 8 | nc-1 = 7 | 2 | **17** |
| factor_mid_opt (P2 fusion, USA nc=20) | nc = 20 | absorbed | 2 | **22** |
| factor_mid_opt (P2 fusion, case8387 nc=8) | nc = 8 | absorbed | 2 | **10** |

→ **USA −64%, case8387 −41% sync 절감**.

## 2. 측정

### 2.1 정확도

모든 케이스 OFF/ON에서 relres 동등 범위 (FP32 reduction 순서 차이로 1.5e-5 ~ 3.1e-5 변동).

### 2.2 wall A/B (5-run median, graph mode, P1+P2 no P4)

```
case                     B        OFF         ON   delta
case30                   1    0.03755    0.03759   0.10%
case30                  64    0.00062    0.00062   noise
case118                  1/64  ...        ...      noise
case8387pegase           1    0.36315    0.35529  -2.17%
case8387pegase           4    0.10375    0.10552  +1.71%
case8387pegase          16    0.04129    0.04172  +1.04%
case8387pegase          32    0.03071    0.03150  +2.58%
case8387pegase          64    0.02535    0.02683  +5.81%
case8387pegase         128    0.02495    0.02413  -3.31%
case_SyntheticUSA        1    2.65724    2.84626  +7.11%
case_SyntheticUSA        4    0.90599    0.92623  +2.23%
case_SyntheticUSA       16    0.54499    0.52199  -4.22%
case_SyntheticUSA       32    0.47981    0.47392  -1.23%
case_SyntheticUSA       64    0.46315    0.45657  -1.42%
case_SyntheticUSA      128    0.47356    0.45254  -4.44%
```

→ 패턴: 
- USA B≥16: 작은 win (−1.2 to −4.4%)
- USA B=1, 4: 회귀 (overhead dominate)
- case8387 B=1, 128: small win, 중간 B에서 noise/회귀

### 2.3 패턴 해석

| 케이스 | nc | sync 절감 비율 | trailing 손실 | 순효과 |
|--------|---:|---------------:|--------------:|------:|
| case8387 (nc=8, row-fused 사용) | 8 → 8 (Phase 1만), absorbed U-solve | 7 syncs (-41%) | 큼 (no staged trailing) | 작은 회귀 |
| USA (nc=20, split form) | 40+19 → 20 | -64% | 중간 | 작은 win |

**핵심 관찰**:
1. **case8387 nc=8 에선 sync 가 이미 작아** (row-fused panel LU가 docs/10 §8에서 적용됨) fusion 의 절감 폭이 작음. 반면 staged trailing 손실은 상수
2. **USA nc=20 에선 sync 가 많아 (61개)** fusion 절감 폭 큼. trailing 손실 absorbed
3. **trade-off threshold**: nc≥16 정도에서 fusion 이득 > trailing 손실

## 3. P4 (shared padding) 폐기 과정

### 3.1 시도

LD = fsz_cap + (fsz_cap % 32 == 0 ? 1 : 0). column writes/reads 의 32-way bank conflict 회피.

### 3.2 결과 — 큰 회귀

5-run median으로 **case8387 B=1 +85%, USA B=1 +20%, USA B=64 +35%** 회귀.

### 3.3 원인 — stage-in의 integer division

P4 활성 시 stage-in 형태가 변경 필요:

```cpp
// 기존: 단순 bulk copy
for (e = t; e < fsz²; e += nt) Fs[e] = F[e];

// P4 필요: LD ≠ fsz 이므로 per-row mapping
for (e = t; e < fsz²; e += nt) {
    int i = e / fsz, j = e % fsz;
    Fs[i*LD + j] = F[e];
}
```

Ampere integer divide 비용: **~30 cyc per division**. fsz²/256 entries × 256 thread × 2 divisions = 거대.

대안 (row-major 명시):
```cpp
for (int i = 0; i < fsz; ++i)
    for (int j = t; j < fsz; j += nt) Fs[i*LD+j] = F[i*fsz+j];
```

이것도 **+85% 회귀**. 이유:
- `nt = 256 > fsz` 인 경우 (case8387 mid fsz≤79) 매 row iteration 마다 fsz threads 만 활성, 나머지 idle
- fsz iter × per-iter 비용 = 거대한 launch wall

### 3.4 결론

P4 의 이론적 win (bank conflict 회피) 보다 **stage-in 패턴 변경의 비용이 더 큼**. fsz²의 stage-in 비용이 LU compute 보다 큰 mid kernel에서 stage-in 최적화가 critical.

→ **P4 폐기**. LD = fsz_cap 유지 (기존 layout).

## 4. P3 (parallel U-solve) 평가

P3 제안: U-solve inner `for i = 0..k-1` 직렬 loop 을 (i, jj) 차원으로 병렬화.

### 4.1 P2 fusion이 P3의 일부 해결

P2 fusion 후, 매 k 단계의 work pool 이 panel_update + usolve_work 로 확장:
- 단순 U-solve (P2 없을 때): uc threads active
- P2 fused: (fsz-k-1)·(nc-k-1) + uc threads active

USA mid k=10: panel_work = 70·10 = 700, usolve_work = 60. Total 760 work / 256 thread = 3 ops/thread. **thread utilization 100%**.

→ P3의 "thread under-utilization" 문제가 P2 fusion으로 자연 해결. 명시적 (i, jj) 병렬화 필요 없음.

### 4.2 P3 명시 구현 시 추가 비용

inner i loop 을 명시 병렬화하면 reduction 필요 (shared scratch 또는 warp shuffle). reduction 자체 비용이 직렬 inner loop 대비 큼:
- 직렬 inner: k cyc per (k, jj). thread별 k FMA 직렬 = k cyc latency.
- 병렬 + reduction: log₂(k) cyc reduction + sync. k=8: 3 cyc reduction + sync ~10 cyc.

k<8 영역에선 직렬이 더 빠름. k>8 영역에서만 병렬 이득.

→ **P3 명시 구현은 ROI 작아 미구현**. P2 fusion으로 충분히 흡수.

## 5. 결정과 ship 권장

### 5.1 default 

`CLS_USE_OPT_MID` **default OFF**. 안정적인 USA-class 워크로드 (B≥16)에서만 opt-in 권장.

### 5.2 opt-in 권장 조합

```bash
CLS_USE_OPT_MID=1
```

USA-class (n≥100k, nc dominant in mid level=20): B=16에서 −4.2%, B=128에서 −4.4% wall 단축. NR 루프에서 typically B=1~64 사용이므로 잠재 1~3% wall 단축.

case8387-class: 적용 안 함 (B=64에서 +5.8% 회귀).

### 5.3 향후 진행 안 함 (defer 결정)

- **P4 폐기 확정**: stage-in 비용 ceiling
- **P3 명시 구현 보류**: P2 fusion에 흡수, 추가 ROI 작음
- **P5 (warp-spec)**: docs/13에서 last resort 마킹. 시도 안 함.

## 6. 메타-결론

### 6.1 docs/10 §9 의 재확인

docs/10 §9: **"ncu barrier stall 절감 ≠ wall 비례 단축"**. 본 실험은 USA nc=20 sync **−64%** 인데 wall **−1 ~ −4%** — 약 1/16의 wall 변환률.

원인 (docs/10 §9에서 정리):
1. barrier stall 외 다른 stall이 빈자리 채움 (long_scoreboard, wait, math_pipe)
2. occupancy가 sync보다 더 큰 영향
3. trailing의 stage 손실이 sync 이득 상쇄

### 6.2 mid/big tier 최적화의 한계

본 실험으로 mid kernel의 micro-optimization 가능성이 거의 다 탐색됨:
- T4.2.A (docs/10): row-fused panel LU → ~0%
- T4.3 (docs/10): cp.async stage-in → ~−3%
- P1+P2 (본 실험): fused U-solve → ~−1~4%

누적 −4~10% 가량 가능하지만 case-dependent. **GEMM (TF32 trailing T1) 과 dispatch (T-split docs/12)** 가 더 큰 lever 유지.

### 6.3 향후 lever

mid/big tier의 더 큰 win 가능 영역:
1. **B-방향 grouped GEMM** (docs/11 T2, 미실시): B=64 trailing의 batch 차원 fusion. 잠재 5-15% 큼
2. **persistent kernel** (docs/09 T4.4, 미실시): factor_big의 multi-launch overhead 제거. USA big tier 잠재 5%
3. **scatter_values / solve phase 최적화** (docs/10 §9 권장): factor 외 phases

본 실험으로 panel LU + U-solve 영역의 "남은 작은 lever" 까지 확인 완료. 향후는 GEMM / non-factor 영역으로 이동.

## 7. 측정 재현

```bash
# Build (mid_opt 포함, default OFF)
cmake -B build -DCLS_INTERNAL_GRAPH=ON ...

# A/B
for opt in 0 1; do
  CLS_USE_OPT_MID=$opt build/custom_linear_solver_run /datasets/.../case_SyntheticUSA \
    --precision fp32 --batch 64 --batch-only --repeat 20
done
```

원본 데이터: `/home/claude/build_cls_refactor/` 빌드 사용.

## 8. 관련 문서

- 제안 도출: [`docs/13`](13-panel-lu-u-solve-bottleneck-2026-06-06.md) P1-P5
- 종전 panel LU 최적화: [`docs/10 §8`](10-t4.1-t4.3-results-2026-06-06.md) T4.2.A row-fused
- 메타-결론 출처: [`docs/10 §9`](10-t4.1-t4.3-results-2026-06-06.md)
- tier 구조: [`docs/12 §9`](../04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md)
- mid/big threshold 근거: [`docs/02-design-analysis/06`](../02-design-analysis/06-tier-threshold-rationale-2026-06-06.md)
