# Small-tier multi-front-per-warp packing 실험 (T-pack) — 2026-06-06

**상태**: 실험 완료 — opt-in 가능, 기본 OFF 권장 (USA B=64 −5.7% 한정 win)
**선행**: [`docs/04-benchmarks-profiling/12 §11.1`](../04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md) 의 small tier 최적화 후보 (Multi-front-per-warp packing)
**관련**: docs/12 §9.1 (small tier ASCII 시각화), docs/10 §9 (sync→wall 변환 한계)

## 0. TL;DR

| 케이스 | B | factor wall (median) | T-pack 효과 |
|--------|--:|---------------------:|------------:|
| case30 | 1/64 | – | no-op (정렬 적용 안됨) |
| case118 | 1/64 | – | no-op |
| case8387 | 1 | 0.356 ms | **+65 ~ +133%** (큰 회귀) |
| case8387 | 64 | 0.0323 ms | **+7 ~ +26%** (회귀) |
| USA | 1 | 2.80 ms | **+21 ~ +39%** (회귀) |
| USA | 64 | 0.535 ms | **−5.7%** (작은 win) |

→ **명확한 negative result**. 유일하게 USA B=64에서만 −5.7% 이득. 원인: launch fragmentation. **MIN_GROUP** 튜닝으로도 해소 안 됨.

## 1. 설계

### 1.1 사전 스케줄링 (analyze-time sort)

`src/plan/analyze.cu`: subtree bucketing 직후 SMALL 레벨 (max_fsz≤32) 안에서 plcols를 (nc, fsz, panel_parent, p) 순으로 정렬. subtree slice가 있으면 각 (subtree, level) 슬라이스 안에서만 정렬해 subtree 경계 보존.

```cpp
auto cmp = [&](int a, int b) {
    if (panels.ncols[a] != panels.ncols[b]) return panels.ncols[a] < panels.ncols[b];
    int fsa = mf.front_ptr[a+1]-mf.front_ptr[a], fsb = mf.front_ptr[b+1]-mf.front_ptr[b];
    if (fsa != fsb) return fsa < fsb;
    if (mf.panel_parent[a] != mf.panel_parent[b]) return mf.panel_parent[a] < mf.panel_parent[b];
    return a < b;
};
```

→ 같은 (nc, fsz) 가 contiguous, 같은 group 안에서 siblings (같은 parent) 인접.

Gate: `CLS_USE_SMALL_PACKED=1` env. default OFF (정렬 자체도 안 함 → 비활성 시 plcols bit-for-bit 동일).

### 1.2 정렬 효과 (case8387 L=2, 722 fronts)

| | unsorted | sorted |
|---|--------:|------:|
| 같은 (nc, fsz) 인접 run 수 | 573 | **197** |
| 최대 run 길이 | 6 | **130** |
| run 크기 ≥ 4 인 front 비율 | 낮음 | **74.3%** |

가장 큰 그룹: `nc=2 fsz=6` 130-front contiguous run.

### 1.3 packed kernel (`src/factorize/small_packed.cuh`)

template `factor_small_packed<FT, FRONTS_PER_WARP>`:
- block = 256 thread = 8 warps
- 1 warp 가 FRONTS_PER_WARP=N (front, batch) tuple 처리
- 32 lane을 N개 sub-warp 으로 분할 (lanes_per_front L = 32/N)
- sub_warp_idx = lane / L, sub_lane = lane % L
- N ∈ {1, 2, 4} 인스턴스 (compile-time)

dispatch 시 fsz 기준 N 선택:
```cpp
if (fsz <= 3)      N = 4;   // fsz²=9, L=8
else if (fsz <= 7) N = 2;   // fsz²=16~49, L=16
else               N = 1;   // L=32 (= 기존 factor_small)
```

**nc=2 unrolled fast path**: outer loop 2회를 인라인 unroll. 75-88%의 small front가 nc=2 → bypass의 hot path.

### 1.4 dispatch (multifrontal.cu)

각 level 안에서:
1. plcols 스캔, 같은 (nc, fsz) 인접 run 식별
2. run_cnt ≥ MIN_GROUP 이고 N ≥ 2 → packed kernel launch
3. 그 외 → 누적 "tiny streak" → 한 번에 모아 factor_small launch (flush)

flush 로직: packed 가 발사되기 전 / level 끝에서 누적된 tiny streak 을 한 번에 처리. **launch 폭증 방지**.

## 2. 측정 결과

### 2.1 MIN_GROUP sweep (5-run median, factor_per_sys_ms)

```
case                     B   OFF    MIN=4     MIN=8     MIN=16    MIN=32
case8387                 1   0.356  +132.6%   +96.1%    +84.4%    +65.3%
case8387                64   0.032  +26.4%    +26.4%    +18.6%    +7.1%
USA                      1   2.80   +39.0%    +24.5%    +23.7%    +20.6%
USA                      64   0.535 -2.2%     -5.7%     -5.8%     -3.4%
```

→ USA B=64 에서 MIN=8~16 에서 **−5.7% / −5.8%** small win. 나머지 모두 회귀.

### 2.2 정확도 (모든 MIN 값에서 동등)

| case | B | OFF relres | ON relres |
|------|--:|-----------:|----------:|
| case30 | 1/64 | 9.5e-7 | 9.5e-7 (동일) |
| case118 | 1/64 | 2.3e-7 | 2.3e-7 |
| case8387 | 1 | 1.5e-5 | 1.5-1.8e-5 |
| case8387 | 64 | 1.5e-5 | 1.4-2.8e-5 |
| USA | 1 | 9.6e-3 | 7.0e-3 ~ 1.0e-2 (동등) |
| USA | 64 | 1.5e-3 | 2.0-3.3e-3 |

정확도는 baseline과 같은 수준 (FP32 reduction 순서 차이로 1e-5 ~ 1e-3 변동).

## 3. 회귀 원인 분석

### 3.1 Launch fragmentation

`subtree multi-stream` (`use_multistream = num_subtrees > 1`)이 활성이면 plcols는 (subtree, level) 슬라이스로 분할되어 sort 가 각 슬라이스 내부에서만 일어남.

case8387 의 subtree 수 K → 각 SMALL level이 K 개의 슬라이스로 쪼개짐. 각 슬라이스 안에서 동일 (nc, fsz) 큰 run 이 형성되지만, **다른 슬라이스의 같은 (nc, fsz) 와는 합쳐지지 않음**.

결과: 한 level 에서 packed launch 수 = K × distinct(nc, fsz). case8387 L=2 에서 분석:
- 197 unique (subtree, nc, fsz) 분포
- packed launches (MIN=8) ≈ 10~20
- flush launches ≈ 2 × K = 4~16
- **Total launches per level ≈ 15~30** (vs 원본 1 launch)

각 launch overhead ~5-10 μs → 100-300 μs/level 누적. B=1에서 factor wall 자체가 350 μs/sys 라 launch overhead 가 wall을 잠식.

### 3.2 B=1 vs B=64 의 차이

B=64 에서는 같은 launch overhead이지만 처리하는 work가 64x. launch overhead 비중이 1/64로 줄어 packed kernel 의 진짜 efficiency가 드러남.

USA B=64 에서 −5.7% win은 launch overhead 가 work 대비 작아진 결과 + USA가 multi-stream subtree fragmentation이 case8387 보다 덜 심한 영향 (USA 가 더 적은 subtree, 더 균등한 분포).

### 3.3 per-warp work 분석 (case8387 L=2 nc=2 fsz=6 run, N=2)

| 단계 | 작업 | per-lane 반복 (L=16) |
|------|------|--------------------:|
| stage-in | 36 entries/front × 2 fronts | 4.5 iter/lane |
| LU k=0 divide | 5 entries × 2 fronts | 0.6 iter |
| LU k=0 trailing | 25 entries × 2 fronts | 3.1 iter |
| LU k=1 divide | 4 entries × 2 fronts | 0.5 iter |
| LU k=1 trailing | 16 entries × 2 fronts | 2.0 iter |
| writeback | ~12 entries × 2 fronts | 1.5 iter |
| extend-add (atomic) | 16 entries × 2 fronts | 2.0 iter |
| **합계** | | ~14 iter/lane (2 fronts) |

vs factor_small (32 lanes, 1 front per warp):
- stage-in 1.1 iter, LU k=0 0.2 + 0.8 iter, LU k=1 0.1 + 0.5 iter, writeback 0.4 iter, extend-add 0.5 iter ≈ ~3.5 iter/lane × 1 front

→ packed: 14 iter / 2 fronts = **7 iter/lane/front**. factor_small: **3.5 iter/lane/front**.

per-front lane work는 packed가 2x. 이게 launch overhead 줄임과 상쇄 + 추가 — net 회귀.

원인: L=16 lane 만 active로 working set이 작아진 게 아니라, sub-warp 내부 loop 횟수가 늘었음. 32 lane 으로 1 front 처리하는 게 sub-warp 16 lane 으로 1 front 처리하는 것보다 lane-parallel 이 좋음.

### 3.4 atomic 충돌 (가설, 미측정)

sibling sort로 인해 동일 warp 안의 2개 front 가 **같은 parent**에 동시 atomicAdd. 같은 parent 의 같은 cache line에 두 lane이 동시 atomic → serialization → wait stall 증가.

ncu로 확인 안 했지만 plausible. sibling 인접화는 atomic locality 면에서는 양날의 검 — cache hit ↑ 하지만 contention ↑.

## 4. 결정 및 follow-up

### 4.1 ship 결정

**기본 OFF 유지**. opt-in (CLS_USE_SMALL_PACKED=1 + CLS_SMALL_PACKED_MIN=8) 으로 USA-class workload 에서 −5% 정도 가능. case8387-class에선 사용 금지.

### 4.2 추가 시도해 볼 만한 것

1. **Multi-group launch**: 한 packed kernel 호출이 여러 (nc, fsz) 그룹을 binary-search로 분기. launch 수 K × distinct(nc, fsz) → 1로 줄임. 가장 큰 잠재 — 구현 부담 있음
2. **subtree-cross sort**: 정렬을 (subtree, level) 슬라이스 안이 아니라 level 전체에서 수행. subtree multi-stream 비활성 (single stream) 모드에서만 적용. multi-stream 효과 (~10-15% 잠재)를 packed kernel과 trade-off
3. **sibling-decoupled scheduling**: 같은 warp에 siblings 가 들어가지 않도록 round-robin 배치. atomic contention ↓
4. **factor_small 전용 nc=2 specialization** (packed 없이): 단지 outer loop 만 unroll. case8387 B=64 small wall의 ~40% 비중에서 작은 이득 가능 (잠재 −1~3%)

### 4.3 측정 인프라

새 파일:
- `src/factorize/small_packed.cuh` (kernel, 3 instantiations)
- `src/plan/analyze.cu` (sort step, env-gated)
- `src/multifrontal.cu` (dispatch + cudaFuncSetAttribute opt-in)
- `src/solver.cpp` (CLS_DUMP_FRONTS 가 plcols 순으로 출력하도록 갱신)

opt-in 환경변수:
- `CLS_USE_SMALL_PACKED=1` (default 0)
- `CLS_SMALL_PACKED_MIN=N` (default 8)

## 5. 메타-회고 — small tier 최적화의 ROI ceiling

docs/12 §11.1 에서 small tier의 최대 잠재 가산을 −10~15% factor wall (case8387 B=64) 로 추정. 본 실험은:
- analyze-time 사전 스케줄링: ✓ (정렬 효과 입증)
- multi-front packed kernel: ✓ (correctness)
- 그러나 **B=1 launch overhead 폭증**, **B=64 packed가 factor_small 보다 per-front efficient 하지 않음**

근본 원인: small front는 너무 작아 32 lane 으로 처리하는 게 16 lane × 2 front 처리보다 효율적. 즉 **packing 이 lane 활용률을 올리는 게 아니라 lane 당 work 만 키움**. lane 자체가 이미 충분히 차 있던 게 아니라, **launch grid 가 부족했던 것**. 그런데 small tier는 grid가 이미 (level_size × B / 8) 만큼 커서 SM saturation 충분.

→ small tier 의 정말 큰 lever는 packing 이 아니라 **launch overhead 자체 줄이기** (multi-level fusion, persistent kernel, CUDA graph)이며, 이미 docs/01 의 D1 (CUDA Graph) 가 production default 로 적용됨. **추가 small tier 최적화 ROI 는 낮다** 가 본 실험의 결론.

이는 docs/10 §9 의 메타-결론 ("ncu stall 절감이 wall로 비례 변환되지 않는다") 의 연장선:
- factor_small B=64 의 wait stall (FMA chain) 은 본질적으로 알고리즘의 dependency chain. packing 으로 줄지 않음.
- launch overhead 는 분리된 문제 — graph capture 가 이미 해결.
- 남은 small wall ~21-40% 는 사실상 **불가피한 work**.
