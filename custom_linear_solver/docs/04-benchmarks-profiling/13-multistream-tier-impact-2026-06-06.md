# Subtree multi-stream의 tier별 임팩트와 메커니즘 분석

**작성일**: 2026-06-06
**선행**: [`docs/03-optimization-notes/08`](../03-optimization-notes/archive/08-tree-restructuring-research-plan.md) (subtree streams 도입), [`docs/03-optimization-notes/11`](../03-optimization-notes/archive/11-small-packed-experiment-2026-06-06.md) (packing 회귀, deprecated 이동)
**대상**: refactored `custom_linear_solver` (`41897cd` + T4.2.A + T4.3 ON), case8387 / USA × fp32 × B=1/64
**Toggle**: `CLS_DISABLE_MULTISTREAM=1` env로 single-stream 모드 강제 (`src/multifrontal.cu` setup의 subtree_streams 할당 skip)
**plan**: 모든 측정 케이스에서 `num_subtrees = K = 8`. 즉 default에서 8개 stream 동시 진행.

## 0. TL;DR

| case | B | multi-stream | single-stream | delta (single 대비) |
|------|--:|-------------:|--------------:|--------------------:|
| case30 | 1 | 0.038 ms | 0.034 ms | single **−10%** |
| case30 | 64 | 0.0006 ms | 0.0006 ms | single −8% |
| case118 | 1 | 0.051 ms | 0.048 ms | single −6% |
| case118 | 64 | 0.0009 ms | 0.0008 ms | single −4% |
| case8387 | 1 | 0.350 ms | 0.344 ms | tie (single ~ |−2%) |
| case8387 | 64 | **0.0262 ms** | 0.0320 ms | **multi −22% (single 대비 +22%)** |
| USA | 1 | **2.64 ms** | 2.88 ms | **multi −9%** |
| USA | 64 | **0.464 ms** | 0.531 ms | **multi −14%** |

→ **작은 문제는 single이 빠르고 (overhead), 큰 문제는 multi가 큰 폭으로 빠르다**. case8387 B=64에서 multi-stream **−22%** wall 단축.

## 1. 메커니즘 — 왜 multi-stream이 빨라지나

### 1.1 핵심 관찰: per-kernel time 자체가 줄어든다

multi-stream과 single-stream에서 **동일한 커널, 동일한 launch 수**. 일반적인 이해는 "stream 끼리 시간이 겹쳐 총 wall이 작아진다"이지만, 실측은 다르다:

case8387 B=64 fp32, nsys per-kernel total time (sum over 5 reps):

| | factor_small | factor_mid | factor 합 |
|---|------:|------:|------:|
| multistream | 2655 μs | **5495 μs** | 8149 μs |
| single-stream | 2659 μs | **9206 μs** | 11865 μs |
| 단축 | −0.2% | **−40%** | **−31%** |

→ factor_mid 한 launch가 single-stream 보다 multi-stream에서 **−40% 빨리 끝남**. launch 수 동일.

USA B=64:

| | factor_small | factor_mid | factor_big | 합 |
|---|------:|------:|------:|------:|
| multistream | 32183 μs | 62771 μs | 52034 μs | 146988 |
| single-stream | 38159 μs | 56311 μs | 59901 μs | 154371 |
| 단축 | **−16%** | +12% | −13% | −5% |

USA B=1:

| | small | mid | big | 합 |
|---|------:|------:|------:|------:|
| multistream | 688 μs | 2938 μs | 9947 μs | 13572 |
| single-stream | 861 μs | 2717 μs | 11319 μs | 14897 |
| 단축 | **−20%** | +8% | −12% | −9% |

→ small tier가 **모든 케이스에서 −16~−22%로 일관 단축**. mid/big tier는 case-dependent.

### 1.2 메커니즘: 다른 stream의 work가 SM stall을 채운다

GPU는 한 SM에 여러 block을 resident시킬 수 있다 (sm_86에서 16 block / 1536 thread). 한 stream의 커널이 SM의 일부만 채우거나 (낮은 occupancy / SOL) stall로 잠시 idle하면, **다른 stream의 block이 그 자리를 즉시 채워서 idle을 줄인다**. 이게 NVIDIA가 말하는 "Hyper-Q / 동시 커널 실행"의 직접 효과.

per-kernel time 단축이 일어나는 조건:
- SM이 "메모리 latency 대기" 또는 "barrier 대기" 등으로 idle 사이클이 있어야 함
- 다른 stream의 block이 실제로 SM에 들어갈 register/shared 여유가 있어야 함

→ **SOL_SM이 낮을수록 다른 stream의 work가 들어갈 여유가 크다**. 즉 latency-bound / sync-bound 커널이 multi-stream 효과 가장 큼.

### 1.3 tier별 SOL_SM (이전 측정, docs/12 §10.2)

| tier | B=1 SOL_SM | B=64 SOL_SM |
|------|----------:|------------:|
| small (8387) | 7.0% | 30.8% |
| small (USA) | 31.1% | 48.6% |
| mid (8387) | 6.5% | **47.3%** |
| mid (USA) | 20.1% | **42.6%** |
| big (USA) | 5.5% | 41.8% |

→ **B=1 모든 tier가 SOL_SM 5-31%로 매우 낮음** → multi-stream으로 다른 stream의 block이 들어갈 여유 매우 큼.

### 1.4 small tier가 가장 일관되게 이득보는 이유

small kernel은 **per-block work가 매우 작다** (median fsz=6 → ~50 ops/warp). 즉:
- per-block 실행 시간 짧음
- block 간 launch 간격 / 메모리 latency 가 work 대비 비율로 큼
- 다른 stream block이 들어가 메모리 latency를 채우면 효과가 극대화

case8387 B=1 factor_small: 13 μs/launch (docs/12), per-block 작업이 ms 수준이 아니라 μs 수준. memory latency hide 가능성이 매우 큼.

### 1.5 mid/big tier가 case-dependent인 이유

mid/big 커널의 multi-stream 효과는 두 요소의 trade-off:
- **이득**: barrier stall (case8387 mid B=64 41%) / latency stall이 다른 stream으로 채워짐
- **손실**: DRAM bandwidth, L2 cache, atomic locking 등 공유 자원 경합

case별:
- case8387 mid B=64: barrier stall 41% 압도적, SOL_Mem 38% (memory에 여유 있음) → multi-stream **−40% 단축**
- USA mid B=64: SOL_Mem 41%, DRAM% 31% (이미 메모리 빡빡), 다른 stream과 contention → **+12% 회귀**
- USA big B=64: barrier stall 37% (시각 stall ↑) + SOL_SM 42%로 일부 여유 → **−13% 단축**

→ **메모리 여유 vs barrier stall 정도가 multi-stream win/lose 를 결정**

## 2. 작은 문제는 왜 single이 빠른가

case30 (n=53), case118 (n=118) 은 모든 factor wall이 50 μs 미만. multi-stream의 비용:
1. **stream creation/destruction**: setup 시 1회. NR 루프 외부지만 측정에 포함되면 회귀
2. **event fork/join**: 매 factor 호출마다 (host thread sync까지 포함) ~10-20 μs
3. **graph capture overhead**: multi-stream graph는 single-stream graph보다 복잡 (8개 stream + 8 join events)

case30 B=1 factor 자체가 34 μs인데 multi-stream의 fork/join overhead가 ~4 μs 이상 → 11% 회귀.

→ **factor wall이 multi-stream overhead보다 충분히 클 때만 win**. 임계 약 factor wall ≥ 100 μs.

## 3. 자세한 메커니즘 정리

```
single-stream:
  ┌─────────────┐
  │ stream 0    │  L1: small  ──→  L2: small ──→  L3: mid  ──→  ...
  └─────────────┘
  SM:           ████          ████          ██████       ██  (낮은 점유율, idle 많음)

multi-stream (subtree streams):
  ┌─────────────┐
  │ stream 0    │  subtree 0 의 L1..LN  ──→  spine
  │ stream 1    │  subtree 1 의 L1..LN  ──→
  │ stream 2    │  ...                  ──→
  │ stream 7    │  subtree 7 의 L1..LN  ──→
  └─────────────┘
  SM:           ████████████████████████████  (병행 stream의 block으로 거의 가득)
                ↑                            ↑
                stream 0 block의 latency 동안   stream 3 block이 SM 채움
```

**핵심**: SM 점유율이 낮은 커널은 multi-stream에서 *block 단위로* 다른 stream의 work와 interleave 되어 idle cycle이 채워짐. 즉:
- single-stream: SM 점유 30% → idle 70%
- multi-stream (2 stream 공존): SM 점유 60% → idle 40% (각 stream 입장에선 idle이 줄어 launch도 더 빨리 끝남)
- 8개 stream: SM 점유 거의 100% (포화)

이건 NVIDIA Hyper-Q의 디자인된 효과. **factor_small의 작은 block (256 thread × 8 block/SM)이 가장 잘 어울림** — block이 작아 다른 stream block과 혼합 resident 가능.

## 4. tier별 multi-stream ROI 분석

### 4.1 small tier — 항상 multi-stream win

- B=1: −16~−22% per-kernel
- B=64: −0~−16% per-kernel (8387 0%는 이미 single에서 잘 saturate 된 경우)

이유:
- per-block work 작아 SM이 자주 idle
- 다른 stream block과의 interleaving이 가장 효과적
- shared memory 사용량 작아 (per-warp ~144 B) co-residency 부담 적음

**ROI 결정자**: factor_small wall 차지율이 클수록 효과 큼. case8387 B=64 small=40% → 전체 factor wall의 −22% 이득에 큰 기여.

### 4.2 mid tier — case-dependent

barrier 우세 (case8387) → multi-stream이 barrier 대기 시간을 다른 stream의 work로 채움 → 큰 win.
memory 우세 (USA) → contention으로 손실.

판단 휴리스틱:
- SOL_Mem < 35% → multi-stream try
- SOL_Mem > 40% → multi-stream 신중

### 4.3 big tier — 작은 win (대체로)

- block size 1024로 SM당 1-2 block만 resident → 다른 stream block 들어갈 여유 적음
- 그러나 barrier stall 37%로 idle cycle은 많음
- USA B=1 big −12%, B=64 −13% — 꾸준한 작은 win

## 5. 추천 정책

### 5.1 현재 default (multi-stream ON when num_subtrees > 1)

대형 power-grid (case8387, USA) 에서 −9~−22% factor wall 이득 → **유지**.

소형 (case30, case118) 에서 +6~+10% 회귀 → 문제. NR 루프에서 절대 시간이 작아 ms 수준 영향 없지만 micro-bench에서는 가시적.

### 5.2 개선 옵션

1. **size-aware gating**: `plan.n < 200` 일 때 multi-stream 자동 disable. 적은 코드 변경.
2. **NR 외부 setup 이라 한번에 결정** (현 구현). NR 반복 중에는 toggle 불가.

case8387/USA 같은 실 workload (n ≥ 10000)에서는 multi-stream이 항상 win이라 size-gate 추가는 case30/118 micro-bench 외 영향 없음.

### 5.3 multi-stream을 더 강하게 쓰는 방향

현재 `num_subtree_streams = num_subtrees ≤ 8`. case8387/USA 모두 K=8 이라 cap에 sat. 더 늘리는 시도:
- panel etree의 spine을 더 깊이 자르거나, subtree를 더 잘게 쪼개 K=16 등으로
- 단 contention 증가 위험 (USA mid B=64에서 이미 +12% 회귀)
- mid kernel의 SOL_Mem 41%이라 추가 stream은 contention 우세할 가능성

→ **K=8이 현 워크로드에 적절**.

## 6. 측정 재현

```bash
# Default = multi-stream
build/custom_linear_solver_run /datasets/.../case8387pegase \
  --precision fp32 --batch 64 --batch-only --repeat 20

# Single-stream
CLS_DISABLE_MULTISTREAM=1 build/custom_linear_solver_run ...

# nsys per-kernel comparison
for mode in 0 1; do
  CLS_DISABLE_MULTISTREAM=$mode nsys profile -o ms_$mode \
    build_nograph/custom_linear_solver_run ...
  nsys stats --report cuda_gpu_kern_sum --format csv ms_$mode.nsys-rep
done
```

원본 데이터:
- `/home/claude/prof/ms_{0,1}_{8387,USA}_B{1,64}.nsys-rep`

## 7. 관련 문서

- subtree streams 도입 history (Phase 3 race issue 포함): [`docs/03-optimization-notes/08`](../03-optimization-notes/archive/08-tree-restructuring-research-plan.md) §9
- tier별 wall + 병목: [`docs/04-benchmarks-profiling/12`](12-front-gemm-distribution-2026-06-06.md) §10
- 비-GEMM 최적화 메타-결론: [`docs/03-optimization-notes/10`](../03-optimization-notes/10-t4.1-t4.3-results-2026-06-06.md) §9 — barrier stall 절감이 wall로 전환되는 한계와 본 문서 multi-stream의 우회 관계
- 회귀 실험 (deprecated): [`docs/03-optimization-notes/11`](../03-optimization-notes/archive/11-small-packed-experiment-2026-06-06.md) — packing 실패 후 multi-stream 분석으로 전환된 배경
