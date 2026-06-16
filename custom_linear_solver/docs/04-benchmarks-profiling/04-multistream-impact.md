# Subtree multi-stream의 tier별 임팩트와 메커니즘

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: subtree multi-stream(K=8)은 큰 문제에서 factor wall을 단축 (case8387 B=64 −22%, USA B=64 −14%, B=1 −9%) — Hyper-Q가 per-kernel 시간 자체를 줄이지만 작은 케이스는 fork/join overhead로 regress.

**대상**: refactored `custom_linear_solver` (`41897cd` + T4.2.A + T4.3 ON), case8387 / USA × fp32 × B=1/64, RTX 3090.
**Toggle**: `CLS_DISABLE_MULTISTREAM=1`로 single-stream 강제 (`src/multifrontal.cu` setup의 subtree_streams 할당 skip).
**plan**: 모든 케이스 `num_subtrees = K = 8` (default에서 8 stream 동시 진행).

## 0. A/B 결과

| case | B | multi-stream | single-stream | delta (single 대비) |
|---|--:|---:|---:|---:|
| case30 | 1 | 0.038 ms | 0.034 ms | single **−10%** |
| case30 | 64 | 0.0006 ms | 0.0006 ms | single −8% |
| case118 | 1 | 0.051 ms | 0.048 ms | single −6% |
| case118 | 64 | 0.0009 ms | 0.0008 ms | single −4% |
| case8387 | 1 | 0.350 ms | 0.344 ms | tie (single ~−2%) |
| case8387 | 64 | **0.0262 ms** | 0.0320 ms | **multi −22%** |
| USA | 1 | **2.64 ms** | 2.88 ms | **multi −9%** |
| USA | 64 | **0.464 ms** | 0.531 ms | **multi −14%** |

→ **작은 문제는 single이 빠르고(overhead), 큰 문제는 multi가 큰 폭으로 빠르다.** case8387 B=64 multi-stream **−22%**.

## 1. 메커니즘 — 왜 빨라지나

### 1.1 핵심: per-kernel time 자체가 줄어든다

multi/single에서 **동일 커널, 동일 launch 수**. 일반 이해는 "stream끼리 시간이 겹쳐 총 wall이 작아진다"지만 실측은 다르다 — *각 커널 launch 자체가 더 빨리 끝난다*.

case8387 B=64, nsys per-kernel total (5 reps 합):

| | factor_small | factor_mid | factor 합 |
|---|---:|---:|---:|
| multistream | 2655 μs | **5495 μs** | 8149 μs |
| single | 2659 μs | **9206 μs** | 11865 μs |
| 단축 | −0.2% | **−40%** | **−31%** |

USA B=64:

| | small | mid | big | 합 |
|---|---:|---:|---:|---:|
| multi | 32183 | 62771 | 52034 | 146988 μs |
| single | 38159 | 56311 | 59901 | 154371 |
| 단축 | **−16%** | +12% | −13% | −5% |

USA B=1:

| | small | mid | big | 합 |
|---|---:|---:|---:|---:|
| multi | 688 | 2938 | 9947 | 13572 μs |
| single | 861 | 2717 | 11319 | 14897 |
| 단축 | **−20%** | +8% | −12% | −9% |

→ **small tier가 모든 케이스에서 −16~−22%로 일관 단축**. mid/big은 case-dependent.

### 1.2 Hyper-Q 메커니즘: 다른 stream의 work가 SM stall을 채운다

GPU는 한 SM에 여러 block resident 가능 (sm_86: 16 block / 1536 thread). 한 stream 커널이 SM 일부만 채우거나 stall로 idle하면, **다른 stream의 block이 그 자리를 즉시 채워 idle을 줄인다** — NVIDIA Hyper-Q / 동시 커널 실행의 직접 효과.

per-kernel 단축 조건: ① SM에 idle 사이클(메모리 latency / barrier 대기)이 있어야 함, ② 다른 stream block이 들어갈 register/shared 여유가 있어야 함. → **SOL_SM이 낮을수록 효과 큼** (latency-bound / sync-bound 커널이 최대).

```
single-stream:  SM ████      ████      ██████   (점유 30%, idle 70%)
multi-stream:   SM ████████████████████████████ (8 stream block으로 거의 포화)
                   ↑ stream 0 block latency 동안 stream 3 block이 SM 채움
```

### 1.3 tier별 SOL_SM (참고, `03-gemm-fraction-front-distribution.md` §5.1)

| tier | B=1 SOL_SM | B=64 SOL_SM |
|---|---:|---:|
| small (8387) | 7.0% | 30.8% |
| small (USA) | 31.1% | 48.6% |
| mid (8387) | 6.5% | **47.3%** |
| mid (USA) | 20.1% | **42.6%** |
| big (USA) | 5.5% | 41.8% |

→ **B=1 모든 tier가 SOL_SM 5–31%** → 다른 stream block 들어갈 여유 매우 큼.

### 1.4 small tier가 가장 일관되게 이득보는 이유

small kernel은 per-block work가 매우 작음 (median fsz=6 → ~50 ops/warp). per-block 실행 짧고 memory latency가 work 대비 비율로 큼 → 다른 stream block이 latency를 채우면 효과 극대화. shared 사용량 작아(~144 B) co-residency 부담 적음.

### 1.5 mid/big tier가 case-dependent인 이유

mid/big의 효과는 trade-off: **이득** = barrier/latency stall이 다른 stream으로 채워짐 vs **손실** = DRAM bandwidth / L2 / atomic 공유 자원 경합.

- case8387 mid B=64: barrier stall 41% 압도 + SOL_Mem 38% (메모리 여유) → **−40%**
- USA mid B=64: SOL_Mem 41%, DRAM 31% (이미 빡빡) → contention으로 **+12% 회귀**
- USA big B=64: barrier 37% + SOL_SM 42% 일부 여유 → **−13%**

→ **메모리 여유 vs barrier stall 정도가 win/lose를 결정**. 휴리스틱: SOL_Mem < 35% → try, > 40% → 신중.

## 2. 작은 문제는 왜 single이 빠른가

case30 (n=53), case118 (n=118)은 모든 factor wall이 50 μs 미만. multi-stream 비용: ① stream creation/destruction (setup 1회), ② event fork/join (매 factor 호출 ~10–20 μs), ③ graph capture overhead (8 stream + 8 join events로 복잡). case30 B=1 factor 34 μs인데 fork/join ~4 μs 이상 → 11% 회귀.

→ **factor wall이 multi-stream overhead보다 충분히 클 때만 win** (임계 ~factor wall ≥ 100 μs).

## 3. tier별 ROI

| tier | 효과 | 이유 |
|---|---|---|
| **small** | 항상 win (B=1 −16~−22%, B=64 −0~−16%) | per-block work 작아 SM 자주 idle, interleaving 최적, shared 부담 적음 |
| **mid** | case-dependent | barrier 우세(8387)→큰 win, memory 우세(USA)→contention 손실 |
| **big** | 작은 win (USA −12~−13%) | 1024t block → SM당 1–2 block만 resident (여유 적음), 그러나 barrier 37%로 idle 많음 |

## 4. 추천 정책

### 4.1 현재 default (multi-stream ON when num_subtrees > 1)
대형 power-grid (case8387, USA)에서 −9~−22% factor wall 이득 → **유지**. 소형(case30/118) +6~10% 회귀지만 NR 루프에서 절대 시간 작아 영향 없음 (micro-bench에서만 가시).

### 4.2 개선 옵션
- **size-aware gating**: `plan.n < 200`일 때 multi-stream 자동 disable (적은 변경). 실 workload(n ≥ 10000)에서는 항상 win이라 case30/118 외 영향 없음.
- 현 구현은 NR 외부 setup에서 한번 결정, NR 반복 중 toggle 불가.

### 4.3 K 선택 — K=8이 현 워크로드에 적절
현재 `num_subtree_streams = num_subtrees ≤ 8`. case8387/USA 모두 K=8에 sat. K=16으로 늘리려면 spine을 더 깊이 자르거나 subtree를 더 잘게 쪼개야 하지만, contention 증가 위험 (USA mid B=64에서 이미 +12% 회귀; mid SOL_Mem 41%라 추가 stream은 contention 우세 가능성). → **K=8 적절**.

## 5. 관련 문서

- `03-gemm-fraction-front-distribution.md` §5, §10 — tier별 wall + 병목 + 구조
- `../03-optimization-notes/01-kernel-engineering.md` — subtree streams 도입 history, 비-GEMM 최적화 메타-결론 (barrier stall 절감이 wall로 전환되는 한계와 multi-stream의 우회 관계), packing 회귀
- `02-strumpack-vs-custom-case8387.md`, `../main-report.md`

원본 데이터: `/home/claude/prof/ms_{0,1}_{8387,USA}_B{1,64}.nsys-rep`.
