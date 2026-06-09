# Panel LU / U-solve is barrier-bound — the dominant 61% of the big kernel

**작성일**: 2026-06-09
**범위**: big factorization 의 지배 비용(panel LU + U-solve)을 phase breakdown + ncu 로 진단.
동기: TC 가 안 보이는 이유가 trailing(TC)이 커널의 극소수라서 (doc 28). panel 을 줄이면 (a) factor
가 빨라지고 (b) trailing/TC 비중이 커진다 — 사용자 가설. 결론: panel 은 **barrier-bound**(순차 pivot
sync), memory 아님. lever 는 sync 체인 단축(blocked LU / block-size).

---

## 1. Phase breakdown: panel 이 61% (70K fp32 B=1, MB 경로로 분리)

| kernel | time | 비중 |
|---|---|---|
| **factor_big_panel** (Phase 1 LU + Phase 2 U-solve) | 30.9ms | **61.4%** |
| factor_big_trailing_mb (Phase 3) | 9.8ms | 19.5% |
| factor_big_extend (Phase 4) | 9.6ms | 19.2% |

(USA 도 동일 비율 61/19/19.) **panel(LU+U-solve)이 압도적.** trailing(TC 대상)은 19% 뿐 — TC 가
안 보이는 이유. panel 을 절반으로 줄이면 trailing 비중이 19%→~32% 로 올라 TC 가 상대적으로 드러나고
커널도 빨라진다.

## 2. 진단: panel 은 BARRIER-bound (memory 아님)

ncu `factor_big_panel<float>` (70K B=64, 1024-thread block):

| metric | 값 | 해석 |
|---|---|---|
| **stall_barrier** | **12–13** | **`__syncthreads` 대기 지배** |
| stall_long_scoreboard | 2.0 | global 지연 아님 |
| lts sector hit | **92%** | panel 은 L2 에 상주 (memory 문제 아님) |
| DRAM | 1.7% | 무시 |
| sm__throughput | 4.5–8% | 연산도 거의 안 함 |
| warps_active | 66% | occupancy 는 양호 |

→ **병목은 순차 pivot 의 1024-thread `__syncthreads` 체인.** right-looking LU 는 pivot k 마다
(divide → sync → rank-1 update 전 폭 → sync) 를 nc 회 반복. 각 pivot 의 실제 일감(rank-1 update
≈ uc·(nc−k) / 1024 thread)은 극소수인데 **1024-thread barrier 비용이 그걸 압도**. U-solve 도 k 마다
sync. 즉 occupancy 도 충분하고 memory 도 L2 가 받쳐주는데, **sync latency 가 임계경로**.

## 3. LU/U 최적화 lever (후보)

1. **Blocked right-looking LU (rank-b panel)**: b 폭 panel 을 먼저 분해(작은 b 회 sync, 좁은 strip)
   후 나머지를 **rank-b GEMM** 한 번으로 갱신. 전 폭 update + sync 횟수가 nc → nc/b 로 감소
   (barrier 비용의 주축 제거). 부수효과: rank-b 갱신이 GEMM 이라 (작은 K=b 지만) 연산이 정규화됨.
2. ~~**Block-size 축소**~~ **(반증 — 2026-06-09)**: 256/512/1024 A/B 결과 **1024 가 최적**, 줄이면
   더 느림 (70K B=1: 256→2.50, 512→2.22, 1024→2.08 ms/sys; B=64·USA 동일 경향). 해석: barrier
   latency 자체가 아니라 **순차 pivot 깊이 × 각 pivot 의 parallel update(uc 행 × nc)** 가 비용 →
   thread 가 많을수록 update 가 빨라 net 이득. 1024 가 이미 max block 이라 더 늘릴 수도 없음. barrier
   가 **노출되는** 진짜 이유는 **big 레벨 underfill**(동시 front 부족 → barrier 가 다른 front 일로
   가려지지 않음; B=1 최악, B=64 도 20–52 블록). → 진짜 lever 는 sync **횟수 감축**(blocked LU) 또는
   **underfill 완화**(동시성↑).
3. **Warp-specialized / 2-stage pipeline**: pivot k+1 의 column 준비를 pivot k 의 update 와 겹쳐
   sync 노출을 줄임 (doc 26 의 pipelining 아이디어를 panel 에 적용).
4. **U-solve 융합**: Phase 2 의 per-k sync 를 panel LU 의 sync 와 공유/축소.

## 4. TC 가설과의 연결

panel 최적화는 fp32/tf32 **양쪽에 동등**하게 적용된다(같은 scalar panel). 따라서:
- **직접 효과**: factor 전체가 빨라짐 (panel 이 61% 이므로 큰 절대 win).
- **TC 가시성**: trailing 비중↑ → tf32-vs-fp32 격차가 상대적으로 커져 보임. 단 trailing 자체는
  thin-K memory-bound(doc 28)라 TC 가 fp32 를 **이기는** 폭은 여전히 작다.
- TC 가 fp32 를 실제로 이기려면 trailing 을 compute-bound 로 바꿔야 함(별개 과제). 본 lever 는
  "TC 가속이 보이게" + "factor 가속" 두 목적을 동시에 달성하는 현실적 다음 스텝.

## 5. 다음 실험 (우선순위)
- ~~block-size A/B~~ **완료 → 반증** (1024 최적, §3-2). barrier 는 block-size 로 못 줄임.
- 남은 후보 (둘 다 sync **횟수** 또는 **노출**을 건드림):
  - **blocked LU(rank-b)**: 전폭 update+sync 를 nc→nc/b. 단 총 update FLOP 은 동일하고 sync
    **횟수**(narrow+wide)는 비슷 — barrier latency 가 아니라 update **효율**(rank-b GEMM 재사용)에서
    이득이 와야 함. 효과 불확실, 프로토타입 비용 큼.
  - **underfill 완화**: big 레벨이 동시 front 부족이라 barrier 가 노출됨. 여러 big 레벨/front 의
    panel 을 **한 grid 로 batch**하면 동시성↑ → barrier 가 다른 front 일로 가려짐. (doc 26 의
    latency-hiding 계열; mid 와 동형 문제.)
- 판별: block-size 가 무효였으므로 sync **latency** 는 (1024 max 에서) irreducible. 남은 건 sync
  **횟수↓**(blocked) 또는 **동시성↑**(underfill 완화). 후자가 더 근본적(big 전 구조가 underfill).
