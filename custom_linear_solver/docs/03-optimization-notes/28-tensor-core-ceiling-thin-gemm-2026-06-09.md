# Tensor-core ceiling: the big trailing is a thin (K=nc) memory-bound GEMM

**작성일**: 2026-06-09
**범위**: mid/big factorization 에서 tensor core 가 fp32 대비 가속되는지 측정·진단. 과제 목표가
TC 가속이므로 중요. 결론: 현재 formulation 으로는 TC 가 fp32 를 못 이긴다. 근본 원인은 trailing
GEMM 의 K=nc 가 작다(~20)는 것. 다음 lever 는 panel LU / U-solve 최적화.

---

## 1. 측정: TC vs fp32 (mid=scalar, big=TC; commit 3f5b959 이후)

end-to-end factor_per_sys_ms, fp32 기준 TC speedup:

| case | B | tf32/fp32 | fp16/fp32 |
|---|---|---|---|
| 70K | 1 | **+6.0%** (느림) | +7.9% |
| 70K | 64 | −2.7% | −6.0% |
| USA | 1 | **+4.2%** (느림) | +11.5% |
| USA | 64 | −1.6% | −1.8% |

**B=1 은 TC 가 더 느리고, B=64 는 marginal.** 과제 목표(TC 가 fp32 대비 가속) 미달.

## 2. 진단: TC pipe ≈ 0 (fused 0.2%, isolated 0.4%)

ncu (70K B=64):
- fused `factor_big_tf32_ptx`: **tensor pipe 0.2%**, sm 8%, FMA 3%, long_scoreboard 3.
- trailing 을 panel/extend 와 분리한 **isolated** `factor_big_trailing_tf32`: **tensor pipe 0.4%**
  (그대로 0), **long_scoreboard 11** (순수 global memory 지연), warps_active 16%, grid 19-39.

→ **trailing 을 분리해도 TC pipe 가 안 올랐다.** batched-TC 의 전제("분리/batching 하면 TC 가
채워진다")가 반증됨. (분리 실험은 correctness 버그도 있어 revert.)

### 2-1. trailing 시간 분해 (CLS_TRAIL_NO_MMA / NO_STAGE 토글, fused 커널 nsys, /30 repeats)

`factor_big_tf32_ptx` 를 staging / mma+drain / panel+extend 로 분해:

| 구성요소 | 70K | USA |
|---|---|---|
| **staging** (L/U 준비, global→shared) | 3.5ms (**1%**) | 15ms (**4%**) |
| **mma+drain** (TC 연산 + C global write) | 92ms (35%) | 133ms (39%) |
| panel+extend (나머지) | 164ms (63%) | 196ms (57%) |

핵심: **staging(준비)은 1–4% 로 싸다** (L/U 패널 작고 L2 캐시). **TC mma 자체도 ~0%**(pipe 0.2%).
"mma+drain" 35–39% 의 정체는 mma 가 아니라 **C drain** — uc²(≈140²) trailing 결과를 **global 에
write** 하는 bandwidth (검산: 50 front × 140² × 4B × 64 batch × 30 ≈ 84ms ≈ 측정 92ms). big front
이 global-resident 라 결과를 global 에 써야 하는 비용. **즉 TC 도 staging 도 느린 게 아니라, 비용은
(a) panel barrier(57–63%) + (b) C drain memory(≈35%).** TC 가 가속할 "느린 연산"이 없음.
(앞서 "staging 지배"라 적은 표현 정정 — 실제 trailing memory 비용은 C drain.)

## 3. 근본 원인: big trailing GEMM 은 thin (K=nc≈20)

trailing = **C[uc×uc] ← C − L[uc×nc]·U[nc×uc]**, 즉 **M=N=uc, K=nc**. 실측 (`--dump-fronts`):

| case | big front 수 | M=N=uc (med/max) | **K=nc** | 시스템당 trailing FLOP |
|---|---|---|---|---|
| 25K | 4 | 119 / 123 | **12** | 1.4 MFLOP |
| USA | 62 | 137 / 205 | **20** | 51 MFLOP |
| 70K | 53 | 141 / 242 | **20** | 47 MFLOP |

전형적 GEMM = **~140×140, K=20**. mma m16n8k8 기준 K=20 = **2.5 K-tile 뿐**.
- TC GEMM 은 A/B 타일 load 를 K-loop 으로 amortize 하는데, K=2.5tile 이면 amortize 할 게 없어
  **L/U staging 이 mma 를 압도** → memory-bound.
- 산술강도: FLOP=2·140²·20≈0.78M, data=C(140²)+L/U(2·140·20)≈100KB → **C(140²) write 가 지배**.
  연산은 ns, 메모리는 ~100ns.
- amalgamation 으로 front 를 키워도 uc(M,N)만 커지고 **K=nc(panel width)는 거의 안 큰다** → TC 를
  근본적으로 못 살린다. (panel_cap 48 은 tf32 −7% 를 줬으나 fp32 가 큰 front 에서 느려진 것이지
  TC 가 빨라진 게 아님 — TC pipe 여전히 낮음.)

## 4. 함의 + 다음 lever (panel LU / U-solve 최적화)

big 커널 시간은 trailing(TC, 0.2%)이 아니라 **panel LU + U-solve (61%) + extend (19%)** 가 지배
(phase breakdown: doc 29). panel 은 global-latency 가 아니라 **barrier-bound**(순차 pivot 의
1024-thread `__syncthreads` 체인; L2 hit 92%, DRAM 1.7%)임이 doc 29 에서 확인됨. TC 가 안 보이는
이유 = TC 가 가속하는 trailing 이 커널 시간의 극소수(19%)라서.

→ **가설 (다음 연구)**: panel LU / U-solve 를 최적화하면 (a) 커널이 빨라지고, (b) trailing 이
커널의 더 큰 비중이 되어 TC 기여가 상대적으로 드러난다. 단, trailing 자체는 thin-K memory-bound
라 TC-vs-fp32 격차는 작게 유지된다 (panel 최적화는 fp32/tf32 양쪽에 동등). TC 가 fp32 를 실제로
이기려면 trailing 을 compute-bound 로 만들어야 하는데(예: front 를 shared-resident 로 유지해
staging 제거), big front 은 shared 예산 초과라 어렵다.

후속: 03-optimization-notes/29 (panel LU / U-solve 최적화 연구) 로 이어짐.
