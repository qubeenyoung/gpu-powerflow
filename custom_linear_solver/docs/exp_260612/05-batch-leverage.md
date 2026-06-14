# 배치 효율성 레버리지 — B=1 벽의 유일한 탈출구 (실측)

> **상태**: 완료   **날짜**: 2026-06-12   **GPU**: RTX 3090 (82 SM)
> **한 줄**: B=1 의 under-fill 벽([04](04-b1-impossibility-analysis.md))은 닫힌 부등식 `P_L ≤ cnt_L` 때문. 배치는 이를 `P_L ≤ cnt_L·B` 로 **열어** per-system factorize 를 **5.2–13.4×(B=64)** 가속한다 — 1.2× 목표를 압도. 단 이는 **처리량**(같은 sparsity 다수 시스템 필요)이며 power-flow 워크로드와 정확히 맞는다.

## 1. 배치 fill = 지배적 레버

per-system factorize_ms (fp32, seed7):

| case | B=1 | B=4 | B=16 | B=64 | **B1/B64** |
|---|---:|---:|---:|---:|---:|
| 13K (8387) | 0.313 | 0.101 | 0.0386 | 0.0239 | **13.1×** |
| 25K | 0.708 | 0.228 | 0.114 | 0.0969 | **7.3×** |
| 70K (USA) | 2.118 | 0.731 | 0.438 | 0.404 | **5.2×** |

이론 그대로: deep level 의 독립 블록 수가 `cnt_L → cnt_L·B` 로 늘어 idle SM 이 배치 차원으로 채워진다. **B=1 에서 막혔던 occupancy 가 풀리는 유일한 정공법.**

## 2. 효율 곡선과 knee (최소 배치)

per-system 시간이 포화되는 지점 = GPU 가 가득 차는 배치:

| case | knee(포화 시작) | 근거 |
|---|---|---|
| 70K (USA) | **~B16** | big front(cnt~2)라 적은 시스템으로 82 SM 채움; B16 이후 memory-bound, 한계이득 |
| 25K | ~B16–32 | mid-dom |
| 13K (8387) | **B64+** | 작은 front 다수라 채우려면 더 큰 배치 필요(B64 까지 계속 개선) |

> **운영점**: 대형 grid 는 **B=16 만으로 대부분 회수**(USA 4.8×, 25K 6.2× per-system). knee 아래는 per-system 급감, knee 위는 포화(처리량은 여전히 선형).

## 3. 배치 내부 추가 레버 (포화 후)

knee 를 넘어 memory/compute-bound 가 되면 TF32 tensor-core 가 추가로 듣는다. per-system @ B=64:

| case | fp32 | tf32 | **추가 이득** |
|---|---:|---:|---:|
| 13K (8387) | 0.0238 | 0.0238 | 1.00× (소형, 미포화) |
| 25K | 0.0970 | 0.0842 | **1.15×** |
| 70K (USA) | 0.404 | 0.370 | **1.09×** |

→ storyline 의 TF32/fused-trailing/staging 기여가 **여기서 살아남는다**(B=1 에선 starved 였음). 배치 fill × tf32 가 곱해진다.

## 4. ordering 은 배치에서 washes out

seed best/worst 민감도:

| case | B=1 spread | B=64 spread |
|---|---:|---:|
| 25K | **1.26×** | 1.02× |
| 70K (USA) | **1.22×** | 1.02× |

→ 배치가 GPU 를 채우면 under-fill critical-path 우위가 사라져 ordering 이 무의미. **`CLS_ORDER_K` best-of-k 는 B=1 전용 레버**(B≥16 에선 불필요). [04](04-b1-impossibility-analysis.md) §6 예측과 일치.

## 5. 정직한 caveat — 무엇을 요구하나

- **처리량(throughput)이지 단일 latency 아님**: B=64 total = 64×per-system. *다수* 시스템을 factorize 할 때만 이득. 단일 시스템 1개의 wall-time 은 그대로 B=1 천장.
- **같은 sparsity pattern 필요**(값만 다른 다수 시스템). power-flow 는 이게 자연스러움:
  - **NR 반복**: 같은 J 구조, iteration 마다 값만 갱신.
  - **N-1 contingency**: 수백–수천 시나리오, 동일 grid topology.
  - **time-series / QSTS / Monte Carlo / OPF inner loop**: 동일 패턴 반복.
- **순수 단일 실시간 1-shot**(예: 단발 state estimate)이면 배치 불가 → B=1 천장(1.05–1.19×)에 갇힘.

## 6. 구현 상태와 권고

- **이미 사용 가능**: `--batch B` 가 B 시스템을 `grid(level_size, B)` 로 처리. 레버는 *지금* 켤 수 있다(코드 변경 불필요).
- **권고 운영점**: 대형 grid `B≈16–32 + tf32` → per-system **6–8× + 9–15%**. 소형은 더 큰 B 가 유리.
- **추가 최적화 여지**(포화 regime 를 더 밀기): staging DRAM 트래픽 절감, mid/big tier 경계 batch-aware 재조정, fused trailing — storyline B 축.

## 7. **같은 B=64 안에서 더 가속 가능한가?** (배치 fill 위에 추가 레버)

배치로 GPU 를 채운 *뒤* 의 병목은 B=1 과 **완전히 다르다**. ncu (B=64, fp32, graph-off):

| case | 지배 커널 | grid | warps_active | sm__thru | DRAM | 진단 |
|---|---|---:|---:|---:|---:|---|
| 25K | factor_mid | 370 | 18–33% | 7–11% | 5–8% | **shared-residency 1 block/SM** → narrow block(128 thr=4 warp)이 warp slot 낭비 |
| 70K | factor_big | 2546 | **81%** | 31% | **34%** | 큰 레벨은 **memory-bound 접근**, 깊은 레벨은 17–20% 로 저하 |

**추가 레버 실측 (per-system @ B=64):**

| lever | 25K | 70K | 비고 |
|---|---:|---:|---|
| baseline (fp32, heuristic) | 0.0969 | 0.404 | — |
| + mid thread 128→256 | 0.0927 (**1.05×**) | 0.381 (**1.06×**) | shared-limited level 의 warp 점유↑. t1024 는 오히려 악화(blocks/SM↓) |
| **+ tf32 tensor core** | **0.0841 (1.15×)** | **0.370 (1.09×)** | B=1 에선 starved 였던 TC 가 GPU full 이라 부활 — **지배 레버** |

→ **같은 B=64 에서 ~1.1–1.15× 추가 가속 가능**(tf32 지배 + thread tuning ~5%, 둘은 거의 안 겹침).

**근본 벽**: mid 커널은 **whole-front shared residency**(fsz²·4B, fsz=120→57KB) 로 **1 block/SM** 에 묶임 → B=64 에서도 occupancy 가 낮다(warps 18–33%). 이걸 깨려면 front 를 shared 에 안 올리고 분할(non-resident/tiled)해야 하는데, **B=64 에서도 re-staging 으로 회귀**(note 07: 0.62–0.68×). 즉 **tf32·thread tuning 너머의 큰 추가 가속은 같은 알고리즘으론 막혀** 있고, 그 다음은 **batched-GEMM 화(시스템 차원을 trailing GEMM 의 batch 로)** 같은 커널 재설계 영역(미착수).

> **요약**: B=64 자체도 **tf32 로 ~1.1–1.15× 더** 가속됨(thread tuning +5%). 한계는 mid 의 shared-residency 1 block/SM 이며, 그 너머는 커널 재설계가 필요.

## 결론

> "B=1 1.2×"는 구조적 불가([04](04-b1-impossibility-analysis.md))지만, **그 한계의 원인(under-fill)이 곧 배치의 기회**다. 같은 under-fill 을 B 로 채우면 per-system **5–13×**, 그 위에 같은 B=64 안에서 **tf32 로 +9–15%** 더. 실제 power-flow 워크로드(NR·contingency·time-series)와 맞는다. **레버는 B=1 가속이 아니라 배치 throughput 에 있고, 배치 안에서도 tf32 가 추가로 듣는다.**

## 재현
```bash
BIN=build/custom_linear_solver_run; C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
$BIN $C --precision fp32 --batch 16 --repeat 15 --warmup 5 --serial-nd --metis-seed 7   # batch_factor_per_sys_ms
$BIN $C --precision tf32 --batch 64 --repeat 15 --warmup 5 --serial-nd --metis-seed 7   # +tf32 lever
```
