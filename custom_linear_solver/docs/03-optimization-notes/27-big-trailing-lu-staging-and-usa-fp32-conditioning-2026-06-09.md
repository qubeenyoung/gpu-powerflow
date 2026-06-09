# Big-trailing L/U staging + USA fp32 conditioning floor

**작성일**: 2026-06-09
**범위**: (1) scalar big trailing 의 L/U shared staging 최적화 (반영, commit `23ec691`). (2) USA fp32
multi-batch 의 큰 relres 가 데이터셋(conditioning) 문제임을 cuDSS 레퍼런스로 확인한 진단.

---

## 1. 반영: factor_big_staged — scalar big trailing 의 L/U staging (commit 23ec691)

### 문제
big tier(fsz>128)의 FP64/FP32 경로(`trailing_update_scalar`)는 front 가 global-resident 라
rank-nc trailing(C ← C − L·U)에서 **각 출력이 L 행·U 열을 global 에서 직독** — L/U 원소가
출력당 재사용되는데 staging 이 없어 **~uc 배(USA 145×, 25K 120×) 중복 global read**. mid tier 는
이미 `trailing_update_staged` 로 L/U 를 shared 에 올려 재사용을 포착하지만 big 은 안 했다.

산술강도: 출력당 nc FMA / 2nc global load = **1/16 FLOP/byte (fp64)** → roofline 상 memory-bound
(L2 가 일부 완화). big-front L/U 패널은 작아(USA uc≤205, nc≤32 → staging 22–64KB) 96KB 예산에 들어간다.

### 변경
`factor_big_staged<T>`: front 는 global 유지, **L/U 패널만 shared 에 staging**(`trailing_update_staged`).
dispatch 가 filled big tier(fp64/fp32)를 `2·level_max_nc·level_max_uc ≤ 96KB` 일 때 라우팅, 초과시
`factor_big`(scalar) fallback. `CLS_NO_BIG_STAGED` A/B 토글.

### 측정 (filled big, B=64)
| | factor_big 커널 | end-to-end factor |
|---|---|---|
| USA fp64 | **−23%** (909→700ms) | **−11%** |
| 25K fp64 | | **−5.7%** |
| case8387 fp64 | | **−5.9%** |
| fp32 (USA/25K) | | −1.6~1.7% |

B=1 big 은 underfill(multi-block 경로)이라 미적용. 안정 케이스 정확성 불변 (25K fp64 e-13,
case8387 fp64 e-14). compute-sanitizer racecheck/synccheck/initcheck/memcheck 전부 clean.

---

## 2. 진단: USA fp32 multi-batch 의 큰 relres 는 데이터셋 conditioning floor

### 현상
USA(SyntheticUSA, ~80K) fp32 B=64 의 `batch_relres` 가 run 마다 **~1e-3 ↔ 0.02–0.06 으로 bimodal**.
fp64 는 e-12~e-11 으로 안정. 처음엔 staging 버그로 의심했으나 **커밋된 baseline(staging 없음)도
동일하게 10/16 run 이 >0.01** → staging 무관.

### cuDSS 레퍼런스 대조 (J.mtx, rhs.mtx; `/opt/nvidia/cudss` 0.7.1)
| solver | precision | relres (B=1 / B=64) |
|---|---|---|
| **cuDSS** | fp64 | 2.02e-12 / 2.11e-12 |
| **cuDSS** | **fp32** | **8.93e-4 / 1.10e-3** |
| ours | fp64 | e-12 ~ e-11 (ND 변동) |
| ours | fp32 | ~1e-3 (good) ↔ 0.02–0.06 (bad) |

→ **cuDSS 도 USA fp32 floor 가 ~1e-3** (fp64 는 e-12). 즉 **USA Jacobian 이 fp32(7자리)로는
~1e-3 가 한계인 ill-conditioned 행렬** — 레퍼런스 솔버가 확인. 우리 "good" run(~1e-3)은 cuDSS 와 일치.

### 두 성분으로 분해
1. **데이터셋 floor ~1e-3** (conditioning × fp32): cuDSS 가 고정 ordering 으로 안정적으로 1e-3.
   불가피 — fp32 정밀도 한계.
2. **우리 parallel-ND ordering 변동** (1e-3 → 0.02–0.06): cuDSS 는 고정 ordering 이라 안정인데,
   우리 multi-threaded METIS-ND 는 run 마다 ordering 이 달라 일부 ordering 이 fp32 오차를 증폭.
   **사전부터 존재**(baseline 동일), staging 무관.

### 함의 / 후속 lever (미실행)
- USA 처럼 ill-conditioned 한 케이스는 **fp32 부적합** — fp64 또는 mixed(fp32 factor + fp64
  iterative refinement)가 필요.
- **ND ordering robustness**: deterministic ordering 또는 conditioning-aware ordering 으로
  fp32 변동을 cuDSS 수준(고정 1e-3)으로 낮출 여지. (현재 parallel-ND 는 속도 우선·비결정적.)
- **iterative refinement (Ootomo-Yokota 류)**: fp32 factor + 한두 번 fp64 refinement 로 relres 를
  e-6 이하로. docs/18 의 M5 lever 와 연결.

---

## 3. 교훈 (검증 절차)
flaky 벤치마크(USA fp32)에서 **baseline 동일조건 대조 없이** correctness 를 판단해, 유효한 −11%
최적화를 revert 직전까지 갔다. 정확성 검증은 항상 (1) **안정 케이스**(fp64 e-13급) + (2) **baseline
동일조건 비교** 로 한다. 단일 flaky 케이스의 절대 relres 로 판단하지 않는다.
