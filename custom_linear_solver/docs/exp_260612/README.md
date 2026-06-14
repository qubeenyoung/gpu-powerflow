# EXP 260612 — factorize 1.2× 가속 시도 (B=1, fp32; symbolic·reorder·factorize 광범위 관찰)

> **상태**: 측정·구현 완료   **날짜**: 2026-06-12   **GPU**: RTX 3090 (sm_86, 82 SM)
> **베이스라인**: 현재 production factorize, B=1, parallel-ND default.
> **⚠️ 정밀도 정정([03](03-precision-correction-and-fp32.md))**: 초기 측정은 `--single-precision fp32`(= fp32 입력 + **fp64 factor**, ncu 가 `<double>` 노출)였음. 진짜 fp32 factor 는 `--precision fp32`(`<float>`). 두 해석의 결과를 모두 보고한다.
> **한 줄 결론**: B=1 factorize 는 **소거트리 깊은 레벨의 under-fill(occupancy) 바운드**이며, scheduling/tiling/sync/amalgamation/thread-width 는 모두 구조벽(1 block/SM, per-front latency-bound)에 막혀 무효다. **유일하게 듣는 레버는 ordering**. factorize-free **`tail_cube` proxy + best-of-k 결정적 선택**(`CLS_ORDER_K`)이 그 천장을 회수.
>
> | 해석 | 13K(8387) | 25K | 70K(USA) | 비고 |
> |---|---:|---:|---:|---|
> | fp64 factor (`--single-precision fp32`) | 1.12× | **1.15×** | 1.13× | 천장 1.13–1.18× ([01](01-findings-b1-factorize-occupancy.md)) |
> | **진짜 fp32 factor (`--precision fp32`)** | **1.14×** | 1.05× | 1.04× | 대형은 default 가 이미 near-optimal ([03](03-precision-correction-and-fp32.md)) |
>
> **목표 1.2× 는 어느 정밀도에서도 세 case 일괄 미달**: under-fill 벽 + ordering 천장이 대형 case 에서 1.2× 아래. B=1 에서 그 갭을 닫는 2차 레버는 (실험·문헌 모두) **존재하지 않는다**. worst-seed 대비로는 1.2×+ 를 만들 수 있으나 cap-inflation(notes 54/55 금지).

> **후속(2026-06-13) — 범위 확장 + 구조적 성과/정정**: 이후 연구가 B=1 단일 목표를 넘어 batched/구조 영역으로 확장됨. 핵심:
> - [`10`](10-panel-resident-mid-kernel.md) **panel-resident 커널 ✓** — batched 대형 front 의 whole-front shared 천장 제거 → DRAM 2–32%→55–65%, **fp32 SyntheticUSA B=64 −11%, ACTIVSg25k −4.3%**(default-on). 06-10 cuDSS 스윕 대비 custom factorize 우위 usa B=64 4.4×→5.05×.
> - [`11`](11-b1-tensorcore-ozaki.md) **B=1 레버 정정 ✓** — 위 "ordering 만이 B=1 레버"는 TC 를 분리 안 한 결론이었음. **B=1 은 latency-bound 라 TC trailing 이 critical path 를 줄여 이득**(USA −17%); **Ozaki-TC 로 fp32 정확도 유지하며 25K −8.5% / USA −17.5%**. (B=64 와 정반대 — 거긴 TC 가 occupancy 깎아 손해.)
> - 음성 결과: [`07`](07-deep-k-amalgamation.md) deep-K amalgamation ✗, [`08`](08-gather-favorable-phase-batched.md) gather/phase-batched ✗ (scatter 구조적 우위). [`09`](09-scatter-factor-occupancy-barriers.md) register/barrier 미세최적화 ~2–3%.

---

## 케이스 매핑 (데이터 가용성)

이 환경에는 lab 표준 `case13659pegase`·`case_ACTIVSg70k` 파일이 **없음**. J-차원 사다리로 최근접 대체:

| 목표 | 대체 케이스 | J dim | nnz | 비고 |
|---|---|---:|---:|---|
| **13K** | `case8387pegase` (+`case6468rte`) | 14908 | 110572 | PEGASE 계열(소형, mid-dom) |
| **25K** | `case_ACTIVSg25k` | 47246 | 318672 | **유일한 정확 매칭(이름)** |
| **70K** | `case_SyntheticUSA` | 156255 | 1052085 | 유일한 big-front-dom 대형 case |

모든 수치는 fp32, B=1, `factorize_ms`(repeat 25–30, warmup 8–10 median).

---

## 헤드라인 결과

honest 베이스라인 = production **parallel-ND default** 의 5-run median (parallel ND 는 비결정적):

| case | baseline median | (range) | **best-of-8 (`CLS_ORDER_K=8`)** | sel seed | **speedup** | ordering 천장(26-seed oracle) |
|---|---:|---:|---:|---:|---:|---:|
| 13K (8387) | 0.572 | 0.538–0.624 | **0.509** | 49 | **1.124×** | 1.121× |
| 25K (ACTIVSg25k) | 1.896 | 1.685–1.998 | **1.646** | 47 | **1.152×** | 1.175× |
| 70K (USA) | 8.476 | 7.887–9.001 | **7.480** | 44 | **1.133×** | 1.133× |

- best-of-8 proxy 가 **13K·70K 는 천장에 도달**, 25K 는 1.152×(천장 1.175× — best seed 7 이 42–49 윈도 밖).
- 부수효과: parallel-ND 의 run-to-run **비결정성(±8–24%)을 결정적으로 제거**. relres 불변(같은 valid permutation).

---

## 무엇이 듣고 무엇이 안 듣나 (이번 실험 + 문헌)

| 레버 | 결과 | 근거 |
|---|---|---|
| **ND ordering 선택** (best-of-k) | ✅ **+12–15%** | 본 실험. fp32 seed 간 factor 시간 **15–67% 변동**. 천장 ~1.13–1.18× |
| multistream subtree 병렬 | ◑ 이미 ON(~1.10×) | 기본값. 끄면 1.10–1.12× 손해 |
| **subtree stream 수 8→16** | ✗ 무효 | [§3](01-findings-b1-factorize-occupancy.md) — deep level 에 독립 front 가 물리적으로 부족(2–66개) |
| **amalgamation (panel width 8→32)** | ✗ 무효 | 25K/70K 평탄(1.852/7.47), 8387 약간 악화. 문헌도 "root 에서 거부" |
| big-front tiling | ✗ 회귀(이전 노트) | re-staging 비용. 문헌: B=1 root 전용 기법 부재 |
| mid sync 감소(fewsync) | ◑ TF32 한정, fp32 무관 | working-tree 프로토타입. baseline=fp32 라 미적용 |

→ **귀결**: B=1 occupancy 천장(1 block/SM, deep-level 독립 front 부족)은 실재하며, ordering 외 레버는 전부 그 벽에 막힌다. 이는 [문헌조사](02-literature-review.md)와 일치(STRUMPACK/CHOLMOD/cuDSS 는 *큰* root 를 가정; B=1 작은-root 는 "literature 의 빈 구멍").

---

## 산출물

- **코드**: `src/analyze/pipeline.cpp` — `CLS_ORDER_K` (default 1 = 기존 동작). >1 이면 결정적 serial-ND k-seed 중 `tail_cube` proxy 최소 plan 선택. `ordering_cost_model()` = Σ(under-filled large level 의 maxfsz³).
- **문서**:
  - [`01-findings-b1-factorize-occupancy.md`](01-findings-b1-factorize-occupancy.md) — 구조 진단, cost model 유도·검증, best-of-k, negative results.
  - [`02-literature-review.md`](02-literature-review.md) — B=1 GPU multifrontal 문헌 종합(ordering·MAGMA-native·look-ahead·amalgamation·power-flow).
  - [`07-deep-k-amalgamation.md`](07-deep-k-amalgamation.md) — **deep-K supernode amalgamation (CLS_AMALG_K) = ✗ 회귀.** subtree 흡수로 nc 를 두껍게(가능). fill↓(0.87×)·level↓(29→16)에도 factor 는 모든 cap 에서 악화(+1.4~+21.6%). 전력망+ND front 는 thin-K 구조적 천장(work-wt nc ~4.6 plateau) → AI 벽 미돌파.
  - [`15-measured-best-of-k-ordering.md`](15-measured-best-of-k-ordering.md) — **ORDERING 실익 ✓ 구현·측정: measured best-of-k(CLS_ORDER_MEASURE_K).** doc 14가 지목한 단 하나의 레버 구현. tail-cube proxy(CLS_ORDER_K)는 **anti-informative**(1순위 픽이 default보다 8% 느림, fill은 시드 무관 동일) → 시드별 **실제 factorize 시간 측정**으로 최적 선택. **B=1 6~13%**(8387 −8%, 25k −11%, USA −8.4%) + parallel-ND 비결정성(USA 18% swing) 제거 + 정밀도 적응(fp64는 다른 시드). 비용=ND 빌드 지배(K16 ~33s, 깨진 proxy와 동급) → factorize 재사용(time-series/N-1) 시에만 amortize. env-gated default-off, production 불변.
  - [`14-tailored-ordering-conclusion.md`](14-tailored-ordering-conclusion.md) — **ORDERING 결론 ✗: tailored ND 반증.** GPU-objective recursion(doc 12 "win"은 single-seed+UFACTOR 버그 artifact → best-of-k 대비 METIS와 동급) + **전기적-weighted bisection(Stage 2)도 모든 depth에서 악화(+9~73%, fill +28%)** — 약한 tie-line cut은 작은/균형 separator가 아님. METIS fill 목적이 이 문제엔 near-optimal. 실익은 **깨진 proxy best-of-k(CLS_ORDER_K) 선택 수정**(단일 시드보다 나쁨)뿐 → **doc 15에서 measured 선택으로 구현·회수.**
  - [`13-ordering-literature-review.md`](13-ordering-literature-review.md) — METIS ND 내부(multilevel/FM) + 선행연구(etree-height=critical-path, cuDSS=METIS-기반 custom ND). fill≈work, ordering 천장 bounded.
  - [`12-gpu-objective-nested-dissection.md`](12-gpu-objective-nested-dissection.md) — ⚠️ **정정됨(doc 14)**: custom GPU-objective ND (CLS_GPU_ND), 보고된 win은 artifact. 재귀/목적/정지 우리 소유 + bisection은 METIS 재사용. fill 대신 **균형(balance) 목적** 최적화 → METIS 대비 −2~5%(leaf 튜닝 −10.6%), **fill 은 오히려↓(0.95–0.98×)**, fp64 1e-13. 기전 확정: λ=0(최소 분리자=fill)는 오히려 악화, **imbalance penalty(λ>0)가 win** — METIS fill 목적이 GPU 병렬성을 남겨둔다는 증명. Ozaki-TC 와 stack.
  - [`11-b1-tensorcore-ozaki.md`](11-b1-tensorcore-ozaki.md) — **B=1 = 정반대 체제 ✓: TC 가 레버.** B=1 은 block-starved(occupancy 무의미) → 패널 중립. trailing 의 mma 가 critical path 단축 → B=1 에서 TC 이득(USA −17%). B=64 와 정반대(거긴 TC 가 occupancy 깎아 손해). **Ozaki-TC** 로 fp32 정확도 유지하며 25K −8.5% / USA −17.5%. 두 체제 최적 커널 정반대 표 정리.
  - [`10-panel-resident-mid-kernel.md`](10-panel-resident-mid-kernel.md) — **STRUCTURAL ✓: panel-resident mid 커널.** B=64 시간의 84%는 꽉 찬 launch(underfill 아님)인데 whole-front shared 때문에 occupancy 8–32%·DRAM 35–40%(큰 front 은 2–3%)에서 굶음. L/U 패널만 shared, CB 는 global+fused extend → shared fsz²→nc(fsz+uc), **DRAM 2–32%→55–65%**. SyntheticUSA(70K) **−9.3%@B64**, ACTIVSg25k −1.4%. fsz≥112 & blocks≥2·SM 게이트, default-on, 회귀 없음.
  - [`09-scatter-factor-occupancy-barriers.md`](09-scatter-factor-occupancy-barriers.md) — **scatter factorize register+barrier 절감 = ✓ default-on ~2–3%.** gather 코드를 hot path 에서 컴파일 제외(mid 60→51, big 48→40 reg) + sync-free U-solve(nc barrier→1) + conditional row-fused LU(nc≤16 & fsz≤96). 25k −2.7%, USA −2.5%. mid/big 경계·TC 는 무효(메모리 latency·barrier bound). 정직한 천장: micro-lever 당 ~1%.
  - [`08-gather-favorable-phase-batched.md`](08-gather-favorable-phase-batched.md) — **gather-favorable 구조(phase-batched assembly, CLS_ASM_MODE=gather_pb) = ✗ 회귀(공정 비교).** assembly 를 별도 high-occupancy pre-pass 로 분리 → 오히려 악화(+78~100%, fused gather 의 −15%보다 나쁨). assemble 커널은 의도대로 high-occupancy(34reg, 57–67%)지만, phase 분리가 dense front(80% 0) global round-trip 을 강제. scatter 우위는 구조적(streaming memset + atomic-free unique scatter + factor-fused extend-add). 최선의 gather=fused-shared(−15%), 어떤 layout/phasing 도 격차 미해소.

## 재현

```bash
BIN=build/custom_linear_solver_run
C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
# baseline (parallel ND, nondeterministic — 여러 번)
$BIN $C --single-precision fp32 --repeat 30 --warmup 10
# best-of-8 ordering (deterministic, tail_cube-selected)
CLS_ORDER_K=8 $BIN $C --single-precision fp32 --repeat 30 --warmup 10 --metis-seed 42
```
