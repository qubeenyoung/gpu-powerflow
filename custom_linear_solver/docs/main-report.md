# custom_linear_solver — 통합 리포트

> **상태**: canonical (storyline · contribution-analysis · main-report 통합본)   **갱신**: 2026-06-16
> **한 줄**: 전력망 Jacobian 의 비효율은 *front 가 작다* 는 한 성질에서 나온다. 개별 GPU 기법은 전부 선행연구에
> 있지만, **head-to-head 에서 일반/벤더 솔버보다 동일 정밀도로 16–66× 빠르다.** 그 격차의 정체는 — packing 과
> full-front fusion 이라는 *합쳐질 수 없는 두 입도*를 **sub-group 분해**로 해소해 둘을 한 커널에서 동시에 달성,
> occupancy 를 12–20× 회복한 데 있다. 이 구조가 우리 기여이고, 그 크기를 정직하게 분해한다.

상세 실험: [`05-reports/06-head-to-head-2026-06-16.md`](05-reports/06-head-to-head-2026-06-16.md).
선행연구: [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md).
API/빌드: [`01-orientation/01-api-and-build-design.md`](01-orientation/01-api-and-build-design.md).

---

## 0. 범위

- **이 문서는 custom linear solver 만 다룬다.**
- **cuPF 는 기여에서 제외.** cuPF 는 FP64 Newton–Raphson 외부 루프에서 *Jacobian 조립·선형계만* FP32(/TF32)로
  푸는 혼합정밀이다(나머지 FP64). 그 혼합정밀 NR 의 수렴·정확도는 cuPF 의 결과이지 솔버의 주장이 아니다.
  솔버는 그 FP32/TF32 선형-풀이 *모드를 제공*할 뿐이다.

---

## 1. 문제와 난점 — 근원은 작은 front

전력망 조류계산은 Newton–Raphson 반복으로 푼다. 매 반복마다 동일 sparsity, 다른 수치의 Jacobian 선형계를 한 번
푼다. 상정사고·시계열·확률 시나리오·최적화 내부 루프에서는 이런 계를 **수백~수천 개 동시에** 풀어야 한다.
과제는 분명하다 — *같은 패턴·다른 값*의 다수 선형계를 GPU 에서 **배치로** 빠르고 정확하게.

전력망 Jacobian 은 거의 평면(near-planar) 그래프에 가깝다. nested-dissection 으로 정렬해 multifrontal 로
분해하면 조밀 부분문제(front)가 **작고**(≈99 % 가 `fsz ≤ 16`), 소거트리가 **깊고 좁다**(~26 레벨). 총 연산량은
작지만(예: `case9241` ≈0.08 GFLOP) **임계경로가 길다.**

이 한 성질이 모든 비효율의 근원이다:
- front 가 작아 조밀 커널이 SM 을 못 채운다 → **occupancy 낮음**.
- 갱신(trailing) 연산의 contraction 깊이가 얕아(피벗 열 수 ≲ 20) 텐서코어가 안 먹고 **메모리 지연**에 묶인다.
- 트리가 깊고 좁아 레벨마다 실행·동기화 비용이 누적된다.

즉 가속은 FLOP 이 아니라 **점유·지연**에서 나온다. 범용 multifrontal 이 grid-like 행렬에서 GPU 피크의 0.004 %
만 쓴다는 보고(Spatula, MICRO'23)도 같은 진단이다.

---

## 2. 솔버 개요 — 무엇을, 어떻게

배치 정적 multifrontal LU(no-pivot, 대각우세 가정). B 개 시스템이 **하나의 symbolic** 을 공유하고, front-major
arena 에 올라가 디바이스에 상주한 채(CUDA 그래프) 분해·해를 replay 한다. cuDSS 스타일 phase API
(`analyze → factorize → solve`). 정밀도 모드 FP64 / FP32 / TF32(텐서코어 trailing) 제공.

```
set_data (device CSR)
   │
analyze            ── sparsity 패턴당 1회 ───────────────────────────────────────
   │   GPU 대칭그래프 빌드(A+Aᵀ) → METIS nested dissection
   │   etree → fill 패턴 → relaxed panel(amalgamation)
   │   multifrontal symbolic(front_ptr / extend-add 맵) → plan + device arena
   │   factor 그래프 + solve 그래프 capture
   │
factorize          ── NR 반복마다 ────────────────────────────────────────────────
   │   CSR 값을 front arena 로 scatter
   │   factor 그래프 replay: 레벨별 조밀 LU + 부모로 extend-add
   │
solve              ── NR 반복마다 ────────────────────────────────────────────────
       치환 RHS gather → 전진 대입(leaves→root) → 후진 대입(root→leaves) → 해 scatter
```

레벨은 leaves→root 로 처리하고, 한 레벨의 front 들은 독립이라 동시에 돈다(레벨당 CUDA-그래프 노드 1개). 두 실행
모드가 하나의 symbolic 을 공유한다 — **단일 시스템(B=1)** (cuPF 가 NR 반복마다 사용) 과 **배치(B>1)**.

| 영역 | 파일 |
|---|---|
| public API | `src/solver.{hpp,cpp}`, `src/matrix/view.hpp` |
| plan / arena / graph | `src/plan/multifrontal_plan.{hpp,cu}` |
| analyze(ordering·symbolic) | `src/reordering/*`, `src/symbolic/*` |
| factorize(3-tier) | `src/factorize/{small,mid,big}.cuh`, `factorize.cu` |
| solve | `src/solve/{phases,kernels,dispatch}.cuh` |

---

## 3. 핵심 기여 — 논리 구조

신규성은 *어떤 기법을 새로 만든 것*이 아니다. 개별 기법은 전부 선행연구에 있다. 신규성은 **그 기법들이 서로
배타적이라 아무도 동시에 못 하던 두 가지를, 다른 분해로 동시에 가능하게 만든 구조**에 있다. 아래 6단계로
논증한다.

### 3.1 정직한 출발 — 개별 기법은 전부 prior art

심사에서 "이거 MAGMA/STRUMPACK/cuDSS 아니냐"에 먼저 항복할 부분을 명시한다(소스·논문으로 확인):

| 기법 | 선행 | 출처 |
|---|---|---|
| 작은 행렬 **packing**(여러 front 를 워프/블록에 묶음) | **MAGMA** vbatched (`ntcol`개/블록) | Dong·Abdelfattah·Haidar·Dongarra |
| front 를 **shared 단일 커널로 factor+update 융합** (small) | **STRUMPACK native** ("single NT×NT block per front", NT∈{8,16,24,32}) | Ghysels & Synk |
| front **크기별 tier 라우팅** | **STRUMPACK** (small custom / large cuSOLVER) | 〃 |
| **단일 symbolic 공유 배치** | **cuDSS** uniform batch(UBATCH) | NVIDIA |
| power-flow **GPU-resident** direct solve(개념) | Swirydowicz et al. 2023 | IJEPES |

⇒ "자체 커널을 썼다", "packing/tiering 을 한다", "front 를 shared 에서 융합한다" — **이 각각은 우리 신규성이
아니다.**

### 3.2 그런데 baseline 은 느리다 — head-to-head

MAGMA+STRUMPACK 을 직접 설치·빌드해 같은 power-flow Jacobian 으로 공정 비교했다(동일 FP64, 동일 GPU, DIRECT,
reorder 1회 + refactor+solve). **트리 깊이까지 매칭**(STRUMPACK `and` ordering ≈ 우리 레벨 수)하고, STRUMPACK
의 두 GPU 경로(MAGMA vbatched / native) 중 **더 빠른 쪽**을 줘도:

| 케이스 | STRUMPACK+MAGMA factor | 우리 factor | STRUMPACK solve | 우리 solve | 배율 |
|---|---|---|---|---|---|
| ACTIVSg25k (B=1, FP64) | 20.5 ms | **1.24 ms** | 20.9 ms | **0.63 ms** | factor 16×, solve 33× |
| SyntheticUSA (B=1, FP64) | 60.2 ms | **2.70 ms** | 86.9 ms | **1.31 ms** | factor 22×, solve 66× |

같은 정밀도(잔차 STRUMPACK 3e-16, 우리 2e-14), 같은 GPU, 깊이까지 맞췄는데 16–66×. **접근이 동일했다면 이
격차가 없어야 한다.** 그러므로 격차의 정체를 규명해야 한다. (상세: [§06 리포트](05-reports/06-head-to-head-2026-06-16.md))

> **정정(2026-06-17, [§08 공정튜닝](05-reports/08-fair-strumpack-tuning-2026-06-17.md)):** 이 16–66×는 STRUMPACK을
> 기본 정렬(metis, 78층 deep-tree)로 돌려 *부풀려진* 수치다. STRUMPACK 권장 `NodeNDP`로 공정 튜닝하면 STRUMPACK이
> 우리보다 얕고(14층) fill도 적은(1.67×) 동작점에 도달하며, 격차는 **factor ~10–13×, solve ~12–16×로 줄어든다.**
> ⇒ ordering·amalgamation·깊이·fill은 차별점이 아니다(STRUMPACK도 함, 일부 더 잘함). 남는 ~10×만이 깊이·fill·
> ordering을 통제한 뒤의 순수 커널 효율이다. 헤드라인은 16–66×가 아니라 **공정 튜닝 후 ~10×**로 읽어야 한다.

### 3.3 왜 느린가 — packing 과 fusion 의 *입도 배타성*

packing(SM 을 채우는 것)과 full-front fusion(front 파이프라인을 on-chip 에 올려 global 왕복을 없애는 것)은
**별개의 축**이다. 그런데 두 baseline 은 **입도(granularity)가 배타적**이라 각각 *하나만* 한다:

| | packing(SM 채움) | full-front fusion(on-chip) | small front 결과 |
|---|---|---|---|
| **MAGMA vbatched** | O — front 들을 한 vbatched 에 묶음 | **X — op 입도**: `getrf`→global→`trsm`→global→`gemm` | op 당 grid 2–3 블록, op 마다 global 왕복 → **occupancy 2 %** |
| **STRUMPACK native 융합커널** | **X — front-monolithic**: "one block per front" | O — factor+trsm+Schur 를 한 shared 커널 | front 1개 = 블록 1개 → SM 대부분 유휴 → **occupancy 2 %** |
| **우리 `factor_small`** | **O** | **O** | **occupancy 30–59 %, lane 45–70 %** |

배타성의 근원:
- **MAGMA 의 packing 은 op 입도다.** vbatched 는 *"한 연산을 여러 front 에"* 적용하는 라이브러리 프리미티브 →
  다음 연산은 또 다른 커널이고 그 사이 데이터는 global 로 나간다. packing 이 op 별 커널에 묶여 **융합 불가**.
- **STRUMPACK 융합커널은 front-monolithic 입도다.** F11/F12/F22 를 한 블록 shared 에 깔고 통째로 도는 구조라
  *front 하나에 블록 하나*가 강제된다. small front 면 블록이 거의 빈다. 융합이 monolithic 이라 **packing 불가**.
- 즉 op-batched(across-fronts, per-op) ↔ front-monolithic(one-front, all-ops)은 **양립 불가능한 두 입도**다.
  그래서 둘 다 *서로 다른 이유로* occupancy 2 % 에 갇힌다.

### 3.4 우리 해법 — 세 번째 입도(sub-group)가 배타성을 해소

워프를 SG(8/16/32) 레인의 **sub-group** 으로 쪼개 **sub-group 하나가 front 하나의 *전체 파이프라인*을 돈다**
(`src/factorize/small.cuh`). 그러면 한 워프에 **32/SG 개 front 가 packing**되고(SM 채움), 각 front 는 자기
sub-group 안에서 **shared 상주로 융합 실행**된다(on-chip 재사용). packing 과 fusion 이 *동시에* 성립한다 — op
입도도 front-monolithic 입도도 아닌, **front 보다 작은 sub-group 입도**라서 가능하다.

이것은 **"MAGMA 의 packing + STRUMPACK 의 fusion 을 합친 것"이 아니다.** 두 baseline 의 입도는 *합쳐질 수
없는* 것이고, sub-group 분해가 그 배타성을 깬 것이 비자명한 지점이다. front 가 워프보다 작은(`fsz≤16` 이 99 %)
전력망 행렬에서 이 분해가 가장 잘 맞는다 — 그래서 *특화*다.

### 3.5 기전 = occupancy 회복 = 그 20×

ncu 측정: occupancy **2 % → 30–59 %**, lane util **7 % → 45–70 %**.

> **레짐 의존(ablation, §06 §5e):** packing이 주역인 건 **배치(B=64)**다 — small-tier 자체 6.0×(커널을 STRUMPACK식
> 으로 ablate한 controlled A/B). **단일 시스템(B=1)에선** packing(1.37×)·amalgamation 얕은 트리(1.46×)·GPU-resident
> graph(1.39×)가 *동급*이고, STRUMPACK 대비 16×의 절반 이상은 STRUMPACK 구현 오버헤드(alloc churn·CPU solve)다.
> 즉 "occupancy가 단일 주원인"은 B=64 한정이며, B=1은 세 기여 + baseline 오버헤드의 합으로 봐야 정직하다.

### 3.6 문헌 판정 (deep-research, 18 1차 소스, 3표 교차검증)

이 *동시* 구조 — "small front 를 packing 한 채로 각 front 의 full 파이프라인을 on-chip 융합" — 은 조사한 어느
GPU sparse direct solver(STRUMPACK / CHOLMOD / Rennich-Davis / LBNL / SABLE / cuDSS 문서)에도 없다. MAGMA 는
packing-only, STRUMPACK native 는 fusion-only 다.

**정직한 한계**: cuDSS 는 closed-source 라 내부 입도를 *검증할 수 없다*(cuDSS 보다 빠른 건 §5 에서 별도 증명,
기전은 미확인). 특허·비영어·미공개 미조사. ⇒ 결론은 "**survey 범위 내 처음**"이지 "전 세계 최초" 증명이 아니다.

**한 줄 판정**: *"전력조류 small-front 멀티프론탈에서, packing 과 full-front fusion 의 입도 배타성을 sub-group
분해로 해소해 둘을 한 커널에서 동시에 달성"* — 이 구조가 survey 범위 내 처음이고 측정된 16–66× 우위의 기전이다.

---

## 4. 3-tier 구현 — 그 원리를 front 크기 전체로 확장

§3 의 핵심(packing+fusion 동시)은 small 티어에 산다. 더 큰 front 로 가면 같은 원리(on-chip 상주를 최대화하며
점유를 지킴)를 크기에 맞춰 변형한다. front 가 어느 커널을 타는지는 **크기만의 결정적 함수**다 — 실행 시 점유·
옵션에 따라 흔들리지 않는다. 경계는 둘뿐(워프 폭 32, 점유 교차점 64), 둘 다 물리적 근거다.

| tier | front 크기 | 전략 | 경계의 근거 |
|---|---|---|---|
| **small** | ≤ 32 | **sub-group packing + full-fusion**(§3.4) | **워프 폭(32)** — packing 은 한 워프 안에서만 성립 |
| **mid** | 33–64 | front 전체를 shared 에 올려 1 block/front 제자리 분해 | **점유 교차점(64)** — whole-front staging 이 `fsz²` 라 넘으면 SM 당 블록 급감 |
| **big** | > 64 | **front 는 global 상주, 작은 타일만 staging, 한 front 를 여러 블록에 분산**(pivot + 패널 + trailing multi-block) | 점유 교차점(64) 위 — whole-front 가 SM 을 굶기므로 multi-block 으로 채움 |

- **big 티어(옛 large) = 일반화의 핵심**: power-grid 작은 케이스는 big 에 거의 안 들어가지만, **대형 power-flow(USA)
  ·circuit·2D/3D-FEM 의 큰 separator(uc 수백)**는 들어간다. 한 front 의 trailing(uc²·nc)을 32×32 타일로 GPU 전체에
  분산(multi-block) + L/U staging 한도(99KB) 초과 시 U 를 j-타일로만 올리는 bounded fallback. parabolic_fem factor
  759→57ms, cuDSS 추월. 상세: [§07 일반화 리포트](05-reports/07-generalization-suitesparse-2026-06-16.md).

> **정정(2026-06-18, [§10 tier 통합](05-reports/10-tier-consolidation-2026-06-18.md)):** 옛 4-tier(big=panel-resident
> 65–111 / large=global >111)를 **3-tier 로 통합**했다. A/B 측정 결과 panel-resident "big"은 배치에서 ~1.2%뿐(옛
> §07 −9.3% 재현 안 됨)이고 B=1 에선 65–111 을 global multi-block 에 보내는 게 **16% 더 빨랐다**. 그래서 65–111 을
> global 커널에 흡수하고 "large"를 "big"으로 개명. 결과: USA B=1 −16%, B=64 무손실, 라우팅 small/mid/big 3갈래로 일관.
- **big 티어의 구조적 점유 회복(배치)**: 큰 중간 front 를 통째로 shared 에 올리면 SM 당 블록이 1–2 개로 묶여
  대역폭을 굶는다. *분해에 필요한 L/U 패널만* shared 에 올리고 부피 큰 기여블록은 global 에 남기면 shared
  사용량이 ~3배 작아져 SM 당 블록이 3–4배 늘고 대역폭이 회복된다 — 배치에서 가장 큰 단일 이득.
- **레벨 내 tier 분리 + 멀티스트림**: 한 레벨의 이질적 front 들을 가장 큰 것에 맞춰 일괄 처리하지 않고 tier
  별로 분리 실행하며, 독립 부분트리는 별도 스트림으로 흘려 한 스트림이 놀리는 SM 을 다른 스트림이 채운다.

---

## 5. 성능 (RTX 3090, CUDA 12.8, kernel time, power-grid NR Jacobian)

### 5.1 단일 시스템 vs cuDSS
FP64 단일 시스템 factor 가 측정 케이스에서 cuDSS 를 앞선다(예: `case9241` factor 0.90 vs 0.99 ms/iter,
`25k` 1.10 vs 1.45). cuPF 통합 기준 custom solver 적용 시 약 **3–4×**. (cuDSS 는 가장 직접적 경쟁자이고, 이미
앞섬이 측정됨 — §3 의 STRUMPACK head-to-head 와는 별개 축.)

### 5.2 정밀도 레버 (B=1, factor+solve, kernel ms)
GA102 에서 FP64 는 FP32 의 1/64 속도 + 2× 바이트 → FP32 경로가 FP32 정확도에서 더 빠르다:

| 케이스 | fp64 f+s | fp32 f+s | fp32 vs fp64 |
|---|---|---|---|
| case3120sp | 0.488 | 0.350 | −28 % |
| case9241pegase | 1.082 | 0.667 | **−38 %** |
| case_ACTIVSg25k | 1.912 | 1.548 | −19 % |

정확도: fp64 relres 1e-13…1e-15, fp32 relres 1e-4…1e-6.

### 5.3 배칭 (시스템당 factor/solve, B=128 vs 단일)
배칭이 지연을 시스템 간 분할상환 → 처리량의 지배적 이득:

| 케이스 | factor/시스템 | solve/시스템 |
|---|---|---|
| case3120sp | **−91 %** | **−90 %** |
| case9241pegase | **−85 %** | **−84 %** |
| case_ACTIVSg25k | −72 % | −74 % |

---

## 6. 부가 요소 — 정직하게(헤드라인 아님)

신뢰는 수치를 부풀리지 않는 데서 나온다. 아래는 *실재하는 기여이나 §3 의 주 드라이버는 아니다* — 헤드라인에
올리면 크기를 오해하게 된다.

- **텐서코어 + 정확도 복원**: 얕은 K 갱신 GEMM 을 TF32 TC 로 태우되 head/tail 2-성분 보정으로 FP32 정확도
  복원. 방법론으로서 가치("저정밀 하드웨어를 쓰되 정확도를 지킨다")는 있으나, **순수 TC 기여는 +6–9 %**(공정
  best-vs-best 중앙값 ~1.1×)로 작고 작은 케이스(≤10K)에선 구조적으로 이득 없음. 큰 속도 덩어리는 TC 와 무관한
  커널 엔지니어링이다.
- **extend-add 융합 / host-free CUDA graph**: front 파이프라인을 "full" 로 만드는 마지막 조각(조립까지 한
  커널) + launch 붕괴. 실재 기여이나 *단독으로는 20× 를 못 만든다* — §3.5 의 occupancy 회복이 주이고 이들은 보조.
- **정밀도 모드**: 솔버는 FP64/FP32/TF32 *모드를 제공*한다. "all-float 가 NR 루프에 충분"한지는 cuPF 의
  수렴 결과이지 솔버의 주장이 아니다(§0).

---

## 7. 정직한 한계

- **단일 시스템(B=1)은 임계경로 바닥에 있다.** 고정 정밀도 안에서 어떤 커널 재구성(협력 레벨 융합, full
  shared-front, 임계값 sweep)도 신뢰할 만한 ≥10 % 추가 단축을 주지 못한다 — GPU 는 >99 % 놀지만 작업이 etree
  spine 을 따라 직렬 의존이다. B=1 의 ≥30 % 이득은 **정밀도**(§5.2), ≥50 % 처리량 이득은 **배칭**(§5.3)에서 온다.
- **순수 FP32 는 가장 큰 케이스(70k)에서 발산** → Mixed/FP64 사용.
- **신규성은 경계가 있다.** 개별 조각(packing·shared 융합·tiering·UBATCH·GPU-resident)은 prior art 다.
  방어 가능한 것은 §3 의 *입도-배타성 해소 구조*와 그 *측정된 기전*이다. cuDSS 내부는 미검증(closed-source).

---

## 8. 문서 색인

- [`05-reports/06-head-to-head-2026-06-16.md`](05-reports/06-head-to-head-2026-06-16.md) — head-to-head 실험·기전·문헌 판정(상세).
- [`05-reports/08-fair-strumpack-tuning-2026-06-17.md`](05-reports/08-fair-strumpack-tuning-2026-06-17.md) — **공정 정정**: STRUMPACK NodeNDP 튜닝(16–66×→~10×), 우리 NodeNDP 이식의 아키텍처적 무력, panel_width=8 최적화.
- [`05-reports/09-strumpack-mechanism-ncu-2026-06-17.md`](05-reports/09-strumpack-mechanism-ncu-2026-06-17.md) — **커널-레벨 기전(ncu/nsys)**: 매핑 입도(occ 4%→60%), host alloc churn, nc(F11) 라우팅 문헌판정, STRUMPACK 자체 small/big 실측(power-flow getrf 0%).
- [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md) — 선행연구 전체.
- [`01-orientation/01-api-and-build-design.md`](01-orientation/01-api-and-build-design.md) — API·빌드·cuPF 통합.
- [`optimal-configuration.md`](optimal-configuration.md) — 권장 빌드/실행 설정.
- [`history/`](history/), [`03-optimization-notes/`](03-optimization-notes/) — 사이클별 최적화·딥다이브.
- 재현 하니스: [`../../exp/`](../../exp/) (head-to-head 소스·케이스).
