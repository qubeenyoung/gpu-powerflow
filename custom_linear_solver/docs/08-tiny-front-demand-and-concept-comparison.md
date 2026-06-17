# tiny-front 전용 솔버의 수요 + STRUMPACK vs custom 개념 비교

> **상태**: reference (2026-06-16 세션 종합)   **목적**: (1) tiny-front 특화 솔버에 시장/연구 수요가 있는가,
> (2) STRUMPACK 개념과 custom 개념을 *추상·자세히·예시* 세 축으로 비교.
> 근거 데이터: [§07 일반화 리포트](05-reports/07-generalization-suitesparse-2026-06-16.md),
> [§06 head-to-head](05-reports/06-head-to-head-2026-06-16.md), [main-report](main-report.md).

---

# PART 1 — tiny-front 전용 솔버의 수요가 있는가?

## 1.1 먼저, "tiny-front 행렬류"의 정의 (측정 기반)
거의 모든 희소행렬은 elimination tree의 *leaf*가 tiny라 **개수**로는 tiny front가 많다. 구분선은 개수가 아니라
**작업(트레일링 FLOP)이 작은 front에 머무느냐**다(§07 측정):

| 행렬 | 작업 in fsz≤64 | 작업 in fsz>160 | 최대 front | 판정 |
|---|---|---|---|---|
| power-flow 25k | **71%** | 0% | ~96 | tiny-front (작업까지) |
| 회로(scircuit) | 55% | 1.7% | ~1개 | tiny-front |
| 2D-FEM(parabolic) | 0.6% | **97%** | 수백 | big-front |
| 3D-FEM(cant) | 0.2% | **99%** | 수백 | big-front |

→ **전력조류·회로는 *작업까지* tiny-front에 갇힌 드문 클래스**다. 전력망이 near-planar + 저차수라 separator가
작게 유지되기 때문(n이 커져도 최대 front가 96→160으로만 자람). 2D/3D PDE는 separator가 √n / n^(2/3)로 커져
작업이 큰 front로 이동한다.

## 1.2 이 클래스에 *수요*가 있는가 — 응용
핵심은 **tiny-front × 같은 패턴 다수 시스템(batched/반복)**의 교집합. 이게 본질적으로 나오는 워크로드:

**(A) 전력계통 — 1차 수요처 (실재·자금 있는 도메인)**
- **상정사고 분석(N-1, N-2)**: 선로/발전기 하나씩 빼며 수백~수천 케이스, 거의 같은 sparsity. ISO·계통운영자가 *상시* 수행.
- **확률·몬테카를로 조류**(재생E·부하 시나리오), **시계열/준정적(QSTS)**, **보안제약 OPF(SCOPF)**, **상태추정**, 실시간 EMS.
- 전부 *작은 시스템 하나로는 GPU가 놀지만 수백~수천 묶으면 꽉 차는* 구조 — 정확히 이 솔버의 강점.

**(B) 회로 시뮬레이션(SPICE/EDA) — 더 큰 산업, 단 진입장벽**
- **과도해석**: 매 시간스텝 Newton, 토폴로지 고정 → same-pattern 반복 factorization. **MC/공정코너**: 배치.
- 단점: 회로는 *대각우세가 아닐 때 pivoting 필요*(우리 no-pivot 가정 깨짐), KLU 등 CPU 솔버가 깊이 박혀 있음.

**(C) 인접 — near-planar 그래프 라플라시안**(스펙트럴/GSP, SPD 약대각우세), **2D PDE UQ/시간적분**(M-matrix).

## 1.3 정직한 수요 평가
- **공급 공백은 실재한다**: GPU 솔버 중 *batched-tiny-front 특화*는 없다. cuDSS는 범용(UBATCH는 있으나 tiny-front
  특화 아님), STRUMPACK은 범용 멀티프론탈(tiny-front에서 §07처럼 15× 느림). 그 틈이 곧 수요.
- **단, niche다**: 범용 HPC 시장이 아니다. 가장 *구체적이고 방어 가능한* 수요는 **전력계통의 배치 워크로드**
  (contingency/UQ/QSTS) — 계산량이 크고 상시 돌고 정확히 이 행렬류다. 회로·그래프는 인접 확장(추가 작업 필요).
- **가치의 크기**: cuPF 통합서 cuDSS 대비 ~3–4×, B=1에서도 대등/우위, STRUMPACK 압도. 수천 케이스를 도는
  contingency/UQ에선 3–4×가 실제 wall-clock·비용 절감.
- **결론**: "범용 솔버 수요"는 없다(cuDSS가 채움). "**batched tiny-front 특화 가속**"의 수요는 **전력계통에서
  분명히** 있고, 회로/그래프로 인접 확장 가능. 시장은 niche지만 깊다.

---

# PART 2 — STRUMPACK 개념 vs custom 개념 (3축 비교)

세 축으로 본다: **(1) 추상, (2) 자세히, (3) 예시.**

## 축 1 — 추상 (한 문장 개념)
| | STRUMPACK | custom |
|---|---|---|
| 한 문장 | "**어떤** 희소행렬이든 nested dissection으로 분해해, 레벨별로 *벤더 배치 BLAS(vbatched)*로 처리하는 **범용** GPU 멀티프론탈." | "front보다 작은 **sub-group을 병렬 원자**로 삼아 *packing과 full-pipeline fusion을 동시에* 하는 **tiny-front 특화** GPU 멀티프론탈." |
| 우선가치 | **일반성**(모든 행렬·크기·정밀도, pivoting으로 안정) | **특화**(작은 front·대각우세·배치에서 최대 속도) |
| 병렬 원자 | **연산(operation)** — 한 BLAS op를 여러 front에 | **sub-group** — front 하나의 *전체 파이프라인*을 레인 묶음에 |
| 비유 | "공장 컨베이어: 한 공정(절삭)을 모든 부품에 → 다음 공정으로" | "셀 생산: 작은 셀이 한 부품을 처음부터 끝까지" |

## 축 2 — 자세히 (메커니즘)
**STRUMPACK** (Ghysels et al., IJHPCA 2025):
- **host-driven 레벨 루프**: etree를 위→아래 레벨별 순회, 레벨마다 `getrf_vbatched`→`trsm_vbatched`×2→`gemm_vbatched`를 **각각 별도 커널 launch**.
- **extend-add(조립)는 분리**: 별도 batched 커널 또는 **CPU**, 일부 경로 host↔device 복사.
- **partial pivoting**(안정성), **solve는 CPU fallback**(논문 명시 한계).
- front별 처리: 작은 front는 native 융합커널(panel+trsm+Schur, *one block per front*), 큰 front는 cuSOLVER/cuBLAS.
- 결과: 일반성·안정성↑, 그러나 tiny-front에서 **op별 분리 + host 오케스트레이션 + one-block-per-front**로 GPU underfill.

**custom**:
- **전 과정 GPU-resident**: factor·solve를 CUDA graph로 캡처해 replay, host↔device는 수렴 스칼라뿐.
- **4-tier 결정적 라우팅**(front 크기 함수): tiny(sub-group packing+full fusion) / small·big(whole-front 또는 panel-resident shared) / large(blocked-LU: pivot + L21/U12 multi-block + trailing multi-block — 이번 세션).
- **factor + extend-add 융합**(atomicAdd로 부모에 scatter), 별도 조립 단계 없음.
- **no-pivot**(대각우세 가정), **B개 same-symbolic 배치**(한 symbolic 공유).
- 결과: tiny-front에서 packing(occupancy)과 fusion(메모리 재사용)을 *동시에* → occupancy 2%→30–59%. 단 일반성↓
  (큰 3D/초대형 8GB arena 벽, no-pivot은 대각우세 한정).

**핵심 개념 차이**: 둘 다 멀티프론탈이지만 — STRUMPACK은 **연산을 front들에 배치(op-batched, across-fronts)**,
custom은 **front를 sub-group으로 분해(sub-front granularity)**. 이 *입도(granularity)*가 갈림길이다. op-batched는
일반적이지만 op 사이 global 왕복·배리어·조립분리를 못 피한다. sub-group은 packing+fusion을 동시에 하지만 front가
워프보다 작아야(=tiny-front) 성립한다.

## 축 3 — 예시 (작은 문제로 두 알고리즘을 직접 따라가기)

작은 문제를 잡자. 6개 변수 그래프에서 자식 supernode 두 개 **A={1,2}, B={3,4}**가 **separator {5,6}**으로
나뉜다고 하자. nested dissection 트리는 *(leaf A, leaf B) → root {5,6}*. A와 B는 둘 다 5,6에 연결된다.

### (i) 한 front의 4단계 — 실제 숫자로
leaf A의 front(4×4, nc=2 자기변수 {1,2}, uc=2 separator 행 {5,6}):
```
      col1 col2 | col5 col6
row1 [  4    2  |  1    0 ]   F11 = [[4,2],[2,5]]   (완전소거 블록)
row2 [  2    5  |  0    1 ]   F12 = [[1,0],[0,1]]   (패널)
row5 [  1    0  |  3    0 ]   F21 = [[1,0],[0,1]]
row6 [  0    1  |  0    3 ]   F22 = [[3,0],[0,3]]   (→ 부모로 갈 기여블록)
```
- **1단계 F11 LU**: `L11=[[1,0],[0.5,1]]`, `U11=[[4,2],[0,4]]`.
- **2단계 패널**: `U12 = L11⁻¹F12 = [[1,0],[-0.5,1]]`, `L21 = F21·U11⁻¹ = [[0.25,-0.125],[0,0.25]]`.
- **3단계 Schur**: `F22 ← F22 − L21·U12 = [[3,0],[0,3]] − [[0.3125,-0.125],[-0.125,0.25]] = [[2.6875,0.125],[0.125,2.75]]`.
- **4단계 extend-add**: 이 2×2 결과를 **부모 front의 {5,6}×{5,6} 위치에 더한다**. leaf B도 똑같이 해서 같은
  위치에 더하면, 부모 front의 {5,6} 블록 = (원래값) + (A의 기여) + (B의 기여) → 그제서야 부모가 자기 4단계 수행.

### (ii) leaf 레벨(front A, B 두 개)을 GPU에 매핑 — 두 방식
이제 *같은 두 front A,B*를 처리하는 방식이 갈린다.

**STRUMPACK (operation-batched, 커널 5개):**
```
커널1  getrf_vbatched([F11_A, F11_B])     → 두 F11을 한 번에 LU. 결과를 global에 씀.
커널2  trsm_vbatched([F12_A, F12_B])      → 두 U12. (global 다시 읽고 씀)
커널3  trsm_vbatched([F21_A, F21_B])      → 두 L21.
커널4  gemm_vbatched(F22 -= L21·U12)      → 두 Schur 갱신.
커널5  extend_add([A,B] → parent)         → 별도 조립(또는 CPU).
```
→ 한 *연산*을 두 front에 묶는다(across-fronts). **단계 사이 front 데이터가 global을 왕복**하고, 두 front가
작으면(2×2~4×4) 각 커널이 GPU를 거의 못 채운다(occupancy↓). 깊은 트리면 레벨마다 이 5커널이 반복.

**custom (sub-group fused, 커널 1개):**
```
커널1개:  한 워프 = 2개 sub-group (SG=16 레인씩).
          sub-group0 ← front A,  sub-group1 ← front B  (packing, 한 워프가 2 front)
          각 sub-group이 자기 4×4 front를 shared에 올려:
              1단계 F11 LU → 2단계 패널 → 3단계 Schur   (전부 shared, global 왕복 없음)
              4단계: F22를 atomicAdd로 부모 front에 바로 scatter   (조립 융합)
```
→ 한 *front*의 전 파이프라인을 *작은 sub-group*에 담아, 여러 front를 한 워프에 packing. 데이터는 shared에 머물고
(fusion), 한 커널이 여러 front × (B 배치)를 채운다(packing). **STRUMPACK의 5커널·global왕복·조립분리가 전부 사라짐.**

### (iii) 핵심
같은 4단계·같은 트리인데 — STRUMPACK은 "연산을 front들에 배치"(커널 5개, global 왕복), custom은 "front를
sub-group으로 쪼개 한 커널에 융합"(커널 1개, shared 상주). front가 워프보다 **작을 때만** custom 매핑이
성립한다(=tiny-front). front가 크면(아래 §3.3-⑦) custom도 타일로 쪼개야 하고, 아주 크면 STRUMPACK의 큰-GEMM
vbatched가 유리해진다.

> (실측 대조는 §07 리포트: scircuit custom 4.4ms vs STRUMPACK 77.9ms / Transport는 반대로 STRUMPACK 우위.)

## 한 줄 정리
STRUMPACK = **"연산을 front들에 배치하는 범용 멀티프론탈"**(큰 3D에 강함, 안정·일반).
custom = **"front를 sub-group으로 쪼개 packing+fusion을 동시에 하는 tiny-front 특화"**(작은 front·배치에 강함).
*입도(operation vs sub-front)*가 두 개념의 본질적 분기점이고, 그 분기가 곧 각자의 강점 영역(큰 3D vs tiny-front)을 결정한다.

---

# PART 3 — 알고리즘 자세히 (그리고 custom이 어떻게 최적화했나)

## 3.0 공통 골격 — 멀티프론탈이 뭘 하는가
둘 다 같은 뼈대를 쓴다. 먼저 이걸 알아야 차이가 보인다.

**(a) ordering & 트리.** 행렬 A의 그래프를 **nested dissection(METIS)**으로 정렬한다. separator로 그래프를
재귀 이분 → **elimination tree(소거트리)**가 생긴다. 트리 노드 = **supernode**(함께 소거되는 열 묶음).

**(b) front(전두행렬).** 각 트리 노드는 작은 **밀집(dense) front** 행렬로 모인다. front를 블록으로 보면:
```
        nc            uc
      ┌──────────┬──────────┐
  nc  │   F11    │   F12    │   F11: fully-summed 피벗 블록 (이 노드에서 완전히 소거됨)
      ├──────────┼──────────┤   F12,F21: 패널 (아직 더 갱신될 행/열)
  uc  │   F21    │   F22    │   F22: contribution block(=Schur 보수). 부모로 넘김.
      └──────────┴──────────┘
```
nc = 피벗 열 수, uc = 갱신(트레일링) 차원.

**(c) 한 front의 4단계.**
1. **F11 분해**: `F11 = L11·U11` (LU).
2. **패널 solve**: `U12 = L11⁻¹·F12`, `L21 = F21·U11⁻¹` (삼각 solve).
3. **Schur 갱신(trailing)**: `F22 ← F22 − L21·U12` (이게 대부분의 FLOP, O(uc²·nc)).
4. **extend-add(조립)**: 갱신된 F22를 **부모 front의 해당 위치로 더한다**(scatter add). 부모는 자식들의 기여를
   다 받은 뒤 자기 차례에 1~4를 수행.

**(d) 처리 순서.** 트리를 **leaf→root**로. 한 레벨의 front들은 서로 독립이라 동시 처리 가능. solve는 factor의
L/U로 전진대입(leaf→root) + 후진대입(root→leaf).

→ **여기까지 STRUMPACK과 custom이 동일하다.** 차이는 *이 4단계와 트리 순회를 GPU에 어떻게 매핑하느냐*다.

## 3.1 STRUMPACK의 매핑 (operation-batched, host-driven)
```
for level L = leaves..root:                      ← HOST 루프 (CPU가 오케스트레이션)
    fronts = 이 레벨의 모든 front
    getrf_vbatched(fronts.F11)                    ← 커널1: 모든 F11을 한 번에 LU (벤더 배치 BLAS)
    trsm_vbatched(fronts.F12); trsm_vbatched(F21) ← 커널2,3: 패널 solve
    gemm_vbatched(F22 -= L21·U12)                 ← 커널4: Schur 갱신
    extend_add(fronts → parents)                  ← 커널5(별도) 또는 CPU: 조립
solve: 전진/후진대입 (대형은 CPU fallback)
```
- **병렬 원자 = "한 연산 × 여러 front"**(vbatched). getrf 따로, trsm 따로, gemm 따로 → **단계 사이 front 데이터가
  global 메모리를 왕복**하고 동기화 배리어가 낀다.
- **조립(extend-add)은 분해와 분리**(별도 커널/CPU).
- **partial pivoting**으로 안정(어떤 행렬도 됨). 작은 front는 native 융합커널(F11+trsm+Schur, *front당 블록 1개*).
- **장점**: 일반·안정, 큰 front에서 vbatched가 큰 GEMM을 효율적으로 → 큰 3D에서 빠름(논문 재현).
- **tiny-front에서 손해**: ① op별 분리 → global 왕복·배리어, ② 깊은 트리(전력망 ~78레벨) × 5커널 = 수백 launch,
  ③ 작은 front를 one-block-per-front로 돌리면 GPU underfill(occupancy 2%), ④ 조립 분리, ⑤ solve CPU fallback.

## 3.2 custom의 매핑 (sub-front granularity, GPU-resident, fused)
```
analyze (패턴당 1회):
    GPU 대칭그래프(A+Aᵀ) → METIS ND → relaxed-panel amalgamation(얕은 트리, 저fill)
    symbolic(front_ptr, extend-add 맵) → device arena + factor/solve CUDA graph 캡처
factorize (매 반복): scatter 값 → factor graph replay:
    for level L (graph node 1개):
        그 레벨 front들을 크기로 4-tier 라우팅, tier별 전용 커널 1회(× B 배치):
          tiny  : sub-group이 front 하나의 [4단계 전체]를 shared에서 융합 실행, 32/SG front/warp packing
          small : whole-front shared
          big   : L/U 패널만 shared(CB는 global) — 배치 점유 회복
          large : blocked-LU (pivot → L21/U12 multi-block → trailing multi-block)
        extend-add는 trailing 끝에 atomicAdd로 부모에 바로 scatter (별도 단계 없음)
solve: 전진/후진대입도 GPU graph, tier별(spine은 전용 커널)
```
- **병렬 원자 = sub-group(front 미만)** + 큰 front는 tile. 즉 "한 front × 전체 파이프라인"을 *작은 단위*로 쪼개
  GPU를 채운다.
- **전 과정 GPU-resident**(CUDA graph): host 오케스트레이션 0, launch는 graph 1회 재생.

## 3.3 custom이 *어떻게* 최적화했나 — 기법별 (무엇을·왜·효과)

**① sub-group packing + full-pipeline fusion (핵심, tiny tier)**
- *무엇*: 워프를 SG(8/16/32) 레인 sub-group으로 쪼개 **sub-group 하나가 front 하나의 4단계 전체**(F11 LU →
  패널 → Schur → extend-add)를 shared에서 수행. 한 워프에 32/SG개 front packing, 다중 warp/block.
- *왜*: STRUMPACK은 packing(vbatched)이나 fusion(native 커널) 중 **하나만** 한다(입도가 배타적: op-batched는
  per-op라 융합 불가, native는 front당 블록이라 packing 불가). 둘 다 occupancy 2%. front가 워프보다 작은
  tiny-front에선 **sub-group 입도**로 둘을 동시에 → SM을 채우면서(packing) 데이터를 on-chip 유지(fusion).
- *효과*: occupancy 2%→30–59%, lane 7%→45–70% → **B=1 동일정밀 STRUMPACK 대비 16–66×**(§06).

**② factor + extend-add 융합 (모든 tier)**
- *무엇*: trailing 직후 F22를 `atomicAdd`로 부모 front에 바로 scatter. 별도 조립 커널/CPU 없음.
- *왜*: STRUMPACK은 조립이 분리 단계(global 왕복/CPU). 부모는 자식보다 높은 레벨이라 race-free.
- *효과*: 조립 패스 제거 + CB의 global write+read 왕복 제거.

**③ GPU-resident CUDA graph (전 과정)**
- *무엇*: factor·solve 커널 시퀀스를 graph로 캡처해 replay. host↔device는 수렴 스칼라 1개뿐.
- *왜*: STRUMPACK은 host 레벨 루프(레벨마다 launch·동기화·일부 cudaMalloc). 깊은 트리에서 launch 폭증.
- *효과*: per-level host 오케스트레이션 제거.

**④ batched same-symbolic (배치)**
- *무엇*: B개 시스템이 **하나의 symbolic** 공유, `blockIdx.y=batch`로 한 launch가 front×B 처리.
- *왜*: contingency/UQ/시계열은 같은 패턴 수천 개 → 작은 시스템 하나론 GPU가 놀지만 묶으면 꽉 참.
- *효과*: per-system factor/solve −70~91% (배치 클수록).

**⑤ no-pivot (대각우세 활용)**
- *무엇*: 대각 원소를 그대로 피벗(열-최대 탐색·행 교환 없음). 0/위험 피벗만 static shift 가드.
- *왜*: 전력망 Jacobian은 대각우세 → pivot 불필요. partial pivot 비용(O(f²))은 작은 front에서 무시 못 함.
- *효과*: 격차의 주원인은 아니나(§5b) 작은 front에서 수~십% 절감 + 코드 단순(조립과 안 얽힘).

**⑥ relaxed-panel amalgamation (analyze)**
- *무엇*: etree chain을 panel_width(=16)까지 병합하되 **패딩 낭비가 true fill의 (1+relax)배 이내**일 때만(이번
  세션에서 가드 추가, separator 횡단 chain의 fill 폭증 방지).
- *왜*: 얕은 트리(레벨↓ = solve launch↓)를 *낮은 fill로* 얻는다. STRUMPACK은 얕게 하려면 fill 6–9× 폭증.
- *효과*: 전력망 ~26레벨을 metis의 ~2× fill로(STRUMPACK 78레벨). solve의 순차 의존 축소.

**⑦ blocked-LU large tier — 큰 front를 GPU 전체로 분산 (이번 세션)**
- *문제*: 큰 front(circuit/2D-FEM separator, uc 수백)는 처음에 ① shared 한도 초과로 launch 실패("OOM" 오진),
  ② one-block-per-front로 under-fill(occupancy 8~17%).
- *해결 단계*:
  - (a) trailing(F22 −= L21·U12, uc²)을 **32×32 타일로 multi-block** — front 하나의 트레일링을 GPU 전체에 분산.
  - (b) shared 초과 시 U만 타일 staging하는 bounded fallback(crash 제거).
  - (c) panel(L21/U12 solve)이 새 병목(47%)이 되자 **정석 blocked-LU 분해**: pivot(nc×nc, front당) →
    L21/U12 solve를 **uc로 multi-block** → trailing multi-block. pivot 외 전부 GPU를 채움.
  - (d) batched 큰-front solve가 shared 초과로 crash → **non-staged fallback**(F 직접 읽기) 라우팅.
- *효과(FP64)*: parabolic factor **759→52ms(14×)**, cant 59→50ms. cuDSS 대비 parabolic **1.5×**, cant 1.26×.
  → 큰 front까지 처리 가능해져 **"tiny-front 클래스"의 경계가 2D-FEM·일부 3D로 확장**(일반화의 핵심).

- *예시로 따라가기*: 큰 front 하나(nc=2, uc=6 → front 8×8)를 생각하자. tiny tier처럼 "front 하나=sub-group
  하나"로 하면 그 sub-group이 8×8 전체를 혼자 처리 → 큰 front엔 일거리 과다, GPU엔 블록 부족(under-fill).
  대신 large tier는 한 front의 일을 *작은 조각으로 쪼개 여러 블록*에 뿌린다:
  - **pivot**: nc×nc=2×2 블록만 한 블록이 LU (작음, front당 1회).
  - **L21/U12 패널**: L21은 uc=6개 *행*이 서로 독립(각 행은 U11에 대한 삼각 solve), U12는 uc=6개 *열*이 독립.
    → 6개 행/열을 **여러 스레드/블록에 나눠** 동시에 solve. (예: 한 블록이 row0..2, 다른 블록이 row3..5)
  - **trailing F22(6×6=36칸) ← F22 − L21·U12**: 36칸을 **32×32 타일**(여기선 1타일이지만 큰 front면 여러 타일)로
    쪼개 각 블록이 자기 타일만 계산. uc=600이면 19×19≈360타일 → 360블록이 GPU를 가득 채움.
  즉 **"front 하나의 4단계"를 (pivot=1블록) + (패널=행/열 분산) + (trailing=타일 분산)**으로 펼쳐, 큰 front
  하나라도 수백 블록으로 GPU를 채운다. ←→ 처음엔 이 8×8을 한 블록이 통째로 직렬 처리해서 GPU가 놀았다(병목).

**측정 요약(B=1, FP64)**
| 행렬 | custom factor/solve | cuDSS | STRUMPACK |
|---|---|---|---|
| scircuit | 4.4 / 2.2 | 5.0 / 1.5 | 77.9 / 87.3 |
| parabolic | 52 / 7.2 | 78.6 / 5.7 | 227 / 125 |

→ ①~⑤가 *작은 front*에서의 압도(STRUMPACK 15–40×, cuDSS 대등)를, ⑦이 *큰 front*로의 확장을 만들었다.

## 3.4 정직한 한계 (알고리즘 차원)
- **8GB front-arena 벽**: all-fronts-GPU-resident라 초대형(G3/Transport/bmwcra) bail-out. front 스트리밍이 필요한데
  그건 GPU-resident graph(③)와 충돌 — 근본적 긴장.
- **no-pivot(⑤)**: 비대각우세 행렬엔 partial pivoting 필요(미구현, 조립과 얽혀 대형 작업).
- **solve floor**: B=1 상위 레벨은 front 수가 적어 under-fill(임계경로 바닥), 하위는 메모리 바운드(거의 최적).
  남은 레버는 batch-interleaved arena(cross-batch coalescing) — 구조 변경.
