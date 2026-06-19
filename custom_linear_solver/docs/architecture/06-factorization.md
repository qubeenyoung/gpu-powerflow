# 06 — 수치 분해 (factorization)

> **층위**: 추상 + 상세. multifrontal LU의 per-front 4단계를 표준 선형대수로 먼저, 그 다음 3개 tier 커널이 같은
> 4단계를 GPU에 어떻게 다르게 매핑하는지. 프리미티브는 `src/factorize/front_ops.cuh`.

---

## 1. 추상 — per-front 4단계

각 front는 elimination tree 순서(leaf→root)로 네 표준 연산을 거친다. front를 pivot 블록과 contribution 블록으로
나눈 좌표계([04 §3](04-memory-layout.md))에서:

```
front F = [ A11  A12 ]   nc 행 (pivot)        목표: A11 = L_pp·U_pp 로 분해하고
          [ A21  A22 ]   uc 행 (contribution)       CB = A22 − L21·U12 를 부모로 넘김
            nc   uc 열
```

| # | 단계 | 연산 | 채우는 곳 | 프리미티브 |
|---|---|---|---|---|
| 1 | **Panel LU** | `A11 = L_pp · U_pp` (no-pivot, 대각 shift만) + `L21 = A21·U_pp⁻¹` | 좌상단 nc×nc + L-panel | `LuMidFront`/`LuPanelFactor`, `GuardedPivot` |
| 2 | **U-panel solve** | `U12 = L_pp⁻¹ · A12` (nc×uc 전진대입) | 우상단 nc×uc | `UPanelSolve`/`UPanelSolveFewsync` |
| 3 | **Trailing update (Schur)** | `CB = A22 − L21 · U12` (uc×uc rank-nc 갱신) | 우하단 uc×uc | `TrailingUpdate*` |
| 4 | **Extend-add** | 자식 CB를 부모 front의 해당 (행,열)에 **`atomicAdd`** | 부모 front | `ExtendAdd` |

분해된 `L`/`U`만 global로 write-back(`WritebackFactored`)하고, CB는 부모로 흘러간다. 이 4단계가 전부이며, tier별
커널은 이 4단계를 **다른 병렬 매핑**으로 구현할 뿐이다(§3).

### extend-add가 race-free인 이유
부모 front는 자식보다 **엄격히 상위 레벨**에 있어 자식들이 scatter할 때 아직 분해 전이다. 형제 자식들은 부모의
같은 원소에 동시에 더할 수 있으므로 `atomicAdd`가 필요하고, 부모가 아직 안 건드려졌으니 그 누적은 정확하다
(`asm_local` 맵이 자식 CB 원소 → 부모 로컬 (행,열)을 준다).

## 2. 입력 조립 — 값을 front로 (`assemble.cuh`)

분해 직전, 등록된 CSR 값 `A`를 front arena로 흩뿌린다(`AssembleFrontValues`). analyze가 만든 `a_pos[k]`가 비영 `k`의
front 내 목적지를 미리 계산해 뒀으므로([04 §2](04-memory-layout.md)), 커널은 `front[a_pos[k]] (+)= value[k]`만 한다.
배치면 `grid.y = b`로 B개를 동시에. 패턴이 중복 없으면 store, 아니면 atomicAdd(`a_pos_unique`).

## 3. 같은 4단계, 3가지 GPU 매핑

### small (`small.cuh` · `FactorSmall`) — sub-group packing
- 한 front를 **sub-group(8/16/32 lane)**에 매핑, 8 warp/block(`kSmallTierWarpsPerBlock`). block barrier 없이
  `__syncwarp`만.
- 1·3단계를 `LuSmallWarp`로 **융합**(right-looking LU: pivot 나눗셈 → rank-1 갱신을 lane-parallel로).
- 수만 개 leaf front를 워프에 packing해 점유를 채우는 게 목적. sub-group 크기는 `fsz`로 결정(`FactorSmallSg`:
  fsz≤8→8, ≤16→16, else 32).
- 텐서코어 없음(small front는 `K=nc`가 1~2라 TC 구조적 무이득).

### mid (`mid.cuh` · `FactorMid`) — whole-front shared-resident
- front 전체를 shared에 staging(`StageInAsync`, Ampere+ `cp.async`), **1 block/front**. phase 간 global 재독 없음.
- `fsz ≤ 48`(`kFusedMidFrontMax`)는 1+3 융합 경로(`LuMidFront`), 초과는 분리: `LuPanelFactor`(1) →
  `UPanelSolveFewsync`(2) → `TrailingUpdate`(3, register-blocked).
- TF32(`UseTC`)면 trailing이 텐서코어(`FactorizeFrontBlockedTf32`/`BlockUpdateTf32Tc`). [09](09-precision-and-tensor-cores.md).
- L|U write-back 후, 부모 있으면 `ExtendAdd`. block thread 수는 GPU fill로 512/256/128(`DispatchFactorMid`).

### big (`big.cuh`) — global-resident, multi-block triple
- big front는 shared-resident가 불가능할 만큼 크다(fsz>64). front를 **global에 두고**, 한 front의 작업을 **여러
  block에 분산**해 GPU를 채운다. 단일 block/front은 점유가 near-zero라 쓰지 않는다.
- **전 정밀도 공통**으로 **3-launch 멀티블록 트리플**(`DispatchFactorBig`):
  `FactorBigPivot`(pivot nc×nc, 1 block/front) → `FactorBigPanels`(L21/U12를 uc로 multi-block) →
  `FactorBigTrail`(uc×uc를 TM×TN 타일로 multi-block, extend-add 융합). launch 간 순서는 graph edge로 보장.
- **TF32**면 trailing만 텐서코어로 교체: `FactorBigTrailTf32`(per-tile Ozaki mma, 블록당 16×64 출력 타일,
  grid=(tile,front,batch) — 스칼라와 같은 멀티블록 fill)가 스칼라 `FactorBigTrail`을 대신한다. 게이트는 pivot 차원
  하나뿐(`nc ≤ kTensorCorePivotColumnCap`로 staged K + per-tile shared를 bound; **uc는 무제한**); 그 외(FP32/FP64,
  또는 nc 초과)는 스칼라 trailing. [09](09-precision-and-tensor-cores.md).
  - 정직한 천장: 전력망 big-front는 K=nc≈8(단일 mma K-타일)이라 텐서 pipe ~7%(shared-load bound)로, 여기서 TF32는
    FP32보다 빠르지 않다 — fronts가 너무 작아 텐서코어가 못 이긴다(small tier가 TC를 건너뛰는 것과 같은 이유,
    [`../README.md` §8](../README.md)).
- 전력망 Jacobian은 big tier가 드물고, circuit·2D/3D-FEM의 큰 separator에서 주로 쓰인다.

## 4. 스케줄 — 누가 언제 launch되나 (`schedule.cuh`)

`IssueFactorLevels`가 분해 전체를 발행한다. 분기점 하나, **`UseSingleSystem(st) == (B == 1)`**:

- **B==1**: `IssueFactorSingleSchedule` — 단일 시스템 융합 경로(`single.cuh`). [08 §4](08-runtime-and-batching.md).
- **B>1**: `IssueFactorBatched` — panel etree를 레벨별로 내려가며, 각 레벨을 **tier 동질 구간으로 쪼개**(analyze가
  미리 tier-정렬) 각 구간을 전용 커널(`Dispatch{Small,Mid,Big}`)로 발행. `grid.y = batch`로 B개를 한 launch에.
  독립 서브트리는 별도 스트림에 fork → spine에서 join(멀티스트림, [08 §3](08-runtime-and-batching.md)).

레벨 안의 front들은 etree상 독립(부모는 상위 레벨)이라 small→mid→big 서브-launch 순서는 무관하고 정확하다.

## 5. 점유 게이트 (`FactorSaturates`, `front_ops.cuh`)

small tier 라우팅에만 붙는 예외: 좁은 레벨이라 small 커널이 GPU를 못 채우면
(`(레벨 front 수)×B < FactorWarpFill()` = SM×워프슬롯), 그 레벨을 **whole-block 커널(`FactorMid`)로 보낸다**
(`IssueFactorLevelRange`). 원리: "packing이 GPU를 못 채우면 packing하지 않는다". fp32 front는 가벼워 work를 ×2로 셈.

## 6. 같은 연산의 코드 변종 (왜 여러 개인가)

본문은 4개 표준 연산으로 설명했지만, 성능 때문에 같은 연산의 변종이 여럿이다. 표준 추상화로 묶으면:

| 표준 연산 | 코드 변종 | 차이(왜 여럿인가) |
|---|---|---|
| Panel LU | `LuMidFront`(융합) / `LuPanelFactor`(row-fused vs 2-phase, nc≤12/16 분기) | barrier 수 vs thread당 직렬 작업 trade-off |
| U-panel solve | `UPanelSolve`(row-parallel) / `UPanelSolveFewsync`(col-parallel, barrier 1회) | 점유 낮은 레벨선 few-sync 우세 |
| Schur GEMM(trailing) | mid: `TrailingUpdateScalar` / `…Rb`(register-block) / `BlockUpdateTf32Tc` · big: `FactorBigTrail`(타일 multi-block) / `FactorBigTrailTf32`(per-tile 텐서코어) | tier·staging·정밀도·융합 여부의 곱 |
| extend-add | `ExtendAdd`(분리) / trailing에 융합(`FactorBigTrail`·`FactorBigTrailTf32`의 fused drain) | CB global 왕복 제거 |

이 변종 폭증은 표준 "Schur GEMM(staging·정밀도·출력 파라미터화)" 하나로 합칠 여지가 있다(현 코드는 의도적
손튜닝 상태). solve 쪽 변종은 [07 §5](07-solve.md).
