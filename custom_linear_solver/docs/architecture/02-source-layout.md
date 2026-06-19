# 02 — 소스 코드 구조

> **층위**: 구조. `src/`가 어떻게 나뉘고 각 파일이 무엇을 책임지는지, 모듈 사이로 데이터가 어떻게 흐르는지.

---

## 1. 큰 그림 — 3 스테이지 + 공용 코어

코드는 cuDSS-유사 phase 모델(analyze → factorize → solve)을 그대로 디렉터리로 가진다.

```
src/
├── solver.{hpp,cpp}     ── 공개 API (Solver, SolverConfig, Status). 세 스테이지를 묶는 얇은 façade.
│
├── analyze/             ── 스테이지 1: 기호 분석 (값 무관, 한 번 실행)
│   ├── pipeline.cpp          오케스트레이터: 아래 단계를 순서대로 호출
│   ├── pattern/              device CSR↔CSC, A+Aᵀ 대칭 그래프 빌드
│   ├── reorder/              METIS nested dissection (fill-reducing 재정렬)
│   ├── symbolic/             elimination tree, fill 패턴, supernode 묶기, multifrontal 기호화
│   └── plan/                 front arena 레이아웃 + level/tier/subtree 분할 → MultifrontalPlan
│
├── factorize/           ── 스테이지 2: 수치 분해 (값 사용, 매 반복)
│   ├── factorize.{cu,hpp}    진입점: 값 scatter + 스케줄 실행(graph replay 또는 직접 발행)
│   ├── schedule.cuh          etree 스케줄: B==1 vs B>1 분기, tier 라우팅, 점유 게이트, 멀티스트림
│   ├── front_ops.cuh         per-front device 프리미티브(LU/U-solve/trailing/extend-add + TF32 mma)
│   ├── assemble.cuh          입력 CSR 값을 front arena로 scatter
│   ├── small.cuh             tier 커널: warp-packed sub-group (FactorSmall)
│   ├── mid.cuh               tier 커널: whole-front shared (FactorMid)
│   ├── big.cuh               tier 커널: global multi-block 트리플 (FactorBigPivot/Panels/Trail[Tf32])
│   └── single.cuh            B=1 단일 시스템 융합 커널 + selinv(pivot 역행렬화)
│
├── solve/               ── 스테이지 3: 삼각 대입 (값 사용, 매 반복)
│   ├── solve.{cu,hpp}        진입점: gather → 레벨별 대입 → scatter, graph 캡처/캐시
│   ├── dispatch.cuh          per-level 전진/후진 디스패치, tier 라우팅, 멀티스트림/단일스트림 sweep
│   ├── kernels.cuh           solve __global__ 진입점(small / fixed-NC / spine 변종)
│   ├── phases.cuh            solve device 프리미티브(전진/후진 substitution, CB 갱신)
│   ├── permute.cuh           gather/scatter 치환 커널(RHS를 perm 순서로, 해를 원복)
│   └── single.cuh            B=1 solve: 역행렬화된 pivot으로 병렬 GEMV
│
└── internal/            ── 공용 코어 (세 스테이지가 공유)
    ├── types.hpp            canonical 상수 + ClassifyFrontTier (tier 분류기) — 단일 진실원
    ├── matrix_view.hpp      CsrMatrixView / DenseVectorView (비소유 device 뷰)
    ├── plan/
    │   ├── multifrontal_plan.{hpp,cu}  MultifrontalPlan(device arena + host mirror, move-only)
    │   └── front_range_caps.hpp        ScanFrontRange: 패널 범위의 max fsz/uc/nc(디스패치 sizing)
    └── runtime/
        ├── state.{hpp,cu}   State(per-Setup 런타임: front/벡터 버퍼, 스트림, 이벤트, graph exec)
        └── setup.cu         Setup(): State 할당, 서브트리 스트림 생성, factor graph 캡처
```

빌드: `CMakeLists.txt`가 `analyze/factorize/solve/internal`의 소스를 `custom_linear_solver_ops` 정적 라이브러리로
묶고, `tests/run_custom_solver.cu`가 그걸 쓰는 CLI다. [03 §3](03-api-config-build.md).

## 2. 파일별 책임 (한 줄)

### `analyze/` — 기호 분석
| 파일 | 핵심 심볼 | 책임 |
|---|---|---|
| `pipeline.cpp` | `BuildPlanFromCsr`, `BuildPlanSeed` | 8단계 파이프라인 오케스트레이션(아래 §3), 선택적 구조 matching(Hopcroft–Karp) |
| `pattern/pattern_kernels.cu` | `BuildCscFromCsrDevice`, `BuildSymmetricGraphDevice`, `PermuteCscDevice` | device에서 CSR→CSC, `A+Aᵀ` 대칭 그래프, 치환 적용 |
| `reorder/metis_nd.cpp` | `MetisNdFromGraph` | METIS nested dissection — serial(`METIS_NodeND`) 또는 parallel ND(분리자 후 양분 재귀) |
| `symbolic/elimination_tree.cpp` | `Etree`, `FillPattern`, `Postorder`, `ColumnCounts` | elimination tree(Liu), 채움 패턴(Davis) |
| `symbolic/supernode.cpp` | `Supernodes`, `RelaxedPanels` | supernode amalgamation — 인접 열을 panel로 묶음(`max_panel_width`) |
| `symbolic/multifrontal.cpp` | `BuildMultifrontalSymbolic` | front 행 집합, panel 부모, extend-add 조립 맵(asm) 산출 |
| `plan/lower.cu` | `AnalyzeMultifrontal` | front arena 오프셋 레이아웃, 패널 레벨, tier 버킷, subtree 분할, `a_pos` scatter 맵, device 업로드 |

### `factorize/` — 수치 분해
| 파일 | 핵심 심볼 | 책임 |
|---|---|---|
| `factorize.cu` | `FactorizeImpl`, `Factorize` | 값을 front로 scatter(`AssembleFrontValues`) 후 graph replay(내부 모드) 또는 레벨 직접 발행 |
| `schedule.cuh` | `IssueFactorLevels`, `IssueFactorLevelRange` | `UseSingleSystem`(B==1) 분기, tier별 디스패치, 점유 게이트(`FactorSaturates`), 멀티스트림 fork/join |
| `front_ops.cuh` | `LuMidFront`/`LuPanelFactor`, `UPanelSolve[Fewsync]`, `TrailingUpdate*`, `ExtendAdd`, `Tf32OzakiPair`, `CLS_MMA_TF32_*` | per-front 4단계 device 빌딩블록 + TF32 mma 매크로 |
| `assemble.cuh` | `AssembleFrontValues` | 배치별 CSR 값을 `a_pos` 맵 따라 front arena로 scatter |
| `small.cuh` | `FactorSmall`, `LuSmallWarp`, `DispatchFactorSmall` | warp당 1 front(sub-group 8/16/32 lane) packing, 8 warp/block |
| `mid.cuh` | `FactorMid`, `StageInAsync`, `FactorizeFrontBlockedTf32`, `DispatchFactorMid` | front 전체 shared staging, 1 block/front; TF32 blocked TC trailing |
| `big.cuh` | `FactorBigPivot/Panels/Trail`, `FactorBigTrailTf32`, `DispatchFactorBig` | global 상주 multi-block 3-launch 트리플(전 정밀도); TF32 trailing은 per-tile 텐서코어(`FactorBigTrailTf32`) |
| `single.cuh` | `FactorSingleLevel`, `FactorSingleBig*`, `FactorSingleInvertPivot`, `IssueFactorSingle*` | B=1 융합 커널 + partitioned-inverse(selinv) |

### `solve/` — 삼각 대입
| 파일 | 핵심 심볼 | 책임 |
|---|---|---|
| `solve.cu` | `Solve`, `IssueSolveLevels` | gather RHS → 레벨별 대입 → scatter 해; 전체 solve graph lazy 캡처/캐시 |
| `dispatch.cuh` | `FwdLevel`, `BwdLevel`, sweep들 | per-level 전진/후진 디스패치, tier 라우팅, fixed-NC 특수화, 멀티스트림/단일스트림 |
| `kernels.cuh` | `SolveFwd[Small/Fixed]`, `SolveBwd*` | solve __global__ 진입점 |
| `phases.cuh` | `FwdSubstitute`, `FwdCbUpdate`, `Bwd*` | warp-parallel 삼각 substitution + CB 갱신 |
| `permute.cuh` | `GatherRhs`, `ScatterSolInverse` | RHS를 perm 순서로 모으고 해를 원래 순서로 흩뿌림 |
| `single.cuh` | `SolveSingleFwd/Bwd` | B=1: 역행렬화된 pivot으로 GEMV 전진/후진 |

### `internal/` — 공용 코어
| 파일 | 핵심 심볼 | 책임 |
|---|---|---|
| `types.hpp` | `ClassifyFrontTier`, `kSmallFrontMax=32`, `kMidFrontMax=64`, `kTensorCoreUcCap`, shared 예산 상수 | tier 분류·HW 상수의 **단일 진실원**(analyze/factorize/solve가 전부 합의) |
| `matrix_view.hpp` | `CsrMatrixView`, `DenseVectorView` | 사용자 device 버퍼의 비소유 뷰 |
| `plan/multifrontal_plan.{hpp,cu}` | `MultifrontalPlan` | symbolic+numeric plan(device arena + host mirror). move-only, 소멸자에서 device free |
| `plan/front_range_caps.hpp` | `ScanFrontRange`, `FrontRangeCaps` | 패널 구간의 max fsz/uc/nc 스캔(커널 thread/shared sizing) |
| `runtime/state.{hpp,cu}` | `State`, `UseSingleSystem`, `Precision` | per-Setup 런타임 상태 + 단일 시스템 여부 분기점 |
| `runtime/setup.cu` | `Setup`, `AllocateState`, `CreateSubtreeStreams` | State 할당/재할당, 서브트리 스트림, factor graph 캡처 |

## 3. 데이터 흐름 (analyze가 만들고, factorize/solve가 소비)

```
CSR(A) ─┐
        ├─▶ pattern_kernels  ─▶ CSC, A+Aᵀ 그래프
        │   metis_nd          ─▶ perm/iperm (재정렬)
        │   symbolic/*        ─▶ etree, fill 패턴, supernode panels, front 행/부모/asm 맵
        └─▶ plan/lower         ─▶ ┌────────────── MultifrontalPlan ──────────────┐
                                  │ host mirror: h_front_ptr, h_ncols, h_plcols,  │
                                  │   panel_level_ptr, tier offsets, subtree 정보 │
                                  │ device arena: front_off/front_ptr/ncols,      │
                                  │   a_pos(scatter 맵), asm_ptr/asm_local(조립)  │
                                  └───────────────────────────────────────────────┘
                                                  │ (값 무관, 한 번)
              ┌───────────────────────────────────┼───────────────────────────────────┐
              ▼                                    ▼                                   ▼
      Setup → State(수치 front 버퍼)        Factorize: AssembleFrontValues       Solve: GatherRhs
                                            → schedule(tier 커널) → L,U          → dispatch(대입) → x
```

핵심: **plan은 "어디에 무엇을 놓고 어떤 순서로 launch할지"의 청사진**이고(값 0개), **State는 "이번 B개의 숫자가
사는 곳"**이다. 두 구조의 상세 필드는 [04](04-memory-layout.md).

## 4. 네이밍·스타일 관례

- **C++/CUDA 심볼**: Google C++ 스타일 `PascalCase` 함수(`FactorMid`, `ClassifyFrontTier`), `kCamelCase` 상수
  (`kMidFrontMax`), `snake_case` 지역/멤버.
- **device hot-loop 약어**: `fsz`(front size), `nc`/`uc`(pivot/contribution dim), `t`/`nt`(thread idx/count),
  `F`/`Fs`(global/shared front), `L`/`U`(분해 인자). host 디스패치·plan 코드에선 풀어쓴다.
- **상수는 한 곳**: 세 스테이지가 합의해야 하는 값(tier 경계, shared 예산, 정렬)은 전부 `internal/types.hpp`. 파일마다
  재선언하지 않는다("must match" 결합 제거).
- **tier 디스패처**는 `Dispatch<Tier>` host 함수(small/mid/big), 커널은 `Factor<Tier>` `__global__`.
- **컴파일타임 토글**은 빌드 모드 1개(`CLS_INTERNAL_GRAPH`)와 내부 PTX 헬퍼(`CLS_MMA_TF32_*`)뿐. 그 외 튜너블은
  `SolverConfig` 필드이거나 `types.hpp` constexpr다(매크로 아님). 근거·이력은 [03 §4](03-api-config-build.md).
