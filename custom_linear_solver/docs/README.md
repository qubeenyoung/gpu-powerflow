# `custom_linear_solver` — 문서

> **상태**: canonical (single source of truth)   **갱신**: 2026-06-18
> **한 줄**: small-front 전력망 Jacobian 용 **배치 GPU multifrontal 직접 솔버**. cuDSS-유사 phase API.
> 이 문서는 개요·진입점이다. 코드 기반 동작 상세(소스 구조·옵션·메모리 레이아웃·tier·알고리즘·런타임·B=1/B>1·정밀도)는
> [`architecture/`](architecture/)(추상+상세 양층, 10개 문서), 과거의 모든 설계·실험·벤치 로그는 [`_legacy/`](_legacy/) 에 있다.

> **재작성 노트(2026-06-18)**: 이전 문서 트리(`01-orientation/` … `05-reports/`, lab-meeting, main-report 등)는
> 누적되며 소스와 어긋난 부분이 많았다(대표적으로 옛 4-tier·`MID_THRESH=128`). 전부 [`_legacy/`](_legacy/) 로 내리고,
> 이 README 를 **현재 소스(`src/`)만을 근거로** 처음부터 다시 썼다. 모든 수치 주장은 `_legacy/` 의 해당 측정 문서를 가리킨다.

---

## 1. 개요

`custom_linear_solver` 는 고정 sparsity 의 비대칭 희소행렬 `A` 를 multifrontal LU 로 GPU 에서 분해/solve 하는
라이브러리다. 타깃은 **Newton–Raphson 전력조류의 Jacobian** — front(dense 부분행렬)가 극단적으로 작고
(대부분 `fsz ≤ 16`), 같은 sparsity 를 값만 바꿔 반복 풀며, 여러 시스템을 한 symbolic 으로 **배치**한다.

- **한 번 analyze, 반복 factorize/solve**: NR 루프에서 sparsity 가 고정 → `Analyze()` 1회, `Factorize()`/`Solve()` 만 반복.
- **B-시스템 배치**: 같은 패턴, 다른 값의 시스템 `B` 개를 한 plan 으로 동시 factor/solve.
- **전 과정 device-resident**: factor/solve 를 CUDA graph 로 캡처해 replay, host↔device 왕복은 수렴 스칼라뿐.
- **no-pivot**: 대각우세(power-grid NR Jacobian) 가정. 정적 대각 shift fallback 만 둔다.

상위 래퍼/구현 인벤토리는 모듈 루트 [`../README.md`](../README.md) 참조.

---

## 2. 공개 API (`src/solver.hpp`)

cuDSS phase 모델을 1:1 모방한다. 전형 시퀀스:

```cpp
custom_linear_solver::Solver solver(config);   // SolverConfig
solver.SetData(matrix);        // A: CSR sparsity + values 포인터 (device)
solver.SetRhs(rhs);            // b (device)
solver.SetSolution(solution);  // x (device, 출력)
solver.Analyze();              // 1회: METIS ND + symbolic + plan 빌드
solver.Setup(/*batch_size=*/B);// B 시스템용 런타임 상태/arena 할당
solver.Factorize();            // 현재 값으로 수치 분해
solver.Solve();                // x = A^{-1} b
```

- `B > 1` 이면 등록 버퍼는 batch-strided: `values[b*nnz+·]`, `rhs[b*n+·]`, `solution[b*n+·]`.
- `SetValues(ptr, value_type)` 로 값만 갱신해 재-factorize (NR 루프).
- 외부 그래프 캡처 모드(`CLS_INTERNAL_GRAPH=OFF`): `SetStream(stream)` 로 caller 스트림에 커널을 직접 발행 →
  cuPF 같은 외부 캡처가 한 iteration 전체를 graph 로 기록.
- `Status` enum: `kSuccess` / `kInvalidValue` / `kInvalidState` / `kAllocationFailed` /
  `kAnalysisFailed` / `kFactorizationFailed` / `kSolveFailed`.

---

## 3. 설정 — `SolverConfig` 기본값

`src/solver.hpp` 의 `SolverConfig` (생성자 인자). **모든 값은 기본값**이며, 크기 기준 tier 경계·TF32 PTX trailing
스택·per-front 라우팅 같은 튜너블은 측정상 off-default 가 전부 회귀해서 **빌드에 baked-in** 돼 있다.

| 필드 | 기본 | 설명 |
|---|---|---|
| `precision` | `FP64` | `FP64` / `FP32` / `TF32` (TF32 = Ozaki mma, **권장**) |
| `max_panel_width` | `8` | supernode panel 최대 열 수. 분석기는 이 값을 `[1,64]`로 clamp 해 전 사이즈에 균일 적용(`lower.cu` `ComputeEffectivePanelWidth`; width 8 이 공정-튜닝 최적, _legacy §08) |
| `use_parallel_nested_dissection` | `true` | 멀티스레드 METIS-ND (비결정적; 재현 벤치는 serial) |
| `use_matching` / `matching` | `false` / `None` | 선택적 구조적 row/col matching (Hopcroft–Karp) |
| `pivot_strategy` | `StaticDiagonalShift` | no-pivot + 특이 pivot 시 정적 대각 shift |
| `enable_shift_retry` | `true` | shift fallback 활성 |
| `shift_retry_epsilon` | `1e-8` | pivot 임계 / shift 크기 |
| `analyze_emit_info` | `false` | analyze 후 front-size·subtree 요약을 stderr 출력 |
| `analyze_dump_fronts_path` | `""` | 비어있지 않으면 per-front CSV `(q,p,fsz,nc,uc,level,…)` 기록 |

> 멀티스트림 서브트리 디스패치는 `SolverConfig` 필드가 아니라 런타임 상태(`State.num_subtree_streams`, 상한
> `kMaxSubtreeStreams=8`)로 제어된다.

---

## 4. 빌드 (`CMakeLists.txt`)

METIS 필수. sm_80+ 필요(TF32 mma).

```bash
cmake -S custom_linear_solver -B build -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86
cmake --build build -j

build/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --precision tf32 --batch 64
```

| CMake 옵션 | 기본 | 효과 |
|---|---|---|
| `CLS_BUILD_CUDA_OPS` | `ON` | multifrontal CUDA 커널 빌드 |
| `CLS_INTERNAL_GRAPH` | `ON` | factor/solve 를 내부 CUDA graph 로 캡처(standalone). `OFF` = 외부 캡처 모드 |
| `CLS_BUILD_SCRIPTS` | `ON` | `custom_linear_solver_run` CLI 빌드 |
| `CLS_BUILD_CUDSS_SCRIPT` | `OFF` | cuDSS 비교 드라이버 빌드 (cuDSS 헤더/lib 필요) |
| `CLS_CUDA_ARCHITECTURES` | `86` | TF32 커널 위해 **≥ 80 필수** |

권장 런타임: `--precision tf32` → front=FP32, trailing=TF32 텐서코어 + **Ozaki 보정(상시 컴파일, 별도 빌드 플래그
없음)** → relres 가 FP32 band(~1e-4)이면서 fp32 보다 빠르다. 배치 `--batch 64`(latency)~`256`(throughput). ND
결정성(재현/벤치)은 `--serial-nd`.

---

## 5. 정밀도 매트릭스

| 모드 | front 저장 | Phase-3 trailing GEMM | accumulate | power-grid Jacobian 정확도 |
|---|---|---|---|---|
| `FP64` | FP64 | scalar FP64 | FP64 | ~1e-13 |
| `FP32` | FP32 | staged-scalar FP32 | FP32 | ~1e-4 |
| `TF32` | FP32 | TF32 PTX `mma.m16n8k8`/k4 (Ozaki 보정) | FP32 | ~1e-4 (TF32 rounding) |

TF32 는 mid/big tier trailing 을 텐서코어로 돌리고, Ozaki(error-free transform) 보정으로 FP32-band 정확도를 회복한다.
small tier 는 텐서코어를 쓰지 않는다(K=nc 가 1~2 라 구조적 무이득 — `_legacy/` 참조).

---

## 6. 파이프라인 구조

### 6.1 Analyze (`src/analyze/pipeline.cpp`)
1. CSR → CSC (device, `pattern_kernels.cu`).
2. 대칭 인접그래프 `A+Aᵀ` device 빌드.
3. **METIS nested dissection** (`reorder/metis_nd.cpp`, parallel 또는 serial `METIS_NodeND`).
4. (선택) 구조적 matching (Hopcroft–Karp, `pipeline.cpp`).
5. **elimination tree** (Liu 1986, `symbolic/elimination_tree.cpp`).
6. **fill pattern** (Davis/CSparse, `symbolic/`) — METIS postorder 라 fill-neutral.
7. **supernode amalgamation** (`symbolic/supernode.cpp`, `max_panel_width`).
8. **multifrontal plan** + level/tier bucketing (`symbolic/multifrontal.cpp`, `internal/plan/multifrontal_plan.cu`).

### 6.2 Factorize (`src/factorize/factorize.cu`, `schedule.cuh`)
`UseSingleSystem(st)` (= `B == 1`) 가 두 일관 경로를 가른다.
- **`B == 1` 단일 시스템**: 배치 schedule 의 per-level launch/barrier latency 가 노출돼, front 당 1 block 으로
  fused 단일 시스템 커널(`single.cuh`) 실행 → pivot 블록을 partitioned-inverse → solve 는 병렬 GEMV.
  TF32 면 big trailing 을 텐서코어(`FactorSingleBigTrailTf32`, Ozaki)로.
- **`B > 1` 배치**: panel etree 를 레벨별로 내려가며 독립 서브트리를 별도 스트림에 fork, 각 (레벨,tier) 동질
  구간을 전용 커널로 디스패치. 라우팅은 front 크기의 결정적 함수(`ClassifyFrontTier`).

### 6.3 Solve (`src/solve/solve.cu`)
gather + 전진/후진 대입 levels + scatter 를 전체 solve graph 로 lazy 캡처해 replay. small tier 는 sub-group
패킹, B=1 은 partitioned-inverse 로 병렬 GEMV. (`dispatch.cuh`/`kernels.cuh`/`phases.cuh`/`single.cuh`.)

---

## 7. 3-tier 커널 라우팅 (`src/internal/types.hpp`)

`ClassifyFrontTier(front_size)` — front 크기만의 **결정적** 함수(점유 휴리스틱 게이트 없음). 경계는 둘 다 물리적:

| tier | front 크기 | 커널 | 근거 |
|---|---|---|---|
| **small** | `fsz ≤ 32` (`kSmallFrontMax = kWarpSize`) | `FactorSmall` — warp-packed sub-group, 8 warps/block, `__syncwarp`만 | small\|mid = **warp 폭 32**. 그 아래선 warp-per-front + sub-group 패킹이 block-per-front 보다 빠름(수만 개 leaf) |
| **mid** | `33 ≤ fsz ≤ 64` (`kMidFrontMax = 64`) | `FactorMid` — whole-front shared-resident, 1 block/front | mid\|big = **점유 교차점 64**. whole-front staging 이 `fsz²·elem` 라 64 초과 시 SM 당 점유가 ~2 block 아래로 |
| **big** | `fsz > 64` | `FactorBig` — global-resident, L/U 패널 타일만 staging, 한 front 를 여러 block 에 분산(pivot+panel+trailing multi-block) | front 가 적고 클 때 GPU 를 채우고, shared-resident 불가한 거대 separator(FEM/circuit root)도 흡수 |

- **small tier 점유 게이트**: 좁은 레벨이라 small 커널이 GPU 를 under-fill 하면 front 당 whole block(`FactorMid`)로
  돌리고, 포화하면 sub-group 패킹(`FactorSmall`). (정밀도와 직교: fp32 front 는 work ×2 로 계산.)
- `WholeFrontSharedMax` (159 float / 111 double) 는 **big 커널 내부 bounded-shared staging 상한**일 뿐 tier 경계가
  아니다. tier 경계는 32 / 64.

> 2026-06-18 통합: 옛 4-tier(big=panel-resident 65–111 / large=global >111)를 3-tier 로 합쳤다. panel-resident
> 티어는 배치에서 ~1.2% 뿐이고 65–111 을 global multi-block 으로 보내는 게 B=1 에서 16% 빨라 제거했다.
> 상세: [`_legacy/05-reports/10-tier-consolidation-2026-06-18.md`](_legacy/05-reports/10-tier-consolidation-2026-06-18.md).

---

## 8. 성능 (정직한 요약)

측정은 RTX 3090(sm_86). **모든 헤드라인엔 공정 통제 후 천장도 함께 적는다.**

- **vs STRUMPACK(+MAGMA), 동일 FP64·깊이 매칭**: 기본 비교 16–66×. 단 STRUMPACK NodeNDP 튜닝까지 공정 적용하면
  **~10×** 로 내려온다. → [`_legacy/05-reports/06-head-to-head-2026-06-16.md`](_legacy/05-reports/06-head-to-head-2026-06-16.md),
  [`_legacy/05-reports/08-fair-strumpack-tuning-2026-06-17.md`](_legacy/05-reports/08-fair-strumpack-tuning-2026-06-17.md),
  기전(ncu/nsys) [`_legacy/05-reports/09-strumpack-mechanism-ncu-2026-06-17.md`](_legacy/05-reports/09-strumpack-mechanism-ncu-2026-06-17.md).
- **vs cuDSS**: raw B=1 비교(5.8–44.7×)는 cuDSS 에 배치가 없어 **과대평가**. 공정한 ubatch+mt setup 은 B=256 tf32
  기준 **~3.6–10×**, 전체 NR 조류계산에선 custom-mixed 가 배치에서 cuDSS 대비 **~4–6×**.
  → 병합 문서 [`_legacy/05-reports/bench-vs-cudss-merged.md`](_legacy/05-reports/bench-vs-cudss-merged.md).
- **TF32 텐서코어의 정직한 천장**: best-vs-best 통제 시 TC 자체 기여는 중앙값 **~1.1×** (large-case +6~16%),
  ≤10K low-fill 에선 K=nc 가 1~2 라 net ≈ 0. → [`_legacy/05-reports/05-tf32-reproduction-2026-06-10.md`](_legacy/05-reports/05-tf32-reproduction-2026-06-10.md),
  [`_legacy/03-optimization-notes/03-tensor-core-investigation.md`](_legacy/03-optimization-notes/03-tensor-core-investigation.md).
- **일반화**: SuiteSparse circuit·2D/3D-FEM 까지 big multi-block 으로 확장(parabolic 에서 cuDSS 추월).
  → [`_legacy/05-reports/07-generalization-suitesparse-2026-06-16.md`](_legacy/05-reports/07-generalization-suitesparse-2026-06-16.md).

---

## 9. 신규성·기여 (정직한 프레이밍)

개별 GPU 기법(no-pivot, host-free CUDA graph, front-size tiering, warp packing, extend-add fusion, TF32 trailing)은
**대부분 선행연구에 있다**. 우리 기여는 — *packing*(occupancy)과 *whole-front fusion*(메모리 재사용)이라는
**합쳐질 수 없는 두 입도**를 front 를 **sub-group 으로 분해**해 한 커널에서 동시에 달성, small-front 에서 occupancy 를
12–20× 회복한 구조다. 무엇이 novel/prior-art 인지의 출처-대조 판정:
[`_legacy/main-report.md`](_legacy/main-report.md) §6, [`_legacy/novelty.html`](_legacy/novelty.html),
[`_legacy/01-orientation/02-related-work-and-novelty.md`](_legacy/01-orientation/02-related-work-and-novelty.md).

---

## 10. 소스 맵 (`src/`)

| 경로 | 역할 |
|---|---|
| `solver.{hpp,cpp}` | 공개 phase API + `SolverConfig` |
| `internal/types.hpp` | **tier 분류기 + 모든 canonical 상수**(`kSmallFrontMax`/`kMidFrontMax` 등) |
| `internal/plan/`, `internal/runtime/` | multifrontal plan, front-range caps, device state/setup, arena |
| `analyze/pipeline.cpp` | analyze 오케스트레이션 |
| `analyze/reorder/metis_nd.*` | METIS nested dissection |
| `analyze/symbolic/` | elimination tree, fill, supernode, multifrontal symbolic |
| `analyze/{pattern,plan}/` | device CSR↔CSC / 대칭그래프 커널, plan lower |
| `factorize/factorize.cu`, `schedule.cuh` | factor 진입점 + etree 스케줄 |
| `factorize/{small,mid,big,single}.cuh` | **tier별 / B=1 전용 커널** |
| `factorize/{front_ops,assemble}.cuh` | per-front phase(LU/U-solve/trailing/extend-add), 입력 조립 |
| `solve/{solve.cu,dispatch,kernels,phases,permute,single}.cuh` | solve 경로 |

---

## 11. `_legacy/`

이전 문서 트리 전체(orientation·design-analysis·optimization-notes·benchmarks·reports·lab-meeting·main-report·
optimal-configuration·HTML 등)와 obsolete 문서들이 [`_legacy/`](_legacy/) 에 보존돼 있다. 측정 원자료(TSV/nsys-rep),
실험 로그, 문헌 판정의 상세 근거가 필요할 때 참조하라. 안내는 [`_legacy/README.md`](_legacy/README.md).

> 주의: _legacy 의 날짜 박힌 로그·옛 설계 문서는 **작성 시점 사실**이다. tier 경계가 `33–159/≥160` 또는
> `MID_THRESH=128` 으로 보이면 그건 2026-06-18 통합 **이전** 표기다. 현재 사실은 항상 이 README 와 `src/` 가 기준.
