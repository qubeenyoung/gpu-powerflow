# 03 — 공개 API · 설정 · 빌드 · 실행

> **층위**: 사용. 라이브러리를 어떻게 호출하고(API), 무엇을 조절하며(SolverConfig / 빌드 옵션), 어떻게 빌드·실행하는지.
> 모든 값은 현재 소스(`src/solver.hpp`, `CMakeLists.txt`, `tests/run_custom_solver.cu`) 기준이다.

---

## 1. 공개 API — phase 모델 (`src/solver.hpp`)

cuDSS phase 모델을 1:1 모방한다. 전형적 시퀀스:

```cpp
using namespace custom_linear_solver;

Solver solver(config);                 // SolverConfig (아래 §2)
solver.SetData(matrix);                // A: CSR sparsity + 값 포인터 (device)
solver.SetRhs(rhs);                    // b (device)
solver.SetSolution(solution);          // x (device, 출력)

solver.Analyze();                      // 1회: METIS ND + 기호 분석 + plan 빌드
solver.Setup(/*batch_size=*/B);        // B 시스템용 런타임 상태/arena 할당
for (each NR iteration) {
  solver.SetValues(new_values);        // 값만 갱신 (패턴 동일)
  solver.Factorize();                  // 수치 분해
  solver.Solve();                      // x = A^{-1} b
}
```

| 메서드 | 역할 |
|---|---|
| `SetData(CsrMatrixView)` | 행렬 패턴 + 값 포인터 등록 |
| `SetValues(const void*, ValueType)` | 값만 갱신(같은 패턴 재-factorize용) |
| `SetRhs / SetSolution(DenseVectorView)` | 우변 `b` / 해 `x` 버퍼 등록 |
| `get_data / get_rhs / get_solution` | 등록된 디스크립터 조회 |
| `Analyze()` | 기호 분석 + plan(한 번; 패턴 고정 가정) |
| `Setup(int batch_size = 1)` | B 시스템 런타임 상태 할당 |
| `Factorize()` / `Solve()` | 수치 분해 / 삼각 solve |
| `SetStream(void*)` | 외부 캡처 모드(`CLS_INTERNAL_GRAPH` off)에서 caller 스트림 바인딩 |

- `B > 1`이면 등록 버퍼는 batch-strided: `values[b*nnz+·]`, `rhs[b*n+·]`, `solution[b*n+·]`.
- 반환 `Status`: `kSuccess` / `kInvalidValue` / `kInvalidState` / `kAllocationFailed` / `kAnalysisFailed` /
  `kFactorizationFailed` / `kSolveFailed`.
- 런타임 동작(graph 캡처, B=1 vs B>1)은 [08](08-runtime-and-batching.md).

## 2. 설정 — `SolverConfig` 기본값

`Solver` 생성자 인자. **모든 값이 기본값**이고, 크기 기준 tier 경계·TF32 PTX 스택·per-front 라우팅처럼 측정상
off-default가 회귀한 튜너블은 **빌드에 baked-in**되어 설정에 노출하지 않는다.

| 필드 | 기본 | 설명 |
|---|---|---|
| `precision` | `FP64` | `FP64` / `FP32` / `TF32`(Ozaki mma, 권장). [09](09-precision-and-tensor-cores.md) |
| `matching` | `MatchingMode::None` | 선택적 구조적 row/col matching(Hopcroft–Karp) |
| `use_parallel_nested_dissection` | `true` | 멀티스레드 METIS-ND. `false`면 결정적 serial `METIS_NodeND` |
| `parallel_nd_depth` | `4` | 병렬 ND 재귀 깊이 |
| `parallel_nd_base_small` | `4000` | `n < 20000`일 때 병렬 ND base-case 임계(이하 serial) |
| `parallel_nd_base_large` | `20000` | `n ≥ 20000`일 때 base-case 임계 |
| `max_panel_width` | `8` | supernode panel 최대 열 수. 분석기가 `[1,64]`로 clamp(전 사이즈 균일) |
| `pivot_strategy` | `StaticDiagonalShift` | no-pivot + 특이 pivot 시 정적 대각 shift (`None` 가능) |
| `shift_retry_epsilon` | `1e-8` | pivot 임계 / shift 크기 (`0`이면 shift 비활성) |
| `analyze_dump_fronts_path` | `""` | 비면 analyze 후 per-front CSV 기록 |
| `analyze_fronts_only` | `false` | numeric arena 없이 symbolic만(분석 툴링용) |
| `analyze_emit_info` | `false` | analyze 후 front-size·subtree 요약을 stderr |

> 멀티스트림 서브트리 디스패치는 `SolverConfig` 필드가 아니라 런타임 상태(`State.num_subtree_streams`, 상한
> `kMaxSubtreeStreams=8`)가 plan의 독립 서브트리 수에서 결정한다. [08 §3](08-runtime-and-batching.md).

## 3. 빌드 — CMake (`CMakeLists.txt`)

METIS 필수. sm_80+ 필요(TF32 mma).

```bash
cmake -S custom_linear_solver -B build -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86
cmake --build build -j
```

| CMake 옵션 | 기본 | 효과 |
|---|---|---|
| `CLS_BUILD_CUDA_OPS` | `ON` | multifrontal CUDA 커널 라이브러리 빌드 |
| `CLS_INTERNAL_GRAPH` | `ON` | factor/solve를 내부 CUDA graph로 캡처(standalone). `OFF`=외부 캡처 모드 |
| `CLS_BUILD_SCRIPTS` | `ON` | `custom_linear_solver_run` CLI 빌드 |
| `CLS_CUDA_ARCHITECTURES` | `"86"` | CUDA arch. TF32 mma 위해 **≥ 80 필수**(CMake가 강제) |

## 4. 컴파일타임 매크로 (실재하는 것만)

런타임 동작에 영향을 주는 `-D` 매크로는 **하나뿐**이고, 나머지 `CLS_*`는 내부 PTX 헬퍼다. (과거 튜닝 매크로
`CLS_PAR_ND_*`, `CLS_TC_UC_CAP`, `CLS_TC_*_MIN`, `CLS_USE_PIVOTING`, `CLS_BUILD_CUDSS_SCRIPT`는 제거됐다 — 각각
`SolverConfig` 필드 / `types.hpp` constexpr `kTensorCoreUcCap` / 항상 참인 게이트라 폐지 / 미사용이라 폐지.)

| 매크로 | 기본 | 위치 | 역할 |
|---|---|---|---|
| `CLS_INTERNAL_GRAPH` | ON(CMake) | `setup.cu`/`factorize.cu`/`solve.cu` | 내부 graph 캡처 vs 외부 캡처 모드(실효 토글) |
| `CLS_MMA_TF32_M16N8K8`, `CLS_MMA_TF32_OZAKI2` | (내부) | `front_ops.cuh` | TF32 PTX `mma.sync` 인라인 asm 래퍼. 사용자 노브 아님 |

> **TF32와 Ozaki**: TF32는 `--precision tf32`(=`SolverConfig.precision=TF32`)만으로 켜진다. 정확도 회복용 Ozaki
> head/tail 4-pass 곱은 TF32 경로에 **상시 컴파일**되어 있고(별도 빌드 플래그 없음), FP32/FP64 인스턴스에는
> `if constexpr`로 **아예 포함되지 않는다**(레지스터 0). 상세는 [09 §3](09-precision-and-tensor-cores.md).

## 5. 실행 — `custom_linear_solver_run` CLI (`tests/run_custom_solver.cu`)

입력은 Matrix Market(coordinate 행렬 / array 벡터, `tests/io.cpp`).

```bash
# 단일 시스템(FP64 참조), 10회 타이밍
build/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --repeat 10 --warmup 2

# 배치(B=64, TF32 = Ozaki mma trailing)
build/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --precision tf32 --batch 64

# 재현 벤치(결정적): serial ND
build/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --precision tf32 --batch 64 --serial-nd

# analyze 산출물만: front 분포 CSV
build/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --dump-fronts fronts.csv
```

`parse_args`가 받는 전체 플래그:

| 플래그 | 기본 | 역할 |
|---|---|---|
| `--matrix <p>` / `--rhs <p>` | (case dir) | 행렬 `A` / RHS `b` 로드 |
| `--solution-out <p>` | "" | 해 `x`를 Matrix Market로 기록 |
| `--repeat <N>` / `--warmup <N>` | 1 / 0 | 타이밍 반복 / 워밍업(중앙값 보고) |
| `--batch <B>` | 0 | 배치 실험(B 시스템, 한 symbolic 공유) |
| `--batch-only` | false | 단일 시스템 실행 생략 |
| `--single-precision fp64\|fp32` | fp64 | **단일 시스템 입력** 정밀도(배치와 독립) |
| `--precision fp64\|fp32\|tf32` | fp64 | **배치 factor** 정밀도(`SolverConfig.precision`로 전달) |
| `--max-panel-width <N>` | 8 | supernode panel 최대 열 수 |
| `--serial-nd` | (parallel) | 결정적 serial METIS NodeND |
| `--matching` | false | 구조적 row/col matching |
| `--pivot-strategy none\|shift` | shift | pivot 전략 |
| `--pivot-epsilon <x>` | 1e-8 | shift 임계/크기 |
| `--dump-fronts <p>` | "" | analyze 후 per-front CSV |
| `-h, --help` | — | 사용법 |

> `--single-precision`은 **단일 시스템 입력**을 float로 올리는 것이고, 배치 정밀도는 `--precision`이 결정한다(두 경로
> 독립). `SolverConfig`의 `analyze_emit_info`/`analyze_fronts_only` 같은 CLI 미노출 필드는 라이브러리 임베드 시 직접 설정.
