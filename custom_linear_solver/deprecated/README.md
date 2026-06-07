# Deprecated 코드 보존

`custom_linear_solver` 리팩토링 (2026-06-05~) 과정에서 제거된 코드의 보존소.
삭제 대신 폴더로 옮긴 이유: (a) 변경 의도와 컨텍스트를 잃지 않기 위함, (b) 향후 재도입 필요 시 참조용, (c) 측정/실험 흔적 재현 가능.

## 폴더 별 내용

### `microbench/`
독립 마이크로벤치 (.cu 단일 파일, 본 라이브러리와 연결되지 않음).

- `tc_trailing_microbench.cu` — trailing GEMM 변형 (FP32 scalar, FP16 WMMA, 등) 의 throughput 비교.
- `wmma_pack_microbench.cu` — WMMA fragment packing 의 측정 실험.

### `profiling_no_trailing/`
*Σ.16 measurement instrument*. 정상 factor 커널의 *trailing update 만 skip 한 NT clone*. NT wall 측정 → trailing GEMM wall 의 직접 분리에 사용. 정확도 깨짐 (잘못된 factor 생성). production 경로 아님.

- `factor_no_trailing.cuh` — `mf_factor_*_NT_b` 4종.
- 활성화 env (옛): `CLS_PROFILE_NO_TRAILING=1`.

### `tc_dedicated/`
TC 전용 entry point. batched 경로의 `BatchPrecision::TC32` 와 별개로 *별도 setup/factorize/solve API* 를 노출했던 실험. 실제로는 batched TC32 와 거의 같은 dispatcher 라 통합 후 삭제.

- `multifrontal_tc.{cu,hpp}` — TCState, tc_setup/tc_factorize/tc_solve.
- `factor_split_cublas.cuh` — cuBLAS gemmStridedBatched 의 trailing 실험 (성능 비교용).
- `spine_kernel.cuh` — spine level 전용 kernel 실험.
- 활성화 env (옛): `--tc` runner flag, `CLS_USE_CUBLAS`, `CLS_USE_REGBLOCK`, `CLS_USE_REGBLOCK_H16`, `CLS_USE_SPINE`, `CLS_BYPASS_GRAPH`, `CLS_TC_SETUP_DBG`, `CLS_CUBLAS_TF32`, `CLS_CUBLAS_MIN_FSZ`.

### `selinv/`
selinv (selective inversion) 사전계산 path. factorize 끝에 각 front 의 nc×nc pivot block 을 미리 역행렬화해서 solve 의 triangular solve 를 GEMV 로 변환. NR loop / contingency 같은 *1 factor + 다수 solve* 시나리오에 이득. power-grid 의 *1 factor + 1 solve* 패턴에선 손해.

- `invert_pivot.cuh` — `mf_invert_pivot_b<FT>` (코드 내부는 FP64 inverse 라 FP32 mode 라도 RTX 3090 FP64 1/64 throughput 에 묶임).
- 활성화 env (옛): `CLS_USE_SELINV=1` (default OFF post-Σ.9), `MF_NO_SELINV=1` (kill-switch).

### `precision_mixed/` (예약 — Phase 3 에서 채워짐)
Mixed precision path (FP64 master front + FP32 working LU). 정확도 ~1e-5, RTX 3090 위 FP32 와 거의 동률 성능. 삭제 이유: API 단순화 (3 mode 만 — FP64/FP32/TC), 메모리 1.5× 점유.

### `amalgamation/` (예약 — Phase 3 에서 채워짐)
Symbolic amalgamation. 작은 child fronts 들을 한 supernode 로 합쳐 trailing GEMM 의 nc 를 키우기. TC mode 의 성능 회복용 실험. NR Jacobian 의 sparsity 구조와 안 맞음.

### `single_system/` (예약 — Phase 5 에서 채워짐)
B=1 전용 factorize / solve 경로. batched(B=1) 와 중복. batched 가 단일 진입점이 되어 삭제.

### `mid_warp/` (2026-06-06)
T4.1 실험. SMALL tier 의 warp-per-front 패턴을 mid range (32 < fsz ≤ MID_WARP_THRESH) 로 확장한 새 커널. ncu 로 barrier stall 41% → 0% 확인했으나 load imbalance 로 occupancy 46% → 12% 추락 → 전체 wall net loss. variance gate (`CLS_MID_WARP_VAR_GATE`) 로 USA opt-in 일부 케이스에서 −4% 가능하나 ROI 낮음. 자세한 측정: `docs/03-optimization-notes/10`.

- `mid_warp.cuh` — `factor_mid_warp<T>` kernel + `lu_mid_warp` helper.
- 활성화 env (옛): `CLS_MID_WARP_THRESH`, `CLS_MID_WARP_VAR_GATE`.

### `mid_opt/` (2026-06-06)
docs/13 의 P1+P2(+P4) 결합 별도 커널. P1 (reciprocal multiply) 은 default kernel 에 흡수되어 retained; P2 (Phase 1+2 fusion) 만 별도 kernel 로 실험. P4 (shared padding) 는 stage-in integer division 비용으로 +85% 회귀 → 폐기. P2 단독으로는 USA B≥16 에서 −1~−4% 작은 win, case8387 noise. mid kernel micro-optimization 한계 도달의 evidence. 자세한 측정: `docs/03-optimization-notes/14`.

- `mid_opt.cuh` — `factor_mid_opt<T>` kernel.
- 활성화 env (옛): `CLS_USE_OPT_MID`.

## 복원 방법

1. 필요 파일을 `deprecated/<area>/` 에서 `src/<해당 폴더>/` 로 복사.
2. `CMakeLists.txt` 의 `CLS_OP_SOURCES` 에 추가.
3. 진입점 (`solver.cpp`, runner) 에 includes / API 복원.
4. 코드 내 env var 검사 복원 (각 `<area>` 의 README 가 명시).

## 정책

- **삭제는 깨끗하게 한 번에**: 흩어진 `#if`/`#ifdef` 대신 폴더 단위로 분리.
- **deprecated 코드는 빌드에서 제외**: CMake 가 deprecated/ 를 스캔하지 않음.
- **이름 / 시그니처 보존**: 추후 git 검색 가능하도록 원본 식별자 유지.
- **이 README 갱신 의무**: 새 폴더 추가 시 / 복원 시 / 영구 삭제 시.
