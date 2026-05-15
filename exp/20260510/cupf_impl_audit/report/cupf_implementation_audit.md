# cuPF Implementation Audit

## 1. Purpose

This audit inspects the current cuPF implementation to support annual-report writing for:

`I. 전년도 연구개발실적 / 2. 연구개발과제의 수행 과정 및 수행 내용`

The focus is implementation and research-progress evidence beyond the linear solver: mismatch, Jacobian update, cuDSS integration, voltage update, convergence check, memory layout, host/device movement, Python/Pybind/C++/CUDA stack, and batch/uniform-batch support. This is not an optimization pass.

## 2. Repository and Environment

| Item | Observation |
|---|---|
| Audit workspace | `/workspace/gpu-powerflow` |
| cuPF source tree inspected | `/workspace/gpu-powerflow-master/cuPF` |
| Note on source location | The requested working directory contains datasets/experiments. The active cuPF source tree was found in the sibling path above. |
| Build system | CMake with optional CUDA, cuDSS, Python bindings, benchmarks, timing, and NVTX options in `CMakeLists.txt`; Python packaging uses `scikit-build-core` and `pybind11` in `pyproject.toml`. |
| CUDA version | `nvcc 12.8.93` |
| GPU model | NVIDIA GeForce RTX 3090, driver 580.126.09, 24576 MiB |
| Relevant dependencies | Eigen3, spdlog, SuiteSparse/KLU, CUDA runtime/cuSPARSE link target, cuDSS, pybind11 |
| cuDSS path used | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss.so` |
| Benchmarks run | Yes. Small smoke tests only: `case14` batch 1 and 4, `case118` batch 1, profile `cuda_mixed_edge`. |

The cuPF source tree had pre-existing uncommitted changes. This audit treats the working tree as the current implementation and did not modify production source code.

Smoke benchmark summary:

| Case | Profile | Batch | Success | Iterations | Total ms | Mismatch ms | Jacobian ms | Factorize ms | Solve ms | Update V ms | Residual |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case14 | cuda_mixed_edge | 1 | true | 3 | 10.018 | 0.021 | 0.017 | 0.094 | 0.084 | 0.018 | 1.466020152496e-10 |
| case14 | cuda_mixed_edge | 4 | true | 3 | 18.351 | 0.018 | 0.018 | 8.366 | 0.087 | 0.018 | 1.481092540279e-10 |
| case118 | cuda_mixed_edge | 1 | true | 4 | 10.819 | 0.023 | 0.028 | 0.207 | 0.176 | 0.027 | 1.501104796020e-12 |

These timings are smoke-test evidence only. They should not be used as final performance claims without a controlled benchmark matrix.

## 3. cuPF Software Stack

The implementation stack is layered as follows:

- Python package: `python/cupf/__init__.py` imports types from `_cupf`, but no Python-side solve wrapper was found.
- Pybind layer: `bindings/pybind_cupf.cpp::PYBIND11_MODULE` exposes `BackendKind`, `ComputePolicy`, cuDSS options, `NewtonOptions`, `NRResult`, and `NewtonSolver` construction. `initialize` and `solve` bindings are marked TODO, so the Python API is currently incomplete for end-to-end solving.
- C++ core: `cpp/src/newton_solver/core/newton_solver.cpp::NewtonSolver` selects CPU FP64, CUDA FP64, CUDA FP32, or CUDA Mixed pipelines and owns initialization, solve/batch solve, and the Newton iteration sequence.
- CUDA kernels: CUDA paths use separate kernels for Ibus, mismatch, norm reduction, Jacobian fill, RHS cast for Mixed, and voltage update/reconstruction.
- cuDSS integration: `cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp` wraps cuDSS handle/config/data/matrix descriptors and calls analysis, factorization/refactorization, and solve phases.

The current C++ benchmark binary is the most complete executable path found for this audit.

## 4. Newton-Raphson Operation Map

| Operation | Implementation | GPU accelerated | Precision | Data location | Source evidence | Notes |
|---|---|---|---|---|---|---|
| initialize | C++ orchestration plus CPU Jacobian indexing/pattern/scatter-map analysis; CUDA storage uploads structure/maps; cuDSS descriptors created | Mixed | Index/pattern metadata on CPU; values later FP64 or FP32 | Host and device | `newton_solver.cpp::initialize`, `jacobian_analysis.cpp::build_*`, `cuda_*_storage.cpp::prepare`, `cuda_cudss.cpp::initialize` | Pattern and scatter maps are reused. cuDSS analysis can run at initialize unless matching requires matrix values. |
| compute mismatch | CUDA path first computes Ibus, then mismatch from cached Ibus; CPU path uses host vectors | Yes for CUDA profiles | FP64 for CPU/CUDA FP64/Mixed residual; FP32 for CUDA FP32 | Host for CPU, device for CUDA | `compute_ibus.cu::compute_ibus_kernel`, `compute_mismatch_from_ibus.cu::compute_mismatch_from_ibus_kernel`, `cpu_mismatch.cpp::CpuMismatchOp::run` | Mixed keeps mismatch in FP64. |
| update Jacobian | Edge-based CPU or CUDA fill using precomputed scatter maps | Yes for CUDA profiles | FP64 J for FP64 paths; FP32 J for FP32/Mixed | Host for CPU, device for CUDA | `fill_jacobian.cpp::CpuJacobianOpF64::run`, `fill_jacobian_gpu.cu::fill_jacobian_gpu_kernel` | No current `atomicAdd` use was found. CUDA `d_J_values` is zeroed every Jacobian update. |
| factorize | CPU KLU numeric factorization or cuDSS factorization/refactorization | Yes through cuDSS | FP64 for CPU/CUDA FP64; FP32 for CUDA FP32/Mixed | Host for CPU, device CSR for CUDA | `cpu_klu.cpp::factorize`, `cuda_cudss.cpp::factorize` | CPU KLU reuses symbolic analysis. cuDSS uses FACTORIZATION first and REFACTORIZATION later when descriptors match. |
| solve | CPU KLU solve or cuDSS solve phase | Yes through cuDSS | FP64 for CPU/CUDA FP64; FP32 for CUDA FP32/Mixed | Host for CPU, device for CUDA | `cpu_klu.cpp::solve`, `cuda_cudss.cpp::solve` | Mixed computes FP32 `dx`; voltage update consumes it against FP64 state. |
| update voltage | CPU host update/reconstruct or CUDA state-update and reconstruct kernels | Yes for CUDA profiles | FP64 for CPU/CUDA FP64; FP32 for CUDA FP32; Mixed FP64 state with FP32 dx | Host for CPU, device for CUDA | `cpu_voltage_update.cpp::run`, `cuda_voltage_update.cu::run`, `voltage_update_kernels.hpp` | Uses `sincos`/`sincosf` helpers; no `__sincosf` occurrence was found. |
| convergence check | CPU norm on host or CUDA norm reduction followed by D2H norm copy and host convergence decision | Partially | FP64 norm for CPU/CUDA FP64/Mixed; FP32 norm for CUDA FP32 | Device reduction plus host scalar/vector decision for CUDA | `reduce_mismatch_norm.cu`, `cuda_mismatch.cu::CudaMismatchNormOp::run` | Per-iteration D2H norm transfer remains. Batch convergence is host-side max over batch norms. |
| data transfer | Storage upload/download and per-iteration norm copy | Transfer itself is not acceleration | Public API is FP64; FP32 path casts on upload/download; Mixed keeps state/residual FP64 | Host and device | `cuda_utils.hpp::DeviceBuffer`, `cuda_fp32_storage.cpp`, `cuda_mixed_storage.cpp`, `cuda_fp64_storage.cpp` | Blocking `cudaMemcpy`/`cudaMemset`; no stream or async memcpy usage found in inspected source. |
| batch handling | `solve_batch` plus batch-major CUDA FP32/Mixed storage and cuDSS uniform-batch config | Yes for CUDA FP32/Mixed | FP32 or Mixed | Host inputs, device batch-major arrays | `newton_solver.cpp::solve_batch`, `pipeline.hpp`, `cuda_mixed_storage.cpp`, `cuda_fp32_storage.cpp`, `cudss_config.hpp` | CPU and CUDA FP64 pipelines do not support batch. Batched Ybus values are not enabled in the public path. |

## 5. Jacobian Implementation

The optimized Jacobian path is edge-based. `jacobian_analysis.cpp::build_jacobian_pattern` builds the sparse NR Jacobian CSR structure once, and `build_jacobian_scatter_map` maps each Ybus edge to up to four Jacobian block positions (`J11`, `J12`, `J21`, `J22`) plus diagonal slots. CUDA storage uploads these maps during `prepare`, so the iteration loop updates values without rebuilding the sparse pattern.

`fill_jacobian_gpu.cu::fill_jacobian_gpu_kernel` launches one thread per Ybus edge per batch. It reads SoA Ybus and voltage arrays, computes off-diagonal terms, and writes to precomputed Jacobian positions. Diagonal correction is handled by the thread for the diagonal Ybus entry. A source search found no `atomicAdd` in the current Jacobian kernel, so the current design avoids atomic accumulation through direct scatter positions and diagonal slots.

The implementation explicitly separates off-diagonal edge terms from diagonal correction. CPU edge code in `fill_jacobian.cpp::CpuJacobianOpF64::run` follows the same pattern. The reference `cpu_naive_jacobian_f64.cpp::CpuNaiveJacobianOpF64::run` is different: it rebuilds derivative sparse matrices and slices subblocks, so it is best interpreted as a PyPower-like baseline rather than the optimized implementation.

Memory layout evidence is favorable but not a complete coalescing proof. CUDA buffers are structure-of-arrays (`d_Ybus_re`, `d_Ybus_im`, `d_V_re`, `d_V_im`, separate index maps), and FP32/Mixed batch buffers use batch-major offsets. Edge arrays are traversed contiguously by edge id, but writes into `d_J_values` are scatter writes through precomputed maps. No NCU-derived coalescing metric was generated during this audit.

Precision by path:

- CPU FP64 and CUDA FP64 use double Jacobian values.
- CUDA FP32 uses float state, mismatch, Jacobian, and dx.
- CUDA Mixed uses double Ybus/V/Sbus/Ibus/F/norm and float Jacobian/dx.

Batch support exists for CUDA FP32 and CUDA Mixed. The Jacobian kernel supports batch-major `J_values`, and the Mixed/FP32 pipelines set `batch_supported=true`. CUDA FP64 is single-case in the current pipeline.

Efficiency notes from code:

- Precomputed mapping tables are used.
- No `atomicAdd` was found in current Jacobian code.
- `d_J_values.memsetZero()` runs every Jacobian update.
- CUDA Mixed and FP32 Jacobian paths use cached Ibus for diagonal correction.
- CUDA FP64 Jacobian currently passes `use_cached_ibus=false`, so it recomputes diagonal current even though a separate Ibus stage exists.

## 6. Mixed Precision Implementation

The Mixed policy is documented in `newton_solver_types.hpp::ComputePolicy` and implemented by `pipeline.hpp::CudaMixedPipeline`.

Mixed storage in `cuda_mixed_storage.hpp::CudaMixedBuffers` keeps:

- FP64: Ybus values, voltage state (`V`, `Va`, `Vm`), Sbus, Ibus, residual `F`, and norm.
- FP32: Jacobian values `d_J_values` and Newton update `d_dx`.

`cuda_cudss.cpp::CudaLinearSolveCuDSS<float, CudaMixedBuffers>::prepare_rhs` launches `prepare_rhs.cu::prepare_rhs_kernel` to cast the FP64 residual to a FP32 RHS buffer on device. The linear solve is then cuDSS float. `cuda_voltage_update.cu::CudaVoltageUpdateOp<CudaMixedBuffers>::run` applies FP32 `dx` to FP64 `Va`/`Vm` and reconstructs FP64 voltage.

Therefore:

- Mismatch remains FP64 in Mixed.
- Convergence norm remains FP64 in Mixed.
- Jacobian and dx are FP32 in Mixed.
- Conversion from FP64 residual to FP32 RHS occurs every Newton iteration, but it is a device-side kernel rather than a host transfer.
- Final voltage remains FP64 in the public result.

The current working tree also contains a full CUDA FP32 path (`CudaFp32Pipeline`, `CudaFp32Buffers`) that casts public FP64 inputs to float on upload and casts final state back to double on download. Pybind exposes `ComputePolicy::FP64` and `ComputePolicy::Mixed`; current binding lines do not expose `FP32`.

## 7. Non-Jacobian CUDA Operations

Ibus is implemented by `compute_ibus.cu::compute_ibus_kernel`, a custom CSR SpMV-like kernel. It uses one row per block and warp reduction. The FP32 and Mixed launchers support batch tiling, but explicitly reject batched Ybus values; current public batch usage assumes shared Ybus values.

Mismatch is implemented by `compute_mismatch_from_ibus.cu::compute_mismatch_from_ibus_kernel`. It consumes device-resident V, Ibus, and Sbus arrays and writes the NR residual. Mixed residual values are double.

Convergence norm is implemented by `reduce_mismatch_norm.cu::reduce_mismatch_norm_kernel`, but `cuda_mismatch.cu::CudaMismatchNormOp::run` copies the norm scalar/vector back to host every iteration and sets `ctx.converged` on CPU. This is a mixed CPU/GPU control path. It avoids copying the full residual unless debug dump is enabled, but it still creates a per-iteration synchronization point.

Voltage update is implemented by `voltage_update_kernels.hpp::apply_voltage_update_kernel` and `reconstruct_voltage_kernel`, launched from `cuda_voltage_update.cu`. The update remains device-resident until final download. Source search found `sincos`/`sincosf`, not `__sincosf`.

Host/device movement is concentrated in storage and convergence:

- Upload per solve: Ybus values, Sbus, V0, and batch input arrays.
- Per iteration: norm scalar/vector D2H; Mixed RHS cast device kernel; cuDSS library work not visible from source.
- Final download: final voltage and norm arrays.
- Debug dumps can add full residual/Jacobian D2H copies if enabled.

No `cudaMemcpyAsync`, `cudaStream`, or `cudaMallocAsync` usage was found in the inspected cuPF source. `DeviceBuffer` uses blocking `cudaMemcpy`, `cudaMalloc`, and `cudaMemset`.

## 8. cuDSS Integration

`cuda_cudss.cpp` implements the cuDSS integration through `CudaLinearSolveCuDSS<T, Buffers>`.

Observed behavior:

- `initialize` creates cuDSS handle, config, data, and descriptors.
- `ensure_descriptors` creates CSR and dense matrix descriptors and recreates them only when batch size, Jacobian dimension, or nnz changes.
- `configure_solver` in `cudss_config.hpp` sets reordering, matching, host thread count, nested-dissection level, pivot epsilon, and `CUDSS_CONFIG_UBATCH_SIZE` when batch size is greater than 1.
- `factorize` runs analysis if needed, then `CUDSS_PHASE_FACTORIZATION` on the first numeric factorization and `CUDSS_PHASE_REFACTORIZATION` on later iterations.
- `solve` runs `CUDSS_PHASE_SOLVE`.
- Mixed uses cuDSS float and a device-side FP64-to-FP32 RHS cast.

Sparsity pattern reuse is implemented at the cuPF level by holding descriptors and analysis state while dimensions and nnz remain unchanged. The exact cuDSS internal workspace allocation and any hidden internal host/device movement cannot be verified from cuPF source.

This section intentionally does not compare solver algorithms or performance beyond confirming integration behavior. Solver benchmarking belongs in a separate linear-solver benchmark.

## 9. Multi-Batch or Uniform-Batch Support

Uniform-batch support exists in the current CUDA FP32 and CUDA Mixed pipelines.

Evidence:

- `newton_solver.cpp::solve_batch` accepts multiple Sbus/V0 cases and rejects batch sizes greater than 1 when the selected pipeline does not support batch.
- `pipeline.hpp` sets `batch_supported=true` for CUDA FP32 and CUDA Mixed, and `false` for CPU FP64 and CUDA FP64.
- `cuda_fp32_storage.cpp` and `cuda_mixed_storage.cpp` allocate batch-major arrays and compute per-batch offsets.
- `fill_jacobian_gpu.cu` updates batch-major Jacobian values.
- `cudss_config.hpp::configure_solver` sets `CUDSS_CONFIG_UBATCH_SIZE` for batch size greater than 1.
- A live smoke test with `case14`, `cuda_mixed_edge`, and `--batch-size 4` converged.

Current limitations:

- The public batch path appears to assume shared Ybus structure and values. `solve_batch` passes a single Ybus view, and `compute_ibus.cu` rejects `ybus_values_batched` for FP32/Mixed.
- Batch-specific Sbus and V0 are supported.
- Batch-specific Jacobian values are supported because state and mismatch differ by batch.
- No active mask or per-case early exit was found. The batch loop uses a host-side convergence decision and reports a common iteration count.
- CPU and CUDA FP64 batch support are not implemented in the current pipelines.

## 10. Evidence for Annual Report

- 전력조류계산의 반복 내부 연산을 GPU 친화적으로 분해하고, Python API와 C++/CUDA 커널을 분리한 cuPF 소프트웨어 스택을 구축하였다.  
  구현 근거: `CMakeLists.txt`, `pyproject.toml`, `bindings/pybind_cupf.cpp`, `newton_solver.cpp`, `pipeline.hpp`에서 Python/Pybind/C++/CUDA/cuDSS 계층이 확인된다.  
  관측 결과: C++ 벤치마크 경로에서 `cuda_mixed_edge` 프로파일이 `case14`와 `case118`에 대해 수렴하였다.  
  남은 병목: Python 바인딩은 Solver 생성까지만 노출되어 있고, `initialize/solve` 바인딩은 TODO 상태이다.

- 자코비안 계산은 Ybus edge 기반 scatter map 방식으로 구현하여 반복 중 희소 패턴 재구성을 피하도록 설계하였다.  
  구현 근거: `jacobian_analysis.cpp`가 CSR 패턴과 `mapJ11/mapJ12/mapJ21/mapJ22`를 생성하고, `fill_jacobian_gpu.cu::fill_jacobian_gpu_kernel`이 edge별로 값을 갱신한다.  
  관측 결과: `case118` smoke test에서 Jacobian stage total은 0.028 ms로 계측되었다.  
  남은 병목: `d_J_values`는 매 Jacobian 갱신마다 zero-fill되며, scatter write의 실제 coalescing 효율은 이번 감사에서 별도 측정하지 않았다.

- 선형계 풀이는 cuDSS의 analysis, factorization/refactorization, solve 단계를 Newton 반복 구조에 통합하였다.  
  구현 근거: `cuda_cudss.cpp`에서 descriptor 생성, analysis, FACTORIZATION/REFACTORIZATION, SOLVE 호출이 확인되고, `cudss_config.hpp`에서 uniform batch 설정이 확인된다.  
  관측 결과: `case14` batch 1 smoke test에서 factorize 0.094 ms, solve 0.084 ms가 계측되었다.  
  남은 병목: cuDSS 내부 workspace allocation이나 숨은 host/device transfer는 cuPF 소스만으로 검증할 수 없다.

- 혼합정밀도 경로는 전력 불일치와 수렴 판정은 FP64로 유지하고, 자코비안과 선형계 보정량은 FP32로 처리하도록 구현하였다.  
  구현 근거: `cuda_mixed_storage.hpp`는 FP64 상태/불일치와 FP32 `J/dx`를 분리하고, `prepare_rhs.cu`는 FP64 residual을 FP32 RHS로 device-side 변환한다.  
  관측 결과: `cuda_mixed_edge` profile로 `case14`, `case118`가 수렴하였다.  
  남은 병목: Mixed RHS cast는 매 반복 수행되며, FP32/FP64 정책별 정확도와 속도 비교는 이번 감사 범위를 넘는다.

- 전력 불일치 계산과 전압 갱신은 자코비안 외부 연산까지 CUDA 커널로 분리 구현하였다.  
  구현 근거: `compute_ibus.cu`, `compute_mismatch_from_ibus.cu`, `reduce_mismatch_norm.cu`, `cuda_voltage_update.cu`에서 각각 Ibus, mismatch, norm, voltage update 커널이 확인된다.  
  관측 결과: `case14` batch 1 smoke test에서 mismatch 0.021 ms, voltage update 0.018 ms가 계측되었다.  
  남은 병목: 수렴 판정은 norm을 매 반복 host로 복사하여 CPU에서 결정한다.

- 동일 계통 구조를 공유하는 uniform-batch 계산을 위한 batch-major 메모리 배치와 cuDSS uniform-batch 설정을 구현하였다.  
  구현 근거: `solve_batch`, `CudaMixedBuffers`, `CudaFp32Buffers`, `CUDSS_CONFIG_UBATCH_SIZE` 설정이 확인된다.  
  관측 결과: `case14` batch 4 smoke test가 수렴하였다.  
  남은 병목: 현재 public path는 batch별 Ybus 값 변경을 지원하지 않는 것으로 보이며, active-mask 기반 조기 종료는 확인되지 않았다.

## 11. Missing Evidence and Follow-up Experiments

Missing or unclear evidence:

- Python end-to-end solve could not be verified because pybind solve/init bindings are incomplete.
- CPU, CUDA FP64, CUDA FP32, and Python profile timings were not rerun in this audit.
- Korean grid or large-grid benchmarks were not run.
- cuDSS internal allocations, workspace reuse, and hidden transfers cannot be verified from cuPF source.
- No stream or async copy strategy was found; all visible transfers use blocking `cudaMemcpy`.
- No current NCU coalescing/occupancy analysis was generated by this audit, although profiling artifacts and instrumentation exist in the repository.
- No per-operation timing evidence was collected for every compute policy and batch size.
- Batch-specific Ybus values are unclear or unsupported in the public path; Ibus launchers explicitly reject batched Ybus values.
- Batch convergence has no active-mask or per-case early-exit evidence.
- CUDA FP64 Jacobian recomputes diagonal current rather than using cached Ibus, unlike Mixed/FP32.
- `d_J_values.memsetZero()` remains a repeated per-iteration operation.

Recommended follow-up before final annual-report performance claims:

- Run a controlled benchmark matrix for CPU FP64, CUDA FP64, CUDA FP32, and CUDA Mixed on the same cases.
- Include at least one medium and one large grid, plus the Korean grid if it is already supported.
- Capture Nsight Compute metrics for Jacobian, Ibus, mismatch, norm, and voltage update kernels.
- Measure H2D/D2H traffic and synchronization overhead, especially norm D2H and final download.
- Verify cuDSS workspace behavior and refactorization reuse with Nsight Systems.
- Add or complete Python solve bindings before making Python API usability claims.
- Decide whether batch-specific Ybus is a requirement; if so, implement and test the Ibus path accordingly.
