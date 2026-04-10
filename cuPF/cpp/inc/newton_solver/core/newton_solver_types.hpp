#pragma once

#include "jacobian_types.hpp"

#include <cstdint>
#include <complex>
#include <vector>


// ---------------------------------------------------------------------------
// CSRView: non-owning view over a CSR matrix.
//
// The caller (Python binding or C++ user) owns the memory. CSRView holds
// raw pointers only — zero-copy, zero allocation.
//
// Binding layer extracts pointers from numpy (.ctypes.data) or torch
// (.data_ptr()) and wraps them here before passing into NewtonSolver.
// ---------------------------------------------------------------------------
template<typename T, typename IndexType = int32_t>
struct CSRView {
    const IndexType* indptr;   // row pointers, size = rows + 1
    const IndexType* indices;  // column indices, size = nnz
    const T*         data;     // non-zero values,  size = nnz

    IndexType rows, cols, nnz;
};




// ---------------------------------------------------------------------------
// NRConfig: Newton-Raphson solver hyperparameters.
// ---------------------------------------------------------------------------
struct NRConfig {
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
};


// ---------------------------------------------------------------------------
// BackendKind: which compute backend to use.
// ---------------------------------------------------------------------------
enum class BackendKind {
    CPU,
    CUDA,
};


// ---------------------------------------------------------------------------
// CpuAlgorithm: selects the CPU-side NR algorithm variant.
//
//   Optimized    — pre-computed JacobianMaps + edge-based fill + SparseLU
//                  symbolic reuse. Default; best CPU performance.
//
//   PyPowerLike  — no pre-analysis, dSbus_dV sparse-matrix Jacobian rebuilt
//                  every iteration, SparseLU one-shot (symbolic + numeric).
//                  Algorithmically equivalent to PyPower's newtonpf().
//                  Use to isolate the pure language (Python→C++) speedup.
// ---------------------------------------------------------------------------
enum class CpuAlgorithm {
    Optimized,
    PyPowerLike,
};


// ---------------------------------------------------------------------------
// PrecisionMode: selects floating-point precision for the NR solver.
//
//   FP32  — end-to-end FP32: public API takes complex<float>, all internal
//            buffers and CUDA kernels stay in FP32. CUDA + n_batch==1 only.
//
//   Mixed — FP64 public API, FP32 Jacobian/solve + FP64 voltage state.
//            This is the original CUDA mixed-precision path (~2× speedup).
//            CUDA + n_batch==1 only during the current precision-selection
//            refactor; multi-batch remains out of scope for now.
//
//   FP64  — end-to-end FP64: public API takes complex<double>, all internal
//            buffers and CUDA kernels stay in FP64. CUDA + n_batch==1 only.
//
// CPU backend: only FP64 is supported. FP32/Mixed combinations throw at
// solver construction time.
// ---------------------------------------------------------------------------
enum class PrecisionMode {
    FP32,
    Mixed,
    FP64,
};


// ---------------------------------------------------------------------------
// NewtonOptions: top-level solver configuration.
// ---------------------------------------------------------------------------
struct NewtonOptions {
    BackendKind         backend       = BackendKind::CPU;
    JacobianBuilderType jacobian      = JacobianBuilderType::EdgeBased;
    int32_t             n_batch       = 1;
    CpuAlgorithm        cpu_algorithm = CpuAlgorithm::Optimized;
    PrecisionMode       precision     = PrecisionMode::FP64;
};


// ---------------------------------------------------------------------------
// Ybus view aliases: precision-typed wrappers around CSRView.
// ---------------------------------------------------------------------------
using YbusViewF32 = CSRView<std::complex<float>>;
using YbusViewF64 = CSRView<std::complex<double>>;

// Backward-compatibility alias — same as YbusViewF64.
using YbusView = YbusViewF64;


// ---------------------------------------------------------------------------
// NRResult variants: output precision matches the solve precision.
// ---------------------------------------------------------------------------
struct NRResultF32 {
    std::vector<std::complex<float>> V;

    int32_t iterations     = 0;
    float   final_mismatch = 0.0f;
    bool    converged      = false;
};

struct NRResultF64 {
    std::vector<std::complex<double>> V;

    int32_t iterations     = 0;
    double  final_mismatch = 0.0;
    bool    converged      = false;
};

// Backward-compatibility alias — same as NRResultF64.
using NRResult = NRResultF64;
