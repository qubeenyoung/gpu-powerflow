#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// CSRView: 희소 행렬에 대한 non-owning view.
//
// 메모리를 소유하지 않으며 할당도 하지 않는다.
// 호출자가 배열 생명주기를 관리한다.
// Python 바인딩 레이어에서 numpy 배열의 raw pointer를 이 구조체로 감싼다.
// ---------------------------------------------------------------------------
template<typename T, typename IndexType = int32_t>
struct CSRView {
    const IndexType* indptr;   // 행 포인터,     크기 = rows + 1
    const IndexType* indices;  // 열 인덱스,     크기 = nnz
    const T*         data;     // 비영 원소 값,  크기 = nnz

    IndexType rows, cols, nnz;
};


// ---------------------------------------------------------------------------
// Ybus view 타입.
// ---------------------------------------------------------------------------
using YbusView = CSRView<std::complex<double>>;


// ---------------------------------------------------------------------------
// NRConfig: Newton-Raphson 수렴 조건 설정.
// ---------------------------------------------------------------------------
struct NRConfig {
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
};


enum class AdjointCacheMode {
    None,
    FinalStateFactorization,
    ReuseLastNewtonFactorizationIfExact,
};

struct SolveOptions {
    bool prepare_adjoint_cache = false;
    AdjointCacheMode adjoint_cache_mode = AdjointCacheMode::FinalStateFactorization;
    bool allow_explicit_transpose_fallback = false;
};


// ---------------------------------------------------------------------------
// BackendKind: 연산 백엔드 선택.
// ---------------------------------------------------------------------------
enum class BackendKind {
    CPU,
    CUDA,
};


// ---------------------------------------------------------------------------
// ComputePolicy: 내부 계산 정밀도 정책.
//
//   FP64  — 내부 전 단계 FP64 (CPU 또는 CUDA).
//   FP32  — GPU 내부 numeric buffers/operators are FP32 (CUDA 전용).
//   Mixed — public I/O는 FP64, 내부 Jacobian/solve는 FP32 (CUDA 전용).
//           고정 프로파일이며, stage별 자유 조합이 아니다.
//
// Mixed 프로파일의 내부 구성:
//   Ybus/V state   — FP64
//   mismatch       — FP64
//   jacobian       — FP32
//   linear solve   — FP32 (cuDSS FP32)
//   voltage update — FP64 상태 기준
// ---------------------------------------------------------------------------
enum class ComputePolicy {
    FP64,
    FP32,
    Mixed,
};


enum class CudaLinearSolverKind {
    CuDSS,
    Custom,
};

enum class CpuLinearSolverKind {
    KLU,
    UMFPACK,
};


enum class CudaJacobianKind {
    Edge,
    EdgeAtomic,
    VertexWarp,
};


// ---------------------------------------------------------------------------
// CuDSSAlgorithm / CuDSSOptions: CUDA direct solver 런타임 설정.
//
// cuDSS 헤더를 public API에 노출하지 않기 위해 자체 enum으로 표현하고,
// CUDA linear solver 내부에서 cudssAlgType_t로 변환한다.
// ---------------------------------------------------------------------------
enum class CuDSSAlgorithm {
    Default,
    Alg1,
    Alg2,
    Alg3,
    Alg4,
    Alg5,
};

struct CuDSSOptions {
    bool use_matching = false;
    CuDSSAlgorithm matching_alg = CuDSSAlgorithm::Default;

    bool auto_pivot_epsilon = true;
    double pivot_epsilon = 0.0;
};


// ---------------------------------------------------------------------------
// NewtonOptions: 생성자에 전달하는 solver 설정.
//
// 사용자는 backend, compute policy와 CUDA direct solver 설정을 선택한다.
// ---------------------------------------------------------------------------
struct NewtonOptions {
    BackendKind          backend = BackendKind::CPU;
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
    // Default GPU path (when backend = CUDA): custom direct solver on the Mixed profile — FP32
    // Jacobian/step, FP64 state — with edge Jacobian assembly. The custom solver reads the FP32
    // Jacobian directly and factors in FP32 (CUPF_CUSTOM_PRECISION overrides). Any field below can
    // still be set explicitly; the CPU backend ignores `compute` (always CPU FP64).
    ComputePolicy        compute = ComputePolicy::Mixed;
    CudaLinearSolverKind cuda_linear_solver = CudaLinearSolverKind::Custom;
#else
    ComputePolicy        compute = ComputePolicy::FP64;
    CudaLinearSolverKind cuda_linear_solver = CudaLinearSolverKind::CuDSS;
#endif
    CpuLinearSolverKind  cpu_linear_solver = CpuLinearSolverKind::KLU;
    CudaJacobianKind     cuda_jacobian = CudaJacobianKind::Edge;
    CuDSSOptions         cudss = {};

    // Capture the whole Newton iteration (ibus -> mismatch -> jacobian -> linear solve -> voltage
    // update) into a single CUDA graph and replay it per step, collapsing the per-iteration kernel
    // launches into one cudaGraphLaunch. Only valid with backend = CUDA and
    // cuda_linear_solver = Custom (cuDSS's cudssExecute is not stream-capturable), and only when
    // the library is built with CUPF_ENABLE_CUDA_GRAPH (which builds custom_linear_solver in its
    // external/capturable mode). Ignored / rejected otherwise.
    bool                 use_cuda_graph = false;
};


// ---------------------------------------------------------------------------
// NRResult: solve() 결과. public I/O는 항상 FP64.
// ---------------------------------------------------------------------------
struct NRResult {
    std::vector<std::complex<double>> V;

    int32_t iterations     = 0;
    double  final_mismatch = 0.0;
    bool    converged      = false;
};


// ---------------------------------------------------------------------------
// NRBatchResult: batch solve() 결과.
//
// 단일 케이스 결과와 동일한 의미를 batch 차원으로 확장한다.
// V는 batch-major contiguous layout [batch_size * n_bus]를 사용한다.
// public I/O는 항상 FP64다.
// ---------------------------------------------------------------------------
struct NRBatchResult {
    std::vector<std::complex<double>> V;

    int32_t n_bus      = 0;
    int32_t batch_size = 0;

    std::vector<int32_t> iterations;
    std::vector<double>  final_mismatch;
    std::vector<uint8_t> converged;
};


// ---------------------------------------------------------------------------
// AdjointOptions / AdjointResult: implicit PF backward API.
//
// The residual convention in cuPF is:
//
//   F(x, Sbus) = S_calc(x) - Sbus
//
// with x = [Va[pv], Va[pq], Vm[pq]] and
// F = [Pmis[pv], Pmis[pq], Qmis[pq]].
//
// A demand/load increase decreases net injection Sbus. Therefore, after
// solving J^T lambda = dL/dx, load gradients are -lambda projected onto the
// corresponding P/Q residual rows.
// ---------------------------------------------------------------------------
struct AdjointOptions {
    bool reuse_forward_factorization = false;
    bool allow_refactorize = true;
    bool require_cached_factorization = false;
    bool allow_refactorize_for_backward = true;
    bool allow_inexact_last_newton_factorization = false;
    bool use_transpose_solve = true;
    bool allow_explicit_transpose_fallback = false;
    bool compute_load_gradients = true;
    bool check_residual = true;
};

struct AdjointResult {
    std::vector<double> lambda;
    std::vector<double> grad_load_p;
    std::vector<double> grad_load_q;

    int32_t n_bus = 0;
    int32_t batch_size = 0;
    int32_t dimF = 0;

    bool success = false;
    bool used_adjoint_cache = false;
    bool adjoint_cache_matches_final_state = false;
    bool reused_forward_factorization = false;
    bool reused_final_state_factorization = false;
    bool refactorized_for_backward = false;
    bool used_explicit_transpose = false;
    bool used_python_scipy = false;
    bool includes_host_device_transfer = false;
    bool zero_copy = false;
    bool torch_extension_zero_copy = false;
    bool raw_pointer_api_used = false;
    bool current_stream_integrated = false;
    bool jt_symbolic_analyzed_at_initialize = false;
    bool jt_values_transposed_on_device = false;
    bool jt_factorized_during_forward_cache = false;
    bool jt_refactorized_during_backward = false;
    bool host_roundtrip_for_jt_transpose = false;

    double jt_residual_norm = 0.0;
    double solve_time_ms = 0.0;
    double transpose_solve_time_ms = 0.0;
    double factorization_time_ms = 0.0;
    double total_time_ms = 0.0;

    std::string backend;
    std::string transpose_solve_backend;
    std::string sign_convention =
        "F=S_calc-S_spec; load increase decreases Sbus; grad_load=-lambda";
};
