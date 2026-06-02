#pragma once

#ifdef CUPF_WITH_CUDA

#include <cstdint>

struct CudaFp64Storage;
struct CudaFp32Storage;
struct CudaMixedStorage;
struct InitializeContext;
struct IterationContext;


struct CudaLinearSolveCustomFp64 {
    CudaLinearSolveCustomFp64();
    ~CudaLinearSolveCustomFp64();

    CudaLinearSolveCustomFp64(CudaLinearSolveCustomFp64&& other) noexcept;
    CudaLinearSolveCustomFp64& operator=(CudaLinearSolveCustomFp64&&) = delete;
    CudaLinearSolveCustomFp64(const CudaLinearSolveCustomFp64&) = delete;
    CudaLinearSolveCustomFp64& operator=(const CudaLinearSolveCustomFp64&) = delete;

    void initialize(CudaFp64Storage& buf, const InitializeContext& ctx);
    void prepare_rhs(CudaFp64Storage& buf, IterationContext& ctx);
    void factorize(CudaFp64Storage& buf, IterationContext& ctx);
    void solve(CudaFp64Storage& buf, IterationContext& ctx);

#ifdef CUPF_ENABLE_CUDA_GRAPH
    // Graph-capture setup: force the library's uniform-batch path (covers B==1 too) and bind the
    // caller's capture stream (cudaStream_t as void*). Call ONCE before cudaStreamBeginCapture — it
    // does the device allocations. Afterwards factorize()/solve() are pure kernel launches with no
    // host sync, so an outer stream capture (cuPF's whole-iteration graph) records them.
    void graph_prepare(CudaFp64Storage& buf, void* stream);
#endif

    void prepare_adjoint_explicit_transpose_cache(CudaFp64Storage& buf,
                                                  IterationContext& ctx,
                                                  double& factorization_time_ms);
    void solve_adjoint_explicit_transpose_host(const double* rhs,
                                               double* solution,
                                               int32_t batch_size,
                                               double& solve_time_ms);
    double* adjoint_rhs_data();
    double* adjoint_solution_data();
    void solve_adjoint_explicit_transpose_cached(double& solve_time_ms);
    bool supports_transpose_solve() const { return false; }
    bool has_adjoint_cache() const { return false; }
    bool has_adjoint_symbolic_analysis() const { return false; }

private:
    struct State;
    State* state_ = nullptr;
};

struct CudaLinearSolveCustomFp32 {
    CudaLinearSolveCustomFp32();
    ~CudaLinearSolveCustomFp32();

    CudaLinearSolveCustomFp32(CudaLinearSolveCustomFp32&& other) noexcept;
    CudaLinearSolveCustomFp32& operator=(CudaLinearSolveCustomFp32&&) = delete;
    CudaLinearSolveCustomFp32(const CudaLinearSolveCustomFp32&) = delete;
    CudaLinearSolveCustomFp32& operator=(const CudaLinearSolveCustomFp32&) = delete;

    void initialize(CudaFp32Storage& buf, const InitializeContext& ctx);
    void prepare_rhs(CudaFp32Storage& buf, IterationContext& ctx);
    void factorize(CudaFp32Storage& buf, IterationContext& ctx);
    void solve(CudaFp32Storage& buf, IterationContext& ctx);

#ifdef CUPF_ENABLE_CUDA_GRAPH
    void graph_prepare(CudaFp32Storage& buf, void* stream);
#endif

    void prepare_adjoint_explicit_transpose_cache(CudaFp32Storage& buf,
                                                  IterationContext& ctx,
                                                  double& factorization_time_ms);
    void solve_adjoint_explicit_transpose_host(const double* rhs,
                                               double* solution,
                                               int32_t batch_size,
                                               double& solve_time_ms);
    float* adjoint_rhs_data();
    float* adjoint_solution_data();
    void solve_adjoint_explicit_transpose_cached(double& solve_time_ms);
    bool supports_transpose_solve() const { return false; }
    bool has_adjoint_cache() const { return false; }
    bool has_adjoint_symbolic_analysis() const { return false; }

private:
    struct State;
    State* state_ = nullptr;
};


// Mixed-profile adapter: drives the custom solver from cuPF's CudaMixedStorage (FP32 Jacobian +
// step, FP64 residual). B==1 uses the solver's FP32 single-case path; B>1 uses the library's
// uniform-batch path. Adjoint/transpose is unsupported, and the Mixed pipeline never calls it.
struct CudaLinearSolveCustomMixed {
    CudaLinearSolveCustomMixed();
    ~CudaLinearSolveCustomMixed();

    CudaLinearSolveCustomMixed(CudaLinearSolveCustomMixed&& other) noexcept;
    CudaLinearSolveCustomMixed& operator=(CudaLinearSolveCustomMixed&&) = delete;
    CudaLinearSolveCustomMixed(const CudaLinearSolveCustomMixed&) = delete;
    CudaLinearSolveCustomMixed& operator=(const CudaLinearSolveCustomMixed&) = delete;

    void initialize(CudaMixedStorage& buf, const InitializeContext& ctx);
    void prepare_rhs(CudaMixedStorage& buf, IterationContext& ctx);
    void factorize(CudaMixedStorage& buf, IterationContext& ctx);
    void solve(CudaMixedStorage& buf, IterationContext& ctx);

#ifdef CUPF_ENABLE_CUDA_GRAPH
    // See CudaLinearSolveCustomFp64::graph_prepare. Call before cudaStreamBeginCapture.
    void graph_prepare(CudaMixedStorage& buf, void* stream);
#endif

    // Adjoint/transpose surface: unsupported (throws). Present only so the pipeline variant
    // visitors compile; signatures mirror the Mixed cuDSS adapter (FP32 step buffers).
    void prepare_adjoint_explicit_transpose_cache(CudaMixedStorage& buf,
                                                  IterationContext& ctx,
                                                  double& factorization_time_ms);
    void solve_adjoint_explicit_transpose_host(const double* rhs,
                                               double* solution,
                                               int32_t batch_size,
                                               double& solve_time_ms);
    float* adjoint_rhs_data();
    float* adjoint_solution_data();
    void solve_adjoint_explicit_transpose_cached(double& solve_time_ms);
    bool supports_transpose_solve() const { return false; }
    bool has_adjoint_cache() const { return false; }
    bool has_adjoint_symbolic_analysis() const { return false; }

private:
    struct State;
    State* state_ = nullptr;
};

#endif  // CUPF_WITH_CUDA
