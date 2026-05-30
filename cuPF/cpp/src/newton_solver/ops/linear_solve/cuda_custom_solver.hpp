#pragma once

#ifdef CUPF_WITH_CUDA

#include <cstdint>

struct CudaFp64Buffers;
struct InitializeContext;
struct IterationContext;


struct CudaLinearSolveCustomFp64 {
    CudaLinearSolveCustomFp64();
    ~CudaLinearSolveCustomFp64();

    CudaLinearSolveCustomFp64(CudaLinearSolveCustomFp64&& other) noexcept;
    CudaLinearSolveCustomFp64& operator=(CudaLinearSolveCustomFp64&&) = delete;
    CudaLinearSolveCustomFp64(const CudaLinearSolveCustomFp64&) = delete;
    CudaLinearSolveCustomFp64& operator=(const CudaLinearSolveCustomFp64&) = delete;

    void initialize(CudaFp64Buffers& buf, const InitializeContext& ctx);
    void prepare_rhs(CudaFp64Buffers& buf, IterationContext& ctx);
    void factorize(CudaFp64Buffers& buf, IterationContext& ctx);
    void solve(CudaFp64Buffers& buf, IterationContext& ctx);

    void prepare_adjoint_explicit_transpose_cache(CudaFp64Buffers& buf,
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

#endif  // CUPF_WITH_CUDA
