#pragma once

#include <variant>

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/ops/mismatch/cpu_mismatch.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/linear_solve/cpu_umfpack.hpp"
#include "newton_solver/ops/voltage_update/cpu_voltage_update.hpp"

#ifdef CUPF_WITH_CUDA
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "newton_solver/ops/mismatch/cuda_mismatch.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian_gpu.hpp"
#include "newton_solver/ops/linear_solve/cuda_cudss.hpp"
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
#include "newton_solver/ops/linear_solve/cuda_custom_solver.hpp"
#endif
#include "newton_solver/ops/voltage_update/cuda_voltage_update.hpp"
#endif


// ---------------------------------------------------------------------------
// CpuFp64Pipeline
// ---------------------------------------------------------------------------
struct CpuLinearSolveAny {
    std::variant<CpuLinearSolveKLU, CpuLinearSolveUMFPACK> v;

    explicit CpuLinearSolveAny(CpuLinearSolverKind kind = CpuLinearSolverKind::KLU)
    {
        if (kind == CpuLinearSolverKind::UMFPACK) {
            v.template emplace<CpuLinearSolveUMFPACK>();
        } else {
            v.template emplace<CpuLinearSolveKLU>();
        }
    }

    void initialize(CpuFp64Storage& buf, const InitializeContext& ctx) {
        std::visit([&](auto& solver) { solver.initialize(buf, ctx); }, v);
    }
    void prepare_rhs(CpuFp64Storage& buf, IterationContext& ctx) {
        std::visit([&](auto& solver) { solver.prepare_rhs(buf, ctx); }, v);
    }
    void factorize(CpuFp64Storage& buf, IterationContext& ctx) {
        std::visit([&](auto& solver) { solver.factorize(buf, ctx); }, v);
    }
    void solve(CpuFp64Storage& buf, IterationContext& ctx) {
        std::visit([&](auto& solver) { solver.solve(buf, ctx); }, v);
    }
    void solve_transpose(const double* rhs, double* solution, int32_t dim, int32_t nrhs = 1) {
        std::visit([&](auto& solver) { solver.solve_transpose(rhs, solution, dim, nrhs); }, v);
    }
    const char* backend_name() const {
        return std::holds_alternative<CpuLinearSolveUMFPACK>(v) ? "cpu_umfpack" : "cpu_klu";
    }
    const char* transpose_backend_name() const {
        return std::holds_alternative<CpuLinearSolveUMFPACK>(v)
            ? "cpu_umfpack_transpose_cached_factorization"
            : "cpu_klu_tsolve_cached_factorization";
    }
};


struct CpuFp64Pipeline {
    CpuFp64Storage    buf;
    CpuJacobianKind   jacobian_kind = CpuJacobianKind::Native;
    CpuLinearSolveAny linear_solve;
    AdjointCache      adjoint_cache;

    explicit CpuFp64Pipeline(CpuJacobianKind jacobian = CpuJacobianKind::Native,
                             CpuLinearSolverKind solver = CpuLinearSolverKind::KLU)
        : jacobian_kind(jacobian)
        , linear_solve(solver)
    {}

    void initialize(const InitializeContext& ctx) {
        buf.prepare(ctx);
        linear_solve.initialize(buf, ctx);
    }

    void upload(const SolveContext& ctx) {
        buf.upload(ctx);
    }

    void download_batch(NRBatchResult& result) const {
        NRResult single;
        buf.download(single);
        result.V = std::move(single.V);
    }

    void ibus(IterationContext& ctx)          { CpuIbusOp{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CpuMismatchOp{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CpuMismatchNormOp{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx) {
        if (jacobian_kind == CpuJacobianKind::Pandapower) {
            CpuPandapowerJacobianOpF64{}.run(buf, ctx);
        } else {
            CpuJacobianOpF64{}.run(buf, ctx);
        }
    }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CpuVoltageUpdateOp{}.run(buf, ctx); }

    static constexpr bool batch_supported = false;
};


#ifdef CUPF_WITH_CUDA

// ---------------------------------------------------------------------------
// CudaFp64Pipeline
// ---------------------------------------------------------------------------
struct CudaFp64Pipeline {
    CudaFp64Storage                                   buf;
    CudaJacobianKind                                  jacobian_kind = CudaJacobianKind::Edge;
    CudaLinearSolveCuDSS<double, CudaFp64Storage>     linear_solve;
    AdjointCache                                      adjoint_cache;

    explicit CudaFp64Pipeline(CuDSSOptions opts = {},
                              CudaJacobianKind jacobian = CudaJacobianKind::Edge)
        : jacobian_kind(jacobian)
        , linear_solve(opts)
    {}

    void initialize(const InitializeContext& ctx) {
        buf.prepare(ctx);
        linear_solve.initialize(buf, ctx);
    }

    void upload(const SolveContext& ctx) {
        buf.upload(ctx);
    }

    void download_batch(NRBatchResult& result) const {
        NRResult single;
        buf.download(single);
        result.V = std::move(single.V);
    }

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaFp64Storage>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaFp64Storage>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaFp64Storage>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<double>{jacobian_kind}.run(buf, ctx); }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CudaVoltageUpdateOp<double>{}.run(buf, ctx); }

    static constexpr bool batch_supported = false;
};


#ifdef CUPF_ENABLE_CUSTOM_SOLVER
// ---------------------------------------------------------------------------
// CudaFp64CustomPipeline
// ---------------------------------------------------------------------------
struct CudaFp64CustomPipeline {
    CudaFp64Storage              buf;
    CudaJacobianKind             jacobian_kind = CudaJacobianKind::Edge;
    CudaLinearSolveCustomFp64    linear_solve;
    AdjointCache                 adjoint_cache;

    explicit CudaFp64CustomPipeline(CudaJacobianKind jacobian = CudaJacobianKind::Edge)
        : jacobian_kind(jacobian)
    {}

    void initialize(const InitializeContext& ctx) {
        buf.prepare(ctx);
        linear_solve.initialize(buf, ctx);
    }

    void upload(const SolveContext& ctx) {
        buf.upload(ctx);
    }

    void download_batch(NRBatchResult& result) const {
        NRResult single;
        buf.download(single);
        result.V = std::move(single.V);
    }

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaFp64Storage>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaFp64Storage>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaFp64Storage>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<double>{jacobian_kind}.run(buf, ctx); }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CudaVoltageUpdateOp<double>{}.run(buf, ctx); }

    static constexpr bool batch_supported = false;
};
#endif


// ---------------------------------------------------------------------------
// CudaFp32Pipeline
// ---------------------------------------------------------------------------
struct CudaFp32Pipeline {
    CudaFp32Storage                                   buf;
    CudaJacobianKind                                  jacobian_kind = CudaJacobianKind::Edge;
    CudaLinearSolveCuDSS<float, CudaFp32Storage>      linear_solve;
    AdjointCache                                      adjoint_cache;

    explicit CudaFp32Pipeline(CuDSSOptions opts = {},
                              CudaJacobianKind jacobian = CudaJacobianKind::Edge)
        : jacobian_kind(jacobian)
        , linear_solve(opts)
    {}

    void initialize(const InitializeContext& ctx) {
        buf.prepare(ctx);
        linear_solve.initialize(buf, ctx);
    }

    void upload(const SolveContext& ctx) {
        buf.upload(ctx);
    }

    void download_batch(NRBatchResult& result) const {
        buf.download_batch(result);
    }

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaFp32Storage>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaFp32Storage>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaFp32Storage>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<float>{jacobian_kind}.run(buf, ctx); }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CudaVoltageUpdateOp<float>{}.run(buf, ctx); }

    static constexpr bool batch_supported = true;
};


// ---------------------------------------------------------------------------
// CudaMixedPipeline
// ---------------------------------------------------------------------------
struct CudaMixedPipeline {
    CudaMixedStorage                                   buf;
    CudaJacobianKind                                   jacobian_kind = CudaJacobianKind::Edge;
    CudaLinearSolveCuDSS<float, CudaMixedStorage>      linear_solve;
    AdjointCache                                       adjoint_cache;

    explicit CudaMixedPipeline(CuDSSOptions opts = {},
                               CudaJacobianKind jacobian = CudaJacobianKind::Edge)
        : jacobian_kind(jacobian)
        , linear_solve(opts)
    {}

    void initialize(const InitializeContext& ctx) {
        buf.prepare(ctx);
        linear_solve.initialize(buf, ctx);
    }

    void upload(const SolveContext& ctx) {
        buf.upload(ctx);
    }

    void download_batch(NRBatchResult& result) const {
        buf.download_batch(result);
    }

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaMixedStorage>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaMixedStorage>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaMixedStorage>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<float>{jacobian_kind}.run(buf, ctx); }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CudaVoltageUpdateOp<float>{}.run(buf, ctx); }

    static constexpr bool batch_supported = true;
};

#endif  // CUPF_WITH_CUDA


// ---------------------------------------------------------------------------
// SolverPipeline: variant over all profiles.
// Forward-declared in newton_solver.hpp; defined here.
// ---------------------------------------------------------------------------
struct SolverPipeline {
    std::variant<
        CpuFp64Pipeline
#ifdef CUPF_WITH_CUDA
        , CudaFp64Pipeline
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
        , CudaFp64CustomPipeline
#endif
        , CudaFp32Pipeline
        , CudaMixedPipeline
#endif
    > v;
};
