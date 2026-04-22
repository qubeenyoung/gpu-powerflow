#pragma once

#include <variant>

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/ops/mismatch/cpu_mismatch.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/voltage_update/cpu_voltage_update.hpp"

#ifdef CUPF_WITH_CUDA
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "newton_solver/ops/mismatch/cuda_mismatch.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian_gpu.hpp"
#include "newton_solver/ops/linear_solve/cuda_cudss.hpp"
#include "newton_solver/ops/voltage_update/cuda_voltage_update.hpp"
#endif


// ---------------------------------------------------------------------------
// CpuFp64Pipeline
// ---------------------------------------------------------------------------
struct CpuFp64Pipeline {
    CpuFp64Buffers    buf;
    CpuLinearSolveKLU linear_solve;

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
    void jacobian(IterationContext& ctx)      { CpuJacobianOpF64{}.run(buf, ctx); }
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
    CudaFp64Buffers                                   buf;
    CudaLinearSolveCuDSS<double, CudaFp64Buffers>     linear_solve;

    explicit CudaFp64Pipeline(CuDSSOptions opts = {}) : linear_solve(opts) {}

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

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaFp64Buffers>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaFp64Buffers>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaFp64Buffers>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<double>{}.run(buf, ctx); }
    void prepare_rhs(IterationContext& ctx)   { linear_solve.prepare_rhs(buf, ctx); }
    void factorize(IterationContext& ctx)     { linear_solve.factorize(buf, ctx); }
    void solve(IterationContext& ctx)         { linear_solve.solve(buf, ctx); }
    void voltage_update(IterationContext& ctx){ CudaVoltageUpdateOp<double>{}.run(buf, ctx); }

    static constexpr bool batch_supported = false;
};


// ---------------------------------------------------------------------------
// CudaMixedPipeline
// ---------------------------------------------------------------------------
struct CudaMixedPipeline {
    CudaMixedBuffers                                   buf;
    CudaLinearSolveCuDSS<float, CudaMixedBuffers>      linear_solve;

    explicit CudaMixedPipeline(CuDSSOptions opts = {}) : linear_solve(opts) {}

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

    void ibus(IterationContext& ctx)          { CudaIbusOp<CudaMixedBuffers>{}.run(buf, ctx); }
    void mismatch(IterationContext& ctx)      { CudaMismatchOp<CudaMixedBuffers>{}.run(buf, ctx); }
    void mismatch_norm(IterationContext& ctx) { CudaMismatchNormOp<CudaMixedBuffers>{}.run(buf, ctx); }
    void jacobian(IterationContext& ctx)      { CudaJacobianOp<float>{}.run(buf, ctx); }
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
        , CudaMixedPipeline
#endif
    > v;
};
