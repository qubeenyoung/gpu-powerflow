#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/backend/i_backend.hpp"
#include "newton_solver/backend/cpu_backend.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

#ifdef WITH_CUDA
#include "newton_solver/backend/cuda_backend.hpp"
#endif

#ifdef WITH_NAIVE_CPU
#include "newton_solver/backend/naive_cpu_backend.hpp"
#endif

#include <string>
#include <vector>
#include <stdexcept>


// ---------------------------------------------------------------------------
// Helper: create the right backend based on NewtonOptions.
//
// Validation:
//   CPU  — only PrecisionMode::FP64 is supported.
//   All backends — current precision-selection refactor is single-case only,
//                  so n_batch must be exactly 1.
// ---------------------------------------------------------------------------
static std::unique_ptr<INewtonSolverBackend> make_backend(const NewtonOptions& opts)
{
    if (opts.n_batch != 1)
        throw std::invalid_argument(
            "NewtonSolver: n_batch != 1 is currently unsupported while "
            "precision selection is being refactored. Use single-case solve only.");

    switch (opts.backend) {
        case BackendKind::CPU:
            if (opts.precision != PrecisionMode::FP64)
                throw std::invalid_argument(
                    "NewtonSolver: CPU backend supports only PrecisionMode::FP64. "
                    "Use CUDA backend for FP32 or Mixed precision.");
#ifdef WITH_NAIVE_CPU
            if (opts.cpu_algorithm == CpuAlgorithm::PyPowerLike)
                return std::make_unique<NaiveCpuNewtonSolverBackend>();
#else
            if (opts.cpu_algorithm == CpuAlgorithm::PyPowerLike)
                throw std::runtime_error(
                    "NewtonSolver: PyPowerLike CPU backend requested but not compiled in. "
                    "Rebuild with -DBUILD_NAIVE_CPU=ON.");
#endif
            return std::make_unique<CpuNewtonSolverBackend>();

        case BackendKind::CUDA:
#ifdef WITH_CUDA
            return std::make_unique<CudaNewtonSolverBackend>(opts.n_batch, opts.precision);
#else
            throw std::runtime_error(
                "NewtonSolver: CUDA backend requested but not compiled in. "
                "Rebuild with -DWITH_CUDA=ON.");
#endif
    }
    return nullptr;
}

namespace {

std::string backend_name(BackendKind backend)
{
    switch (backend) {
        case BackendKind::CPU:
            return "cpu";
        case BackendKind::CUDA:
            return "cuda";
    }
    return "unknown";
}

void sync_backend_for_timing(INewtonSolverBackend& backend)
{
#ifdef CUPF_ENABLE_TIMING
    backend.synchronizeForTiming();
#else
    (void)backend;
#endif
}

}  // namespace


// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
NewtonSolver::NewtonSolver(const NewtonOptions& options)
    : options_(options)
    , jac_builder_(options.jacobian)
    , backend_(make_backend(options))
{}

NewtonSolver::~NewtonSolver() = default;


// ---------------------------------------------------------------------------
// analyze (FP64/Mixed): build Jacobian sparsity maps, initialize backend.
// ---------------------------------------------------------------------------
void NewtonSolver::analyze(
    const YbusViewF64& ybus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    if (options_.precision == PrecisionMode::FP32)
        throw std::invalid_argument(
            "NewtonSolver::analyze(YbusViewF64): solver was constructed with "
            "PrecisionMode::FP32. Use YbusViewF32 for FP32 mode.");

    LOG_INFO(std::string("NR analyze start backend=") + backend_name(options_.backend) +
             " buses=" + std::to_string(ybus.rows) +
             " pv=" + std::to_string(n_pv) +
             " pq=" + std::to_string(n_pq));

    newton_solver::utils::ScopedTimer total_timer("NR.analyze.total");

    // PyPowerLike: no pre-analysis. Just hand Ybus to the backend and stop.
    // Python's newtonpf() has no equivalent of this phase — Jacobian sparsity
    // is derived implicitly every iteration inside dSbus_dV().
    if (options_.backend == BackendKind::CPU &&
        options_.cpu_algorithm == CpuAlgorithm::PyPowerLike)
    {
        backend_->analyze(ybus, JacobianMaps{}, JacobianStructure{}, ybus.rows);
        sync_backend_for_timing(*backend_);
        analyzed_ = true;
        LOG_INFO("NR analyze done (PyPowerLike — no JacobianBuilder phase)");
        return;
    }

    {
        newton_solver::utils::ScopedTimer timer("NR.analyze.jacobianBuilder");
        auto result = jac_builder_.analyze(ybus, pv, n_pv, pq, n_pq);
        jac_maps_   = std::move(result.maps);
        J_          = std::move(result.J);
    }

    {
        newton_solver::utils::ScopedTimer timer("NR.analyze.backend");
        backend_->analyze(ybus, jac_maps_, J_, ybus.rows);
        sync_backend_for_timing(*backend_);
    }
    analyzed_ = true;

    LOG_INFO(std::string("NR analyze done backend=") + backend_name(options_.backend) +
             " dimF=" + std::to_string(J_.dim) +
             " j_nnz=" + std::to_string(J_.nnz));
}


// ---------------------------------------------------------------------------
// solve (FP64/Mixed): Newton-Raphson loop for a single power flow case.
// ---------------------------------------------------------------------------
void NewtonSolver::solve(
    const YbusViewF64&          ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    NRResultF64&                result)
{
    if (options_.precision == PrecisionMode::FP32)
        throw std::invalid_argument(
            "NewtonSolver::solve(YbusViewF64): solver was constructed with "
            "PrecisionMode::FP32. Use FP32 overloads for FP32 mode.");

    const int32_t dimF = n_pv + 2 * n_pq;

    std::vector<double> F(dimF);
    std::vector<double> dx(dimF);

    LOG_INFO(std::string("NR solve start backend=") + backend_name(options_.backend) +
             " buses=" + std::to_string(ybus.rows) +
             " pv=" + std::to_string(n_pv) +
             " pq=" + std::to_string(n_pq) +
             " max_iter=" + std::to_string(config.max_iter));

    newton_solver::utils::ScopedTimer solve_total_timer("NR.solve.total");
    {
        newton_solver::utils::ScopedTimer timer("NR.solve.initialize");
        backend_->initialize(ybus, sbus, V0);
        sync_backend_for_timing(*backend_);
    }

    double  normF     = 0.0;
    bool    converged = false;
    int32_t iter      = 0;

    while (iter < config.max_iter) {
        newton_solver::utils::ScopedTimer iter_timer("NR.iteration.total");

        {
            newton_solver::utils::ScopedTimer timer("NR.computeMismatch");
            backend_->computeMismatch(pv, n_pv, pq, n_pq, F.data(), normF);
            sync_backend_for_timing(*backend_);
        }

        if (normF < config.tolerance) {
            converged = true;
            break;
        }

        {
            newton_solver::utils::ScopedTimer timer("NR.updateJacobian");
            backend_->updateJacobian();
            sync_backend_for_timing(*backend_);
        }
        {
            newton_solver::utils::ScopedTimer timer("NR.solveLinearSystem");
            backend_->solveLinearSystem(F.data(), dx.data());
            sync_backend_for_timing(*backend_);
        }
        {
            newton_solver::utils::ScopedTimer timer("NR.updateVoltage");
            backend_->updateVoltage(dx.data(), pv, n_pv, pq, n_pq);
            sync_backend_for_timing(*backend_);
        }

        ++iter;
    }

    result.V.resize(ybus.rows);
    {
        newton_solver::utils::ScopedTimer timer("NR.solve.downloadV");
        backend_->downloadV(result.V.data(), ybus.rows);
        sync_backend_for_timing(*backend_);
    }

    result.converged      = converged;
    result.iterations     = iter;
    result.final_mismatch = normF;

    LOG_INFO(std::string("NR solve done backend=") + backend_name(options_.backend) +
             " iterations=" + std::to_string(iter) +
             " converged=" + (converged ? "true" : "false") +
             " mismatch=" + std::to_string(normF));
}


// ---------------------------------------------------------------------------
// solve_batch: run NR for n_batch independent cases in parallel.
//
// If the backend supports native batch (CUDA UBATCH), all cases are solved
// simultaneously using SpMM + cuDSS UBATCH.
// Otherwise, falls back to sequential single-case solves.
//
// NR loop (batch path):
//   1. initialize_batch(V0_batch, Sbus_batch)
//   2. per iteration:
//      a. computeMismatch_batch → F_batch, normF_batch
//      b. check per-case convergence
//      c. updateJacobian_batch
//      d. solveLinearSystem_batch(F_batch) → dx_batch (in d_x_f_batch on GPU)
//      e. updateVoltage_batch (reads d_x_f_batch directly)
//   3. downloadV_batch → results[b].V
// ---------------------------------------------------------------------------
void NewtonSolver::solve_batch(
    const YbusView&             ybus,
    const std::complex<double>* sbus_batch,
    const std::complex<double>* V0_batch,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    int32_t                     n_batch,
    const NRConfig&             config,
    NRResult*                   results)
{
    (void)ybus;
    (void)sbus_batch;
    (void)V0_batch;
    (void)pv;
    (void)n_pv;
    (void)pq;
    (void)n_pq;
    (void)n_batch;
    (void)config;
    (void)results;

    throw std::invalid_argument(
        "NewtonSolver::solve_batch(): multi-batch is currently out of scope for "
        "the precision-selection refactor. Use n_batch == 1.");
}


// ---------------------------------------------------------------------------
// analyze (FP32): build Jacobian sparsity maps, initialize FP32 backend.
//
// JacobianBuilder only reads structure (indptr, indices, rows, nnz) — not
// the complex values. A structural-only FP64 view is constructed with a null
// data pointer so the builder can run without a data copy.
//
// Only valid when PrecisionMode::FP32 is active (CUDA, n_batch == 1).
// ---------------------------------------------------------------------------
void NewtonSolver::analyze(
    const YbusViewF32& ybus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    if (options_.precision != PrecisionMode::FP32)
        throw std::invalid_argument(
            "NewtonSolver::analyze(YbusViewF32): solver was not constructed with "
            "PrecisionMode::FP32. Use YbusViewF64 for FP64/Mixed modes.");

    LOG_INFO(std::string("NR analyze (FP32) start backend=") + backend_name(options_.backend) +
             " buses=" + std::to_string(ybus.rows) +
             " pv=" + std::to_string(n_pv) +
             " pq=" + std::to_string(n_pq));

    newton_solver::utils::ScopedTimer total_timer("NR.analyze.total");

    // JacobianBuilder only uses indptr, indices, rows, nnz — not data values.
    // Build a structural-only F64 view so the existing builder can run unchanged.
    YbusViewF64 ybus_struct { ybus.indptr, ybus.indices, nullptr, ybus.rows, ybus.cols, ybus.nnz };

    {
        newton_solver::utils::ScopedTimer timer("NR.analyze.jacobianBuilder");
        auto result = jac_builder_.analyze(ybus_struct, pv, n_pv, pq, n_pq);
        jac_maps_   = std::move(result.maps);
        J_          = std::move(result.J);
    }

    {
        newton_solver::utils::ScopedTimer timer("NR.analyze.backend");
        backend_->analyze_f32(ybus, jac_maps_, J_, ybus.rows);
        sync_backend_for_timing(*backend_);
    }
    analyzed_ = true;

    LOG_INFO(std::string("NR analyze (FP32) done backend=") + backend_name(options_.backend) +
             " dimF=" + std::to_string(J_.dim) +
             " j_nnz=" + std::to_string(J_.nnz));
}


// ---------------------------------------------------------------------------
// solve (FP32): Newton-Raphson loop with end-to-end FP32 pipeline.
//
// NR iteration:
//   1. computeMismatch_f32 → F (float), normF (float)
//   2. if normF < tol: converged, break
//   3. updateJacobian (dispatches internally to FP32 path)
//   4. solveLinearSystem_f32(F) → dx (float, GPU-resident for CUDA)
//   5. updateVoltage_f32(dx)
//
// Only valid when PrecisionMode::FP32 is active (CUDA, n_batch == 1).
// ---------------------------------------------------------------------------
void NewtonSolver::solve(
    const YbusViewF32&         ybus,
    const std::complex<float>* sbus,
    const std::complex<float>* V0,
    const int32_t*             pv, int32_t n_pv,
    const int32_t*             pq, int32_t n_pq,
    const NRConfig&            config,
    NRResultF32&               result)
{
    if (options_.precision != PrecisionMode::FP32)
        throw std::invalid_argument(
            "NewtonSolver::solve(YbusViewF32): solver was not constructed with "
            "PrecisionMode::FP32. Use YbusViewF64 for FP64/Mixed modes.");

    const int32_t dimF = n_pv + 2 * n_pq;

    std::vector<float> F(dimF);
    std::vector<float> dx(dimF);

    LOG_INFO(std::string("NR solve (FP32) start backend=") + backend_name(options_.backend) +
             " buses=" + std::to_string(ybus.rows) +
             " pv=" + std::to_string(n_pv) +
             " pq=" + std::to_string(n_pq) +
             " max_iter=" + std::to_string(config.max_iter));

    newton_solver::utils::ScopedTimer solve_total_timer("NR.solve.total");
    {
        newton_solver::utils::ScopedTimer timer("NR.solve.initialize");
        backend_->initialize_f32(ybus, sbus, V0);
        sync_backend_for_timing(*backend_);
    }

    float   normF     = 0.0f;
    bool    converged = false;
    int32_t iter      = 0;
    const float tol_f = static_cast<float>(config.tolerance);

    while (iter < config.max_iter) {
        newton_solver::utils::ScopedTimer iter_timer("NR.iteration.total");

        {
            newton_solver::utils::ScopedTimer timer("NR.computeMismatch");
            backend_->computeMismatch_f32(pv, n_pv, pq, n_pq, F.data(), normF);
            sync_backend_for_timing(*backend_);
        }

        if (normF < tol_f) {
            converged = true;
            break;
        }

        {
            newton_solver::utils::ScopedTimer timer("NR.updateJacobian");
            backend_->updateJacobian();
            sync_backend_for_timing(*backend_);
        }
        {
            newton_solver::utils::ScopedTimer timer("NR.solveLinearSystem");
            backend_->solveLinearSystem_f32(F.data(), dx.data());
            sync_backend_for_timing(*backend_);
        }
        {
            newton_solver::utils::ScopedTimer timer("NR.updateVoltage");
            backend_->updateVoltage_f32(dx.data(), pv, n_pv, pq, n_pq);
            sync_backend_for_timing(*backend_);
        }

        ++iter;
    }

    result.V.resize(ybus.rows);
    {
        newton_solver::utils::ScopedTimer timer("NR.solve.downloadV");
        backend_->downloadV_f32(result.V.data(), ybus.rows);
        sync_backend_for_timing(*backend_);
    }

    result.converged      = converged;
    result.iterations     = iter;
    result.final_mismatch = normF;

    LOG_INFO(std::string("NR solve (FP32) done backend=") + backend_name(options_.backend) +
             " iterations=" + std::to_string(iter) +
             " converged=" + (converged ? "true" : "false") +
             " mismatch=" + std::to_string(normF));
}
