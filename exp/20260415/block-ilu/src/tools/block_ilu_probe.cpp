#include "dump_case_loader.hpp"
#include "assembly/reduced_jacobian_assembler.hpp"
#include "linear/cusparse_ilu0_block.hpp"
#include "model/reduced_jacobian.hpp"
#include "solver/powerflow_block_ilu_solver.hpp"
#include "utils/cuda_utils.hpp"

#include <Eigen/KLUSupport>
#include <Eigen/Sparse>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BLOCK_ILU_DEFAULT_DATASET_ROOT
#define BLOCK_ILU_DEFAULT_DATASET_ROOT "/workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps"
#endif

namespace {

struct CliOptions {
    std::filesystem::path dataset_root = BLOCK_ILU_DEFAULT_DATASET_ROOT;
    std::string case_name = "case_ACTIVSg200";
    bool pattern_only = false;
    bool factor_only = false;
    double nonlinear_tolerance = 1e-8;
    double linear_tolerance = 1e-6;
    int32_t max_outer_iterations = 10;
    int32_t max_inner_iterations = 500;
    exp_20260415::block_ilu::SchurLinearSolverKind linear_solver_kind =
        exp_20260415::block_ilu::SchurLinearSolverKind::Bicgstab;
    int32_t gmres_restart = 30;
    int32_t residual_check_interval = 5;
    exp_20260415::block_ilu::J11ReorderMode j11_reorder_mode =
        exp_20260415::block_ilu::J11ReorderMode::None;
    exp_20260415::block_ilu::J11SolverKind j11_solver_kind =
        exp_20260415::block_ilu::J11SolverKind::Ilu0;
    int32_t j11_dense_block_size = 32;
    exp_20260415::block_ilu::J11DenseBackend j11_dense_backend =
        exp_20260415::block_ilu::J11DenseBackend::CublasGetrf;
    bool j11_dense_backend_explicit = false;
    exp_20260415::block_ilu::J11PartitionMode j11_partition_mode =
        exp_20260415::block_ilu::J11PartitionMode::Bfs;
    exp_20260415::block_ilu::InnerPrecision inner_precision =
        exp_20260415::block_ilu::InnerPrecision::Fp64;
    bool collect_timing_breakdown = true;
    bool continue_on_linear_failure = false;
    bool compare_direction = false;
    std::filesystem::path dump_dx_log;
    bool enable_line_search = false;
    exp_20260415::block_ilu::SchurPreconditionerKind schur_preconditioner_kind =
        exp_20260415::block_ilu::SchurPreconditionerKind::None;
    int32_t line_search_max_trials = 8;
    double line_search_reduction = 0.5;
};

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0
              << " [--dataset-root PATH] [--case NAME]"
              << " [--pattern-only] [--factor-only]"
              << " [--max-outer INT] [--inner-max-iter INT]"
              << " [--linear-tol FLOAT] [--nonlinear-tol FLOAT]"
              << " [--linear-solver bicgstab|gmres]"
              << " [--gmres-restart INT] [--residual-check-interval INT]"
              << " [--inner-precision fp64|fp32]"
              << " [--j11-solver ilu0|partition-dense-lu|exact-klu]"
              << " [--j11-dense-backend cublas|tc|cusolver]"
              << " [--j11-partition bfs|metis]"
              << " [--j11-reorder none|amd|colamd|rcm]"
              << " [--j11-block-size INT]"
              << " [--continue-on-linear-failure] [--no-timing-breakdown]"
              << " [--compare-direction] [--dump-dx-log PATH]"
              << " [--schur-preconditioner none|j22-ilu0|j22-block-dense-lu]"
              << " [--line-search] [--line-search-max-trials INT]"
              << " [--line-search-reduction FLOAT]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            options.dataset_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_name = argv[++i];
        } else if (arg == "--pattern-only") {
            options.pattern_only = true;
        } else if (arg == "--factor-only") {
            options.factor_only = true;
        } else if (arg == "--max-outer" && i + 1 < argc) {
            options.max_outer_iterations = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--inner-max-iter" && i + 1 < argc) {
            options.max_inner_iterations = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--linear-solver" && i + 1 < argc) {
            options.linear_solver_kind =
                exp_20260415::block_ilu::parse_schur_linear_solver_kind(argv[++i]);
        } else if (arg == "--gmres-restart" && i + 1 < argc) {
            options.gmres_restart = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--residual-check-interval" && i + 1 < argc) {
            options.residual_check_interval = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--linear-tol" && i + 1 < argc) {
            options.linear_tolerance = std::stod(argv[++i]);
        } else if (arg == "--nonlinear-tol" && i + 1 < argc) {
            options.nonlinear_tolerance = std::stod(argv[++i]);
        } else if (arg == "--j11-reorder" && i + 1 < argc) {
            options.j11_reorder_mode =
                exp_20260415::block_ilu::parse_j11_reorder_mode(argv[++i]);
        } else if (arg == "--j11-solver" && i + 1 < argc) {
            options.j11_solver_kind =
                exp_20260415::block_ilu::parse_j11_solver_kind(argv[++i]);
        } else if (arg == "--j11-block-size" && i + 1 < argc) {
            options.j11_dense_block_size = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--j11-dense-backend" && i + 1 < argc) {
            options.j11_dense_backend =
                exp_20260415::block_ilu::parse_j11_dense_backend(argv[++i]);
            options.j11_dense_backend_explicit = true;
        } else if (arg == "--j11-partition" && i + 1 < argc) {
            options.j11_partition_mode =
                exp_20260415::block_ilu::parse_j11_partition_mode(argv[++i]);
        } else if (arg == "--inner-precision" && i + 1 < argc) {
            options.inner_precision =
                exp_20260415::block_ilu::parse_inner_precision(argv[++i]);
        } else if (arg == "--continue-on-linear-failure") {
            options.continue_on_linear_failure = true;
        } else if (arg == "--no-timing-breakdown") {
            options.collect_timing_breakdown = false;
        } else if (arg == "--compare-direction") {
            options.compare_direction = true;
        } else if (arg == "--dump-dx-log" && i + 1 < argc) {
            options.dump_dx_log = argv[++i];
        } else if (arg == "--line-search") {
            options.enable_line_search = true;
        } else if (arg == "--schur-preconditioner" && i + 1 < argc) {
            options.schur_preconditioner_kind =
                exp_20260415::block_ilu::parse_schur_preconditioner_kind(argv[++i]);
        } else if (arg == "--line-search-max-trials" && i + 1 < argc) {
            options.line_search_max_trials = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--line-search-reduction" && i + 1 < argc) {
            options.line_search_reduction = std::stod(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.compare_direction && options.max_outer_iterations != 1) {
        throw std::runtime_error("--compare-direction compares the first Newton step; use --max-outer 1");
    }
    if (options.compare_direction && (options.pattern_only || options.factor_only)) {
        throw std::runtime_error("--compare-direction requires the Newton linear solve to run");
    }
    if (!options.dump_dx_log.empty() && (options.pattern_only || options.factor_only)) {
        throw std::runtime_error("--dump-dx-log requires the Newton linear solve to run");
    }
    if (options.j11_dense_block_size <= 0) {
        throw std::runtime_error("--j11-block-size must be positive");
    }
    if (options.gmres_restart <= 0 || options.residual_check_interval <= 0) {
        throw std::runtime_error("invalid GMRES parameters");
    }
    if (options.inner_precision == exp_20260415::block_ilu::InnerPrecision::Fp32 &&
        options.j11_solver_kind != exp_20260415::block_ilu::J11SolverKind::PartitionDenseLu &&
        options.j11_solver_kind != exp_20260415::block_ilu::J11SolverKind::ExactKlu) {
        throw std::runtime_error("--inner-precision fp32 currently requires --j11-solver partition-dense-lu or exact-klu");
    }
    if (options.j11_solver_kind == exp_20260415::block_ilu::J11SolverKind::ExactKlu &&
        options.inner_precision != exp_20260415::block_ilu::InnerPrecision::Fp32) {
        throw std::runtime_error("--j11-solver exact-klu currently requires --inner-precision fp32");
    }
    if (options.line_search_max_trials <= 0 ||
        options.line_search_reduction <= 0.0 ||
        options.line_search_reduction >= 1.0) {
        throw std::runtime_error("invalid line-search parameters");
    }
    if (!options.j11_dense_backend_explicit &&
        options.inner_precision == exp_20260415::block_ilu::InnerPrecision::Fp32 &&
        options.j11_solver_kind == exp_20260415::block_ilu::J11SolverKind::PartitionDenseLu) {
        options.j11_dense_backend = exp_20260415::block_ilu::J11DenseBackend::TcNoPivot;
    }
    if (options.j11_dense_backend == exp_20260415::block_ilu::J11DenseBackend::TcNoPivot &&
        (options.inner_precision != exp_20260415::block_ilu::InnerPrecision::Fp32 ||
         options.j11_solver_kind != exp_20260415::block_ilu::J11SolverKind::PartitionDenseLu)) {
        throw std::runtime_error("--j11-dense-backend tc requires --inner-precision fp32 and --j11-solver partition-dense-lu");
    }
    if (options.j11_dense_backend == exp_20260415::block_ilu::J11DenseBackend::CusolverGetrf &&
        (options.inner_precision != exp_20260415::block_ilu::InnerPrecision::Fp32 ||
         options.j11_solver_kind != exp_20260415::block_ilu::J11SolverKind::PartitionDenseLu)) {
        throw std::runtime_error("--j11-dense-backend cusolver currently requires --inner-precision fp32 and --j11-solver partition-dense-lu");
    }
    if (options.schur_preconditioner_kind !=
            exp_20260415::block_ilu::SchurPreconditionerKind::None &&
        options.inner_precision != exp_20260415::block_ilu::InnerPrecision::Fp32) {
        throw std::runtime_error("--schur-preconditioner currently requires --inner-precision fp32");
    }
    if (options.linear_solver_kind == exp_20260415::block_ilu::SchurLinearSolverKind::Gmres &&
        options.inner_precision != exp_20260415::block_ilu::InnerPrecision::Fp32) {
        throw std::runtime_error("--linear-solver gmres currently requires --inner-precision fp32");
    }
    return options;
}

void split_complex_vector(const std::vector<std::complex<double>>& input,
                          std::vector<double>& re,
                          std::vector<double>& im)
{
    re.resize(input.size());
    im.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        re[i] = input[i].real();
        im[i] = input[i].imag();
    }
}

double frobenius_norm(const std::vector<double>& values)
{
    double sum = 0.0;
    for (double value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

double block_frobenius_from_full(const exp_20260415::block_ilu::HostCsrPattern& full,
                                 const std::vector<double>& values,
                                 int32_t row_begin,
                                 int32_t row_end,
                                 int32_t col_begin,
                                 int32_t col_end)
{
    double sum = 0.0;
    for (int32_t row = row_begin; row < row_end; ++row) {
        for (int32_t pos = full.row_ptr[static_cast<std::size_t>(row)];
             pos < full.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = full.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_end) {
                const double value = values[static_cast<std::size_t>(pos)];
                sum += value * value;
            }
        }
    }
    return std::sqrt(sum);
}

struct HostMismatch {
    std::vector<double> values;
    double norm2 = 0.0;
    double norm_inf = 0.0;
};

struct LinearResidualMetrics {
    double norm2 = 0.0;
    double rel2 = 0.0;
    double norm_inf = 0.0;
};

struct DirectionMetrics {
    double reference_norm = 0.0;
    double iterative_norm = 0.0;
    double diff_norm = 0.0;
    double relative_diff = 0.0;
    double cosine = 0.0;
    double angle_deg = 0.0;
};

HostMismatch compute_host_mismatch(const cupf::tests::DumpCaseData& data,
                                   const exp_20260415::block_ilu::ReducedJacobianPatterns& patterns,
                                   const std::vector<double>& y_re,
                                   const std::vector<double>& y_im,
                                   const std::vector<double>& v_re,
                                   const std::vector<double>& v_im,
                                   const std::vector<double>& sbus_re,
                                   const std::vector<double>& sbus_im)
{
    const int32_t dim = patterns.index.dim;
    HostMismatch result;
    result.values.resize(static_cast<std::size_t>(dim));

    for (int32_t tid = 0; tid < dim; ++tid) {
        int32_t bus = 0;
        bool take_q = false;
        if (tid < patterns.index.n_pv) {
            bus = patterns.index.pv[static_cast<std::size_t>(tid)];
        } else if (tid < patterns.index.n_pv + patterns.index.n_pq) {
            bus = patterns.index.pq[static_cast<std::size_t>(tid - patterns.index.n_pv)];
        } else {
            bus = patterns.index.pq[
                static_cast<std::size_t>(tid - patterns.index.n_pv - patterns.index.n_pq)];
            take_q = true;
        }

        double i_re = 0.0;
        double i_im = 0.0;
        for (int32_t pos = data.indptr[static_cast<std::size_t>(bus)];
             pos < data.indptr[static_cast<std::size_t>(bus + 1)];
             ++pos) {
            const int32_t col = data.indices[static_cast<std::size_t>(pos)];
            const double yr = y_re[static_cast<std::size_t>(pos)];
            const double yi = y_im[static_cast<std::size_t>(pos)];
            const double vr = v_re[static_cast<std::size_t>(col)];
            const double vi = v_im[static_cast<std::size_t>(col)];
            i_re += yr * vr - yi * vi;
            i_im += yr * vi + yi * vr;
        }

        const double vr = v_re[static_cast<std::size_t>(bus)];
        const double vi = v_im[static_cast<std::size_t>(bus)];
        const double mis_p = vr * i_re + vi * i_im - sbus_re[static_cast<std::size_t>(bus)];
        const double mis_q = vi * i_re - vr * i_im - sbus_im[static_cast<std::size_t>(bus)];
        const double value = take_q ? mis_q : mis_p;
        result.values[static_cast<std::size_t>(tid)] = value;
        result.norm2 += value * value;
        result.norm_inf = std::max(result.norm_inf, std::abs(value));
    }

    result.norm2 = std::sqrt(result.norm2);
    return result;
}

std::vector<double> negate_vector(const std::vector<double>& values)
{
    std::vector<double> output(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        output[i] = -values[i];
    }
    return output;
}

std::vector<double> solve_direct_klu(
    const exp_20260415::block_ilu::HostCsrPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& rhs)
{
    if (pattern.rows != pattern.cols ||
        pattern.rows != static_cast<int32_t>(rhs.size()) ||
        values.size() != pattern.col_idx.size()) {
        throw std::runtime_error("solve_direct_klu received inconsistent dimensions");
    }

    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Triplet = Eigen::Triplet<double, int32_t>;
    std::vector<Triplet> triplets;
    triplets.reserve(values.size());
    for (int32_t row = 0; row < pattern.rows; ++row) {
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            triplets.emplace_back(row,
                                  pattern.col_idx[static_cast<std::size_t>(pos)],
                                  values[static_cast<std::size_t>(pos)]);
        }
    }

    SparseMatrix jacobian(pattern.rows, pattern.cols);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
    jacobian.makeCompressed();

    Eigen::KLU<SparseMatrix> klu;
    klu.analyzePattern(jacobian);
    if (klu.info() != Eigen::Success) {
        throw std::runtime_error("KLU symbolic analysis failed for direction comparison");
    }
    klu.factorize(jacobian);
    if (klu.info() != Eigen::Success) {
        throw std::runtime_error("KLU numeric factorization failed for direction comparison");
    }

    Eigen::Map<const Eigen::VectorXd> rhs_map(rhs.data(), static_cast<Eigen::Index>(rhs.size()));
    const Eigen::VectorXd dx = klu.solve(rhs_map);
    if (klu.info() != Eigen::Success) {
        throw std::runtime_error("KLU solve failed for direction comparison");
    }

    return std::vector<double>(dx.data(), dx.data() + dx.size());
}

LinearResidualMetrics compute_linear_residual(
    const exp_20260415::block_ilu::HostCsrPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& rhs,
    const std::vector<double>& dx)
{
    if (pattern.rows != static_cast<int32_t>(rhs.size()) ||
        pattern.cols != static_cast<int32_t>(dx.size()) ||
        values.size() != pattern.col_idx.size()) {
        throw std::runtime_error("compute_linear_residual received inconsistent dimensions");
    }

    LinearResidualMetrics result;
    double rhs_norm2 = 0.0;
    for (double value : rhs) {
        rhs_norm2 += value * value;
    }
    rhs_norm2 = std::sqrt(rhs_norm2);

    for (int32_t row = 0; row < pattern.rows; ++row) {
        double ax = 0.0;
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = pattern.col_idx[static_cast<std::size_t>(pos)];
            ax += values[static_cast<std::size_t>(pos)] *
                  dx[static_cast<std::size_t>(col)];
        }
        const double residual = rhs[static_cast<std::size_t>(row)] - ax;
        result.norm2 += residual * residual;
        result.norm_inf = std::max(result.norm_inf, std::abs(residual));
    }

    result.norm2 = std::sqrt(result.norm2);
    result.rel2 = result.norm2 / std::max(rhs_norm2, std::numeric_limits<double>::min());
    return result;
}

std::vector<double> host_csr_matvec(
    const exp_20260415::block_ilu::HostCsrPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& x)
{
    if (pattern.cols != static_cast<int32_t>(x.size()) ||
        values.size() != pattern.col_idx.size()) {
        throw std::runtime_error("host_csr_matvec received inconsistent dimensions");
    }

    std::vector<double> y(static_cast<std::size_t>(pattern.rows), 0.0);
    for (int32_t row = 0; row < pattern.rows; ++row) {
        double sum = 0.0;
        for (int32_t pos = pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = pattern.col_idx[static_cast<std::size_t>(pos)];
            sum += values[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(col)];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

DirectionMetrics compare_direction_vectors(const std::vector<double>& reference,
                                           const std::vector<double>& iterative,
                                           int32_t offset,
                                           int32_t count)
{
    if (reference.size() != iterative.size() ||
        offset < 0 ||
        count < 0 ||
        offset + count > static_cast<int32_t>(reference.size())) {
        throw std::runtime_error("compare_direction_vectors received inconsistent dimensions");
    }

    DirectionMetrics result;
    double dot = 0.0;
    for (int32_t local = 0; local < count; ++local) {
        const std::size_t i = static_cast<std::size_t>(offset + local);
        const double ref = reference[i];
        const double iter = iterative[i];
        const double diff = iter - ref;
        dot += ref * iter;
        result.reference_norm += ref * ref;
        result.iterative_norm += iter * iter;
        result.diff_norm += diff * diff;
    }

    result.reference_norm = std::sqrt(result.reference_norm);
    result.iterative_norm = std::sqrt(result.iterative_norm);
    result.diff_norm = std::sqrt(result.diff_norm);
    result.relative_diff =
        result.diff_norm / std::max(result.reference_norm, std::numeric_limits<double>::min());

    const double denom = result.reference_norm * result.iterative_norm;
    if (denom > 0.0) {
        result.cosine = std::clamp(dot / denom, -1.0, 1.0);
    } else {
        result.cosine = (result.reference_norm == result.iterative_norm) ? 1.0 : 0.0;
    }

    constexpr double kPi = 3.141592653589793238462643383279502884;
    result.angle_deg = std::acos(result.cosine) * 180.0 / kPi;
    return result;
}

void apply_host_voltage_update(
    const exp_20260415::block_ilu::ReducedJacobianPatterns& patterns,
    const std::vector<double>& v_re,
    const std::vector<double>& v_im,
    const std::vector<double>& dx,
    std::vector<double>& updated_re,
    std::vector<double>& updated_im)
{
    if (dx.size() != static_cast<std::size_t>(patterns.index.dim) ||
        v_re.size() != v_im.size()) {
        throw std::runtime_error("apply_host_voltage_update received inconsistent dimensions");
    }

    const int32_t n_bus = patterns.index.n_bus;
    std::vector<double> va(static_cast<std::size_t>(n_bus));
    std::vector<double> vm(static_cast<std::size_t>(n_bus));
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const std::size_t i = static_cast<std::size_t>(bus);
        va[i] = std::atan2(v_im[i], v_re[i]);
        vm[i] = std::hypot(v_re[i], v_im[i]);
    }

    for (int32_t tid = 0; tid < patterns.index.dim; ++tid) {
        const double value = dx[static_cast<std::size_t>(tid)];
        if (tid < patterns.index.n_pv) {
            va[static_cast<std::size_t>(patterns.index.pv[static_cast<std::size_t>(tid)])] +=
                value;
        } else if (tid < patterns.index.n_pv + patterns.index.n_pq) {
            const int32_t local = tid - patterns.index.n_pv;
            va[static_cast<std::size_t>(patterns.index.pq[static_cast<std::size_t>(local)])] +=
                value;
        } else {
            const int32_t local = tid - patterns.index.n_pv - patterns.index.n_pq;
            vm[static_cast<std::size_t>(patterns.index.pq[static_cast<std::size_t>(local)])] +=
                value;
        }
    }

    updated_re.resize(v_re.size());
    updated_im.resize(v_im.size());
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const std::size_t i = static_cast<std::size_t>(bus);
        updated_re[i] = vm[i] * std::cos(va[i]);
        updated_im[i] = vm[i] * std::sin(va[i]);
    }
}

void write_dx_log(const std::filesystem::path& path,
                  const exp_20260415::block_ilu::ReducedJacobianPatterns& patterns,
                  const exp_20260415::block_ilu::BlockIluSolveStats& stats)
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open dx log for writing: " + path.string());
    }

    out << std::boolalpha << std::scientific << std::setprecision(17);
    out << "outer,component,variable,bus,applied,alpha,dx,applied_dx\n";
    for (const auto& trace : stats.dx_trace) {
        if (trace.dx.size() != static_cast<std::size_t>(patterns.index.dim)) {
            throw std::runtime_error("dx trace has inconsistent dimension");
        }
        for (int32_t component = 0; component < patterns.index.dim; ++component) {
            const bool is_theta = component < patterns.index.n_pvpq;
            const int32_t local =
                is_theta ? component : component - patterns.index.n_pvpq;
            const int32_t bus =
                is_theta
                    ? patterns.index.pvpq[static_cast<std::size_t>(local)]
                    : patterns.index.pq[static_cast<std::size_t>(local)];
            out << trace.outer_iteration << ','
                << component << ','
                << (is_theta ? "theta" : "vm") << ','
                << bus << ','
                << trace.dx_was_applied << ','
                << trace.alpha << ','
                << trace.dx[static_cast<std::size_t>(component)] << ','
                << (trace.dx_was_applied
                        ? trace.alpha * trace.dx[static_cast<std::size_t>(component)]
                        : 0.0)
                << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const auto case_dir = options.dataset_root / options.case_name;
        const cupf::tests::DumpCaseData data = cupf::tests::load_dump_case(case_dir);
        const auto patterns =
            exp_20260415::block_ilu::build_reduced_jacobian_patterns(data.rows,
                                                                      data.pv,
                                                                      data.pq,
                                                                      data.indptr,
                                                                      data.indices);

        std::cout << "BLOCK_ILU_PATTERN "
                  << "case=" << data.case_name << ' '
                  << "n_bus=" << data.rows << ' '
                  << "n_pv=" << patterns.index.n_pv << ' '
                  << "n_pq=" << patterns.index.n_pq << ' '
                  << "n_pvpq=" << patterns.index.n_pvpq << ' '
                  << "dim=" << patterns.index.dim << ' '
                  << "full_nnz=" << patterns.full.nnz() << ' '
                  << "j11_nnz=" << patterns.j11.nnz() << ' '
                  << "j12_nnz=" << patterns.j12.nnz() << ' '
                  << "j21_nnz=" << patterns.j21.nnz() << ' '
                  << "j22_nnz=" << patterns.j22.nnz()
                  << '\n';

        if (options.pattern_only) {
            return 0;
        }

        std::vector<double> y_re;
        std::vector<double> y_im;
        std::vector<double> v_re;
        std::vector<double> v_im;
        std::vector<double> sbus_re;
        std::vector<double> sbus_im;
        split_complex_vector(data.ybus_data, y_re, y_im);
        split_complex_vector(data.v0, v_re, v_im);
        split_complex_vector(data.sbus, sbus_re, sbus_im);

        DeviceBuffer<int32_t> d_y_row_ptr;
        DeviceBuffer<int32_t> d_y_col_idx;
        DeviceBuffer<double> d_y_re;
        DeviceBuffer<double> d_y_im;
        DeviceBuffer<double> d_sbus_re;
        DeviceBuffer<double> d_sbus_im;
        DeviceBuffer<double> d_v_re;
        DeviceBuffer<double> d_v_im;
        d_y_row_ptr.assign(data.indptr.data(), data.indptr.size());
        d_y_col_idx.assign(data.indices.data(), data.indices.size());
        d_y_re.assign(y_re.data(), y_re.size());
        d_y_im.assign(y_im.data(), y_im.size());
        d_sbus_re.assign(sbus_re.data(), sbus_re.size());
        d_sbus_im.assign(sbus_im.data(), sbus_im.size());
        d_v_re.assign(v_re.data(), v_re.size());
        d_v_im.assign(v_im.data(), v_im.size());

        exp_20260415::block_ilu::ReducedJacobianAssembler assembler;
        assembler.analyze(patterns);
        assembler.assemble(d_y_re.data(), d_y_im.data(), d_v_re.data(), d_v_im.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> full_values;
        std::vector<double> j11_values;
        std::vector<double> j12_values;
        std::vector<double> j21_values;
        std::vector<double> j22_values;
        assembler.download_full_values(full_values);
        assembler.download_j11_values(j11_values);
        assembler.download_j12_values(j12_values);
        assembler.download_j21_values(j21_values);
        assembler.download_j22_values(j22_values);

        const int32_t n_pvpq = patterns.index.n_pvpq;
        const int32_t n_pq = patterns.index.n_pq;
        const double full_j11_fro =
            block_frobenius_from_full(patterns.full, full_values, 0, n_pvpq, 0, n_pvpq);
        const double full_j12_fro =
            block_frobenius_from_full(patterns.full,
                                      full_values,
                                      0,
                                      n_pvpq,
                                      n_pvpq,
                                      n_pvpq + n_pq);
        const double full_j21_fro =
            block_frobenius_from_full(patterns.full,
                                      full_values,
                                      n_pvpq,
                                      n_pvpq + n_pq,
                                      0,
                                      n_pvpq);
        const double full_j22_fro =
            block_frobenius_from_full(patterns.full,
                                      full_values,
                                      n_pvpq,
                                      n_pvpq + n_pq,
                                      n_pvpq,
                                      n_pvpq + n_pq);

        std::cout << std::scientific << std::setprecision(12)
                  << "BLOCK_ILU_VALUES "
                  << "case=" << data.case_name << ' '
                  << "j11_fro=" << full_j11_fro << ' '
                  << "j12_fro=" << full_j12_fro << ' '
                  << "j21_fro=" << full_j21_fro << ' '
                  << "j22_fro=" << full_j22_fro << ' '
                  << "j11_block_fro=" << frobenius_norm(j11_values) << ' '
                  << "j12_block_fro=" << frobenius_norm(j12_values) << ' '
                  << "j21_block_fro=" << frobenius_norm(j21_values) << ' '
                  << "j22_block_fro=" << frobenius_norm(j22_values)
                  << '\n';

        exp_20260415::block_ilu::CusparseIlu0Block ilu_j11("J11");
        exp_20260415::block_ilu::CusparseIlu0Block ilu_j22("J22");
        ilu_j11.analyze(assembler.j11_view());
        ilu_j22.analyze(assembler.j22_view());
        ilu_j11.factorize();
        ilu_j22.factorize();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "BLOCK_ILU_FACTORIZATION "
                  << "case=" << data.case_name << ' '
                  << "j11_zero_pivot=" << ilu_j11.last_zero_pivot() << ' '
                  << "j22_zero_pivot=" << ilu_j22.last_zero_pivot()
                  << '\n';

        if (options.factor_only) {
            return 0;
        }

        // Reset the voltage buffers before the Newton solve. The factorization
        // smoke above does not modify V, but this keeps the probe flow explicit.
        d_v_re.assign(v_re.data(), v_re.size());
        d_v_im.assign(v_im.data(), v_im.size());

        exp_20260415::block_ilu::BlockIluSolverOptions solver_options;
        solver_options.nonlinear_tolerance = options.nonlinear_tolerance;
        solver_options.linear_tolerance = options.linear_tolerance;
        solver_options.max_outer_iterations = options.max_outer_iterations;
        solver_options.max_inner_iterations = options.max_inner_iterations;
        solver_options.linear_solver_kind = options.linear_solver_kind;
        solver_options.gmres_restart = options.gmres_restart;
        solver_options.gmres_residual_check_interval = options.residual_check_interval;
        solver_options.j11_reorder_mode = options.j11_reorder_mode;
        solver_options.j11_solver_kind = options.j11_solver_kind;
        solver_options.j11_dense_block_size = options.j11_dense_block_size;
        solver_options.j11_dense_backend = options.j11_dense_backend;
        solver_options.j11_partition_mode = options.j11_partition_mode;
        solver_options.inner_precision = options.inner_precision;
        solver_options.collect_timing_breakdown = options.collect_timing_breakdown;
        solver_options.continue_on_linear_failure = options.continue_on_linear_failure;
        solver_options.collect_dx_trace = !options.dump_dx_log.empty();
        solver_options.enable_line_search = options.enable_line_search;
        solver_options.schur_preconditioner_kind = options.schur_preconditioner_kind;
        solver_options.line_search_max_trials = options.line_search_max_trials;
        solver_options.line_search_reduction = options.line_search_reduction;

        exp_20260415::block_ilu::HostPowerFlowStructure structure{
            .n_bus = data.rows,
            .ybus_row_ptr = data.indptr,
            .ybus_col_idx = data.indices,
            .pv = data.pv,
            .pq = data.pq,
        };
        exp_20260415::block_ilu::DevicePowerFlowState state{
            .ybus_row_ptr = d_y_row_ptr.data(),
            .ybus_col_idx = d_y_col_idx.data(),
            .ybus_re = d_y_re.data(),
            .ybus_im = d_y_im.data(),
            .sbus_re = d_sbus_re.data(),
            .sbus_im = d_sbus_im.data(),
            .voltage_re = d_v_re.data(),
            .voltage_im = d_v_im.data(),
        };

        exp_20260415::block_ilu::PowerFlowBlockIluSolver solver(solver_options);
        solver.analyze(structure);
        const exp_20260415::block_ilu::BlockIluSolveStats stats = solver.solve(state);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!options.dump_dx_log.empty()) {
            write_dx_log(options.dump_dx_log, patterns, stats);
        }

        std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                  << "BLOCK_ILU_SOLVE "
                  << "case=" << data.case_name << ' '
                  << "converged=" << stats.converged << ' '
                  << "linear_solver="
                  << exp_20260415::block_ilu::schur_linear_solver_kind_name(
                         options.linear_solver_kind)
                  << ' '
                  << "inner_precision="
                  << exp_20260415::block_ilu::inner_precision_name(options.inner_precision)
                  << ' '
                  << "j11_solver="
                  << exp_20260415::block_ilu::j11_solver_kind_name(options.j11_solver_kind)
                  << ' '
                  << "j11_dense_backend="
                  << exp_20260415::block_ilu::j11_dense_backend_name(options.j11_dense_backend)
                  << ' '
                  << "j11_reorder="
                  << exp_20260415::block_ilu::j11_reorder_mode_name(options.j11_reorder_mode)
                  << ' '
                  << "j11_partition="
                  << exp_20260415::block_ilu::j11_partition_mode_name(
                         options.j11_partition_mode)
                  << ' '
                  << "j11_block_size=" << options.j11_dense_block_size << ' '
                  << "schur_preconditioner="
                  << exp_20260415::block_ilu::schur_preconditioner_kind_name(
                         options.schur_preconditioner_kind)
                  << ' '
                  << "line_search=" << options.enable_line_search << ' '
                  << "line_search_max_trials=" << options.line_search_max_trials << ' '
                  << "line_search_reduction=" << options.line_search_reduction << ' '
                  << "outer_iterations=" << stats.outer_iterations << ' '
                  << "final_mismatch=" << stats.final_mismatch << ' '
                  << "total_inner_iterations=" << stats.total_inner_iterations << ' '
                  << "total_restart_cycles=" << stats.total_restart_cycles << ' '
                  << "total_schur_matvec_calls=" << stats.total_schur_matvec_calls << ' '
                  << "total_schur_preconditioner_applies="
                  << stats.total_schur_preconditioner_applies << ' '
                  << "total_spmv_calls=" << stats.total_spmv_calls << ' '
                  << "total_j11_solve_calls=" << stats.total_j11_solve_calls << ' '
                  << "total_preconditioner_applies=" << stats.total_preconditioner_applies << ' '
                  << "total_reduction_calls=" << stats.total_reduction_calls << ' '
                  << "total_linear_solve_sec=" << stats.total_linear_solve_sec << ' '
                  << "total_linear_iteration_sec=" << stats.total_linear_iteration_sec << ' '
                  << "linear_preconditioner_sec="
                  << stats.total_linear_timing.schur_j11_solve_sec << ' '
                  << "linear_schur_preconditioner_sec="
                  << stats.total_linear_timing.schur_preconditioner_sec << ' '
                  << "linear_spmv_sec=" << stats.total_linear_timing.schur_spmv_sec << ' '
                  << "linear_reduction_sec=" << stats.total_linear_timing.reduction_sec << ' '
                  << "linear_vector_update_sec="
                  << stats.total_linear_timing.vector_update_sec << ' '
                  << "linear_small_solve_sec="
                  << stats.total_linear_timing.small_solve_sec << ' '
                  << "linear_residual_refresh_sec="
                  << stats.total_linear_timing.residual_refresh_sec << ' '
                  << "schur_rhs_sec=" << stats.total_linear_timing.schur_rhs_sec << ' '
                  << "schur_matvec_sec=" << stats.total_linear_timing.schur_matvec_sec << ' '
                  << "schur_recover_sec=" << stats.total_linear_timing.schur_recover_sec << ' '
                  << "avg_linear_iteration_sec="
                  << (stats.total_inner_iterations > 0
                          ? stats.total_linear_iteration_sec /
                                static_cast<double>(stats.total_inner_iterations)
                          : 0.0)
                  << ' '
                  << "preconditioner_rebuilds=" << stats.preconditioner_rebuilds << ' '
                  << "j11_zero_pivot=" << stats.j11_zero_pivot << ' '
                  << "j22_zero_pivot=" << stats.j22_zero_pivot << ' '
                  << "failure_reason="
                  << (stats.failure_reason.empty() ? "none" : stats.failure_reason)
                  << '\n';
        for (const auto& step : stats.step_trace) {
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "BLOCK_ILU_OUTER "
                      << "case=" << data.case_name << ' '
                      << "outer=" << step.outer_iteration << ' '
                      << "before=" << step.before_mismatch << ' '
                      << "after=" << step.after_mismatch << ' '
                      << "dx_was_applied=" << step.dx_was_applied << ' '
                      << "preconditioner_factorized=" << step.preconditioner_factorized << ' '
                      << "outer_iteration_sec=" << step.outer_iteration_sec << ' '
                      << "mismatch_sec=" << step.mismatch_sec << ' '
                      << "jacobian_assembly_sec=" << step.jacobian_assembly_sec << ' '
                      << "preconditioner_factorize_sec="
                      << step.preconditioner_factorize_sec << ' '
                      << "voltage_update_sec=" << step.voltage_update_sec << ' '
                      << "after_mismatch_sec=" << step.after_mismatch_sec << ' '
                      << "line_search_sec=" << step.line_search_sec << ' '
                      << "line_search_alpha=" << step.line_search_alpha << ' '
                      << "line_search_trials=" << step.line_search_trials << ' '
                      << "line_search_accepted=" << step.line_search_accepted
                      << '\n';
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "BLOCK_ILU_INNER "
                      << "case=" << data.case_name << ' '
                      << "outer=" << step.outer_iteration << ' '
                      << "linear_solver="
                      << exp_20260415::block_ilu::schur_linear_solver_kind_name(
                             options.linear_solver_kind)
                      << ' '
                      << "inner_precision="
                      << exp_20260415::block_ilu::inner_precision_name(options.inner_precision)
                      << ' '
                      << "j11_solver="
                      << exp_20260415::block_ilu::j11_solver_kind_name(
                             options.j11_solver_kind)
                      << ' '
                      << "j11_dense_backend="
                      << exp_20260415::block_ilu::j11_dense_backend_name(
                             options.j11_dense_backend)
                      << ' '
                      << "j11_reorder="
                      << exp_20260415::block_ilu::j11_reorder_mode_name(
                             options.j11_reorder_mode)
                      << ' '
                      << "j11_partition="
                      << exp_20260415::block_ilu::j11_partition_mode_name(
                             options.j11_partition_mode)
                      << ' '
                      << "j11_block_size=" << options.j11_dense_block_size << ' '
                      << "schur_preconditioner="
                      << exp_20260415::block_ilu::schur_preconditioner_kind_name(
                             options.schur_preconditioner_kind)
                      << ' '
                      << "linear_converged=" << step.linear_converged << ' '
                      << "linear_relative_residual=" << step.linear_relative_residual << ' '
                      << "linear_solve_sec=" << step.linear_solve_sec << ' '
                      << "linear_avg_iteration_sec=" << step.linear_avg_iteration_sec << ' '
                      << "inner_iterations=" << step.inner_iterations << ' '
                      << "restart_cycles=" << step.restart_cycles << ' '
                      << "schur_matvec_calls=" << step.schur_matvec_calls << ' '
                      << "spmv_calls=" << step.spmv_calls << ' '
                      << "j11_solve_calls=" << step.j11_solve_calls << ' '
                      << "schur_preconditioner_applies="
                      << step.schur_preconditioner_applies << ' '
                      << "preconditioner_applies=" << step.preconditioner_applies << ' '
                      << "reduction_calls=" << step.reduction_calls << ' '
                      << "linear_preconditioner_sec=" << step.linear_preconditioner_sec << ' '
                      << "linear_schur_preconditioner_sec="
                      << step.linear_schur_preconditioner_sec << ' '
                      << "linear_spmv_sec=" << step.linear_spmv_sec << ' '
                      << "linear_reduction_sec=" << step.linear_reduction_sec << ' '
                      << "linear_vector_update_sec=" << step.linear_vector_update_sec << ' '
                      << "linear_small_solve_sec=" << step.linear_small_solve_sec << ' '
                      << "linear_residual_refresh_sec="
                      << step.linear_residual_refresh_sec << ' '
                      << "schur_rhs_sec=" << step.schur_rhs_sec << ' '
                      << "schur_matvec_sec=" << step.schur_matvec_sec << ' '
                      << "schur_recover_sec=" << step.schur_recover_sec
                      << '\n';
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "BLOCK_ILU_STEP "
                      << "case=" << data.case_name << ' '
                      << "outer=" << step.outer_iteration << ' '
                      << "linear_converged=" << step.linear_converged << ' '
                      << "linear_solver="
                      << exp_20260415::block_ilu::schur_linear_solver_kind_name(
                             options.linear_solver_kind)
                      << ' '
                      << "inner_precision="
                      << exp_20260415::block_ilu::inner_precision_name(options.inner_precision)
                      << ' '
                      << "j11_solver="
                      << exp_20260415::block_ilu::j11_solver_kind_name(
                             options.j11_solver_kind)
                      << ' '
                      << "j11_dense_backend="
                      << exp_20260415::block_ilu::j11_dense_backend_name(
                             options.j11_dense_backend)
                      << ' '
                      << "j11_reorder="
                      << exp_20260415::block_ilu::j11_reorder_mode_name(
                             options.j11_reorder_mode)
                      << ' '
                      << "j11_partition="
                      << exp_20260415::block_ilu::j11_partition_mode_name(
                             options.j11_partition_mode)
                      << ' '
                      << "j11_block_size=" << options.j11_dense_block_size << ' '
                      << "schur_preconditioner="
                      << exp_20260415::block_ilu::schur_preconditioner_kind_name(
                             options.schur_preconditioner_kind)
                      << ' '
                      << "dx_was_applied=" << step.dx_was_applied << ' '
                      << "line_search_alpha=" << step.line_search_alpha << ' '
                      << "line_search_trials=" << step.line_search_trials << ' '
                      << "line_search_accepted=" << step.line_search_accepted << ' '
                      << "before=" << step.before_mismatch << ' '
                      << "after=" << step.after_mismatch << ' '
                      << "linear_relative_residual=" << step.linear_relative_residual << ' '
                      << "linear_solve_sec=" << step.linear_solve_sec << ' '
                      << "linear_avg_iteration_sec=" << step.linear_avg_iteration_sec << ' '
                      << "linear_preconditioner_sec=" << step.linear_preconditioner_sec << ' '
                      << "linear_schur_preconditioner_sec="
                      << step.linear_schur_preconditioner_sec << ' '
                      << "linear_spmv_sec=" << step.linear_spmv_sec << ' '
                      << "linear_reduction_sec=" << step.linear_reduction_sec << ' '
                      << "linear_vector_update_sec=" << step.linear_vector_update_sec << ' '
                      << "linear_small_solve_sec=" << step.linear_small_solve_sec << ' '
                      << "linear_residual_refresh_sec="
                      << step.linear_residual_refresh_sec << ' '
                      << "inner_iterations=" << step.inner_iterations << ' '
                      << "restart_cycles=" << step.restart_cycles << ' '
                      << "schur_matvec_calls=" << step.schur_matvec_calls << ' '
                      << "spmv_calls=" << step.spmv_calls << ' '
                      << "j11_solve_calls=" << step.j11_solve_calls << ' '
                      << "schur_preconditioner_applies="
                      << step.schur_preconditioner_applies << ' '
                      << "preconditioner_applies=" << step.preconditioner_applies << ' '
                      << "reduction_calls=" << step.reduction_calls
                      << '\n';
        }

        if (options.compare_direction) {
            std::vector<double> iterative_dx;
            solver.download_last_dx(iterative_dx);

            const HostMismatch mismatch0 = compute_host_mismatch(data,
                                                                 patterns,
                                                                 y_re,
                                                                 y_im,
                                                                 v_re,
                                                                 v_im,
                                                                 sbus_re,
                                                                 sbus_im);
            const std::vector<double> rhs0 = negate_vector(mismatch0.values);
            const std::vector<double> reference_dx =
                solve_direct_klu(patterns.full, full_values, rhs0);

            const DirectionMetrics full_metrics =
                compare_direction_vectors(reference_dx, iterative_dx, 0, patterns.index.dim);
            const DirectionMetrics theta_metrics =
                compare_direction_vectors(reference_dx, iterative_dx, 0, patterns.index.n_pvpq);
            const DirectionMetrics vm_metrics = compare_direction_vectors(reference_dx,
                                                                          iterative_dx,
                                                                          patterns.index.n_pvpq,
                                                                          patterns.index.n_pq);

            const LinearResidualMetrics reference_residual =
                compute_linear_residual(patterns.full, full_values, rhs0, reference_dx);
            const LinearResidualMetrics iterative_residual =
                compute_linear_residual(patterns.full, full_values, rhs0, iterative_dx);

            std::vector<double> iterative_dvm(static_cast<std::size_t>(patterns.index.n_pq));
            std::copy(iterative_dx.begin() + patterns.index.n_pvpq,
                      iterative_dx.end(),
                      iterative_dvm.begin());
            const std::vector<double> j12_dvm =
                host_csr_matvec(patterns.j12, j12_values, iterative_dvm);
            std::vector<double> theta_rhs(static_cast<std::size_t>(patterns.index.n_pvpq));
            for (int32_t i = 0; i < patterns.index.n_pvpq; ++i) {
                theta_rhs[static_cast<std::size_t>(i)] =
                    rhs0[static_cast<std::size_t>(i)] - j12_dvm[static_cast<std::size_t>(i)];
            }
            const std::vector<double> exact_j11_theta =
                solve_direct_klu(patterns.j11, j11_values, theta_rhs);
            std::vector<double> exact_j11_recovered_dx(
                static_cast<std::size_t>(patterns.index.dim));
            std::copy(exact_j11_theta.begin(),
                      exact_j11_theta.end(),
                      exact_j11_recovered_dx.begin());
            std::copy(iterative_dvm.begin(),
                      iterative_dvm.end(),
                      exact_j11_recovered_dx.begin() + patterns.index.n_pvpq);

            const DirectionMetrics exact_j11_full_metrics =
                compare_direction_vectors(reference_dx,
                                          exact_j11_recovered_dx,
                                          0,
                                          patterns.index.dim);
            const DirectionMetrics exact_j11_theta_metrics =
                compare_direction_vectors(reference_dx,
                                          exact_j11_recovered_dx,
                                          0,
                                          patterns.index.n_pvpq);
            const DirectionMetrics exact_j11_vm_metrics =
                compare_direction_vectors(reference_dx,
                                          exact_j11_recovered_dx,
                                          patterns.index.n_pvpq,
                                          patterns.index.n_pq);
            const LinearResidualMetrics exact_j11_residual =
                compute_linear_residual(patterns.full,
                                        full_values,
                                        rhs0,
                                        exact_j11_recovered_dx);

            std::vector<double> ref_v_re;
            std::vector<double> ref_v_im;
            std::vector<double> iter_v_re;
            std::vector<double> iter_v_im;
            std::vector<double> exact_j11_v_re;
            std::vector<double> exact_j11_v_im;
            apply_host_voltage_update(patterns, v_re, v_im, reference_dx, ref_v_re, ref_v_im);
            apply_host_voltage_update(patterns, v_re, v_im, iterative_dx, iter_v_re, iter_v_im);
            apply_host_voltage_update(patterns,
                                      v_re,
                                      v_im,
                                      exact_j11_recovered_dx,
                                      exact_j11_v_re,
                                      exact_j11_v_im);
            const HostMismatch reference_after = compute_host_mismatch(data,
                                                                       patterns,
                                                                       y_re,
                                                                       y_im,
                                                                       ref_v_re,
                                                                       ref_v_im,
                                                                       sbus_re,
                                                                       sbus_im);
            const HostMismatch iterative_after = compute_host_mismatch(data,
                                                                       patterns,
                                                                       y_re,
                                                                       y_im,
                                                                       iter_v_re,
                                                                       iter_v_im,
                                                                       sbus_re,
                                                                       sbus_im);
            const HostMismatch exact_j11_after = compute_host_mismatch(data,
                                                                       patterns,
                                                                       y_re,
                                                                       y_im,
                                                                       exact_j11_v_re,
                                                                       exact_j11_v_im,
                                                                       sbus_re,
                                                                       sbus_im);

            const exp_20260415::block_ilu::BlockIluStepTrace empty_step;
            const auto& first_step = stats.step_trace.empty() ? empty_step
                                                              : stats.step_trace.front();
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "DIRECTION_COMPARE "
                      << "case=" << data.case_name << ' '
                      << "reference_solver=eigen_klu "
                      << "iterative_solver="
                      << exp_20260415::block_ilu::schur_linear_solver_kind_name(
                             options.linear_solver_kind)
                      << ' '
                      << "inner_precision="
                      << exp_20260415::block_ilu::inner_precision_name(options.inner_precision)
                      << ' '
                      << "j11_solver="
                      << exp_20260415::block_ilu::j11_solver_kind_name(
                             options.j11_solver_kind)
                      << ' '
                      << "j11_dense_backend="
                      << exp_20260415::block_ilu::j11_dense_backend_name(
                             options.j11_dense_backend)
                      << ' '
                      << "j11_reorder="
                      << exp_20260415::block_ilu::j11_reorder_mode_name(
                             options.j11_reorder_mode)
                      << ' '
                      << "j11_partition="
                      << exp_20260415::block_ilu::j11_partition_mode_name(
                             options.j11_partition_mode)
                      << ' '
                      << "j11_block_size=" << options.j11_dense_block_size << ' '
                      << "linear_tolerance=" << options.linear_tolerance << ' '
                      << "inner_iterations=" << first_step.inner_iterations << ' '
                      << "linear_converged=" << first_step.linear_converged << ' '
                      << "linear_relative_residual=" << first_step.linear_relative_residual << ' '
                      << "mismatch_before=" << mismatch0.norm_inf << ' '
                      << "cos_dx=" << full_metrics.cosine << ' '
                      << "angle_dx_deg=" << full_metrics.angle_deg << ' '
                      << "rel_dx_diff=" << full_metrics.relative_diff << ' '
                      << "norm_dx_ref=" << full_metrics.reference_norm << ' '
                      << "norm_dx_iter=" << full_metrics.iterative_norm << ' '
                      << "cos_theta=" << theta_metrics.cosine << ' '
                      << "angle_theta_deg=" << theta_metrics.angle_deg << ' '
                      << "rel_theta_diff=" << theta_metrics.relative_diff << ' '
                      << "cos_vm=" << vm_metrics.cosine << ' '
                      << "angle_vm_deg=" << vm_metrics.angle_deg << ' '
                      << "rel_vm_diff=" << vm_metrics.relative_diff << ' '
                      << "ref_linear_relres2=" << reference_residual.rel2 << ' '
                      << "iter_linear_relres2=" << iterative_residual.rel2 << ' '
                      << "ref_linear_resinf=" << reference_residual.norm_inf << ' '
                      << "iter_linear_resinf=" << iterative_residual.norm_inf << ' '
                      << "mismatch_after_ref=" << reference_after.norm_inf << ' '
                      << "mismatch_after_iter=" << iterative_after.norm_inf
                      << '\n';
            std::cout << std::boolalpha << std::scientific << std::setprecision(12)
                      << "J11_EXACT_RECOVER_COMPARE "
                      << "case=" << data.case_name << ' '
                      << "reference_solver=eigen_klu_full "
                      << "vm_source="
                      << exp_20260415::block_ilu::schur_linear_solver_kind_name(
                             options.linear_solver_kind)
                      << ' '
                      << "theta_recover_solver=eigen_klu_j11 "
                      << "inner_precision="
                      << exp_20260415::block_ilu::inner_precision_name(options.inner_precision)
                      << ' '
                      << "j11_solver="
                      << exp_20260415::block_ilu::j11_solver_kind_name(
                             options.j11_solver_kind)
                      << ' '
                      << "j11_dense_backend="
                      << exp_20260415::block_ilu::j11_dense_backend_name(
                             options.j11_dense_backend)
                      << ' '
                      << "j11_reorder="
                      << exp_20260415::block_ilu::j11_reorder_mode_name(
                             options.j11_reorder_mode)
                      << ' '
                      << "j11_partition="
                      << exp_20260415::block_ilu::j11_partition_mode_name(
                             options.j11_partition_mode)
                      << ' '
                      << "j11_block_size=" << options.j11_dense_block_size << ' '
                      << "linear_tolerance=" << options.linear_tolerance << ' '
                      << "inner_iterations=" << first_step.inner_iterations << ' '
                      << "linear_converged=" << first_step.linear_converged << ' '
                      << "cos_dx=" << exact_j11_full_metrics.cosine << ' '
                      << "angle_dx_deg=" << exact_j11_full_metrics.angle_deg << ' '
                      << "rel_dx_diff=" << exact_j11_full_metrics.relative_diff << ' '
                      << "cos_theta=" << exact_j11_theta_metrics.cosine << ' '
                      << "angle_theta_deg=" << exact_j11_theta_metrics.angle_deg << ' '
                      << "rel_theta_diff=" << exact_j11_theta_metrics.relative_diff << ' '
                      << "cos_vm=" << exact_j11_vm_metrics.cosine << ' '
                      << "angle_vm_deg=" << exact_j11_vm_metrics.angle_deg << ' '
                      << "rel_vm_diff=" << exact_j11_vm_metrics.relative_diff << ' '
                      << "linear_relres2=" << exact_j11_residual.rel2 << ' '
                      << "linear_resinf=" << exact_j11_residual.norm_inf << ' '
                      << "mismatch_after_ref=" << reference_after.norm_inf << ' '
                      << "mismatch_after_iter=" << iterative_after.norm_inf << ' '
                      << "mismatch_after_exact_j11_recover=" << exact_j11_after.norm_inf
                      << '\n';
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "block_ilu_probe failed: " << ex.what() << '\n';
        return 1;
    }
}
