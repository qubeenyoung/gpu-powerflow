#include "cuiter/solver/gmres_solver.hpp"

#include "cuiter/kernels/gmres_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuiter {
namespace {

template <typename Fn>
double gpu_timed(Fn&& fn)
{
    CudaEventTimer timer;
    timer.start();
    fn();
    return timer.stop();
}

template <typename Fn>
double host_timed_with_sync(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

std::size_t basis_offset(int32_t n, int32_t column)
{
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(column);
}

void validate_matrix_values(const CsrMatrix& matrix)
{
    if (matrix.rows <= 0 || matrix.rows != matrix.cols ||
        static_cast<int32_t>(matrix.row_ptr.size()) != matrix.rows + 1 ||
        matrix.nnz() <= 0) {
        throw std::runtime_error("GmresSolver requires a nonempty square CSR matrix");
    }
}

double coefficient_of_variation(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());
    if (mean <= std::numeric_limits<double>::min()) {
        return 0.0;
    }
    double square_sum = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        square_sum += diff * diff;
    }
    return std::sqrt(square_sum / static_cast<double>(values.size())) / mean;
}

void scale_stats(const std::vector<double>& values,
                 double& min_value,
                 double& max_value,
                 double& geomean)
{
    if (values.empty()) {
        min_value = 1.0;
        max_value = 1.0;
        geomean = 1.0;
        return;
    }

    min_value = std::numeric_limits<double>::infinity();
    max_value = 0.0;
    double log_sum = 0.0;
    for (double value : values) {
        const double safe_value = std::max(std::abs(value), std::numeric_limits<double>::min());
        min_value = std::min(min_value, safe_value);
        max_value = std::max(max_value, safe_value);
        log_sum += std::log(safe_value);
    }
    geomean = std::exp(log_sum / static_cast<double>(values.size()));
}

}  // namespace

GmresSolver::GmresSolver(GmresSolverOptions options)
    : options_(std::move(options))
{
    CUITER_CUBLAS_CHECK(cublasCreate(&cublas_));
    CUITER_CUBLAS_CHECK(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE));
}

GmresSolver::~GmresSolver()
{
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void GmresSolver::set_options(const GmresSolverOptions& options)
{
    if (analyzed_) {
        throw std::runtime_error("set_options must be called before analyze");
    }
    options_ = options;
}

bool GmresSolver::using_metis_block_jacobi() const
{
    return options_.preconditioner == "metis_block_jacobi" ||
           options_.preconditioner == "metis_block_jacobi_coarse" ||
           options_.preconditioner == "ras_overlap1";
}

bool GmresSolver::using_ruiz_scaling() const
{
    return options_.linear_scaling == "ruiz" || options_.linear_scaling == "ruiz_row_col";
}

bool GmresSolver::using_field_scaling() const
{
    return options_.linear_scaling == "field" || options_.linear_scaling == "field_wise";
}

bool GmresSolver::using_linear_scaling() const
{
    return using_ruiz_scaling() || using_field_scaling();
}

void GmresSolver::analyze(const CsrMatrix& matrix)
{
    validate_matrix_values(matrix);
    if (options_.max_iters <= 0 || options_.restart <= 0 ||
        options_.rel_tolerance < 0.0 || options_.abs_tolerance < 0.0) {
        throw std::runtime_error("invalid GMRES options");
    }
    if (!options_.use_right_preconditioning) {
        throw std::runtime_error("this implementation only supports right preconditioning");
    }
    if (options_.preconditioner != "none" &&
        options_.preconditioner != "metis_block_jacobi" &&
        options_.preconditioner != "metis_block_jacobi_coarse" &&
        options_.preconditioner != "ras_overlap1") {
        throw std::runtime_error("unknown GMRES preconditioner: " + options_.preconditioner);
    }
    if (options_.partition_mode != "unknown_metis" &&
        options_.partition_mode != "bus_weighted_metis") {
        throw std::runtime_error("unknown partition mode: " + options_.partition_mode);
    }
    if (options_.linear_scaling != "none" &&
        options_.linear_scaling != "ruiz" &&
        options_.linear_scaling != "ruiz_row_col" &&
        options_.linear_scaling != "field" &&
        options_.linear_scaling != "field_wise") {
        throw std::runtime_error("unknown linear scaling: " + options_.linear_scaling);
    }
    if (using_ruiz_scaling()) {
        if (!using_metis_block_jacobi() || !options_.use_mr1_fast_path) {
            throw std::runtime_error("Ruiz scaling is implemented for MR1 + METIS block-Jacobi only");
        }
        if (options_.scaling_iters <= 0 || options_.scaling_clamp < 1.0 ||
            options_.scaling_eps <= 0.0 || !std::isfinite(options_.scaling_clamp) ||
            !std::isfinite(options_.scaling_eps)) {
            throw std::runtime_error("invalid Ruiz scaling options");
        }
        if (options_.scaling_norm != "l2") {
            throw std::runtime_error("only l2 Ruiz scaling norm is implemented");
        }
    }
    if (using_field_scaling()) {
        if (!using_metis_block_jacobi()) {
            throw std::runtime_error("field scaling requires METIS block-Jacobi");
        }
        if (static_cast<int32_t>(options_.index_field.size()) != matrix.rows) {
            throw std::runtime_error("field scaling requires index field metadata");
        }
        if (options_.scaling_iters <= 0 || options_.scaling_clamp < 1.0 ||
            options_.scaling_eps <= 0.0 || !std::isfinite(options_.scaling_clamp) ||
            !std::isfinite(options_.scaling_eps)) {
            throw std::runtime_error("invalid field scaling options");
        }
    }

    matrix_ = matrix;
    d_row_ptr_.assign(matrix_.row_ptr.data(), matrix_.row_ptr.size());
    d_col_idx_.assign(matrix_.col_idx.data(), matrix_.col_idx.size());
    d_original_values_.resize(static_cast<std::size_t>(matrix_.nnz()));

    if (using_metis_block_jacobi()) {
        MetisBlockJacobiOptions precond_options;
        precond_options.block_size = options_.block_size;
        precond_options.use_fp32_preconditioner = options_.use_fp32_preconditioner;
        precond_options.apply_mode = options_.block_jacobi_apply;
        precond_options.diagonal_shift = options_.block_jacobi_diagonal_shift;
        precond_options.enable_coarse = options_.preconditioner == "metis_block_jacobi_coarse";
        precond_options.ras_overlap = options_.preconditioner == "ras_overlap1" ? 1 : 0;
        precond_options.coarse_vars_per_block = options_.coarse_vars_per_block;
        precond_options.coarse_refresh = options_.coarse_refresh;
        precond_options.coarse_precision = options_.coarse_precision;
        precond_options.coarse_diag_shift_scale = options_.coarse_diag_shift_scale;
        precond_options.partition_mode = options_.partition_mode;
        precond_options.bus_edge_weight = options_.bus_edge_weight;
        precond_options.bus_edge_weight_scale = options_.bus_edge_weight_scale;
        precond_options.bus_edge_weight_clamp = options_.bus_edge_weight_clamp;
        precond_options.target_block_unknowns = options_.target_block_unknowns;
        precond_options.n_bus = options_.n_bus;
        precond_options.index_to_bus = options_.index_to_bus;
        precond_options.index_field = options_.index_field;
        preconditioner_.analyze(matrix_, precond_options);
    }

    analyzed_ = true;
    setup_ready_ = false;
}

void GmresSolver::setup(const double* d_values)
{
    if (!analyzed_ || d_values == nullptr) {
        throw std::runtime_error("GmresSolver::setup called before analyze or with null values");
    }

    last_scaling_timings_ = {};
    dr_min_ = 1.0;
    dr_max_ = 1.0;
    dr_geomean_ = 1.0;
    dc_min_ = 1.0;
    dc_max_ = 1.0;
    dc_geomean_ = 1.0;
    row_norm_cv_before_ = 0.0;
    row_norm_cv_after_ = 0.0;
    col_norm_cv_before_ = 0.0;
    col_norm_cv_after_ = 0.0;
    if (using_metis_block_jacobi()) {
        if (using_linear_scaling()) {
            preconditioner_.refresh_permuted_values(d_values);
            const DeviceCsrMatrixView raw_matrix = preconditioner_.permuted_matrix_view();
            ensure_scaling_workspace(raw_matrix.rows, raw_matrix.nnz);
            CUITER_CUDA_CHECK(cudaMemcpy(d_unscaled_permuted_values_.data(),
                                         raw_matrix.values,
                                         static_cast<std::size_t>(raw_matrix.nnz) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
            if (using_ruiz_scaling()) {
                compute_ruiz_scaling(unscaled_permuted_matrix_view());
            } else {
                compute_field_scaling(unscaled_permuted_matrix_view());
            }
            preconditioner_.setup_permuted_values(d_scaled_values_.data());
        } else {
            preconditioner_.setup(d_values);
        }
    } else {
        const auto start = std::chrono::steady_clock::now();
        CUITER_CUDA_CHECK(cudaMemcpy(d_original_values_.data(),
                                     d_values,
                                     static_cast<std::size_t>(matrix_.nnz()) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaDeviceSynchronize());
        (void)start;
    }
    setup_ready_ = true;
}

void GmresSolver::refresh_matrix_values(const double* d_values)
{
    if (!setup_ready_ || d_values == nullptr) {
        setup(d_values);
        return;
    }
    if (using_linear_scaling()) {
        setup(d_values);
        return;
    }
    if (using_metis_block_jacobi()) {
        preconditioner_.refresh_permuted_values(d_values);
    } else {
        CUITER_CUDA_CHECK(cudaMemcpy(d_original_values_.data(),
                                     d_values,
                                     static_cast<std::size_t>(matrix_.nnz()) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }
}

void GmresSolver::ensure_workspace(int32_t n, int32_t restart)
{
    if (workspace_n_ == n && workspace_restart_ == restart) {
        return;
    }

    workspace_n_ = n;
    workspace_restart_ = restart;
    const std::size_t rows = static_cast<std::size_t>(n);
    d_rhs_work_.resize(rows);
    d_x_work_.resize(rows);
    d_r_.resize(rows);
    d_w_.resize(rows);
    d_ax_.resize(rows);
    d_v_basis_.resize(rows * static_cast<std::size_t>(restart + 1));
    d_z_basis_.resize(rows * static_cast<std::size_t>(restart));
    d_h_col_.resize(static_cast<std::size_t>(restart + 1));
    d_y_.resize(static_cast<std::size_t>(restart));
    d_mr1_dots_.resize(8);
    d_bicgstab_r_hat_.resize(rows);
    d_bicgstab_p_.resize(rows);
    d_bicgstab_p_hat_.resize(rows);
    d_bicgstab_s_.resize(rows);
    d_bicgstab_s_hat_.resize(rows);
    d_bicgstab_v_.resize(rows);
    d_bicgstab_t_.resize(rows);
    d_rhs_unscaled_.resize(rows);
    d_initial_guess_permuted_.resize(rows);

    h_col_host_.assign(static_cast<std::size_t>(restart + 1), 0.0);
    hessenberg_.assign(static_cast<std::size_t>(restart + 1) *
                           static_cast<std::size_t>(restart),
                       0.0);
    givens_c_.assign(static_cast<std::size_t>(restart), 0.0);
    givens_s_.assign(static_cast<std::size_t>(restart), 0.0);
    g_.assign(static_cast<std::size_t>(restart + 1), 0.0);
    y_host_.assign(static_cast<std::size_t>(restart), 0.0);
}

void GmresSolver::ensure_scaling_workspace(int32_t n, int32_t nnz)
{
    d_unscaled_permuted_values_.resize(static_cast<std::size_t>(nnz));
    d_scaled_values_.resize(static_cast<std::size_t>(nnz));
    d_row_scale_.resize(static_cast<std::size_t>(n));
    d_col_scale_.resize(static_cast<std::size_t>(n));
    d_row_norms_.resize(static_cast<std::size_t>(n));
    d_col_norms_.resize(static_cast<std::size_t>(n));
    d_initial_guess_permuted_.resize(static_cast<std::size_t>(n));
}

DeviceCsrMatrixView GmresSolver::active_matrix_view() const
{
    if (using_metis_block_jacobi()) {
        return preconditioner_.permuted_matrix_view();
    }
    return DeviceCsrMatrixView{matrix_.rows,
                               matrix_.cols,
                               matrix_.nnz(),
                               d_row_ptr_.data(),
                               d_col_idx_.data(),
                               d_original_values_.data()};
}

DeviceCsrMatrixView GmresSolver::unscaled_permuted_matrix_view() const
{
    return DeviceCsrMatrixView{matrix_.rows,
                               matrix_.cols,
                               matrix_.nnz(),
                               preconditioner_.permuted_matrix_view().row_ptr,
                               preconditioner_.permuted_matrix_view().col_idx,
                               d_unscaled_permuted_values_.data()};
}

void GmresSolver::compute_ruiz_scaling(const DeviceCsrMatrixView& raw_matrix)
{
    const int32_t n = raw_matrix.rows;
    kernels::launch_set_constant(n, 1.0, d_row_scale_.data());
    kernels::launch_set_constant(n, 1.0, d_col_scale_.data());

    last_scaling_timings_.scaling_row_norm_seconds += gpu_timed([&] {
        kernels::launch_compute_scaled_row_l2_norms(n,
                                                    raw_matrix.row_ptr,
                                                    raw_matrix.col_idx,
                                                    raw_matrix.values,
                                                    d_row_scale_.data(),
                                                    d_col_scale_.data(),
                                                    d_row_norms_.data());
    });
    last_scaling_timings_.scaling_col_norm_seconds += gpu_timed([&] {
        kernels::launch_compute_scaled_col_l2_norms(n,
                                                    raw_matrix.cols,
                                                    raw_matrix.row_ptr,
                                                    raw_matrix.col_idx,
                                                    raw_matrix.values,
                                                    d_row_scale_.data(),
                                                    d_col_scale_.data(),
                                                    d_col_norms_.data());
    });

    std::vector<double> row_before(static_cast<std::size_t>(n));
    std::vector<double> col_before(static_cast<std::size_t>(n));
    d_row_norms_.copy_to(row_before.data(), row_before.size());
    d_col_norms_.copy_to(col_before.data(), col_before.size());
    row_norm_cv_before_ = coefficient_of_variation(row_before);
    col_norm_cv_before_ = coefficient_of_variation(col_before);

    for (int32_t iter = 0; iter < options_.scaling_iters; ++iter) {
        last_scaling_timings_.scaling_row_norm_seconds += gpu_timed([&] {
            kernels::launch_compute_scaled_row_l2_norms(n,
                                                        raw_matrix.row_ptr,
                                                        raw_matrix.col_idx,
                                                        raw_matrix.values,
                                                        d_row_scale_.data(),
                                                        d_col_scale_.data(),
                                                        d_row_norms_.data());
            kernels::launch_update_ruiz_scale(n,
                                              d_row_norms_.data(),
                                              options_.scaling_eps,
                                              options_.scaling_clamp,
                                              d_row_scale_.data());
        });
        last_scaling_timings_.scaling_col_norm_seconds += gpu_timed([&] {
            kernels::launch_compute_scaled_col_l2_norms(n,
                                                        raw_matrix.cols,
                                                        raw_matrix.row_ptr,
                                                        raw_matrix.col_idx,
                                                        raw_matrix.values,
                                                        d_row_scale_.data(),
                                                        d_col_scale_.data(),
                                                        d_col_norms_.data());
            kernels::launch_update_ruiz_scale(n,
                                              d_col_norms_.data(),
                                              options_.scaling_eps,
                                              options_.scaling_clamp,
                                              d_col_scale_.data());
        });
    }

    last_scaling_timings_.scaling_row_norm_seconds += gpu_timed([&] {
        kernels::launch_compute_scaled_row_l2_norms(n,
                                                    raw_matrix.row_ptr,
                                                    raw_matrix.col_idx,
                                                    raw_matrix.values,
                                                    d_row_scale_.data(),
                                                    d_col_scale_.data(),
                                                    d_row_norms_.data());
    });
    last_scaling_timings_.scaling_col_norm_seconds += gpu_timed([&] {
        kernels::launch_compute_scaled_col_l2_norms(n,
                                                    raw_matrix.cols,
                                                    raw_matrix.row_ptr,
                                                    raw_matrix.col_idx,
                                                    raw_matrix.values,
                                                    d_row_scale_.data(),
                                                    d_col_scale_.data(),
                                                    d_col_norms_.data());
    });
    last_scaling_timings_.scaling_apply_values_seconds += gpu_timed([&] {
        kernels::launch_apply_scaled_csr_values(n,
                                                raw_matrix.row_ptr,
                                                raw_matrix.col_idx,
                                                raw_matrix.values,
                                                d_row_scale_.data(),
                                                d_col_scale_.data(),
                                                d_scaled_values_.data());
    });
    last_scaling_timings_.scaling_total_seconds =
        last_scaling_timings_.scaling_row_norm_seconds +
        last_scaling_timings_.scaling_col_norm_seconds +
        last_scaling_timings_.scaling_apply_values_seconds;

    std::vector<double> row_after(static_cast<std::size_t>(n));
    std::vector<double> col_after(static_cast<std::size_t>(n));
    std::vector<double> row_scale(static_cast<std::size_t>(n));
    std::vector<double> col_scale(static_cast<std::size_t>(n));
    d_row_norms_.copy_to(row_after.data(), row_after.size());
    d_col_norms_.copy_to(col_after.data(), col_after.size());
    d_row_scale_.copy_to(row_scale.data(), row_scale.size());
    d_col_scale_.copy_to(col_scale.data(), col_scale.size());
    row_norm_cv_after_ = coefficient_of_variation(row_after);
    col_norm_cv_after_ = coefficient_of_variation(col_after);
    scale_stats(row_scale, dr_min_, dr_max_, dr_geomean_);
    scale_stats(col_scale, dc_min_, dc_max_, dc_geomean_);
}

void GmresSolver::compute_field_scaling(const DeviceCsrMatrixView& raw_matrix)
{
    const int32_t n = raw_matrix.rows;
    std::vector<int32_t> row_ptr(static_cast<std::size_t>(n + 1));
    std::vector<int32_t> col_idx(static_cast<std::size_t>(raw_matrix.nnz));
    std::vector<double> values(static_cast<std::size_t>(raw_matrix.nnz));
    CUITER_CUDA_CHECK(cudaMemcpy(row_ptr.data(),
                                 raw_matrix.row_ptr,
                                 row_ptr.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost));
    CUITER_CUDA_CHECK(cudaMemcpy(col_idx.data(),
                                 raw_matrix.col_idx,
                                 col_idx.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost));
    CUITER_CUDA_CHECK(cudaMemcpy(values.data(),
                                 raw_matrix.values,
                                 values.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost));

    std::vector<int32_t> field(static_cast<std::size_t>(n), 0);
    const auto& new_to_old = preconditioner_.permutation().new_to_old;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t old_index = new_to_old[static_cast<std::size_t>(i)];
        int32_t value = 0;
        if (old_index >= 0 && static_cast<std::size_t>(old_index) < options_.index_field.size()) {
            value = options_.index_field[static_cast<std::size_t>(old_index)];
        }
        field[static_cast<std::size_t>(i)] = value == 1 ? 1 : 0;
    }

    auto clamp_scale = [&](double value) {
        const double lo = 1.0 / options_.scaling_clamp;
        const double hi = options_.scaling_clamp;
        return std::min(hi, std::max(lo, value));
    };

    auto compute_norm_vectors = [&](const std::array<double, 2>& dr,
                                    const std::array<double, 2>& dc,
                                    std::vector<double>& row_norms,
                                    std::vector<double>& col_norms) {
        std::fill(row_norms.begin(), row_norms.end(), 0.0);
        std::fill(col_norms.begin(), col_norms.end(), 0.0);
        for (int32_t row = 0; row < n; ++row) {
            const int32_t rf = field[static_cast<std::size_t>(row)];
            double row_sum = 0.0;
            for (int32_t pos = row_ptr[static_cast<std::size_t>(row)];
                 pos < row_ptr[static_cast<std::size_t>(row + 1)];
                 ++pos) {
                const int32_t col = col_idx[static_cast<std::size_t>(pos)];
                const int32_t cf = field[static_cast<std::size_t>(col)];
                const double scaled = dr[static_cast<std::size_t>(rf)] *
                                      values[static_cast<std::size_t>(pos)] *
                                      dc[static_cast<std::size_t>(cf)];
                const double square = scaled * scaled;
                row_sum += square;
                col_norms[static_cast<std::size_t>(col)] += square;
            }
            row_norms[static_cast<std::size_t>(row)] = std::sqrt(row_sum);
        }
        for (double& norm : col_norms) {
            norm = std::sqrt(norm);
        }
    };

    std::array<int32_t, 2> field_counts = {0, 0};
    for (int32_t value : field) {
        ++field_counts[static_cast<std::size_t>(value == 1 ? 1 : 0)];
    }

    std::array<double, 2> dr = {1.0, 1.0};
    std::array<double, 2> dc = {1.0, 1.0};
    std::vector<double> row_norms(static_cast<std::size_t>(n), 0.0);
    std::vector<double> col_norms(static_cast<std::size_t>(n), 0.0);
    compute_norm_vectors(dr, dc, row_norms, col_norms);
    row_norm_cv_before_ = coefficient_of_variation(row_norms);
    col_norm_cv_before_ = coefficient_of_variation(col_norms);

    const auto scaling_start = std::chrono::steady_clock::now();
    for (int32_t iter = 0; iter < options_.scaling_iters; ++iter) {
        compute_norm_vectors(dr, dc, row_norms, col_norms);
        std::array<double, 2> row_square_sum = {0.0, 0.0};
        for (int32_t row = 0; row < n; ++row) {
            const int32_t rf = field[static_cast<std::size_t>(row)];
            const double norm = row_norms[static_cast<std::size_t>(row)];
            row_square_sum[static_cast<std::size_t>(rf)] += norm * norm;
        }
        for (int32_t f = 0; f < 2; ++f) {
            const double rms = std::sqrt(
                row_square_sum[static_cast<std::size_t>(f)] /
                static_cast<double>(std::max(1, field_counts[static_cast<std::size_t>(f)])));
            dr[static_cast<std::size_t>(f)] = clamp_scale(
                dr[static_cast<std::size_t>(f)] /
                std::sqrt(std::max(rms, options_.scaling_eps)));
        }

        compute_norm_vectors(dr, dc, row_norms, col_norms);
        std::array<double, 2> col_square_sum = {0.0, 0.0};
        for (int32_t col = 0; col < n; ++col) {
            const int32_t cf = field[static_cast<std::size_t>(col)];
            const double norm = col_norms[static_cast<std::size_t>(col)];
            col_square_sum[static_cast<std::size_t>(cf)] += norm * norm;
        }
        for (int32_t f = 0; f < 2; ++f) {
            const double rms = std::sqrt(
                col_square_sum[static_cast<std::size_t>(f)] /
                static_cast<double>(std::max(1, field_counts[static_cast<std::size_t>(f)])));
            dc[static_cast<std::size_t>(f)] = clamp_scale(
                dc[static_cast<std::size_t>(f)] /
                std::sqrt(std::max(rms, options_.scaling_eps)));
        }
    }
    compute_norm_vectors(dr, dc, row_norms, col_norms);
    row_norm_cv_after_ = coefficient_of_variation(row_norms);
    col_norm_cv_after_ = coefficient_of_variation(col_norms);

    std::vector<double> row_scale(static_cast<std::size_t>(n), 1.0);
    std::vector<double> col_scale(static_cast<std::size_t>(n), 1.0);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t f = field[static_cast<std::size_t>(i)];
        row_scale[static_cast<std::size_t>(i)] = dr[static_cast<std::size_t>(f)];
        col_scale[static_cast<std::size_t>(i)] = dc[static_cast<std::size_t>(f)];
    }
    d_row_scale_.assign(row_scale.data(), row_scale.size());
    d_col_scale_.assign(col_scale.data(), col_scale.size());
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    last_scaling_timings_.scaling_row_norm_seconds =
        0.5 * std::chrono::duration<double>(std::chrono::steady_clock::now() - scaling_start).count();
    last_scaling_timings_.scaling_col_norm_seconds =
        last_scaling_timings_.scaling_row_norm_seconds;

    last_scaling_timings_.scaling_apply_values_seconds += gpu_timed([&] {
        kernels::launch_apply_scaled_csr_values(n,
                                                raw_matrix.row_ptr,
                                                raw_matrix.col_idx,
                                                raw_matrix.values,
                                                d_row_scale_.data(),
                                                d_col_scale_.data(),
                                                d_scaled_values_.data());
    });
    last_scaling_timings_.scaling_total_seconds =
        last_scaling_timings_.scaling_row_norm_seconds +
        last_scaling_timings_.scaling_col_norm_seconds +
        last_scaling_timings_.scaling_apply_values_seconds;

    scale_stats(row_scale, dr_min_, dr_max_, dr_geomean_);
    scale_stats(col_scale, dc_min_, dc_max_, dc_geomean_);
}

void GmresSolver::attach_scaling_result_metadata(LinearSolveResult& result) const
{
    result.timings.scaling_row_norm_seconds = last_scaling_timings_.scaling_row_norm_seconds;
    result.timings.scaling_col_norm_seconds = last_scaling_timings_.scaling_col_norm_seconds;
    result.timings.scaling_apply_values_seconds =
        last_scaling_timings_.scaling_apply_values_seconds;
    result.timings.scaling_total_seconds = last_scaling_timings_.scaling_total_seconds +
                                           result.timings.scaling_apply_rhs_seconds;
    result.dr_min = dr_min_;
    result.dr_max = dr_max_;
    result.dr_geomean = dr_geomean_;
    result.dc_min = dc_min_;
    result.dc_max = dc_max_;
    result.dc_geomean = dc_geomean_;
    result.row_norm_cv_before = row_norm_cv_before_;
    result.row_norm_cv_after = row_norm_cv_after_;
    result.col_norm_cv_before = col_norm_cv_before_;
    result.col_norm_cv_after = col_norm_cv_after_;
}

double GmresSolver::norm_device(int32_t n, const double* d_x)
{
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_x, 1, d_h_col_.data()));
    double value = 0.0;
    CUITER_CUDA_CHECK(cudaMemcpy(&value, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return value;
}

double GmresSolver::compute_residual(const DeviceCsrMatrixView& matrix,
                                     int32_t n,
                                     const double* d_rhs,
                                     const double* d_x,
                                     double rhs_norm,
                                     LinearSolveResult& result)
{
    double norm = 0.0;
    result.timings.final_residual_seconds += gpu_timed([&] {
        kernels::launch_csr_spmv(matrix.rows,
                                 matrix.row_ptr,
                                 matrix.col_idx,
                                 matrix.values,
                                 d_x,
                                 d_ax_.data());
        kernels::launch_residual(n, d_rhs, d_ax_.data(), d_r_.data());
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
    });
    const auto copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    const auto copy_stop = std::chrono::steady_clock::now();
    result.timings.dot_reduction_seconds +=
        std::chrono::duration<double>(copy_stop - copy_start).count();
    result.residual_norm2 = norm;
    result.relative_residual_norm2 = norm / rhs_norm;
    return norm;
}

void GmresSolver::update_solution(int32_t n, int32_t basis_count, double* d_x)
{
    d_y_.assign(y_host_.data(), static_cast<std::size_t>(basis_count));
    kernels::launch_combine_solution(n, basis_count, d_z_basis_.data(), d_y_.data(), d_x);
}

bool GmresSolver::solve_small_upper(int32_t basis_count)
{
    std::fill(y_host_.begin(), y_host_.end(), 0.0);
    for (int32_t row = basis_count - 1; row >= 0; --row) {
        double rhs = g_[static_cast<std::size_t>(row)];
        for (int32_t col = row + 1; col < basis_count; ++col) {
            rhs -= hessenberg_[static_cast<std::size_t>(h_index(row, col))] *
                   y_host_[static_cast<std::size_t>(col)];
        }

        const double diag = hessenberg_[static_cast<std::size_t>(h_index(row, row))];
        if (!std::isfinite(diag) || std::abs(diag) <= std::numeric_limits<double>::min()) {
            return false;
        }
        y_host_[static_cast<std::size_t>(row)] = rhs / diag;
    }
    return true;
}

double GmresSolver::apply_givens_and_residual(int32_t column, double rhs_norm)
{
    for (int32_t row = 0; row < column; ++row) {
        const int32_t idx0 = h_index(row, column);
        const int32_t idx1 = h_index(row + 1, column);
        const double h0 = hessenberg_[static_cast<std::size_t>(idx0)];
        const double h1 = hessenberg_[static_cast<std::size_t>(idx1)];
        const double c = givens_c_[static_cast<std::size_t>(row)];
        const double s = givens_s_[static_cast<std::size_t>(row)];
        hessenberg_[static_cast<std::size_t>(idx0)] = c * h0 + s * h1;
        hessenberg_[static_cast<std::size_t>(idx1)] = -s * h0 + c * h1;
    }

    const int32_t diag_idx = h_index(column, column);
    const int32_t subdiag_idx = h_index(column + 1, column);
    const double diag = hessenberg_[static_cast<std::size_t>(diag_idx)];
    const double subdiag = hessenberg_[static_cast<std::size_t>(subdiag_idx)];
    const double radius = std::hypot(diag, subdiag);
    double c = 1.0;
    double s = 0.0;
    if (radius > std::numeric_limits<double>::min()) {
        c = diag / radius;
        s = subdiag / radius;
    }
    givens_c_[static_cast<std::size_t>(column)] = c;
    givens_s_[static_cast<std::size_t>(column)] = s;
    hessenberg_[static_cast<std::size_t>(diag_idx)] = c * diag + s * subdiag;
    hessenberg_[static_cast<std::size_t>(subdiag_idx)] = 0.0;

    const double old_g = g_[static_cast<std::size_t>(column)];
    g_[static_cast<std::size_t>(column)] = c * old_g;
    g_[static_cast<std::size_t>(column + 1)] = -s * old_g;
    return std::abs(g_[static_cast<std::size_t>(column + 1)]) / rhs_norm;
}

int32_t GmresSolver::h_index(int32_t row, int32_t col) const
{
    return col * (workspace_restart_ + 1) + row;
}

LinearSolveResult GmresSolver::solve(const std::vector<double>& values,
                                     const std::vector<double>& rhs)
{
    if (!analyzed_) {
        throw std::runtime_error("GmresSolver::solve called before analyze");
    }
    if (static_cast<int32_t>(values.size()) != matrix_.nnz() ||
        static_cast<int32_t>(rhs.size()) != matrix_.rows) {
        throw std::runtime_error("GmresSolver::solve received mismatched host vector sizes");
    }

    DeviceBuffer<double> d_values(values.size());
    DeviceBuffer<double> d_rhs(rhs.size());
    DeviceBuffer<double> d_x(rhs.size());
    d_values.assign(values.data(), values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    setup(d_values.data());
    LinearSolveResult result = solve_device(d_values.data(), d_rhs.data(), d_x.data());
    result.solution.resize(rhs.size());
    d_x.copy_to(result.solution.data(), result.solution.size());
    return result;
}

LinearSolveResult GmresSolver::solve_mr1_device(const DeviceCsrMatrixView& matrix,
                                                const double* d_rhs,
                                                double* d_x,
                                                std::chrono::steady_clock::time_point solve_start)
{
    const int32_t n = matrix_.rows;
    const bool scaled = using_ruiz_scaling();
    LinearSolveResult result;
    if (using_metis_block_jacobi()) {
        const auto& setup_timings = preconditioner_.timings();
        result.timings.metis_partition_seconds = setup_timings.metis_partition_seconds;
        result.timings.permutation_build_seconds = setup_timings.permutation_build_seconds;
        result.timings.weighted_graph_build_seconds = setup_timings.weighted_graph_build_seconds;
        result.timings.block_extract_seconds = setup_timings.block_extract_seconds;
        result.timings.block_lu_seconds = setup_timings.block_lu_seconds;
        result.timings.ras_symbolic_seconds = setup_timings.ras_symbolic_seconds;
        result.timings.ras_setup_seconds = setup_timings.ras_setup_seconds;
        result.timings.setup_total_seconds = setup_timings.setup_total_seconds;
        result.block_stats = preconditioner_.block_stats();
    }

    result.timings.rhs_permute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_permute_vector(n,
                                           preconditioner_.d_new_to_old(),
                                           d_rhs,
                                           d_rhs_unscaled_.data());
        } else {
            kernels::launch_copy(n, d_rhs, d_rhs_unscaled_.data());
        }
        kernels::launch_set_zero(n, d_x_work_.data());
    });
    if (scaled) {
        result.timings.scaling_apply_rhs_seconds += gpu_timed([&] {
            kernels::launch_multiply_by_scale(
                n, d_row_scale_.data(), d_rhs_unscaled_.data(), d_rhs_work_.data());
        });
    } else {
        result.timings.rhs_permute_seconds += gpu_timed([&] {
            kernels::launch_copy(n, d_rhs_unscaled_.data(), d_rhs_work_.data());
        });
    }
    attach_scaling_result_metadata(result);

    double rhs_norm = 0.0;
    result.timings.dot_reduction_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_rhs_work_.data(), 1, d_h_col_.data()));
    });
    const auto rhs_copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&rhs_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    result.timings.bicgstab_scalar_sync_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_copy_start).count();
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());

    result.timings.preconditioner_apply_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            preconditioner_.apply(d_rhs_work_.data(), d_z_basis_.data());
        } else {
            kernels::launch_copy(n, d_rhs_work_.data(), d_z_basis_.data());
        }
    });
    if (using_metis_block_jacobi()) {
        const auto& apply_timings = preconditioner_.last_apply_timings();
        result.timings.block_jacobi_apply_seconds = apply_timings.block_jacobi_apply_seconds;
        result.timings.coarse_az0_spmv_seconds = apply_timings.coarse_az0_spmv_seconds;
        result.timings.coarse_compress_seconds = apply_timings.coarse_compress_seconds;
        result.timings.coarse_solve_seconds = apply_timings.coarse_solve_seconds;
        result.timings.coarse_expand_seconds = apply_timings.coarse_expand_seconds;
        result.timings.coarse_total_seconds = apply_timings.coarse_total_seconds;
        result.timings.ras_apply_seconds = apply_timings.ras_apply_seconds;
        result.timings.ras_gather_seconds = apply_timings.ras_gather_seconds;
        result.timings.ras_local_gemv_seconds = apply_timings.ras_local_gemv_seconds;
        result.timings.ras_scatter_seconds = apply_timings.ras_scatter_seconds;
        result.timings.preconditioner_total_seconds = apply_timings.preconditioner_total_seconds;
        result.timings.coarse_failed = apply_timings.coarse_failed;
    }

    result.timings.spmv_seconds += gpu_timed([&] {
        kernels::launch_csr_spmv(matrix.rows,
                                 matrix.row_ptr,
                                 matrix.col_idx,
                                 matrix.values,
                                 d_z_basis_.data(),
                                 d_w_.data());
    });
    result.timings.mr1_spmv_seconds = result.timings.spmv_seconds;

    double dots[2] = {0.0, 0.0};
    double fused_dot_seconds = gpu_timed([&] {
        kernels::launch_mr1_two_dot_reduction(
            n, d_w_.data(), d_rhs_work_.data(), d_mr1_dots_.data());
    });
    result.timings.dot_reduction_seconds += fused_dot_seconds;
    const auto dot_copy_start = std::chrono::steady_clock::now();
    d_mr1_dots_.copy_to(dots, 2);
    const auto dot_copy_stop = std::chrono::steady_clock::now();
    const double dot_copy_seconds =
        std::chrono::duration<double>(dot_copy_stop - dot_copy_start).count();
    result.timings.dot_reduction_seconds += dot_copy_seconds;
    result.timings.mr1_fused_dot_seconds = fused_dot_seconds + dot_copy_seconds;

    const double denominator = dots[1];
    if (!std::isfinite(dots[0]) || !std::isfinite(denominator) ||
        denominator <= std::numeric_limits<double>::min()) {
        result.stop_reason = "mr1_dot_breakdown";
        result.residual_norm2 = rhs_norm;
        result.relative_residual_norm2 = 1.0;
    } else {
        const double alpha = dots[0] / denominator;
        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_scale_copy(n, alpha, d_z_basis_.data(), d_x_work_.data());
            kernels::launch_residual_scaled(n,
                                            d_rhs_work_.data(),
                                            d_w_.data(),
                                            alpha,
                                            d_r_.data());
        });
        result.timings.mr1_update_seconds = result.timings.solution_update_seconds;

        double residual_norm = 0.0;
        result.timings.final_residual_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
        });
        CUITER_CUDA_CHECK(cudaMemcpy(&residual_norm,
                                     d_h_col_.data(),
                                     sizeof(double),
                                     cudaMemcpyDeviceToHost));
        result.scaled_residual_norm2 = residual_norm;
        result.scaled_relative_residual_norm2 = residual_norm / rhs_norm;
        if (scaled) {
            result.timings.solution_update_seconds += gpu_timed([&] {
                kernels::launch_multiply_by_scale(
                    n, d_col_scale_.data(), d_x_work_.data(), d_x_work_.data());
            });
            double unscaled_rhs_norm = 0.0;
            result.timings.final_residual_seconds += gpu_timed([&] {
                CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_,
                                               n,
                                               d_rhs_unscaled_.data(),
                                               1,
                                               d_h_col_.data()));
            });
            CUITER_CUDA_CHECK(cudaMemcpy(&unscaled_rhs_norm,
                                         d_h_col_.data(),
                                         sizeof(double),
                                         cudaMemcpyDeviceToHost));
            unscaled_rhs_norm = std::max(unscaled_rhs_norm, std::numeric_limits<double>::min());
            const DeviceCsrMatrixView raw_matrix = unscaled_permuted_matrix_view();
            result.timings.final_residual_seconds += gpu_timed([&] {
                kernels::launch_csr_spmv(raw_matrix.rows,
                                         raw_matrix.row_ptr,
                                         raw_matrix.col_idx,
                                         raw_matrix.values,
                                         d_x_work_.data(),
                                         d_ax_.data());
                kernels::launch_residual(n, d_rhs_unscaled_.data(), d_ax_.data(), d_r_.data());
                CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
            });
            double unscaled_residual_norm = 0.0;
            CUITER_CUDA_CHECK(cudaMemcpy(&unscaled_residual_norm,
                                         d_h_col_.data(),
                                         sizeof(double),
                                         cudaMemcpyDeviceToHost));
            result.unscaled_residual_norm2 = unscaled_residual_norm;
            result.unscaled_relative_residual_norm2 = unscaled_residual_norm / unscaled_rhs_norm;
            result.residual_norm2 = result.unscaled_residual_norm2;
            result.relative_residual_norm2 = result.unscaled_relative_residual_norm2;
        } else {
            result.unscaled_residual_norm2 = residual_norm;
            result.unscaled_relative_residual_norm2 = residual_norm / rhs_norm;
            result.residual_norm2 = residual_norm;
            result.relative_residual_norm2 = residual_norm / rhs_norm;
        }
        result.iterations = 1;
        const double target_norm = std::max(options_.abs_tolerance, options_.rel_tolerance * rhs_norm);
        result.converged = result.scaled_residual_norm2 <= target_norm;
        result.stop_reason = result.converged ? "mr1_converged" : "mr1_fixed_iter";
    }

    if (!result.stop_reason.empty() && result.iterations == 0) {
        result.scaled_residual_norm2 = rhs_norm;
        result.scaled_relative_residual_norm2 = 1.0;
        if (scaled) {
            double unscaled_rhs_norm = 0.0;
            result.timings.final_residual_seconds += gpu_timed([&] {
                CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_,
                                               n,
                                               d_rhs_unscaled_.data(),
                                               1,
                                               d_h_col_.data()));
            });
            CUITER_CUDA_CHECK(cudaMemcpy(&unscaled_rhs_norm,
                                         d_h_col_.data(),
                                         sizeof(double),
                                         cudaMemcpyDeviceToHost));
            result.unscaled_residual_norm2 = unscaled_rhs_norm;
            result.unscaled_relative_residual_norm2 = 1.0;
        } else {
            result.unscaled_residual_norm2 = rhs_norm;
            result.unscaled_relative_residual_norm2 = 1.0;
        }
        result.residual_norm2 = result.unscaled_residual_norm2;
        result.relative_residual_norm2 = result.unscaled_relative_residual_norm2;
    }

    result.timings.unpermute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_unpermute_vector(n, preconditioner_.d_new_to_old(), d_x_work_.data(), d_x);
        } else {
            kernels::launch_copy(n, d_x_work_.data(), d_x);
        }
    });
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.middle_solver_total_seconds =
        result.timings.preconditioner_apply_seconds +
        result.timings.spmv_seconds +
        result.timings.dot_reduction_seconds +
        result.timings.solution_update_seconds;
    result.timings.gmres_loop_seconds = result.timings.middle_solver_total_seconds;
    result.timings.solve_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return result;
}

LinearSolveResult GmresSolver::solve_mr2_device(const DeviceCsrMatrixView& matrix,
                                                const double* d_rhs,
                                                double* d_x,
                                                std::chrono::steady_clock::time_point solve_start)
{
    const int32_t n = matrix_.rows;
    LinearSolveResult result;
    if (!using_metis_block_jacobi() || options_.preconditioner != "metis_block_jacobi_coarse") {
        result.stop_reason = "mr2_requires_metis_block_jacobi_coarse";
        return result;
    }

    const auto& setup_timings = preconditioner_.timings();
    result.timings.metis_partition_seconds = setup_timings.metis_partition_seconds;
    result.timings.permutation_build_seconds = setup_timings.permutation_build_seconds;
    result.timings.weighted_graph_build_seconds = setup_timings.weighted_graph_build_seconds;
    result.timings.block_extract_seconds = setup_timings.block_extract_seconds;
    result.timings.block_lu_seconds = setup_timings.block_lu_seconds;
    result.timings.setup_total_seconds = setup_timings.setup_total_seconds;
    result.block_stats = preconditioner_.block_stats();

    double* d_z0 = d_z_basis_.data();
    double* d_z1 = d_v_basis_.data();
    double* d_w0 = d_w_.data();
    double* d_w1 = d_ax_.data();

    result.timings.rhs_permute_seconds += gpu_timed([&] {
        kernels::launch_permute_vector(n, preconditioner_.d_new_to_old(), d_rhs, d_rhs_work_.data());
        kernels::launch_set_zero(n, d_x_work_.data());
    });

    double rhs_norm = 0.0;
    result.timings.dot_reduction_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_rhs_work_.data(), 1, d_h_col_.data()));
    });
    const auto rhs_copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&rhs_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    result.timings.bicgstab_scalar_sync_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_copy_start).count();
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());

    result.timings.block_jacobi_apply_seconds += gpu_timed([&] {
        preconditioner_.apply_local(d_rhs_work_.data(), d_z0);
    });

    result.timings.coarse_az0_spmv_seconds += gpu_timed([&] {
        kernels::launch_csr_spmv(matrix.rows,
                                 matrix.row_ptr,
                                 matrix.col_idx,
                                 matrix.values,
                                 d_z0,
                                 d_w0);
    });
    result.timings.solution_update_seconds += gpu_timed([&] {
        kernels::launch_residual(n, d_rhs_work_.data(), d_w0, d_r_.data());
    });

    preconditioner_.apply_coarse_correction(d_r_.data(), d_z1);
    const auto& coarse_timings = preconditioner_.last_apply_timings();
    result.timings.coarse_compress_seconds = coarse_timings.coarse_compress_seconds;
    result.timings.coarse_solve_seconds = coarse_timings.coarse_solve_seconds;
    result.timings.coarse_expand_seconds = coarse_timings.coarse_expand_seconds;
    result.timings.coarse_total_seconds = coarse_timings.coarse_total_seconds;
    result.timings.coarse_failed = coarse_timings.coarse_failed;
    result.timings.preconditioner_total_seconds =
        result.timings.block_jacobi_apply_seconds + coarse_timings.preconditioner_total_seconds;
    result.timings.preconditioner_apply_seconds = result.timings.preconditioner_total_seconds;

    result.timings.mr2_w1_spmv_seconds += gpu_timed([&] {
        kernels::launch_csr_spmv(matrix.rows,
                                 matrix.row_ptr,
                                 matrix.col_idx,
                                 matrix.values,
                                 d_z1,
                                 d_w1);
    });
    result.timings.spmv_seconds =
        result.timings.coarse_az0_spmv_seconds + result.timings.mr2_w1_spmv_seconds;
    result.timings.mr1_spmv_seconds = result.timings.spmv_seconds;

    double dots[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double fused_dot_seconds = gpu_timed([&] {
        kernels::launch_mr2_five_dot_reduction(
            n, d_w0, d_w1, d_rhs_work_.data(), d_mr1_dots_.data());
    });
    result.timings.dot_reduction_seconds += fused_dot_seconds;
    const auto dot_copy_start = std::chrono::steady_clock::now();
    d_mr1_dots_.copy_to(dots, 5);
    const auto dot_copy_stop = std::chrono::steady_clock::now();
    const double dot_copy_seconds =
        std::chrono::duration<double>(dot_copy_stop - dot_copy_start).count();
    result.timings.dot_reduction_seconds += dot_copy_seconds;
    result.timings.mr1_fused_dot_seconds = fused_dot_seconds + dot_copy_seconds;

    const double b0 = dots[0];
    const double b1 = dots[1];
    const double g00 = dots[2];
    const double g01 = dots[3];
    const double g11 = dots[4];
    const double det = g00 * g11 - g01 * g01;
    double alpha0 = 0.0;
    double alpha1 = 0.0;
    bool degenerate = false;
    if (!std::isfinite(det) ||
        std::abs(det) <= 1.0e-14 * std::max(g00 * g11, std::numeric_limits<double>::min())) {
        degenerate = true;
        if (std::isfinite(g00) && g00 > std::numeric_limits<double>::min()) {
            alpha0 = b0 / g00;
        } else {
            result.stop_reason = "mr2_dot_breakdown";
            result.residual_norm2 = rhs_norm;
            result.relative_residual_norm2 = 1.0;
        }
    } else {
        alpha0 = (b0 * g11 - b1 * g01) / det;
        alpha1 = (g00 * b1 - g01 * b0) / det;
    }

    if (result.stop_reason.empty()) {
        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_linear_combination2(n, alpha0, d_z0, alpha1, d_z1, d_x_work_.data());
            kernels::launch_residual_two_scaled(
                n, d_rhs_work_.data(), d_w0, alpha0, d_w1, alpha1, d_r_.data());
        });
        result.timings.mr1_update_seconds = result.timings.solution_update_seconds;

        double residual_norm = 0.0;
        result.timings.final_residual_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
        });
        CUITER_CUDA_CHECK(cudaMemcpy(&residual_norm,
                                     d_h_col_.data(),
                                     sizeof(double),
                                     cudaMemcpyDeviceToHost));
        result.residual_norm2 = residual_norm;
        result.relative_residual_norm2 = residual_norm / rhs_norm;
        result.iterations = 1;
        const double target_norm = std::max(options_.abs_tolerance, options_.rel_tolerance * rhs_norm);
        result.converged = residual_norm <= target_norm;
        if (degenerate) {
            result.stop_reason = result.converged ? "mr2_degenerate_converged" :
                                                    "mr2_degenerate_to_local";
        } else {
            result.stop_reason = result.converged ? "mr2_converged" : "mr2_fixed_iter";
        }
    }

    result.timings.unpermute_seconds += gpu_timed([&] {
        kernels::launch_unpermute_vector(n, preconditioner_.d_new_to_old(), d_x_work_.data(), d_x);
    });
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.middle_solver_total_seconds =
        result.timings.preconditioner_apply_seconds +
        result.timings.spmv_seconds +
        result.timings.dot_reduction_seconds +
        result.timings.solution_update_seconds;
    result.timings.gmres_loop_seconds = result.timings.middle_solver_total_seconds;
    result.timings.solve_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return result;
}

LinearSolveResult GmresSolver::solve_bicgstab_device(
    const DeviceCsrMatrixView& matrix,
    const double* d_rhs,
    double* d_x,
    std::chrono::steady_clock::time_point solve_start)
{
    const int32_t n = matrix_.rows;
    const bool scaled = using_field_scaling();
    LinearSolveResult result;
    if (using_ruiz_scaling()) {
        result.stop_reason = "bicgstab_rejects_ruiz_scaling";
        return result;
    }
    if (using_metis_block_jacobi()) {
        const auto& setup_timings = preconditioner_.timings();
        result.timings.metis_partition_seconds = setup_timings.metis_partition_seconds;
        result.timings.permutation_build_seconds = setup_timings.permutation_build_seconds;
        result.timings.weighted_graph_build_seconds = setup_timings.weighted_graph_build_seconds;
        result.timings.block_extract_seconds = setup_timings.block_extract_seconds;
        result.timings.block_lu_seconds = setup_timings.block_lu_seconds;
        result.timings.ras_symbolic_seconds = setup_timings.ras_symbolic_seconds;
        result.timings.ras_setup_seconds = setup_timings.ras_setup_seconds;
        result.timings.setup_total_seconds = setup_timings.setup_total_seconds;
        result.block_stats = preconditioner_.block_stats();
    }

    result.timings.rhs_permute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_permute_vector(
                n, preconditioner_.d_new_to_old(), d_rhs, d_rhs_unscaled_.data());
        } else {
            kernels::launch_copy(n, d_rhs, d_rhs_unscaled_.data());
        }
        if (scaled) {
            kernels::launch_multiply_by_scale(
                n, d_row_scale_.data(), d_rhs_unscaled_.data(), d_rhs_work_.data());
        } else {
            kernels::launch_copy(n, d_rhs_unscaled_.data(), d_rhs_work_.data());
        }
        kernels::launch_set_zero(n, d_bicgstab_v_.data());
        kernels::launch_set_zero(n, d_bicgstab_p_.data());
        if (options_.use_initial_guess) {
            if (using_metis_block_jacobi()) {
                kernels::launch_permute_vector(
                    n, preconditioner_.d_new_to_old(), d_x, d_initial_guess_permuted_.data());
            } else {
                kernels::launch_copy(n, d_x, d_initial_guess_permuted_.data());
            }
            kernels::launch_copy(n, d_initial_guess_permuted_.data(), d_x_work_.data());
        } else {
            kernels::launch_set_zero(n, d_x_work_.data());
        }
    });
    if (scaled) {
        attach_scaling_result_metadata(result);
    }

    double rhs_norm = 0.0;
    result.timings.dot_reduction_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_rhs_work_.data(), 1, d_h_col_.data()));
    });
    const auto rhs_copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&rhs_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    result.timings.bicgstab_scalar_sync_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - rhs_copy_start).count();
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());

    result.timings.solution_update_seconds += gpu_timed([&] {
        if (options_.use_initial_guess) {
            kernels::launch_csr_spmv(matrix.rows,
                                     matrix.row_ptr,
                                     matrix.col_idx,
                                     matrix.values,
                                     d_x_work_.data(),
                                     d_ax_.data());
            kernels::launch_residual(n, d_rhs_work_.data(), d_ax_.data(), d_r_.data());
        } else {
            kernels::launch_copy(n, d_rhs_work_.data(), d_r_.data());
        }
        kernels::launch_copy(n, d_rhs_work_.data(), d_bicgstab_r_hat_.data());
    });

    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    const double target_norm = std::max(options_.abs_tolerance, options_.rel_tolerance * rhs_norm);
    const auto add_apply_timings = [&] {
        const auto& apply_timings = preconditioner_.last_apply_timings();
        result.timings.block_jacobi_apply_seconds += apply_timings.block_jacobi_apply_seconds;
        result.timings.coarse_az0_spmv_seconds += apply_timings.coarse_az0_spmv_seconds;
        result.timings.coarse_compress_seconds += apply_timings.coarse_compress_seconds;
        result.timings.coarse_solve_seconds += apply_timings.coarse_solve_seconds;
        result.timings.coarse_expand_seconds += apply_timings.coarse_expand_seconds;
        result.timings.coarse_total_seconds += apply_timings.coarse_total_seconds;
        result.timings.ras_apply_seconds += apply_timings.ras_apply_seconds;
        result.timings.ras_gather_seconds += apply_timings.ras_gather_seconds;
        result.timings.ras_local_gemv_seconds += apply_timings.ras_local_gemv_seconds;
        result.timings.ras_scatter_seconds += apply_timings.ras_scatter_seconds;
        result.timings.preconditioner_total_seconds += apply_timings.preconditioner_total_seconds;
        result.timings.coarse_failed = result.timings.coarse_failed || apply_timings.coarse_failed;
    };

    for (int32_t iter = 0; iter < options_.max_iters; ++iter) {
        double rho = 0.0;
        result.timings.dot_reduction_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas_,
                                           n,
                                           d_bicgstab_r_hat_.data(),
                                           1,
                                           d_r_.data(),
                                           1,
                                           d_h_col_.data()));
        });
        const auto rho_copy_start = std::chrono::steady_clock::now();
        CUITER_CUDA_CHECK(cudaMemcpy(&rho, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
        const double rho_copy_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - rho_copy_start).count();
        result.timings.dot_reduction_seconds += rho_copy_seconds;
        result.timings.bicgstab_scalar_sync_seconds += rho_copy_seconds;
        if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
            result.stop_reason = "bicgstab_rho_breakdown";
            break;
        }

        result.timings.solution_update_seconds += gpu_timed([&] {
            if (iter == 0) {
                kernels::launch_copy(n, d_r_.data(), d_bicgstab_p_.data());
            } else {
                const double beta = (rho / rho_old) * (alpha / omega);
                kernels::launch_bicgstab_update_p(
                    n, d_r_.data(), beta, omega, d_bicgstab_v_.data(), d_bicgstab_p_.data());
            }
        });

        result.timings.preconditioner_apply_seconds += gpu_timed([&] {
            if (using_metis_block_jacobi()) {
                preconditioner_.apply(d_bicgstab_p_.data(), d_bicgstab_p_hat_.data());
            } else {
                kernels::launch_copy(n, d_bicgstab_p_.data(), d_bicgstab_p_hat_.data());
            }
        });
        if (using_metis_block_jacobi()) {
            add_apply_timings();
        }

        result.timings.spmv_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(matrix.rows,
                                     matrix.row_ptr,
                                     matrix.col_idx,
                                     matrix.values,
                                     d_bicgstab_p_hat_.data(),
                                     d_bicgstab_v_.data());
        });

        double rhat_v = 0.0;
        result.timings.dot_reduction_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas_,
                                           n,
                                           d_bicgstab_r_hat_.data(),
                                           1,
                                           d_bicgstab_v_.data(),
                                           1,
                                           d_h_col_.data()));
        });
        const auto rv_copy_start = std::chrono::steady_clock::now();
        CUITER_CUDA_CHECK(cudaMemcpy(&rhat_v, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
        const double rv_copy_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - rv_copy_start).count();
        result.timings.dot_reduction_seconds += rv_copy_seconds;
        result.timings.bicgstab_scalar_sync_seconds += rv_copy_seconds;
        if (!std::isfinite(rhat_v) || std::abs(rhat_v) <= std::numeric_limits<double>::min()) {
            result.stop_reason = "bicgstab_alpha_breakdown";
            break;
        }
        alpha = rho / rhat_v;
        if (!std::isfinite(alpha)) {
            result.stop_reason = "bicgstab_alpha_nan";
            break;
        }

        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_residual_scaled(
                n, d_r_.data(), d_bicgstab_v_.data(), alpha, d_bicgstab_s_.data());
        });

        result.timings.preconditioner_apply_seconds += gpu_timed([&] {
            if (using_metis_block_jacobi()) {
                preconditioner_.apply(d_bicgstab_s_.data(), d_bicgstab_s_hat_.data());
            } else {
                kernels::launch_copy(n, d_bicgstab_s_.data(), d_bicgstab_s_hat_.data());
            }
        });
        if (using_metis_block_jacobi()) {
            add_apply_timings();
        }

        result.timings.spmv_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(matrix.rows,
                                     matrix.row_ptr,
                                     matrix.col_idx,
                                     matrix.values,
                                     d_bicgstab_s_hat_.data(),
                                     d_bicgstab_t_.data());
        });

        double dots[2] = {0.0, 0.0};
        const double fused_dot_seconds = gpu_timed([&] {
            kernels::launch_mr1_two_dot_reduction(
                n, d_bicgstab_t_.data(), d_bicgstab_s_.data(), d_mr1_dots_.data());
        });
        result.timings.dot_reduction_seconds += fused_dot_seconds;
        const auto dots_copy_start = std::chrono::steady_clock::now();
        d_mr1_dots_.copy_to(dots, 2);
        const double dots_copy_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - dots_copy_start).count();
        result.timings.dot_reduction_seconds += dots_copy_seconds;
        result.timings.bicgstab_scalar_sync_seconds += dots_copy_seconds;
        const double omega_den = dots[1];
        if (!std::isfinite(dots[0]) || !std::isfinite(omega_den) ||
            omega_den <= std::numeric_limits<double>::min()) {
            result.stop_reason = "bicgstab_omega_breakdown";
            break;
        }
        omega = dots[0] / omega_den;
        if (!std::isfinite(omega)) {
            result.stop_reason = "bicgstab_omega_nan";
            break;
        }

        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_bicgstab_update_x_r(n,
                                                alpha,
                                                d_bicgstab_p_hat_.data(),
                                                omega,
                                                d_bicgstab_s_hat_.data(),
                                                d_bicgstab_s_.data(),
                                                d_bicgstab_t_.data(),
                                                d_x_work_.data(),
                                                d_r_.data());
        });

        rho_old = rho;
        result.iterations = iter + 1;
        if (std::abs(omega) <= std::numeric_limits<double>::min()) {
            result.stop_reason = "bicgstab_omega_zero";
            break;
        }
    }

    double residual_norm = 0.0;
    result.timings.final_residual_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
    });
    const auto residual_copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&residual_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    result.timings.bicgstab_scalar_sync_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - residual_copy_start).count();
    result.residual_norm2 = residual_norm;
    result.relative_residual_norm2 = residual_norm / rhs_norm;
    result.scaled_residual_norm2 = residual_norm;
    result.scaled_relative_residual_norm2 = result.relative_residual_norm2;
    if (result.stop_reason.empty()) {
        result.converged = residual_norm <= target_norm;
        result.stop_reason = result.converged ? "bicgstab_converged" : "bicgstab_fixed_iter";
    }

    if (scaled) {
        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_multiply_by_scale(
                n, d_col_scale_.data(), d_x_work_.data(), d_x_work_.data());
        });
        double unscaled_rhs_norm = 0.0;
        result.timings.final_residual_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_,
                                           n,
                                           d_rhs_unscaled_.data(),
                                           1,
                                           d_h_col_.data()));
        });
        CUITER_CUDA_CHECK(cudaMemcpy(&unscaled_rhs_norm,
                                     d_h_col_.data(),
                                     sizeof(double),
                                     cudaMemcpyDeviceToHost));
        unscaled_rhs_norm = std::max(unscaled_rhs_norm, std::numeric_limits<double>::min());
        const DeviceCsrMatrixView raw_matrix = unscaled_permuted_matrix_view();
        result.timings.final_residual_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(raw_matrix.rows,
                                     raw_matrix.row_ptr,
                                     raw_matrix.col_idx,
                                     raw_matrix.values,
                                     d_x_work_.data(),
                                     d_ax_.data());
            kernels::launch_residual(n, d_rhs_unscaled_.data(), d_ax_.data(), d_r_.data());
            CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
        });
        double unscaled_residual_norm = 0.0;
        CUITER_CUDA_CHECK(cudaMemcpy(&unscaled_residual_norm,
                                     d_h_col_.data(),
                                     sizeof(double),
                                     cudaMemcpyDeviceToHost));
        result.unscaled_residual_norm2 = unscaled_residual_norm;
        result.unscaled_relative_residual_norm2 = unscaled_residual_norm / unscaled_rhs_norm;
        result.residual_norm2 = result.unscaled_residual_norm2;
        result.relative_residual_norm2 = result.unscaled_relative_residual_norm2;
    } else {
        result.unscaled_residual_norm2 = residual_norm;
        result.unscaled_relative_residual_norm2 = result.relative_residual_norm2;
    }

    result.timings.unpermute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_unpermute_vector(
                n, preconditioner_.d_new_to_old(), d_x_work_.data(), d_x);
        } else {
            kernels::launch_copy(n, d_x_work_.data(), d_x);
        }
    });
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.bicgstab_spmv_seconds = result.timings.spmv_seconds;
    result.timings.bicgstab_dot_reduction_seconds = result.timings.dot_reduction_seconds;
    result.timings.bicgstab_update_seconds = result.timings.solution_update_seconds;
    result.timings.bicgstab_total_seconds =
        result.timings.preconditioner_apply_seconds +
        result.timings.spmv_seconds +
        result.timings.dot_reduction_seconds +
        result.timings.solution_update_seconds;
    result.timings.middle_solver_total_seconds = result.timings.bicgstab_total_seconds;
    result.timings.gmres_loop_seconds = result.timings.middle_solver_total_seconds;
    result.timings.solve_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return result;
}

LinearSolveResult GmresSolver::solve_bicgstab_fused_fixed2_device(
    const DeviceCsrMatrixView& matrix,
    const double* d_rhs,
    double* d_x,
    std::chrono::steady_clock::time_point solve_start)
{
    const int32_t n = matrix_.rows;
    const bool scaled = using_field_scaling();
    LinearSolveResult result;
    if (using_ruiz_scaling()) {
        result.stop_reason = "bicgstab_fused_rejects_ruiz_scaling";
        return result;
    }
    if (using_metis_block_jacobi()) {
        const auto& setup_timings = preconditioner_.timings();
        result.timings.metis_partition_seconds = setup_timings.metis_partition_seconds;
        result.timings.permutation_build_seconds = setup_timings.permutation_build_seconds;
        result.timings.weighted_graph_build_seconds = setup_timings.weighted_graph_build_seconds;
        result.timings.block_extract_seconds = setup_timings.block_extract_seconds;
        result.timings.block_lu_seconds = setup_timings.block_lu_seconds;
        result.timings.ras_symbolic_seconds = setup_timings.ras_symbolic_seconds;
        result.timings.ras_setup_seconds = setup_timings.ras_setup_seconds;
        result.timings.setup_total_seconds = setup_timings.setup_total_seconds;
        result.block_stats = preconditioner_.block_stats();
    }

    result.timings.rhs_permute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_permute_vector(
                n, preconditioner_.d_new_to_old(), d_rhs, d_rhs_unscaled_.data());
        } else {
            kernels::launch_copy(n, d_rhs, d_rhs_unscaled_.data());
        }
        if (scaled) {
            kernels::launch_multiply_by_scale(
                n, d_row_scale_.data(), d_rhs_unscaled_.data(), d_rhs_work_.data());
        } else {
            kernels::launch_copy(n, d_rhs_unscaled_.data(), d_rhs_work_.data());
        }
        kernels::launch_set_zero(n, d_bicgstab_v_.data());
        kernels::launch_set_zero(n, d_bicgstab_p_.data());
        kernels::launch_set_zero(n, d_x_work_.data());
        kernels::launch_set_constant(8, 1.0, d_mr1_dots_.data());
    });
    if (scaled) {
        attach_scaling_result_metadata(result);
    }

    double rhs_norm = 0.0;
    result.timings.dot_reduction_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_rhs_work_.data(), 1, d_h_col_.data()));
    });
    CUITER_CUDA_CHECK(cudaMemcpy(&rhs_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    const double target_norm = std::max(options_.abs_tolerance, options_.rel_tolerance * rhs_norm);

    result.timings.solution_update_seconds += gpu_timed([&] {
        kernels::launch_copy(n, d_rhs_work_.data(), d_r_.data());
        kernels::launch_copy(n, d_rhs_work_.data(), d_bicgstab_r_hat_.data());
    });

    const auto add_apply_timings = [&] {
        const auto& apply_timings = preconditioner_.last_apply_timings();
        result.timings.block_jacobi_apply_seconds += apply_timings.block_jacobi_apply_seconds;
        result.timings.coarse_az0_spmv_seconds += apply_timings.coarse_az0_spmv_seconds;
        result.timings.coarse_compress_seconds += apply_timings.coarse_compress_seconds;
        result.timings.coarse_solve_seconds += apply_timings.coarse_solve_seconds;
        result.timings.coarse_expand_seconds += apply_timings.coarse_expand_seconds;
        result.timings.coarse_total_seconds += apply_timings.coarse_total_seconds;
        result.timings.ras_apply_seconds += apply_timings.ras_apply_seconds;
        result.timings.ras_gather_seconds += apply_timings.ras_gather_seconds;
        result.timings.ras_local_gemv_seconds += apply_timings.ras_local_gemv_seconds;
        result.timings.ras_scatter_seconds += apply_timings.ras_scatter_seconds;
        result.timings.preconditioner_total_seconds += apply_timings.preconditioner_total_seconds;
        result.timings.coarse_failed = result.timings.coarse_failed || apply_timings.coarse_failed;
    };

    for (int32_t iter = 0; iter < 2; ++iter) {
        result.timings.dot_reduction_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas_,
                                           n,
                                           d_bicgstab_r_hat_.data(),
                                           1,
                                           d_r_.data(),
                                           1,
                                           d_mr1_dots_.data()));
        });

        result.timings.solution_update_seconds += gpu_timed([&] {
            if (iter == 0) {
                kernels::launch_copy(n, d_r_.data(), d_bicgstab_p_.data());
            } else {
                kernels::launch_bicgstab_compute_beta(d_mr1_dots_.data());
                kernels::launch_bicgstab_update_p_device_scalar(n,
                                                                d_r_.data(),
                                                                d_mr1_dots_.data() + 7,
                                                                d_mr1_dots_.data() + 6,
                                                                d_bicgstab_v_.data(),
                                                                d_bicgstab_p_.data());
            }
        });

        result.timings.preconditioner_apply_seconds += gpu_timed([&] {
            if (using_metis_block_jacobi()) {
                preconditioner_.apply(d_bicgstab_p_.data(), d_bicgstab_p_hat_.data());
            } else {
                kernels::launch_copy(n, d_bicgstab_p_.data(), d_bicgstab_p_hat_.data());
            }
        });
        if (using_metis_block_jacobi()) {
            add_apply_timings();
        }

        result.timings.spmv_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(matrix.rows,
                                     matrix.row_ptr,
                                     matrix.col_idx,
                                     matrix.values,
                                     d_bicgstab_p_hat_.data(),
                                     d_bicgstab_v_.data());
        });

        result.timings.dot_reduction_seconds += gpu_timed([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas_,
                                           n,
                                           d_bicgstab_r_hat_.data(),
                                           1,
                                           d_bicgstab_v_.data(),
                                           1,
                                           d_mr1_dots_.data() + 1));
        });
        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_bicgstab_compute_alpha(d_mr1_dots_.data());
            kernels::launch_residual_scaled_device_scalar(n,
                                                          d_r_.data(),
                                                          d_bicgstab_v_.data(),
                                                          d_mr1_dots_.data() + 5,
                                                          d_bicgstab_s_.data());
        });

        result.timings.preconditioner_apply_seconds += gpu_timed([&] {
            if (using_metis_block_jacobi()) {
                preconditioner_.apply(d_bicgstab_s_.data(), d_bicgstab_s_hat_.data());
            } else {
                kernels::launch_copy(n, d_bicgstab_s_.data(), d_bicgstab_s_hat_.data());
            }
        });
        if (using_metis_block_jacobi()) {
            add_apply_timings();
        }

        result.timings.spmv_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(matrix.rows,
                                     matrix.row_ptr,
                                     matrix.col_idx,
                                     matrix.values,
                                     d_bicgstab_s_hat_.data(),
                                     d_bicgstab_t_.data());
        });

        result.timings.dot_reduction_seconds += gpu_timed([&] {
            kernels::launch_mr1_two_dot_reduction(
                n, d_bicgstab_t_.data(), d_bicgstab_s_.data(), d_mr1_dots_.data() + 2);
        });
        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_bicgstab_compute_omega_and_advance(d_mr1_dots_.data());
            kernels::launch_bicgstab_update_x_r_device_scalar(n,
                                                              d_mr1_dots_.data() + 5,
                                                              d_bicgstab_p_hat_.data(),
                                                              d_mr1_dots_.data() + 6,
                                                              d_bicgstab_s_hat_.data(),
                                                              d_bicgstab_s_.data(),
                                                              d_bicgstab_t_.data(),
                                                              d_x_work_.data(),
                                                              d_r_.data());
        });
        result.iterations = iter + 1;
    }

    double residual_norm = 0.0;
    result.timings.final_residual_seconds += gpu_timed([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_r_.data(), 1, d_h_col_.data()));
    });
    const auto residual_copy_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaMemcpy(&residual_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    result.timings.bicgstab_scalar_sync_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - residual_copy_start).count();
    result.residual_norm2 = residual_norm;
    result.relative_residual_norm2 = residual_norm / rhs_norm;
    result.scaled_residual_norm2 = residual_norm;
    result.scaled_relative_residual_norm2 = result.relative_residual_norm2;
    result.unscaled_residual_norm2 = residual_norm;
    result.unscaled_relative_residual_norm2 = result.relative_residual_norm2;
    result.converged = residual_norm <= target_norm;
    result.stop_reason =
        result.converged ? "bicgstab_fused_fixed2_converged" : "bicgstab_fused_fixed2";

    result.timings.unpermute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_unpermute_vector(
                n, preconditioner_.d_new_to_old(), d_x_work_.data(), d_x);
        } else {
            kernels::launch_copy(n, d_x_work_.data(), d_x);
        }
    });
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.bicgstab_spmv_seconds = result.timings.spmv_seconds;
    result.timings.bicgstab_dot_reduction_seconds = result.timings.dot_reduction_seconds;
    result.timings.bicgstab_update_seconds = result.timings.solution_update_seconds;
    result.timings.bicgstab_total_seconds =
        result.timings.preconditioner_apply_seconds +
        result.timings.spmv_seconds +
        result.timings.dot_reduction_seconds +
        result.timings.solution_update_seconds;
    result.timings.middle_solver_total_seconds = result.timings.bicgstab_total_seconds;
    result.timings.gmres_loop_seconds = result.timings.middle_solver_total_seconds;
    result.timings.solve_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return result;
}

LinearSolveResult GmresSolver::solve_device(const double* d_values,
                                            const double* d_rhs,
                                            double* d_x)
{
    if (!setup_ready_ || d_values == nullptr || d_rhs == nullptr || d_x == nullptr) {
        throw std::runtime_error("GmresSolver::solve_device called before setup or with null input");
    }
    const int32_t n = matrix_.rows;
    const int32_t restart = options_.use_bicgstab_fixed_path
                                ? 2
                                : std::min(options_.restart, options_.max_iters);
    ensure_workspace(n, restart);

    LinearSolveResult result;
    if (using_metis_block_jacobi()) {
        const auto& setup_timings = preconditioner_.timings();
        result.timings.metis_partition_seconds = setup_timings.metis_partition_seconds;
        result.timings.permutation_build_seconds = setup_timings.permutation_build_seconds;
        result.timings.weighted_graph_build_seconds = setup_timings.weighted_graph_build_seconds;
        result.timings.block_extract_seconds = setup_timings.block_extract_seconds;
        result.timings.block_lu_seconds = setup_timings.block_lu_seconds;
        result.timings.ras_symbolic_seconds = setup_timings.ras_symbolic_seconds;
        result.timings.ras_setup_seconds = setup_timings.ras_setup_seconds;
        result.timings.setup_total_seconds = setup_timings.setup_total_seconds;
        result.block_stats = preconditioner_.block_stats();
    }

    const DeviceCsrMatrixView matrix = active_matrix_view();
    const auto solve_start = std::chrono::steady_clock::now();
    if (options_.use_mr2_fast_path) {
        return solve_mr2_device(matrix, d_rhs, d_x, solve_start);
    }
    if (options_.use_mr1_fast_path) {
        return solve_mr1_device(matrix, d_rhs, d_x, solve_start);
    }
    if (options_.use_bicgstab_fixed_path) {
        if (options_.use_bicgstab_fused_fixed2 && options_.max_iters == 2) {
            return solve_bicgstab_fused_fixed2_device(matrix, d_rhs, d_x, solve_start);
        }
        return solve_bicgstab_device(matrix, d_rhs, d_x, solve_start);
    }

    result.timings.rhs_permute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_permute_vector(n, preconditioner_.d_new_to_old(), d_rhs, d_rhs_work_.data());
        } else {
            kernels::launch_copy(n, d_rhs, d_rhs_work_.data());
        }
        kernels::launch_set_zero(n, d_x_work_.data());
    });

    double rhs_norm = 0.0;
    result.timings.dot_reduction_seconds += host_timed_with_sync([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_rhs_work_.data(), 1, d_h_col_.data()));
    });
    CUITER_CUDA_CHECK(cudaMemcpy(&rhs_norm, d_h_col_.data(), sizeof(double), cudaMemcpyDeviceToHost));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());

    result.timings.solution_update_seconds += gpu_timed([&] {
        kernels::launch_copy(n, d_rhs_work_.data(), d_r_.data());
    });
    double beta = norm_device(n, d_r_.data());
    result.residual_norm2 = beta;
    result.relative_residual_norm2 = beta / rhs_norm;
    const double target_norm = std::max(options_.abs_tolerance, options_.rel_tolerance * rhs_norm);
    if (beta <= target_norm) {
        result.converged = true;
        result.stop_reason = "initial_residual";
    }

    const auto loop_start = std::chrono::steady_clock::now();
    while (!result.converged && result.iterations < options_.max_iters) {
        const int32_t remaining = options_.max_iters - result.iterations;
        const int32_t cycle_limit = std::min(restart, remaining);
        std::fill(hessenberg_.begin(), hessenberg_.end(), 0.0);
        std::fill(g_.begin(), g_.end(), 0.0);
        std::fill(givens_c_.begin(), givens_c_.end(), 0.0);
        std::fill(givens_s_.begin(), givens_s_.end(), 0.0);
        g_[0] = beta;

        result.timings.solution_update_seconds += gpu_timed([&] {
            kernels::launch_scale_copy(n, 1.0 / beta, d_r_.data(), d_v_basis_.data());
        });

        bool solution_updated = false;
        int32_t steps_this_cycle = 0;
        for (int32_t j = 0; j < cycle_limit; ++j) {
            const double* d_vj = d_v_basis_.data() + basis_offset(n, j);
            double* d_zj = d_z_basis_.data() + basis_offset(n, j);
            double* d_vnext = d_v_basis_.data() + basis_offset(n, j + 1);

            result.timings.preconditioner_apply_seconds += gpu_timed([&] {
                if (using_metis_block_jacobi()) {
                    preconditioner_.apply(d_vj, d_zj);
                } else {
                    kernels::launch_copy(n, d_vj, d_zj);
                }
            });

            result.timings.spmv_seconds += gpu_timed([&] {
                kernels::launch_csr_spmv(matrix.rows,
                                         matrix.row_ptr,
                                         matrix.col_idx,
                                         matrix.values,
                                         d_zj,
                                         d_w_.data());
            });

            for (int32_t i = 0; i <= j; ++i) {
                const double* d_vi = d_v_basis_.data() + basis_offset(n, i);
                result.timings.dot_reduction_seconds += gpu_timed([&] {
                    CUITER_CUBLAS_CHECK(cublasDdot(cublas_, n, d_w_.data(), 1, d_vi, 1,
                                                   d_h_col_.data() + i));
                });
                result.timings.orthogonalization_seconds += gpu_timed([&] {
                    kernels::launch_sub_scaled_device_scalar(
                        n, d_vi, d_h_col_.data() + i, d_w_.data());
                });
            }
            result.timings.dot_reduction_seconds += gpu_timed([&] {
                CUITER_CUBLAS_CHECK(cublasDnrm2(cublas_, n, d_w_.data(), 1,
                                               d_h_col_.data() + j + 1));
            });
            const auto scalar_copy_start = std::chrono::steady_clock::now();
            d_h_col_.copy_to(h_col_host_.data(), static_cast<std::size_t>(j + 2));
            const auto scalar_copy_stop = std::chrono::steady_clock::now();
            result.timings.dot_reduction_seconds +=
                std::chrono::duration<double>(scalar_copy_stop - scalar_copy_start).count();

            for (int32_t row = 0; row <= j + 1; ++row) {
                hessenberg_[static_cast<std::size_t>(h_index(row, j))] =
                    h_col_host_[static_cast<std::size_t>(row)];
            }

            const double next_norm = h_col_host_[static_cast<std::size_t>(j + 1)];
            if (!std::isfinite(next_norm)) {
                result.stop_reason = "arnoldi_norm_nan_or_inf";
                break;
            }
            const bool happy_breakdown =
                next_norm <= std::numeric_limits<double>::epsilon() * rhs_norm;
            if (!happy_breakdown) {
                result.timings.solution_update_seconds += gpu_timed([&] {
                    kernels::launch_scale_copy(n, 1.0 / next_norm, d_w_.data(), d_vnext);
                });
            }

            const double estimated_relative = apply_givens_and_residual(j, rhs_norm);
            result.residual_estimates.push_back(estimated_relative);
            ++result.iterations;
            steps_this_cycle = j + 1;
            result.residual_norm2 = estimated_relative * rhs_norm;
            result.relative_residual_norm2 = estimated_relative;

            if (result.residual_norm2 <= target_norm || happy_breakdown ||
                result.iterations == options_.max_iters) {
                const int32_t basis_count = j + 1;
                if (!solve_small_upper(basis_count)) {
                    result.stop_reason = "small_upper_breakdown";
                    break;
                }
                result.timings.solution_update_seconds += gpu_timed([&] {
                    update_solution(n, basis_count, d_x_work_.data());
                });
                solution_updated = true;

                beta = compute_residual(matrix, n, d_rhs_work_.data(), d_x_work_.data(), rhs_norm, result);
                if (beta <= target_norm) {
                    result.converged = true;
                    result.stop_reason = happy_breakdown ? "happy_breakdown" : "converged";
                }
                break;
            }
        }

        if (result.converged || !result.stop_reason.empty()) {
            break;
        }

        if (!solution_updated) {
            if (steps_this_cycle <= 0) {
                result.stop_reason = "empty_restart_cycle";
                break;
            }
            if (!solve_small_upper(steps_this_cycle)) {
                result.stop_reason = "small_upper_breakdown";
                break;
            }
            result.timings.solution_update_seconds += gpu_timed([&] {
                update_solution(n, steps_this_cycle, d_x_work_.data());
            });
            beta = compute_residual(matrix, n, d_rhs_work_.data(), d_x_work_.data(), rhs_norm, result);
            if (beta <= target_norm) {
                result.converged = true;
                result.stop_reason = "converged";
                break;
            }
        }
    }

    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.gmres_loop_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();

    if (result.stop_reason.empty()) {
        result.stop_reason = result.converged ? "converged" : "max_iters";
    }
    if (options_.compute_true_residual && result.stop_reason == "max_iters") {
        compute_residual(matrix, n, d_rhs_work_.data(), d_x_work_.data(), rhs_norm, result);
    }

    result.timings.unpermute_seconds += gpu_timed([&] {
        if (using_metis_block_jacobi()) {
            kernels::launch_unpermute_vector(n, preconditioner_.d_new_to_old(), d_x_work_.data(), d_x);
        } else {
            kernels::launch_copy(n, d_x_work_.data(), d_x);
        }
    });
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    result.timings.solve_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    return result;
}

}  // namespace cuiter
