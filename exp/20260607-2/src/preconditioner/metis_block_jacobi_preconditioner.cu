#include "cuiter/preconditioner/metis_block_jacobi_preconditioner.hpp"

#include "cuiter/kernels/block_jacobi_kernels.hpp"
#include "cuiter/kernels/gmres_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuiter {
namespace {

template <typename Fn>
double host_timed(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <typename Fn>
double gpu_timed(Fn&& fn)
{
    CudaEventTimer timer;
    timer.start();
    fn();
    return timer.stop();
}

}  // namespace

BlockJacobiApplyMode parse_block_jacobi_apply_mode(const std::string& value)
{
    if (value == "inverse_gemv") {
        return BlockJacobiApplyMode::InverseGemv;
    }
    if (value == "lu_solve") {
        return BlockJacobiApplyMode::LuSolve;
    }
    throw std::runtime_error("unknown block Jacobi apply mode: " + value);
}

std::string to_string(BlockJacobiApplyMode mode)
{
    switch (mode) {
    case BlockJacobiApplyMode::InverseGemv:
        return "inverse_gemv";
    case BlockJacobiApplyMode::LuSolve:
        return "lu_solve";
    }
    return "unknown";
}

MetisBlockJacobiPreconditioner::MetisBlockJacobiPreconditioner()
{
    CUITER_CUBLAS_CHECK(cublasCreate(&cublas_));
    CUITER_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
}

MetisBlockJacobiPreconditioner::~MetisBlockJacobiPreconditioner()
{
    if (cusolver_ != nullptr) {
        cusolverDnDestroy(cusolver_);
        cusolver_ = nullptr;
    }
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

void MetisBlockJacobiPreconditioner::analyze(const CsrMatrix& matrix,
                                             const MetisBlockJacobiOptions& options)
{
    if (options.block_size != 4 && options.block_size != 8 &&
        options.block_size != 16 && options.block_size != 32 && options.block_size != 64) {
        throw std::runtime_error(
            "MetisBlockJacobiPreconditioner currently supports block size 4, 8, 16, 32, or 64");
    }
    if (options.partition_mode != "unknown_metis" &&
        options.partition_mode != "bus_weighted_metis") {
        throw std::runtime_error("unknown partition mode: " + options.partition_mode);
    }
    if (options.partition_mode == "bus_weighted_metis") {
        if (options.target_block_unknowns <= 0 ||
            options.n_bus <= 0 ||
            static_cast<int32_t>(options.index_to_bus.size()) != matrix.rows ||
            static_cast<int32_t>(options.index_field.size()) != matrix.rows) {
            throw std::runtime_error("invalid bus weighted METIS metadata");
        }
        if (options.bus_edge_weight != "jacobian_frobenius") {
            throw std::runtime_error("only bus_edge_weight=jacobian_frobenius is implemented");
        }
        if (options.bus_edge_weight_scale <= 0.0 || options.bus_edge_weight_clamp <= 0) {
            throw std::runtime_error("invalid bus edge weight scaling options");
        }
    }
    if (options.diagonal_shift < 0.0 || !std::isfinite(options.diagonal_shift)) {
        throw std::runtime_error("invalid diagonal shift");
    }
    if (options.ras_overlap != 0 && options.ras_overlap != 1) {
        throw std::runtime_error("RAS overlap must be 0 or 1");
    }
    if (options.enable_coarse) {
        if (options.coarse_vars_per_block != 1 && options.coarse_vars_per_block != 2) {
            throw std::runtime_error("coarse_vars_per_block must be 1 or 2");
        }
        if (options.coarse_vars_per_block == 2 &&
            static_cast<int32_t>(options.index_field.size()) != matrix.rows) {
            throw std::runtime_error("coarse_vars_per_block=2 requires index field metadata");
        }
        if (options.coarse_refresh != "bootstrap_only" &&
            options.coarse_refresh != "every_iter" &&
            options.coarse_refresh != "after_cudss_fallback") {
            throw std::runtime_error("unknown coarse refresh policy: " + options.coarse_refresh);
        }
        if (options.coarse_precision != "fp32" && options.coarse_precision != "fp64") {
            throw std::runtime_error("coarse precision must be fp32 or fp64");
        }
        if (options.coarse_diag_shift_scale < 0.0 ||
            !std::isfinite(options.coarse_diag_shift_scale)) {
            throw std::runtime_error("invalid coarse diagonal shift scale");
        }
    }

    options_ = options;
    matrix_ = matrix;
    rows_ = matrix.rows;
    nnz_ = matrix.nnz();
    leading_dim_ = options.block_size;
    num_blocks_ = 0;
    coarse_dim_ = 0;
    setup_ready_ = false;
    partition_ready_ = false;
    coarse_ready_ = false;
    coarse_failed_ = false;
    ras_enabled_ = options_.ras_overlap == 1;
    ras_stats_ = {};

    if (options_.partition_mode == "unknown_metis") {
        permutation_ = build_metis_permutation(matrix, options.block_size);
        initialize_from_permutation();
    }
    analyzed_ = true;
}

void MetisBlockJacobiPreconditioner::initialize_from_permutation()
{
    num_blocks_ = static_cast<int32_t>(permutation_.block_sizes.size());
    permuted_pattern_ = build_permuted_csr_pattern(matrix_, permutation_);
    timings_.metis_partition_seconds = permutation_.timings.metis_partition_seconds;
    timings_.permutation_build_seconds = permutation_.timings.permutation_build_seconds;
    timings_.weighted_graph_build_seconds = permutation_.timings.weighted_graph_build_seconds;

    d_row_ptr_.assign(permuted_pattern_.row_ptr.data(), permuted_pattern_.row_ptr.size());
    d_col_idx_.assign(permuted_pattern_.col_idx.data(), permuted_pattern_.col_idx.size());
    d_value_source_index_.assign(permuted_pattern_.value_source_index.data(),
                                 permuted_pattern_.value_source_index.size());
    d_new_to_old_.assign(permutation_.new_to_old.data(), permutation_.new_to_old.size());
    d_block_starts_.assign(permutation_.block_starts.data(), permutation_.block_starts.size());
    d_block_sizes_.assign(permutation_.block_sizes.data(), permutation_.block_sizes.size());
    d_permuted_values_.resize(static_cast<std::size_t>(nnz_));

    build_dense_extract_maps();
    build_ras_metadata();
    build_coarse_metadata();
    allocate_dense_buffers();
    allocate_ras_buffers();
    allocate_coarse_buffers();
    rebuild_pointer_arrays();
    rebuild_ras_pointer_arrays();
    rebuild_coarse_pointer_arrays();
    partition_ready_ = true;
}

void MetisBlockJacobiPreconditioner::build_bus_weighted_partition_if_needed(const double* d_values)
{
    if (partition_ready_ || options_.partition_mode != "bus_weighted_metis") {
        return;
    }
    std::vector<double> host_values(static_cast<std::size_t>(nnz_), 0.0);
    CUITER_CUDA_CHECK(cudaMemcpy(host_values.data(),
                                 d_values,
                                 static_cast<std::size_t>(nnz_) * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    BusWeightedPartitionOptions partition_options;
    partition_options.n_bus = options_.n_bus;
    partition_options.target_block_unknowns = options_.target_block_unknowns;
    partition_options.bus_edge_weight = options_.bus_edge_weight;
    partition_options.bus_edge_weight_scale = options_.bus_edge_weight_scale;
    partition_options.bus_edge_weight_clamp = options_.bus_edge_weight_clamp;
    partition_options.index_to_bus = options_.index_to_bus;
    partition_options.index_field = options_.index_field;
    permutation_ = build_bus_weighted_metis_permutation(matrix_, host_values, partition_options);
    initialize_from_permutation();
}

void MetisBlockJacobiPreconditioner::update_numeric_partition_stats(const std::vector<double>& host_values)
{
    if (!partition_ready_) {
        return;
    }
    permutation_.stats = compute_partition_stats(matrix_,
                                                 &host_values,
                                                 permutation_.old_to_new,
                                                 permutation_.block_starts,
                                                 permutation_.block_sizes,
                                                 options_.partition_mode == "bus_weighted_metis" ?
                                                     options_.target_block_unknowns :
                                                     options_.block_size,
                                                 options_.index_to_bus.empty() ? nullptr : &options_.index_to_bus,
                                                 options_.index_field.empty() ? nullptr : &options_.index_field,
                                                 options_.partition_mode);
}

void MetisBlockJacobiPreconditioner::build_dense_extract_maps()
{
    std::vector<int32_t> block_of_row(static_cast<std::size_t>(rows_), -1);
    std::vector<int32_t> local_of_row(static_cast<std::size_t>(rows_), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(permutation_.block_sizes.size()); ++block) {
        const int32_t begin = permutation_.block_starts[static_cast<std::size_t>(block)];
        const int32_t size = permutation_.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < size; ++local) {
            block_of_row[static_cast<std::size_t>(begin + local)] = block;
            local_of_row[static_cast<std::size_t>(begin + local)] = local;
        }
    }

    std::vector<int32_t> dense_block_offsets(static_cast<std::size_t>(nnz_), -1);
    std::vector<int32_t> dense_local_rows(static_cast<std::size_t>(nnz_), 0);
    std::vector<int32_t> dense_local_cols(static_cast<std::size_t>(nnz_), 0);

    for (int32_t row = 0; row < rows_; ++row) {
        const int32_t row_block = block_of_row[static_cast<std::size_t>(row)];
        const int32_t local_row = local_of_row[static_cast<std::size_t>(row)];
        for (int32_t pos = permuted_pattern_.row_ptr[static_cast<std::size_t>(row)];
             pos < permuted_pattern_.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = permuted_pattern_.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_row[static_cast<std::size_t>(col)];
            if (row_block == col_block && row_block >= 0) {
                const int32_t local_col = local_of_row[static_cast<std::size_t>(col)];
                dense_block_offsets[static_cast<std::size_t>(pos)] =
                    row_block * leading_dim_ * leading_dim_;
                dense_local_rows[static_cast<std::size_t>(pos)] = local_row;
                dense_local_cols[static_cast<std::size_t>(pos)] = local_col * leading_dim_;
            }
        }
    }

    d_dense_block_offsets_.assign(dense_block_offsets.data(), dense_block_offsets.size());
    d_dense_local_rows_.assign(dense_local_rows.data(), dense_local_rows.size());
    d_dense_local_cols_.assign(dense_local_cols.data(), dense_local_cols.size());
}

void MetisBlockJacobiPreconditioner::build_ras_metadata()
{
    if (!ras_enabled_) {
        ras_leading_dim_ = 0;
        ras_extract_nnz_ = 0;
        return;
    }

    const auto start_time = std::chrono::steady_clock::now();
    std::vector<int32_t> block_of_row(static_cast<std::size_t>(rows_), -1);
    for (int32_t block = 0; block < num_blocks_; ++block) {
        const int32_t begin = permutation_.block_starts[static_cast<std::size_t>(block)];
        const int32_t size = permutation_.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < size; ++local) {
            block_of_row[static_cast<std::size_t>(begin + local)] = block;
        }
    }

    std::vector<std::vector<int32_t>> neighbors(static_cast<std::size_t>(num_blocks_));
    for (int32_t row = 0; row < rows_; ++row) {
        const int32_t row_block = block_of_row[static_cast<std::size_t>(row)];
        for (int32_t pos = permuted_pattern_.row_ptr[static_cast<std::size_t>(row)];
             pos < permuted_pattern_.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = permuted_pattern_.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_row[static_cast<std::size_t>(col)];
            if (row_block >= 0 && col_block >= 0 && row_block != col_block) {
                neighbors[static_cast<std::size_t>(row_block)].push_back(col_block);
                neighbors[static_cast<std::size_t>(col_block)].push_back(row_block);
            }
        }
    }
    for (auto& list : neighbors) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    std::vector<int32_t> local_offsets(static_cast<std::size_t>(num_blocks_ + 1), 0);
    std::vector<int32_t> local_to_global;
    std::vector<int32_t> owned_sizes(static_cast<std::size_t>(num_blocks_), 0);
    std::vector<int32_t> local_sizes(static_cast<std::size_t>(num_blocks_), 0);
    std::vector<int32_t> source_pos;
    std::vector<int32_t> dense_offsets;
    std::vector<int32_t> dense_rows;
    std::vector<int32_t> dense_cols;
    std::vector<int32_t> global_to_local(static_cast<std::size_t>(rows_), -1);

    ras_leading_dim_ = 0;
    int32_t min_owned = rows_;
    int32_t max_owned = 0;
    int32_t min_overlap = rows_;
    int32_t max_overlap = 0;
    int32_t min_neighbors = num_blocks_;
    int32_t max_neighbors = 0;
    double sum_owned = 0.0;
    double sum_overlap = 0.0;
    double sum_neighbors = 0.0;
    double setup_work = 0.0;
    double apply_work = 0.0;

    for (int32_t block = 0; block < num_blocks_; ++block) {
        local_offsets[static_cast<std::size_t>(block)] =
            static_cast<int32_t>(local_to_global.size());
        const int32_t owned_begin = permutation_.block_starts[static_cast<std::size_t>(block)];
        const int32_t owned_size = permutation_.block_sizes[static_cast<std::size_t>(block)];
        owned_sizes[static_cast<std::size_t>(block)] = owned_size;
        for (int32_t local = 0; local < owned_size; ++local) {
            local_to_global.push_back(owned_begin + local);
        }
        for (int32_t neighbor : neighbors[static_cast<std::size_t>(block)]) {
            const int32_t begin = permutation_.block_starts[static_cast<std::size_t>(neighbor)];
            const int32_t size = permutation_.block_sizes[static_cast<std::size_t>(neighbor)];
            for (int32_t local = 0; local < size; ++local) {
                local_to_global.push_back(begin + local);
            }
        }
        const int32_t offset = local_offsets[static_cast<std::size_t>(block)];
        const int32_t local_size =
            static_cast<int32_t>(local_to_global.size()) - offset;
        local_sizes[static_cast<std::size_t>(block)] = local_size;
        ras_leading_dim_ = std::max(ras_leading_dim_, local_size);

        min_owned = std::min(min_owned, owned_size);
        max_owned = std::max(max_owned, owned_size);
        min_overlap = std::min(min_overlap, local_size);
        max_overlap = std::max(max_overlap, local_size);
        const int32_t neighbor_count =
            static_cast<int32_t>(neighbors[static_cast<std::size_t>(block)].size());
        min_neighbors = std::min(min_neighbors, neighbor_count);
        max_neighbors = std::max(max_neighbors, neighbor_count);
        sum_owned += static_cast<double>(owned_size);
        sum_overlap += static_cast<double>(local_size);
        sum_neighbors += static_cast<double>(neighbor_count);
        setup_work += std::pow(static_cast<double>(local_size), 3.0);
        apply_work += std::pow(static_cast<double>(local_size), 2.0);
    }
    local_offsets[static_cast<std::size_t>(num_blocks_)] =
        static_cast<int32_t>(local_to_global.size());

    for (int32_t block = 0; block < num_blocks_; ++block) {
        const int32_t offset = local_offsets[static_cast<std::size_t>(block)];
        const int32_t local_size = local_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < local_size; ++local) {
            global_to_local[static_cast<std::size_t>(local_to_global[static_cast<std::size_t>(offset + local)])] =
                local;
        }
        const int32_t dense_base = block * ras_leading_dim_ * ras_leading_dim_;
        for (int32_t local_row = 0; local_row < local_size; ++local_row) {
            const int32_t row =
                local_to_global[static_cast<std::size_t>(offset + local_row)];
            for (int32_t pos = permuted_pattern_.row_ptr[static_cast<std::size_t>(row)];
                 pos < permuted_pattern_.row_ptr[static_cast<std::size_t>(row + 1)];
                 ++pos) {
                const int32_t col = permuted_pattern_.col_idx[static_cast<std::size_t>(pos)];
                const int32_t local_col = global_to_local[static_cast<std::size_t>(col)];
                if (local_col >= 0) {
                    source_pos.push_back(pos);
                    dense_offsets.push_back(dense_base);
                    dense_rows.push_back(local_row);
                    dense_cols.push_back(local_col * ras_leading_dim_);
                }
            }
        }
        for (int32_t local = 0; local < local_size; ++local) {
            global_to_local[static_cast<std::size_t>(local_to_global[static_cast<std::size_t>(offset + local)])] =
                -1;
        }
    }

    ras_extract_nnz_ = static_cast<int32_t>(source_pos.size());
    d_ras_local_offsets_.assign(local_offsets.data(), local_offsets.size());
    d_ras_local_to_global_.assign(local_to_global.data(), local_to_global.size());
    d_ras_owned_sizes_.assign(owned_sizes.data(), owned_sizes.size());
    d_ras_local_sizes_.assign(local_sizes.data(), local_sizes.size());
    d_ras_extract_source_pos_.assign(source_pos.data(), source_pos.size());
    d_ras_dense_block_offsets_.assign(dense_offsets.data(), dense_offsets.size());
    d_ras_dense_local_rows_.assign(dense_rows.data(), dense_rows.size());
    d_ras_dense_local_cols_.assign(dense_cols.data(), dense_cols.size());

    const double block_count = static_cast<double>(std::max(1, num_blocks_));
    ras_stats_.overlap = 1;
    ras_stats_.num_blocks = num_blocks_;
    ras_stats_.min_owned_dim = min_owned == rows_ ? 0 : min_owned;
    ras_stats_.max_owned_dim = max_owned;
    ras_stats_.avg_owned_dim = sum_owned / block_count;
    ras_stats_.min_overlap_dim = min_overlap == rows_ ? 0 : min_overlap;
    ras_stats_.max_overlap_dim = max_overlap;
    ras_stats_.avg_overlap_dim = sum_overlap / block_count;
    ras_stats_.min_neighbor_count = min_neighbors == num_blocks_ ? 0 : min_neighbors;
    ras_stats_.max_neighbor_count = max_neighbors;
    ras_stats_.avg_neighbor_count = sum_neighbors / block_count;
    ras_stats_.overlap_dim_growth =
        ras_stats_.avg_owned_dim > 0.0 ? ras_stats_.avg_overlap_dim / ras_stats_.avg_owned_dim : 0.0;
    ras_stats_.estimated_dense_storage_mb =
        static_cast<double>(num_blocks_) * static_cast<double>(ras_leading_dim_) *
        static_cast<double>(ras_leading_dim_) *
        static_cast<double>(options_.use_fp32_preconditioner ? sizeof(float) : sizeof(double)) /
        (1024.0 * 1024.0);
    ras_stats_.estimated_setup_work = setup_work;
    ras_stats_.estimated_apply_work = apply_work;
    ras_stats_.local_nnz_total = ras_extract_nnz_;
    ras_stats_.risk = ras_stats_.max_overlap_dim > 512 || ras_stats_.estimated_dense_storage_mb > 2048.0;
    timings_.ras_symbolic_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
}

void MetisBlockJacobiPreconditioner::build_coarse_metadata()
{
    std::vector<int32_t> block_ids(static_cast<std::size_t>(rows_), 0);
    std::vector<int32_t> coarse_ids(static_cast<std::size_t>(rows_), 0);
    std::vector<float> weights_f32(static_cast<std::size_t>(rows_), 1.0f);
    std::vector<double> weights_f64(static_cast<std::size_t>(rows_), 1.0);
    coarse_dim_ = num_blocks_ * std::max(1, options_.coarse_vars_per_block);

    std::vector<std::array<int32_t, 2>> field_counts(static_cast<std::size_t>(num_blocks_));
    if (options_.coarse_vars_per_block == 2) {
        for (int32_t block = 0; block < num_blocks_; ++block) {
            field_counts[static_cast<std::size_t>(block)] = {0, 0};
            const int32_t begin = permutation_.block_starts[static_cast<std::size_t>(block)];
            const int32_t size = permutation_.block_sizes[static_cast<std::size_t>(block)];
            for (int32_t local = 0; local < size; ++local) {
                const int32_t new_index = begin + local;
                const int32_t old_index =
                    permutation_.new_to_old[static_cast<std::size_t>(new_index)];
                int32_t field = 0;
                if (old_index >= 0 &&
                    static_cast<std::size_t>(old_index) < options_.index_field.size()) {
                    field = options_.index_field[static_cast<std::size_t>(old_index)];
                }
                field = field == 1 ? 1 : 0;
                ++field_counts[static_cast<std::size_t>(block)][static_cast<std::size_t>(field)];
            }
        }
    }

    for (int32_t block = 0; block < num_blocks_; ++block) {
        const int32_t begin = permutation_.block_starts[static_cast<std::size_t>(block)];
        const int32_t size = permutation_.block_sizes[static_cast<std::size_t>(block)];
        for (int32_t local = 0; local < size; ++local) {
            const int32_t row = begin + local;
            int32_t coarse_id = block;
            double weight = 1.0 / std::sqrt(static_cast<double>(std::max(1, size)));
            if (options_.coarse_vars_per_block == 2) {
                const int32_t old_index =
                    permutation_.new_to_old[static_cast<std::size_t>(row)];
                int32_t field = 0;
                if (old_index >= 0 &&
                    static_cast<std::size_t>(old_index) < options_.index_field.size()) {
                    field = options_.index_field[static_cast<std::size_t>(old_index)];
                }
                field = field == 1 ? 1 : 0;
                coarse_id = block * 2 + field;
                const int32_t field_count =
                    field_counts[static_cast<std::size_t>(block)][static_cast<std::size_t>(field)];
                weight = 1.0 / std::sqrt(static_cast<double>(std::max(1, field_count)));
            }
            block_ids[static_cast<std::size_t>(row)] = block;
            coarse_ids[static_cast<std::size_t>(row)] = coarse_id;
            weights_f32[static_cast<std::size_t>(row)] = static_cast<float>(weight);
            weights_f64[static_cast<std::size_t>(row)] = weight;
        }
    }
    d_block_ids_.assign(block_ids.data(), block_ids.size());
    d_coarse_ids_.assign(coarse_ids.data(), coarse_ids.size());
    d_coarse_weights_f32_.assign(weights_f32.data(), weights_f32.size());
    d_coarse_weights_f64_.assign(weights_f64.data(), weights_f64.size());
}

void MetisBlockJacobiPreconditioner::allocate_dense_buffers()
{
    const std::size_t num_blocks = permutation_.block_sizes.size();
    const std::size_t dense_count =
        num_blocks * static_cast<std::size_t>(leading_dim_) * static_cast<std::size_t>(leading_dim_);
    const std::size_t pivot_count = num_blocks * static_cast<std::size_t>(leading_dim_);

    d_pivots_.resize(pivot_count);
    d_info_.resize(num_blocks);
    if (options_.use_fp32_preconditioner) {
        d_dense_f32_.resize(dense_count);
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            d_inverse_f32_.resize(dense_count);
        }
    } else {
        d_dense_f64_.resize(dense_count);
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            d_inverse_f64_.resize(dense_count);
        }
    }
}

void MetisBlockJacobiPreconditioner::allocate_ras_buffers()
{
    if (!ras_enabled_) {
        return;
    }
    if (ras_leading_dim_ <= 0) {
        throw std::runtime_error("RAS metadata is empty");
    }
    if (ras_leading_dim_ > 1024) {
        throw std::runtime_error("RAS overlap local dimension exceeds pilot limit");
    }
    const std::size_t num_blocks = permutation_.block_sizes.size();
    const std::size_t dense_count =
        num_blocks * static_cast<std::size_t>(ras_leading_dim_) *
        static_cast<std::size_t>(ras_leading_dim_);
    const std::size_t pivot_count = num_blocks * static_cast<std::size_t>(ras_leading_dim_);
    d_ras_pivots_.resize(pivot_count);
    d_ras_info_.resize(num_blocks);
    if (options_.use_fp32_preconditioner) {
        d_ras_dense_f32_.resize(dense_count);
        d_ras_inverse_f32_.resize(dense_count);
    } else {
        d_ras_dense_f64_.resize(dense_count);
        d_ras_inverse_f64_.resize(dense_count);
    }
}

void MetisBlockJacobiPreconditioner::allocate_coarse_buffers()
{
    if (!options_.enable_coarse) {
        return;
    }
    const std::size_t coarse_dim = static_cast<std::size_t>(coarse_dim_);
    const std::size_t coarse_count = coarse_dim * coarse_dim;
    d_coarse_az0_.resize(static_cast<std::size_t>(rows_));
    d_coarse_q_.resize(static_cast<std::size_t>(rows_));
    d_coarse_pivots_.resize(coarse_dim);
    d_coarse_info_.resize(1);
    coarse_lwork_ = 0;
    if (options_.coarse_precision == "fp32") {
        d_coarse_lu_f32_.resize(coarse_count);
        d_coarse_rhs_f32_.resize(coarse_dim);
        CUITER_CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_,
                                                          coarse_dim_,
                                                          coarse_dim_,
                                                          d_coarse_lu_f32_.data(),
                                                          coarse_dim_,
                                                          &coarse_lwork_));
        d_coarse_work_f32_.resize(static_cast<std::size_t>(coarse_lwork_));
    } else {
        d_coarse_lu_f64_.resize(coarse_count);
        d_coarse_rhs_f64_.resize(coarse_dim);
        CUITER_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolver_,
                                                          coarse_dim_,
                                                          coarse_dim_,
                                                          d_coarse_lu_f64_.data(),
                                                          coarse_dim_,
                                                          &coarse_lwork_));
        d_coarse_work_f64_.resize(static_cast<std::size_t>(coarse_lwork_));
    }
}

void MetisBlockJacobiPreconditioner::rebuild_pointer_arrays()
{
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    std::vector<float*> dense_ptrs_f32(static_cast<std::size_t>(num_blocks), nullptr);
    std::vector<const float*> dense_const_ptrs_f32(static_cast<std::size_t>(num_blocks), nullptr);
    std::vector<float*> inverse_ptrs_f32(static_cast<std::size_t>(num_blocks), nullptr);
    std::vector<double*> dense_ptrs_f64(static_cast<std::size_t>(num_blocks), nullptr);
    std::vector<const double*> dense_const_ptrs_f64(static_cast<std::size_t>(num_blocks), nullptr);
    std::vector<double*> inverse_ptrs_f64(static_cast<std::size_t>(num_blocks), nullptr);
    const std::size_t stride =
        static_cast<std::size_t>(leading_dim_) * static_cast<std::size_t>(leading_dim_);

    if (options_.use_fp32_preconditioner) {
        for (int32_t block = 0; block < num_blocks; ++block) {
            dense_ptrs_f32[static_cast<std::size_t>(block)] =
                d_dense_f32_.data() + static_cast<std::size_t>(block) * stride;
            dense_const_ptrs_f32[static_cast<std::size_t>(block)] =
                d_dense_f32_.data() + static_cast<std::size_t>(block) * stride;
            if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
                inverse_ptrs_f32[static_cast<std::size_t>(block)] =
                    d_inverse_f32_.data() + static_cast<std::size_t>(block) * stride;
            }
        }
        d_dense_ptrs_f32_.assign(dense_ptrs_f32.data(), dense_ptrs_f32.size());
        d_dense_const_ptrs_f32_.assign(dense_const_ptrs_f32.data(), dense_const_ptrs_f32.size());
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            d_inverse_ptrs_f32_.assign(inverse_ptrs_f32.data(), inverse_ptrs_f32.size());
        }
    } else {
        for (int32_t block = 0; block < num_blocks; ++block) {
            dense_ptrs_f64[static_cast<std::size_t>(block)] =
                d_dense_f64_.data() + static_cast<std::size_t>(block) * stride;
            dense_const_ptrs_f64[static_cast<std::size_t>(block)] =
                d_dense_f64_.data() + static_cast<std::size_t>(block) * stride;
            if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
                inverse_ptrs_f64[static_cast<std::size_t>(block)] =
                    d_inverse_f64_.data() + static_cast<std::size_t>(block) * stride;
            }
        }
        d_dense_ptrs_f64_.assign(dense_ptrs_f64.data(), dense_ptrs_f64.size());
        d_dense_const_ptrs_f64_.assign(dense_const_ptrs_f64.data(), dense_const_ptrs_f64.size());
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            d_inverse_ptrs_f64_.assign(inverse_ptrs_f64.data(), inverse_ptrs_f64.size());
        }
    }
}

void MetisBlockJacobiPreconditioner::rebuild_ras_pointer_arrays()
{
    if (!ras_enabled_) {
        return;
    }
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    const std::size_t stride =
        static_cast<std::size_t>(ras_leading_dim_) * static_cast<std::size_t>(ras_leading_dim_);
    if (options_.use_fp32_preconditioner) {
        std::vector<float*> dense_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        std::vector<const float*> dense_const_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        std::vector<float*> inverse_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        for (int32_t block = 0; block < num_blocks; ++block) {
            dense_ptrs[static_cast<std::size_t>(block)] =
                d_ras_dense_f32_.data() + static_cast<std::size_t>(block) * stride;
            dense_const_ptrs[static_cast<std::size_t>(block)] =
                d_ras_dense_f32_.data() + static_cast<std::size_t>(block) * stride;
            inverse_ptrs[static_cast<std::size_t>(block)] =
                d_ras_inverse_f32_.data() + static_cast<std::size_t>(block) * stride;
        }
        d_ras_dense_ptrs_f32_.assign(dense_ptrs.data(), dense_ptrs.size());
        d_ras_dense_const_ptrs_f32_.assign(dense_const_ptrs.data(), dense_const_ptrs.size());
        d_ras_inverse_ptrs_f32_.assign(inverse_ptrs.data(), inverse_ptrs.size());
    } else {
        std::vector<double*> dense_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        std::vector<const double*> dense_const_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        std::vector<double*> inverse_ptrs(static_cast<std::size_t>(num_blocks), nullptr);
        for (int32_t block = 0; block < num_blocks; ++block) {
            dense_ptrs[static_cast<std::size_t>(block)] =
                d_ras_dense_f64_.data() + static_cast<std::size_t>(block) * stride;
            dense_const_ptrs[static_cast<std::size_t>(block)] =
                d_ras_dense_f64_.data() + static_cast<std::size_t>(block) * stride;
            inverse_ptrs[static_cast<std::size_t>(block)] =
                d_ras_inverse_f64_.data() + static_cast<std::size_t>(block) * stride;
        }
        d_ras_dense_ptrs_f64_.assign(dense_ptrs.data(), dense_ptrs.size());
        d_ras_dense_const_ptrs_f64_.assign(dense_const_ptrs.data(), dense_const_ptrs.size());
        d_ras_inverse_ptrs_f64_.assign(inverse_ptrs.data(), inverse_ptrs.size());
    }
}

void MetisBlockJacobiPreconditioner::rebuild_coarse_pointer_arrays()
{
    if (!options_.enable_coarse) {
        return;
    }
    if (options_.coarse_precision == "fp32") {
        std::vector<float*> lu_ptrs(1, d_coarse_lu_f32_.data());
        std::vector<const float*> lu_const_ptrs(1, d_coarse_lu_f32_.data());
        std::vector<float*> rhs_ptrs(1, d_coarse_rhs_f32_.data());
        d_coarse_lu_ptrs_f32_.assign(lu_ptrs.data(), lu_ptrs.size());
        d_coarse_lu_const_ptrs_f32_.assign(lu_const_ptrs.data(), lu_const_ptrs.size());
        d_coarse_rhs_ptrs_f32_.assign(rhs_ptrs.data(), rhs_ptrs.size());
    } else {
        std::vector<double*> lu_ptrs(1, d_coarse_lu_f64_.data());
        std::vector<const double*> lu_const_ptrs(1, d_coarse_lu_f64_.data());
        std::vector<double*> rhs_ptrs(1, d_coarse_rhs_f64_.data());
        d_coarse_lu_ptrs_f64_.assign(lu_ptrs.data(), lu_ptrs.size());
        d_coarse_lu_const_ptrs_f64_.assign(lu_const_ptrs.data(), lu_const_ptrs.size());
        d_coarse_rhs_ptrs_f64_.assign(rhs_ptrs.data(), rhs_ptrs.size());
    }
}

void MetisBlockJacobiPreconditioner::refresh_permuted_values(const double* d_values)
{
    if (!analyzed_ || !partition_ready_ || d_values == nullptr) {
        throw std::runtime_error("refresh_permuted_values called before analyze or with null values");
    }
    kernels::launch_scatter_csr_values(
        nnz_, d_value_source_index_.data(), d_values, d_permuted_values_.data());
}

void MetisBlockJacobiPreconditioner::setup(const double* d_values)
{
    if (!analyzed_) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::setup called before analyze");
    }

    const auto total_start = std::chrono::steady_clock::now();
    if (options_.partition_mode == "bus_weighted_metis") {
        build_bus_weighted_partition_if_needed(d_values);
    }
    if (!partition_ready_) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner partition is not ready");
    }
    std::vector<double> host_values;
    if (!options_.index_to_bus.empty()) {
        host_values.resize(static_cast<std::size_t>(nnz_));
        CUITER_CUDA_CHECK(cudaMemcpy(host_values.data(),
                                     d_values,
                                     static_cast<std::size_t>(nnz_) * sizeof(double),
                                     cudaMemcpyDeviceToHost));
        update_numeric_partition_stats(host_values);
    }
    const double refresh_seconds = host_timed([&] {
        refresh_permuted_values(d_values);
    });
    setup_permuted_values(d_permuted_values_.data());
    timings_.block_extract_seconds += refresh_seconds;

    const auto total_stop = std::chrono::steady_clock::now();
    timings_.setup_total_seconds =
        std::chrono::duration<double>(total_stop - total_start).count();
}

void MetisBlockJacobiPreconditioner::setup_permuted_values(const double* d_permuted_values)
{
    if (!analyzed_ || !partition_ready_ || d_permuted_values == nullptr) {
        throw std::runtime_error("setup_permuted_values called before analyze or with null values");
    }

    const auto total_start = std::chrono::steady_clock::now();
    timings_.block_extract_seconds = host_timed([&] {
        if (d_permuted_values != d_permuted_values_.data()) {
            CUITER_CUDA_CHECK(cudaMemcpy(d_permuted_values_.data(),
                                         d_permuted_values,
                                         static_cast<std::size_t>(nnz_) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        }
        if (ras_enabled_) {
            const std::size_t dense_count = permutation_.block_sizes.size() *
                                            static_cast<std::size_t>(ras_leading_dim_) *
                                            static_cast<std::size_t>(ras_leading_dim_);
            if (options_.use_fp32_preconditioner) {
                CUITER_CUDA_CHECK(cudaMemset(d_ras_dense_f32_.data(), 0, dense_count * sizeof(float)));
                kernels::launch_extract_ras_dense_blocks_f32(ras_extract_nnz_,
                                                             d_ras_extract_source_pos_.data(),
                                                             d_ras_dense_block_offsets_.data(),
                                                             d_ras_dense_local_rows_.data(),
                                                             d_ras_dense_local_cols_.data(),
                                                             d_permuted_values_.data(),
                                                             d_ras_dense_f32_.data());
                kernels::launch_add_block_diagonal_shift_f32(num_blocks_,
                                                             ras_leading_dim_,
                                                             d_ras_local_sizes_.data(),
                                                             static_cast<float>(options_.diagonal_shift),
                                                             d_ras_dense_f32_.data());
            } else {
                CUITER_CUDA_CHECK(cudaMemset(d_ras_dense_f64_.data(), 0, dense_count * sizeof(double)));
                kernels::launch_extract_ras_dense_blocks_f64(ras_extract_nnz_,
                                                             d_ras_extract_source_pos_.data(),
                                                             d_ras_dense_block_offsets_.data(),
                                                             d_ras_dense_local_rows_.data(),
                                                             d_ras_dense_local_cols_.data(),
                                                             d_permuted_values_.data(),
                                                             d_ras_dense_f64_.data());
                kernels::launch_add_block_diagonal_shift_f64(num_blocks_,
                                                             ras_leading_dim_,
                                                             d_ras_local_sizes_.data(),
                                                             options_.diagonal_shift,
                                                             d_ras_dense_f64_.data());
            }
        } else {
            const std::size_t dense_count = permutation_.block_sizes.size() *
                                            static_cast<std::size_t>(leading_dim_) *
                                            static_cast<std::size_t>(leading_dim_);
            if (options_.use_fp32_preconditioner) {
                CUITER_CUDA_CHECK(cudaMemset(d_dense_f32_.data(), 0, dense_count * sizeof(float)));
                kernels::launch_extract_dense_blocks_f32(nnz_,
                                                         d_dense_block_offsets_.data(),
                                                         d_dense_local_rows_.data(),
                                                         d_dense_local_cols_.data(),
                                                         d_permuted_values_.data(),
                                                         d_dense_f32_.data());
                kernels::launch_add_block_diagonal_shift_f32(
                    static_cast<int32_t>(permutation_.block_sizes.size()),
                    leading_dim_,
                    d_block_sizes_.data(),
                    static_cast<float>(options_.diagonal_shift),
                    d_dense_f32_.data());
            } else {
                CUITER_CUDA_CHECK(cudaMemset(d_dense_f64_.data(), 0, dense_count * sizeof(double)));
                kernels::launch_extract_dense_blocks_f64(nnz_,
                                                         d_dense_block_offsets_.data(),
                                                         d_dense_local_rows_.data(),
                                                         d_dense_local_cols_.data(),
                                                         d_permuted_values_.data(),
                                                         d_dense_f64_.data());
                kernels::launch_add_block_diagonal_shift_f64(
                    static_cast<int32_t>(permutation_.block_sizes.size()),
                    leading_dim_,
                    d_block_sizes_.data(),
                    options_.diagonal_shift,
                    d_dense_f64_.data());
            }
        }
    });

    timings_.block_lu_seconds = host_timed([&] {
        if (ras_enabled_) {
            factorize_or_invert_ras();
        } else {
            factorize_or_invert();
        }
        setup_coarse_if_needed();
    });
    timings_.ras_setup_seconds = ras_enabled_ ?
        timings_.block_extract_seconds + timings_.block_lu_seconds : 0.0;
    const auto total_stop = std::chrono::steady_clock::now();
    timings_.setup_total_seconds =
        std::chrono::duration<double>(total_stop - total_start).count();
    setup_ready_ = true;
}

void MetisBlockJacobiPreconditioner::factorize_or_invert()
{
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    if (num_blocks <= 0) {
        throw std::runtime_error("no METIS blocks to factorize");
    }

    if (options_.use_fp32_preconditioner) {
        CUITER_CUBLAS_CHECK(cublasSgetrfBatched(cublas_,
                                                leading_dim_,
                                                d_dense_ptrs_f32_.data(),
                                                leading_dim_,
                                                d_pivots_.data(),
                                                d_info_.data(),
                                                num_blocks));
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            CUITER_CUBLAS_CHECK(cublasSgetriBatched(
                cublas_,
                leading_dim_,
                d_dense_const_ptrs_f32_.data(),
                leading_dim_,
                d_pivots_.data(),
                d_inverse_ptrs_f32_.data(),
                leading_dim_,
                d_info_.data(),
                num_blocks));
        }
    } else {
        CUITER_CUBLAS_CHECK(cublasDgetrfBatched(cublas_,
                                                leading_dim_,
                                                d_dense_ptrs_f64_.data(),
                                                leading_dim_,
                                                d_pivots_.data(),
                                                d_info_.data(),
                                                num_blocks));
        if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
            CUITER_CUBLAS_CHECK(cublasDgetriBatched(
                cublas_,
                leading_dim_,
                d_dense_const_ptrs_f64_.data(),
                leading_dim_,
                d_pivots_.data(),
                d_inverse_ptrs_f64_.data(),
                leading_dim_,
                d_info_.data(),
                num_blocks));
        }
    }
}

void MetisBlockJacobiPreconditioner::factorize_or_invert_ras()
{
    if (!ras_enabled_) {
        return;
    }
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    if (num_blocks <= 0 || ras_leading_dim_ <= 0) {
        throw std::runtime_error("no RAS blocks to factorize");
    }

    if (options_.use_fp32_preconditioner) {
        CUITER_CUBLAS_CHECK(cublasSgetrfBatched(cublas_,
                                                ras_leading_dim_,
                                                d_ras_dense_ptrs_f32_.data(),
                                                ras_leading_dim_,
                                                d_ras_pivots_.data(),
                                                d_ras_info_.data(),
                                                num_blocks));
        CUITER_CUBLAS_CHECK(cublasSgetriBatched(cublas_,
                                                ras_leading_dim_,
                                                d_ras_dense_const_ptrs_f32_.data(),
                                                ras_leading_dim_,
                                                d_ras_pivots_.data(),
                                                d_ras_inverse_ptrs_f32_.data(),
                                                ras_leading_dim_,
                                                d_ras_info_.data(),
                                                num_blocks));
    } else {
        CUITER_CUBLAS_CHECK(cublasDgetrfBatched(cublas_,
                                                ras_leading_dim_,
                                                d_ras_dense_ptrs_f64_.data(),
                                                ras_leading_dim_,
                                                d_ras_pivots_.data(),
                                                d_ras_info_.data(),
                                                num_blocks));
        CUITER_CUBLAS_CHECK(cublasDgetriBatched(cublas_,
                                                ras_leading_dim_,
                                                d_ras_dense_const_ptrs_f64_.data(),
                                                ras_leading_dim_,
                                                d_ras_pivots_.data(),
                                                d_ras_inverse_ptrs_f64_.data(),
                                                ras_leading_dim_,
                                                d_ras_info_.data(),
                                                num_blocks));
    }
}

void MetisBlockJacobiPreconditioner::setup_coarse_if_needed()
{
    if (!options_.enable_coarse || coarse_failed_) {
        return;
    }
    if ((options_.coarse_refresh == "bootstrap_only" ||
         options_.coarse_refresh == "after_cudss_fallback") &&
        coarse_ready_) {
        return;
    }
    assemble_and_factorize_coarse();
}

void MetisBlockJacobiPreconditioner::assemble_and_factorize_coarse()
{
    if (!options_.enable_coarse || coarse_dim_ <= 0) {
        return;
    }

    const std::size_t coarse_dim = static_cast<std::size_t>(coarse_dim_);
    const std::size_t coarse_count = coarse_dim * coarse_dim;
    if (options_.coarse_precision == "fp32") {
        CUITER_CUDA_CHECK(cudaMemset(d_coarse_lu_f32_.data(), 0, coarse_count * sizeof(float)));
        kernels::launch_assemble_coarse_matrix_f32(rows_,
                                                   coarse_dim_,
                                                   d_row_ptr_.data(),
                                                   d_col_idx_.data(),
                                                   d_permuted_values_.data(),
                                                   d_coarse_ids_.data(),
                                                   d_coarse_weights_f32_.data(),
                                                   d_coarse_lu_f32_.data());
        std::vector<float> coarse_host(coarse_count, 0.0f);
        d_coarse_lu_f32_.copy_to(coarse_host.data(), coarse_host.size());
        double diag_sum = 0.0;
        for (int32_t i = 0; i < coarse_dim_; ++i) {
            diag_sum += std::abs(static_cast<double>(
                coarse_host[static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(i) * coarse_dim]));
        }
        const double diag_mean = diag_sum / static_cast<double>(std::max(1, coarse_dim_));
        const double shift = options_.coarse_diag_shift_scale *
                             (diag_mean > std::numeric_limits<double>::min() ? diag_mean : 1.0);
        for (int32_t i = 0; i < coarse_dim_; ++i) {
            coarse_host[static_cast<std::size_t>(i) +
                        static_cast<std::size_t>(i) * coarse_dim] += static_cast<float>(shift);
        }
        d_coarse_lu_f32_.assign(coarse_host.data(), coarse_host.size());
        const cusolverStatus_t status = cusolverDnSgetrf(cusolver_,
                                                         coarse_dim_,
                                                         coarse_dim_,
                                                         d_coarse_lu_f32_.data(),
                                                         coarse_dim_,
                                                         d_coarse_work_f32_.data(),
                                                         d_coarse_pivots_.data(),
                                                         d_coarse_info_.data());
        int32_t info = 0;
        CUITER_CUDA_CHECK(cudaMemcpy(&info, d_coarse_info_.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (status != CUSOLVER_STATUS_SUCCESS || info != 0) {
            coarse_failed_ = true;
            coarse_ready_ = false;
            return;
        }
    } else {
        CUITER_CUDA_CHECK(cudaMemset(d_coarse_lu_f64_.data(), 0, coarse_count * sizeof(double)));
        kernels::launch_assemble_coarse_matrix_f64(rows_,
                                                   coarse_dim_,
                                                   d_row_ptr_.data(),
                                                   d_col_idx_.data(),
                                                   d_permuted_values_.data(),
                                                   d_coarse_ids_.data(),
                                                   d_coarse_weights_f64_.data(),
                                                   d_coarse_lu_f64_.data());
        std::vector<double> coarse_host(coarse_count, 0.0);
        d_coarse_lu_f64_.copy_to(coarse_host.data(), coarse_host.size());
        double diag_sum = 0.0;
        for (int32_t i = 0; i < coarse_dim_; ++i) {
            diag_sum += std::abs(coarse_host[static_cast<std::size_t>(i) +
                                             static_cast<std::size_t>(i) * coarse_dim]);
        }
        const double diag_mean = diag_sum / static_cast<double>(std::max(1, coarse_dim_));
        const double shift = options_.coarse_diag_shift_scale *
                             (diag_mean > std::numeric_limits<double>::min() ? diag_mean : 1.0);
        for (int32_t i = 0; i < coarse_dim_; ++i) {
            coarse_host[static_cast<std::size_t>(i) +
                        static_cast<std::size_t>(i) * coarse_dim] += shift;
        }
        d_coarse_lu_f64_.assign(coarse_host.data(), coarse_host.size());
        const cusolverStatus_t status = cusolverDnDgetrf(cusolver_,
                                                         coarse_dim_,
                                                         coarse_dim_,
                                                         d_coarse_lu_f64_.data(),
                                                         coarse_dim_,
                                                         d_coarse_work_f64_.data(),
                                                         d_coarse_pivots_.data(),
                                                         d_coarse_info_.data());
        int32_t info = 0;
        CUITER_CUDA_CHECK(cudaMemcpy(&info, d_coarse_info_.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (status != CUSOLVER_STATUS_SUCCESS || info != 0) {
            coarse_failed_ = true;
            coarse_ready_ = false;
            return;
        }
    }
    coarse_ready_ = true;
}

void MetisBlockJacobiPreconditioner::apply_block_jacobi(const double* d_rhs, double* d_out)
{
    if (!setup_ready_ || d_rhs == nullptr || d_out == nullptr) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::apply called before setup or with null input");
    }
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    if (options_.apply_mode == BlockJacobiApplyMode::InverseGemv) {
        if (options_.use_fp32_preconditioner) {
            kernels::launch_block_inverse_apply_f32(num_blocks,
                                                    leading_dim_,
                                                    d_block_starts_.data(),
                                                    d_block_sizes_.data(),
                                                    d_inverse_f32_.data(),
                                                    d_rhs,
                                                    d_out);
        } else {
            kernels::launch_block_inverse_apply_f64(num_blocks,
                                                    leading_dim_,
                                                    d_block_starts_.data(),
                                                    d_block_sizes_.data(),
                                                    d_inverse_f64_.data(),
                                                    d_rhs,
                                                    d_out);
        }
    } else {
        if (options_.use_fp32_preconditioner) {
            kernels::launch_block_lu_solve_apply_f32(num_blocks,
                                                     leading_dim_,
                                                     d_block_starts_.data(),
                                                     d_block_sizes_.data(),
                                                     d_dense_f32_.data(),
                                                     d_pivots_.data(),
                                                     d_rhs,
                                                     d_out);
        } else {
            kernels::launch_block_lu_solve_apply_f64(num_blocks,
                                                     leading_dim_,
                                                     d_block_starts_.data(),
                                                     d_block_sizes_.data(),
                                                     d_dense_f64_.data(),
                                                     d_pivots_.data(),
                                                     d_rhs,
                                                     d_out);
        }
    }
}

void MetisBlockJacobiPreconditioner::apply_ras(const double* d_rhs, double* d_out)
{
    if (!setup_ready_ || !ras_enabled_ || d_rhs == nullptr || d_out == nullptr) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::apply_ras called before setup or with null input");
    }
    const int32_t num_blocks = static_cast<int32_t>(permutation_.block_sizes.size());
    if (options_.use_fp32_preconditioner) {
        kernels::launch_ras_inverse_apply_f32(num_blocks,
                                              ras_leading_dim_,
                                              d_ras_local_offsets_.data(),
                                              d_ras_local_to_global_.data(),
                                              d_ras_owned_sizes_.data(),
                                              d_ras_local_sizes_.data(),
                                              d_ras_inverse_f32_.data(),
                                              d_rhs,
                                              d_out);
    } else {
        kernels::launch_ras_inverse_apply_f64(num_blocks,
                                              ras_leading_dim_,
                                              d_ras_local_offsets_.data(),
                                              d_ras_local_to_global_.data(),
                                              d_ras_owned_sizes_.data(),
                                              d_ras_local_sizes_.data(),
                                              d_ras_inverse_f64_.data(),
                                              d_rhs,
                                              d_out);
    }
}

void MetisBlockJacobiPreconditioner::apply_local(const double* d_rhs, double* d_out)
{
    if (!setup_ready_ || d_rhs == nullptr || d_out == nullptr) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::apply_local called before setup or with null input");
    }
    apply_block_jacobi(d_rhs, d_out);
}

void MetisBlockJacobiPreconditioner::apply_coarse_correction(const double* d_residual,
                                                             double* d_out)
{
    if (!setup_ready_ || d_residual == nullptr || d_out == nullptr) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::apply_coarse_correction called before setup or with null input");
    }

    last_apply_timings_ = {};
    const auto total_start = std::chrono::steady_clock::now();
    if (!options_.enable_coarse || !coarse_ready_ || coarse_failed_) {
        kernels::launch_set_zero(rows_, d_out);
        CUITER_CUDA_CHECK(cudaDeviceSynchronize());
        last_apply_timings_.coarse_failed = true;
        last_apply_timings_.preconditioner_total_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
        return;
    }

    last_apply_timings_.coarse_compress_seconds += gpu_timed([&] {
        if (options_.coarse_precision == "fp32") {
            CUITER_CUDA_CHECK(cudaMemset(d_coarse_rhs_f32_.data(),
                                         0,
                                         static_cast<std::size_t>(coarse_dim_) * sizeof(float)));
            kernels::launch_compress_coarse_rhs_f32(rows_,
                                                    d_coarse_ids_.data(),
                                                    d_coarse_weights_f32_.data(),
                                                    d_residual,
                                                    d_coarse_rhs_f32_.data());
        } else {
            CUITER_CUDA_CHECK(cudaMemset(d_coarse_rhs_f64_.data(),
                                         0,
                                         static_cast<std::size_t>(coarse_dim_) * sizeof(double)));
            kernels::launch_compress_coarse_rhs_f64(rows_,
                                                    d_coarse_ids_.data(),
                                                    d_coarse_weights_f64_.data(),
                                                    d_residual,
                                                    d_coarse_rhs_f64_.data());
        }
    });

    last_apply_timings_.coarse_solve_seconds += gpu_timed([&] {
        cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
        if (options_.coarse_precision == "fp32") {
            status = cusolverDnSgetrs(cusolver_,
                                      CUBLAS_OP_N,
                                      coarse_dim_,
                                      1,
                                      d_coarse_lu_f32_.data(),
                                      coarse_dim_,
                                      d_coarse_pivots_.data(),
                                      d_coarse_rhs_f32_.data(),
                                      coarse_dim_,
                                      d_coarse_info_.data());
        } else {
            status = cusolverDnDgetrs(cusolver_,
                                      CUBLAS_OP_N,
                                      coarse_dim_,
                                      1,
                                      d_coarse_lu_f64_.data(),
                                      coarse_dim_,
                                      d_coarse_pivots_.data(),
                                      d_coarse_rhs_f64_.data(),
                                      coarse_dim_,
                                      d_coarse_info_.data());
        }
        if (status != CUSOLVER_STATUS_SUCCESS) {
            coarse_failed_ = true;
        }
    });

    if (!coarse_failed_) {
        last_apply_timings_.coarse_expand_seconds += gpu_timed([&] {
            kernels::launch_set_zero(rows_, d_out);
            if (options_.coarse_precision == "fp32") {
                kernels::launch_expand_add_coarse_solution_f32(rows_,
                                                               d_coarse_ids_.data(),
                                                               d_coarse_weights_f32_.data(),
                                                               d_coarse_rhs_f32_.data(),
                                                               d_out);
            } else {
                kernels::launch_expand_add_coarse_solution_f64(rows_,
                                                               d_coarse_ids_.data(),
                                                               d_coarse_weights_f64_.data(),
                                                               d_coarse_rhs_f64_.data(),
                                                               d_out);
            }
        });
    }

    last_apply_timings_.coarse_failed = coarse_failed_;
    last_apply_timings_.coarse_total_seconds =
        last_apply_timings_.coarse_compress_seconds +
        last_apply_timings_.coarse_solve_seconds +
        last_apply_timings_.coarse_expand_seconds;
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    last_apply_timings_.preconditioner_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
}

void MetisBlockJacobiPreconditioner::apply(const double* d_rhs, double* d_out)
{
    if (!setup_ready_ || d_rhs == nullptr || d_out == nullptr) {
        throw std::runtime_error("MetisBlockJacobiPreconditioner::apply called before setup or with null input");
    }
    last_apply_timings_ = {};

    const auto total_start = std::chrono::steady_clock::now();
    if (ras_enabled_) {
        last_apply_timings_.ras_apply_seconds += gpu_timed([&] {
            apply_ras(d_rhs, d_out);
        });
        last_apply_timings_.ras_local_gemv_seconds =
            last_apply_timings_.ras_apply_seconds;
        last_apply_timings_.block_jacobi_apply_seconds +=
            last_apply_timings_.ras_apply_seconds;
    } else {
        last_apply_timings_.block_jacobi_apply_seconds += gpu_timed([&] {
            apply_block_jacobi(d_rhs, d_out);
        });
    }

    if (options_.enable_coarse && coarse_ready_ && !coarse_failed_) {
        last_apply_timings_.coarse_az0_spmv_seconds += gpu_timed([&] {
            kernels::launch_csr_spmv(rows_,
                                     d_row_ptr_.data(),
                                     d_col_idx_.data(),
                                     d_permuted_values_.data(),
                                     d_out,
                                     d_coarse_az0_.data());
        });
        last_apply_timings_.coarse_compress_seconds += gpu_timed([&] {
            kernels::launch_residual(rows_, d_rhs, d_coarse_az0_.data(), d_coarse_q_.data());
            if (options_.coarse_precision == "fp32") {
                CUITER_CUDA_CHECK(cudaMemset(d_coarse_rhs_f32_.data(),
                                             0,
                                             static_cast<std::size_t>(coarse_dim_) * sizeof(float)));
                kernels::launch_compress_coarse_rhs_f32(rows_,
                                                        d_coarse_ids_.data(),
                                                        d_coarse_weights_f32_.data(),
                                                        d_coarse_q_.data(),
                                                        d_coarse_rhs_f32_.data());
            } else {
                CUITER_CUDA_CHECK(cudaMemset(d_coarse_rhs_f64_.data(),
                                             0,
                                             static_cast<std::size_t>(coarse_dim_) * sizeof(double)));
                kernels::launch_compress_coarse_rhs_f64(rows_,
                                                        d_coarse_ids_.data(),
                                                        d_coarse_weights_f64_.data(),
                                                        d_coarse_q_.data(),
                                                        d_coarse_rhs_f64_.data());
            }
        });
        last_apply_timings_.coarse_solve_seconds += gpu_timed([&] {
            cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
            if (options_.coarse_precision == "fp32") {
                status = cusolverDnSgetrs(cusolver_,
                                          CUBLAS_OP_N,
                                          coarse_dim_,
                                          1,
                                          d_coarse_lu_f32_.data(),
                                          coarse_dim_,
                                          d_coarse_pivots_.data(),
                                          d_coarse_rhs_f32_.data(),
                                          coarse_dim_,
                                          d_coarse_info_.data());
            } else {
                status = cusolverDnDgetrs(cusolver_,
                                          CUBLAS_OP_N,
                                          coarse_dim_,
                                          1,
                                          d_coarse_lu_f64_.data(),
                                          coarse_dim_,
                                          d_coarse_pivots_.data(),
                                          d_coarse_rhs_f64_.data(),
                                          coarse_dim_,
                                          d_coarse_info_.data());
            }
            if (status != CUSOLVER_STATUS_SUCCESS) {
                coarse_failed_ = true;
            }
        });
        if (!coarse_failed_) {
            last_apply_timings_.coarse_expand_seconds += gpu_timed([&] {
                if (options_.coarse_precision == "fp32") {
                    kernels::launch_expand_add_coarse_solution_f32(rows_,
                                                                   d_coarse_ids_.data(),
                                                                   d_coarse_weights_f32_.data(),
                                                                   d_coarse_rhs_f32_.data(),
                                                                   d_out);
                } else {
                    kernels::launch_expand_add_coarse_solution_f64(rows_,
                                                                   d_coarse_ids_.data(),
                                                                   d_coarse_weights_f64_.data(),
                                                                   d_coarse_rhs_f64_.data(),
                                                                   d_out);
                }
            });
        }
    }

    last_apply_timings_.coarse_failed = coarse_failed_;
    last_apply_timings_.coarse_total_seconds =
        last_apply_timings_.coarse_az0_spmv_seconds +
        last_apply_timings_.coarse_compress_seconds +
        last_apply_timings_.coarse_solve_seconds +
        last_apply_timings_.coarse_expand_seconds;
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    last_apply_timings_.preconditioner_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
}

DeviceCsrMatrixView MetisBlockJacobiPreconditioner::permuted_matrix_view() const
{
    return DeviceCsrMatrixView{rows_,
                               rows_,
                               nnz_,
                               d_row_ptr_.data(),
                               d_col_idx_.data(),
                               d_permuted_values_.data()};
}

}  // namespace cuiter
