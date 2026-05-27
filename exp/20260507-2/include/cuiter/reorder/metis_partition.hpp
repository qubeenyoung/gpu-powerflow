#pragma once

#include "cuiter/core/csr_matrix.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cuiter {

struct MetisPartitionTimings {
    double metis_partition_seconds = 0.0;
    double permutation_build_seconds = 0.0;
    double weighted_graph_build_seconds = 0.0;
};

struct BlockStructureStats {
    int32_t block_size_target = 0;
    int32_t num_blocks = 0;
    int32_t min_block_size = 0;
    int32_t max_block_size = 0;
    double avg_block_size = 0.0;
    double std_block_size = 0.0;
    double diagonal_block_density_avg = 0.0;
    double diagonal_block_nnz_ratio = 0.0;
    double offblock_nnz_ratio = 0.0;
    double total_weighted_coupling = 0.0;
    double diagonal_weighted_coupling_ratio = 0.0;
    double offblock_weighted_coupling_ratio = 0.0;
    double j11_diagonal_weighted_ratio = 0.0;
    double j12_diagonal_weighted_ratio = 0.0;
    double j21_diagonal_weighted_ratio = 0.0;
    double j22_diagonal_weighted_ratio = 0.0;
    int32_t theta_vmag_split_count = 0;
    int32_t pq_split_count = 0;
    std::string partition_mode = "unknown_metis";
};

struct MetisPermutation {
    std::vector<int32_t> new_to_old;
    std::vector<int32_t> old_to_new;
    std::vector<int32_t> block_starts;
    std::vector<int32_t> block_sizes;
    BlockStructureStats stats;
    MetisPartitionTimings timings;
};

struct PermutedCsrPattern {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<int32_t> value_source_index;
};

struct BusWeightedPartitionOptions {
    int32_t n_bus = 0;
    int32_t target_block_unknowns = 64;
    std::string bus_edge_weight = "jacobian_frobenius";
    double bus_edge_weight_scale = 1000.0;
    int32_t bus_edge_weight_clamp = 1000000;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;
};

MetisPermutation build_metis_permutation(const CsrMatrix& matrix, int32_t target_block_size);
MetisPermutation build_bus_weighted_metis_permutation(const CsrMatrix& matrix,
                                                      const std::vector<double>& values,
                                                      const BusWeightedPartitionOptions& options);

BlockStructureStats compute_partition_stats(const CsrMatrix& matrix,
                                            const std::vector<double>* values,
                                            const std::vector<int32_t>& old_to_new,
                                            const std::vector<int32_t>& block_starts,
                                            const std::vector<int32_t>& block_sizes,
                                            int32_t target_block_size,
                                            const std::vector<int32_t>* index_to_bus,
                                            const std::vector<int32_t>* index_field,
                                            const std::string& partition_mode);

PermutedCsrPattern build_permuted_csr_pattern(const CsrMatrix& matrix,
                                              const MetisPermutation& permutation);

}  // namespace cuiter
