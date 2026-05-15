#include "cuiter/reorder/metis_partition.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
using metis_idx_t = int32_t;
using metis_real_t = float;

int METIS_PartGraphKway(metis_idx_t* nvtxs,
                        metis_idx_t* ncon,
                        metis_idx_t* xadj,
                        metis_idx_t* adjncy,
                        metis_idx_t* vwgt,
                        metis_idx_t* vsize,
                        metis_idx_t* adjwgt,
                        metis_idx_t* nparts,
                        metis_real_t* tpwgts,
                        metis_real_t* ubvec,
                        metis_idx_t* options,
                        metis_idx_t* objval,
                        metis_idx_t* part);
}

namespace cuiter {
namespace {

constexpr int kMetisOk = 1;

template <typename Fn>
double host_timed(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

void validate_pattern(const CsrMatrix& matrix)
{
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.rows != matrix.cols) {
        throw std::runtime_error("METIS block Jacobi requires a square CSR matrix");
    }
    if (static_cast<int32_t>(matrix.row_ptr.size()) != matrix.rows + 1) {
        throw std::runtime_error("invalid CSR row_ptr length");
    }
    if (matrix.row_ptr.front() != 0 ||
        matrix.row_ptr.back() != static_cast<int32_t>(matrix.col_idx.size())) {
        throw std::runtime_error("invalid CSR row_ptr bounds");
    }
    for (int32_t row = 0; row < matrix.rows; ++row) {
        if (matrix.row_ptr[static_cast<std::size_t>(row)] >
            matrix.row_ptr[static_cast<std::size_t>(row + 1)]) {
            throw std::runtime_error("CSR row_ptr is not monotonic");
        }
    }
    for (int32_t col : matrix.col_idx) {
        if (col < 0 || col >= matrix.cols) {
            throw std::runtime_error("CSR col_idx out of range");
        }
    }
}

std::vector<std::vector<int32_t>> build_symmetrized_adjacency(const CsrMatrix& matrix)
{
    std::vector<std::vector<int32_t>> adjacency(static_cast<std::size_t>(matrix.rows));
    for (int32_t row = 0; row < matrix.rows; ++row) {
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            if (row == col) {
                continue;
            }
            adjacency[static_cast<std::size_t>(row)].push_back(col);
            adjacency[static_cast<std::size_t>(col)].push_back(row);
        }
    }

    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    return adjacency;
}

void append_group_chunks(const std::vector<int32_t>& sorted_group,
                         int32_t target_block_size,
                         std::vector<std::vector<int32_t>>& blocks)
{
    if (sorted_group.empty()) {
        return;
    }

    const int32_t group_size = static_cast<int32_t>(sorted_group.size());
    const int32_t chunks = std::max(1, (group_size + target_block_size - 1) / target_block_size);
    int32_t begin = 0;
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
        const int32_t remaining = group_size - begin;
        const int32_t remaining_chunks = chunks - chunk;
        const int32_t chunk_size = (remaining + remaining_chunks - 1) / remaining_chunks;
        const int32_t end = begin + chunk_size;
        blocks.emplace_back(sorted_group.begin() + static_cast<std::ptrdiff_t>(begin),
                            sorted_group.begin() + static_cast<std::ptrdiff_t>(end));
        begin = end;
    }
}

uint64_t edge_key(int32_t a, int32_t b)
{
    const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

int32_t edge_key_lo(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
}

int32_t edge_key_hi(uint64_t key)
{
    return static_cast<int32_t>(static_cast<uint32_t>(key & 0xffffffffu));
}

int32_t clamp_weight(double value, int32_t clamp)
{
    if (!std::isfinite(value) || value <= 0.0) {
        return 1;
    }
    const double rounded = std::round(value);
    const double bounded = std::min<double>(std::max<double>(1.0, rounded), clamp);
    return static_cast<int32_t>(bounded);
}

std::vector<int32_t> block_of_old_indices(const CsrMatrix& matrix,
                                          const std::vector<int32_t>& old_to_new,
                                          const std::vector<int32_t>& block_starts,
                                          const std::vector<int32_t>& block_sizes)
{
    std::vector<int32_t> block_of_new(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t block = 0; block < static_cast<int32_t>(block_sizes.size()); ++block) {
        const int32_t begin = block_starts[static_cast<std::size_t>(block)];
        const int32_t end = begin + block_sizes[static_cast<std::size_t>(block)];
        for (int32_t row = begin; row < end; ++row) {
            block_of_new[static_cast<std::size_t>(row)] = block;
        }
    }

    std::vector<int32_t> block_of_old(static_cast<std::size_t>(matrix.rows), -1);
    for (int32_t old_index = 0; old_index < matrix.rows; ++old_index) {
        const int32_t new_index = old_to_new[static_cast<std::size_t>(old_index)];
        block_of_old[static_cast<std::size_t>(old_index)] =
            block_of_new[static_cast<std::size_t>(new_index)];
    }
    return block_of_old;
}

}  // namespace

BlockStructureStats compute_partition_stats(const CsrMatrix& matrix,
                                            const std::vector<double>* values,
                                            const std::vector<int32_t>& old_to_new,
                                            const std::vector<int32_t>& block_starts,
                                            const std::vector<int32_t>& block_sizes,
                                            int32_t target_block_size,
                                            const std::vector<int32_t>* index_to_bus,
                                            const std::vector<int32_t>* index_field,
                                            const std::string& partition_mode)
{
    BlockStructureStats stats;
    stats.block_size_target = target_block_size;
    stats.num_blocks = static_cast<int32_t>(block_sizes.size());
    stats.partition_mode = partition_mode;
    if (block_sizes.empty()) {
        return stats;
    }

    stats.min_block_size = *std::min_element(block_sizes.begin(), block_sizes.end());
    stats.max_block_size = *std::max_element(block_sizes.begin(), block_sizes.end());
    stats.avg_block_size =
        static_cast<double>(matrix.rows) / static_cast<double>(block_sizes.size());
    double variance = 0.0;
    for (int32_t size : block_sizes) {
        const double diff = static_cast<double>(size) - stats.avg_block_size;
        variance += diff * diff;
    }
    stats.std_block_size =
        std::sqrt(variance / static_cast<double>(std::max<std::size_t>(1, block_sizes.size())));

    const std::vector<int32_t> block_of_old =
        block_of_old_indices(matrix, old_to_new, block_starts, block_sizes);

    std::vector<int64_t> diag_nnz(static_cast<std::size_t>(block_sizes.size()), 0);
    int64_t offblock_nnz = 0;
    const int64_t total_nnz = static_cast<int64_t>(matrix.col_idx.size());
    double weighted_total = 0.0;
    double weighted_diag = 0.0;
    double field_total[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double field_diag[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int32_t old_row = 0; old_row < matrix.rows; ++old_row) {
        const int32_t row_block = block_of_old[static_cast<std::size_t>(old_row)];
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(old_row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++pos) {
            const int32_t old_col = matrix.col_idx[static_cast<std::size_t>(pos)];
            const int32_t col_block = block_of_old[static_cast<std::size_t>(old_col)];
            const bool same_block = row_block == col_block;
            if (row_block == col_block) {
                ++diag_nnz[static_cast<std::size_t>(row_block)];
            } else {
                ++offblock_nnz;
            }
            if (values != nullptr && static_cast<std::size_t>(pos) < values->size()) {
                const double weighted = (*values)[static_cast<std::size_t>(pos)] *
                                        (*values)[static_cast<std::size_t>(pos)];
                weighted_total += weighted;
                if (same_block) {
                    weighted_diag += weighted;
                }
                if (index_field != nullptr &&
                    static_cast<std::size_t>(old_row) < index_field->size() &&
                    static_cast<std::size_t>(old_col) < index_field->size()) {
                    const int32_t row_field = (*index_field)[static_cast<std::size_t>(old_row)];
                    const int32_t col_field = (*index_field)[static_cast<std::size_t>(old_col)];
                    if (row_field >= 0 && row_field < 2 && col_field >= 0 && col_field < 2) {
                        field_total[row_field][col_field] += weighted;
                        if (same_block) {
                            field_diag[row_field][col_field] += weighted;
                        }
                    }
                }
            }
        }
    }

    double density_sum = 0.0;
    for (int32_t block = 0; block < static_cast<int32_t>(block_sizes.size()); ++block) {
        const double size = static_cast<double>(block_sizes[static_cast<std::size_t>(block)]);
        density_sum += static_cast<double>(diag_nnz[static_cast<std::size_t>(block)]) /
                       std::max(1.0, size * size);
    }
    stats.diagonal_block_density_avg =
        density_sum / static_cast<double>(block_sizes.size());
    stats.diagonal_block_nnz_ratio =
        total_nnz > 0 ? 1.0 - static_cast<double>(offblock_nnz) / static_cast<double>(total_nnz) : 0.0;
    stats.offblock_nnz_ratio =
        total_nnz > 0 ? static_cast<double>(offblock_nnz) / static_cast<double>(total_nnz) : 0.0;
    stats.total_weighted_coupling = std::sqrt(weighted_total);
    stats.diagonal_weighted_coupling_ratio =
        weighted_total > 0.0 ? weighted_diag / weighted_total : 0.0;
    stats.offblock_weighted_coupling_ratio =
        weighted_total > 0.0 ? 1.0 - stats.diagonal_weighted_coupling_ratio : 0.0;
    stats.j11_diagonal_weighted_ratio =
        field_total[0][0] > 0.0 ? field_diag[0][0] / field_total[0][0] : 0.0;
    stats.j12_diagonal_weighted_ratio =
        field_total[0][1] > 0.0 ? field_diag[0][1] / field_total[0][1] : 0.0;
    stats.j21_diagonal_weighted_ratio =
        field_total[1][0] > 0.0 ? field_diag[1][0] / field_total[1][0] : 0.0;
    stats.j22_diagonal_weighted_ratio =
        field_total[1][1] > 0.0 ? field_diag[1][1] / field_total[1][1] : 0.0;

    if (index_to_bus != nullptr) {
        std::unordered_map<int32_t, int32_t> field0_block;
        std::unordered_map<int32_t, int32_t> field1_block;
        for (int32_t index = 0; index < matrix.rows; ++index) {
            if (static_cast<std::size_t>(index) >= index_to_bus->size() ||
                index_field == nullptr ||
                static_cast<std::size_t>(index) >= index_field->size()) {
                continue;
            }
            const int32_t bus = (*index_to_bus)[static_cast<std::size_t>(index)];
            const int32_t field = (*index_field)[static_cast<std::size_t>(index)];
            const int32_t block = block_of_old[static_cast<std::size_t>(index)];
            if (field == 0) {
                field0_block[bus] = block;
            } else if (field == 1) {
                field1_block[bus] = block;
            }
        }
        for (const auto& item : field0_block) {
            const auto it = field1_block.find(item.first);
            if (it != field1_block.end() && it->second != item.second) {
                ++stats.theta_vmag_split_count;
                ++stats.pq_split_count;
            }
        }
    }
    return stats;
}

MetisPermutation build_metis_permutation(const CsrMatrix& matrix, int32_t target_block_size)
{
    validate_pattern(matrix);
    if (target_block_size <= 0) {
        throw std::runtime_error("target_block_size must be positive");
    }

    MetisPermutation permutation;
    std::vector<metis_idx_t> part(static_cast<std::size_t>(matrix.rows), 0);
    metis_idx_t nparts = std::max<metis_idx_t>(
        1, (matrix.rows + target_block_size - 1) / target_block_size);

    permutation.timings.metis_partition_seconds = host_timed([&] {
        const auto adjacency = build_symmetrized_adjacency(matrix);
        std::vector<metis_idx_t> xadj(static_cast<std::size_t>(matrix.rows + 1), 0);
        std::vector<metis_idx_t> adjncy;
        for (int32_t row = 0; row < matrix.rows; ++row) {
            xadj[static_cast<std::size_t>(row)] = static_cast<metis_idx_t>(adjncy.size());
            for (int32_t neighbor : adjacency[static_cast<std::size_t>(row)]) {
                adjncy.push_back(static_cast<metis_idx_t>(neighbor));
            }
        }
        xadj[static_cast<std::size_t>(matrix.rows)] = static_cast<metis_idx_t>(adjncy.size());

        if (nparts == 1 || adjncy.empty()) {
            std::fill(part.begin(), part.end(), 0);
            return;
        }

        metis_idx_t nvtxs = matrix.rows;
        metis_idx_t ncon = 1;
        metis_idx_t edgecut = 0;
        const int status = METIS_PartGraphKway(&nvtxs,
                                               &ncon,
                                               xadj.data(),
                                               adjncy.data(),
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               &nparts,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               &edgecut,
                                               part.data());
        if (status != kMetisOk) {
            throw std::runtime_error("METIS_PartGraphKway failed with status=" +
                                     std::to_string(status));
        }
    });

    permutation.timings.permutation_build_seconds = host_timed([&] {
        std::vector<std::vector<int32_t>> groups(static_cast<std::size_t>(nparts));
        for (int32_t node = 0; node < matrix.rows; ++node) {
            const int32_t group = part[static_cast<std::size_t>(node)];
            if (group < 0 || group >= nparts) {
                throw std::runtime_error("METIS returned an out-of-range part id");
            }
            groups[static_cast<std::size_t>(group)].push_back(node);
        }

        std::vector<std::vector<int32_t>> blocks;
        blocks.reserve(static_cast<std::size_t>(nparts));
        for (auto& group : groups) {
            std::sort(group.begin(), group.end());
            append_group_chunks(group, target_block_size, blocks);
        }

        permutation.new_to_old.clear();
        permutation.old_to_new.assign(static_cast<std::size_t>(matrix.rows), -1);
        permutation.block_starts.clear();
        permutation.block_sizes.clear();
        permutation.new_to_old.reserve(static_cast<std::size_t>(matrix.rows));
        for (const auto& block_nodes : blocks) {
            if (block_nodes.empty()) {
                continue;
            }
            permutation.block_starts.push_back(
                static_cast<int32_t>(permutation.new_to_old.size()));
            permutation.block_sizes.push_back(static_cast<int32_t>(block_nodes.size()));
            for (int32_t old_index : block_nodes) {
                const int32_t new_index = static_cast<int32_t>(permutation.new_to_old.size());
                permutation.new_to_old.push_back(old_index);
                permutation.old_to_new[static_cast<std::size_t>(old_index)] = new_index;
            }
        }

        if (static_cast<int32_t>(permutation.new_to_old.size()) != matrix.rows) {
            throw std::runtime_error("permutation did not include every unknown");
        }
        for (int32_t value : permutation.old_to_new) {
            if (value < 0) {
                throw std::runtime_error("permutation has an unassigned old index");
            }
        }
    });

    permutation.stats = compute_partition_stats(matrix,
                                                nullptr,
                                                permutation.old_to_new,
                                                permutation.block_starts,
                                                permutation.block_sizes,
                                                target_block_size,
                                                nullptr,
                                                nullptr,
                                                "unknown_metis");
    return permutation;
}

MetisPermutation build_bus_weighted_metis_permutation(const CsrMatrix& matrix,
                                                      const std::vector<double>& values,
                                                      const BusWeightedPartitionOptions& options)
{
    validate_pattern(matrix);
    if (static_cast<int32_t>(values.size()) != matrix.nnz()) {
        throw std::runtime_error("bus weighted METIS requires numeric values for every nonzero");
    }
    if (options.n_bus <= 0 ||
        static_cast<int32_t>(options.index_to_bus.size()) != matrix.rows ||
        static_cast<int32_t>(options.index_field.size()) != matrix.rows) {
        throw std::runtime_error("invalid bus weighted partition metadata");
    }
    if (options.target_block_unknowns <= 0 ||
        options.bus_edge_weight_scale <= 0.0 ||
        options.bus_edge_weight_clamp <= 0) {
        throw std::runtime_error("invalid bus weighted partition options");
    }
    if (options.bus_edge_weight != "jacobian_frobenius") {
        throw std::runtime_error("only bus_edge_weight=jacobian_frobenius is implemented");
    }

    MetisPermutation permutation;
    permutation.stats.partition_mode = "bus_weighted_metis";
    std::vector<int32_t> unknown_count(static_cast<std::size_t>(options.n_bus), 0);
    std::vector<int32_t> field0_index(static_cast<std::size_t>(options.n_bus), -1);
    std::vector<int32_t> field1_index(static_cast<std::size_t>(options.n_bus), -1);
    for (int32_t index = 0; index < matrix.rows; ++index) {
        const int32_t bus = options.index_to_bus[static_cast<std::size_t>(index)];
        const int32_t field = options.index_field[static_cast<std::size_t>(index)];
        if (bus < 0 || bus >= options.n_bus || field < 0 || field > 1) {
            throw std::runtime_error("invalid index-to-bus metadata");
        }
        ++unknown_count[static_cast<std::size_t>(bus)];
        if (field == 0) {
            field0_index[static_cast<std::size_t>(bus)] = index;
        } else {
            field1_index[static_cast<std::size_t>(bus)] = index;
        }
    }

    std::vector<int32_t> active_buses;
    std::vector<int32_t> bus_to_vertex(static_cast<std::size_t>(options.n_bus), -1);
    active_buses.reserve(static_cast<std::size_t>(matrix.rows));
    int32_t total_unknowns = 0;
    for (int32_t bus = 0; bus < options.n_bus; ++bus) {
        if (unknown_count[static_cast<std::size_t>(bus)] <= 0) {
            continue;
        }
        bus_to_vertex[static_cast<std::size_t>(bus)] = static_cast<int32_t>(active_buses.size());
        active_buses.push_back(bus);
        total_unknowns += unknown_count[static_cast<std::size_t>(bus)];
    }
    if (total_unknowns != matrix.rows || active_buses.empty()) {
        throw std::runtime_error("bus weighted partition metadata does not cover every unknown");
    }

    std::unordered_map<uint64_t, double> edge_sumsq;
    permutation.timings.weighted_graph_build_seconds = host_timed([&] {
        for (int32_t row = 0; row < matrix.rows; ++row) {
            const int32_t row_bus = options.index_to_bus[static_cast<std::size_t>(row)];
            const int32_t row_vertex = bus_to_vertex[static_cast<std::size_t>(row_bus)];
            if (row_vertex < 0) {
                continue;
            }
            for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
                 pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
                 ++pos) {
                const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
                const int32_t col_bus = options.index_to_bus[static_cast<std::size_t>(col)];
                const int32_t col_vertex = bus_to_vertex[static_cast<std::size_t>(col_bus)];
                if (col_vertex < 0 || row_vertex == col_vertex) {
                    continue;
                }
                const double value = values[static_cast<std::size_t>(pos)];
                edge_sumsq[edge_key(row_vertex, col_vertex)] += value * value;
            }
        }
    });

    const int32_t nv = static_cast<int32_t>(active_buses.size());
    std::vector<std::vector<std::pair<int32_t, int32_t>>> adjacency(static_cast<std::size_t>(nv));
    std::vector<double> raw_weights;
    raw_weights.reserve(edge_sumsq.size());
    for (const auto& item : edge_sumsq) {
        const double raw = std::sqrt(std::max(0.0, item.second));
        if (raw > 0.0) {
            raw_weights.push_back(raw);
        }
    }
    double median = 0.0;
    if (!raw_weights.empty()) {
        const std::size_t middle = raw_weights.size() / 2;
        std::nth_element(raw_weights.begin(), raw_weights.begin() + static_cast<std::ptrdiff_t>(middle), raw_weights.end());
        median = raw_weights[middle];
    }
    for (const auto& item : edge_sumsq) {
        const int32_t u = edge_key_lo(item.first);
        const int32_t v = edge_key_hi(item.first);
        const double raw = std::sqrt(std::max(0.0, item.second));
        const double scaled = median > 0.0 ? options.bus_edge_weight_scale * raw / median : 1.0;
        const int32_t weight = clamp_weight(scaled, options.bus_edge_weight_clamp);
        adjacency[static_cast<std::size_t>(u)].push_back({v, weight});
        adjacency[static_cast<std::size_t>(v)].push_back({u, weight});
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
    }

    metis_idx_t nparts = std::max<metis_idx_t>(
        1, (total_unknowns + options.target_block_unknowns - 1) / options.target_block_unknowns);
    std::vector<metis_idx_t> part(static_cast<std::size_t>(nv), 0);
    permutation.timings.metis_partition_seconds = host_timed([&] {
        std::vector<metis_idx_t> xadj(static_cast<std::size_t>(nv + 1), 0);
        std::vector<metis_idx_t> adjncy;
        std::vector<metis_idx_t> adjwgt;
        std::vector<metis_idx_t> vwgt(static_cast<std::size_t>(nv), 1);
        for (int32_t vertex = 0; vertex < nv; ++vertex) {
            const int32_t bus = active_buses[static_cast<std::size_t>(vertex)];
            vwgt[static_cast<std::size_t>(vertex)] =
                std::max<metis_idx_t>(1, unknown_count[static_cast<std::size_t>(bus)]);
            xadj[static_cast<std::size_t>(vertex)] = static_cast<metis_idx_t>(adjncy.size());
            for (const auto& edge : adjacency[static_cast<std::size_t>(vertex)]) {
                adjncy.push_back(edge.first);
                adjwgt.push_back(edge.second);
            }
        }
        xadj[static_cast<std::size_t>(nv)] = static_cast<metis_idx_t>(adjncy.size());
        if (nparts == 1 || adjncy.empty()) {
            std::fill(part.begin(), part.end(), 0);
            return;
        }

        metis_idx_t nvtxs = nv;
        metis_idx_t ncon = 1;
        metis_idx_t edgecut = 0;
        const int status = METIS_PartGraphKway(&nvtxs,
                                               &ncon,
                                               xadj.data(),
                                               adjncy.data(),
                                               vwgt.data(),
                                               nullptr,
                                               adjwgt.data(),
                                               &nparts,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               &edgecut,
                                               part.data());
        if (status != kMetisOk) {
            throw std::runtime_error("bus weighted METIS_PartGraphKway failed with status=" +
                                     std::to_string(status));
        }
    });

    permutation.timings.permutation_build_seconds = host_timed([&] {
        std::vector<std::vector<int32_t>> part_buses(static_cast<std::size_t>(nparts));
        for (int32_t vertex = 0; vertex < nv; ++vertex) {
            const int32_t group = part[static_cast<std::size_t>(vertex)];
            if (group < 0 || group >= nparts) {
                throw std::runtime_error("bus weighted METIS returned out-of-range part id");
            }
            part_buses[static_cast<std::size_t>(group)].push_back(
                active_buses[static_cast<std::size_t>(vertex)]);
        }

        permutation.new_to_old.clear();
        permutation.old_to_new.assign(static_cast<std::size_t>(matrix.rows), -1);
        permutation.block_starts.clear();
        permutation.block_sizes.clear();
        permutation.new_to_old.reserve(static_cast<std::size_t>(matrix.rows));
        for (auto& buses : part_buses) {
            std::sort(buses.begin(), buses.end());
            if (buses.empty()) {
                continue;
            }
            const int32_t block_begin = static_cast<int32_t>(permutation.new_to_old.size());
            permutation.block_starts.push_back(block_begin);
            for (int32_t bus : buses) {
                const int32_t theta = field0_index[static_cast<std::size_t>(bus)];
                const int32_t vmag = field1_index[static_cast<std::size_t>(bus)];
                if (theta >= 0) {
                    const int32_t new_index = static_cast<int32_t>(permutation.new_to_old.size());
                    permutation.new_to_old.push_back(theta);
                    permutation.old_to_new[static_cast<std::size_t>(theta)] = new_index;
                }
                if (vmag >= 0) {
                    const int32_t new_index = static_cast<int32_t>(permutation.new_to_old.size());
                    permutation.new_to_old.push_back(vmag);
                    permutation.old_to_new[static_cast<std::size_t>(vmag)] = new_index;
                }
            }
            permutation.block_sizes.push_back(
                static_cast<int32_t>(permutation.new_to_old.size()) - block_begin);
        }
        if (static_cast<int32_t>(permutation.new_to_old.size()) != matrix.rows) {
            throw std::runtime_error("bus weighted permutation did not include every unknown");
        }
        for (int32_t value : permutation.old_to_new) {
            if (value < 0) {
                throw std::runtime_error("bus weighted permutation has an unassigned old index");
            }
        }
    });

    permutation.stats = compute_partition_stats(matrix,
                                                &values,
                                                permutation.old_to_new,
                                                permutation.block_starts,
                                                permutation.block_sizes,
                                                options.target_block_unknowns,
                                                &options.index_to_bus,
                                                &options.index_field,
                                                "bus_weighted_metis");
    permutation.stats.num_blocks = static_cast<int32_t>(permutation.block_sizes.size());
    permutation.timings.weighted_graph_build_seconds +=
        permutation.stats.num_blocks >= 0 ? 0.0 : 0.0;
    return permutation;
}

PermutedCsrPattern build_permuted_csr_pattern(const CsrMatrix& matrix,
                                              const MetisPermutation& permutation)
{
    validate_pattern(matrix);
    if (static_cast<int32_t>(permutation.new_to_old.size()) != matrix.rows ||
        static_cast<int32_t>(permutation.old_to_new.size()) != matrix.rows) {
        throw std::runtime_error("invalid permutation size");
    }

    PermutedCsrPattern pattern;
    pattern.rows = matrix.rows;
    pattern.cols = matrix.cols;
    pattern.row_ptr.assign(static_cast<std::size_t>(matrix.rows + 1), 0);
    pattern.col_idx.reserve(matrix.col_idx.size());
    pattern.value_source_index.reserve(matrix.col_idx.size());

    std::vector<std::pair<int32_t, int32_t>> row_entries;
    for (int32_t new_row = 0; new_row < matrix.rows; ++new_row) {
        const int32_t old_row = permutation.new_to_old[static_cast<std::size_t>(new_row)];
        row_entries.clear();
        for (int32_t old_pos = matrix.row_ptr[static_cast<std::size_t>(old_row)];
             old_pos < matrix.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++old_pos) {
            const int32_t old_col = matrix.col_idx[static_cast<std::size_t>(old_pos)];
            const int32_t new_col = permutation.old_to_new[static_cast<std::size_t>(old_col)];
            row_entries.emplace_back(new_col, old_pos);
        }
        std::sort(row_entries.begin(), row_entries.end());
        pattern.row_ptr[static_cast<std::size_t>(new_row)] =
            static_cast<int32_t>(pattern.col_idx.size());
        for (const auto& entry : row_entries) {
            pattern.col_idx.push_back(entry.first);
            pattern.value_source_index.push_back(entry.second);
        }
    }
    pattern.row_ptr[static_cast<std::size_t>(matrix.rows)] =
        static_cast<int32_t>(pattern.col_idx.size());
    return pattern;
}

}  // namespace cuiter
