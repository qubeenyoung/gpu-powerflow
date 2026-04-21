#include "linear/metis_partition.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

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

namespace exp_20260415::block_ilu {
namespace {

constexpr int kMetisOk = 1;

void assign_chunked_blocks(const std::vector<std::vector<int32_t>>& groups,
                           int32_t rows,
                           int32_t max_block_size,
                           std::vector<int32_t>& block_of_new,
                           std::vector<int32_t>& local_of_new,
                           std::vector<int32_t>& block_sizes)
{
    block_of_new.assign(static_cast<std::size_t>(rows), -1);
    local_of_new.assign(static_cast<std::size_t>(rows), -1);
    block_sizes.clear();

    for (std::vector<int32_t> group : groups) {
        std::sort(group.begin(), group.end());
        for (std::size_t begin = 0; begin < group.size();
             begin += static_cast<std::size_t>(max_block_size)) {
            const std::size_t end =
                std::min(group.size(), begin + static_cast<std::size_t>(max_block_size));
            const int32_t block = static_cast<int32_t>(block_sizes.size());
            const int32_t block_size = static_cast<int32_t>(end - begin);
            block_sizes.push_back(block_size);
            for (int32_t local = 0; local < block_size; ++local) {
                const int32_t node = group[begin + static_cast<std::size_t>(local)];
                block_of_new[static_cast<std::size_t>(node)] = block;
                local_of_new[static_cast<std::size_t>(node)] = local;
            }
        }
    }

    for (int32_t node = 0; node < rows; ++node) {
        if (block_of_new[static_cast<std::size_t>(node)] < 0 ||
            local_of_new[static_cast<std::size_t>(node)] < 0) {
            throw std::runtime_error("METIS partition left an unassigned J11 node");
        }
    }
}

std::vector<std::vector<int32_t>> build_new_coordinate_adjacency(
    const HostCsrPattern& host_pattern,
    const std::vector<int32_t>& old_to_new)
{
    const int32_t rows = host_pattern.rows;
    std::vector<std::vector<int32_t>> adjacency(static_cast<std::size_t>(rows));
    for (int32_t old_row = 0; old_row < rows; ++old_row) {
        const int32_t new_row = old_to_new[static_cast<std::size_t>(old_row)];
        for (int32_t pos = host_pattern.row_ptr[static_cast<std::size_t>(old_row)];
             pos < host_pattern.row_ptr[static_cast<std::size_t>(old_row + 1)];
             ++pos) {
            const int32_t old_col = host_pattern.col_idx[static_cast<std::size_t>(pos)];
            const int32_t new_col = old_to_new[static_cast<std::size_t>(old_col)];
            if (new_row == new_col) {
                continue;
            }
            adjacency[static_cast<std::size_t>(new_row)].push_back(new_col);
            adjacency[static_cast<std::size_t>(new_col)].push_back(new_row);
        }
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    return adjacency;
}

}  // namespace

void build_metis_graph_partitions(const HostCsrPattern& host_pattern,
                                  const std::vector<int32_t>& old_to_new,
                                  int32_t max_block_size,
                                  std::vector<int32_t>& block_of_new,
                                  std::vector<int32_t>& local_of_new,
                                  std::vector<int32_t>& block_sizes)
{
    if (host_pattern.rows <= 0 || host_pattern.rows != host_pattern.cols ||
        static_cast<int32_t>(old_to_new.size()) != host_pattern.rows ||
        max_block_size <= 0) {
        throw std::runtime_error("invalid input for METIS J11 partitioning");
    }

    const int32_t rows = host_pattern.rows;
    if (rows <= max_block_size) {
        std::vector<std::vector<int32_t>> one_group(1);
        one_group[0].resize(static_cast<std::size_t>(rows));
        std::iota(one_group[0].begin(), one_group[0].end(), 0);
        assign_chunked_blocks(one_group,
                              rows,
                              max_block_size,
                              block_of_new,
                              local_of_new,
                              block_sizes);
        return;
    }

    const auto adjacency = build_new_coordinate_adjacency(host_pattern, old_to_new);
    std::vector<metis_idx_t> xadj(static_cast<std::size_t>(rows + 1), 0);
    std::vector<metis_idx_t> adjncy;
    for (int32_t row = 0; row < rows; ++row) {
        xadj[static_cast<std::size_t>(row)] =
            static_cast<metis_idx_t>(adjncy.size());
        for (int32_t neighbor : adjacency[static_cast<std::size_t>(row)]) {
            adjncy.push_back(static_cast<metis_idx_t>(neighbor));
        }
    }
    xadj[static_cast<std::size_t>(rows)] = static_cast<metis_idx_t>(adjncy.size());

    if (adjncy.empty()) {
        std::vector<std::vector<int32_t>> groups;
        for (int32_t begin = 0; begin < rows; begin += max_block_size) {
            std::vector<int32_t> group;
            const int32_t end = std::min(rows, begin + max_block_size);
            for (int32_t node = begin; node < end; ++node) {
                group.push_back(node);
            }
            groups.push_back(std::move(group));
        }
        assign_chunked_blocks(groups,
                              rows,
                              max_block_size,
                              block_of_new,
                              local_of_new,
                              block_sizes);
        return;
    }

    metis_idx_t nvtxs = rows;
    metis_idx_t ncon = 1;
    metis_idx_t nparts = (rows + max_block_size - 1) / max_block_size;
    metis_idx_t edgecut = 0;
    std::vector<metis_idx_t> part(static_cast<std::size_t>(rows), 0);

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

    std::vector<std::vector<int32_t>> groups(static_cast<std::size_t>(nparts));
    for (int32_t node = 0; node < rows; ++node) {
        const int32_t group = part[static_cast<std::size_t>(node)];
        if (group < 0 || group >= nparts) {
            throw std::runtime_error("METIS returned an invalid partition id");
        }
        groups[static_cast<std::size_t>(group)].push_back(node);
    }
    assign_chunked_blocks(groups,
                          rows,
                          max_block_size,
                          block_of_new,
                          local_of_new,
                          block_sizes);
}

}  // namespace exp_20260415::block_ilu
