#pragma once

#include "model/reduced_jacobian.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260415::block_ilu {

void build_metis_graph_partitions(const HostCsrPattern& host_pattern,
                                  const std::vector<int32_t>& old_to_new,
                                  int32_t max_block_size,
                                  std::vector<int32_t>& block_of_new,
                                  std::vector<int32_t>& local_of_new,
                                  std::vector<int32_t>& block_sizes);

}  // namespace exp_20260415::block_ilu
