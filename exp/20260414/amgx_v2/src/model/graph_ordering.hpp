#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260414::amgx_v2 {

enum class GraphOrderingMethod {
    Natural,
    ReverseCuthillMcKee,
};

struct BusOrdering {
    std::vector<int32_t> ordered_bus;
    std::vector<int32_t> bus_to_position;
    GraphOrderingMethod method = GraphOrderingMethod::Natural;
};

GraphOrderingMethod parse_graph_ordering_method(const std::string& name);

BusOrdering build_bus_ordering(int32_t n_bus,
                               const std::vector<int32_t>& ybus_row_ptr,
                               const std::vector<int32_t>& ybus_col_idx,
                               GraphOrderingMethod method);

}  // namespace exp_20260414::amgx_v2
