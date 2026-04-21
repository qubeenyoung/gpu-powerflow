#include "graph_ordering.hpp"

#include <algorithm>
#include <deque>
#include <numeric>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

void validate_graph(int32_t n_bus,
                    const std::vector<int32_t>& row_ptr,
                    const std::vector<int32_t>& col_idx)
{
    if (n_bus <= 0) {
        throw std::runtime_error("bus graph must have at least one vertex");
    }
    if (row_ptr.size() != static_cast<std::size_t>(n_bus + 1)) {
        throw std::runtime_error("bus graph row_ptr size does not match n_bus");
    }
    if (row_ptr.front() != 0 || row_ptr.back() != static_cast<int32_t>(col_idx.size())) {
        throw std::runtime_error("bus graph CSR pointers are inconsistent");
    }
    for (int32_t row = 0; row < n_bus; ++row) {
        if (row_ptr[row] > row_ptr[row + 1]) {
            throw std::runtime_error("bus graph CSR row_ptr must be nondecreasing");
        }
    }
    for (int32_t col : col_idx) {
        if (col < 0 || col >= n_bus) {
            throw std::runtime_error("bus graph column index is out of range");
        }
    }
}

std::vector<int32_t> compute_degrees(int32_t n_bus,
                                     const std::vector<int32_t>& row_ptr,
                                     const std::vector<int32_t>& col_idx)
{
    std::vector<int32_t> degree(static_cast<std::size_t>(n_bus), 0);
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        int32_t count = 0;
        for (int32_t pos = row_ptr[bus]; pos < row_ptr[bus + 1]; ++pos) {
            if (col_idx[static_cast<std::size_t>(pos)] != bus) {
                ++count;
            }
        }
        degree[static_cast<std::size_t>(bus)] = count;
    }
    return degree;
}

BusOrdering make_ordering_from_bus_list(std::vector<int32_t> ordered_bus,
                                        GraphOrderingMethod method,
                                        int32_t n_bus)
{
    if (ordered_bus.size() != static_cast<std::size_t>(n_bus)) {
        throw std::runtime_error("ordering length does not match n_bus");
    }

    std::vector<int32_t> bus_to_position(static_cast<std::size_t>(n_bus), -1);
    for (int32_t pos = 0; pos < n_bus; ++pos) {
        const int32_t bus = ordered_bus[static_cast<std::size_t>(pos)];
        if (bus < 0 || bus >= n_bus || bus_to_position[static_cast<std::size_t>(bus)] >= 0) {
            throw std::runtime_error("ordering is not a valid permutation");
        }
        bus_to_position[static_cast<std::size_t>(bus)] = pos;
    }
    return BusOrdering{std::move(ordered_bus), std::move(bus_to_position), method};
}

BusOrdering build_natural_ordering(int32_t n_bus)
{
    std::vector<int32_t> ordered_bus(static_cast<std::size_t>(n_bus));
    std::iota(ordered_bus.begin(), ordered_bus.end(), 0);
    return make_ordering_from_bus_list(std::move(ordered_bus),
                                       GraphOrderingMethod::Natural,
                                       n_bus);
}

BusOrdering build_rcm_ordering(int32_t n_bus,
                               const std::vector<int32_t>& row_ptr,
                               const std::vector<int32_t>& col_idx)
{
    const std::vector<int32_t> degree = compute_degrees(n_bus, row_ptr, col_idx);
    std::vector<char> visited(static_cast<std::size_t>(n_bus), 0);
    std::vector<int32_t> cm_order;
    cm_order.reserve(static_cast<std::size_t>(n_bus));

    // RCM works component-by-component. Starting from a low-degree vertex tends
    // to reduce bandwidth on sparse grid-like graphs such as power networks.
    while (static_cast<int32_t>(cm_order.size()) < n_bus) {
        int32_t seed = -1;
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            if (visited[static_cast<std::size_t>(bus)]) {
                continue;
            }
            if (seed < 0 ||
                degree[static_cast<std::size_t>(bus)] < degree[static_cast<std::size_t>(seed)]) {
                seed = bus;
            }
        }

        std::vector<int32_t> component_order;
        std::deque<int32_t> queue;
        visited[static_cast<std::size_t>(seed)] = 1;
        queue.push_back(seed);

        while (!queue.empty()) {
            const int32_t bus = queue.front();
            queue.pop_front();
            component_order.push_back(bus);

            std::vector<int32_t> neighbors;
            for (int32_t pos = row_ptr[bus]; pos < row_ptr[bus + 1]; ++pos) {
                const int32_t col = col_idx[static_cast<std::size_t>(pos)];
                if (col != bus && !visited[static_cast<std::size_t>(col)]) {
                    neighbors.push_back(col);
                    visited[static_cast<std::size_t>(col)] = 1;
                }
            }
            std::sort(neighbors.begin(), neighbors.end(), [&](int32_t lhs, int32_t rhs) {
                const int32_t lhs_degree = degree[static_cast<std::size_t>(lhs)];
                const int32_t rhs_degree = degree[static_cast<std::size_t>(rhs)];
                return lhs_degree == rhs_degree ? lhs < rhs : lhs_degree < rhs_degree;
            });
            for (int32_t neighbor : neighbors) {
                queue.push_back(neighbor);
            }
        }

        std::reverse(component_order.begin(), component_order.end());
        cm_order.insert(cm_order.end(), component_order.begin(), component_order.end());
    }

    return make_ordering_from_bus_list(std::move(cm_order),
                                       GraphOrderingMethod::ReverseCuthillMcKee,
                                       n_bus);
}

}  // namespace

GraphOrderingMethod parse_graph_ordering_method(const std::string& name)
{
    if (name == "natural") {
        return GraphOrderingMethod::Natural;
    }
    if (name == "rcm" || name == "reverse_cuthill_mckee") {
        return GraphOrderingMethod::ReverseCuthillMcKee;
    }
    throw std::runtime_error("unknown graph ordering method: " + name);
}

BusOrdering build_bus_ordering(int32_t n_bus,
                               const std::vector<int32_t>& ybus_row_ptr,
                               const std::vector<int32_t>& ybus_col_idx,
                               GraphOrderingMethod method)
{
    validate_graph(n_bus, ybus_row_ptr, ybus_col_idx);
    switch (method) {
    case GraphOrderingMethod::Natural:
        return build_natural_ordering(n_bus);
    case GraphOrderingMethod::ReverseCuthillMcKee:
        return build_rcm_ordering(n_bus, ybus_row_ptr, ybus_col_idx);
    }
    throw std::runtime_error("unhandled graph ordering method");
}

}  // namespace exp_20260414::amgx_v2
