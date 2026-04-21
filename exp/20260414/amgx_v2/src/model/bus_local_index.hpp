#pragma once

#include "model/graph_ordering.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

enum class BusEquationKind {
    Fixed,
    Active,
};

enum class BusType {
    Slack,
    Pv,
    Pq,
};

struct BusLocalIndex {
    int32_t n_bus = 0;
    int32_t dim = 0;

    std::vector<int32_t> ordered_bus;
    std::vector<int32_t> bus_to_position;
    std::vector<BusType> bus_type;

    // These arrays are indexed by physical bus id. Every bus has two formal
    // slots: theta and Vm. Fixed slots are later represented by identity rows.
    std::vector<int32_t> theta_slot;
    std::vector<int32_t> vm_slot;
    std::vector<BusEquationKind> p_equation;
    std::vector<BusEquationKind> q_equation;

    int32_t theta(int32_t bus) const;
    int32_t vm(int32_t bus) const;
    bool is_p_active(int32_t bus) const;
    bool is_q_active(int32_t bus) const;
    bool is_slot_active(int32_t slot) const;
};

BusLocalIndex build_bus_local_index(int32_t n_bus,
                                    const std::vector<int32_t>& pv,
                                    const std::vector<int32_t>& pq,
                                    const BusOrdering& ordering);

}  // namespace exp_20260414::amgx_v2
