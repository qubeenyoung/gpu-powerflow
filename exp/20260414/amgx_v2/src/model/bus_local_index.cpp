#include "bus_local_index.hpp"

#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

void validate_bus_list(int32_t n_bus, const std::vector<int32_t>& buses, const char* name)
{
    std::vector<char> seen(static_cast<std::size_t>(n_bus), 0);
    for (int32_t bus : buses) {
        if (bus < 0 || bus >= n_bus) {
            throw std::runtime_error(std::string(name) + " bus index is out of range");
        }
        if (seen[static_cast<std::size_t>(bus)]) {
            throw std::runtime_error(std::string(name) + " bus list contains duplicates");
        }
        seen[static_cast<std::size_t>(bus)] = 1;
    }
}

}  // namespace

int32_t BusLocalIndex::theta(int32_t bus) const
{
    if (bus < 0 || bus >= n_bus) {
        throw std::runtime_error("theta index requested for invalid bus");
    }
    return theta_slot[static_cast<std::size_t>(bus)];
}

int32_t BusLocalIndex::vm(int32_t bus) const
{
    if (bus < 0 || bus >= n_bus) {
        throw std::runtime_error("Vm index requested for invalid bus");
    }
    return vm_slot[static_cast<std::size_t>(bus)];
}

bool BusLocalIndex::is_p_active(int32_t bus) const
{
    if (bus < 0 || bus >= n_bus) {
        throw std::runtime_error("P equation requested for invalid bus");
    }
    return p_equation[static_cast<std::size_t>(bus)] == BusEquationKind::Active;
}

bool BusLocalIndex::is_q_active(int32_t bus) const
{
    if (bus < 0 || bus >= n_bus) {
        throw std::runtime_error("Q equation requested for invalid bus");
    }
    return q_equation[static_cast<std::size_t>(bus)] == BusEquationKind::Active;
}

bool BusLocalIndex::is_slot_active(int32_t slot) const
{
    if (slot < 0 || slot >= dim) {
        throw std::runtime_error("activity requested for invalid slot");
    }
    const int32_t position = slot / 2;
    const int32_t bus = ordered_bus[static_cast<std::size_t>(position)];
    return (slot % 2 == 0) ? is_p_active(bus) : is_q_active(bus);
}

BusLocalIndex build_bus_local_index(int32_t n_bus,
                                    const std::vector<int32_t>& pv,
                                    const std::vector<int32_t>& pq,
                                    const BusOrdering& ordering)
{
    if (n_bus <= 0) {
        throw std::runtime_error("bus-local index requires at least one bus");
    }
    validate_bus_list(n_bus, pv, "PV");
    validate_bus_list(n_bus, pq, "PQ");
    if (ordering.ordered_bus.size() != static_cast<std::size_t>(n_bus) ||
        ordering.bus_to_position.size() != static_cast<std::size_t>(n_bus)) {
        throw std::runtime_error("bus ordering does not match n_bus");
    }

    BusLocalIndex index;
    index.n_bus = n_bus;
    index.dim = 2 * n_bus;
    index.ordered_bus = ordering.ordered_bus;
    index.bus_to_position = ordering.bus_to_position;
    index.bus_type.assign(static_cast<std::size_t>(n_bus), BusType::Slack);
    index.theta_slot.assign(static_cast<std::size_t>(n_bus), -1);
    index.vm_slot.assign(static_cast<std::size_t>(n_bus), -1);
    index.p_equation.assign(static_cast<std::size_t>(n_bus), BusEquationKind::Fixed);
    index.q_equation.assign(static_cast<std::size_t>(n_bus), BusEquationKind::Fixed);

    for (int32_t bus : pv) {
        index.bus_type[static_cast<std::size_t>(bus)] = BusType::Pv;
        index.p_equation[static_cast<std::size_t>(bus)] = BusEquationKind::Active;
        index.q_equation[static_cast<std::size_t>(bus)] = BusEquationKind::Fixed;
    }
    for (int32_t bus : pq) {
        if (index.bus_type[static_cast<std::size_t>(bus)] != BusType::Slack) {
            throw std::runtime_error("bus cannot be both PV and PQ");
        }
        index.bus_type[static_cast<std::size_t>(bus)] = BusType::Pq;
        index.p_equation[static_cast<std::size_t>(bus)] = BusEquationKind::Active;
        index.q_equation[static_cast<std::size_t>(bus)] = BusEquationKind::Active;
    }

    for (int32_t position = 0; position < n_bus; ++position) {
        const int32_t bus = index.ordered_bus[static_cast<std::size_t>(position)];
        index.theta_slot[static_cast<std::size_t>(bus)] = 2 * position;
        index.vm_slot[static_cast<std::size_t>(bus)] = 2 * position + 1;
    }

    return index;
}

}  // namespace exp_20260414::amgx_v2
