#include "jacobian_build.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace exp20260421::vertex_edge::newton_solver {
namespace {

uint64_t make_key(int32_t row, int32_t col)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint32_t>(col);
}

int32_t key_row(uint64_t key)
{
    return static_cast<int32_t>(key >> 32);
}

int32_t key_col(uint64_t key)
{
    return static_cast<int32_t>(key & 0xffffffffu);
}

void add_pattern(std::vector<uint64_t>& keys, int32_t row, int32_t col)
{
    if (row >= 0 && col >= 0) {
        keys.push_back(make_key(row, col));
    }
}

int32_t find_coeff_index(const JacobianPattern& pattern, int32_t row, int32_t col)
{
    if (row < 0 || col < 0) {
        return -1;
    }

    const int32_t begin = pattern.row_ptr[static_cast<std::size_t>(row)];
    const int32_t end = pattern.row_ptr[static_cast<std::size_t>(row + 1)];
    const auto first = pattern.col_idx.begin() + begin;
    const auto last = pattern.col_idx.begin() + end;
    const auto it = std::lower_bound(first, last, col);
    if (it == last || *it != col) {
        return -1;
    }
    return static_cast<int32_t>(it - pattern.col_idx.begin());
}

}  // namespace

JacobianBuild buildJacobian(const YbusGraph& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq)
{
    if (ybus.n_bus <= 0 || ybus.n_edges <= 0 ||
        ybus.row_ptr == nullptr || ybus.col == nullptr) {
        throw std::invalid_argument("buildJacobian: bad Ybus graph");
    }
    if (n_pv < 0 || n_pq < 0 || n_pv + n_pq <= 0) {
        throw std::invalid_argument("buildJacobian: bad pv/pq sizes");
    }
    if ((n_pv > 0 && pv == nullptr) || (n_pq > 0 && pq == nullptr)) {
        throw std::invalid_argument("buildJacobian: pv/pq pointer is null");
    }

    JacobianBuild build;
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim = n_pvpq + n_pq;

    build.index.n_pvpq = n_pvpq;
    build.index.n_pq = n_pq;
    build.index.dim = dim;
    build.index.pvpq.resize(static_cast<std::size_t>(n_pvpq));
    build.index.bus_to_pvpq.assign(static_cast<std::size_t>(ybus.n_bus), -1);
    build.index.bus_to_pq.assign(static_cast<std::size_t>(ybus.n_bus), -1);
    std::vector<uint8_t> bus_mark(static_cast<std::size_t>(ybus.n_bus), 0);
    for (int32_t i = 0; i < n_pv; ++i) {
        const int32_t bus = pv[i];
        if (bus < 0 || bus >= ybus.n_bus) {
            throw std::invalid_argument("buildJacobian: pv bus index out of range");
        }
        if (bus_mark[static_cast<std::size_t>(bus)] != 0) {
            throw std::invalid_argument("buildJacobian: duplicate pv bus index");
        }
        bus_mark[static_cast<std::size_t>(bus)] = 1;
        build.index.pvpq[static_cast<std::size_t>(i)] = bus;
        build.index.bus_to_pvpq[static_cast<std::size_t>(bus)] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[i];
        if (bus < 0 || bus >= ybus.n_bus) {
            throw std::invalid_argument("buildJacobian: pq bus index out of range");
        }
        if (bus_mark[static_cast<std::size_t>(bus)] == 1) {
            throw std::invalid_argument("buildJacobian: pv/pq bus sets overlap");
        }
        if (bus_mark[static_cast<std::size_t>(bus)] == 2) {
            throw std::invalid_argument("buildJacobian: duplicate pq bus index");
        }
        bus_mark[static_cast<std::size_t>(bus)] = 2;
        build.index.pvpq[static_cast<std::size_t>(n_pv + i)] = bus;
        build.index.bus_to_pvpq[static_cast<std::size_t>(bus)] = n_pv + i;
        build.index.bus_to_pq[static_cast<std::size_t>(bus)] = n_pvpq + i;
    }

    std::vector<int32_t> rmap_pvpq(static_cast<std::size_t>(ybus.n_bus), -1);
    std::vector<int32_t> rmap_pq(static_cast<std::size_t>(ybus.n_bus), -1);
    std::vector<int32_t> cmap_pvpq(static_cast<std::size_t>(ybus.n_bus), -1);
    std::vector<int32_t> cmap_pq(static_cast<std::size_t>(ybus.n_bus), -1);

    for (int32_t i = 0; i < n_pvpq; ++i) {
        const int32_t bus = build.index.pvpq[static_cast<std::size_t>(i)];
        rmap_pvpq[static_cast<std::size_t>(bus)] = i;
        cmap_pvpq[static_cast<std::size_t>(bus)] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[i];
        rmap_pq[static_cast<std::size_t>(bus)] = i + n_pvpq;
        cmap_pq[static_cast<std::size_t>(bus)] = i + n_pvpq;
    }

    std::vector<uint64_t> keys;
    keys.reserve(static_cast<std::size_t>(4 * ybus.n_edges + 4 * ybus.n_bus));

    for (int32_t row = 0; row < ybus.n_bus; ++row) {
        for (int32_t k = ybus.row_ptr[row]; k < ybus.row_ptr[row + 1]; ++k) {
            const int32_t col = ybus.col[k];
            if (col < 0 || col >= ybus.n_bus) {
                throw std::invalid_argument("buildJacobian: Ybus column index out of range");
            }
            if (row == col) {
                continue;
            }

            const int32_t ri_pvpq = rmap_pvpq[static_cast<std::size_t>(row)];
            const int32_t ri_pq = rmap_pq[static_cast<std::size_t>(row)];
            const int32_t cj_pvpq = cmap_pvpq[static_cast<std::size_t>(col)];
            const int32_t cj_pq = cmap_pq[static_cast<std::size_t>(col)];

            add_pattern(keys, ri_pvpq, cj_pvpq);
            add_pattern(keys, ri_pq, cj_pvpq);
            add_pattern(keys, ri_pvpq, cj_pq);
            add_pattern(keys, ri_pq, cj_pq);
        }
    }

    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t ri_pvpq = rmap_pvpq[static_cast<std::size_t>(bus)];
        const int32_t ri_pq = rmap_pq[static_cast<std::size_t>(bus)];
        const int32_t cj_pvpq = cmap_pvpq[static_cast<std::size_t>(bus)];
        const int32_t cj_pq = cmap_pq[static_cast<std::size_t>(bus)];

        add_pattern(keys, ri_pvpq, cj_pvpq);
        add_pattern(keys, ri_pq, cj_pvpq);
        add_pattern(keys, ri_pvpq, cj_pq);
        add_pattern(keys, ri_pq, cj_pq);
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    build.pattern.dim = dim;
    build.pattern.nnz = static_cast<int32_t>(keys.size());
    build.pattern.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    build.pattern.col_idx.resize(keys.size());

    for (uint64_t key : keys) {
        ++build.pattern.row_ptr[static_cast<std::size_t>(key_row(key) + 1)];
    }
    for (int32_t row = 0; row < dim; ++row) {
        build.pattern.row_ptr[static_cast<std::size_t>(row + 1)] +=
            build.pattern.row_ptr[static_cast<std::size_t>(row)];
    }
    for (std::size_t pos = 0; pos < keys.size(); ++pos) {
        build.pattern.col_idx[pos] = key_col(keys[pos]);
    }

    build.map.offdiagJ11.assign(static_cast<std::size_t>(ybus.n_edges), -1);
    build.map.offdiagJ12.assign(static_cast<std::size_t>(ybus.n_edges), -1);
    build.map.offdiagJ21.assign(static_cast<std::size_t>(ybus.n_edges), -1);
    build.map.offdiagJ22.assign(static_cast<std::size_t>(ybus.n_edges), -1);

    int32_t edge = 0;
    for (int32_t row = 0; row < ybus.n_bus; ++row) {
        for (int32_t k = ybus.row_ptr[row]; k < ybus.row_ptr[row + 1]; ++k, ++edge) {
            const int32_t col = ybus.col[k];
            const int32_t ri_pvpq = rmap_pvpq[static_cast<std::size_t>(row)];
            const int32_t ri_pq = rmap_pq[static_cast<std::size_t>(row)];
            const int32_t cj_pvpq = cmap_pvpq[static_cast<std::size_t>(col)];
            const int32_t cj_pq = cmap_pq[static_cast<std::size_t>(col)];

            build.map.offdiagJ11[static_cast<std::size_t>(edge)] =
                find_coeff_index(build.pattern, ri_pvpq, cj_pvpq);
            build.map.offdiagJ21[static_cast<std::size_t>(edge)] =
                find_coeff_index(build.pattern, ri_pq, cj_pvpq);
            build.map.offdiagJ12[static_cast<std::size_t>(edge)] =
                find_coeff_index(build.pattern, ri_pvpq, cj_pq);
            build.map.offdiagJ22[static_cast<std::size_t>(edge)] =
                find_coeff_index(build.pattern, ri_pq, cj_pq);
        }
    }
    if (edge != ybus.n_edges) {
        throw std::invalid_argument("buildJacobian: n_edges does not match row_ptr");
    }

    build.map.diagJ11.assign(static_cast<std::size_t>(ybus.n_bus), -1);
    build.map.diagJ12.assign(static_cast<std::size_t>(ybus.n_bus), -1);
    build.map.diagJ21.assign(static_cast<std::size_t>(ybus.n_bus), -1);
    build.map.diagJ22.assign(static_cast<std::size_t>(ybus.n_bus), -1);

    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t ri_pvpq = rmap_pvpq[static_cast<std::size_t>(bus)];
        const int32_t ri_pq = rmap_pq[static_cast<std::size_t>(bus)];
        const int32_t cj_pvpq = cmap_pvpq[static_cast<std::size_t>(bus)];
        const int32_t cj_pq = cmap_pq[static_cast<std::size_t>(bus)];

        build.map.diagJ11[static_cast<std::size_t>(bus)] =
            find_coeff_index(build.pattern, ri_pvpq, cj_pvpq);
        build.map.diagJ21[static_cast<std::size_t>(bus)] =
            find_coeff_index(build.pattern, ri_pq, cj_pvpq);
        build.map.diagJ12[static_cast<std::size_t>(bus)] =
            find_coeff_index(build.pattern, ri_pvpq, cj_pq);
        build.map.diagJ22[static_cast<std::size_t>(bus)] =
            find_coeff_index(build.pattern, ri_pq, cj_pq);
    }

    return build;
}

}  // namespace exp20260421::vertex_edge::newton_solver
