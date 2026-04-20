#include "jacobian_build.hpp"

#include <algorithm>

namespace {

using PatternRows = std::vector<std::vector<int32_t>>;


void push_pattern(PatternRows& rows, int32_t row, int32_t col)
{
    if (row >= 0 && col >= 0) {
        rows[row].push_back(col);
    }
}

}  // namespace


BusIndexMap buildBusIndexMap(const YbusGraph& ybus,
                             const int32_t* pv,
                             int32_t n_pv,
                             const int32_t* pq,
                             int32_t n_pq)
{
    BusIndexMap index;
    index.n_pvpq = n_pv + n_pq;
    index.n_pq = n_pq;
    index.dim = index.n_pvpq + n_pq;
    index.pvpq.resize(index.n_pvpq);
    index.bus_to_pvpq.assign(ybus.n_bus, -1);
    index.bus_to_pq.assign(ybus.n_bus, -1);

    for (int32_t pos = 0; pos < n_pv; ++pos) {
        const int32_t bus = pv[pos];
        index.pvpq[pos] = bus;
        index.bus_to_pvpq[bus] = pos;
    }

    for (int32_t pos = 0; pos < n_pq; ++pos) {
        const int32_t bus = pq[pos];
        const int32_t pvpq_pos = n_pv + pos;
        index.pvpq[pvpq_pos] = bus;
        index.bus_to_pvpq[bus] = pvpq_pos;
        index.bus_to_pq[bus] = index.n_pvpq + pos;
    }

    return index;
}


JacobianPattern buildJacobianPattern(const YbusGraph& ybus,
                                     const BusIndexMap& index)
{
    PatternRows rows(index.dim);
    for (int32_t k = 0; k < ybus.n_edges; ++k) {
        const int32_t i = ybus.row[k];
        const int32_t j = ybus.col[k];
        if (i == j) {
            continue;
        }

        const int32_t row_pvpq = index.bus_to_pvpq[i];
        const int32_t row_pq = index.bus_to_pq[i];
        const int32_t col_pvpq = index.bus_to_pvpq[j];
        const int32_t col_pq = index.bus_to_pq[j];

        push_pattern(rows, row_pvpq, col_pvpq);
        push_pattern(rows, row_pq, col_pvpq);
        push_pattern(rows, row_pvpq, col_pq);
        push_pattern(rows, row_pq, col_pq);
    }

    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t pvpq = index.bus_to_pvpq[bus];
        const int32_t pq = index.bus_to_pq[bus];

        push_pattern(rows, pvpq, pvpq);
        push_pattern(rows, pq, pvpq);
        push_pattern(rows, pvpq, pq);
        push_pattern(rows, pq, pq);
    }

    JacobianPattern pattern;
    pattern.dim = index.dim;
    pattern.row_ptr.assign(pattern.dim + 1, 0);

    for (int32_t row = 0; row < pattern.dim; ++row) {
        auto& cols = rows[row];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        const int32_t row_nnz = cols.size();
        pattern.row_ptr[row + 1] = pattern.row_ptr[row] + row_nnz;
    }

    pattern.nnz = pattern.row_ptr.back();
    pattern.col_idx.reserve(pattern.nnz);
    for (const auto& cols : rows) {
        pattern.col_idx.insert(pattern.col_idx.end(), cols.begin(), cols.end());
    }

    return pattern;
}


JacobianMap buildJacobianMap(const YbusGraph& ybus,
                             const BusIndexMap& index,
                             const JacobianPattern& pattern)
{
    const CoeffLookup lookup = buildCoeffLookup(pattern);

    JacobianMap map;
    map.offdiagJ11.assign(ybus.n_edges, -1);
    map.offdiagJ12.assign(ybus.n_edges, -1);
    map.offdiagJ21.assign(ybus.n_edges, -1);
    map.offdiagJ22.assign(ybus.n_edges, -1);

    map.diagJ11.assign(ybus.n_bus, -1);
    map.diagJ12.assign(ybus.n_bus, -1);
    map.diagJ21.assign(ybus.n_bus, -1);
    map.diagJ22.assign(ybus.n_bus, -1);

    for (int32_t k = 0; k < ybus.n_edges; ++k) {
        const int32_t i = ybus.row[k];
        const int32_t j = ybus.col[k];
        if (i == j) {
            continue;
        }

        const int32_t row_pvpq = index.bus_to_pvpq[i];
        const int32_t row_pq = index.bus_to_pq[i];
        const int32_t col_pvpq = index.bus_to_pvpq[j];
        const int32_t col_pq = index.bus_to_pq[j];

        map.offdiagJ11[k] = coeffIndex(lookup, row_pvpq, col_pvpq);
        map.offdiagJ21[k] = coeffIndex(lookup, row_pq, col_pvpq);
        map.offdiagJ12[k] = coeffIndex(lookup, row_pvpq, col_pq);
        map.offdiagJ22[k] = coeffIndex(lookup, row_pq, col_pq);
    }

    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t pvpq = index.bus_to_pvpq[bus];
        const int32_t pq = index.bus_to_pq[bus];

        map.diagJ11[bus] = coeffIndex(lookup, pvpq, pvpq);
        map.diagJ21[bus] = coeffIndex(lookup, pq, pvpq);
        map.diagJ12[bus] = coeffIndex(lookup, pvpq, pq);
        map.diagJ22[bus] = coeffIndex(lookup, pq, pq);
    }

    return map;
}


JacobianBuild buildJacobian(const YbusGraph& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq)
{
    JacobianBuild build;
    build.index = buildBusIndexMap(ybus, pv, n_pv, pq, n_pq);
    build.pattern = buildJacobianPattern(ybus, build.index);
    build.map = buildJacobianMap(ybus, build.index, build.pattern);
    return build;
}


CoeffLookup buildCoeffLookup(const JacobianPattern& pattern)
{
    CoeffLookup lookup(pattern.dim);

    for (int32_t row = 0; row < pattern.dim; ++row) {
        const int32_t row_begin = pattern.row_ptr[row];
        const int32_t row_end = pattern.row_ptr[row + 1];
        lookup[row].reserve(row_end - row_begin);

        for (int32_t pos = row_begin; pos < row_end; ++pos) {
            lookup[row][pattern.col_idx[pos]] = pos;
        }
    }

    return lookup;
}


int32_t coeffIndex(const CoeffLookup& lookup, int32_t row, int32_t col)
{
    if (row < 0 || col < 0 || row >= lookup.size()) {
        return -1;
    }

    const auto found = lookup[row].find(col);
    if (found == lookup[row].end()) {
        return -1;
    }
    return found->second;
}
