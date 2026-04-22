// ---------------------------------------------------------------------------
// jacobian_analysis.cpp
//
// One-time Jacobian analysis for Newton-Raphson power flow.
//
//   1. make_jacobian_indexing()
//      Builds PV/PQ indexing tables.
//   2. JacobianPatternGenerator::generate()
//      Builds the CSR sparsity pattern of the 2x2 Jacobian.
//   3. JacobianMapBuilder::build()
//      Builds scatter maps from Ybus entries/buses into Jacobian value slots.
// ---------------------------------------------------------------------------

#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

constexpr int32_t kLinearCoeffSearchLimit = 16;

struct ScatterPositions {
    int32_t p11 = -1;
    int32_t p21 = -1;
    int32_t p12 = -1;
    int32_t p22 = -1;
};

void require_pointer(const int32_t* ptr, const char* name, int32_t count)
{
    if (count > 0 && ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
}

int32_t find_coeff_index(const JacobianPattern& pattern, int32_t row, int32_t col)
{
    if (row < 0 || col < 0 || row >= pattern.dim || col >= pattern.dim) {
        return -1;
    }

    const int32_t begin = pattern.row_ptr[row];
    const int32_t end = pattern.row_ptr[row + 1];
    const int32_t row_nnz = end - begin;
    const int32_t* cols = pattern.col_idx.data();

    if (row_nnz <= kLinearCoeffSearchLimit) {
        for (int32_t pos = begin; pos < end; ++pos) {
            const int32_t value = cols[pos];
            if (value == col) {
                return pos;
            }
            if (value > col) {
                return -1;
            }
        }
        return -1;
    }

    const int32_t* first = cols + begin;
    const int32_t* last = cols + end;
    const int32_t* it = std::lower_bound(first, last, col);
    if (it == last || *it != col) {
        return -1;
    }
    return static_cast<int32_t>(it - cols);
}

int32_t first_ybus_entry(
    const YbusView& ybus,
    int32_t row,
    int32_t col,
    bool include_equal)
{
    const int32_t begin = ybus.indptr[row];
    const int32_t end = ybus.indptr[row + 1];
    const int32_t* first = ybus.indices + begin;
    const int32_t* last = ybus.indices + end;
    const int32_t* it = include_equal
        ? std::lower_bound(first, last, col)
        : std::upper_bound(first, last, col);
    return static_cast<int32_t>(begin + (it - first));
}

void append_jacobian_columns(
    std::vector<std::vector<int32_t>>& rows,
    const JacobianIndexing& indexing,
    int32_t row_bus,
    int32_t col_bus)
{
    const int32_t ri_pvpq = indexing.row_pvpq[row_bus];
    const int32_t ri_pq = indexing.row_pq[row_bus];
    const int32_t cj_pvpq = indexing.col_pvpq[col_bus];
    const int32_t cj_pq = indexing.col_pq[col_bus];

    if (ri_pvpq >= 0 && cj_pvpq >= 0) {
        rows[ri_pvpq].push_back(cj_pvpq);
    }
    if (ri_pq >= 0 && cj_pvpq >= 0) {
        rows[ri_pq].push_back(cj_pvpq);
    }
    if (ri_pvpq >= 0 && cj_pq >= 0) {
        rows[ri_pvpq].push_back(cj_pq);
    }
    if (ri_pq >= 0 && cj_pq >= 0) {
        rows[ri_pq].push_back(cj_pq);
    }
}

ScatterPositions find_directed_scatter_positions(
    const JacobianPattern& pattern,
    const JacobianIndexing& indexing,
    int32_t row_bus,
    int32_t col_bus)
{
    const int32_t ri_pvpq = indexing.row_pvpq[row_bus];
    const int32_t ri_pq = indexing.row_pq[row_bus];
    const int32_t cj_pvpq = indexing.col_pvpq[col_bus];
    const int32_t cj_pq = indexing.col_pq[col_bus];

    ScatterPositions positions;

    if (ri_pvpq >= 0 && cj_pvpq >= 0) {
        positions.p11 = find_coeff_index(pattern, ri_pvpq, cj_pvpq);
    }
    if (ri_pq >= 0 && cj_pvpq >= 0) {
        positions.p21 = find_coeff_index(pattern, ri_pq, cj_pvpq);
    }
    if (ri_pvpq >= 0 && cj_pq >= 0) {
        positions.p12 = find_coeff_index(pattern, ri_pvpq, cj_pq);
    }
    if (ri_pq >= 0 && cj_pq >= 0) {
        positions.p22 = find_coeff_index(pattern, ri_pq, cj_pq);
    }

    return positions;
}

}  // namespace


JacobianIndexing make_jacobian_indexing(
    int32_t n_bus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    if (n_bus <= 0 || n_pv < 0 || n_pq < 0 || n_pv + n_pq <= 0) {
        throw std::invalid_argument("make_jacobian_indexing: invalid dimensions");
    }
    require_pointer(pv, "pv", n_pv);
    require_pointer(pq, "pq", n_pq);

    JacobianIndexing indexing;
    indexing.n_bus = n_bus;
    indexing.n_pvpq = n_pv + n_pq;
    indexing.n_pq = n_pq;
    indexing.pvpq.resize(indexing.n_pvpq);

    indexing.row_pvpq.assign(n_bus, -1);
    indexing.row_pq.assign(n_bus, -1);
    indexing.col_pvpq.assign(n_bus, -1);
    indexing.col_pq.assign(n_bus, -1);

    for (int32_t i = 0; i < n_pv; ++i) {
        const int32_t bus = pv[i];
        if (bus < 0 || bus >= n_bus) {
            throw std::invalid_argument("make_jacobian_indexing: pv bus index out of range");
        }
        indexing.pvpq[i] = bus;
        indexing.row_pvpq[bus] = i;
        indexing.col_pvpq[bus] = i;
    }

    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[i];
        if (bus < 0 || bus >= n_bus) {
            throw std::invalid_argument("make_jacobian_indexing: pq bus index out of range");
        }
        const int32_t pvpq_idx = n_pv + i;
        const int32_t pq_idx = indexing.n_pvpq + i;
        indexing.pvpq[pvpq_idx] = bus;
        indexing.row_pvpq[bus] = pvpq_idx;
        indexing.col_pvpq[bus] = pvpq_idx;
        indexing.row_pq[bus] = pq_idx;
        indexing.col_pq[bus] = pq_idx;
    }

    return indexing;
}


JacobianPattern JacobianPatternGenerator::generate(
    const YbusView& ybus,
    const JacobianIndexing& indexing) const
{
    if (indexing.n_bus != ybus.rows) {
        throw std::invalid_argument("JacobianPatternGenerator::generate: indexing/Ybus size mismatch");
    }

    const int32_t dim = indexing.n_pvpq + indexing.n_pq;
    std::vector<std::vector<int32_t>> rows(dim);
    for (int32_t bus = 0; bus < ybus.rows; ++bus) {
        const int32_t y_degree = ybus.indptr[bus + 1] - ybus.indptr[bus];
        const int32_t reserve_count = 2 * y_degree + 2;

        const int32_t ri_pvpq = indexing.row_pvpq[bus];
        const int32_t ri_pq = indexing.row_pq[bus];

        if (ri_pvpq >= 0) {
            rows[ri_pvpq].reserve(reserve_count);
        }
        if (ri_pq >= 0) {
            rows[ri_pq].reserve(reserve_count);
        }
    }

    for (int32_t row = 0; row < ybus.rows; ++row) {
        const int32_t upper_begin = first_ybus_entry(ybus, row, row, false);
        for (int32_t k = upper_begin; k < ybus.indptr[row + 1]; ++k) {
            const int32_t col = ybus.indices[k];
            if (col < 0 || col >= ybus.cols) {
                throw std::invalid_argument("JacobianPatternGenerator::generate: Ybus column index out of range");
            }
            append_jacobian_columns(rows, indexing, row, col);
            append_jacobian_columns(rows, indexing, col, row);
        }
    }

    for (int32_t bus = 0; bus < ybus.rows; ++bus) {
        append_jacobian_columns(rows, indexing, bus, bus);
    }

    JacobianPattern pattern;
    pattern.dim = dim;
    pattern.row_ptr.assign(dim + 1, 0);

    int64_t nnz = 0;
    for (int32_t row = 0; row < dim; ++row) {
        auto& cols = rows[row];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

        nnz += cols.size();
        if (nnz > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("JacobianPatternGenerator::generate: Jacobian nnz exceeds int32 range");
        }
        pattern.row_ptr[row + 1] = static_cast<int32_t>(nnz);
    }

    pattern.nnz = static_cast<int32_t>(nnz);
    pattern.col_idx.reserve(nnz);
    for (const auto& cols : rows) {
        pattern.col_idx.insert(pattern.col_idx.end(), cols.begin(), cols.end());
    }

    return pattern;
}


JacobianScatterMap JacobianMapBuilder::build(
    const YbusView& ybus,
    const JacobianIndexing& indexing,
    const JacobianPattern& pattern) const
{
    if (indexing.n_bus != ybus.rows) {
        throw std::invalid_argument("JacobianMapBuilder::build: indexing/Ybus size mismatch");
    }
    if (pattern.dim != indexing.n_pvpq + indexing.n_pq) {
        throw std::invalid_argument("JacobianMapBuilder::build: pattern dimension mismatch");
    }

    JacobianScatterMap map;
    map.n_pvpq = indexing.n_pvpq;
    map.n_pq = indexing.n_pq;
    map.pvpq = indexing.pvpq;

    map.mapJ11.assign(ybus.nnz, -1);
    map.mapJ12.assign(ybus.nnz, -1);
    map.mapJ21.assign(ybus.nnz, -1);
    map.mapJ22.assign(ybus.nnz, -1);
    map.diagJ11.assign(ybus.rows, -1);
    map.diagJ12.assign(ybus.rows, -1);
    map.diagJ21.assign(ybus.rows, -1);
    map.diagJ22.assign(ybus.rows, -1);

    for (int32_t row = 0; row < ybus.rows; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            const int32_t col = ybus.indices[k];
            if (col < 0 || col >= ybus.cols) {
                throw std::invalid_argument("JacobianMapBuilder::build: Ybus column index out of range");
            }

            const ScatterPositions positions =
                find_directed_scatter_positions(pattern, indexing, row, col);

            map.mapJ11[k] = positions.p11;
            map.mapJ21[k] = positions.p21;
            map.mapJ12[k] = positions.p12;
            map.mapJ22[k] = positions.p22;

            if (row == col) {
                const int32_t bus = row;
                map.diagJ11[bus] = positions.p11;
                map.diagJ21[bus] = positions.p21;
                map.diagJ12[bus] = positions.p12;
                map.diagJ22[bus] = positions.p22;
            }
        }
    }

    return map;
}
