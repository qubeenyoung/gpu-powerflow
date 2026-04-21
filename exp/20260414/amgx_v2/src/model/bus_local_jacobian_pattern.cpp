#include "bus_local_jacobian_pattern.hpp"

#include <algorithm>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

void add_unique(std::vector<int32_t>& row, int32_t col)
{
    if (col >= 0) {
        row.push_back(col);
    }
}

void finalize_row(std::vector<int32_t>& row)
{
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
}

}  // namespace

CsrMatrixView BusLocalJacobianPattern::host_view(const std::vector<double>& values) const
{
    if (values.size() != static_cast<std::size_t>(nnz)) {
        throw std::runtime_error("value array size does not match bus-local Jacobian pattern");
    }
    return CsrMatrixView{
        .rows = dim,
        .nnz = nnz,
        .row_ptr = row_ptr.data(),
        .col_idx = col_idx.data(),
        .values = values.data(),
    };
}

BusLocalJacobianPattern build_bus_local_jacobian_pattern(
    const BusLocalIndex& index,
    const std::vector<int32_t>& ybus_row_ptr,
    const std::vector<int32_t>& ybus_col_idx)
{
    if (ybus_row_ptr.size() != static_cast<std::size_t>(index.n_bus + 1)) {
        throw std::runtime_error("Ybus row_ptr size does not match bus-local index");
    }
    if (ybus_row_ptr.front() != 0 ||
        ybus_row_ptr.back() != static_cast<int32_t>(ybus_col_idx.size())) {
        throw std::runtime_error("Ybus CSR pointers are inconsistent");
    }

    std::vector<std::vector<int32_t>> rows(static_cast<std::size_t>(index.dim));

    for (int32_t position = 0; position < index.n_bus; ++position) {
        const int32_t row_bus = index.ordered_bus[static_cast<std::size_t>(position)];
        const int32_t p_row = index.theta(row_bus);
        const int32_t q_row = index.vm(row_bus);

        if (index.is_p_active(row_bus)) {
            for (int32_t pos = ybus_row_ptr[row_bus]; pos < ybus_row_ptr[row_bus + 1]; ++pos) {
                const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(pos)];
                add_unique(rows[static_cast<std::size_t>(p_row)], index.theta(col_bus));
                add_unique(rows[static_cast<std::size_t>(p_row)], index.vm(col_bus));
            }
            add_unique(rows[static_cast<std::size_t>(p_row)], index.theta(row_bus));
            add_unique(rows[static_cast<std::size_t>(p_row)], index.vm(row_bus));
        } else {
            // Fixed slots become identity rows. This keeps the augmented
            // 2-slot-per-bus system nonsingular and forces the update to zero.
            add_unique(rows[static_cast<std::size_t>(p_row)], p_row);
        }

        if (index.is_q_active(row_bus)) {
            for (int32_t pos = ybus_row_ptr[row_bus]; pos < ybus_row_ptr[row_bus + 1]; ++pos) {
                const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(pos)];
                add_unique(rows[static_cast<std::size_t>(q_row)], index.theta(col_bus));
                add_unique(rows[static_cast<std::size_t>(q_row)], index.vm(col_bus));
            }
            add_unique(rows[static_cast<std::size_t>(q_row)], index.theta(row_bus));
            add_unique(rows[static_cast<std::size_t>(q_row)], index.vm(row_bus));
        } else {
            add_unique(rows[static_cast<std::size_t>(q_row)], q_row);
        }
    }

    BusLocalJacobianPattern pattern;
    pattern.dim = index.dim;
    pattern.row_ptr.assign(static_cast<std::size_t>(pattern.dim + 1), 0);
    for (auto& row : rows) {
        finalize_row(row);
    }
    for (int32_t row = 0; row < pattern.dim; ++row) {
        pattern.row_ptr[static_cast<std::size_t>(row + 1)] =
            pattern.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(rows[static_cast<std::size_t>(row)].size());
    }
    pattern.nnz = pattern.row_ptr.back();
    pattern.col_idx.reserve(static_cast<std::size_t>(pattern.nnz));
    for (const auto& row : rows) {
        pattern.col_idx.insert(pattern.col_idx.end(), row.begin(), row.end());
    }

    return pattern;
}

}  // namespace exp_20260414::amgx_v2
