#include "model/reduced_jacobian.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace exp_20260415::block_ilu {
namespace {

void validate_bus(int32_t n_bus, int32_t bus, const char* label)
{
    if (bus < 0 || bus >= n_bus) {
        throw std::runtime_error(std::string(label) + " bus index is out of range");
    }
}

void validate_bus_list(int32_t n_bus, const std::vector<int32_t>& buses, const char* label)
{
    std::vector<char> seen(static_cast<std::size_t>(n_bus), 0);
    for (int32_t bus : buses) {
        validate_bus(n_bus, bus, label);
        if (seen[static_cast<std::size_t>(bus)] != 0) {
            throw std::runtime_error(std::string(label) + " bus list contains duplicates");
        }
        seen[static_cast<std::size_t>(bus)] = 1;
    }
}

void validate_ybus_csr(int32_t n_bus,
                       const std::vector<int32_t>& row_ptr,
                       const std::vector<int32_t>& col_idx)
{
    if (n_bus <= 0) {
        throw std::runtime_error("Ybus requires a nonempty bus count");
    }
    if (row_ptr.size() != static_cast<std::size_t>(n_bus + 1)) {
        throw std::runtime_error("Ybus row_ptr size does not match n_bus");
    }
    if (row_ptr.front() != 0 ||
        row_ptr.back() != static_cast<int32_t>(col_idx.size())) {
        throw std::runtime_error("Ybus CSR pointers are inconsistent");
    }
    for (int32_t row = 0; row < n_bus; ++row) {
        if (row_ptr[static_cast<std::size_t>(row)] >
            row_ptr[static_cast<std::size_t>(row + 1)]) {
            throw std::runtime_error("Ybus row_ptr is not monotone");
        }
    }
    for (int32_t col : col_idx) {
        validate_bus(n_bus, col, "Ybus column");
    }
}

void add_if_valid(std::vector<int32_t>& row, int32_t col)
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

HostCsrPattern finalize_pattern(int32_t rows_count,
                                int32_t cols_count,
                                std::vector<std::vector<int32_t>> rows)
{
    if (static_cast<int32_t>(rows.size()) != rows_count) {
        throw std::runtime_error("pattern row workspace has wrong size");
    }

    for (auto& row : rows) {
        finalize_row(row);
    }

    HostCsrPattern pattern;
    pattern.rows = rows_count;
    pattern.cols = cols_count;
    pattern.row_ptr.assign(static_cast<std::size_t>(rows_count + 1), 0);
    for (int32_t row = 0; row < rows_count; ++row) {
        pattern.row_ptr[static_cast<std::size_t>(row + 1)] =
            pattern.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(rows[static_cast<std::size_t>(row)].size());
    }

    pattern.col_idx.reserve(static_cast<std::size_t>(pattern.row_ptr.back()));
    for (const auto& row : rows) {
        pattern.col_idx.insert(pattern.col_idx.end(), row.begin(), row.end());
    }
    return pattern;
}

int32_t find_csr_position(const HostCsrPattern& pattern, int32_t row, int32_t col)
{
    if (row < 0 || row >= pattern.rows || col < 0 || col >= pattern.cols) {
        return -1;
    }
    const auto begin = pattern.col_idx.begin() + pattern.row_ptr[static_cast<std::size_t>(row)];
    const auto end = pattern.col_idx.begin() + pattern.row_ptr[static_cast<std::size_t>(row + 1)];
    const auto it = std::lower_bound(begin, end, col);
    if (it == end || *it != col) {
        return -1;
    }
    return static_cast<int32_t>(it - pattern.col_idx.begin());
}

}  // namespace

int32_t HostCsrPattern::nnz() const
{
    return static_cast<int32_t>(col_idx.size());
}

bool ReducedJacobianIndex::has_theta(int32_t bus) const
{
    validate_bus(n_bus, bus, "theta");
    return theta_slot[static_cast<std::size_t>(bus)] >= 0;
}

bool ReducedJacobianIndex::has_vm(int32_t bus) const
{
    validate_bus(n_bus, bus, "Vm");
    return vm_slot[static_cast<std::size_t>(bus)] >= 0;
}

int32_t ReducedJacobianIndex::theta(int32_t bus) const
{
    validate_bus(n_bus, bus, "theta");
    return theta_slot[static_cast<std::size_t>(bus)];
}

int32_t ReducedJacobianIndex::vm(int32_t bus) const
{
    validate_bus(n_bus, bus, "Vm");
    return vm_slot[static_cast<std::size_t>(bus)];
}

int32_t ReducedJacobianIndex::theta_block(int32_t bus) const
{
    validate_bus(n_bus, bus, "theta block");
    return theta_block_slot[static_cast<std::size_t>(bus)];
}

int32_t ReducedJacobianIndex::vm_block(int32_t bus) const
{
    validate_bus(n_bus, bus, "Vm block");
    return vm_block_slot[static_cast<std::size_t>(bus)];
}

ReducedJacobianIndex build_reduced_jacobian_index(int32_t n_bus,
                                                  const std::vector<int32_t>& pv,
                                                  const std::vector<int32_t>& pq)
{
    if (n_bus <= 0) {
        throw std::runtime_error("reduced Jacobian index requires at least one bus");
    }
    validate_bus_list(n_bus, pv, "PV");
    validate_bus_list(n_bus, pq, "PQ");

    ReducedJacobianIndex index;
    index.n_bus = n_bus;
    index.n_pv = static_cast<int32_t>(pv.size());
    index.n_pq = static_cast<int32_t>(pq.size());
    index.n_pvpq = index.n_pv + index.n_pq;
    index.dim = index.n_pvpq + index.n_pq;
    index.pv = pv;
    index.pq = pq;
    index.pvpq.reserve(static_cast<std::size_t>(index.n_pvpq));
    index.pvpq.insert(index.pvpq.end(), pv.begin(), pv.end());
    index.pvpq.insert(index.pvpq.end(), pq.begin(), pq.end());
    index.theta_slot.assign(static_cast<std::size_t>(n_bus), -1);
    index.vm_slot.assign(static_cast<std::size_t>(n_bus), -1);
    index.theta_block_slot.assign(static_cast<std::size_t>(n_bus), -1);
    index.vm_block_slot.assign(static_cast<std::size_t>(n_bus), -1);

    std::vector<char> is_pv(static_cast<std::size_t>(n_bus), 0);
    for (int32_t bus : pv) {
        is_pv[static_cast<std::size_t>(bus)] = 1;
    }

    for (int32_t pos = 0; pos < index.n_pvpq; ++pos) {
        const int32_t bus = index.pvpq[static_cast<std::size_t>(pos)];
        index.theta_slot[static_cast<std::size_t>(bus)] = pos;
        index.theta_block_slot[static_cast<std::size_t>(bus)] = pos;
    }

    for (int32_t pos = 0; pos < index.n_pq; ++pos) {
        const int32_t bus = pq[static_cast<std::size_t>(pos)];
        if (is_pv[static_cast<std::size_t>(bus)] != 0) {
            throw std::runtime_error("bus cannot be both PV and PQ");
        }
        index.vm_slot[static_cast<std::size_t>(bus)] = index.n_pvpq + pos;
        index.vm_block_slot[static_cast<std::size_t>(bus)] = pos;
    }

    return index;
}

ReducedJacobianPatterns build_reduced_jacobian_patterns(
    int32_t n_bus,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq,
    const std::vector<int32_t>& ybus_row_ptr,
    const std::vector<int32_t>& ybus_col_idx)
{
    validate_ybus_csr(n_bus, ybus_row_ptr, ybus_col_idx);

    ReducedJacobianPatterns out;
    out.index = build_reduced_jacobian_index(n_bus, pv, pq);
    const ReducedJacobianIndex& index = out.index;

    std::vector<std::vector<int32_t>> full_rows(static_cast<std::size_t>(index.dim));
    std::vector<std::vector<int32_t>> j11_rows(static_cast<std::size_t>(index.n_pvpq));
    std::vector<std::vector<int32_t>> j12_rows(static_cast<std::size_t>(index.n_pvpq));
    std::vector<std::vector<int32_t>> j21_rows(static_cast<std::size_t>(index.n_pq));
    std::vector<std::vector<int32_t>> j22_rows(static_cast<std::size_t>(index.n_pq));

    for (int32_t row_bus = 0; row_bus < n_bus; ++row_bus) {
        const int32_t full_p_row = index.theta(row_bus);
        const int32_t full_q_row = index.vm(row_bus);
        const int32_t block_p_row = index.theta_block(row_bus);
        const int32_t block_q_row = index.vm_block(row_bus);

        if (full_p_row >= 0) {
            for (int32_t pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
                 pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
                 ++pos) {
                const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(pos)];
                add_if_valid(full_rows[static_cast<std::size_t>(full_p_row)],
                             index.theta(col_bus));
                add_if_valid(full_rows[static_cast<std::size_t>(full_p_row)],
                             index.vm(col_bus));
                add_if_valid(j11_rows[static_cast<std::size_t>(block_p_row)],
                             index.theta_block(col_bus));
                add_if_valid(j12_rows[static_cast<std::size_t>(block_p_row)],
                             index.vm_block(col_bus));
            }
            add_if_valid(full_rows[static_cast<std::size_t>(full_p_row)], index.theta(row_bus));
            add_if_valid(full_rows[static_cast<std::size_t>(full_p_row)], index.vm(row_bus));
            add_if_valid(j11_rows[static_cast<std::size_t>(block_p_row)],
                         index.theta_block(row_bus));
            add_if_valid(j12_rows[static_cast<std::size_t>(block_p_row)],
                         index.vm_block(row_bus));
        }

        if (full_q_row >= 0) {
            for (int32_t pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
                 pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
                 ++pos) {
                const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(pos)];
                add_if_valid(full_rows[static_cast<std::size_t>(full_q_row)],
                             index.theta(col_bus));
                add_if_valid(full_rows[static_cast<std::size_t>(full_q_row)],
                             index.vm(col_bus));
                add_if_valid(j22_rows[static_cast<std::size_t>(block_q_row)],
                             index.vm_block(col_bus));
                add_if_valid(j21_rows[static_cast<std::size_t>(block_q_row)],
                             index.theta_block(col_bus));
            }
            add_if_valid(full_rows[static_cast<std::size_t>(full_q_row)], index.theta(row_bus));
            add_if_valid(full_rows[static_cast<std::size_t>(full_q_row)], index.vm(row_bus));
            add_if_valid(j22_rows[static_cast<std::size_t>(block_q_row)],
                         index.vm_block(row_bus));
            add_if_valid(j21_rows[static_cast<std::size_t>(block_q_row)],
                         index.theta_block(row_bus));
        }
    }

    out.full = finalize_pattern(index.dim, index.dim, std::move(full_rows));
    out.j11 = finalize_pattern(index.n_pvpq, index.n_pvpq, std::move(j11_rows));
    out.j12 = finalize_pattern(index.n_pvpq, index.n_pq, std::move(j12_rows));
    out.j21 = finalize_pattern(index.n_pq, index.n_pvpq, std::move(j21_rows));
    out.j22 = finalize_pattern(index.n_pq, index.n_pq, std::move(j22_rows));

    const int32_t ybus_nnz = static_cast<int32_t>(ybus_col_idx.size());
    out.ybus_row.assign(static_cast<std::size_t>(ybus_nnz), 0);
    out.ybus_col = ybus_col_idx;
    out.full_maps.map11.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.full_maps.map12.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.full_maps.map21.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.full_maps.map22.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.j11_maps.map.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.j12_maps.map.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.j21_maps.map.assign(static_cast<std::size_t>(ybus_nnz), -1);
    out.j22_maps.map.assign(static_cast<std::size_t>(ybus_nnz), -1);

    for (int32_t row_bus = 0; row_bus < n_bus; ++row_bus) {
        for (int32_t y_pos = ybus_row_ptr[static_cast<std::size_t>(row_bus)];
             y_pos < ybus_row_ptr[static_cast<std::size_t>(row_bus + 1)];
             ++y_pos) {
            const int32_t col_bus = ybus_col_idx[static_cast<std::size_t>(y_pos)];
            out.ybus_row[static_cast<std::size_t>(y_pos)] = row_bus;

            const int32_t p_row = index.theta(row_bus);
            const int32_t q_row = index.vm(row_bus);
            if (p_row >= 0) {
                out.full_maps.map11[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.full, p_row, index.theta(col_bus));
                out.full_maps.map12[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.full, p_row, index.vm(col_bus));
                out.j11_maps.map[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.j11,
                                      index.theta_block(row_bus),
                                      index.theta_block(col_bus));
                out.j12_maps.map[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.j12,
                                      index.theta_block(row_bus),
                                      index.vm_block(col_bus));
            }
            if (q_row >= 0) {
                out.full_maps.map21[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.full, q_row, index.theta(col_bus));
                out.full_maps.map22[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.full, q_row, index.vm(col_bus));
                out.j21_maps.map[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.j21,
                                      index.vm_block(row_bus),
                                      index.theta_block(col_bus));
                out.j22_maps.map[static_cast<std::size_t>(y_pos)] =
                    find_csr_position(out.j22,
                                      index.vm_block(row_bus),
                                      index.vm_block(col_bus));
            }
        }
    }

    out.full_maps.diag11.assign(static_cast<std::size_t>(n_bus), -1);
    out.full_maps.diag12.assign(static_cast<std::size_t>(n_bus), -1);
    out.full_maps.diag21.assign(static_cast<std::size_t>(n_bus), -1);
    out.full_maps.diag22.assign(static_cast<std::size_t>(n_bus), -1);
    out.j11_maps.diag.assign(static_cast<std::size_t>(n_bus), -1);
    out.j12_maps.diag.assign(static_cast<std::size_t>(n_bus), -1);
    out.j21_maps.diag.assign(static_cast<std::size_t>(n_bus), -1);
    out.j22_maps.diag.assign(static_cast<std::size_t>(n_bus), -1);

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const int32_t p_row = index.theta(bus);
        const int32_t q_row = index.vm(bus);
        if (p_row >= 0) {
            out.full_maps.diag11[static_cast<std::size_t>(bus)] =
                find_csr_position(out.full, p_row, index.theta(bus));
            out.full_maps.diag12[static_cast<std::size_t>(bus)] =
                find_csr_position(out.full, p_row, index.vm(bus));
            out.j11_maps.diag[static_cast<std::size_t>(bus)] =
                find_csr_position(out.j11,
                                  index.theta_block(bus),
                                  index.theta_block(bus));
            out.j12_maps.diag[static_cast<std::size_t>(bus)] =
                find_csr_position(out.j12,
                                  index.theta_block(bus),
                                  index.vm_block(bus));
        }
        if (q_row >= 0) {
            out.full_maps.diag21[static_cast<std::size_t>(bus)] =
                find_csr_position(out.full, q_row, index.theta(bus));
            out.full_maps.diag22[static_cast<std::size_t>(bus)] =
                find_csr_position(out.full, q_row, index.vm(bus));
            out.j21_maps.diag[static_cast<std::size_t>(bus)] =
                find_csr_position(out.j21,
                                  index.vm_block(bus),
                                  index.theta_block(bus));
            out.j22_maps.diag[static_cast<std::size_t>(bus)] =
                find_csr_position(out.j22,
                                  index.vm_block(bus),
                                  index.vm_block(bus));
        }
    }

    return out;
}

}  // namespace exp_20260415::block_ilu
