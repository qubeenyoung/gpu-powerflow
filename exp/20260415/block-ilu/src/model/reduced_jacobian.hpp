#pragma once

#include <cstdint>
#include <vector>

namespace exp_20260415::block_ilu {

struct HostCsrPattern {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;

    int32_t nnz() const;
};

struct ReducedJacobianIndex {
    int32_t n_bus = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t n_pvpq = 0;
    int32_t dim = 0;

    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
    std::vector<int32_t> pvpq;

    // Global full-J slots. -1 means the bus is not active in that variable set.
    std::vector<int32_t> theta_slot;
    std::vector<int32_t> vm_slot;

    // Local block slots for J11/J22. J11 uses theta_block_slot; J22 uses
    // vm_block_slot. They are kept separate to make block extraction explicit.
    std::vector<int32_t> theta_block_slot;
    std::vector<int32_t> vm_block_slot;

    bool has_theta(int32_t bus) const;
    bool has_vm(int32_t bus) const;
    int32_t theta(int32_t bus) const;
    int32_t vm(int32_t bus) const;
    int32_t theta_block(int32_t bus) const;
    int32_t vm_block(int32_t bus) const;
};

struct FullJacobianMaps {
    std::vector<int32_t> map11;
    std::vector<int32_t> map12;
    std::vector<int32_t> map21;
    std::vector<int32_t> map22;
    std::vector<int32_t> diag11;
    std::vector<int32_t> diag12;
    std::vector<int32_t> diag21;
    std::vector<int32_t> diag22;
};

struct DiagonalBlockMaps {
    std::vector<int32_t> map;
    std::vector<int32_t> diag;
};

struct ReducedJacobianPatterns {
    ReducedJacobianIndex index;
    HostCsrPattern full;
    HostCsrPattern j11;
    HostCsrPattern j12;
    HostCsrPattern j21;
    HostCsrPattern j22;
    FullJacobianMaps full_maps;
    DiagonalBlockMaps j11_maps;
    DiagonalBlockMaps j12_maps;
    DiagonalBlockMaps j21_maps;
    DiagonalBlockMaps j22_maps;
    std::vector<int32_t> ybus_row;
    std::vector<int32_t> ybus_col;
};

ReducedJacobianIndex build_reduced_jacobian_index(int32_t n_bus,
                                                  const std::vector<int32_t>& pv,
                                                  const std::vector<int32_t>& pq);

ReducedJacobianPatterns build_reduced_jacobian_patterns(
    int32_t n_bus,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq,
    const std::vector<int32_t>& ybus_row_ptr,
    const std::vector<int32_t>& ybus_col_idx);

}  // namespace exp_20260415::block_ilu
