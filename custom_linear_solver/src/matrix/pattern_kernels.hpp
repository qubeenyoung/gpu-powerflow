#pragma once

#include <cstddef>
#include <vector>

#include "solver.hpp"

namespace custom_linear_solver::matrix {

struct IntDeviceBuffer {
    int* ptr = nullptr;
    std::size_t count = 0;

    IntDeviceBuffer() = default;
    ~IntDeviceBuffer();
    IntDeviceBuffer(const IntDeviceBuffer&) = delete;
    IntDeviceBuffer& operator=(const IntDeviceBuffer&) = delete;
    IntDeviceBuffer(IntDeviceBuffer&& other) noexcept;
    IntDeviceBuffer& operator=(IntDeviceBuffer&& other) noexcept;

    void reset();
    Status allocate(std::size_t values);
    Status upload(const std::vector<int>& values);
};

struct DeviceCscPattern {
    int n = 0;
    int nnz = 0;
    IntDeviceBuffer col_ptr;
    IntDeviceBuffer row_idx;
    IntDeviceBuffer source_pos;
};

Status build_csc_from_csr_device(int n, int nnz, const int* d_csr_row_ptr,
                                 const int* d_csr_col_idx, DeviceCscPattern& csc);

Status permute_csc_device(const DeviceCscPattern& csc, const int* d_iperm,
                          DeviceCscPattern& ordered);

Status download_csc_structure(const DeviceCscPattern& csc, std::vector<int>& col_ptr,
                              std::vector<int>& row_idx);

}  // namespace custom_linear_solver::matrix
