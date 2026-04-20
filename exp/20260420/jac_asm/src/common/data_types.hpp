#pragma once

#include <cstdint>
#include <vector>

struct YbusGraph {
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    // coo format ybus
    const int32_t* row = nullptr;
    const int32_t* col = nullptr;
    const float* real = nullptr;
    const float* imag = nullptr;

    // csr format ybus
    const int32_t* row_ptr = nullptr;
};

struct BusIndexMap {
    std::vector<int32_t> pvpq;
    std::vector<int32_t> bus_to_pvpq;
    std::vector<int32_t> bus_to_pq;

    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    int32_t dim = 0;
};

struct JacobianPattern {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;

    int32_t dim = 0;
    int32_t nnz = 0;
};

struct JacobianMap {
    std::vector<int32_t> offdiagJ11;
    std::vector<int32_t> offdiagJ12;
    std::vector<int32_t> offdiagJ21;
    std::vector<int32_t> offdiagJ22;

    std::vector<int32_t> diagJ11;
    std::vector<int32_t> diagJ12;
    std::vector<int32_t> diagJ21;
    std::vector<int32_t> diagJ22;
};

struct JacobianBuild {
    BusIndexMap index;
    JacobianPattern pattern;
    JacobianMap map;
};
