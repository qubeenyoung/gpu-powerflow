#pragma once

// experimental minimal cuPF NR port

#include "cupf_minimal/case_data.hpp"

#include <cstdint>
#include <vector>

namespace cupf_minimal {

struct JacobianIndexing {
    int32_t n_bus = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    std::vector<int32_t> pvpq;
    std::vector<int32_t> row_pvpq;
    std::vector<int32_t> row_pq;
    std::vector<int32_t> col_pvpq;
    std::vector<int32_t> col_pq;
};

struct JacobianPattern {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    int32_t dim = 0;
    int32_t nnz = 0;
};

struct JacobianScatterMap {
    std::vector<int32_t> mapJ11;
    std::vector<int32_t> mapJ12;
    std::vector<int32_t> mapJ21;
    std::vector<int32_t> mapJ22;
    std::vector<int32_t> diagJ11;
    std::vector<int32_t> diagJ12;
    std::vector<int32_t> diagJ21;
    std::vector<int32_t> diagJ22;
    std::vector<int32_t> pvpq;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
};

JacobianIndexing make_jacobian_indexing(int32_t n_bus,
                                         const int32_t* pv,
                                         int32_t n_pv,
                                         const int32_t* pq,
                                         int32_t n_pq);

class JacobianPatternGenerator {
public:
    JacobianPattern generate(const YbusView& ybus, const JacobianIndexing& indexing) const;
};

class JacobianMapBuilder {
public:
    JacobianScatterMap build(const YbusView& ybus,
                             const JacobianIndexing& indexing,
                             const JacobianPattern& pattern) const;
};

}  // namespace cupf_minimal

