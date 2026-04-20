#pragma once

#include "data_types.hpp"

#include <unordered_map>

using CoeffLookup = std::vector<std::unordered_map<int32_t, int32_t>>;

BusIndexMap buildBusIndexMap(const YbusGraph& ybus,
                             const int32_t* pv,
                             int32_t n_pv,
                             const int32_t* pq,
                             int32_t n_pq);

JacobianPattern buildJacobianPattern(const YbusGraph& ybus,
                                     const BusIndexMap& index);

JacobianMap buildJacobianMap(const YbusGraph& ybus,
                             const BusIndexMap& index,
                             const JacobianPattern& pattern);

JacobianBuild buildJacobian(const YbusGraph& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq);

CoeffLookup buildCoeffLookup(const JacobianPattern& pattern);

int32_t coeffIndex(const CoeffLookup& lookup, int32_t row, int32_t col);
