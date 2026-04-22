#pragma once

#include "data_types.hpp"

#include <cstdint>

namespace exp20260421::vertex_edge::newton_solver {

__global__ void fill_jacobian_edge_batch(YbusGraph ybus,
                                         const double* v_re,
                                         const double* v_im,
                                         int32_t batch_size,
                                         const int32_t* mapJ11,
                                         const int32_t* mapJ21,
                                         const int32_t* mapJ12,
                                         const int32_t* mapJ22,
                                         const int32_t* diagJ11,
                                         const int32_t* diagJ21,
                                         const int32_t* diagJ12,
                                         const int32_t* diagJ22,
                                         int32_t jac_nnz,
                                         float* J_values);

}  // namespace exp20260421::vertex_edge::newton_solver
