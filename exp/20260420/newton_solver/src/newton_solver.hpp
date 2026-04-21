#pragma once

#include "cudss_lu.hpp"
#include "data_types.hpp"
#include "mismatch.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace exp20260420::newton_solver {

struct NewtonOptions {
    int32_t max_iter = 20;
    double tolerance = 1e-6;
    int32_t batch_size = 1;
};

struct NewtonResult {
    int32_t iterations = 0;
    double final_mismatch = 0.0;
    bool converged = false;
};

struct NewtonWorkspace {
    int32_t n_bus = 0;
    int32_t n_edges = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t n_pvpq = 0;
    int32_t batch_size = 1;
    int32_t dim = 0;
    int32_t jac_nnz = 0;

    YbusGraph ybus{};

    int32_t* pv = nullptr;
    int32_t* pq = nullptr;
    int32_t* pvpq = nullptr;

    int32_t* J_row_ptr = nullptr;
    int32_t* J_col_idx = nullptr;
    float* J_values = nullptr;

    int32_t* offdiagJ11 = nullptr;
    int32_t* offdiagJ12 = nullptr;
    int32_t* offdiagJ21 = nullptr;
    int32_t* offdiagJ22 = nullptr;
    int32_t* diagJ11 = nullptr;
    int32_t* diagJ12 = nullptr;
    int32_t* diagJ21 = nullptr;
    int32_t* diagJ22 = nullptr;

    float* sbus_re = nullptr;
    float* sbus_im = nullptr;
    double* v_re = nullptr;
    double* v_im = nullptr;
    double* F = nullptr;
    float* dx = nullptr;

    exp20260420::mismatch::MismatchWorkspace mismatch;
    exp20260420::linear_solver::CudssLuWorkspace lu;
};

void newtonAnalyze(NewtonWorkspace& ws,
                   const YbusGraph& host_ybus,
                   const YbusGraph& device_ybus,
                   const int32_t* pv,
                   int32_t n_pv,
                   const int32_t* pq,
                   int32_t n_pq,
                   int32_t batch_size = 1,
                   cudaStream_t stream = nullptr);

NewtonResult newtonSolve(NewtonWorkspace& ws,
                         const float* sbus_re,
                         const float* sbus_im,
                         const double* v0_re,
                         const double* v0_im,
                         const NewtonOptions& options,
                         double* out_v_re = nullptr,
                         double* out_v_im = nullptr,
                         cudaStream_t stream = nullptr);

void newtonDestroy(NewtonWorkspace& ws);

}  // namespace exp20260420::newton_solver
