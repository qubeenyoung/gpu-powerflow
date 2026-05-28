#pragma once

#include "newton_solver/core/pipeline.hpp"
#include "newton_solver/core/newton_solver_types.hpp"

#include <cstdint>

void solve_adjoint_pipeline(CpuFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result);

#ifdef CUPF_WITH_CUDA
void solve_adjoint_pipeline(CudaFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result);

void solve_adjoint_pipeline(CudaFp32Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result);

void solve_adjoint_pipeline(CudaMixedPipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result);
#endif
