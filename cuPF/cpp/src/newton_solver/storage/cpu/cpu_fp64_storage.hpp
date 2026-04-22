#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"

#include <Eigen/Sparse>

#include <vector>
#include <complex>

using CpuYbusMatrixF64     = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor, int32_t>;
using CpuJacobianMatrixF64 = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;


// ---------------------------------------------------------------------------
// CpuFp64Buffers: CPU FP64 경로의 host-side 버퍼.
//
// 메모리와 레이아웃만 소유한다. KLU solver 상태는 CpuLinearSolveKLU가 갖는다.
// ---------------------------------------------------------------------------
struct CpuFp64Buffers {
    void prepare(const InitializeContext& ctx);
    void upload(const SolveContext& ctx);
    void download(NRResult& result) const;

    CpuYbusMatrixF64     Ybus;
    CpuJacobianMatrixF64 J;

    JacobianScatterMap   maps;
    JacobianPattern      J_pattern;

    std::vector<double>               F;
    std::vector<double>               dx;

    std::vector<double>               Va;
    std::vector<double>               Vm;
    std::vector<std::complex<double>> V;

    std::vector<int32_t>              Ybus_indptr;
    std::vector<int32_t>              Ybus_indices;
    std::vector<std::complex<double>> Ybus_data;
    std::vector<std::complex<double>> Ibus;
    bool                              has_cached_Ibus = false;

    std::vector<std::complex<double>> Sbus;

    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;
};
