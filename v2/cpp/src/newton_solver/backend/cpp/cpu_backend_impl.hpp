#pragma once

#include "newton_solver/backend/cpu_backend.hpp"

#include <Eigen/Sparse>
#include <Eigen/KLUSupport>

#include <complex>
#include <vector>


// JacobianMatrix: Eigen CSC used internally by the CPU backend.
// ColMajor because KLU expects CSC input.
using JacobianMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;


struct CpuNewtonSolverBackend::Impl {
    using cxd  = std::complex<double>;
    using SpCx = Eigen::SparseMatrix<cxd, Eigen::ColMajor, int32_t>;
    using VXcd = Eigen::Matrix<cxd, Eigen::Dynamic, 1>;
    using VXd  = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // --- Ybus (Eigen CSC) built from CSRView in initialize() ---
    SpCx Ybus;

    // --- Ybus in CSR format (from YbusView, stored for CSR-order iteration) ---
    // updateJacobian iterates these to match JacobianMaps which are CSR-indexed.
    std::vector<int32_t> ybus_indptr;
    std::vector<int32_t> ybus_indices;
    std::vector<cxd>     ybus_data;

    // --- Jacobian (Eigen CSC, built from JacobianStructure in analyze) ---
    JacobianMatrix J;

    // --- JacobianMaps with CSC-remapped positions ---
    // Received from JacobianBuilder in CSR positions; remapped to CSC in analyze()
    // so updateJacobian can write directly into J.valuePtr()[pos].
    JacobianMaps maps;

    // --- Per-solve voltage state ---
    VXcd V;
    VXd  Vm;
    VXd  Va;
    VXcd Sbus;

    // --- Cached sparse products shared within one NR iteration ---
    VXcd Ibus;
    bool has_cached_Ibus = false;

    // --- Linear solver ---
    Eigen::KLU<JacobianMatrix> lu;

    int32_t n_bus = 0;
};
