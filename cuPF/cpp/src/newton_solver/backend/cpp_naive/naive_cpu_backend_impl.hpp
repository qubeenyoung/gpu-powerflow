#pragma once

#include "newton_solver/backend/naive_cpu_backend.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <complex>
#include <vector>


// Eigen types used throughout the naive backend.
using NaiveJacMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;

struct NaiveCpuNewtonSolverBackend::Impl {
    using cxd  = std::complex<double>;
    using SpCx = Eigen::SparseMatrix<cxd, Eigen::ColMajor, int32_t>;
    using VXcd = Eigen::Matrix<cxd, Eigen::Dynamic, 1>;
    using VXd  = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    // Ybus as Eigen CSC — built once in analyze(), reused across solves.
    SpCx Ybus;

    // Per-solve voltage state (updated in-place each NR iteration).
    VXcd V;
    VXd  Vm;
    VXd  Va;
    VXcd Sbus;

    // Jacobian rebuilt from scratch every updateJacobian() call.
    NaiveJacMat J;

    // pv/pq indices stored by computeMismatch(); used by updateJacobian().
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
    std::vector<int32_t> pvpq;  // concatenation: [pv | pq]

    int32_t n_bus = 0;
};
