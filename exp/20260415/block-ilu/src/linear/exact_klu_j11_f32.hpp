#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <Eigen/KLUSupport>
#include <Eigen/Sparse>

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

class ExactKluJ11F32 {
public:
    explicit ExactKluJ11F32(std::string name);

    ExactKluJ11F32(const ExactKluJ11F32&) = delete;
    ExactKluJ11F32& operator=(const ExactKluJ11F32&) = delete;

    void analyze(DeviceCsrMatrixView matrix, const HostCsrPattern& host_pattern);
    void factorize();
    void solve(const float* rhs_device, float* out_device);

    int32_t last_zero_pivot() const { return last_zero_pivot_; }

private:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Triplet = Eigen::Triplet<double, int32_t>;

    void build_symbolic_pattern();

    std::string name_;
    DeviceCsrMatrixView matrix_;
    HostCsrPattern host_pattern_;
    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    int32_t last_zero_pivot_ = -1;
    bool analyzed_ = false;
    bool symbolic_analyzed_ = false;
    bool factorized_ = false;

    std::vector<double> h_values_;
    std::vector<float> h_rhs_f32_;
    std::vector<float> h_solution_f32_;
    Eigen::VectorXd h_rhs_;
    Eigen::VectorXd h_solution_;
    SparseMatrix sparse_;
    Eigen::KLU<SparseMatrix> klu_;
};

}  // namespace exp_20260415::block_ilu
