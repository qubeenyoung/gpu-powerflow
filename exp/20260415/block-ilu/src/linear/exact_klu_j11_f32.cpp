#include "linear/exact_klu_j11_f32.hpp"

#include <stdexcept>
#include <utility>

namespace exp_20260415::block_ilu {

ExactKluJ11F32::ExactKluJ11F32(std::string name)
    : name_(std::move(name))
{
}

void ExactKluJ11F32::analyze(DeviceCsrMatrixView matrix,
                             const HostCsrPattern& host_pattern)
{
    if (matrix.rows <= 0 || matrix.rows != matrix.cols || matrix.nnz <= 0 ||
        host_pattern.rows != matrix.rows || host_pattern.cols != matrix.cols ||
        host_pattern.nnz() != matrix.nnz) {
        throw std::runtime_error(name_ + "::analyze received invalid inputs");
    }

    matrix_ = matrix;
    host_pattern_ = host_pattern;
    rows_ = matrix.rows;
    nnz_ = matrix.nnz;
    h_values_.resize(static_cast<std::size_t>(nnz_));
    h_rhs_f32_.resize(static_cast<std::size_t>(rows_));
    h_solution_f32_.resize(static_cast<std::size_t>(rows_));
    h_rhs_.resize(rows_);
    h_solution_.resize(rows_);
    build_symbolic_pattern();
    analyzed_ = true;
    factorized_ = false;
}

void ExactKluJ11F32::build_symbolic_pattern()
{
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz_));
    for (int32_t row = 0; row < rows_; ++row) {
        for (int32_t pos = host_pattern_.row_ptr[static_cast<std::size_t>(row)];
             pos < host_pattern_.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            triplets.emplace_back(row,
                                  host_pattern_.col_idx[static_cast<std::size_t>(pos)],
                                  1.0);
        }
    }
    sparse_.resize(rows_, rows_);
    sparse_.setFromTriplets(triplets.begin(), triplets.end());
    sparse_.makeCompressed();
    klu_.analyzePattern(sparse_);
    if (klu_.info() != Eigen::Success) {
        throw std::runtime_error(name_ + " KLU symbolic analysis failed");
    }
    symbolic_analyzed_ = true;
}

void ExactKluJ11F32::factorize()
{
    if (!analyzed_ || !symbolic_analyzed_) {
        throw std::runtime_error(name_ + "::factorize called before analyze");
    }

    CUDA_CHECK(cudaMemcpy(h_values_.data(),
                          matrix_.values,
                          static_cast<std::size_t>(nnz_) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz_));
    for (int32_t row = 0; row < rows_; ++row) {
        for (int32_t pos = host_pattern_.row_ptr[static_cast<std::size_t>(row)];
             pos < host_pattern_.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            triplets.emplace_back(row,
                                  host_pattern_.col_idx[static_cast<std::size_t>(pos)],
                                  h_values_[static_cast<std::size_t>(pos)]);
        }
    }
    sparse_.setZero();
    sparse_.resize(rows_, rows_);
    sparse_.setFromTriplets(triplets.begin(), triplets.end());
    sparse_.makeCompressed();

    klu_.factorize(sparse_);
    if (klu_.info() != Eigen::Success) {
        last_zero_pivot_ = 0;
        factorized_ = false;
        throw std::runtime_error(name_ + " KLU numeric factorization failed");
    }
    last_zero_pivot_ = -1;
    factorized_ = true;
}

void ExactKluJ11F32::solve(const float* rhs_device, float* out_device)
{
    if (!factorized_) {
        throw std::runtime_error(name_ + "::solve called before factorize");
    }
    if (rhs_device == nullptr || out_device == nullptr) {
        throw std::runtime_error(name_ + "::solve received null device pointer");
    }

    CUDA_CHECK(cudaMemcpy(h_rhs_f32_.data(),
                          rhs_device,
                          static_cast<std::size_t>(rows_) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int32_t i = 0; i < rows_; ++i) {
        h_rhs_[i] = static_cast<double>(h_rhs_f32_[static_cast<std::size_t>(i)]);
    }

    h_solution_ = klu_.solve(h_rhs_);
    if (klu_.info() != Eigen::Success) {
        throw std::runtime_error(name_ + " KLU solve failed");
    }

    for (int32_t i = 0; i < rows_; ++i) {
        h_solution_f32_[static_cast<std::size_t>(i)] =
            static_cast<float>(h_solution_[i]);
    }
    CUDA_CHECK(cudaMemcpy(out_device,
                          h_solution_f32_.data(),
                          static_cast<std::size_t>(rows_) * sizeof(float),
                          cudaMemcpyHostToDevice));
}

}  // namespace exp_20260415::block_ilu
