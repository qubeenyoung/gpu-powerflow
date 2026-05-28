#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "matrix/sparse_matrix.hpp"

namespace sparse_direct::solver::utils {

void require_square(const matrix::CsrMatrix& csr, std::string_view solver_name);
void require_square(const matrix::CscMatrix& csc, std::string_view solver_name);
void require_rhs_size(int rows, std::size_t rhs_size);

std::vector<int> to_int_indices(const std::vector<matrix::Index>& values);
std::vector<unsigned int> to_unsigned_indices(const std::vector<matrix::Index>& values);

void ensure_mpi_initialized();

int positive_env_or(std::string_view name, int fallback);
double finite_env_or(std::string_view name, double fallback);

void check_cuda(cudaError_t status, std::string_view action);
void require_cuda_device(int device_id = 0);
void synchronize_cuda(std::string_view action);

bool contains_nonfinite(const std::vector<double>& values);

}  // namespace sparse_direct::solver::utils
