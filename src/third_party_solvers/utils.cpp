#include "third_party_solvers/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include <mpi.h>

namespace sparse_direct::solver::utils {

void require_square(const matrix::CsrMatrix& csr, std::string_view solver_name)
{
    csr.validate();
    if (csr.rows != csr.cols) {
        throw std::runtime_error(std::string(solver_name) + " requires a square matrix");
    }
}

void require_square(const matrix::CscMatrix& csc, std::string_view solver_name)
{
    csc.validate();
    if (csc.rows != csc.cols) {
        throw std::runtime_error(std::string(solver_name) + " requires a square matrix");
    }
}

void require_rhs_size(int rows, std::size_t rhs_size)
{
    if (rhs_size != static_cast<std::size_t>(rows)) {
        throw std::runtime_error("RHS size does not match matrix rows");
    }
}

std::vector<int> to_int_indices(const std::vector<matrix::Index>& values)
{
    std::vector<int> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        converted[i] = static_cast<int>(values[i]);
    }
    return converted;
}

std::vector<unsigned int> to_unsigned_indices(const std::vector<matrix::Index>& values)
{
    std::vector<unsigned int> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0) {
            throw std::runtime_error("sparse index must be non-negative");
        }
        converted[i] = static_cast<unsigned int>(values[i]);
    }
    return converted;
}

void ensure_mpi_initialized()
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
    }
}

int positive_env_or(std::string_view name, int fallback)
{
    const std::string env_name(name);
    const char* value = std::getenv(env_name.c_str());
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return fallback;
    }

    return static_cast<int>(parsed);
}

double finite_env_or(std::string_view name, double fallback)
{
    const std::string env_name(name);
    const char* value = std::getenv(env_name.c_str());
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) {
        return fallback;
    }

    return parsed;
}

void check_cuda(cudaError_t status, std::string_view action)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + " failed: " + cudaGetErrorString(status));
    }
}

void require_cuda_device(int device_id)
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= device_id) {
        throw std::runtime_error("requested CUDA device is not available");
    }
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void synchronize_cuda(std::string_view action)
{
    check_cuda(cudaDeviceSynchronize(), action);
}

bool contains_nonfinite(const std::vector<double>& values)
{
    return std::any_of(values.begin(), values.end(), [](double value) {
        return !std::isfinite(value);
    });
}

}  // namespace sparse_direct::solver::utils
