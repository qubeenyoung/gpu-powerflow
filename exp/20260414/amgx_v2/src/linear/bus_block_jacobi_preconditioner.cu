#include "bus_block_jacobi_preconditioner.hpp"

#include "utils/cuda_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace exp_20260414::amgx_v2 {
namespace {

constexpr int32_t kBlockDim = 2;
constexpr int32_t kBlockSize = 256;
constexpr double kPivotFloor = 1e-18;

__global__ void apply_bus_block_jacobi_kernel(int32_t block_rows,
                                              const int32_t* __restrict__ diagonal_value_base,
                                              const double* __restrict__ values,
                                              const double* __restrict__ rhs,
                                              double* __restrict__ x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= block_rows) {
        return;
    }

    const int32_t base = diagonal_value_base[row];
    const double a00 = values[base + 0];
    const double a01 = values[base + 1];
    const double a10 = values[base + 2];
    const double a11 = values[base + 3];
    const double b0 = rhs[kBlockDim * row + 0];
    const double b1 = rhs[kBlockDim * row + 1];

    const double det = a00 * a11 - a01 * a10;
    if (isfinite(det) && fabs(det) > kPivotFloor) {
        const double inv_det = 1.0 / det;
        x[kBlockDim * row + 0] = (a11 * b0 - a01 * b1) * inv_det;
        x[kBlockDim * row + 1] = (-a10 * b0 + a00 * b1) * inv_det;
        return;
    }

    // Keep singular or nearly singular local blocks from poisoning FGMRES. This
    // fallback is deliberately conservative and only uses reliable diagonal
    // entries from the same block.
    x[kBlockDim * row + 0] = fabs(a00) > kPivotFloor ? b0 / a00 : b0;
    x[kBlockDim * row + 1] = fabs(a11) > kPivotFloor ? b1 / a11 : b1;
}

}  // namespace

void BusBlockJacobiPreconditioner::setup(BusBlockJacobiView view)
{
    if (view.block_rows <= 0 || view.diagonal_value_base == nullptr || view.values == nullptr) {
        throw std::runtime_error("BusBlockJacobiPreconditioner::setup received invalid input");
    }
    view_ = view;
}

void BusBlockJacobiPreconditioner::apply(const double* rhs_device,
                                         double* x_device,
                                         int32_t scalar_dim) const
{
    if (!ready() || rhs_device == nullptr || x_device == nullptr ||
        scalar_dim != kBlockDim * view_.block_rows) {
        throw std::runtime_error("BusBlockJacobiPreconditioner::apply received invalid input");
    }

    const int32_t grid = (view_.block_rows + kBlockSize - 1) / kBlockSize;
    apply_bus_block_jacobi_kernel<<<grid, kBlockSize>>>(
        view_.block_rows,
        view_.diagonal_value_base,
        view_.values,
        rhs_device,
        x_device);
    CUDA_CHECK(cudaGetLastError());
}

bool BusBlockJacobiPreconditioner::ready() const
{
    return view_.block_rows > 0 && view_.diagonal_value_base != nullptr && view_.values != nullptr;
}

}  // namespace exp_20260414::amgx_v2
