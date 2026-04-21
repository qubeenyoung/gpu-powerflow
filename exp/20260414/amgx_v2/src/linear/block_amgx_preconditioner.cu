#include "block_amgx_preconditioner.hpp"

#include <stdexcept>

namespace exp_20260414::amgx_v2 {

void BlockAmgxPreconditioner::setup(const BlockCsrMatrixView& matrix,
                                    AmgxBlockSmoother smoother)
{
    if (matrix.block_dim != 2) {
        throw std::runtime_error("BlockAmgxPreconditioner requires fixed 2x2 blocks");
    }
    impl_.setup(matrix, smoother);
}

void BlockAmgxPreconditioner::apply(const double* rhs_device,
                                    double* x_device,
                                    int32_t scalar_dim) const
{
    impl_.apply(rhs_device, x_device, scalar_dim);
}

bool BlockAmgxPreconditioner::ready() const
{
    return impl_.ready();
}

}  // namespace exp_20260414::amgx_v2
