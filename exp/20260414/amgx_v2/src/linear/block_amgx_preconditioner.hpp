#pragma once

#include "linear/amgx_preconditioner.hpp"

#include <cstdint>

namespace exp_20260414::amgx_v2 {

// Thin, named wrapper for fixed-block AMGX preconditioning. Keeping this type
// separate makes scalar AMGX, AMGX block Jacobi, and AMGX block DILU explicit at
// the solver level while sharing the low-level AMGX handle management.
class BlockAmgxPreconditioner {
public:
    void setup(const BlockCsrMatrixView& matrix, AmgxBlockSmoother smoother);
    void apply(const double* rhs_device, double* x_device, int32_t scalar_dim) const;
    bool ready() const;

private:
    AmgxPreconditioner impl_;
};

}  // namespace exp_20260414::amgx_v2
