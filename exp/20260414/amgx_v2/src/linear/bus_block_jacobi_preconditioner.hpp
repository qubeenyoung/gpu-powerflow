#pragma once

#include <cstdint>

namespace exp_20260414::amgx_v2 {

struct BusBlockJacobiView {
    int32_t block_rows = 0;
    const int32_t* diagonal_value_base = nullptr;
    const double* values = nullptr;
};

// Device-side 2x2 block Jacobi preconditioner over the bus-local block matrix.
//
// It applies each bus diagonal block independently. This is intentionally small
// and explicit so it can serve as a controlled fallback if AMGX's BLOCK_JACOBI
// smoother rejects fixed 2x2 local matrices.
class BusBlockJacobiPreconditioner {
public:
    void setup(BusBlockJacobiView view);
    void apply(const double* rhs_device, double* x_device, int32_t scalar_dim) const;
    bool ready() const;

private:
    BusBlockJacobiView view_;
};

}  // namespace exp_20260414::amgx_v2
