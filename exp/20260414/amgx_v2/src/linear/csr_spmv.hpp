#pragma once

#include "linear/amgx_preconditioner.hpp"

namespace exp_20260414::amgx_v2 {

// Minimal device CSR SpMV for the assembled bus-local Jacobian.
//
// This deliberately stays small instead of pulling solver logic into the
// matrix wrapper. FGMRES can use this as its operator callback while AMGX owns
// the preconditioner application.
class CsrSpmv {
public:
    void bind(CsrMatrixView matrix);
    void apply(const double* x_device, double* y_device) const;

    int32_t rows() const { return matrix_.rows; }

private:
    CsrMatrixView matrix_;
};

}  // namespace exp_20260414::amgx_v2
