#pragma once

#include <cstdint>

namespace exp_20260414::amgx_v2 {

struct CsrMatrixView {
    int32_t rows = 0;
    int32_t nnz = 0;
    const int32_t* row_ptr = nullptr;
    const int32_t* col_idx = nullptr;
    const double* values = nullptr;
};

struct BlockCsrMatrixView {
    int32_t rows = 0;
    int32_t nnz = 0;
    int32_t block_dim = 0;
    const int32_t* row_ptr = nullptr;
    const int32_t* col_idx = nullptr;
    const double* values = nullptr;
};

enum class AmgxBlockSmoother {
    MulticolorDilu,
    BlockJacobi,
};

class AmgxPreconditioner {
public:
    AmgxPreconditioner();
    ~AmgxPreconditioner();

    AmgxPreconditioner(const AmgxPreconditioner&) = delete;
    AmgxPreconditioner& operator=(const AmgxPreconditioner&) = delete;

    void setup(const CsrMatrixView& matrix);
    void setup(const BlockCsrMatrixView& matrix);
    void setup(const BlockCsrMatrixView& matrix, AmgxBlockSmoother smoother);
    void apply(const double* rhs_device, double* x_device, int32_t n) const;
    bool ready() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    void setup_impl(int32_t rows,
                    int32_t nnz,
                    int32_t block_dim,
                    AmgxBlockSmoother smoother,
                    const int32_t* row_ptr,
                    const int32_t* col_idx,
                    const double* values);
};

}  // namespace exp_20260414::amgx_v2
