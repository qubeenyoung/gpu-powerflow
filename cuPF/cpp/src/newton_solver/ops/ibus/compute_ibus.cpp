#include "compute_ibus.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>


void compute_ibus(CpuFp64Buffers& buf)
{
    if (buf.n_bus <= 0) {
        throw std::runtime_error("compute_ibus: buffers are not prepared");
    }

    // Ibus = Ybus * V, with Ybus stored CSC (column-major):
    //   for each column c with V[c] != 0,
    //     for each k in [indptr[c], indptr[c+1]):
    //       Ibus[indices[k]] += values[k] * V[c]
    std::fill(buf.Ibus.begin(), buf.Ibus.end(), std::complex<double>(0.0, 0.0));

    const int32_t* col_ptr = buf.Ybus.outerIndexPtr();
    const int32_t* row_idx = buf.Ybus.innerIndexPtr();
    const std::complex<double>* vals = buf.Ybus.valuePtr();

    for (int32_t col = 0; col < buf.n_bus; ++col) {
        const std::complex<double> vc = buf.V[static_cast<std::size_t>(col)];
        const int32_t k_end = col_ptr[col + 1];
        for (int32_t k = col_ptr[col]; k < k_end; ++k) {
            buf.Ibus[static_cast<std::size_t>(row_idx[k])] += vals[k] * vc;
        }
    }
    buf.has_cached_Ibus = true;
}


void CpuIbusOp::run(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    compute_ibus(buf);
}
