#include "compute_ibus.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <Eigen/Sparse>
#include <complex>
#include <stdexcept>


namespace {
using CpuComplexVectorF64 = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
}


void compute_ibus(CpuFp64Buffers& buf)
{
    if (buf.n_bus <= 0) {
        throw std::runtime_error("compute_ibus: buffers are not prepared");
    }

    Eigen::Map<const CpuComplexVectorF64> V(buf.V.data(), buf.n_bus);
    Eigen::Map<CpuComplexVectorF64>       Ibus(buf.Ibus.data(), buf.n_bus);

    Ibus = buf.Ybus * V;
    buf.has_cached_Ibus = true;
}


void CpuIbusOp::run(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    compute_ibus(buf);
}
