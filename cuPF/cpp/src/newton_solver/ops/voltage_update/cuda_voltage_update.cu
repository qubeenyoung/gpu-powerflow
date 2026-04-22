// ---------------------------------------------------------------------------
// cuda_voltage_update.cu
//
// CUDA voltage update. T selects the linear-solve correction dtype while the
// voltage state/cache remains double.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_voltage_update.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "voltage_update_kernels.hpp"

#include <stdexcept>


void CudaVoltageUpdateOp<double>::run(CudaFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0) {
        throw std::runtime_error("CudaVoltageUpdateOp<double>::run: buffers are not prepared");
    }

    const int32_t n_pv = buf.n_pvpq - buf.n_pq;
    launch_voltage_update_state<double>(
        1,
        buf.n_bus,
        buf.dimF,
        n_pv,
        buf.n_pq,
        buf.d_Va.data(),
        buf.d_Vm.data(),
        buf.d_dx.data(),
        buf.d_pv.data(),
        buf.d_pq.data());

    launch_reconstruct_voltage(
        buf.n_bus,
        buf.d_Va.data(),
        buf.d_Vm.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data());

    sync_cuda_for_timing();
}


void CudaVoltageUpdateOp<float>::run(CudaMixedBuffers& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaVoltageUpdateOp<float>::run: buffers are not prepared");
    }

    const int32_t n_pv = buf.n_pvpq - buf.n_pq;
    const int32_t total_buses = buf.batch_size * buf.n_bus;
    launch_voltage_update_state<float>(
        buf.batch_size,
        buf.n_bus,
        buf.dimF,
        n_pv,
        buf.n_pq,
        buf.d_Va.data(),
        buf.d_Vm.data(),
        buf.d_dx.data(),
        buf.d_pv.data(),
        buf.d_pq.data());

    launch_reconstruct_voltage(
        total_buses,
        buf.d_Va.data(),
        buf.d_Vm.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data());

    sync_cuda_for_timing();
}

#endif  // CUPF_WITH_CUDA
