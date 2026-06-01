// ---------------------------------------------------------------------------
// cuda_voltage_update.cu
//
// CUDA voltage-update stage — applies one Newton step and rebuilds the voltage.
// Two device passes (kernels in voltage_update_kernels.hpp):
//   1. apply_voltage_update : (Va, Vm) -= dx   at the pv/pq slots dx maps to
//   2. reconstruct_voltage  : V = Vm * (cos Va + i sin Va)   (polar -> rect)
// The rectangular V is what the next iteration's Ibus/mismatch stages read.
//
// Precision is parameterized by two scalars:
//   StateScalar - element type of the voltage state (Va/Vm/V_re/V_im)
//   DxScalar    - element type of the Newton step dx coming out of the solve
// The profiles map as:
//   FP64  -> <double, double>   (state and dx both double)
//   FP32  -> <float , float >   (everything float)
//   Mixed -> <double, float >   (double state, float dx; cast on apply)
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_voltage_update.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "voltage_update_kernels.hpp"

#include <stdexcept>


namespace {

// Shared body for all three profiles: validate, apply the step, reconstruct V.
// Storage is one of the CudaBatchedStorage<...> profiles; StateScalar/DxScalar
// pick the kernel instantiation (see file header). Behavior is identical for
// every batch size — the kernels stride over [batch_size * dimF] (the step) and
// [batch_size * n_bus] (the buses).
template <typename StateScalar, typename DxScalar, typename Storage>
void run_voltage_update(Storage& buf)
{
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaVoltageUpdateOp::run: buffers are not prepared");
    }

    // dx is laid out [n_pv angle | n_pq angle | n_pq magnitude] per case, so the
    // angle-equation count n_pv is the total pv+pq rows minus the pq rows.
    const int32_t n_pv = buf.n_pvpq - buf.n_pq;
    const int32_t total_buses = buf.batch_size * buf.n_bus;

    // Pass 1: subtract the Newton step from the polar state (Va at pv/pq buses,
    // Vm at pq buses). One thread per dx entry across the whole batch.
    launch_apply_voltage_update<StateScalar, DxScalar>(
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

    // Pass 2: rebuild the rectangular voltage cache from the updated polar
    // state. One thread per bus across the whole batch.
    launch_reconstruct_voltage<StateScalar>(
        total_buses,
        buf.d_Va.data(),
        buf.d_Vm.data(),
        buf.d_V_re.data(),
        buf.d_V_im.data());

    sync_cuda_for_timing();
}

}  // namespace


// FP64 profile: double state, double dx.
void CudaVoltageUpdateOp<double>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_voltage_update<double, double>(buf);
}


// FP32 profile: float state, float dx.
void CudaVoltageUpdateOp<float>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_voltage_update<float, float>(buf);
}


// Mixed profile: double state updated by a float dx (cast inside the kernel).
void CudaVoltageUpdateOp<float>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_voltage_update<double, float>(buf);
}

#endif  // CUPF_WITH_CUDA
