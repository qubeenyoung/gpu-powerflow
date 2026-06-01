// ---------------------------------------------------------------------------
// compute_mismatch_from_ibus.cu
//
// Newton residual stage:  F = (power computed from V, Ibus) - Sbus.
//
// The bus current Ibus = Ybus * V is produced by ops/ibus beforehand. The
// complex injected power at a bus is S(V) = V * conj(Ibus), so the residual is
//   F_complex = V * conj(Ibus) - Sbus
// and the real Newton residual vector keeps the active-power part (Re) at the
// PV/PQ angle rows and the reactive-power part (Im) at the PQ magnitude rows:
//   F = [ dP@pv | dP@pq | dQ@pq ]   (length dimF = n_pv + 2*n_pq)
// matching the CPU mismatch ordering and the Jacobian row layout.
//
// One templated kernel serves all three precision profiles; the three public
// launchers are thin wrappers that pick the scalar (FP32 -> float, FP64/Mixed
// -> double state) and forward the batched buffers.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <stdexcept>


namespace {

template <typename Scalar>
__global__ void compute_mismatch_from_ibus_kernel(
    int32_t total_entries,    // batch_size * dimF (one thread per residual entry)
    int32_t dimF,
    int32_t n_bus,
    int32_t n_pv,
    int32_t n_pq,
    const Scalar* __restrict__ v_re,
    const Scalar* __restrict__ v_im,
    const Scalar* __restrict__ ibus_re,
    const Scalar* __restrict__ ibus_im,
    const Scalar* __restrict__ sbus_re,
    const Scalar* __restrict__ sbus_im,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    Scalar* __restrict__ F)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_entries) {
        return;
    }

    // Map the flat thread id to (batch case, position within that case's F).
    const int32_t batch = tid / dimF;
    const int32_t local = tid - batch * dimF;

    // The residual layout [dP@pv | dP@pq | dQ@pq] tells us which bus this entry
    // belongs to and whether it is an active-power (Re) or reactive-power (Im)
    // equation. PV buses have only the dP (angle) row; PQ buses have both.
    int32_t bus = 0;
    bool take_imag = false;
    if (local < n_pv) {
        bus = pv[local];                     // dP at a pv bus
    } else if (local < n_pv + n_pq) {
        bus = pq[local - n_pv];              // dP at a pq bus
    } else {
        bus = pq[local - n_pv - n_pq];       // dQ at a pq bus
        take_imag = true;
    }

    // Injected power S(V) = V * conj(Ibus). With V = vr + i*vi, Ibus = ir + i*ii:
    //   Re(S) = vr*ir + vi*ii   (active power P)
    //   Im(S) = vi*ir - vr*ii   (reactive power Q)
    // The residual subtracts the specified injection Sbus.
    const int32_t bus_idx = batch * n_bus + bus;   // batch-major per-bus index
    const Scalar vr = v_re[bus_idx];
    const Scalar vi = v_im[bus_idx];
    const Scalar ir = ibus_re[bus_idx];
    const Scalar ii = ibus_im[bus_idx];

    const Scalar mis_re = vr * ir + vi * ii - sbus_re[bus_idx];   // P mismatch
    const Scalar mis_im = vi * ir - vr * ii - sbus_im[bus_idx];   // Q mismatch
    F[tid] = take_imag ? mis_im : mis_re;
}

// Shared launch: one thread per residual entry across the whole batch.
template <typename Scalar, typename Storage>
void launch_mismatch_impl(Storage& storage)
{
    if (storage.n_bus <= 0 || storage.dimF <= 0 || storage.batch_size <= 0) {
        throw std::runtime_error("launch_compute_mismatch_from_ibus: storage is not prepared");
    }

    constexpr int32_t block = 256;
    const int32_t total_entries = storage.batch_size * storage.dimF;
    const int32_t grid = (total_entries + block - 1) / block;

    compute_mismatch_from_ibus_kernel<Scalar><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        total_entries,
        storage.dimF,
        storage.n_bus,
        storage.n_pvpq - storage.n_pq,   // n_pv = (pv+pq) angle rows minus pq rows
        storage.n_pq,
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_Ibus_re.data(),
        storage.d_Ibus_im.data(),
        storage.d_Sbus_re.data(),
        storage.d_Sbus_im.data(),
        storage.d_pv.data(),
        storage.d_pq.data(),
        storage.d_F.data());
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

}  // namespace


// FP64 / FP32 / Mixed wrappers — pick the state scalar (Mixed keeps FP64 state).
void launch_compute_mismatch_from_ibus(CudaFp64Storage& storage)
{
    launch_mismatch_impl<double>(storage);
}

void launch_compute_mismatch_from_ibus(CudaFp32Storage& storage)
{
    launch_mismatch_impl<float>(storage);
}

void launch_compute_mismatch_from_ibus(CudaMixedStorage& storage)
{
    launch_mismatch_impl<double>(storage);
}

#endif  // CUPF_WITH_CUDA
