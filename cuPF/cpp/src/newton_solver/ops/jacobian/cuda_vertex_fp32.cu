// ---------------------------------------------------------------------------
// cuda_vertex_fp32.cu — FP32 vertex-based Jacobian orchestrator
//
// The main kernels live in:
//   fill_jacobian_vertex_offdiag_fp32.cu
//   fill_jacobian_diag_from_ibus_fp32.cu
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_vertex_fp32.hpp"

#include "cuda_jacobian_fp32_kernels.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"

#include <stdexcept>


CudaJacobianOpVertexFp32::CudaJacobianOpVertexFp32(IStorage& storage)
    : storage_(storage) {}


void CudaJacobianOpVertexFp32::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.d_pvpq.empty() || storage.d_J_values.empty() || storage.d_Ibus_re.empty()) {
        throw std::runtime_error("CudaJacobianOpVertexFp32::run: storage is not prepared");
    }

    // Kept for now as planned: full removal needs a poison/coverage check that
    // every Jacobian slot is written exactly once before diagonal accumulation.
    storage.d_J_values.memsetZero();

    launch_fill_jacobian_vertex_offdiag_fp32(storage);
    launch_fill_jacobian_diag_from_ibus_fp32(storage);
}

#endif  // CUPF_WITH_CUDA
