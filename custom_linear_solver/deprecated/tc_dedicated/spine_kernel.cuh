#pragma once

// Phase 4 — Spine persistent kernel.
//
// The spine is the contiguous "cnt=1 chain" at the top of the panel etree (computed in
// `analyze_multifrontal`, stored in `plan.h_spine_panels` / `plan.d_spine_panels`). Each
// panel in the chain has exactly one parent (the next spine panel above) and exactly one
// child (the next spine panel below — except for the spine bottom, whose children are the
// K subtree roots).
//
// In the current dispatch each spine level launches its own `mf_factor_*_b` kernel with
// grid=(1, B). That is N_spine kernel launches with B-block grids -- pure dispatch
// overhead since B is small. This kernel fuses all spine panels into a SINGLE launch with
// grid=(1, B), where each block (= one batch) walks the spine in a device-side loop:
//
//   for each spine panel s (bottom -> top):
//       factor F[s] (dense LU, in shared)
//       if not last: extend-add CB into F[parent]
//
// The dependency between spine panels is sequential within a batch; across batches
// independent. This is exactly the persistent-kernel pattern in the Megakernels paper
// applied at small scope (only the spine -- a few panels), which keeps register pressure
// bounded.
//
// FP32 only (TC-dedicated path / batched float-front modes). The spine fronts are usually
// the LARGEST in the tree (root area), so they may exceed the shared-mem cap. This kernel
// runs against the global front directly (no shared staging of the whole front), with
// dynamic shared used only for the panel/U scratch.

#include <cuda_runtime.h>

#include "batched/lu_device.cuh"   // lu_panel_factor, u_panel_solve, trailing_update_scalar, extend_add

namespace custom_linear_solver::tc {
namespace {

using namespace custom_linear_solver::batched;

// One block per batch. Walks `spine_panels[]` in order (bottom -> top) -- caller must
// ensure that array is in factor order, so a panel's child has already been factored and
// extend-added (which is true if all non-spine levels were factored before this kernel
// launches).
__global__ void mf_spine_factor_b(int n_spine, const int* __restrict__ spine_panels,
                                  const int* __restrict__ front_off,
                                  const int* __restrict__ front_ptr,
                                  const int* __restrict__ ncols,
                                  const int* __restrict__ panel_parent,
                                  const int* __restrict__ asm_ptr,
                                  const int* __restrict__ asm_local, float* frontB,
                                  long front_total, int* sing, int do_extend)
{
    if (blockIdx.x != 0) return;  // grid=(1, B); blockIdx.y is the batch
    float* front = frontB + (long)blockIdx.y * front_total;
    const int t = threadIdx.x, nt = blockDim.x;

    for (int idx = 0; idx < n_spine; ++idx) {
        const int p = spine_panels[idx];
        const int fsz = front_ptr[p + 1] - front_ptr[p];
        const int nc = ncols[p];
        const int uc = fsz - nc;
        float* F = front + front_off[p];

        if (fsz <= 48) {
            lu_small_front<float>(F, fsz, nc, t, nt, sing);
        } else {
            lu_panel_factor<float>(F, fsz, nc, t, nt, sing);
            u_panel_solve<float>(F, fsz, nc, uc, t, nt);
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        }

        const int par = panel_parent[p];
        if (par < 0 || !do_extend) continue;
        __syncthreads();
        float* Fp = front + front_off[par];
        const int pfsz = front_ptr[par + 1] - front_ptr[par];
        const int abase = asm_ptr[p];
        extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
        __syncthreads();  // ensure parent's CB is updated before we factor it next iteration
    }
}

}  // namespace
}  // namespace custom_linear_solver::tc
