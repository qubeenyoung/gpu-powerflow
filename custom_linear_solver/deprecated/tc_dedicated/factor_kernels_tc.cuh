#pragma once

// TC-dedicated factor kernels. Phase 2: lower the TC trailing fsz threshold so fronts in 32-48 also
// go through the WMMA path (mid_tc32_b<true> in batched/ uses fsz<=48 -> scalar fallback). Only
// included by src/tc/multifrontal_tc.cu.
//
// Mirrors mf_factor_mid_tc32_b<true> but with a TEMPLATE parameter `MIN_FSZ_FOR_TC` controlling the
// threshold (instead of a hard-coded 48). At instantiation time the compiler dead-codes the unused
// branch. With cap>=16 (nc=16, full WMMA K-tile) and MIN_FSZ_FOR_TC=24, fsz 25-128 fronts all go
// through TC trailing. cap=8 (nc=8, K-padded to 16, ~50% TC efficiency) is also exercised by the
// same code path -- still measurable vs the scalar fallback.

#include <cuda_runtime.h>
#include <mma.h>

#include "batched/lu_device.cuh"
#include "tc/factor_tc.cuh"  // tc_trailing_wmma_f32 (same WMMA helper)

namespace custom_linear_solver::tc {
namespace {

// Same shape as mf_factor_mid_tc32_b<true> but TC threshold is the template arg. The "extend-add"
// at the end is kept in shared (CB never round-trips to global).
template <int MIN_FSZ_FOR_TC>
__global__ void mf_factor_mid_tc_lo_b(int lbegin, int lend,
                                      const int* __restrict__ plcols,
                                      const int* __restrict__ front_off,
                                      const int* __restrict__ front_ptr,
                                      const int* __restrict__ ncols,
                                      const int* __restrict__ panel_parent,
                                      const int* __restrict__ asm_ptr,
                                      const int* __restrict__ asm_local, float* frontB,
                                      long front_total, int* sing, int do_extend, int ucp_max,
                                      int fsz_cap)
{
    using namespace custom_linear_solver::batched;
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    extern __shared__ char smem_mid_lo[];
    __half* Lh = reinterpret_cast<__half*>(smem_mid_lo);
    __half* Uh = Lh + (long)ucp_max * 32;
    float* Csc = reinterpret_cast<float*>(Uh + (long)32 * ucp_max);
    float* Fs = Csc + 4 * 256;
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    // TC path threshold lowered from 48 to MIN_FSZ_FOR_TC. WMMA also requires nc<=32, uc<=256
    // (shared staging cap), else fall back to scalar.
    if (fsz < MIN_FSZ_FOR_TC) {
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        if (nc <= 32 && uc <= 256) {
            tc_trailing_wmma_f32(Fs, fsz, nc, uc, Lh, Uh, Csc, ucp_max, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    }
    __syncthreads();
    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

}  // namespace
}  // namespace custom_linear_solver::tc
