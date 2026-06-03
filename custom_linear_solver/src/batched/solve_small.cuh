#pragma once

// Internal — included only by batched/multifrontal_batched.cu (single TU). WARP-PACKED forward /
// backward solve kernels for the SMALL bottom etree levels, which dominate the batched solve the
// same way they dominate the factor (the bottom level alone is ~half of the forward solve). The
// general solve kernels launch one 32-thread block per (front,batch); for the tens of thousands of
// tiny leaf fronts that is one warp in its own block = block-launch/scheduling overhead and poor
// SM packing. These pack WARPS_PER_BLOCK warps per block, one (front,batch) per warp, mapping work
// index w in [0, level_size*B) to front fl = w % level_size, batch bb = w / level_size.
//
// Per-warp shared: the forward kernel needs sh_piv[nc] (nc <= MF_MAX_NC); the backward kernel needs
// rhs[nc] + the x-cache xsh[cb]. Laid out as a per-warp slab of `slab` FT elements (slab = 64 + cb
// cap, sized at launch). Behavior matches mf_fwd_level_b / mf_bwd_level_b (incl. the selinv GEMV
// path) but with __syncwarp() and lane-local strides.

#include <cuda_runtime.h>

namespace custom_linear_solver::batched {
namespace {

// Forward solve (L y = b) for a small level, one warp per (front,batch).
template <typename FT>
__global__ void mf_fwd_small_warp_b(int lbegin, int level_size, int B, int slab,
                                    const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ front_rows, const FT* frontB, FT* yB,
                                    long front_total, int n, int selinv)
{
    extern __shared__ unsigned char fsm_raw[];
    FT* slabs = reinterpret_cast<FT*>(fsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= level_size * B) return;
    const int fl = wg % level_size, bb = wg / level_size;
    const FT* front = frontB + (long)bb * front_total;
    FT* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    FT* sh_piv = slabs + (long)warp_in_blk * slab;

    if (selinv) {
        for (int k = lane; k < nc; k += 32) {
            FT v = y[fr[k]];
            for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];
            sh_piv[k] = v;
        }
        __syncwarp();
        for (int k = lane; k < nc; k += 32) y[fr[k]] = sh_piv[k];
    } else {
        FT part = FT(0), sk = FT(0);
        for (int k = 0; k < nc; ++k) {
            if (lane == k) { sk = y[fr[k]] + part; sh_piv[k] = sk; y[fr[k]] = sk; }
            sk = __shfl_sync(0xffffffffu, sk, k);
            if (lane > k && lane < nc) part -= F[(long)lane * fsz + k] * sk;
        }
    }
    __syncwarp();
    for (int i = nc + lane; i < fsz; i += 32) {
        FT upd = FT(0);
        for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
        atomicAdd(&y[fr[i]], -upd);
    }
}

// Backward solve (U x = y) for a small level, one warp per (front,batch). slab >= 64 + cb.
template <typename FT>
__global__ void mf_bwd_small_warp_b(int lbegin, int level_size, int B, int slab,
                                    const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ front_rows, const FT* frontB, FT* yB,
                                    long front_total, int n, int selinv)
{
    extern __shared__ unsigned char bsm_raw[];
    FT* slabs = reinterpret_cast<FT*>(bsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= level_size * B) return;
    const int fl = wg % level_size, bb = wg / level_size;
    const FT* front = frontB + (long)bb * front_total;
    FT* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const FT* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int cb = fsz - nc;
    FT* rhs = slabs + (long)warp_in_blk * slab;  // [0,nc)
    FT* xsh = rhs + 64;                           // [0,cb)

    for (int k = lane; k < nc; k += 32) rhs[k] = y[fr[k]];
    for (int j = lane; j < cb; j += 32) xsh[j] = y[fr[nc + j]];
    __syncwarp();
    if (lane < nc) {
        FT pk = FT(0);
        for (int j = 0; j < cb; ++j) pk += F[(long)lane * fsz + (nc + j)] * xsh[j];
        rhs[lane] -= pk;
    }
    __syncwarp();
    if (selinv) {
        for (int k = lane; k < nc; k += 32) {
            FT v = FT(0);
            for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
            y[fr[k]] = v;
        }
    } else {
        FT part = FT(0), xk = FT(0);
        for (int k = nc - 1; k >= 0; --k) {
            if (lane == k) { xk = (rhs[k] + part) / F[(long)k * fsz + k]; y[fr[k]] = xk; }
            xk = __shfl_sync(0xffffffffu, xk, k);
            if (lane < k) part -= F[(long)lane * fsz + k] * xk;
        }
    }
}

}  // namespace
}  // namespace custom_linear_solver::batched
