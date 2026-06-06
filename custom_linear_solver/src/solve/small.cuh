#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// Warp-packed forward / backward solve for the SMALL bottom etree levels. These dominate the
// batched solve the same way they dominate the factor: the bottom level alone is ~half of the
// forward solve. Block-per-front would launch one 32-thread block per (front, batch); for tens
// of thousands of tiny leaf fronts that means one warp in its own block — block-launch /
// scheduling overhead plus poor SM packing.
//
// Layout: WARPS_PER_BLOCK warps per block, one (front, batch) per warp. Work index w in
// [0, level_size * B) maps to (front, batch) = (w % level_size, w / level_size). Same body as
// the block kernels but with __syncwarp() in place of __syncthreads(), and per-warp shared
// slabs.

#include <cuda_runtime.h>

#include "solve/primitives.cuh"

namespace custom_linear_solver {
namespace {

// Forward solve (L y = b) for a small-front level. slab elements per warp in dynamic shared.
template <typename T>
__global__ void solve_fwd_small(int lbegin, int level_size, int B, int slab,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    extern __shared__ unsigned char fsm_raw[];
    T* slabs = reinterpret_cast<T*>(fsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= level_size * B) return;
    const int fl = wg % level_size, bb = wg / level_size;
    const T* front = frontB + (long)bb * front_total;
    T* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    T* sh_piv = slabs + (long)warp_in_blk * slab;

    fwd_substitute<T>(F, fsz, nc, fr, y, sh_piv, lane);
    __syncwarp();
    fwd_cb_update<T>(F, fsz, nc, /*cb=*/fsz - nc, fr, y, sh_piv, lane, /*nt=*/32);
}

// Backward solve (U x = y) for a small-front level. slab >= MF_MAX_NC + cb.
template <typename T>
__global__ void solve_bwd_small(int lbegin, int level_size, int B, int slab,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ front_rows,
                                const T* frontB, T* yB, long front_total, int n)
{
    extern __shared__ unsigned char bsm_raw[];
    T* slabs = reinterpret_cast<T*>(bsm_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wg >= level_size * B) return;
    const int fl = wg % level_size, bb = wg / level_size;
    const T* front = frontB + (long)bb * front_total;
    T* y = yB + (long)bb * n;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    const T* F = front + front_off[p];
    const int* fr = front_rows + s;
    const int cb = fsz - nc;
    T* rhs = slabs + (long)warp_in_blk * slab;   // [0, nc)
    T* xsh = rhs + MF_MAX_NC;                    // [0, cb)

    bwd_load_rhs_and_x<T>(y, fr, nc, cb, rhs, xsh, lane, /*nt=*/32);
    __syncwarp();
    bwd_cb_subtract<T>(F, fsz, nc, cb, xsh, rhs, lane, /*width=*/32);
    __syncwarp();
    bwd_substitute<T>(F, fsz, nc, fr, y, rhs, lane);
}

}  // namespace
}  // namespace custom_linear_solver
