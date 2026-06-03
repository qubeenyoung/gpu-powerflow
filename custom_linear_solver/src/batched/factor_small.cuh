#pragma once

// Internal — included only by batched/multifrontal_batched.cu (single TU). TINY-FRONT factor
// kernel for the batched path.
//
// Why: ncu on the power-grid Jacobians shows batched factor time is dominated by the bottom etree
// levels, which are tens of thousands of TINY fronts (fsz<=~26: 95% of all fronts but ~5% of the
// flops). The general per-front kernel gives each front a 128-thread (4-warp) block, so a 16x16
// front (256 elements, nc<=8) leaves ~3/4 of the threads idle and pays a full BLOCK __syncthreads
// barrier on every one of its nc rank-1 passes -> latency/occupancy bound, not compute bound.
//
// Fix: one WARP per (front, batch), many warps per block. The dense no-pivot LU runs lane-parallel
// with __syncwarp() (a warp barrier, far cheaper than a 4-warp block barrier), and packing W fronts
// per block keeps far more fronts in flight per SM. The front stays in global (it is already L1-
// resident for these sizes; DRAM throughput on these levels is ~12%), so this is purely a barrier /
// occupancy / idle-thread win. Used for levels whose max fsz is small (the dominant bottom levels);
// larger-front levels keep the per-front block kernel (+ tensor-core trailing).

#include <cuda_runtime.h>

namespace custom_linear_solver::batched {
namespace {

// Warp-parallel fused no-pivot LU for a small front F (row-major, ld=fsz), nc pivot columns.
// lane in [0,32). Mirrors lu_small_front but with __syncwarp() and no cross-warp barrier.
template <typename FT>
__device__ __forceinline__ void lu_small_warp(FT* F, int fsz, int nc, int lane, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        FT piv = F[(long)k * fsz + k];
        if (piv == FT(0)) {
            if (lane == 0) *sing = 1;
            piv = FT(1);
        }
        for (int i = k + 1 + lane; i < fsz; i += 32) F[(long)i * fsz + k] /= piv;
        __syncwarp();
        const int m = fsz - k - 1;
        for (int e = lane; e < m * m; e += 32) {
            const int ii = k + 1 + e / m, jj = k + 1 + e % m;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        __syncwarp();
    }
}

// One warp per (front, batch); blockDim = WARPS_PER_BLOCK*32. The level's fronts are
// plcols[lbegin..lend) (level_size of them); work index w in [0, level_size*B) maps to
// front fl = w % level_size, batch bb = w / level_size. FT = front element type (double / float).
//
// The front (a contiguous fsz*fsz block in the arena) is staged into per-warp SHARED memory with a
// COALESCED load, factored there, written back, and its CB extend-added from shared. This kills the
// strided/uncoalesced global access of the column-k divide and the rank-1 update (F[i*fsz+k] across
// lanes i is stride-fsz in global) and keeps the nc repeated passes in low-latency shared. The
// dynamic shared is sized WARPS_PER_BLOCK * fsz2cap elements (fsz2cap = the level's max fsz^2).
template <typename FT>
__global__ void mf_factor_small_warp_b(int lbegin, int level_size, int B, int fsz2cap,
                                       const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols,
                                       const int* __restrict__ panel_parent,
                                       const int* __restrict__ asm_ptr,
                                       const int* __restrict__ asm_local, FT* frontB,
                                       long front_total, int* sing, int do_extend)
{
    extern __shared__ unsigned char smem_sw_raw[];
    FT* smem_sw = reinterpret_cast<FT*>(smem_sw_raw);
    const int warp_in_blk = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_global >= level_size * B) return;
    const int fl = warp_global % level_size;
    const int bb = warp_global / level_size;
    FT* front = frontB + (long)bb * front_total;
    const int p = plcols[lbegin + fl];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    FT* F = front + front_off[p];
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    FT* Fs = smem_sw + (long)warp_in_blk * fsz2cap;

    for (int e = lane; e < fsz2; e += 32) Fs[e] = F[e];  // coalesced stage-in
    __syncwarp();
    lu_small_warp<FT>(Fs, fsz, nc, lane, sing);
    __syncwarp();
    // Write back only the L/U the solve reads; the CB stays in shared for the extend-add below.
    writeback_factored<FT, FT>(F, Fs, fsz, nc, uc, lane, 32);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    FT* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    for (int e = lane; e < uc * uc; e += 32) {
        const int a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * fsz + (nc + b)]);
    }
}

}  // namespace
}  // namespace custom_linear_solver::batched
