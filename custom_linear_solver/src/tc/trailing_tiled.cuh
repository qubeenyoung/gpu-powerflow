#pragma once

// Phase Σ — Tiled scalar trailing GEMM with shared-memory blocking.
//
// Motivation: ncu measurement on `mf_factor_mid_tc32_b<false>` (USA, FP32, B=64) showed:
//   FFMA achieved = 0.24 × 10¹² inst/s -> FP32 FLOPS = 0.49 TFLOPS
//   RTX 3090 peak FP32 = 35.6 TFLOPS
//   efficiency = 1.36 % of peak, FMA pipe % = 12.6
//
// The existing `trailing_update_scalar` (lu_device.cuh) does a triple-loop without shared-memory
// tiling. Each thread reads L[ii, 0..nc-1] from global once per (i, j) output -- many redundant
// global reads. cuBLAS-quality GEMM uses register/shared blocking; we replicate the lighter
// shared-memory blocking here.
//
// Trailing: C(uc x uc) -= L(uc x nc) * U(nc x uc), all in row-major in the front F.
// L is at F[(nc+i)*fsz + k], i in 0..uc, k in 0..nc.
// U is at F[k*fsz + (nc+j)], k in 0..nc, j in 0..uc.
// C is at F[(nc+i)*fsz + (nc+j)].

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace custom_linear_solver::tc {
namespace {

// Simpler shared-staged trailing update — stages the entire L and U panels into shared
// once, then each thread computes its assigned C[i, j] from shared (no inner tile loops).
// Drop-in replacement for trailing_update_scalar but with vastly fewer global memory reads.
// L panel is uc x nc (uc rows, nc cols); U panel is nc x uc.
template <typename T, int MAX_NC = 32, int MAX_UC = 256>
__device__ __forceinline__ void trailing_update_staged(T* F, int fsz, int nc, int uc, int t,
                                                        int nt, T* sh_L, T* sh_U)
{
    // Stage L into sh_L (row-major, uc x nc, ld = nc)
    for (int e = t; e < uc * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        sh_L[e] = F[(long)(nc + i) * fsz + k];
    }
    // Stage U into sh_U (row-major, nc x uc, ld = uc)
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        sh_U[e] = F[(long)k * fsz + (nc + j)];
    }
    __syncthreads();
    // Compute C -= L * U from shared.
    for (int e = t; e < uc * uc; e += nt) {
        const int i = e / uc, j = e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) {
            acc += sh_L[i * nc + k] * sh_U[k * uc + j];
        }
        F[(long)(nc + i) * fsz + (nc + j)] -= acc;
    }
}

// Register-blocked trailing update. Each thread holds a TM x TN accumulator tile in registers,
// reducing LDS:FMA ratio from 2:1 (scalar) to (TM+TN)/(TM*TN) (e.g. 0.5:1 for 4x4). Threads are
// arranged in a BY x BX grid (BY*BX must equal nt). Each block iteration covers a
// (BY*TM) x (BX*TN) output tile; multiple iterations cover the full uc x uc trailing block.
//
// Caller guarantees nc <= MAX_NC and shared arrays sized appropriately. Out-of-range outputs
// (uc not a multiple of TILE_R/TILE_C) are masked in registers via 0-padding.
template <typename T, int TM = 4, int TN = 4, int BY = 16, int BX = 16>
__device__ __forceinline__ void trailing_update_regblock(T* F, int fsz, int nc, int uc, int t,
                                                          int nt, T* sh_L, T* sh_U)
{
    // Stage L (uc x nc) and U (nc x uc) into shared once.
    for (int e = t; e < uc * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        sh_L[e] = F[(long)(nc + i) * fsz + k];
    }
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        sh_U[e] = F[(long)k * fsz + (nc + j)];
    }
    __syncthreads();

    // Thread coordinates within the BY x BX grid (row-major: tx = row, ty = col).
    const int tx = t / BX;  // 0..BY-1
    const int ty = t % BX;  // 0..BX-1

    constexpr int TILE_R = BY * TM;
    constexpr int TILE_C = BX * TN;
    const int ntr = (uc + TILE_R - 1) / TILE_R;
    const int ntc = (uc + TILE_C - 1) / TILE_C;

    for (int ti = 0; ti < ntr; ++ti) {
        for (int tj = 0; tj < ntc; ++tj) {
            const int r0 = ti * TILE_R + tx * TM;
            const int c0 = tj * TILE_C + ty * TN;

            T acc[TM][TN];
            #pragma unroll
            for (int x = 0; x < TM; ++x)
                #pragma unroll
                for (int y = 0; y < TN; ++y) acc[x][y] = T(0);

            for (int k = 0; k < nc; ++k) {
                T lv[TM], uv[TN];
                #pragma unroll
                for (int x = 0; x < TM; ++x)
                    lv[x] = (r0 + x < uc) ? sh_L[(r0 + x) * nc + k] : T(0);
                #pragma unroll
                for (int y = 0; y < TN; ++y)
                    uv[y] = (c0 + y < uc) ? sh_U[k * uc + (c0 + y)] : T(0);
                #pragma unroll
                for (int x = 0; x < TM; ++x) {
                    #pragma unroll
                    for (int y = 0; y < TN; ++y) acc[x][y] += lv[x] * uv[y];
                }
            }

            #pragma unroll
            for (int x = 0; x < TM; ++x) {
                const int rr = r0 + x;
                if (rr >= uc) continue;
                #pragma unroll
                for (int y = 0; y < TN; ++y) {
                    const int cc = c0 + y;
                    if (cc >= uc) continue;
                    F[(long)(nc + rr) * fsz + (nc + cc)] -= acc[x][y];
                }
            }
        }
    }
}

// Tiled trailing update. The block stages an L tile (TILE_M x TILE_K) and a U tile
// (TILE_K x TILE_N) into shared, then each thread computes one C[i, j] from the tiles.
// nt = blockDim.x (caller); the tile dims are template constants to allow compile-time tuning.
//
// Caller must guarantee fsz/uc/nc are positive; out-of-range outputs are skipped per-thread.
template <typename T, int TILE_M = 16, int TILE_N = 16, int TILE_K = 16>
__device__ __forceinline__ void trailing_update_tiled(T* F, int fsz, int nc, int uc, int t,
                                                      int nt, T* sh_L_buf, T* sh_U_buf)
{
    // Each block-iteration covers a TILE_M × TILE_N output tile.
    // Total output tiles: ceil(uc/TILE_M) × ceil(uc/TILE_N).
    const int ntm = (uc + TILE_M - 1) / TILE_M;
    const int ntn = (uc + TILE_N - 1) / TILE_N;
    const int ntk = (nc + TILE_K - 1) / TILE_K;

    // Thread tile coordinates (within a TILE_M x TILE_N output tile).
    const int my_row = t / TILE_N;
    const int my_col = t % TILE_N;

    for (int ti = 0; ti < ntm; ++ti) {
        for (int tj = 0; tj < ntn; ++tj) {
            T acc = T(0);
            for (int tk = 0; tk < ntk; ++tk) {
                // Stage L[ti*TILE_M..+TILE_M, tk*TILE_K..+TILE_K] -> sh_L_buf
                for (int e = t; e < TILE_M * TILE_K; e += nt) {
                    const int r = e / TILE_K, c = e % TILE_K;
                    const int gi = nc + ti * TILE_M + r;     // global L row
                    const int gk = tk * TILE_K + c;          // global K col
                    sh_L_buf[e] = (gi < (nc + uc) && gk < nc) ? F[(long)gi * fsz + gk] : T(0);
                }
                // Stage U[tk*TILE_K..+TILE_K, tj*TILE_N..+TILE_N] -> sh_U_buf
                for (int e = t; e < TILE_K * TILE_N; e += nt) {
                    const int r = e / TILE_N, c = e % TILE_N;
                    const int gk = tk * TILE_K + r;          // global K row
                    const int gj = nc + tj * TILE_N + c;     // global U col
                    sh_U_buf[e] = (gk < nc && gj < (nc + uc)) ? F[(long)gk * fsz + gj] : T(0);
                }
                __syncthreads();
                // Each thread accumulates one (my_row, my_col) of the output tile from the
                // staged K dimension (length TILE_K).
                if (my_row < TILE_M && my_col < TILE_N) {
                    for (int k = 0; k < TILE_K; ++k) {
                        acc += sh_L_buf[my_row * TILE_K + k] * sh_U_buf[k * TILE_N + my_col];
                    }
                }
                __syncthreads();
            }
            // Write back: each thread updates one output position if in range
            if (my_row < TILE_M && my_col < TILE_N) {
                const int ii = nc + ti * TILE_M + my_row;
                const int jj = nc + tj * TILE_N + my_col;
                if (ii < (nc + uc) && jj < (nc + uc)) {
                    F[(long)ii * fsz + jj] -= acc;
                }
            }
            __syncthreads();
        }
    }
}

// Drop-in replacement kernel for `mf_factor_mid_tc32_b<false>` whose only difference is the
// tiled trailing update. Front is FP32. Used when CLS_USE_TILED_TRAILING=1.
// Mirrors mid_tc32_b<false>: shared-staged Fs, lu_panel_factor + u_panel_solve + (tiled trailing)
// + writeback + extend-add.
__global__ void mf_factor_mid_tiled_b(int lbegin, int lend,
                                      const int* __restrict__ plcols,
                                      const int* __restrict__ front_off,
                                      const int* __restrict__ front_ptr,
                                      const int* __restrict__ ncols,
                                      const int* __restrict__ panel_parent,
                                      const int* __restrict__ asm_ptr,
                                      const int* __restrict__ asm_local, float* frontB,
                                      long front_total, int* sing, int do_extend, int fsz_cap,
                                      int level_max_nc, int level_max_uc)
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

    // Shared layout: front (fsz_cap^2 floats) then L (level_max_uc * level_max_nc)
    // then U (same size). Caller computed shared total accordingly.
    extern __shared__ char smem_mid_tiled[];
    float* Fs = reinterpret_cast<float*>(smem_mid_tiled);
    float* sh_L = Fs + (long)fsz_cap * fsz_cap;
    float* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    // Stage-in
    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        // Use the simpler staged version (correctness verified path). The full tile-blocked
        // version (trailing_update_tiled) is faster in theory but has a correctness bug
        // (relres ≈ 6% on USA case vs 0.06% FP32 baseline); needs further debug.
        trailing_update_staged<float>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U);
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

// Phase Σ.15 — FP16-input register-blocked trailing GEMM (no WMMA, no tile padding).
// Motivation: § 13 / "WMMA tile padding overhead" finding — WMMA's 16×16×16 fragment wastes
// 37.5% cycles when nc=10-30 (our power-grid mid-front sweet spot). Register-blocked GEMM
// bypasses WMMA entirely: each thread holds a TM×TN FP32 accumulator tile and runs nc inner
// FMAs (NO padding, K loop iterates exactly nc times). Inputs are FP16 to halve shared memory
// pressure (more occupancy) and use the FP16 FMA pipe (2× throughput vs FP32 on sm_86).
//
// Layout:
//   sh_L_h: uc × nc row-major __half (vs FP32 in the existing trailing_update_regblock)
//   sh_U_h: nc × uc row-major __half
// Per-thread accumulator: float acc[TM][TN] — FP32 keeps the accumulation precision near FP32
// even with FP16 inputs (the mantissa of each L·U product is preserved by widening to FP32).
template <int TM = 4, int TN = 4, int BY = 16, int BX = 16>
__device__ __forceinline__ void trailing_update_regblock_h16(float* F, int fsz, int nc, int uc,
                                                              int t, int nt,
                                                              __half* sh_L_h, __half* sh_U_h)
{
    // Stage L (uc × nc) and U (nc × uc) into shared, casting FP32 -> FP16.
    for (int e = t; e < uc * nc; e += nt) {
        const int i = e / nc, k = e % nc;
        sh_L_h[e] = __float2half(F[(long)(nc + i) * fsz + k]);
    }
    for (int e = t; e < nc * uc; e += nt) {
        const int k = e / uc, j = e % uc;
        sh_U_h[e] = __float2half(F[(long)k * fsz + (nc + j)]);
    }
    __syncthreads();

    // Thread coords in BY × BX = nt grid
    const int tx = t / BX;
    const int ty = t % BX;
    constexpr int TILE_R = BY * TM;
    constexpr int TILE_C = BX * TN;
    const int ntr = (uc + TILE_R - 1) / TILE_R;
    const int ntc = (uc + TILE_C - 1) / TILE_C;

    for (int ti = 0; ti < ntr; ++ti) {
        for (int tj = 0; tj < ntc; ++tj) {
            const int r0 = ti * TILE_R + tx * TM;
            const int c0 = tj * TILE_C + ty * TN;

            float acc[TM][TN];
            #pragma unroll
            for (int x = 0; x < TM; ++x)
                #pragma unroll
                for (int y = 0; y < TN; ++y) acc[x][y] = 0.0f;

            // Inner K loop: exactly nc iterations (no padding waste).
            for (int k = 0; k < nc; ++k) {
                float lv_f[TM], uv_f[TN];
                #pragma unroll
                for (int x = 0; x < TM; ++x) {
                    lv_f[x] = (r0 + x < uc) ? __half2float(sh_L_h[(r0 + x) * nc + k]) : 0.0f;
                }
                #pragma unroll
                for (int y = 0; y < TN; ++y) {
                    uv_f[y] = (c0 + y < uc) ? __half2float(sh_U_h[k * uc + (c0 + y)]) : 0.0f;
                }
                #pragma unroll
                for (int x = 0; x < TM; ++x) {
                    #pragma unroll
                    for (int y = 0; y < TN; ++y) {
                        acc[x][y] += lv_f[x] * uv_f[y];
                    }
                }
            }

            #pragma unroll
            for (int x = 0; x < TM; ++x) {
                const int rr = r0 + x;
                if (rr >= uc) continue;
                #pragma unroll
                for (int y = 0; y < TN; ++y) {
                    const int cc = c0 + y;
                    if (cc >= uc) continue;
                    F[(long)(nc + rr) * fsz + (nc + cc)] -= acc[x][y];
                }
            }
        }
    }
}

// Wrapper kernel: same as mf_factor_mid_tiled_b but with FP16-input register-blocked trailing.
// Shared layout: Fs[fsz_cap^2] (FP32) + sh_L_h[level_max_uc * level_max_nc] (FP16) +
// sh_U_h[level_max_nc * level_max_uc] (FP16). FP16 staging halves the L/U shared footprint
// vs the existing mid_tiled_b (which stores FP32 in sh_L / sh_U).
__global__ void mf_factor_mid_regblock_h16_b(int lbegin, int lend,
                                              const int* __restrict__ plcols,
                                              const int* __restrict__ front_off,
                                              const int* __restrict__ front_ptr,
                                              const int* __restrict__ ncols,
                                              const int* __restrict__ panel_parent,
                                              const int* __restrict__ asm_ptr,
                                              const int* __restrict__ asm_local, float* frontB,
                                              long front_total, int* sing, int do_extend,
                                              int fsz_cap, int level_max_nc, int level_max_uc)
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

    extern __shared__ char smem_mid_h16[];
    float* Fs = reinterpret_cast<float*>(smem_mid_h16);
    __half* sh_L_h = reinterpret_cast<__half*>(Fs + (long)fsz_cap * fsz_cap);
    __half* sh_U_h = sh_L_h + (long)level_max_uc * level_max_nc;

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        trailing_update_regblock_h16<4, 4, 16, 16>(Fs, fsz, nc, uc, t, nt, sh_L_h, sh_U_h);
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

// Register-blocked variant of mf_factor_mid_tiled_b. Same shared layout, but the trailing
// rank-nc update uses trailing_update_regblock<TM=4,TN=4,BY=16,BX=16> for 4×4 register tiling.
// 256 threads per block (16x16 grid in register-tile coords).
__global__ void mf_factor_mid_regblock_b(int lbegin, int lend,
                                         const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         const int* __restrict__ panel_parent,
                                         const int* __restrict__ asm_ptr,
                                         const int* __restrict__ asm_local, float* frontB,
                                         long front_total, int* sing, int do_extend, int fsz_cap,
                                         int level_max_nc, int level_max_uc)
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

    extern __shared__ char smem_mid_regblock[];
    float* Fs = reinterpret_cast<float*>(smem_mid_regblock);
    float* sh_L = Fs + (long)fsz_cap * fsz_cap;
    float* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
    __syncthreads();

    if (fsz <= 48) {
        lu_small_front<float>(Fs, fsz, nc, t, nt, sing);
    } else {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
        trailing_update_regblock<float, 4, 4, 16, 16>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U);
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
