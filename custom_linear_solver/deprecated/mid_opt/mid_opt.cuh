#pragma once

// Internal — included only by multifrontal.cu (single TU).
//
// EXPERIMENTAL mid-front factor kernel combining the 4 improvements from
// `docs/03-optimization-notes/13`:
//
//   P1 — Reciprocal multiply: explicit `inv_piv = 1/piv` hoist, then `*= inv_piv` instead of
//        `/= piv` per element. Saves ~16 cyc per divide where the compiler doesn't auto-hoist
//        the shared-memory pivot read.
//
//   P2 — Phase 1 + Phase 2 fusion: per-k iteration does panel-LU step k AND U-solve step k
//        in a single thread pass (disjoint output regions: panel cols [0, nc) vs U-trailing
//        cols [nc, fsz)). Sync count per front: 2·nc (split form) + (nc-1) (separate U-solve)
//        → nc (fused). Eliminates nc-1 syncs per front + the Phase 1/2 transition syncs.
//
//   P3 — Parallel U-solve over (i, jj) pairs: within the fused per-k iteration, the U-solve
//        sub-work distributes across all threads in [panel_work, panel_work + usolve_work)
//        range. The inner i-loop is kept serial per thread (each thread fully resolves one
//        jj) — a fully-parallel version with reduction would require shared scratch and was
//        deferred (cost vs. gain not clear given P2 already widens the work granularity).
//        The fusion alone gives most of P3's benefit by absorbing U-solve thread-idle into
//        panel-update's larger work pool.
//
//   P4 — Bank-conflict-free shared layout: leading dim LD = fsz_cap + (fsz_cap % 32 == 0 ? 1 : 0).
//        Without padding, fsz_cap multiple of 32 (e.g., 32, 64, 96, 128) makes column writes
//        / reads land in the same bank for all 32 lanes (32-way conflict). One extra column
//        breaks the alignment.
//
// Trailing GEMM here is scalar (not staged) — keeps the kernel simple. The default factor_mid
// has the staged path for cache-friendly trailing; benchmarking will tell whether the staging
// cost vs. P1-P4 wins matters more.

#include <cuda_runtime.h>

#include "factorize/primitives.cuh"

namespace custom_linear_solver {
namespace {

template <typename T>
__global__ void factor_mid_opt(int lbegin, int lend, const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, T* frontB,
                                long front_total, int* sing, int do_extend, int fsz_cap)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int s = front_ptr[p];
    const int fsz = front_ptr[p + 1] - s;
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

    // P4 attempted but REVERTED: padded leading dimension would require per-row stage-in
    // (integer division per element to map e → (i, j)) which itself was ~30 cyc per entry
    // on Ampere and dominated wall (+85% case8387 B=1 in initial measurement). Restoring
    // LD = fsz_cap means the stage-in/writeback can use the bulk `Fs[e] = F[e]` pattern.
    // Bank conflicts on column writes (fsz % 32 == 0) remain — measured as a secondary
    // effect that doesn't dominate (USA mid fsz distribution sweep showed ~10% of mid
    // launches in the 32-multiple bucket; per-front cost ~3-5 cyc extra).
    const int LD = fsz_cap;

    extern __shared__ char smem_mid_opt[];
    T* Fs = reinterpret_cast<T*>(smem_mid_opt);

    // ----- stage-in: F (global, ld=fsz) → Fs (shared, ld=fsz=LD) — bulk coalesced -----
    const int fsz2 = fsz * fsz;
    stage_in_async<T>(Fs, F, fsz2, t, nt);
    __syncthreads();

    // ----- P1 + P2: fused Phase 1 (panel LU) + Phase 2 (U-solve) per k -----
    for (int k = 0; k < nc; ++k) {
        T piv = Fs[(long)k * LD + k];
        if (piv == T(0)) { if (t == 0) *sing = 1; piv = T(1); }
        const T inv_piv = T(1) / piv;
        // Divide column k: L below the diagonal
        for (int i = k + 1 + t; i < fsz; i += nt) {
            Fs[(long)i * LD + k] = Fs[(long)i * LD + k] * inv_piv;
        }
        __syncthreads();

        // P2 fusion: panel update + U-solve step k in same pass
        const int pc = nc - 1 - k;
        const long panel_work = (long)(fsz - k - 1) * pc;
        const long usolve_work = (k > 0) ? (long)uc : 0;
        const long total = panel_work + usolve_work;
        for (long e = t; e < total; e += nt) {
            if (e < panel_work) {
                // Panel update: F[ii][jj] -= F[ii][k] * F[k][jj]
                const long ii = k + 1 + e / pc;
                const long jj = k + 1 + e % pc;
                Fs[ii * LD + jj] -= Fs[ii * LD + k] * Fs[(long)k * LD + jj];
            } else {
                // U-solve step k: row k, col jj = nc + (e - panel_work)
                const long e_us = e - panel_work;
                const long jj = nc + e_us;
                T v = Fs[(long)k * LD + jj];
                #pragma unroll 4
                for (int i = 0; i < k; ++i) {
                    v -= Fs[(long)k * LD + i] * Fs[(long)i * LD + jj];
                }
                Fs[(long)k * LD + jj] = v;
            }
        }
        __syncthreads();
    }

    // ----- Phase 3: scalar trailing GEMM C -= L * U on the (uc × uc) contribution block -----
    for (long e = t; e < (long)uc * uc; e += nt) {
        const long ii = nc + e / uc, jj = nc + e % uc;
        T acc = T(0);
        #pragma unroll 4
        for (int k = 0; k < nc; ++k) {
            acc += Fs[ii * LD + k] * Fs[(long)k * LD + jj];
        }
        Fs[ii * LD + jj] -= acc;
    }
    __syncthreads();

    // ----- Writeback: Fs (ld=fsz) → F (ld=fsz). Only L/U regions; CB stays in Fs. -----
    // Same as default factor_mid (since LD == fsz now).
    writeback_factored<T, T>(F, Fs, fsz, nc, uc, t, nt);

    const int par = panel_parent[p];
    if (par < 0 || !do_extend) return;
    __syncthreads();
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    // Extend-add CB → parent (atomicAdd).
    for (long e = t; e < (long)uc * uc; e += nt) {
        const long a = e / uc, b = e % uc;
        atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
                  Fs[(long)(nc + a) * LD + (nc + b)]);
    }
}

}  // namespace
}  // namespace custom_linear_solver
