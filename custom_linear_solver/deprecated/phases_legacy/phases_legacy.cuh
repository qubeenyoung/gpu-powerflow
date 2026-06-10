// [ARCHIVED 2026-06-10] Dead factorize/phases.cuh helpers — unreachable from the active
// kernels (factor_small / factor_mid_blocked / factor_big) after the kernel unification.
//   trailing_update_mma_tf32_direct_shared : trailing of the removed factor_mid_tf32_ptx
//   *_warp / trailing_update_mma_tf32_warp_shared : removed warp-per-front TF32 path
//   u_panel_solve_column_owned : superseded by the inline column-usolve in factorize_front_blocked_tf32
// Reference only — not compiled.

// --- u_panel_solve_column_owned ---
template <typename T>
__device__ __forceinline__ void u_panel_solve_column_owned(T* F, int fsz, int nc, int uc,
                                                           int t, int nt)
{
    // Each U column is independent once the panel LU is complete. Keep one column on the same
    // thread for the whole triangular solve so the per-row block-wide barriers disappear.
    for (int e = t; e < uc; e += nt) {
        const int jj = nc + e;
        for (int k = 1; k < nc; ++k) {
            T v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) {
                v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            }
            F[(long)k * fsz + jj] = v;
        }
    }
    __syncthreads();
}

// --- trailing_update_mma_tf32_direct_shared ---
// TF32 PTX trailing that reads L/U directly from a shared-resident front Fs. This is intended
// for the mid tier: the full front is already in shared memory, so re-staging L/U into padded
// scratch can cost more than it saves. Out-of-bounds K/M/N lanes are zeroed in registers.
template <bool FuseExtend = false>
__device__ __forceinline__ void trailing_update_mma_tf32_direct_shared(float* F, int fsz,
                                                                        int nc, int uc,
                                                                        int t, int nt,
                                                                        float* Fp = nullptr,
                                                                        int pfsz = 0,
                                                                        const int* asm_local = nullptr,
                                                                        int abase = 0)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP  = ((nc + 7)  / 8)  * 8;

    const int ntj16 = UCP / 16;
    const int ntj8  = UCP / 8;
    const int nks   = KP  / 8;
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;
    const int laneC = (lane & 3) * 2;

    auto load_l = [&](int r, int k) {
        return (r < uc && k < nc) ? F[(long)(nc + r) * fsz + k] : 0.0f;
    };
    auto load_u = [&](int k, int col) {
        return (k < nc && col < uc) ? F[(long)k * fsz + (nc + col)] : 0.0f;
    };
    auto drain = [&](int r, int col, float c) {
        if (r >= uc || col >= uc) return;
        const long off = (long)(nc + r) * fsz + (nc + col);
        if constexpr (FuseExtend) {
            atomicAdd(&Fp[(long)asm_local[abase + r] * pfsz + asm_local[abase + col]],
                      F[off] - c);
        } else {
            F[off] -= c;
        }
    };

    constexpr int NTJ8_MAX =
        8;
    if (ntj8 <= NTJ8_MAX) {
        for (int ti = warp; ti < ntj16; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
            for (int kc = 0; kc < nks; ++kc) {
                const int k0 = kc * 8 + laneC;
                const int k1 = k0 + 1;
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(load_l(r_top, k0));
                const Tf32Pair a1 = tf32_ozaki_pair(load_l(r_bot, k0));
                const Tf32Pair a2 = tf32_ozaki_pair(load_l(r_top, k1));
                const Tf32Pair a3 = tf32_ozaki_pair(load_l(r_bot, k1));
#else
                const unsigned a0 = __float_as_uint(load_l(r_top, k0));
                const unsigned a1 = __float_as_uint(load_l(r_bot, k0));
                const unsigned a2 = __float_as_uint(load_l(r_top, k1));
                const unsigned a3 = __float_as_uint(load_l(r_bot, k1));
#endif
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= ntj8) break;
                    const int col = tj8 * 8 + laneR;
#ifdef CLS_TF32_OZAKI_TC2
                    const Tf32Pair b0 = tf32_ozaki_pair(load_u(k0, col));
                    const Tf32Pair b1 = tf32_ozaki_pair(load_u(k1, col));
                    CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                        a0, a1, a2, a3, b0, b1);
#else
                    const unsigned b0 = __float_as_uint(load_u(k0, col));
                    const unsigned b1 = __float_as_uint(load_u(k1, col));
                    CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                         a0, a1, a2, a3, b0, b1);
#endif
                }
            }
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= ntj8) break;
                const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
                drain(r_top, col0, c[tj8][0]);
                drain(r_top, col1, c[tj8][1]);
                drain(r_bot, col0, c[tj8][2]);
                drain(r_bot, col1, c[tj8][3]);
            }
        }
        return;
    }

    for (int ti = warp; ti < ntj16; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < ntj8; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < nks; ++kc) {
                const int k0 = kc * 8 + laneC;
                const int k1 = k0 + 1;
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair a0 = tf32_ozaki_pair(load_l(r_top, k0));
                const Tf32Pair a1 = tf32_ozaki_pair(load_l(r_bot, k0));
                const Tf32Pair a2 = tf32_ozaki_pair(load_l(r_top, k1));
                const Tf32Pair a3 = tf32_ozaki_pair(load_l(r_bot, k1));
                const int col = tj8 * 8 + laneR;
                const Tf32Pair b0 = tf32_ozaki_pair(load_u(k0, col));
                const Tf32Pair b1 = tf32_ozaki_pair(load_u(k1, col));
                CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
                const unsigned a0 = __float_as_uint(load_l(r_top, k0));
                const unsigned a1 = __float_as_uint(load_l(r_bot, k0));
                const unsigned a2 = __float_as_uint(load_l(r_top, k1));
                const unsigned a3 = __float_as_uint(load_l(r_bot, k1));
                const int col = tj8 * 8 + laneR;
                const unsigned b0 = __float_as_uint(load_u(k0, col));
                const unsigned b1 = __float_as_uint(load_u(k1, col));
                CLS_MMA_TF32_M16N8K8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
            }
            const int col0 = tj8 * 8 + laneC;
            const int col1 = col0 + 1;
            drain(r_top, col0, c0);
            drain(r_top, col1, c1);
            drain(r_bot, col0, c2);
            drain(r_bot, col1, c3);
        }
    }
}

// --- lu_panel_factor_warp ---
__device__ __forceinline__ void lu_panel_factor_warp(float* F, int fsz, int nc, int sl,
                                                     unsigned mask, int* sing)
{
    for (int k = 0; k < nc; ++k) {
        float piv = F[(long)k * fsz + k];
        if (piv == 0.0f) { if (sl == 0) *sing = 1; piv = 1.0f; }
        const float inv_piv = 1.0f / piv;
        for (int i = k + 1 + sl; i < fsz; i += kWarpSize) {
            F[(long)i * fsz + k] *= inv_piv;
        }
        __syncwarp(mask);
        const int pc = nc - 1 - k;
        for (int e = sl; e < (fsz - k - 1) * pc; e += kWarpSize) {
            const int ii = k + 1 + e / pc;
            const int jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncwarp(mask);
    }
}

// --- u_panel_solve_warp ---
__device__ __forceinline__ void u_panel_solve_warp(float* F, int fsz, int nc, int uc, int sl,
                                                   unsigned mask)
{
    for (int k = 1; k < nc; ++k) {
        for (int e = sl; e < uc; e += kWarpSize) {
            const int jj = nc + e;
            float v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncwarp(mask);
    }
}

// --- u_panel_solve_warp_column_owned ---
__device__ __forceinline__ void u_panel_solve_warp_column_owned(float* F, int fsz, int nc,
                                                                int uc, int sl, unsigned mask)
{
    for (int e = sl; e < uc; e += kWarpSize) {
        const int jj = nc + e;
        for (int k = 1; k < nc; ++k) {
            float v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) {
                v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            }
            F[(long)k * fsz + jj] = v;
        }
    }
    __syncwarp(mask);
}

// --- trailing_update_mma_tf32_warp_shared ---
__device__ __forceinline__ void trailing_update_mma_tf32_warp_shared(float* F, int fsz, int nc,
                                                                     int uc, int lane)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP = ((nc + 7) / 8) * 8;
    const int ntj16 = UCP / 16;
    const int ntj8 = UCP / 8;
    const int nks = KP / 8;
    const int laneR = lane >> 2;
    const int laneC = (lane & 3) * 2;

    auto load_l = [&](int r, int k) {
        return (r < uc && k < nc) ? F[(long)(nc + r) * fsz + k] : 0.0f;
    };
    auto load_u = [&](int k, int col) {
        return (k < nc && col < uc) ? F[(long)k * fsz + (nc + col)] : 0.0f;
    };
    auto drain = [&](int r, int col, float c) {
        if (r < uc && col < uc) F[(long)(nc + r) * fsz + (nc + col)] -= c;
    };

    constexpr int NTJ8_MAX = 4;
    for (int ti = 0; ti < ntj16; ++ti) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        float c[NTJ8_MAX][4];
        #pragma unroll
        for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
        for (int kc = 0; kc < nks; ++kc) {
            const int k0 = kc * 8 + laneC;
            const int k1 = k0 + 1;
#ifdef CLS_TF32_OZAKI_TC2
            const Tf32Pair a0 = tf32_ozaki_pair(load_l(r_top, k0));
            const Tf32Pair a1 = tf32_ozaki_pair(load_l(r_bot, k0));
            const Tf32Pair a2 = tf32_ozaki_pair(load_l(r_top, k1));
            const Tf32Pair a3 = tf32_ozaki_pair(load_l(r_bot, k1));
#else
            const unsigned a0 = __float_as_uint(load_l(r_top, k0));
            const unsigned a1 = __float_as_uint(load_l(r_bot, k0));
            const unsigned a2 = __float_as_uint(load_l(r_top, k1));
            const unsigned a3 = __float_as_uint(load_l(r_bot, k1));
#endif
            #pragma unroll
            for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                if (tj8 >= ntj8) break;
                const int col = tj8 * 8 + laneR;
#ifdef CLS_TF32_OZAKI_TC2
                const Tf32Pair b0 = tf32_ozaki_pair(load_u(k0, col));
                const Tf32Pair b1 = tf32_ozaki_pair(load_u(k1, col));
                CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                    a0, a1, a2, a3, b0, b1);
#else
                const unsigned b0 = __float_as_uint(load_u(k0, col));
                const unsigned b1 = __float_as_uint(load_u(k1, col));
                CLS_MMA_TF32_M16N8K8(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3],
                                     a0, a1, a2, a3, b0, b1);
#endif
            }
        }
        #pragma unroll
        for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
            if (tj8 >= ntj8) break;
            const int col0 = tj8 * 8 + laneC;
            const int col1 = col0 + 1;
            drain(r_top, col0, c[tj8][0]);
            drain(r_top, col1, c[tj8][1]);
            drain(r_bot, col0, c[tj8][2]);
            drain(r_bot, col1, c[tj8][3]);
        }
    }
}
