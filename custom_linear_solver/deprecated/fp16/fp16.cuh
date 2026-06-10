// [ARCHIVED 2026-06-10] FP16 PTX trailing kernels + helpers, removed from the active
// build (Precision::FP16 dropped). Reference only — not compiled.

// ===== from phases.cuh =====
// --- trailing_update_mma_fp16_ptx (orig lines 366..505) ---
// ---------------------------------------------------------------------------------------
//  FP16 PTX K=8       (mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32)
//
// FP16 inputs, FP32 accumulate. Per-lane register-direct trailing, structurally identical to
// the TF32 K=8 path below (same 16×8 output tile, same (ti, kc, tj8) A-reuse loop, same drain
// into F) — the only differences are that the staging panels hold __half and that FP16 packs
// two elements per 32-bit register, so m16n8k8.f16 takes 2 A-regs + 1 B-reg instead of TF32's
// 4 + 2.
//
// Per-lane multiplicand layout (probed). groupID = laneR = lane >> 2 (0..7); tid = lane & 3
// (0..3); laneC = tid * 2 ∈ {0,2,4,6}:
//   A (16×8, row-major Lh, ld = KP): two .f16 packed per reg, contiguous in K →
//       a0 = pack(A[laneR + 0, laneC], A[laneR + 0, laneC + 1])   (M_top, K pair)
//       a1 = pack(A[laneR + 8, laneC], A[laneR + 8, laneC + 1])   (M_bot, K pair)
//     loadable as a single 4-byte read of two adjacent halves (laneC even ⇒ aligned).
//   B (8×8, row-major Uh, ld = UCP): two .f16 packed per reg, contiguous in K (stride UCP in
//     memory, so packed explicitly) →
//       b0 = pack(B[laneC, laneR], B[laneC + 1, laneR])           (K pair, N = laneR)
//   C accumulator: identical to every m16n8 shape — see the section header.
template <bool FuseExtend = false>
__device__ __forceinline__ void trailing_update_mma_fp16_ptx(float* F, int fsz, int nc, int uc,
                                                             __half* Lh, __half* Uh,
                                                             int t, int nt,
                                                             float* Fp = nullptr,
                                                             int pfsz = 0,
                                                             const int* asm_local = nullptr,
                                                             int abase = 0)
{
    const int UCP = ((uc + 15) / 16) * 16;
    const int KP  = ((nc + 7)  / 8)  * 8;

    // (a) Stage L → Lh and U → Uh as __half, padded with zeros.
    for (int e = t; e < UCP * KP; e += nt) {
        const int i = e / KP, k = e % KP;
        Lh[e] = (i < uc && k < nc) ? __float2half(F[(long)(nc + i) * fsz + k]) : __half(0.0f);
    }
    for (int e = t; e < KP * UCP; e += nt) {
        const int k = e / UCP, j = e % UCP;
        Uh[e] = (k < nc && j < uc) ? __float2half(F[(long)k * fsz + (nc + j)]) : __half(0.0f);
    }
    __syncthreads();

    const int ntj16 = UCP / 16;            // 16-row tiles (M)
    const int ntj8  = UCP / 8;             // 8-col tiles  (N)
    const int nks   = KP  / 8;             // K-loop count (mma K=8)
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;           // 0..7
    const int laneC = (lane & 3) * 2;      // 0,2,4,6

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

    // A-reuse hoisted path. Capped at NTJ8_MAX = 8 N-tiles (UCP ≤ 64); the unrolled tj8 loop
    // keeps the per-tile accumulators in named registers for the inline-asm "+f" binding.
    constexpr int NTJ8_MAX = 8;
    if (ntj8 <= NTJ8_MAX) {
        for (int ti = warp; ti < ntj16; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) {
                c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;
            }
            // (b) Outer K loop: load A once per K-tile (two contiguous halves → one 4-byte
            //     read), then sweep all N tiles inside.
            for (int kc = 0; kc < nks; ++kc) {
                const unsigned a0 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 0) * KP + kc * 8 + laneC]);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 8) * KP + kc * 8 + laneC]);
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= ntj8) break;
                    // B pair is strided by UCP in memory → pack the two halves by hand.
                    const __half2 bh = __halves2half2(
                        Uh[(long)(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR],
                        Uh[(long)(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(&bh);
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+f"(c[tj8][0]), "+f"(c[tj8][1]), "+f"(c[tj8][2]), "+f"(c[tj8][3])
                        : "r"(a0), "r"(a1), "r"(b0));
                }
            }
            // (c) Drain accumulators straight into F with uc bounds checks.
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

    // Fall-through for UCP > 64 (big-tier strips wider than NTJ8_MAX × 8). The (ti, tj8, kc)
    // ordering reloads A per tj8, but the absolute A-reuse saving is smaller at this size.
    for (int ti = warp; ti < ntj16; ti += nwarp) {
        const int r_top = ti * 16 + laneR;
        const int r_bot = r_top + 8;
        for (int tj8 = 0; tj8 < ntj8; ++tj8) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (int kc = 0; kc < nks; ++kc) {
                const unsigned a0 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 0) * KP + kc * 8 + laneC]);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(
                    &Lh[(long)(ti * 16 + laneR + 8) * KP + kc * 8 + laneC]);
                const __half2 bh = __halves2half2(
                    Uh[(long)(kc * 8 + laneC + 0) * UCP + tj8 * 8 + laneR],
                    Uh[(long)(kc * 8 + laneC + 1) * UCP + tj8 * 8 + laneR]);
                const unsigned b0 = *reinterpret_cast<const unsigned*>(&bh);
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(b0));
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            drain(r_top, col0, c0);
            drain(r_top, col1, c1);
            drain(r_bot, col0, c2);
            drain(r_bot, col1, c3);
        }
    }
}

// --- block_update_mma_fp16_direct_shared (orig lines 942..1030) ---
__device__ __forceinline__ void block_update_mma_fp16_direct_shared(float* F, int fsz,
                                                                    int row0, int col0,
                                                                    int dim, int k0, int kb,
                                                                    int t, int nt)
{
    if (dim <= 0 || kb <= 0) return;
    const int DP = ((dim + 15) / 16) * 16;
    const int KP = ((kb + 7) / 8) * 8;
    const int ntj16 = DP / 16;
    const int ntj8  = DP / 8;
    const int nks   = KP / 8;
    const int warp  = t >> 5;
    const int nwarp = nt >> 5;
    const int lane  = t & 31;
    const int laneR = lane >> 2;
    const int laneC = (lane & 3) * 2;

    auto load_l = [&](int r, int k) {
        return (r < dim && k < kb) ? F[(long)(row0 + r) * fsz + (k0 + k)] : 0.0f;
    };
    auto load_u = [&](int k, int col) {
        return (k < kb && col < dim) ? F[(long)(k0 + k) * fsz + (col0 + col)] : 0.0f;
    };
    auto drain = [&](int r, int col, float c) {
        if (r < dim && col < dim) F[(long)(row0 + r) * fsz + (col0 + col)] -= c;
    };

    constexpr int NTJ8_MAX = 16;
    if (ntj8 <= NTJ8_MAX) {
        for (int ti = warp; ti < ntj16; ti += nwarp) {
            const int r_top = ti * 16 + laneR;
            const int r_bot = r_top + 8;
            float c[NTJ8_MAX][4];
            #pragma unroll
            for (int j = 0; j < NTJ8_MAX; ++j) { c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
            for (int kc = 0; kc < nks; ++kc) {
                const int k = kc * 8 + laneC;
                const unsigned a0 = pack_f16x2(load_l(r_top, k + 0), load_l(r_top, k + 1));
                const unsigned a1 = pack_f16x2(load_l(r_bot, k + 0), load_l(r_bot, k + 1));
                #pragma unroll
                for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
                    if (tj8 >= ntj8) break;
                    const int col = tj8 * 8 + laneR;
                    const unsigned b0 = pack_f16x2(load_u(k + 0, col), load_u(k + 1, col));
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                        : "+f"(c[tj8][0]), "+f"(c[tj8][1]), "+f"(c[tj8][2]), "+f"(c[tj8][3])
                        : "r"(a0), "r"(a1), "r"(b0));
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
                const int k = kc * 8 + laneC;
                const unsigned a0 = pack_f16x2(load_l(r_top, k + 0), load_l(r_top, k + 1));
                const unsigned a1 = pack_f16x2(load_l(r_bot, k + 0), load_l(r_bot, k + 1));
                const int col = tj8 * 8 + laneR;
                const unsigned b0 = pack_f16x2(load_u(k + 0, col), load_u(k + 1, col));
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a0), "r"(a1), "r"(b0));
            }
            const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
            drain(r_top, col0, c0);
            drain(r_top, col1, c1);
            drain(r_bot, col0, c2);
            drain(r_bot, col1, c3);
        }
    }
}

// --- factorize_front_blocked_fp16 (orig lines 1116..1155) ---
__device__ __forceinline__ void factorize_front_blocked_fp16(float* F, int fsz, int nc,
                                                             int t, int nt, int* sing)
{
    constexpr int BK = 8;
    for (int k0 = 0; k0 < nc; k0 += BK) {
        const int kb = (k0 + BK <= nc) ? BK : (nc - k0);
        const int next = k0 + kb;

        for (int kk = 0; kk < kb; ++kk) {
            const int k = k0 + kk;
            float piv = F[(long)k * fsz + k];
            if (piv == 0.0f) { if (t == 0) *sing = 1; piv = 1.0f; }
            const float inv_piv = 1.0f / piv;
            for (int i = k + 1 + t; i < fsz; i += nt) {
                const float lik = F[(long)i * fsz + k] * inv_piv;
                F[(long)i * fsz + k] = lik;
                for (int jj = k + 1; jj < next; ++jj) {
                    F[(long)i * fsz + jj] -= lik * F[(long)k * fsz + jj];
                }
            }
            __syncthreads();
        }

        for (int kk = 0; kk < kb; ++kk) {
            const int row = k0 + kk;
            for (int j = next + t; j < fsz; j += nt) {
                float v = F[(long)row * fsz + j];
                for (int i = k0; i < row; ++i) v -= F[(long)row * fsz + i] * F[(long)i * fsz + j];
                F[(long)row * fsz + j] = v;
            }
            __syncthreads();
        }

        const int dim = fsz - next;
        if (dim > 0) {
            block_update_mma_fp16_direct_shared(F, fsz, next, next, dim, k0, kb, t, nt);
            __syncthreads();
        }
    }
}

// ===== from kernels.cuh =====
// --- factor_mid_fp16_ptx (orig lines 219..282) ---
// FP16 PTX mid trailing on a shared-resident front. The front stays as FP32 in shared memory;
// only the L/U panels are converted to __half before mma.sync. Non-root TC fronts drain the
// accumulator directly into the parent front, because the mid-tier C block is only consumed by
// extend-add and is not written back. Fronts that are too small or too skinny for Tensor Cores
// keep the scalar staged path inside the same kernel.
__global__ void factor_mid_fp16_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ panel_parent,
                                    const int* __restrict__ asm_ptr,
                                    const int* __restrict__ asm_local, float* frontB,
                                    long front_total, int* sing, int do_extend, int fsz_cap,
                                    int level_max_nc, int level_max_uc, int ucp_max,
                                    int kp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;

    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;
    const int par = panel_parent[p];
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 10 && nc <= 32 && uc <= 256);
    const bool use_fused_trailing =
        (use_tc && par >= 0 && do_extend && extend_add_allowed_for_uc(uc));
    float* Fp = use_fused_trailing ? (front + front_off[par]) : nullptr;
    const int pfsz = use_fused_trailing ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = use_fused_trailing ? asm_ptr[p] : 0;

    extern __shared__ char smem_mid_fp16_ptx[];
    float* Fs   = reinterpret_cast<float*>(smem_mid_fp16_ptx);
    __half* Lh  = reinterpret_cast<__half*>(Fs + (long)fsz_cap * fsz_cap);
    __half* Uh  = Lh + (long)ucp_max * kp_max;

    stage_in_async<float>(Fs, F, fsz2, t, nt);
    __syncthreads();

    factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (use_fused_trailing) {
            trailing_update_mma_fp16_ptx<true>(Fs, fsz, nc, uc, Lh, Uh, t, nt, Fp, pfsz,
                                               asm_local, abase);
        } else if (use_tc) {
            trailing_update_mma_fp16_ptx<false>(Fs, fsz, nc, uc, Lh, Uh, t, nt);
        } else {
            trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt);
        }
    });

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* parent_front = front + front_off[par];
    const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
    const int asm_base = asm_ptr[p];
    extend_add<float, float>(parent_front, parent_fsz, Fs, fsz, nc, uc, asm_local, asm_base,
                             t, nt);
}

// --- factor_big_fp16_ptx (orig lines 575..653) ---
// FP16 PTX trailing on the global front (Precision::FP16). 512-thread block with
// __launch_bounds__(512, 2) so two blocks stay resident per SM on sm_86. Shared scratch is
// only the __half L/U staging panels; the inline-asm mma writes the accumulator to per-lane
// registers, so no Csc readback scratch is needed.
__global__ void __launch_bounds__(512, 2)
                                factor_big_fp16_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, float* frontB,
                                long front_total, int* sing, int do_extend, int ucp_max,
                                int kp_max)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    float* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    float* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;

#ifdef CLS_FUSE_FP16_TRAIL_EXTEND
    const int par = panel_parent[p];
    const bool use_fused_trailing =
        (par >= 0 && do_extend && extend_add_allowed_for_uc(uc) &&
         fsz > 48 && nc <= 32 && uc <= 256);
    float* Fp = use_fused_trailing ? (front + front_off[par]) : nullptr;
    const int pfsz = use_fused_trailing ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = use_fused_trailing ? asm_ptr[p] : 0;
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_fp16_ptx[];
            __half* Lh = reinterpret_cast<__half*>(smem_big_fp16_ptx);
            __half* Uh = Lh + (long)ucp_max * kp_max;
            if (use_fused_trailing) {
                trailing_update_mma_fp16_ptx<true>(F, fsz, nc, uc, Lh, Uh, t, nt, Fp, pfsz,
                                                   asm_local, abase);
            } else {
                trailing_update_mma_fp16_ptx<false>(F, fsz, nc, uc, Lh, Uh, t, nt);
            }
        }
    });

    if (use_fused_trailing || par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    {
        float* parent_front = front + front_off[par];
        const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
        const int asm_base = asm_ptr[p];
        extend_add<float, float>(parent_front, parent_fsz, F, fsz, nc, uc, asm_local, asm_base,
                                 t, nt);
    }
#else
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_fp16_ptx[];
            __half* Lh = reinterpret_cast<__half*>(smem_big_fp16_ptx);
            __half* Uh = Lh + (long)ucp_max * kp_max;
            trailing_update_mma_fp16_ptx(F, fsz, nc, uc, Lh, Uh, t, nt);
        }
    });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
#endif
}
