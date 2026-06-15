// [ARCHIVED 2026-06-10] Big-tier kernels superseded by factor_big_unified<T,UseTC>:
//   factor_big_tf32_ptx (naive thin-K TF32) ; multi-block underfill path
//   (factor_big_panel + factor_big_trailing_mb + factor_big_extend + launch_factor_big_mb).
// Reference only — not compiled.

// ===== from kernels.cuh =====
// --- factor_big_panel ---
// ----- Multi-block big-front split (B=1 / underfilled-level path) ----------------------------
//
// When a big level has too few fronts to fill the GPU (level_size × B < num_SMs, e.g. the deep
// levels of a single-system solve), the fused factor_big runs one block per front — a handful
// of busy SMs while the rest idle, with the FLOP-heavy trailing GEMM serialized onto those few
// blocks. These three kernels split the work so the trailing fans out across many blocks:
//
//   factor_big_panel<T>     Phase 1 (panel LU) + Phase 2 (U-solve), one block per front. Small
//                           fronts (fsz ≤ 48) take the fused lu_small_front (panel + trailing in
//                           one pass) so the trailing kernel can skip them.
//   factor_big_trailing_mb  Phase 3 (scalar trailing) split into blockIdx.z element tiles of
//                           `elems_per_block` C entries — grid (front, batch, tiles).
//   factor_big_extend<T>    Phase 4 (extend-add) into the parent front, one block per front.
//
// Each C output element is owned by exactly one tile and L/U are read-only, so the tiles are
// race-free; the kernel-launch boundaries order panel → trailing → extend within the stream.
template <typename T>
__global__ void factor_big_panel(int lbegin, int lend, const int* __restrict__ plcols,
                                 const int* __restrict__ front_off,
                                 const int* __restrict__ front_ptr,
                                 const int* __restrict__ ncols, T* frontB,
                                 long front_total, int* sing)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    if (fsz <= 48) {
        lu_small_front<T>(F, fsz, nc, t, nt, sing);   // Phase 1 + 3 fused; trailing kernel skips these
    } else {
        lu_panel_factor<T>(F, fsz, nc, t, nt, sing);  // Phase 1
        u_panel_solve<T>(F, fsz, nc, uc, t, nt);      // Phase 2 (trailing deferred to the MB kernel)
    }
}

// --- factor_big_trailing_mb ---
template <typename T>
__global__ void factor_big_trailing_mb(int lbegin, int lend, const int* __restrict__ plcols,
                                       const int* __restrict__ front_off,
                                       const int* __restrict__ front_ptr,
                                       const int* __restrict__ ncols, T* frontB,
                                       long front_total, int elems_per_block)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    if (fsz <= 48) return;                               // already trailed by the fused panel kernel
    const long total = (long)uc * uc;
    const long base = (long)blockIdx.z * elems_per_block;
    if (base >= total) return;
    const long end = (base + elems_per_block < total) ? base + elems_per_block : total;
    T* F = frontB + (long)blockIdx.y * front_total + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    // Scalar trailing C[ii,jj] -= Σ_k L[ii,k]·U[k,jj] over this block's element slice.
    for (long ee = base + t; ee < end; ee += nt) {
        const int ii = nc + (int)(ee / uc), jj = nc + (int)(ee % uc);
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

// --- factor_big_extend ---
template <typename T>
__global__ void factor_big_extend(int lbegin, int lend, const int* __restrict__ plcols,
                                  const int* __restrict__ front_off,
                                  const int* __restrict__ front_ptr,
                                  const int* __restrict__ ncols,
                                  const int* __restrict__ panel_parent,
                                  const int* __restrict__ asm_ptr,
                                  const int* __restrict__ asm_local, T* frontB,
                                  long front_total)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;
    const int p = plcols[idx];
    const int par = panel_parent[p];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    const int uc = fsz - nc;
    if (par < 0 || !extend_add_allowed_for_uc(uc)) return;
    T* front = frontB + (long)blockIdx.y * front_total;
    T* F = front + front_off[p];
    T* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    const int t = threadIdx.x, nt = blockDim.x;
    extend_add<T, T>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}

// --- factor_big_tf32_ptx ---
// TF32 PTX trailing on the global front (Precision::TF32). The 512-thread block plus
// NOTE: __launch_bounds__ removed. The old EXP-B `__launch_bounds__(512, 2)` capped registers
// at 64 to keep 2 blocks/SM resident — good for the non-fused trailing, but with the fused
// trail+extend drain the kernel needs ~106 registers, so the 64 cap forced a 24-byte spill
// (STACK:24) that regressed batch (USA B=256 +4.3%). Without the bound nvcc uses 106 regs / 0
// spill (1 block/SM), which is the better trade for this kernel. See docs note on the
// launch_bounds × fused interaction.
__global__ void factor_big_tf32_ptx(int lbegin, int lend, const int* __restrict__ plcols,
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

#ifdef CLS_FUSE_TF32_TRAIL_EXTEND
    // Fused trailing + extend-add: drain the TF32 mma accumulator straight into the parent front
    // via atomicAdd, eliminating the uncoalesced C write-back (to F) + read-back (by extend_add)
    // round-trip that docs note 30/53 identified as the real big-front memory bottleneck
    // (C-drain global STORE 9.98 sector/req, 2.5x uncoalesced). Mirrors the FP16 fused path
    // (CLS_FUSE_FP16_TRAIL_EXTEND). Only TC-eligible non-root fronts fuse; scalar (nc>32 / uc>256)
    // and root fronts keep the split write-then-extend path.
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
            extern __shared__ char smem_big_tf32_ptx[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_ptx);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            if (use_fused_trailing) {
                trailing_update_mma_tf32_ptx<true>(F, fsz, nc, uc, Ltf, Utf, t, nt, Fp, pfsz,
                                                   asm_local, abase);
            } else {
                trailing_update_mma_tf32_ptx<false>(F, fsz, nc, uc, Ltf, Utf, t, nt);
            }
        }
    });

    if (use_fused_trailing || par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    {
        float* Fp2 = front + front_off[par];
        const int pfsz2 = front_ptr[par + 1] - front_ptr[par];
        const int abase2 = asm_ptr[p];
        extend_add<float, float>(Fp2, pfsz2, F, fsz, nc, uc, asm_local, abase2, t, nt);
    }
#else
    auto tf32_big_trailing = [&] {
        if (nc > 32 || uc > 256) {
            trailing_update_scalar<float>(F, fsz, nc, uc, t, nt);
        } else {
            extern __shared__ char smem_big_tf32_ptx[];
            float* Ltf = reinterpret_cast<float*>(smem_big_tf32_ptx);
            float* Utf = Ltf + (long)ucp_max * kp_max;
            trailing_update_mma_tf32_ptx(F, fsz, nc, uc, Ltf, Utf, t, nt);
        }
    };
    factorize_front<float>(F, fsz, nc, uc, t, nt, sing, [&] { tf32_big_trailing(); });

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
#endif
}

// ===== from dispatch.cuh =====
// --- launch_factor_big_mb ---
// Multi-block big-front path for underfilled levels (B=1 deep levels): split the scalar big
// kernel into panel → multi-block trailing → extend so the FLOP-heavy trailing fans out across
// SMs instead of running one-block-per-front. `panel_blk` is the panel/extend block size,
// `level_max_uc` sizes the trailing tile grid (blockIdx.z).
template <typename T>
static inline void launch_factor_big_mb(int b, int e, int level_size, int B, int panel_blk,
                                        int level_max_uc, const MultifrontalPlan& plan,
                                        const int* d_plc, T* frontB, int* sing, int do_extend,
                                        cudaStream_t stream)
{
    const dim3 grid_pf(level_size, B);
    factor_big_panel<T><<<grid_pf, panel_blk, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, frontB,
        plan.front_total, sing);

    constexpr int TRAIL_BLK = 256;
    const int eb = TRAIL_BLK * 8;                              // C elements per trailing block
    const long max_uc2 = (long)level_max_uc * level_max_uc;
    const int ztiles = (int)((max_uc2 + eb - 1) / eb);
    if (ztiles > 0) {
        const dim3 grid_tr(level_size, B, ztiles);
        factor_big_trailing_mb<T><<<grid_tr, TRAIL_BLK, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols, frontB,
            plan.front_total, eb);
    }
    if (do_extend)
        factor_big_extend<T><<<grid_pf, panel_blk, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, frontB, plan.front_total);
}
