// [ARCHIVED 2026-06-10] Legacy mid kernels superseded by factor_mid_blocked<T,UseTC>.
// factor_mid<T> = staged-scalar mid; factor_mid_tf32_ptx = TF32 direct/blocked mid (B>=64).
// Reference only — not compiled.

// FP64 / FP32 staged-scalar trailing.
// Shared layout:  Fs[fsz_cap²] | sh_L[level_max_uc · level_max_nc] | sh_U[level_max_nc · level_max_uc].
template <typename T>
__global__ void factor_mid(int lbegin, int lend, const int* __restrict__ plcols,
                           const int* __restrict__ front_off,
                           const int* __restrict__ front_ptr,
                           const int* __restrict__ ncols,
                           const int* __restrict__ panel_parent,
                           const int* __restrict__ asm_ptr,
                           const int* __restrict__ asm_local, T* frontB,
                           long front_total, int* sing, int do_extend, int fsz_cap,
                           int level_max_nc, int level_max_uc)
{
    const int idx = lbegin + blockIdx.x;
    if (idx >= lend) return;

    // (batch, panel) → front pointer.
    T* front = frontB + (long)blockIdx.y * front_total;
    const int p = plcols[idx];
    const int fsz = front_ptr[p + 1] - front_ptr[p];
    const int nc = ncols[p];
    T* F = front + front_off[p];
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    const int fsz2 = fsz * fsz;

    // Shared layout — Fs first (largest, alignment), then the L / U staging panels.
    extern __shared__ char smem_mid[];
    T* Fs   = reinterpret_cast<T*>(smem_mid);
    T* sh_L = Fs + (long)fsz_cap * fsz_cap;
    T* sh_U = sh_L + (long)level_max_uc * level_max_nc;

    const int par = panel_parent[p];
#ifdef CLS_FUSE_SCALAR_TRAIL_EXTEND
    // fsz <= 48 fronts use lu_small_front inside factorize_front, which fuses Phase 1+3 and never
    // invokes the trailing() functor — so the fused drain can't run there. Restrict to fsz > 48.
    const bool use_fused = (fsz > 48 && par >= 0 && do_extend && extend_add_allowed_for_uc(uc));
#else
    constexpr bool use_fused = false;
#endif
    T* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
    const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
    const int abase = asm_ptr[p];

    stage_in_async<T>(Fs, F, fsz2, t, nt);
    __syncthreads();
    factorize_front<T>(Fs, fsz, nc, uc, t, nt, sing, [&] {
        if (use_fused)
            trailing_update_staged<T, true>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U, Fp, pfsz, asm_local, abase);
        else
            trailing_update_staged<T>(Fs, fsz, nc, uc, t, nt, sh_L, sh_U);
    });
    __syncthreads();
    writeback_factored<T, T>(F, Fs, fsz, nc, uc, t, nt);

    if (use_fused || par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    extend_add<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
}

// TF32 PTX mid trailing on a shared-resident FP32 front. This mirrors the FP16 mid TC path
// but keeps L/U staging in float and lets mma.sync's .tf32 ABI truncate multiplicands.
__global__ void factor_mid_tf32_ptx(int lbegin, int lend, const int* __restrict__ plcols,
                                    const int* __restrict__ front_off,
                                    const int* __restrict__ front_ptr,
                                    const int* __restrict__ ncols,
                                    const int* __restrict__ panel_parent,
                                    const int* __restrict__ asm_ptr,
                                    const int* __restrict__ asm_local, float* frontB,
                                    long front_total, int* sing, int do_extend, int fsz_cap,
                                    int ucp_max, int kp_max, int direct_shared_mode)
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
    const bool use_tc =
#if   defined(CLS_MID_TF32_LOW_TC)
        ((fsz > 48 && uc >= 32) ||
         (direct_shared_mode && fsz > 16 && uc >= 16)) &&
        nc >= 4 && nc <= 32 && uc <= 256;
#else
        (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);
#endif

	extern __shared__ char smem_mid_tf32_ptx[];
	float* Fs  = reinterpret_cast<float*>(smem_mid_tf32_ptx);
	(void)ucp_max;
	(void)kp_max;

	stage_in_async<float>(Fs, F, fsz2, t, nt);
	__syncthreads();

    const bool direct_shared_tc =
#ifdef CLS_MID_TF32_DIRECT_SHARED
        true;
#else
        (direct_shared_mode != 0);
	#endif
	    bool extend_fused = false;
	    (void)direct_shared_tc;
	    auto direct_tf32_trailing = [&] {
	        trailing_update_mma_tf32_direct_shared(Fs, fsz, nc, uc, t, nt);
	    };
    auto direct_tf32_factorize = [&] {
        lu_panel_factor<float>(Fs, fsz, nc, t, nt, sing);
#ifdef CLS_TF32_COLUMN_USOLVE
        u_panel_solve_column_owned<float>(Fs, fsz, nc, uc, t, nt);
#else
        u_panel_solve<float>(Fs, fsz, nc, uc, t, nt);
#endif
        direct_tf32_trailing();
    };

    if (use_tc) {
#ifdef CLS_MID_TF32_DIRECT_SHARED
        direct_tf32_factorize();
#else
        if (direct_shared_mode) {
            direct_tf32_factorize();
        } else {
            factorize_front_blocked_tf32(Fs, fsz, nc, t, nt, sing);
        }
#endif
    } else {
        factorize_front<float>(Fs, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(Fs, fsz, nc, uc, t, nt); });
    }

    writeback_factored<float, float>(F, Fs, fsz, nc, uc, t, nt);

    if (par < 0 || !do_extend || extend_fused || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* parent_front = front + front_off[par];
    const int parent_fsz = front_ptr[par + 1] - front_ptr[par];
    const int asm_base = asm_ptr[p];
    extend_add<float, float>(parent_front, parent_fsz, Fs, fsz, nc, uc, asm_local, asm_base,
                             t, nt);
}
