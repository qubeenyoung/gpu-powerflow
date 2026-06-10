// [ARCHIVED 2026-06-10] Dormant global-resident blocked TF32 big-high kernel. Prototype
// measured negative (repeated global L/U traffic; see TC investigation §2). Not compiled.

// Big-front right-looking blocked TF32 path. Unlike factor_big_tf32_ptx, this uses Tensor Cores
// for the block update after each pivot block, so TC covers remaining panel columns as well as C.
__global__ void __launch_bounds__(512, 2)
                                factor_big_tf32_blocked_ptx(int lbegin, int lend,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols,
                                const int* __restrict__ panel_parent,
                                const int* __restrict__ asm_ptr,
                                const int* __restrict__ asm_local, float* frontB,
                                long front_total, int* sing, int do_extend)
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
    const bool use_tc = (fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256);

    if (use_tc) {
        factorize_front_blocked_tf32(F, fsz, nc, t, nt, sing);
    } else {
        factorize_front<float>(F, fsz, nc, uc, t, nt, sing,
            [&] { trailing_update_scalar<float>(F, fsz, nc, uc, t, nt); });
    }

    const int par = panel_parent[p];
    if (par < 0 || !do_extend || !extend_add_allowed_for_uc(uc)) return;
    __syncthreads();
    float* Fp = front + front_off[par];
    const int pfsz = front_ptr[par + 1] - front_ptr[par];
    const int abase = asm_ptr[p];
    extend_add<float, float>(Fp, pfsz, F, fsz, nc, uc, asm_local, abase, t, nt);
}
