#include "tc/multifrontal_tc.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// TC-dedicated multifrontal factor/solve. Mirrors batched/multifrontal_batched.cu but with the
// precision mode hard-wired to TC32 (FP32 front + FP16 WMMA trailing). Reuses the batched-side
// device kernels (factor_small.cuh, factor_kernels.cuh, factor_tc.cuh, solve_kernels.cuh, ...)
// because the build is CUDA_SEPARABLE_COMPILATION OFF -- the kernels are templated and the batched
// TU already instantiates them; the TC TU links to those instantiations via the shared headers.
//
// Phase 1 (this revision):
//   - API + state isolation; dispatch logic identical to batched(TC32).
//   - Threshold knobs (SMALL_THRESH=32, fsz>=49 for TC trailing inside mf_factor_mid_tc32_b<true>)
//     unchanged. Baseline equivalence measurement.
// Phase 2 (planned):
//   - Lower TC trailing fsz threshold (49 -> 32) by introducing a TC-specific kernel that runs
//     WMMA on smaller fronts.
// Phase 3 (planned):
//   - Deep-tail dense LU absorption (L17-L29 collapse via cuBLAS sgetrfBatched or equivalent).
#include "batched/scatter.cuh"          // scatter_batched<FT,VT>
#include "batched/factor_kernels.cuh"   // mf_factor_extend_*_b, mf_invert_pivot_b<FT>
#include "batched/factor_small.cuh"     // mf_factor_small_warp_b<FT>
#include "tc/factor_tc.cuh"             // mf_factor_extend_tc32_b
#include "tc/factor_kernels_tc.cuh"     // Phase 2: mf_factor_mid_tc_lo_b<MIN_FSZ_FOR_TC>
#include "tc/spine_kernel.cuh"          // Phase 4: mf_spine_factor_b
#include "tc/trailing_tiled.cuh"        // Phase Σ.1: mf_factor_mid_tiled_b (tiled scalar trailing)
#include "tc/factor_split_cublas.cuh"   // Phase Σ.6: phaseA/B kernels for cuBLAS trailing
#include "batched/solve_kernels.cuh"    // gather_rhs_b, scatter_sol_b, mf_{fwd,bwd}_level_b
#include "batched/solve_small.cuh"      // mf_{fwd,bwd}_small_warp_b<FT>

namespace custom_linear_solver::tc {

using custom_linear_solver::plan::MultifrontalPlan;

// All TC kernels and helpers live in the `custom_linear_solver::batched` namespace (because that's
// where the .cuh files declare them). We pull them in unqualified inside this anonymous namespace
// so the dispatch reads cleanly while keeping the host API in `tc::`.
namespace {
using namespace custom_linear_solver::batched;
}

// Process-global cuBLAS handle — hoisted out of tc_setup to save the ~10 ms cublasCreate
// driver init on every analyze cycle. Created lazily on the first tc_setup call, destroyed
// on the next call after a process atexit-like teardown (kept alive otherwise).
namespace {
cublasHandle_t g_cublas_handle = nullptr;
cublasHandle_t get_or_create_cublas_handle()
{
    if (g_cublas_handle) return g_cublas_handle;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    g_cublas_handle = h;
    return h;
}
}  // namespace

TCState::~TCState()
{
    if (factor_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(factor_graph_exec));
    if (solve_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(solve_graph_exec));
    if (d_frontBf) cudaFree(d_frontBf);
    if (d_yBf) cudaFree(d_yBf);
    if (d_sing) cudaFree(d_sing);
    if (fork_event) cudaEventDestroy(static_cast<cudaEvent_t>(fork_event));
    for (int k = 0; k < num_subtree_streams; ++k) {
        if (join_events[k]) cudaEventDestroy(static_cast<cudaEvent_t>(join_events[k]));
        if (subtree_streams[k]) cudaStreamDestroy(static_cast<cudaStream_t>(subtree_streams[k]));
    }
    // cublas_handle is the process-global g_cublas_handle — DO NOT destroy it here; it is
    // reused across multiple TCState lifetimes. Leak on process exit is acceptable.
    cublas_handle = nullptr;
    if (d_Aptrs) cudaFree(d_Aptrs);
    if (d_Bptrs) cudaFree(d_Bptrs);
    if (d_Cptrs) cudaFree(d_Cptrs);
    if (d_pivotsB) cudaFree(d_pivotsB);
    if (stream && owns_stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

// ---- factor / solve issuance (mirror of batched, fixed to TC32 path) ----------------------

// Issue all panels in a given (lbegin, lend) range of plcols at level L onto `stream`.
// Range is a contiguous slice of plcols (already sorted by subtree id inside analyze).
static void issue_level_range(const MultifrontalPlan& plan, TCState& st, cudaStream_t stream,
                              int L, int b, int e)
{
    const int B = st.B;
    const int do_extend = 1;
    constexpr int SMALL_THRESH = 32;
    constexpr int SMALL_WARPS = 8;
    constexpr int MID_THRESH = 128;
    if (e <= b) return;
    const int level_size = e - b;

    int max_fsz = 0, max_uc = 1;
    for (int q = b; q < e; ++q) {
        const int pp = plan.h_plcols[q];
        const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
        const int uc = fsz - plan.h_ncols[pp];
        if (fsz > max_fsz) max_fsz = fsz;
        if (uc > max_uc && uc <= 256) max_uc = uc;
    }

    if (max_fsz <= SMALL_THRESH) {
        const long warps = (long)level_size * B;
        const int blk = SMALL_WARPS * 32;
        const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
        const int fsz2cap = max_fsz * max_fsz;
        const size_t shb = (size_t)SMALL_WARPS * fsz2cap * sizeof(float);
        mf_factor_small_warp_b<float><<<gx, blk, shb, stream>>>(
            b, level_size, B, fsz2cap, plan.d_plcols, plan.d_front_off,
            plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
            plan.d_asm_local, st.d_frontBf, plan.front_total, st.d_sing, do_extend);
        (void)L;
        return;
    }
    // Phase Σ.7 — env gates (read once). Shared between MID and BIG cuBLAS paths.
    static const bool s_use_cublas_outer = []() {
        const char* on = std::getenv("CLS_USE_CUBLAS");
        return on && on[0] && on[0] != '0';
    }();
    static const bool s_cublas_tf32 = []() {
        const char* on = std::getenv("CLS_CUBLAS_TF32");
        return on && on[0] && on[0] != '0';
    }();

    if (max_fsz <= MID_THRESH) {
        const int fsz_cap = max_fsz;
        const int ucp_max = ((max_uc + 15) / 16) * 16;
        // Phase Σ.7: MID cuBLAS path (CLS_USE_CUBLAS=1). Split the existing single-kernel
        // (stage + LU + U-solve + trailing + writeback + extend-add) into:
        //   (1) phase A: stage + LU + U-solve + writeback L/U (custom shared kernel)
        //   (2) cublasSgemmGroupedBatched: trailing C -= L*U for all K panels at this level
        //       in ONE call (K groups × B matrices each)
        //   (3) phase B: extend-add C from global into parent
        // CLS_CUBLAS_MIN_FSZ (default 64) gates cuBLAS at MID: levels whose max_fsz is below
        // this threshold stay on the custom shared-staged kernel because the per-front work
        // is too small for the cuBLAS-grouped overhead to amortize (case8387 L0 has 4000+
        // panels at fsz≈18 -- the K*B GEMMs are below the cuBLAS sweet spot).
        static const int s_cublas_min_fsz = []() {
            const char* mf = std::getenv("CLS_CUBLAS_MIN_FSZ");
            return mf ? std::atoi(mf) : 64;
        }();
        if (s_use_cublas_outer && st.cublas_handle && st.d_Aptrs && !plan.h_front_off.empty() &&
            max_fsz >= s_cublas_min_fsz) {
            const int mblk = 256;
            const size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float);
            dim3 grid(level_size, B);
            if (st.use_pivoting && st.d_pivotsB) {
                mf_factor_mid_phaseA_pp_b<<<grid, mblk, shb, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    st.d_frontBf, plan.front_total, st.d_sing, fsz_cap,
                    st.d_pivotsB, plan.d_pivot_offset, plan.total_pivots);
            } else {
                mf_factor_mid_phaseA_b<<<grid, mblk, shb, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    st.d_frontBf, plan.front_total, st.d_sing, fsz_cap);
            }
            // cublasSgemmGroupedBatched: one call, K groups, each group is B matrices
            // sharing the (M=uc, N=uc, K=nc, lda=ldb=ldc=fsz) of that panel.
            cublasHandle_t handle = static_cast<cublasHandle_t>(st.cublas_handle);
            cublasSetStream(handle, stream);
            cublasMath_t prev_mode;
            cublasGetMathMode(handle, &prev_mode);
            if (s_cublas_tf32) cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            const cublasOperation_t* transa_arr =
                reinterpret_cast<const cublasOperation_t*>(st.h_transa.data()) + b;
            cublasSgemmGroupedBatched(handle, transa_arr, transa_arr,
                st.h_m.data() + b, st.h_n.data() + b, st.h_k.data() + b,
                st.h_alpha.data() + b,
                const_cast<const float**>(st.d_Aptrs) + (long)b * B, st.h_lda.data() + b,
                const_cast<const float**>(st.d_Bptrs) + (long)b * B, st.h_lda.data() + b,
                st.h_beta.data() + b,
                st.d_Cptrs + (long)b * B, st.h_lda.data() + b,
                level_size,
                st.h_gsize.data() + b);
            if (s_cublas_tf32) cublasSetMathMode(handle, prev_mode);
            // Phase B: extend-add (reuse big-front phaseB kernel — works for any fsz since
            // it just reads C panel from global and atomically scatters into parent).
            mf_factor_big_phaseB_b<<<grid, mblk, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                plan.front_total, do_extend);
            return;
        }
        // Phase Σ.1: shared-staged trailing GEMM. Default ON after measurement showed
        // case8387 −15..−36 % factor wall, USA −3..−5 %, accuracy equal or better.
        // CLS_NO_TILED_TRAILING=1 to disable (regression check).
        // Phase Σ.2: register-blocked trailing GEMM (4×4 register tile per thread). Reduces
        // shared LDS:FMA ratio 4× vs scalar staged. Enabled when CLS_USE_REGBLOCK=1 (opt-in
        // until measurement confirms win); shares the same shared-memory layout as the staged
        // path.
        static const bool s_use_tiled = []() {
            const char* off = std::getenv("CLS_NO_TILED_TRAILING");
            if (off && off[0] && off[0] != '0') return false;
            return true;
        }();
        static const bool s_use_regblock = []() {
            const char* on = std::getenv("CLS_USE_REGBLOCK");
            return on && on[0] && on[0] != '0';
        }();
        // Phase Σ.15 — FP16-input register-blocked trailing (no WMMA padding waste). Opt-in
        // via CLS_USE_REGBLOCK_H16=1.
        static const bool s_use_regblock_h16 = []() {
            const char* on = std::getenv("CLS_USE_REGBLOCK_H16");
            return on && on[0] && on[0] != '0';
        }();
        if (s_use_tiled && max_fsz >= 48) {  // tiled win only for larger fronts
            // Shared = front + staged L (uc x nc) + staged U (nc x uc)
            // Compute actual max nc / max uc within this level.
            int level_max_nc = 1, level_max_uc = 1;
            for (int q = b; q < e; ++q) {
                const int pp = plan.h_plcols[q];
                const int fsz_p = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
                const int nc_p = plan.h_ncols[pp];
                if (nc_p > level_max_nc) level_max_nc = nc_p;
                if ((fsz_p - nc_p) > level_max_uc) level_max_uc = fsz_p - nc_p;
            }
            size_t shb_tiled = (size_t)fsz_cap * fsz_cap * sizeof(float)
                                + 2 * (size_t)level_max_nc * level_max_uc * sizeof(float);
            // Skip tiled path if total shared would exceed limit (fall back to default).
            if (shb_tiled > 96 * 1024) goto fallback_to_default;
            const int mblk_tiled = 256;
            dim3 grid_tiled(level_size, B);
            // FP16 register-blocked variant — shared L/U as __half (halves L/U footprint).
            if (s_use_regblock_h16) {
                size_t shb_h16 = (size_t)fsz_cap * fsz_cap * sizeof(float)
                                 + 2 * (size_t)level_max_nc * level_max_uc * sizeof(__half);
                if (shb_h16 > 96 * 1024) goto fallback_to_default;
                mf_factor_mid_regblock_h16_b<<<grid_tiled, mblk_tiled, shb_h16, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                return;
            }
            // regblock variant (FP32) when opted in; uses same dispatch + shared layout.
            if (s_use_regblock) {
                mf_factor_mid_regblock_b<<<grid_tiled, mblk_tiled, shb_tiled, stream>>>(
                    b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                    plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
                return;
            }
            mf_factor_mid_tiled_b<<<grid_tiled, mblk_tiled, shb_tiled, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
            return;
        }
    fallback_to_default:
        size_t shb = (size_t)fsz_cap * fsz_cap * sizeof(float);
        shb += (size_t)2 * ucp_max * 32 * sizeof(__half) + 4 * 256 * sizeof(float);
        const int mblk = max_fsz <= 48 ? 64 : (max_fsz <= 80 ? 128 : 256);
        dim3 grid(level_size, B);
        mf_factor_mid_tc_lo_b<24><<<grid, mblk, shb, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, st.d_sing, do_extend, ucp_max, fsz_cap);
        return;
    }
    // BIG separator (> MID_THRESH)
    // Phase Σ.7 — opt-in cuBLAS trailing using grouped batched. One call per LEVEL handles
    // all K big panels (K groups × B matrices). Per-call cuBLAS overhead amortized across
    // K*B GEMMs instead of K separate strided calls.
    const int bigT = 1024;
    dim3 bgrid(level_size, B);
    if (s_use_cublas_outer && st.cublas_handle && st.d_Aptrs && !plan.h_front_off.empty()) {
        if (st.use_pivoting && st.d_pivotsB) {
            mf_factor_big_phaseA_pp_b<<<bgrid, bigT, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                st.d_frontBf, plan.front_total, st.d_sing,
                st.d_pivotsB, plan.d_pivot_offset, plan.total_pivots);
        } else {
            mf_factor_big_phaseA_b<<<bgrid, bigT, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                st.d_frontBf, plan.front_total, st.d_sing);
        }
        cublasHandle_t handle = static_cast<cublasHandle_t>(st.cublas_handle);
        cublasSetStream(handle, stream);
        cublasMath_t prev_mode;
        cublasGetMathMode(handle, &prev_mode);
        if (s_cublas_tf32) cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        const cublasOperation_t* transa_arr =
            reinterpret_cast<const cublasOperation_t*>(st.h_transa.data()) + b;
        cublasSgemmGroupedBatched(handle, transa_arr, transa_arr,
            st.h_m.data() + b, st.h_n.data() + b, st.h_k.data() + b,
            st.h_alpha.data() + b,
            const_cast<const float**>(st.d_Aptrs) + (long)b * B, st.h_lda.data() + b,
            const_cast<const float**>(st.d_Bptrs) + (long)b * B, st.h_lda.data() + b,
            st.h_beta.data() + b,
            st.d_Cptrs + (long)b * B, st.h_lda.data() + b,
            level_size,
            st.h_gsize.data() + b);
        if (s_cublas_tf32) cublasSetMathMode(handle, prev_mode);
        mf_factor_big_phaseB_b<<<bgrid, bigT, 0, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, do_extend);
        return;
    }

    const int ucp_max = ((max_uc + 15) / 16) * 16;
    const size_t shbytes =
        (size_t)2 * ucp_max * 32 * sizeof(__half) + (bigT / 32) * 256 * sizeof(float);
    mf_factor_extend_tc32_b<<<bgrid, bigT, shbytes, stream>>>(
        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
        plan.front_total, st.d_sing, do_extend, ucp_max);
}

static void issue_factor_levels(const MultifrontalPlan& plan, TCState& st, cudaStream_t stream)
{
    const int B = st.B;

    // Phase Σ.8 — init pivot storage to per-panel identity at the start of every factor.
    // Non-pivoting factor paths leave the pivots untouched; pivoting paths overwrite this
    // identity with their swap choices. Solve then applies whatever is in the array, so
    // mixing pivoting/non-pivoting kernels in one level dispatch stays correct.
    if (st.use_pivoting && st.d_pivotsB && plan.d_pivot_offset && plan.num_panels > 0) {
        dim3 ig(plan.num_panels, B);
        mf_init_pivots_identity_b<<<ig, 64, 0, stream>>>(
            st.d_pivotsB, plan.d_pivot_offset, plan.d_ncols, plan.num_panels,
            plan.total_pivots, B);
    }

    // Phase 4 toggle (still opt-in after measurement, see plan doc Phase 4 outcome).
    const char* spine_env = std::getenv("CLS_USE_SPINE");
    const bool use_spine = spine_env && spine_env[0] && spine_env[0] != '0' &&
                           !plan.h_spine_panels.empty() && plan.d_spine_panels != nullptr;
    const int spine_lo = use_spine ? plan.spine_start_level : plan.num_plevels;

    // Phase 3 / Σ.11 — multi-stream subtree dispatch. K subtrees fork onto K streams at the
    // bottom of the tree; spine work joins back onto the main stream. **Default ON** after
    // Σ.11 fix to subtree partition (was: stranded panels with parent-in-spine direct caused
    // garbage; fix: subtree root = first ancestor with spine parent, cap K=8 with spillover
    // bin). Disable with CLS_NO_MULTISTREAM=1.
    const char* multi_no = std::getenv("CLS_NO_MULTISTREAM");
    const bool disable_multi = multi_no && multi_no[0] && multi_no[0] != '0';
    const char* multi_env = std::getenv("CLS_USE_MULTISTREAM");
    const bool legacy_enable = multi_env && multi_env[0] && multi_env[0] != '0';
    const bool enable_multi = !disable_multi || legacy_enable;
    const bool use_multistream = enable_multi && st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();

    if (use_multistream) {
        // Fork: main stream signals fork_event; each subtree stream waits on it.
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);
        }
        // Each subtree: walk its levels (only the panels that belong to it).
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < plan.num_plevels; ++L) {
                if (L >= spine_lo) break;       // spine handled separately on main stream
                if (cnt[L] == 0) continue;
                issue_level_range(plan, st, ks, L, off[L], off[L] + cnt[L]);
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }
        // Join: main stream waits all K subtree events.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        }
        // Bug fix — when use_spine=false (default), the spine panels are NOT visited inside the
        // K subtree streams (they have subtree_of_panel=-1). Process those levels on the main
        // stream after the join, on a per-level basis. (When use_spine=true, the spine fused
        // kernel below handles them.)
        if (!use_spine && plan.spine_start_level >= 0) {
            for (int L = plan.spine_start_level; L < plan.num_plevels; ++L) {
                const int b = plan.plptr[L], e = plan.plptr[L + 1];
                issue_level_range(plan, st, stream, L, b, e);
            }
        }
    } else {
        // Single-stream fallback (original): iterate levels on main stream.
        for (int L = 0; L < plan.num_plevels; ++L) {
            if (L >= spine_lo) continue;
            const int b = plan.plptr[L], e = plan.plptr[L + 1];
            issue_level_range(plan, st, stream, L, b, e);
        }
    }
    // Phase 4: fused spine kernel. Runs after all non-spine levels so each spine front already
    // has the CB from its child(ren) accumulated. One block per batch; the block walks the
    // spine panels in order. Default OFF (see Phase 4 measurement above).
    const int do_extend = 1;
    if (use_spine) {
        const int n_spine = (int)plan.h_spine_panels.size();
        // Max fsz across the spine -- determines reasonable block width.
        int max_fsz = 0;
        for (int sp : plan.h_spine_panels) {
            const int fsz = plan.h_front_ptr[sp + 1] - plan.h_front_ptr[sp];
            if (fsz > max_fsz) max_fsz = fsz;
        }
        const int spine_T = max_fsz <= 64 ? 128 : (max_fsz <= 128 ? 256 : 384);
        dim3 grid(1, B);
        custom_linear_solver::tc::mf_spine_factor_b<<<grid, spine_T, 0, stream>>>(
            n_spine, plan.d_spine_panels, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_frontBf,
            plan.front_total, st.d_sing, do_extend);
    }

    if (st.selinv) {
        const dim3 ig(plan.num_panels, B);
        mf_invert_pivot_b<float><<<ig, 32, 0, stream>>>(
            plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols, st.d_frontBf,
            plan.front_total);
    }
}

static void issue_solve_levels(const MultifrontalPlan& plan, TCState& st, cudaStream_t stream)
{
    const int B = st.B;
    const int sel = st.selinv ? 1 : 0;
    constexpr int SMALL_THRESH = 32, SMALL_WARPS = 8;
    constexpr int selt = (int)sizeof(float);
    auto level_max_fsz = [&](int b, int e) {
        int m = 0;
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int fsz = plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp];
            if (fsz > m) m = fsz;
        }
        return m;
    };
    for (int L = 0; L < plan.num_plevels; ++L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const size_t shb = (size_t)SMALL_WARPS * 64 * selt;
            if (st.use_pivoting && st.d_pivotsB) {
                mf_fwd_small_warp_pp_b<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n, st.d_pivotsB, plan.d_pivot_offset, plan.total_pivots);
            } else {
                mf_fwd_small_warp_b<float><<<gx, blk, shb, stream>>>(
                    b, e - b, B, 64, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                    plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                    plan.n, sel);
            }
            continue;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 fg(e - b, B);
        if (st.use_pivoting && st.d_pivotsB) {
            mf_fwd_level_pp_b<float><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n,
                st.d_pivotsB, plan.d_pivot_offset, plan.total_pivots);
        } else {
            mf_fwd_level_b<float><<<fg, tsb, 0, stream>>>(
                b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
        }
    }
    for (int L = plan.num_plevels - 1; L >= 0; --L) {
        const int b = plan.plptr[L], e = plan.plptr[L + 1];
        if (e <= b) continue;
        int max_cb = 1;
        for (int q = b; q < e; ++q) {
            const int pp = plan.h_plcols[q];
            const int cbq = (plan.h_front_ptr[pp + 1] - plan.h_front_ptr[pp]) - plan.h_ncols[pp];
            if (cbq > max_cb) max_cb = cbq;
        }
        if (level_max_fsz(b, e) <= SMALL_THRESH) {
            const long warps = (long)(e - b) * B;
            const int blk = SMALL_WARPS * 32;
            const int gx = (int)((warps + SMALL_WARPS - 1) / SMALL_WARPS);
            const int slab = 64 + max_cb;
            const size_t shb = (size_t)SMALL_WARPS * slab * selt;
            mf_bwd_small_warp_b<float><<<gx, blk, shb, stream>>>(
                b, e - b, B, slab, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
                plan.d_ncols, plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total,
                plan.n, sel);
            continue;
        }
        const int mf = level_max_fsz(b, e);
        const int tsb = mf <= 64 ? 64 : (mf <= 128 ? 128 : 256);
        const dim3 bg(e - b, B);
        mf_bwd_level_b<float><<<bg, tsb, (size_t)max_cb * sizeof(float), stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_front_rows, st.d_frontBf, st.d_yBf, plan.front_total, plan.n, sel);
    }
}

// ---- setup ------------------------------------------------------------------------------------

bool tc_warmup()
{
    return get_or_create_cublas_handle() != nullptr;
}

bool tc_setup(const MultifrontalPlan& plan, int B, TCState& st)
{
    if (plan.num_panels == 0 || B <= 0) return false;
    const bool dbg = std::getenv("CLS_TC_SETUP_DBG") != nullptr;
    auto stamp = [dbg](const char* name, double& t) {
        if (!dbg) return;
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
        fprintf(stderr, "[tc_setup] %s : +%.2f ms\n", name, ms - t);
        t = ms;
    };
    double t = 0;
    if (dbg) { cudaDeviceSynchronize(); t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    st.B = B;
    st.front_total = plan.front_total;
    st.n = plan.n;
    // selinv = partitioned-inverse (pivot block inverted in factor) trades factor time for
    // solve time. For 1-factor-1-solve workloads (Newton-Raphson power flow) selinv is a
    // net LOSS: factor +40 μs to save solve +2 μs. Default OFF; CLS_USE_SELINV=1 to re-enable.
    st.selinv = (std::getenv("CLS_USE_SELINV") != nullptr) &&
                (std::getenv("MF_NO_SELINV") == nullptr);
    const long fe = (long)B * plan.front_total;
    if (cudaMalloc(&st.d_frontBf, fe * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_yBf, (long)B * plan.n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&st.d_sing, sizeof(int)) != cudaSuccess) return false;
    stamp("cudaMalloc(state arenas)", t);

    // Phase Σ.8 — pivot storage. CLS_USE_PIVOTING=1 + pivot_offset available -> allocate
    // B * total_pivots ints. Zeros are valid identity pivots (each step's swap target = k).
    {
        const char* on = std::getenv("CLS_USE_PIVOTING");
        st.use_pivoting = on && on[0] && on[0] != '0' && plan.total_pivots > 0 &&
                          plan.d_pivot_offset != nullptr;
        if (st.use_pivoting) {
            const long pe = (long)B * plan.total_pivots;
            if (cudaMalloc(&st.d_pivotsB, pe * sizeof(int)) != cudaSuccess) return false;
            cudaMemset(st.d_pivotsB, 0, pe * sizeof(int));
        }
    }

    // Phase Σ.7 — cuBLAS handle + grouped-batched arrays. Created unconditionally so the
    // factor graph captures a fully-initialized state; runtime cost is one cublasCreate +
    // 4 cudaMallocs + 3 H2D copies of pointer arrays. CLS_USE_CUBLAS=1 still gates whether
    // the cuBLAS path is *used* at dispatch time.
    {
        cublasHandle_t h = get_or_create_cublas_handle();
        if (!h) return false;
        st.cublas_handle = h;
        stamp("cublasCreate (cached)", t);

        const int P = plan.num_panels;
        if (P > 0 && !plan.h_front_off.empty() && !plan.h_front_ptr.empty() &&
            !plan.h_ncols.empty() && !plan.h_plcols.empty()) {
            st.h_m.assign(P, 0);
            st.h_n.assign(P, 0);
            st.h_k.assign(P, 0);
            st.h_lda.assign(P, 0);
            st.h_gsize.assign(P, B);
            st.h_transa.assign(P, static_cast<int>(CUBLAS_OP_N));
            st.h_alpha.assign(P, -1.0f);
            st.h_beta.assign(P, 1.0f);

            // Build host scalar arrays (P entries each) AND per-panel within-front offsets
            // (3 * P ints). The big P*B device pointer arrays are then materialized by a
            // single kernel from these offsets — saves the 3*P*B*8-byte H2D copy that
            // dominated tc_setup for large matrices (USA: 55 ms -> ~1 ms).
            std::vector<int> h_within_U(P), h_within_L(P), h_within_C(P), h_panel_front_off(P);
            for (int i = 0; i < P; ++i) {
                const int p = plan.h_plcols[i];
                const long foff = plan.h_front_off[p];
                const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
                const int nc = plan.h_ncols[p];
                const int uc = fsz - nc;
                // CRITICAL: for fsz<=48 the phaseA kernel calls `lu_small_front` which fuses
                // LU + U-solve + trailing in one pass. The cuBLAS trailing afterwards would
                // double-subtract L*U on those panels. Mark m=n=k=0 here so the grouped
                // batched call no-ops them (cuBLAS treats zero-dim groups as no-ops).
                if (fsz <= 48) {
                    st.h_m[i] = 0; st.h_n[i] = 0; st.h_k[i] = 0;
                } else {
                    st.h_m[i] = uc; st.h_n[i] = uc; st.h_k[i] = nc;
                }
                st.h_lda[i] = fsz;
                h_panel_front_off[i] = (int)foff;
                h_within_U[i] = nc;
                h_within_L[i] = nc * fsz;
                h_within_C[i] = nc * fsz + nc;
            }
            if (cudaMalloc(&st.d_Aptrs, sizeof(float*) * (size_t)P * B) != cudaSuccess) return false;
            if (cudaMalloc(&st.d_Bptrs, sizeof(float*) * (size_t)P * B) != cudaSuccess) return false;
            if (cudaMalloc(&st.d_Cptrs, sizeof(float*) * (size_t)P * B) != cudaSuccess) return false;
            // Upload only the per-panel offsets (4 small int arrays of length P).
            int *d_foff = nullptr, *d_uoff = nullptr, *d_loff = nullptr, *d_coff = nullptr;
            cudaMalloc(&d_foff, sizeof(int) * P);
            cudaMalloc(&d_uoff, sizeof(int) * P);
            cudaMalloc(&d_loff, sizeof(int) * P);
            cudaMalloc(&d_coff, sizeof(int) * P);
            cudaMemcpy(d_foff, h_panel_front_off.data(), sizeof(int) * P, cudaMemcpyHostToDevice);
            cudaMemcpy(d_uoff, h_within_U.data(), sizeof(int) * P, cudaMemcpyHostToDevice);
            cudaMemcpy(d_loff, h_within_L.data(), sizeof(int) * P, cudaMemcpyHostToDevice);
            cudaMemcpy(d_coff, h_within_C.data(), sizeof(int) * P, cudaMemcpyHostToDevice);
            const int bs_blk = (B + 63) / 64;
            dim3 grid(P, bs_blk);
            mf_build_cublas_ptrs_b<<<grid, std::min(B, 64), 0, 0>>>(
                P, B, plan.front_total, d_foff, d_uoff, d_loff, d_coff,
                st.d_frontBf, st.d_Aptrs, st.d_Bptrs, st.d_Cptrs);
            cudaDeviceSynchronize();
            cudaFree(d_foff); cudaFree(d_uoff); cudaFree(d_loff); cudaFree(d_coff);
            stamp("cuBLAS arrays build (device kernel)", t);
        }
    }

    // Opt in to the sm_86 99 KB dynamic-shared limit for the kernels that exceed 48 KB.
    cudaFuncSetAttribute(mf_factor_small_warp_b<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_tc_lo_b<24>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_tiled_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_regblock_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_extend_tc32_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_phaseA_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_phaseA_pp_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    cudaFuncSetAttribute(mf_factor_mid_regblock_h16_b,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    stamp("cudaFuncSetAttribute x7", t);

#ifdef CLS_INTERNAL_GRAPH
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    st.stream = stream;
    st.owns_stream = true;

    // Phase 3 / Σ.11 — allocate K subtree streams. **Default ON** after the Σ.11 fix to
    // subtree partition. Disable via CLS_NO_MULTISTREAM=1 (also gates the dispatch).
    const char* multi_no_su = std::getenv("CLS_NO_MULTISTREAM");
    const bool disable_multi_su = multi_no_su && multi_no_su[0] && multi_no_su[0] != '0';
    const bool enable_multi_su = !disable_multi_su;
    if (enable_multi_su && plan.num_subtrees > 1 && plan.num_subtrees <= 8) {
        st.num_subtree_streams = plan.num_subtrees;
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks;
            cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking);
            st.subtree_streams[k] = ks;
            cudaEvent_t je;
            cudaEventCreateWithFlags(&je, cudaEventDisableTiming);
            st.join_events[k] = je;
        }
        cudaEvent_t fe;
        cudaEventCreateWithFlags(&fe, cudaEventDisableTiming);
        st.fork_event = fe;
    }

    // Drain any sticky error left by the prior cudaFuncSetAttribute calls — for kernels not
    // instantiated by the current dispatch path, the SetAttribute leaves a cudaErrorInvalidValue
    // that would otherwise poison the final cudaGetLastError() of this function.
    (void)cudaGetLastError();
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_factor_levels(plan, st, stream);
    cudaGraph_t g;
    cudaStreamEndCapture(stream, &g);
    stamp("factor graph capture", t);
    cudaGraphExec_t ge;
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    st.factor_graph_exec = ge;
    stamp("factor graph instantiate", t);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    issue_solve_levels(plan, st, stream);
    cudaGraph_t sg;
    cudaStreamEndCapture(stream, &sg);
    stamp("solve graph capture", t);
    cudaGraphExec_t sge;
    cudaGraphInstantiate(&sge, sg, nullptr, nullptr, 0);
    cudaGraphDestroy(sg);
    st.solve_graph_exec = sge;
    stamp("solve graph instantiate", t);
    return cudaGetLastError() == cudaSuccess;
#else
    return true;
#endif
}

void tc_set_stream(TCState& st, void* stream)
{
    st.stream = stream;
    st.owns_stream = false;
}

// ---- factorize / solve ------------------------------------------------------------------------

bool tc_factorize(const MultifrontalPlan& plan, TCState& st, const float* d_valuesB,
                  const int* d_o2c, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const long fe = (long)st.B * plan.front_total;
    const int T = 256;
    const dim3 sgrid((plan.nnz_a + T - 1) / T, st.B);
    auto issue_scatter = [&]() {
        cudaMemsetAsync(st.d_sing, 0, sizeof(int), stream);
        cudaMemsetAsync(st.d_frontBf, 0, fe * sizeof(float), stream);
        scatter_batched<float, float><<<sgrid, T, 0, stream>>>(plan.nnz_a, plan.front_total, d_o2c,
                                                               plan.d_a_pos, d_valuesB, st.d_frontBf);
    };
#ifdef CLS_INTERNAL_GRAPH
    // Diagnostic: CLS_BYPASS_GRAPH=1 skips cudaGraphLaunch and dispatches kernels directly
    // (lets us isolate graph-capture bugs from algorithmic bugs).
    static const bool s_bypass_graph = []() {
        const char* on = std::getenv("CLS_BYPASS_GRAPH");
        return on && on[0] && on[0] != '0';
    }();
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    issue_scatter();
    if (s_bypass_graph) {
        issue_factor_levels(plan, st, stream);
    } else {
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.factor_graph_exec), stream);
    }
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
#else
    (void)kernel_ms;
    issue_scatter();
    issue_factor_levels(plan, st, stream);
    return true;
#endif
}

bool tc_solve(const MultifrontalPlan& plan, TCState& st, const float* d_rhsB, float* d_solB,
              const int* d_perm, double* kernel_ms)
{
    cudaStream_t stream = static_cast<cudaStream_t>(st.stream);
    const int n = plan.n;
    const int T = 256;
    const dim3 vg((n + T - 1) / T, st.B);
    auto issue_gather = [&]() {
        gather_rhs_b<float, float><<<vg, T, 0, stream>>>(n, d_rhsB, d_perm, st.d_yBf);
    };
    auto issue_scatter_sol = [&]() {
        scatter_sol_b<float, float><<<vg, T, 0, stream>>>(n, st.d_yBf, d_perm, d_solB);
    };
#ifdef CLS_INTERNAL_GRAPH
    cudaEvent_t k0, k1;
    cudaEventCreate(&k0);
    cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);
    issue_gather();
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(st.solve_graph_exec), stream);
    issue_scatter_sol();
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    if (kernel_ms) { float ms = 0; cudaEventElapsedTime(&ms, k0, k1); *kernel_ms = ms; }
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    return cudaGetLastError() == cudaSuccess;
#else
    (void)kernel_ms;
    issue_gather();
    issue_solve_levels(plan, st, stream);
    issue_scatter_sol();
    return true;
#endif
}

}  // namespace custom_linear_solver::tc
