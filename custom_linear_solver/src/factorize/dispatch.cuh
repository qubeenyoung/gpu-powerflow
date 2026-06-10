#pragma once

// FACTORIZE — host-side level dispatch.
//
// Internal — included only by numeric_engine.cu (single TU; CUDA_SEPARABLE_COMPILATION OFF).
//
// Two layers:
//
//   issue_factor_levels        — outer dispatcher. Walks the panel-etree level-by-level
//                                (`plan.num_plevels`). When the plan exposes independent
//                                subtrees, each subtree's pre-spine levels are enqueued on
//                                its own CUDA stream and joined into the main stream before
//                                the spine — a fork/wait/join pattern that lets independent
//                                subtree work overlap on the GPU. With one subtree (or with
//                                multistream disabled) it reduces to a single sequential
//                                level walk on the main stream.
//
//   issue_factor_level_range   — per-range dispatcher. For one (sub)range of panels at one
//                                level it scans the range (scan_front_range), picks a tier
//                                with classify_front_tier, and calls dispatch_factor_small /
//                                _mid / _big. The mid dispatch falls through to the big tier
//                                when its shared layout overflows kMidSharedMemoryBudgetBytes.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef CLS_CUBLAS_TF32_TRAILING
#include <cublas_v2.h>
#endif

#include "plan/front_range_caps.hpp"
#include "numeric_engine.hpp"
#include "factorize/kernels.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

#ifdef CLS_DISABLE_EXTEND_ADD
inline constexpr int kFactorDoExtend = 0;
#else
inline constexpr int kFactorDoExtend = 1;
#endif

static int factor_num_sms()
{
    static const int v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int sm = 1;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        return sm;
    }();
    return v;
}

static int factor_max_block()
{
    static const int v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int mt = 1024;
        cudaDeviceGetAttribute(&mt, cudaDevAttrMaxThreadsPerBlock, dev);
        return mt;
    }();
    return v;
}

static long factor_warp_fill();  // defined below (used by the small-tier sub-group gate)

// Launch helper: instantiates factor_small for a concrete (front type, sub-group size). Keeps
// the runtime sub_group_size switch in issue_factor_level_range compact.
template <typename FrontType, int sub_group_size>
static inline void launch_factor_small(int num_blocks, int threads_per_block, size_t shared_bytes, cudaStream_t stream,
                                       int b, int level_size, int B, int front_area,
                                       const MultifrontalPlan& plan, const int* d_plc,
                                       FrontType* frontB, int* sing, int do_extend)
{
    factor_small<FrontType, sub_group_size><<<num_blocks, threads_per_block, shared_bytes, stream>>>(
        b, level_size, B, front_area, d_plc, plan.d_front_off, plan.d_front_ptr,
        plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, frontB,
        plan.front_total, sing, do_extend);
}

// Pick the small-tier sub-group size: sub_group_size ∈ {8,16,32}; packing (sub_group_size<32) only applies once the
// packed grid still fills the GPU (mirrors solve/dispatch.cuh solve_small_sg).
static int factor_small_sg(int max_fsz, long warps_unpacked)
{
#ifdef CLS_SMALL_SG32_ONLY
    (void)max_fsz; (void)warps_unpacked;
    return 32;
#else
    int sg = (max_fsz <= 8) ? 8 : (max_fsz <= 16 ? 16 : 32);
    if (warps_unpacked / (32 / sg) < factor_warp_fill()) sg = 32;
    return sg;
#endif
}

// Launch the small kernel with (front type fixed, sub_group_size resolved at the call site).
template <typename FrontType>
static inline void launch_factor_small_t(int sub_group_size, int num_blocks, int threads_per_block, size_t shared_bytes, cudaStream_t stream,
                                         int b, int level_size, int B, int front_area,
                                         const MultifrontalPlan& plan, const int* d_plc,
                                         FrontType* frontB, int* sing, int do_extend)
{
    if (sub_group_size == 8)       launch_factor_small<FrontType, 8>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend);
    else if (sub_group_size == 16) launch_factor_small<FrontType, 16>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend);
    else               launch_factor_small<FrontType, 32>(num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area, plan, d_plc, frontB, sing, do_extend);
}

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

#ifdef CLS_CUBLAS_TF32_TRAILING
static inline bool cublas_uses_tier_order(const MultifrontalPlan& plan, const int* h_plc)
{
    return !plan.h_plcols_tier.empty() && h_plc == plan.h_plcols_tier.data();
}

static inline bool cublas_tf32_trailing_ready(const State& st)
{
    return st.batch_count >= 64 && st.precision == Precision::TF32 && st.cublas_handle &&
           st.d_cublas_Aptrs && st.d_cublas_Bptrs && st.d_cublas_Cptrs;
}

static bool cublas_range_has_only_deferred_trailing(const MultifrontalPlan& plan,
                                                    const int* h_plc, int b, int e)
{
    for (int q = b; q < e; ++q) {
        const int p = h_plc[q];
        const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
        if (fsz <= 48) return false;
    }
    return true;
}

static bool launch_cublas_tf32_trailing(const MultifrontalPlan& plan, State& st,
                                        cudaStream_t stream, int b, int e,
                                        const int* d_plc, const int* h_plc)
{
    const int B = st.batch_count;
    const int level_size = e - b;
    if (level_size <= 0 || !cublas_tf32_trailing_ready(st)) {
        return false;
    }
    (void)d_plc;

    const bool tier_order = cublas_uses_tier_order(plan, h_plc);
    const std::vector<int>& hm = tier_order ? st.cublas_m_tier : st.cublas_m;
    const std::vector<int>& hn = tier_order ? st.cublas_n_tier : st.cublas_n;
    const std::vector<int>& hk = tier_order ? st.cublas_k_tier : st.cublas_k;
    const std::vector<int>& hlda = tier_order ? st.cublas_lda_tier : st.cublas_lda;
    if (hm.empty() || hn.empty() || hk.empty() || hlda.empty()) return false;
    float** Aptrs = tier_order ? st.d_cublas_Aptrs_tier : st.d_cublas_Aptrs;
    float** Bptrs = tier_order ? st.d_cublas_Bptrs_tier : st.d_cublas_Bptrs;
    float** Cptrs = tier_order ? st.d_cublas_Cptrs_tier : st.d_cublas_Cptrs;
    if (!Aptrs || !Bptrs || !Cptrs) return false;

    cublasHandle_t handle = static_cast<cublasHandle_t>(st.cublas_handle);
    cublasSetStream(handle, stream);
    const cublasStatus_t status = cublasSgemmGroupedBatched(
        handle,
        reinterpret_cast<const cublasOperation_t*>(st.cublas_trans.data()) + b,
        reinterpret_cast<const cublasOperation_t*>(st.cublas_trans.data()) + b,
        hm.data() + b, hn.data() + b, hk.data() + b,
        st.cublas_alpha.data() + b,
        reinterpret_cast<const float* const*>(Aptrs + (long)b * B),
        hlda.data() + b,
        reinterpret_cast<const float* const*>(Bptrs + (long)b * B),
        hlda.data() + b,
        st.cublas_beta.data() + b,
        Cptrs + (long)b * B,
        hlda.data() + b,
        level_size,
        st.cublas_group_size.data() + b);
    // If cuBLAS reports an error here, falling back would be incorrect because phaseA has
    // already modified the front. Let the later sync/residual path expose the failure.
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char* dbg = std::getenv("CLS_CUBLAS_STATUS_DEBUG");
        if (dbg && dbg[0] && dbg[0] != '0') {
            std::fprintf(stderr, "[CLS] cublasSgemmGroupedBatched status=%d range=[%d,%d) B=%d\n",
                         static_cast<int>(status), b, e, B);
        }
    }
    return true;
}
#endif

// ---------------------------------------------------------------------------------------
// Per-tier dispatch helpers (small / mid / big), each launching the kernel variant for the
// active Precision. issue_factor_level_range (below) scans the range, classifies the tier, and
// calls the matching helper.
// ---------------------------------------------------------------------------------------

// SMALL tier: sub-group-packed kernel. One sub-group of sub_group_size lanes owns one front; 32/sub_group_size fronts
// pack per warp, sub_group_size chosen by max_fsz so the dominant tiny fronts keep all sub_group_size lanes busy. The
// factor_small_sg gate keeps the classic one-warp-per-front form (sub_group_size=32) until the packed grid
// fills the GPU.
static void dispatch_factor_small(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                  int b, int e, const int* d_plc, const FrontRangeCaps& m)
{
#ifdef CLS_DISABLE_SMALL_FACTOR
    (void)plan; (void)st; (void)stream; (void)b; (void)e; (void)d_plc; (void)m;
    return;
#else
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;
    const long warps_unpacked = (long)level_size * B;            // unpacked (one-warp-per-front) count
    const int sub_group_size = factor_small_sg(m.max_fsz, warps_unpacked), fronts_per_warp = kWarpSize / sub_group_size;
    const int threads_per_block = kSmallTierWarpsPerBlock * kWarpSize;
    const int num_blocks = (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp + kSmallTierWarpsPerBlock - 1) / kSmallTierWarpsPerBlock);
    const int front_area = m.max_fsz * m.max_fsz;
    const size_t element_bytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
    const size_t shared_bytes = (size_t)kSmallTierWarpsPerBlock * fronts_per_warp * front_area * element_bytes;
#ifdef CLS_SMALL_TF32_TC
    if (precision == Precision::TF32 && B >= 128 && m.max_fsz > 16) {
        const int tc_num_blocks = (int)((warps_unpacked + kSmallTierWarpsPerBlock - 1) /
                                        kSmallTierWarpsPerBlock);
        const size_t tc_shared_bytes =
            (size_t)kSmallTierWarpsPerBlock * front_area * sizeof(float);
        factor_small_tf32_tc<<<tc_num_blocks, threads_per_block, tc_shared_bytes, stream>>>(
            b, level_size, B, front_area, d_plc, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
            st.d_front_batch_f, plan.front_total, st.d_sing, do_extend);
        return;
    }
#endif
    if (precision == Precision::FP64)
        launch_factor_small_t<double>(sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area,
                                      plan, d_plc, st.d_front_batch, st.d_sing, do_extend);
    else
        launch_factor_small_t<float>(sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B, front_area,
                                     plan, d_plc, st.d_front_batch_f, st.d_sing, do_extend);
#endif
}

// MID tier: shared-resident kernel, one block per (front, batch). Returns false when the chosen
// shared layout overflows the per-block budget (e.g. FP64 with a large fsz_cap), so the caller
// falls through to the big tier on the same range.
static bool dispatch_factor_mid(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                int b, int e, const int* d_plc, const int* h_plc,
                                const FrontRangeCaps& m)
{
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = m;
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;
    const int fsz_cap = max_fsz;
    (void)max_uc;
    dim3 grid(level_size, B);

    const size_t element_bytes = (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
    const size_t shared_bytes_tiled = (size_t)fsz_cap * fsz_cap * element_bytes              // Fs
                           + (size_t)2 * level_max_nc * level_max_uc * element_bytes;  // sh_L + sh_U
    // Underfilled levels (block count ≤ num_SMs, e.g. B=1 deep narrow levels) each run a single
    // latency-bound block; widen it to the device max so the per-pivot rank-1 update parallelises
    // over the SM. Once the level fills the GPU with blocks, keep the default 256.
    const int threads_per_block = ((long)(e - b) * B <= factor_num_sms()) ? factor_max_block() : 256;
#if defined(CLS_CUBLAS_TF32_TRAILING) && defined(CLS_CUBLAS_TF32_MID)
    if (precision == Precision::TF32 && fsz_cap >= 64 && cublas_tf32_trailing_ready(st) &&
        cublas_range_has_only_deferred_trailing(plan, h_plc, b, e)) {
        constexpr int phaseA_threads = 256;
        const size_t shared_bytes_phaseA = (size_t)fsz_cap * fsz_cap * sizeof(float);
        if (shared_bytes_phaseA <= kMidSharedMemoryBudgetBytes) {
            factor_mid_cublas_phaseA<<<grid, phaseA_threads, shared_bytes_phaseA, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                st.d_front_batch_f, plan.front_total, st.d_sing, fsz_cap);
            if (launch_cublas_tf32_trailing(plan, st, stream, b, e, d_plc, h_plc)) {
#ifndef CLS_DISABLE_EXTEND_ADD
                factor_big_extend<float><<<grid, phaseA_threads, 0, stream>>>(
                    b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                    plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
                    st.d_front_batch_f, plan.front_total);
#endif
                return true;
            }
        }
    }
#endif
#ifdef CLS_MID_TF32_TC
#ifndef CLS_MID_TF32_MIN_FSZ
#define CLS_MID_TF32_MIN_FSZ 48
#endif
    const bool mid_tf32_low_tc_enabled =
#ifdef CLS_MID_TF32_LOW_TC_FORCE_ALL
        true;
#elif defined(CLS_MID_TF32_LOW_TC)
        (plan.num_rows >= 20000 && plan.num_rows < 80000);
#else
        false;
#endif
	    if (precision == Precision::TF32 && B >= 64 &&
	        (fsz_cap > CLS_MID_TF32_MIN_FSZ || (mid_tf32_low_tc_enabled && fsz_cap > 16))) {
	        const int ucp_max = round_up_to_multiple(max_uc, 16);
	        const int kp_max = round_up_to_multiple(std::min(level_max_nc, kTensorCorePivotColumnCap), 8);
	        const int mid_tf32_direct_shared =
#if defined(CLS_MID_TF32_FORCE_BLOCKED)
	            0;
#elif defined(CLS_MID_TF32_DIRECT_SHARED)
	            1;
#elif defined(CLS_MID_TF32_DIRECT_HIGH)
	            (fsz_cap > kMidSplitFrontMax) ? 1 : 0;
#else
	            0;
#endif
	        constexpr int mid_tf32_tc_threads =
	#ifdef CLS_MID_TF32_TC_THREADS_128
	            128;
	#elif defined(CLS_MID_TF32_TC_THREADS_512)
	            512;
	#else
	            256;
	#endif
	        size_t ozaki_stage_bytes = 0;
#if defined(CLS_TF32_OZAKI_TC2) && defined(CLS_TF32_OZAKI_STAGE_DIRECT)
	        if (mid_tf32_direct_shared && fsz_cap <= kMidSplitFrontMax) {
	            ozaki_stage_bytes = (size_t)4 * ucp_max * kp_max * sizeof(unsigned);
	        }
#endif
	        const size_t shared_bytes_tf32_tc =
	            (size_t)fsz_cap * fsz_cap * sizeof(float) + ozaki_stage_bytes;
	        if (shared_bytes_tf32_tc <= kMidSharedMemoryBudgetBytes) {
	            factor_mid_tf32_ptx<<<grid, mid_tf32_tc_threads, shared_bytes_tf32_tc, stream>>>(
	                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
	                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total, st.d_sing, do_extend, fsz_cap, ucp_max, kp_max,
                mid_tf32_direct_shared);
            return true;
        }
    }
#endif
#ifdef CLS_MID_FP16_TC
    // Mid FP16 TC is useful on the 13K/25K-sized mid-front workloads but was unsafe on the
    // smaller nc=8-dominant 8387 case and on USA's ill-conditioned parallel-ND ordering.
    // Keep the default policy narrow; CLS_MID_FP16_TC=OFF restores the scalar mid path.
    const bool mid_fp16_tc_size_enabled =
#ifdef CLS_MID_FP16_TC_FORCE_ALL
        true;
#else
        (plan.num_rows >= 20000 && plan.num_rows < 80000);
#endif
    if (precision == Precision::FP16 && mid_fp16_tc_size_enabled) {
        const int ucp_max = round_up_to_multiple(max_uc, 16);
        const int kp_max = round_up_to_multiple(std::min(level_max_nc, kTensorCorePivotColumnCap), 8);
        constexpr int mid_fp16_tc_threads =
#ifdef CLS_MID_FP16_TC_THREADS_256
            256;
#else
            512;
#endif
        const size_t shared_bytes_fp16_tc =
#ifdef CLS_FP16_BLOCKED_SHARED_TC
            (size_t)fsz_cap * fsz_cap * sizeof(float);
#else
            (size_t)fsz_cap * fsz_cap * sizeof(float) +
            (size_t)2 * ucp_max * kp_max * sizeof(__half);
#endif
        if (shared_bytes_fp16_tc <= kMidSharedMemoryBudgetBytes) {
            factor_mid_fp16_ptx<<<grid, mid_fp16_tc_threads, shared_bytes_fp16_tc, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc,
                ucp_max, kp_max);
            return true;
        }
    }
#endif
    if (shared_bytes_tiled <= kMidSharedMemoryBudgetBytes) {
        if (precision == Precision::FP64)
            factor_mid<double><<<grid, threads_per_block, shared_bytes_tiled, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
        else
            factor_mid<float><<<grid, threads_per_block, shared_bytes_tiled, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total, st.d_sing, do_extend, fsz_cap, level_max_nc, level_max_uc);
        return true;
    }
    // The mid shared layout did not fit; tell the caller to fall through to the big tier.
    return false;
}

// BIG tier: global-memory kernel. Underfilled levels (level_size × B < num_SMs, e.g. the deep
// levels of a single-system solve) split the scalar path across blocks so the trailing GEMM
// fans out across SMs instead of running one block per front; filled levels keep the fused kernel.
static void dispatch_factor_big(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                int b, int e, const int* d_plc, const int* h_plc,
                                const FrontRangeCaps& m)
{
    const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = m;
    const Precision precision = st.precision;
    const int B = st.batch_count;
    const int level_size = e - b;
    constexpr int do_extend = kFactorDoExtend;
    (void)max_fsz;
    dim3 grid(level_size, B);
#ifdef CLS_BIG_MB
    const bool big_underfill = ((long)level_size * B < factor_num_sms());
#else
    // Default: no multi-block. Every big front runs the fused single-block kernel, so tf32/fp16
    // use the tensor-core trailing everywhere (the MB path was scalar). Define CLS_BIG_MB to
    // restore the underfill multi-block path (faster for fp32 at B=1; see docs note 30).
    const bool big_underfill = false;
#endif
    if (precision == Precision::FP64) {
        // FP64 uses a smaller block (128) — the scalar trailing on global memory does not
        // amortise the larger occupancy cost of a 1024-thread block at FP64 register pressure.
        constexpr int T = 128;
        if (big_underfill) {
            launch_factor_big_mb<double>(b, e, level_size, B, T, level_max_uc, plan, d_plc,
                                         st.d_front_batch, st.d_sing, do_extend, stream);
            return;
        }
        const size_t big_lu_bytes = (size_t)2 * level_max_nc * level_max_uc * sizeof(double);
#ifndef CLS_NO_BIG_STAGED
        if (big_lu_bytes <= kMidSharedMemoryBudgetBytes) {
            factor_big_staged<double><<<grid, T, big_lu_bytes, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
                plan.front_total, st.d_sing, do_extend, level_max_nc, level_max_uc);
            return;
        }
#endif
        factor_big<double><<<grid, T, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
            plan.front_total, st.d_sing, do_extend);
        return;
    }
    constexpr int bigT = 1024;
    // TF32 path: the TC-fused trailing is already fast, so fanning out costs the TC speedup and
    // only pays under *severe* underfill — a handful of large fronts on a near-idle GPU (the
    // deep levels of a single-system solve). There the FP32 scalar multi-block path beats
    // TC-fused (SM fanout, no panel staging) and is more accurate. Tighter gate than the scalar
    // path (≈ num_SMs/8) so moderately-underfilled levels, where TC-fused still wins, keep the
    // TC kernel. (FP16 stays on the fused kernel — its TC trailing is fast enough that even
    // severe-underfill routing nets a small loss on the smaller cases.)
#ifdef CLS_BIG_MB
    const bool tf32_severe_underfill = (precision == Precision::TF32) &&
                                       ((long)level_size * B * 8 < factor_num_sms());
#else
    const bool tf32_severe_underfill = false;
#endif
    if (tf32_severe_underfill) {
        constexpr int bigT_tc = 512;
        launch_factor_big_mb<float>(b, e, level_size, B, bigT_tc, level_max_uc, plan, d_plc,
                                    st.d_front_batch_f, st.d_sing, do_extend, stream);
        return;
    }
    if (precision == Precision::FP16) {
#ifdef CLS_FP16_BLOCKED_SHARED_TC
        const size_t shared_front_bytes = (size_t)max_fsz * max_fsz * sizeof(float);
        if (shared_front_bytes <= kDynamicSharedMemoryOptInBytes) {
            constexpr int bigT_fp16_shared = 256;
            factor_big_shared_fp16_blocked<<<grid, bigT_fp16_shared, shared_front_bytes, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total, st.d_sing, do_extend, max_fsz);
            return;
        }
#endif
        // 512-thread block + __launch_bounds__(512, 2) on factor_big_fp16_ptx so two blocks
        // fit per SM on sm_86. Shared scratch is only the __half staging panels (Lh, Uh) —
        // the front stays in global memory and the inline-asm mma needs no Csc readback.
        constexpr int bigT_fp16 =
#ifdef CLS_BIG_FP16_TC_THREADS_256
            256;
#else
            512;
#endif
        const int ucp_max = round_up_to_multiple(max_uc, 16);
        const int kp_max = round_up_to_multiple(std::min(level_max_nc, kTensorCorePivotColumnCap), 8);
        const size_t shbytes = (size_t)2 * ucp_max * kp_max * sizeof(__half);
        factor_big_fp16_ptx<<<grid, bigT_fp16, shbytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, ucp_max, kp_max);
        return;
    }
#ifdef CLS_CUBLAS_TF32_TRAILING
    if (precision == Precision::TF32 && cublas_tf32_trailing_ready(st) &&
        cublas_range_has_only_deferred_trailing(plan, h_plc, b, e)) {
        constexpr int bigT_cublas = 1024;
        factor_big_panel<float><<<grid, bigT_cublas, 0, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            st.d_front_batch_f, plan.front_total, st.d_sing);
        if (launch_cublas_tf32_trailing(plan, st, stream, b, e, d_plc, h_plc)) {
#ifndef CLS_DISABLE_EXTEND_ADD
            factor_big_extend<float><<<grid, bigT_cublas, 0, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total);
#endif
            return;
        }
    }
#endif
    if (precision == Precision::TF32) {
#ifdef CLS_BIG_TF32_BLOCKED_TC
        const size_t shared_front_bytes = (size_t)max_fsz * max_fsz * sizeof(float);
        if (shared_front_bytes <= kDynamicSharedMemoryOptInBytes) {
            constexpr int bigT_tf32_shared =
#ifdef CLS_BIG_TF32_SHARED_THREADS_128
                128;
#elif defined(CLS_BIG_TF32_SHARED_THREADS_512)
                512;
#else
                256;
#endif
            factor_big_shared_tf32_blocked<<<grid, bigT_tf32_shared, shared_front_bytes, stream>>>(
                b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
                plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
                plan.front_total, st.d_sing, do_extend, max_fsz);
            return;
        }
#endif
        // 512-thread block. The matching `__launch_bounds__(512, 2)` on factor_big_tf32_ptx
        // constrains nvcc so two blocks of 512 threads fit per SM on sm_86 (the alternative
        // 1024-thread block would be capped at one resident block by the thread-per-SM limit).
        // PTX path needs no Csc scratch.
        constexpr int bigT_tf32 =
#ifdef CLS_BIG_TF32_THREADS_256
            256;
#elif defined(CLS_BIG_TF32_THREADS_384)
            384;
#else
            512;
#endif
        const int ucp_max = round_up_to_multiple(max_uc, 16);
        const int kp_max = round_up_to_multiple(std::min(level_max_nc, kTensorCorePivotColumnCap), 8);
        const size_t shbytes = (size_t)(2 * ucp_max * kp_max + 4 * kp_max) * sizeof(float);  // +4*kp_max: Utf LDB pad
        factor_big_tf32_ptx<<<grid, bigT_tf32, shbytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, ucp_max, kp_max);
        return;
    }
    // FP32 big trailing on the global front.
    if (big_underfill) {
        launch_factor_big_mb<float>(b, e, level_size, B, bigT, level_max_uc, plan, d_plc,
                                    st.d_front_batch_f, st.d_sing, do_extend, stream);
        return;
    }
    const size_t big_lu_bytes = (size_t)2 * level_max_nc * level_max_uc * sizeof(float);
#ifndef CLS_NO_BIG_STAGED
    if (big_lu_bytes <= kMidSharedMemoryBudgetBytes) {
        factor_big_staged<float><<<grid, bigT, big_lu_bytes, stream>>>(
            b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
            plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
            plan.front_total, st.d_sing, do_extend, level_max_nc, level_max_uc);
        return;
    }
#endif
    factor_big<float><<<grid, bigT, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
        plan.front_total, st.d_sing, do_extend);
}

// Per-range dispatcher: scan the panels in plcols[b..e), pick the tier from the level's max front
// size, and dispatch one right-sized kernel. A mid range whose shared layout overflows the
// per-block budget falls through to the big tier on the same range.
static void issue_factor_level_range(const MultifrontalPlan& plan, State& st,
                                     cudaStream_t stream, int b, int e,
                                     const int* d_plc, const int* h_plc)
{
    if (e <= b) return;
    const FrontRangeCaps m = scan_front_range(plan, h_plc, b, e);
    switch (classify_front_tier(m.max_fsz)) {
        case FrontTier::kSmall:
            dispatch_factor_small(plan, st, stream, b, e, d_plc, m);
            return;
        case FrontTier::kMid:
            if (dispatch_factor_mid(plan, st, stream, b, e, d_plc, h_plc, m)) return;
            [[fallthrough]];
        case FrontTier::kBig:
            dispatch_factor_big(plan, st, stream, b, e, d_plc, h_plc, m);
            return;
    }
}

// ---------------------------------------------------------------------------------------
// Outer dispatcher: iterate etree levels.
//
// Single-stream path:
//   for L in 0..num_plevels:
//       issue_factor_level_range(plan, st, stream, plptr[L], plptr[L+1])
//
// Multi-stream path (fork / wait / join):
//   - st.subtree_streams[0..K-1] hold K extra streams (one per subtree, K ≤ 8).
//   - plan.h_subtree_level_off / _cnt indexes plcols[] so each subtree gets its own
//     (level → range) slice below the spine.
//
//     [main]  ----- fork_event ----┐
//                                  ├──> subtree 0 stream:  L0 ranges → L1 ranges → ...  ── join_events[0] ─┐
//                                  ├──> subtree 1 stream:  L0 ranges → L1 ranges → ...  ── join_events[1] ─┤
//                                  ├──> ...                                                                  │
//     [main]                       └──> subtree K-1 stream: ...                          ── join_events[K-1] ┤
//     [main]  --wait(join_events[0..K-1])--->  spine levels (L_spine_start..num_plevels)
//
//   The spine (`plan.spine_start_level..num_plevels`) holds the levels above the subtree
//   roots and runs on the main stream after all subtree streams have signalled completion.
//   The use_multistream gate also handles the "no spine" case (spine_start_level < 0) by
//   not enqueueing any spine levels on the main stream.
// Occupancy gate: warp-packing a small-tier range beats the block-per-front kernel only once the
// small fronts × B fill the GPU's warp slots. Below that — notably B=1, the latency regime — the
// block-per-front kernel gives each small front 8× the threads and a mixed range already has enough
// fronts to occupy the device, so keeping the range merged on the larger kernel wins (unsplit
// regressed B=1 factor by up to ~80%). The threshold is a pure hardware quantity (SMs × warps/SM
// via device attrs), so the rule generalizes across matrices and GPUs.
static long factor_warp_fill()
{
    static const long v = [] {
        int dev = 0; cudaGetDevice(&dev);
        int sm = 1, tpm = 1536;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        return (long)sm * (tpm / 32);
    }();
    return v;
}

// Dispatch one (level- or cell-) range described by its (kNumTiers+1) tier boundaries `tb` into
// d_plc/h_plc. Splits into tier-homogeneous sub-launches when the occupancy gate passes, else
// dispatches the whole range on the larger kernel (pre-split behaviour). Same range = independent
// fronts, so the small→mid→big sub-launches are order-free and correct.
static void issue_factor_tiered(const MultifrontalPlan& plan, State& st, cudaStream_t stream,
                                const int* tb, const int* d_plc, const int* h_plc)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    constexpr int NS = MultifrontalPlan::kSmallTiers;
    const long small_cnt = tb[NS] - tb[0];
    const bool mixed = (tb[NT] - tb[0]) > small_cnt;  // a larger-tier front is present
    if (st.tier_split && small_cnt > 0 && mixed && small_cnt * (long)st.batch_count >= factor_warp_fill()) {
#if defined(CLS_SMALL_BUCKET_SPLIT_16)
        if (NS > 1 && st.batch_count < 128) {
            issue_factor_level_range(plan, st, stream, tb[0], tb[NS], d_plc, h_plc);
            for (int t = NS; t < NT; ++t)
                if (tb[t + 1] > tb[t])
                    issue_factor_level_range(plan, st, stream, tb[t], tb[t + 1], d_plc, h_plc);
            return;
        }
#endif
        for (int t = 0; t < NT; ++t)
            if (tb[t + 1] > tb[t])
                issue_factor_level_range(plan, st, stream, tb[t], tb[t + 1], d_plc, h_plc);
    } else {
        issue_factor_level_range(plan, st, stream, tb[0], tb[NT], d_plc, h_plc);
    }
}

static void issue_factor_levels(const MultifrontalPlan& plan, State& st, cudaStream_t stream)
{
    constexpr int NT = MultifrontalPlan::kNumTiers;
    const bool use_multistream = st.num_subtree_streams > 1 &&
                                 plan.num_subtrees == st.num_subtree_streams &&
                                 !plan.h_subtree_level_off.empty();
    const int* lvl_tb = plan.h_level_tier_off.data();           // single-stream: per-level ranges
    const int* sub_tb = plan.h_subtree_level_tier_off.data();   // multistream: per-cell ranges
    const bool have_lvl = !plan.h_level_tier_off.empty() && plan.d_plcols_tier;
    const bool have_sub = !plan.h_subtree_level_tier_off.empty();

    if (use_multistream) {
        const int spine_lo = (plan.spine_start_level >= 0) ? plan.spine_start_level : plan.num_plevels;

        // Fork: record an event on the main stream and have each subtree stream wait on it
        // so the subtree work cannot start until any prior main-stream work has retired.
        cudaEventRecord(static_cast<cudaEvent_t>(st.fork_event), stream);
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(static_cast<cudaStream_t>(st.subtree_streams[k]),
                                static_cast<cudaEvent_t>(st.fork_event), 0);
        }

        // Each subtree stream sweeps levels 0..spine_lo-1 over its own panel slice, tier-splitting
        // each (subtree, level) cell under the same occupancy gate as the single-stream walk.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStream_t ks = static_cast<cudaStream_t>(st.subtree_streams[k]);
            const int* off = plan.h_subtree_level_off.data() + (long)k * plan.num_plevels;
            const int* cnt = plan.h_subtree_level_cnt.data() + (long)k * plan.num_plevels;
            for (int L = 0; L < plan.num_plevels; ++L) {
                if (L >= spine_lo) break;
                if (cnt[L] == 0) continue;
                if (have_sub)
                    issue_factor_tiered(plan, st, ks,
                                        sub_tb + ((long)k * plan.num_plevels + L) * (NT + 1),
                                        plan.d_plcols, plan.h_plcols.data());
                else
                    issue_factor_level_range(plan, st, ks, off[L], off[L] + cnt[L],
                                             plan.d_plcols, plan.h_plcols.data());
            }
            cudaEventRecord(static_cast<cudaEvent_t>(st.join_events[k]), ks);
        }

        // Join: main stream waits on every subtree's join event before issuing the spine.
        for (int k = 0; k < st.num_subtree_streams; ++k) {
            cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(st.join_events[k]), 0);
        }

        // Spine levels (if any) on the main stream — cnt=1 chain, nothing to tier-split.
        if (plan.spine_start_level >= 0) {
            for (int L = plan.spine_start_level; L < plan.num_plevels; ++L) {
                const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
                issue_factor_level_range(plan, st, stream, b, e,
                                         plan.d_plcols, plan.h_plcols.data());
            }
        }
    } else {
        // Single stream: tier-split each level under the occupancy gate (see issue_factor_tiered).
        for (int L = 0; L < plan.num_plevels; ++L) {
            if (have_lvl)
                issue_factor_tiered(plan, st, stream, lvl_tb + (long)L * (NT + 1),
                                    plan.d_plcols_tier, plan.h_plcols_tier.data());
            else
                issue_factor_level_range(plan, st, stream, plan.panel_level_ptr[L], plan.panel_level_ptr[L + 1],
                                         plan.d_plcols, plan.h_plcols.data());
        }
    }
}

}  // namespace
}  // namespace custom_linear_solver
