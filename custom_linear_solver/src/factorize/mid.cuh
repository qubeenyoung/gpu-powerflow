#pragma once

// FACTORIZE — MID tier (front fits the per-precision opt-in shared budget at
// full occupancy). The whole front is staged into shared and factorized in
// place; TF32 uses the blocked Tensor-Core trailing. Kernel + blocked-TC
// helpers + dispatch, all here.

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// =======================================================================================
//  STAGE-IN  —  global → shared
// =======================================================================================
//
// Copies the whole fsz×fsz front from global to dynamic shared so the four
// phases below touch only shared memory. On Ampere+ each element is issued
// through cp.async via
// __pipeline_memcpy_async, then committed and waited as one batch; on
// pre-Ampere the helper falls back to a synchronous strided copy. The async
// path lets a thread queue many loads before stalling, giving the SM something
// else to schedule while the loads are in flight.
template <typename T>
__device__ __forceinline__ void StageInAsync(T* __restrict__ Fs,
                                             const T* __restrict__ F, int fsz2,
                                             int t, int nt) {
#if __CUDA_ARCH__ >= 800
  constexpr size_t SHAPE = sizeof(T);
  for (int e = t; e < fsz2; e += nt) {
    __pipeline_memcpy_async(&Fs[e], &F[e], SHAPE);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
#else
  for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
#endif
}

__device__ __forceinline__ void BlockUpdateTf32Tc(float* F, int fsz, int row0,
                                                  int col0, int dim, int k0,
                                                  int kb, int t, int nt) {
  if (dim <= 0 || kb <= 0) return;
  const int DP = ((dim + 15) / 16) * 16;
  const int nc_pad = ((kb + 7) / 8) * 8;
  const int m_tiles = DP / 16;
  const int n_tiles = DP / 8;
  const int k_tiles = nc_pad / 8;
  const int warp = t >> 5;
  const int nwarp = nt >> 5;
  const int lane = t & 31;
  const int laneR = lane >> 2;
  const int laneC = (lane & 3) * 2;

  auto load_l = [&](int r, int k) {
    return (r < dim && k < kb) ? F[(long)(row0 + r) * fsz + (k0 + k)] : 0.0f;
  };
  auto load_u = [&](int k, int col) {
    return (k < kb && col < dim) ? F[(long)(k0 + k) * fsz + (col0 + col)]
                                 : 0.0f;
  };
  auto store_c = [&](int r, int col, float c) {
    if (r < dim && col < dim) F[(long)(row0 + r) * fsz + (col0 + col)] -= c;
  };

  constexpr int NTJ8_MAX = 16;
  if (n_tiles <= NTJ8_MAX) {
    for (int ti = warp; ti < m_tiles; ti += nwarp) {
      const int r_top = ti * 16 + laneR;
      const int r_bot = r_top + 8;
      float c[NTJ8_MAX][4];
#pragma unroll
      for (int j = 0; j < NTJ8_MAX; ++j) {
        c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
      }
      for (int kc = 0; kc < k_tiles; ++kc) {
        const int k = kc * 8 + laneC;
        const Tf32Pair a0 = Tf32OzakiPair(load_l(r_top, k + 0));
        const Tf32Pair a1 = Tf32OzakiPair(load_l(r_bot, k + 0));
        const Tf32Pair a2 = Tf32OzakiPair(load_l(r_top, k + 1));
        const Tf32Pair a3 = Tf32OzakiPair(load_l(r_bot, k + 1));
#pragma unroll
        for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
          if (tj8 >= n_tiles) break;
          const int col = tj8 * 8 + laneR;
          const Tf32Pair b0 = Tf32OzakiPair(load_u(k + 0, col));
          const Tf32Pair b1 = Tf32OzakiPair(load_u(k + 1, col));
          CLS_MMA_TF32_OZAKI2(c[tj8][0], c[tj8][1], c[tj8][2], c[tj8][3], a0,
                              a1, a2, a3, b0, b1);
        }
      }
#pragma unroll
      for (int tj8 = 0; tj8 < NTJ8_MAX; ++tj8) {
        if (tj8 >= n_tiles) break;
        const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
        store_c(r_top, col0, c[tj8][0]);
        store_c(r_top, col1, c[tj8][1]);
        store_c(r_bot, col0, c[tj8][2]);
        store_c(r_bot, col1, c[tj8][3]);
      }
    }
    return;
  }

  for (int ti = warp; ti < m_tiles; ti += nwarp) {
    const int r_top = ti * 16 + laneR;
    const int r_bot = r_top + 8;
    for (int tj8 = 0; tj8 < n_tiles; ++tj8) {
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      for (int kc = 0; kc < k_tiles; ++kc) {
        const int k = kc * 8 + laneC;
        const Tf32Pair a0 = Tf32OzakiPair(load_l(r_top, k + 0));
        const Tf32Pair a1 = Tf32OzakiPair(load_l(r_bot, k + 0));
        const Tf32Pair a2 = Tf32OzakiPair(load_l(r_top, k + 1));
        const Tf32Pair a3 = Tf32OzakiPair(load_l(r_bot, k + 1));
        const int col = tj8 * 8 + laneR;
        const Tf32Pair b0 = Tf32OzakiPair(load_u(k + 0, col));
        const Tf32Pair b1 = Tf32OzakiPair(load_u(k + 1, col));
        CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
      }
      const int col0 = tj8 * 8 + laneC, col1 = col0 + 1;
      store_c(r_top, col0, c0);
      store_c(r_top, col1, c1);
      store_c(r_bot, col0, c2);
      store_c(r_bot, col1, c3);
    }
  }
}

__device__ __forceinline__ void FactorizeFrontBlockedTf32(
    float* F, int fsz, int nc, int t, int nt, int* sing, bool static_pivoting,
    double pivot_threshold, double pivot_shift) {
  constexpr int BK = 8;
  for (int k0 = 0; k0 < nc; k0 += BK) {
    const int kb = (k0 + BK <= nc) ? BK : (nc - k0);
    const int next = k0 + kb;

    // Factor the current diagonal block and L panel below it. Only the current
    // block's panel columns are updated here; the right-looking TC update below
    // handles the rest.
    for (int kk = 0; kk < kb; ++kk) {
      const int k = k0 + kk;
      const long diag = (long)k * fsz + k;
      float piv = GuardedPivot(F[diag], static_pivoting, pivot_threshold,
                               pivot_shift, sing, t == 0);
      if (t == 0) F[diag] = piv;
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

    // Solve the block row U over all remaining columns, including both the rest
    // of the panel and the contribution block. Row-by-row: row r depends on
    // rows < r in this pivot block, so a __syncthreads() follows each row.
    for (int kk = 0; kk < kb; ++kk) {
      const int row = k0 + kk;
      for (int j = next + t; j < fsz; j += nt) {
        float v = F[(long)row * fsz + j];
        for (int i = k0; i < row; ++i)
          v -= F[(long)row * fsz + i] * F[(long)i * fsz + j];
        F[(long)row * fsz + j] = v;
      }
      __syncthreads();
    }

    const int dim = fsz - next;
    if (dim > 0) {
      BlockUpdateTf32Tc(F, fsz, next, next, dim, k0, kb, t, nt);
      __syncthreads();
    }
  }
}

// Shared-resident blocked kernel — the unified SMALL-tier kernel for every
// precision. The full front is staged into shared (fits the 99 KiB opt-in
// budget for fsz below the per-precision shared limit) and factorized in place.
// For UseTC (T = float, TF32 precision) eligible shapes use the blocked TF32
// Tensor-Core trailing; otherwise (FP64/FP32, or TC-ineligible shapes) the same
// shared-resident blocked structure runs scalar trailing. Supersedes FactorMid
// / factor_mid_tf32_ptx / factor_big_shared_tf32_blocked.
template <typename T, bool UseTC>
__global__ void FactorMid(int lbegin, int lend, const int* __restrict__ plcols,
                          const int* __restrict__ front_off,
                          const int* __restrict__ front_ptr,
                          const int* __restrict__ ncols,
                          const int* __restrict__ panel_parent,
                          const int* __restrict__ asm_ptr,
                          const int* __restrict__ asm_local, T* frontB,
                          long front_total, int* sing, int do_extend,
                          int fsz_cap, int rb_width, bool static_pivoting,
                          double pivot_threshold, double pivot_shift) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  T* front = frontB + (long)blockIdx.y * front_total;
  const int p = plcols[idx];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  T* F = front + front_off[p];
  const int t = threadIdx.x, nt = blockDim.x;
  const int uc = fsz - nc;
  const int fsz2 = fsz * fsz;

  extern __shared__ char smem_mid_blocked[];
  T* Fs = reinterpret_cast<T*>(smem_mid_blocked);
  StageInAsync<T>(Fs, F, fsz2, t, nt);
  __syncthreads();

  // Factorize the front in place: TF32-eligible shapes take the blocked
  // Tensor-Core path; all other shapes / precisions run the scalar phases
  // (Phase 1 panel LU + Phase 2 U-Solve + Phase 3 trailing, with the
  // fsz<=kFusedMidFrontMax fused Phase 1+3 fast path).
  bool did_tc = false;
  if constexpr (UseTC) {
    if (fsz > CLS_TC_FSZ_MIN && uc >= CLS_TC_UC_MIN && nc >= CLS_TC_NC_MIN &&
        nc <= kTensorCorePivotColumnCap && uc <= CLS_TC_UC_CAP) {
      FactorizeFrontBlockedTf32(Fs, fsz, nc, t, nt, sing, static_pivoting,
                                pivot_threshold, pivot_shift);
      did_tc = true;
    }
  }
  if (!did_tc) {
    if (fsz <= kFusedMidFrontMax) {
      LuMidFront<T>(Fs, fsz, nc, t, nt, sing, static_pivoting, pivot_threshold,
                    pivot_shift);  // Phase 1 + Phase 3 fused
    } else {
      LuPanelFactor<T>(Fs, fsz, nc, t, nt, sing, static_pivoting,
                       pivot_threshold, pivot_shift);  // Phase 1
      UPanelSolveFewsync<T>(Fs, fsz, nc, uc, t,
                            nt);  // Phase 2 (barrier-cut default)
      TrailingUpdate<T>(Fs, fsz, nc, uc, t, nt,
                        rb_width);  // Phase 3 (register-blocked)
    }
  }

  WritebackFactored<T, T>(F, Fs, fsz, nc, uc, t, nt);

  const int par = panel_parent[p];
  if (par < 0 || !do_extend) return;
  __syncthreads();
  T* Fp = front + front_off[par];
  const int pfsz = front_ptr[par + 1] - front_ptr[par];
  const int abase = asm_ptr[p];
  ExtendAdd<T, T>(Fp, pfsz, Fs, fsz, nc, uc, asm_local, abase, t, nt);
  (void)fsz_cap;
}

// MID tier — whole-front shared-resident kernel (FactorMid). The entire
// fsz×fsz front is staged into shared and factorized in place; TF32 runs the
// blocked Tensor-Core trailing. The tier boundary (fsz <= kMidFrontMax = 64)
// guarantees the whole front fits the shared budget for every precision, so
// this never overflows. Routed deterministically by tier — no occupancy/opt-in
// gate. Thread count follows the swept occupancy heuristic (underfilled levels
// parallelise each front more).
static void DispatchFactorMid(const MultifrontalPlan& plan, State& st,
                              cudaStream_t stream, int b, int e,
                              const int* d_plc, const int* h_plc,
                              const FrontRangeCaps& caps) {
  const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
  const Precision precision = st.precision;
  const int B = st.batch_count;
  const int level_size = e - b;
  constexpr int do_extend = kFactorDoExtend;
  const int fsz_cap = max_fsz;
  // Register-blocked trailing width (exp_260612 sweep optimum): each thread
  // owns a 4-wide column strip of the trailing block, raising the FMA:load
  // ratio over the scalar form.
  constexpr int rb_width = 4;
  (void)max_uc;
  (void)level_max_nc;
  (void)level_max_uc;
  (void)h_plc;
  dim3 grid(level_size, B);

  const size_t element_bytes =
      (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
  const size_t shared_bytes =
      (size_t)fsz_cap * fsz_cap * element_bytes;  // whole front in shared
  const long blocks = (long)level_size * B;
  const long sms = FactorNumSms();
  // NB: bumping the under-fill case 512->1024 threads was measured neutral
  // (each front is dependency-latency-bound on the sequential panel-LU chain).
  // Kept at 512. (exp_260612)
  const int threads = (blocks < sms) ? 512 : (blocks < 4 * sms ? 256 : 128);

  if (precision == Precision::FP64) {
    FactorMid<double, false><<<grid, threads, shared_bytes, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
        plan.front_total, st.d_sing, do_extend, fsz_cap, rb_width,
        st.static_pivoting, st.pivot_threshold, st.pivot_shift);
  } else if (precision == Precision::TF32) {
    FactorMid<float, true><<<grid, threads, shared_bytes, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
        st.d_front_batch_f, plan.front_total, st.d_sing, do_extend, fsz_cap,
        rb_width, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
  } else {  // FP32
    FactorMid<float, false><<<grid, threads, shared_bytes, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
        st.d_front_batch_f, plan.front_total, st.d_sing, do_extend, fsz_cap,
        rb_width, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
  }
}

}  // namespace
}  // namespace custom_linear_solver
