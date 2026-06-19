#pragma once

// SOLVE — single-system (B=1) schedule. Ported from cuDSS_reproduce
// (mysolver/gpu/gpu_mf.cu).
//
// Multifrontal forward (L y = b, leaves->root) then backward (U x = y,
// root->leaves), one CUDA block per front, per panel level, with per-level
// block-size heuristics. The B=1 factor (Factorize/ single.cuh) inverts each
// front's pivot block, so the per-front triangular solves become parallel GEMVs
// (y = Linv @ rhs, x = Uinv @ rhs). Operates in place on the B=1 slice of
// State.d_y_batch[_f], reading the front the B=1 factor produced
// (State.d_front_batch[_f]); the I/O permutation (gather/scatter) is shared
// with the batched Solve driver.

#include <cuda_runtime.h>

#include <algorithm>

#include "internal/plan/multifrontal_plan.hpp"
#include "internal/runtime/state.hpp"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

constexpr int kSingleMaxNc = 64;     // backward rhs[] bound (nc <= panel cap)
constexpr int kSingleBwdRegNc = 16;  // backward register-partial bound

// Forward Solve L y = b. Step 1: the nc x nc unit-lower pivot block via the
// inverted-pivot GEMV (sh_piv = Linv @ rhs). Step 2: apply the L panel to the
// CB rows -> atomicAdd into the global y.
template <typename FT>
__global__ void SolveSingleFwd(int lbegin, int lend,
                               const int* __restrict__ plcols,
                               const int* __restrict__ front_off,
                               const int* __restrict__ front_ptr,
                               const int* __restrict__ ncols,
                               const int* __restrict__ front_rows,
                               const FT* __restrict__ front, FT* y) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const FT* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  // Accumulate in the front precision FT (double for FP64, float for FP32/TF32):
  // the front + inverted pivots are already FT, so a double accumulate buys no
  // accuracy, only conversions, and FT keeps the FP32 path all-float.
  __shared__ FT sh_piv[64];

  // Step 1: pivot block via inverted-pivot GEMV (sh_piv = Linv @ rhs).
  for (int k = t; k < nc; k += nt) {
    FT v = y[fr[k]];
    for (int i = 0; i < k; ++i)
      v += F[(long)k * fsz + i] * y[fr[i]];  // Linv[k][i], i<k
    sh_piv[k] = v;
  }
  __syncthreads();

  // Step 2: write pivots back, then apply the L panel to the CB rows
  // (atomicAdd because siblings scatter into the same parent y entries).
  for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
  for (int i = nc + t; i < fsz; i += nt) {
    FT upd = 0;
    for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
    atomicAdd(&y[fr[i]], -upd);
  }
}

// Backward Solve U x = y. The CB-column product (bulk) is a parallel reduction
// into nc register partials; the nc x nc upper back-Solve runs as the
// inverted-pivot GEMV (x = Uinv @ rhs).
template <typename FT>
__global__ void SolveSingleBwd(int lbegin, int lend,
                               const int* __restrict__ plcols,
                               const int* __restrict__ front_off,
                               const int* __restrict__ front_ptr,
                               const int* __restrict__ ncols,
                               const int* __restrict__ front_rows,
                               const FT* __restrict__ front, FT* y) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const FT* F = front + front_off[p];
  const int* fr = front_rows + s;
  const int t = threadIdx.x, nt = blockDim.x;
  const int cb = fsz - nc;
  // Accumulate in the front precision FT: the front + inverted pivots are
  // already FT, so a double accumulate buys no accuracy on the FP32 path, only
  // conversions; FT keeps the FP32 solve all-float and halves the shared
  // rhs/wsum footprint.
  __shared__ FT rhs[kSingleMaxNc];
  __shared__ FT wsum[(256 / 32) * kSingleBwdRegNc];

  // Step 1: load rhs[] from y, then reduce the CB-column product into nc
  // register partials per thread.
  for (int k = t; k < nc; k += nt) rhs[k] = __ldg(&y[fr[k]]);
  FT part[kSingleBwdRegNc];
  for (int k = 0; k < nc; ++k) part[k] = 0;
  for (int j = t; j < cb; j += nt) {
    const FT xj = __ldg(&y[fr[nc + j]]);
    for (int k = 0; k < nc; ++k) part[k] += F[(long)k * fsz + (nc + j)] * xj;
  }
  const int lane = t & 31, warp = t >> 5;

  // Single-warp fast path: warp-reduce partials into rhs, then the nc x nc
  // upper back-solve as the inverted-pivot GEMV. No wsum staging, no block sync.
  if (nt <= 32) {
    __syncwarp();
    for (int k = 0; k < nc; ++k) {
      FT v = part[k];
      for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0) rhs[k] -= v;
    }
    __syncwarp();
    for (int k = t; k < nc; k += nt) {
      FT v = 0;
      for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
      y[fr[k]] = v;
    }
    return;
  }

  // Step 2: multi-warp reduce of partials through wsum into rhs.
  for (int k = 0; k < nc; ++k) {
    FT v = part[k];
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) wsum[warp * nc + k] = v;
  }
  __syncthreads();
  if (t == 0) {
    const int nw = (nt + 31) / 32;
    for (int k = 0; k < nc; ++k) {
      FT sm = 0;
      for (int w = 0; w < nw; ++w) sm += wsum[w * nc + k];
      rhs[k] -= sm;
    }
  }
  __syncthreads();

  // Step 3: nc x nc upper back-solve as the inverted-pivot GEMV (x = Uinv @ rhs).
  for (int k = t; k < nc; k += nt) {
    FT v = 0;
    for (int j = k; j < nc; ++j) v += F[(long)k * fsz + j] * rhs[j];
    y[fr[k]] = v;
  }
}

// Per-level Solve block size (gpu_mf level_ts): big circuit fronts want more
// threads, the many tiny levels want 32 for occupancy, the narrow mid-size
// spine wants ~96, the narrow tiny spine a single warp (triggers the kernel's
// single-warp fast path).
static int SingleSolveBlockSize(const MultifrontalPlan& plan, int L) {
  const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
  const int cnt = e - b;
  int mx = 0;
  for (int q = b; q < e; ++q) {
    const int p = plan.h_plcols[q];
    mx = std::max(mx, plan.h_front_ptr[p + 1] - plan.h_front_ptr[p]);
  }
  if (mx >= 256) return 192;
  if (mx <= 48 && cnt >= 512) return 32;
  if (cnt < 82 && mx >= 64) return 96;
  if (cnt < 82 && mx <= 40) return 32;
  return 64;
}

template <typename FT>
static void IssueSolveSingleTyped(const MultifrontalPlan& plan, State& st,
                                  const FT* front, FT* y, cudaStream_t stream) {
  // Forward sweep leaves->root, one front per block, per panel level.
  for (int L = 0; L < plan.num_plevels; ++L) {
    const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
    if (e <= b) continue;
    SolveSingleFwd<FT><<<e - b, SingleSolveBlockSize(plan, L), 0, stream>>>(
        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, front, y);
  }

  // Backward sweep root->leaves.
  for (int L = plan.num_plevels - 1; L >= 0; --L) {
    const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
    if (e <= b) continue;
    SolveSingleBwd<FT><<<e - b, SingleSolveBlockSize(plan, L), 0, stream>>>(
        b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_front_rows, front, y);
  }
}

// The working vector y matches the front precision: the gather casts the RHS
// into d_y_batch (double) for FP64 or d_y_batch_f (float) for FP32/TF32, the
// type the templated fwd/bwd kernels read/write.
static void IssueSolveSingle(const MultifrontalPlan& plan, State& st,
                             cudaStream_t stream) {
  if (IsFp32Front(st.precision))
    IssueSolveSingleTyped<float>(plan, st, st.d_front_batch_f, st.d_y_batch_f,
                                 stream);
  else
    IssueSolveSingleTyped<double>(plan, st, st.d_front_batch, st.d_y_batch,
                                  stream);
}

}  // namespace
}  // namespace custom_linear_solver
