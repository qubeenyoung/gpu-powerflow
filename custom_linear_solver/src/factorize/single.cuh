#pragma once

// FACTORIZE — single-system (B=1) schedule.
//
// The batched schedule (schedule.cuh) amortizes per-level launch / barrier
// latency across B systems; at B=1 that latency is exposed and the tier-split +
// occupancy-gate launches buy nothing. This path is tuned for B=1: one block
// per front, a FUSED factor+extend-add per level, per-level block-size
// heuristics, the multi-block big-front triple, and the partitioned-inverse
// pivot blocks that turn the Solve's triangular back-solves into parallel GEMVs
// (Solve/single.cuh).
//
// Runs against the SAME MultifrontalPlan device arrays as the batched path and
// the B=1 slice of the per-state front arena (State.d_front_batch[_f]). The
// numeric scatter (AssembleFrontValues) is shared; only the level walk and the
// per-front kernels differ here.

#include <cuda_runtime.h>

#include <algorithm>

#include "internal/plan/multifrontal_plan.hpp"
#include "internal/runtime/state.hpp"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

constexpr int kSingleRegNc =
    16;  // panel-cap bound for the partitioned-inverse shared buffers
constexpr int kSingleTrailTile = 16;  // big-front trailing tile size

// FUSED factor + extend-add, one block per front, one launch per level. Small
// fronts use a per-pivot rank-1 loop; big fronts use the blocked panel /
// U-panel / rank-nc trailing decomposition. The contribution block then
// extend-adds into the parent (a strictly higher, not-yet-factored level ->
// race-free atomicAdd).
template <typename FT>
__global__ void FactorSingleLevel(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ panel_parent,
    const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
    FT* front, int* sing, int do_extend) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  FT* F = front + front_off[p];
  const int t = threadIdx.x, nt = blockDim.x;
  const int uc = fsz - nc;

  if (fsz <= 48) {
    // Small front: per-pivot rank-1 LU over the whole remaining block.
    for (int k = 0; k < nc; ++k) {
      FT piv = F[(long)k * fsz + k];
      if (piv == FT(0)) {
        if (t == 0) *sing = 1;
        piv = FT(1);
      }
      for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
      __syncthreads();
      const int m = fsz - k - 1;
      for (int e = t; e < m * m; e += nt) {
        const int ii = k + 1 + e / m, jj = k + 1 + e % m;
        F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
      }
      __syncthreads();
    }
  } else {
    // Big front: blocked panel LU → U-panel Solve → rank-nc trailing.
    // Phase 1: panel LU on the pivot block (rank-1 over the nc-wide panel only).
    for (int k = 0; k < nc; ++k) {
      FT piv = F[(long)k * fsz + k];
      if (piv == FT(0)) {
        if (t == 0) *sing = 1;
        piv = FT(1);
      }
      for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
      __syncthreads();
      const int pc = nc - 1 - k;
      for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
        const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
        F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
      }
      if (pc > 0) __syncthreads();
    }

    // Phase 2: U-panel Solve (forward-subst against L11), accumulated in FT.
    for (int k = 1; k < nc; ++k) {
      for (int e = t; e < uc; e += nt) {
        const int jj = nc + e;
        FT v = F[(long)k * fsz + jj];
        for (int i = 0; i < k; ++i)
          v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
        F[(long)k * fsz + jj] = v;
      }
      __syncthreads();
    }

    // Phase 3: rank-nc trailing update over the uc×uc contribution block.
    for (int e = t; e < uc * uc; e += nt) {
      const int ii = nc + e / uc, jj = nc + e % uc;
      FT acc = 0;
      for (int k = 0; k < nc; ++k)
        acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
      F[(long)ii * fsz + jj] -= acc;
    }
  }

  // Phase 4: extend-add the contribution block into the parent (race-free
  // atomicAdd — parent is a strictly higher, not-yet-factored level).
  const int par = panel_parent[p];
  if (par < 0 || !do_extend) return;
  __syncthreads();
  FT* Fp = front + front_off[par];
  const int pfsz = front_ptr[par + 1] - front_ptr[par];
  const int abase = asm_ptr[p];
  for (int e = t; e < uc * uc; e += nt) {
    const int a = e / uc, b = e % uc;
    atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
              F[(long)(nc + a) * fsz + (nc + b)]);
  }
}

// Multi-block big-front triple. The few big fronts near the root sit on the
// sequential critical path; one block each under-uses the SMs, so the panel/U-
// panel run one block/front while the embarrassingly-parallel trailing + extend
// spread many blocks. Templated on the front type so the FP32 single-system
// path gets multi-block big fronts too.
template <typename FT>
__global__ void FactorSingleBigPanel(int lbegin, int lend,
                                     const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ ncols, FT* front,
                                     int* sing) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  FT* F = front + front_off[p];
  const int t = threadIdx.x, nt = blockDim.x;
  const int uc = fsz - nc;

  // Phase 1: panel LU on the nc-wide pivot panel.
  for (int k = 0; k < nc; ++k) {
    FT piv = F[(long)k * fsz + k];
    if (piv == FT(0)) {
      if (t == 0) *sing = 1;
      piv = FT(1);
    }
    for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
    __syncthreads();
    const int pc = nc - 1 - k;
    for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
      const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
      F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
    }
    if (pc > 0) __syncthreads();
  }

  // Phase 2: U-panel Solve (forward-subst against L11).
  for (int k = 1; k < nc; ++k) {
    for (int e = t; e < uc; e += nt) {
      const int jj = nc + e;
      FT v = F[(long)k * fsz + jj];
      for (int i = 0; i < k; ++i)
        v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
      F[(long)k * fsz + jj] = v;
    }
    __syncthreads();
  }
}

// Multi-block scalar big-front trailing: each block owns one
// kSingleTrailTile² tile of the uc×uc contribution block. Stages its L rows and
// U cols into shared, then C -= L·U over the nc K-dim. FP64/FP32.
template <typename FT>
__global__ void FactorSingleBigTrail(int lbegin, int lend,
                                     const int* __restrict__ plcols,
                                     const int* __restrict__ front_off,
                                     const int* __restrict__ front_ptr,
                                     const int* __restrict__ ncols, FT* front) {
  const int idx = lbegin + blockIdx.y;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  FT* F = front + front_off[p];
  const int uc = fsz - nc;
  const int ntc = (uc + kSingleTrailTile - 1) / kSingleTrailTile;
  if (ntc <= 0) return;
  const int ti = blockIdx.x / ntc, tj = blockIdx.x % ntc;
  if (ti >= ntc) return;
  extern __shared__ __align__(sizeof(double)) unsigned char smem_bt[];
  FT* sh = reinterpret_cast<FT*>(smem_bt);
  FT* Lsh = sh;
  FT* Ush = sh + kSingleTrailTile * nc;
  const int t = threadIdx.x;

  // Stage this tile's L rows and U cols into shared.
  for (int e = t; e < kSingleTrailTile * nc;
       e += kSingleTrailTile * kSingleTrailTile) {
    const int rr = e / nc, k = e % nc;
    const int ii = nc + ti * kSingleTrailTile + rr;
    Lsh[e] = (ii < fsz) ? F[(long)ii * fsz + k] : FT(0);
  }
  for (int e = t; e < nc * kSingleTrailTile;
       e += kSingleTrailTile * kSingleTrailTile) {
    const int k = e / kSingleTrailTile, cc = e % kSingleTrailTile;
    const int jj = nc + tj * kSingleTrailTile + cc;
    Ush[e] = (jj < fsz) ? F[(long)k * fsz + jj] : FT(0);
  }
  __syncthreads();

  // Dot over nc and subtract from this thread's (ii, jj) trailing entry.
  const int r = t / kSingleTrailTile, c = t % kSingleTrailTile;
  const int ii = nc + ti * kSingleTrailTile + r,
            jj = nc + tj * kSingleTrailTile + c;
  if (ii < fsz && jj < fsz) {
    FT acc = 0;
    for (int k = 0; k < nc; ++k)
      acc += Lsh[r * nc + k] * Ush[k * kSingleTrailTile + c];
    F[(long)ii * fsz + jj] -= acc;
  }
}

// B=1 TF32 big-front trailing: rank-nc update C -= L @ U via shared-staged TF32
// Ozaki mma (m16n8k8). Each block owns one kTcRow x kTcCol output tile (16 x
// 64); the 8 warps each compute one 16x8 N-subtile, all sharing the single
// staged 16xnc L-tile (L read from global once per row tile, not per 16-wide
// column). Multi-block over (row tile, col strip, front). nc<=8 (panel cap) ->
// the K dimension is one 8-wide mma tile. Float front only.
constexpr int kTcRow = 16;  // M (rows per output tile)
constexpr int kTcCol = 64;  // N (cols per output tile) = 8 warps x 8
__global__ void FactorSingleBigTrailTf32(int lbegin, int lend,
                                         const int* __restrict__ plcols,
                                         const int* __restrict__ front_off,
                                         const int* __restrict__ front_ptr,
                                         const int* __restrict__ ncols,
                                         float* front) {
  const int idx = lbegin + blockIdx.y;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  float* F = front + front_off[p];
  const int uc = fsz - nc;
  const int nrt = (uc + kTcRow - 1) / kTcRow;  // row tiles
  const int nct = (uc + kTcCol - 1) / kTcCol;  // col strips
  if (nrt <= 0 || nct <= 0) return;
  const int ti = blockIdx.x / nct, tj = blockIdx.x % nct;
  if (ti >= nrt) return;
  const int row0 = nc + ti * kTcRow;  // first trailing row of this tile
  const int col0 = nc + tj * kTcCol;  // first trailing col of this tile

  // Stage the 16xnc L-tile (shared by all 8 warps) and the ncx64 U-strip.
  __shared__ float Lsh[kTcRow * 8];  // Lsh[r * 8 + k]
  __shared__ float Ush[8 * kTcCol];  // Ush[k * kTcCol + c]
  const int t = threadIdx.x, nt = blockDim.x;
  for (int e = t; e < kTcRow * 8; e += nt) {
    const int r = e >> 3, k = e & 7;  // r in [0,16), k in [0,8)
    const int gr = row0 + r;
    Lsh[e] = (gr < fsz && k < nc) ? F[(long)gr * fsz + k] : 0.0f;
  }
  for (int e = t; e < 8 * kTcCol; e += nt) {
    const int k = e / kTcCol, c = e % kTcCol;  // k in [0,8), c in [0,64)
    const int gc = col0 + c;
    Ush[e] = (k < nc && gc < fsz) ? F[(long)k * fsz + gc] : 0.0f;
  }
  __syncthreads();

  // Tensor-core rank-nc mma: one 8-wide K tile (nc<=8). Ozaki head/tail 4-pass
  // recovers ~FP32; plain single-pass TF32 diverges on these ill-conditioned
  // no-pivot+selinv Jacobians, so the Ozaki passes are required.
  const int warp = t >> 5, lane = t & 31;  // 8 warps, one N-subtile each
  const int laneR = lane >> 2, laneC = (lane & 3) * 2;
  const int r_top = laneR, r_bot = laneR + 8;  // A fragment rows
  const int ncol = warp * 8;                   // this warp's N-subtile base col
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  const Tf32Pair a0 = Tf32OzakiPair(Lsh[r_top * 8 + laneC]);
  const Tf32Pair a1 = Tf32OzakiPair(Lsh[r_bot * 8 + laneC]);
  const Tf32Pair a2 = Tf32OzakiPair(Lsh[r_top * 8 + laneC + 1]);
  const Tf32Pair a3 = Tf32OzakiPair(Lsh[r_bot * 8 + laneC + 1]);
  const Tf32Pair b0 = Tf32OzakiPair(Ush[laneC * kTcCol + ncol + laneR]);
  const Tf32Pair b1 = Tf32OzakiPair(Ush[(laneC + 1) * kTcCol + ncol + laneR]);
  CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

  // Drain C[r][ncol+laneC(+1)] -= c (mma C-fragment layout: laneC selects col).
  const int oc0 = ncol + laneC, oc1 = oc0 + 1;
  auto drain = [&](int rr, int cc, float v) {
    const int gr = row0 + rr, gc = col0 + cc;
    if (gr < fsz && gc < fsz) F[(long)gr * fsz + gc] -= v;
  };
  drain(r_top, oc0, c0);
  drain(r_top, oc1, c1);
  drain(r_bot, oc0, c2);
  drain(r_bot, oc1, c3);
}

// Multi-block extend-add: scatter the uc×uc contribution block into the parent
// front via atomicAdd, grid-strided across blocks.
template <typename FT>
__global__ void FactorSingleBigExtend(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ panel_parent,
    const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
    FT* front) {
  const int idx = lbegin + blockIdx.y;
  if (idx >= lend) return;
  const int p = plcols[idx];
  const int par = panel_parent[p];
  if (par < 0) return;
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  const int uc = fsz - nc;
  FT* F = front + front_off[p];
  FT* Fp = front + front_off[par];
  const int pfsz = front_ptr[par + 1] - front_ptr[par];
  const int abase = asm_ptr[p];
  const int t = threadIdx.x, nt = blockDim.x;
  for (long e = (long)blockIdx.x * nt + t; e < (long)uc * uc;
       e += (long)gridDim.x * nt) {
    const int a = e / uc, b = e % uc;
    atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
              F[(long)(nc + a) * fsz + (nc + b)]);
  }
}

// Partitioned-inverse: invert each front's nc x nc pivot block in place (U_pp
// upper incl diag, L_pp unit-lower strict-lower). The single-system Solve then
// uses parallel GEMVs instead of sequential triangular solves. One block per
// front, one thread per inverse column.
template <typename FT>
__global__ void FactorSingleInvertPivot(int npanels,
                                        const int* __restrict__ front_ptr,
                                        const int* __restrict__ front_off,
                                        const int* __restrict__ ncols,
                                        FT* front) {
  const int p = blockIdx.x;
  if (p >= npanels) return;
  const int s = front_ptr[p];
  const int fsz = front_ptr[p + 1] - s;
  const int nc = ncols[p];
  FT* F = front + front_off[p];
  const int j = threadIdx.x;
  __shared__ FT Ui[kSingleRegNc * kSingleRegNc];
  __shared__ FT Li[kSingleRegNc * kSingleRegNc];
  if (j < nc) {
    Ui[j * nc + j] = FT(1) / F[(long)j * fsz + j];
    for (int i = j - 1; i >= 0; --i) {
      FT v = 0;
      for (int k = i + 1; k <= j; ++k)
        v -= F[(long)i * fsz + k] * Ui[k * nc + j];
      Ui[i * nc + j] = v / F[(long)i * fsz + i];
    }
    Li[j * nc + j] = FT(1);
    for (int i = j + 1; i < nc; ++i) {
      FT v = 0;
      for (int k = j; k < i; ++k) v -= F[(long)i * fsz + k] * Li[k * nc + j];
      Li[i * nc + j] = v;
    }
  }
  __syncthreads();
  if (j < nc) {
    for (int i = 0; i <= j; ++i) F[(long)i * fsz + j] = Ui[i * nc + j];
    for (int i = j + 1; i < nc; ++i) F[(long)i * fsz + j] = Li[i * nc + j];
  }
}

// Per-level factor block size (gpu_mf level_ft): big serial fronts want more
// threads, the many tiny Leaf fronts want small blocks for occupancy, the
// medium hotspot wants 384.
static int SingleFactorBlockSize(const MultifrontalPlan& plan, int L) {
  const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
  const int cnt = e - b;
  int mx = 0;
  for (int q = b; q < e; ++q) {
    const int p = plan.h_plcols[q];
    mx = std::max(mx, plan.h_front_ptr[p + 1] - plan.h_front_ptr[p]);
  }
  if (mx >= 257) return 768;
  if (mx <= 48 && cnt >= 1024) return 128;
  return 384;
}

// Partitioned-inverse pivot blocks (selinv): the single-system B=1 default for
// every precision. The Solve (Solve/single.cuh) reads the inverted blocks and
// runs parallel GEMVs. Dispatched on the active front precision. Shared with
// the FP32/TF32 B=1 path, which factors with the tensor-core whole-front
// kernels (Factorize/mid.cuh) and then inverts here.
static void IssueFactorSingleInvert(const MultifrontalPlan& plan, State& st,
                                    cudaStream_t stream) {
  if (IsFp32Front(st.precision))
    FactorSingleInvertPivot<float><<<plan.num_panels, 32, 0, stream>>>(
        plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols,
        st.d_front_batch_f);
  else
    FactorSingleInvertPivot<double><<<plan.num_panels, 32, 0, stream>>>(
        plan.num_panels, plan.d_front_ptr, plan.d_front_off, plan.d_ncols,
        st.d_front_batch);
}

// FP64 fused single-system factor levels (no selinv — schedule.cuh inverts
// uniformly afterward).
template <typename FT, bool UseTC = false>
static void IssueFactorSingleLevelsTyped(const MultifrontalPlan& plan,
                                         State& st, FT* front,
                                         cudaStream_t stream) {
  constexpr int do_extend = 1;
  // All precisions: big fronts (mxf >= mb_thresh) take the multi-block big-front
  // triple (panel one block/front, trailing + extend many blocks); one block per
  // big front would collapse occupancy. For TF32 (UseTC) the big trailing is the
  // dedicated FactorSingleBigTrailTf32 (shared-staged mma); FP64/FP32 use the
  // scalar FactorSingleBigTrail. Smaller fronts use the scalar FactorSingleLevel
  // (TF32 mma does not pay off on tiny fronts).
  const bool bigmulti = true;
  const int mb_thresh = 81;
  for (int L = 0; L < plan.num_plevels; ++L) {
    const int b = plan.panel_level_ptr[L], e = plan.panel_level_ptr[L + 1];
    if (e <= b) continue;
    const int T = SingleFactorBlockSize(plan, L);
    int mxf = 0, mxnc = 1;
    for (int q = b; q < e; ++q) {
      const int p = plan.h_plcols[q];
      mxf = std::max(mxf, plan.h_front_ptr[p + 1] - plan.h_front_ptr[p]);
      mxnc = std::max(mxnc, plan.h_ncols[p]);
    }
    if (bigmulti && mxf >= mb_thresh) {
      FactorSingleBigPanel<FT><<<e - b, T, 0, stream>>>(
          b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
          front, st.d_sing);
      const int mxtc = (mxf + kSingleTrailTile - 1) / kSingleTrailTile;
      if constexpr (UseTC) {
        // TF32: dedicated shared-staged mma big trailing. One block per (16x64
        // output tile, front), 256 threads = 8 warps (one N-subtile each).
        const int mxrt = (mxf + kTcRow - 1) / kTcRow;
        const int mxct = (mxf + kTcCol - 1) / kTcCol;
        FactorSingleBigTrailTf32<<<dim3(mxrt * mxct, e - b), 256, 0, stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, reinterpret_cast<float*>(front));
      } else {
        const size_t trail_sh = (size_t)2 * kSingleTrailTile * mxnc * sizeof(FT);
        FactorSingleBigTrail<FT><<<dim3(mxtc * mxtc, e - b),
                                   kSingleTrailTile * kSingleTrailTile, trail_sh,
                                   stream>>>(
            b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr,
            plan.d_ncols, front);
      }
      const int xtiles = 16;
      FactorSingleBigExtend<FT><<<dim3(xtiles, e - b), T, 0, stream>>>(
          b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
          plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, front);
    } else {
      FactorSingleLevel<FT><<<e - b, T, 0, stream>>>(
          b, e, plan.d_plcols, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
          plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, front,
          st.d_sing, do_extend);
    }
  }
}

// Fused single-system factor levels, no selinv — schedule.cuh inverts afterward.
// FP64/FP32 use the scalar FactorSingleLevel + the FT big-front triple; TF32 uses
// the same path but the big-front trailing runs on the Tensor Cores (the
// dedicated FactorSingleBigTrailTf32, selected by the UseTC template argument).
static void IssueFactorSingleLevels(const MultifrontalPlan& plan, State& st,
                                    cudaStream_t stream) {
  if (st.precision == Precision::TF32)
    IssueFactorSingleLevelsTyped<float, true>(plan, st, st.d_front_batch_f,
                                              stream);
  else if (IsFp32Front(st.precision))
    IssueFactorSingleLevelsTyped<float, false>(plan, st, st.d_front_batch_f,
                                               stream);
  else
    IssueFactorSingleLevelsTyped<double, false>(plan, st, st.d_front_batch,
                                                stream);
}

}  // namespace
}  // namespace custom_linear_solver
