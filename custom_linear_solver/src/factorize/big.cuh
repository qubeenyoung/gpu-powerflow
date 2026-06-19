#pragma once

// FACTORIZE — BIG tier (front too large for shared; stays global-resident).
// Stages only the L/U panels, drains the CB into the parent (fused). TF32 uses
// the mma trailing. Kernel + staged/TC trailing + multi-block dispatch.

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Multi-block TF32 tensor-core big-tier trailing: the contribution block
// C(uc×uc) -= L21 @ U12 via shared-staged TF32 Ozaki mma (m16n8k8), fused into
// the parent. Slots in as launch 3 of the big-tier triple for TF32 —
// FactorBigPivot/FactorBigPanels (multi-block) factor the panel exactly as for
// FP32/FP64, leaving L/U in global; this kernel runs the tensor-core trailing.
//
// Grid (tile, front, batch): one block owns one kBigTcM×kBigTcN (16×64) output
// tile of one front's contribution block, so the trailing fans out across the
// whole GPU even when a level holds only a few large separators — the same
// per-tile multi-block split the scalar FactorBigTrail uses, now on the tensor
// cores. 256 threads = 8 warps, each warp one 16×8 N-subtile (laneR the row,
// laneC the mma C-fragment column pair). K (= nc) is swept in 8-wide mma tiles
// and Ozaki head/tail 4-pass recovers ~FP32 from TF32. nc <=
// kTensorCorePivotColumnCap (DispatchFactorBig gate) bounds the staged K; uc is
// unbounded (the per-tile split removes the whole-front staging limit).
// nc_pad_max = roundUp8(level_max_nc) fixes the shared strides so the launch
// shared size matches across the level; per-front nc only sets how many K-tiles
// run.
constexpr int kBigTcM = 16;  // rows per output tile (mma M)
constexpr int kBigTcN = 64;  // cols per output tile = 8 warps × 8 (mma N)
__global__ void FactorBigTrailTf32(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ panel_parent,
    const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
    float* frontB, long front_total, int do_extend, int level_max_nc) {
  const int fi = lbegin + blockIdx.y;
  if (fi >= lend) return;
  float* front = frontB + (long)blockIdx.z * front_total;
  const int p = plcols[fi];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  const int uc = fsz - nc;
  const int nrt = (uc + kBigTcM - 1) / kBigTcM;  // row tiles
  const int nct = (uc + kBigTcN - 1) / kBigTcN;  // col tiles
  if (nrt <= 0 || nct <= 0) return;
  if (blockIdx.x >= nrt * nct) return;  // surplus tiles for smaller fronts: drop
  const int ti = blockIdx.x / nct, tj = blockIdx.x % nct;
  const int row0 = nc + ti * kBigTcM;  // first contribution row of this tile
  const int col0 = nc + tj * kBigTcN;  // first contribution col of this tile
  float* F = front + front_off[p];

  const int par = panel_parent[p];
  // Big fronts are always fsz > kFusedMidFrontMax, so a present parent fuses the
  // contribution-block drain straight into the trailing (no separate extend).
  const bool fused = (par >= 0 && do_extend && fsz > kFusedMidFrontMax);
  float* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
  const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
  const int abase = asm_ptr[p];

  const int nc_pad_max = ((level_max_nc + 7) / 8) * 8;  // K padding (fixed)
  const int nc_pad = ((nc + 7) / 8) * 8;                // per-front K padding
  const int k_tiles = nc_pad / 8;
  const int LshStride = nc_pad_max + 1;  // +1 pad → conflict-free A-read banks
                                         // (row stride coprime to 32; 8·laneR
                                         // would otherwise alias banks)
  const int UshStride = kBigTcN + 4;     // +4 pad → conflict-free B-read banks

  // Stage this tile's L rows (kBigTcM × nc) and U cols (nc × kBigTcN) into
  // shared, zero-padded past nc and past the front edge.
  extern __shared__ float smem_big_tf32[];
  float* Lsh = smem_big_tf32;                    // [kBigTcM][LshStride]
  float* Ush = Lsh + (long)kBigTcM * LshStride;  // [nc_pad_max][UshStride]
  const int t = threadIdx.x, nt = blockDim.x;
  for (int e = t; e < kBigTcM * nc_pad; e += nt) {
    const int r = e / nc_pad, k = e % nc_pad;
    const int gr = row0 + r;
    Lsh[r * LshStride + k] =
        (gr < fsz && k < nc) ? F[(long)gr * fsz + k] : 0.0f;
  }
  for (int e = t; e < nc_pad * kBigTcN; e += nt) {
    const int k = e / kBigTcN, c = e % kBigTcN;
    const int gc = col0 + c;
    Ush[k * UshStride + c] =
        (k < nc && gc < fsz) ? F[(long)k * fsz + gc] : 0.0f;
  }
  __syncthreads();

  // One warp = one 16×8 N-subtile; sweep K in 8-wide mma tiles (Ozaki 4-pass).
  const int warp = t >> 5, lane = t & 31;
  const int laneR = lane >> 2, laneC = (lane & 3) * 2;
  const int r_top = laneR, r_bot = laneR + 8;  // A-fragment rows in the tile
  const int ncol = warp * 8;                   // this warp's N-subtile base col
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int kc = 0; kc < k_tiles; ++kc) {
    const int kb = kc * 8;
    const Tf32Pair a0 = Tf32OzakiPair(Lsh[r_top * LshStride + kb + laneC]);
    const Tf32Pair a1 = Tf32OzakiPair(Lsh[r_bot * LshStride + kb + laneC]);
    const Tf32Pair a2 = Tf32OzakiPair(Lsh[r_top * LshStride + kb + laneC + 1]);
    const Tf32Pair a3 = Tf32OzakiPair(Lsh[r_bot * LshStride + kb + laneC + 1]);
    const Tf32Pair b0 =
        Tf32OzakiPair(Ush[(kb + laneC) * UshStride + ncol + laneR]);
    const Tf32Pair b1 =
        Tf32OzakiPair(Ush[(kb + laneC + 1) * UshStride + ncol + laneR]);
    CLS_MMA_TF32_OZAKI2(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
  }

  // Drain each accumulator into F (or fused into the parent). Per-lane output
  // positions follow the mma C-fragment layout: rows {r_top, r_bot}, cols
  // {ncol+laneC, ncol+laneC+1}.
  auto drain = [&](int rr, int cc, float v) {
    const int gr = row0 + rr, gc = col0 + cc;
    if (gr >= fsz || gc >= fsz) return;
    const long off = (long)gr * fsz + gc;
    if (fused) {
      const int li = gr - nc, lj = gc - nc;  // uc-local contribution index
      atomicAdd(&Fp[(long)asm_local[abase + li] * pfsz + asm_local[abase + lj]],
                F[off] - v);
    } else {
      F[off] -= v;
    }
  };
  const int oc0 = ncol + laneC, oc1 = oc0 + 1;
  drain(r_top, oc0, c0);
  drain(r_top, oc1, c1);
  drain(r_bot, oc0, c2);
  drain(r_bot, oc1, c3);
}

// Multi-block BIG tier — one front spread across many blocks.
//
//  One block per front is under-fill bound: a level holds only a few large
//  separator fronts, so a single block per front leaves the GPU near-idle while
//  each block grinds a uc²·nc trailing serially. Split the work so the panel and
//  the embarrassingly-parallel trailing fan out across the whole GPU:
//    (1) FactorBigPivot  : factor the nc×nc A11 block only (one block/front).
//    (2) FactorBigPanels : multi-block over uc — L21 = A21·U11⁻¹ (per row) and
//                          U12 = L11⁻¹·A12 (per col). Fills the GPU.
//    (3) FactorBigTrail  : a 3D grid (tile, front, batch) — each block owns one
//        TM×TN tile of the front's uc×uc trailing, reads L/U from global
//        (written by 1–2, ordered by the graph edge between launches), and fuses
//        the contribution straight into the parent. (TF32 instead runs the
//        tensor-core FactorBigTrailTf32 above when the level is TC-eligible.)
//  Power-grid fronts never reach this tier; exercised only by circuit/2D-FEM
//  matrices with large separators.
template <typename T>
__global__ void FactorBigPivot(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, T* frontB, long front_total, int* sing,
    bool static_pivoting, double pivot_threshold, double pivot_shift) {
  const int idx = lbegin + blockIdx.x;
  if (idx >= lend) return;
  T* front = frontB + (long)blockIdx.y * front_total;
  const int p = plcols[idx];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  T* F = front + front_off[p];
  const int t = threadIdx.x, nt = blockDim.x;
  // No-pivot (or static-shift) LU of the nc×nc top-left A11 block only.
  for (int k = 0; k < nc; ++k) {
    const long diag = (long)k * fsz + k;
    T piv = GuardedPivot(F[diag], static_pivoting, pivot_threshold, pivot_shift,
                         sing, t == 0);
    if (t == 0) F[diag] = piv;
    const T inv = T(1) / piv;
    for (int i = k + 1 + t; i < nc; i += nt) {
      const T lik = F[(long)i * fsz + k] * inv;
      F[(long)i * fsz + k] = lik;
      for (int jj = k + 1; jj < nc; ++jj)
        F[(long)i * fsz + jj] -= lik * F[(long)k * fsz + jj];
    }
    __syncthreads();
  }
}

template <typename T, int TILE>
__global__ void FactorBigPanels(int lbegin, int lend,
                                const int* __restrict__ plcols,
                                const int* __restrict__ front_off,
                                const int* __restrict__ front_ptr,
                                const int* __restrict__ ncols, T* frontB,
                                long front_total, int level_max_nc) {
  const int fi = lbegin + blockIdx.y;
  if (fi >= lend) return;
  T* front = frontB + (long)blockIdx.z * front_total;
  const int p = plcols[fi];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  const int uc = fsz - nc;
  const int g =
      blockIdx.x * TILE + threadIdx.x;  // uc index this thread owns (row & col)
  T* F = front + front_off[p];

  // Stage the factored nc×nc pivot block (L11 unit-lower + U11 upper) into
  // shared.
  extern __shared__ char smem_panels[];
  T* P = reinterpret_cast<T*>(smem_panels);  // nc × nc, row-major
  for (int e = threadIdx.x; e < nc * nc; e += TILE)
    P[e] = F[(long)(e / nc) * fsz + (e % nc)];
  __syncthreads();
  if (g >= uc) return;

  // L21 row g: forward-subst against U11. L[k] =
  // (A21[k]-Σ_{j<k}L[j]·U11[j][k])/U11[k][k].
  T* Lrow = F + (long)(nc + g) * fsz;  // F[(nc+g), 0..nc)
  for (int k = 0; k < nc; ++k) {
    T s = Lrow[k];
    for (int j = 0; j < k; ++j) s -= Lrow[j] * P[(long)j * nc + k];
    Lrow[k] = s / P[(long)k * nc + k];
  }

  // U12 col g: forward-subst against L11 (unit lower). U[k] =
  // A12[k]-Σ_{i<k}L11[k][i]·U[i].
  for (int k = 0; k < nc; ++k) {
    T s = F[(long)k * fsz + (nc + g)];
    for (int i = 0; i < k; ++i)
      s -= P[(long)k * nc + i] * F[(long)i * fsz + (nc + g)];
    F[(long)k * fsz + (nc + g)] = s;
  }
  (void)level_max_nc;
}

template <typename T, int TM, int TN>
__global__ void FactorBigTrail(
    int lbegin, int lend, const int* __restrict__ plcols,
    const int* __restrict__ front_off, const int* __restrict__ front_ptr,
    const int* __restrict__ ncols, const int* __restrict__ panel_parent,
    const int* __restrict__ asm_ptr, const int* __restrict__ asm_local,
    T* frontB, long front_total, int do_extend, int level_max_nc) {
  const int fi = lbegin + blockIdx.y;
  if (fi >= lend) return;
  T* front = frontB + (long)blockIdx.z * front_total;
  const int p = plcols[fi];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  const int uc = fsz - nc;
  const int ntx = (uc + TM - 1) / TM, nty = (uc + TN - 1) / TN;
  if (blockIdx.x >= ntx * nty) return;  // extra tiles for smaller fronts: drop
  const int ti = blockIdx.x / nty, tj = blockIdx.x % nty;
  const int i0 = ti * TM, j0 = tj * TN;
  T* F = front + front_off[p];
  const int par = panel_parent[p];
  const bool fused = (par >= 0 && do_extend && fsz > kFusedMidFrontMax);
  T* Fp = (par >= 0) ? (front + front_off[par]) : nullptr;
  const int pfsz = (par >= 0) ? (front_ptr[par + 1] - front_ptr[par]) : 0;
  const int abase = asm_ptr[p];
  const int t = threadIdx.x, nt = blockDim.x;

  // Stage this tile's L rows (TM×nc) and U cols (nc×TN) into shared for reuse
  // across the nc dot.
  extern __shared__ char smem_trail[];
  T* sL = reinterpret_cast<T*>(smem_trail);  // TM × nc
  T* sU = sL + (long)TM * level_max_nc;      // nc × TN
  for (int e = t; e < TM * nc; e += nt) {
    const int i = e / nc, k = e % nc, gi = i0 + i;
    sL[i * nc + k] = (gi < uc) ? F[(long)(nc + gi) * fsz + k] : T(0);
  }
  for (int e = t; e < nc * TN; e += nt) {
    const int k = e / TN, j = e % TN, gj = j0 + j;
    sU[k * TN + j] = (gj < uc) ? F[(long)k * fsz + (nc + gj)] : T(0);
  }
  __syncthreads();

  // Dot over nc and drain each (gi, gj) — fused into the parent or in place.
  for (int e = t; e < TM * TN; e += nt) {
    const int i = e / TN, j = e % TN, gi = i0 + i, gj = j0 + j;
    if (gi >= uc || gj >= uc) continue;
    T acc = T(0);
    for (int k = 0; k < nc; ++k) acc += sL[i * nc + k] * sU[k * TN + j];
    const long off = (long)(nc + gi) * fsz + (nc + gj);
    if (fused) {
      atomicAdd(&Fp[(long)asm_local[abase + gi] * pfsz + asm_local[abase + gj]],
                F[off] - acc);
    } else {
      F[off] -= acc;
    }
  }
}

// BIG tier dispatch (fronts too large to be shared-resident: fsz >
// WholeFrontSharedMax). All precisions run the blocked multi-block triple:
// pivot (nc×nc, per-front) → L21/U12 panels (multi-block) → trailing
// (multi-block), with graph edges ordering the launches. Routed
// deterministically by tier — no occupancy/opt-in gate.
static void DispatchFactorBig(const MultifrontalPlan& plan, State& st,
                              cudaStream_t stream, int b, int e,
                              const int* d_plc, const int* h_plc,
                              const FrontRangeCaps& caps) {
  const auto& [max_fsz, max_uc, level_max_nc, level_max_uc] = caps;
  const Precision precision = st.precision;
  constexpr int do_extend = kFactorDoExtend;
  (void)max_fsz;
  (void)h_plc;
  dim3 grid(e - b, st.batch_count);

  // Only the small pivot block is per-front; everything else fills the GPU. The
  // uc index space uses level_max_uc (unclamped).
  constexpr int PT = 256;  // panel-Solve tile width
  const int p_ntx = (level_max_uc + PT - 1) / PT;
  const dim3 pgrid(p_ntx > 0 ? p_ntx : 1, e - b, st.batch_count);
  if (precision == Precision::FP64) {
    // Launch 1: per-front nc×nc pivot LU.
    FactorBigPivot<double><<<grid, 64, 0, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        st.d_front_batch, plan.front_total, st.d_sing, st.static_pivoting,
        st.pivot_threshold, st.pivot_shift);

    // Launch 2: multi-block L21/U12 panel solves.
    FactorBigPanels<double, PT>
        <<<pgrid, PT, (size_t)level_max_nc * level_max_nc * sizeof(double),
           stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                     plan.d_ncols, st.d_front_batch, plan.front_total,
                     level_max_nc);
    // Launch 3: multi-block TM×TN trailing + fused extend. grid.x must cover
    // the biggest front's uc² — use level_max_uc (unclamped); max_uc is clamped
    // to the TF32 tile cap and would silently drop tiles → wrong result.
    constexpr int TM = 32, TN = 32;
    const int ntx = (level_max_uc + TM - 1) / TM,
              nty = (level_max_uc + TN - 1) / TN;
    const dim3 tgrid(ntx * nty > 0 ? ntx * nty : 1, e - b, st.batch_count);
    const size_t sh =
        (size_t)(TM + TN) * level_max_nc * sizeof(double);  // ≤ 48KB (nc≤64)
    FactorBigTrail<double, TM, TN><<<tgrid, 256, sh, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch,
        plan.front_total, do_extend, level_max_nc);
    return;
  }
  // FP32 and TF32 share the float-front panel factorization (launches 1–2). The
  // trailing differs: FP32 runs the scalar multi-block FactorBigTrail; TF32 runs
  // the per-tile tensor-core FactorBigTrailTf32 when the level is TC-eligible,
  // else falls back to the same scalar trailing.
  (void)max_uc;

  // Launch 1: per-front nc×nc pivot LU.
  FactorBigPivot<float><<<grid, 64, 0, stream>>>(
      b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
      st.d_front_batch_f, plan.front_total, st.d_sing, st.static_pivoting,
      st.pivot_threshold, st.pivot_shift);

  // Launch 2: multi-block L21/U12 panel solves.
  FactorBigPanels<float, PT>
      <<<pgrid, PT, (size_t)level_max_nc * level_max_nc * sizeof(float),
         stream>>>(b, e, d_plc, plan.d_front_off, plan.d_front_ptr,
                   plan.d_ncols, st.d_front_batch_f, plan.front_total,
                   level_max_nc);

  // Launch 3 (TF32): multi-block per-tile tensor-core trailing. One block per
  // (16×64 output tile, front, batch) — fills the GPU like the scalar
  // FactorBigTrail, with the trailing on the tensor cores. nc is capped so the
  // staged K + per-tile shared stay bounded; uc is unbounded. Any other level
  // (FP32/FP64, or nc past the cap) takes the scalar trailing below.
  //
  // Honest note: on the power-grid target the big tier is K = nc ≈ 8 (one mma
  // K-tile), so the tensor pipe runs ~7% (shared-load bound) and TF32 here is no
  // faster than FP32 — the fronts are simply too small for tensor cores to pay
  // (same reason the small tier skips them). See ../README.md §8.
  if (precision == Precision::TF32 &&
      level_max_nc <= kTensorCorePivotColumnCap) {
    constexpr int M = 16, N = 64;  // == kBigTcM, kBigTcN
    const int nrt = (level_max_uc + M - 1) / M;
    const int nct = (level_max_uc + N - 1) / N;
    const dim3 tcgrid(nrt * nct > 0 ? nrt * nct : 1, e - b, st.batch_count);
    const int nc_pad_max = ((level_max_nc + 7) / 8) * 8;
    const size_t tc_sh =
        ((size_t)M * (nc_pad_max + 1) + (size_t)nc_pad_max * (N + 4)) *
        sizeof(float);
    FactorBigTrailTf32<<<tcgrid, 256, tc_sh, stream>>>(
        b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
        plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local,
        st.d_front_batch_f, plan.front_total, do_extend, level_max_nc);
    return;
  }

  // Launch 3 (FP32, or TF32 level not TC-eligible): multi-block TM×TN scalar
  // trailing + fused extend.
  constexpr int TM = 32, TN = 32;
  const int ntx = (level_max_uc + TM - 1) / TM,
            nty = (level_max_uc + TN - 1) / TN;
  const dim3 tgrid(ntx * nty > 0 ? ntx * nty : 1, e - b, st.batch_count);
  const size_t sh =
      (size_t)(TM + TN) * level_max_nc * sizeof(float);  // ≤ 24KB (nc≤64)
  FactorBigTrail<float, TM, TN><<<tgrid, 256, sh, stream>>>(
      b, e, d_plc, plan.d_front_off, plan.d_front_ptr, plan.d_ncols,
      plan.d_panel_parent, plan.d_asm_ptr, plan.d_asm_local, st.d_front_batch_f,
      plan.front_total, do_extend, level_max_nc);
}

}  // namespace
}  // namespace custom_linear_solver
