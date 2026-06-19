#pragma once

// FACTORIZE — SMALL tier (max_fsz <= kSmallFrontMax = warp). One sub-group of
// lanes owns a whole front; sub-groups pack into warps. Kernel + launch +
// dispatch, all here.

#include "factorize/front_ops.cuh"

namespace custom_linear_solver {

using custom_linear_solver::plan::MultifrontalPlan;

namespace {

// Fused panel LU + trailing update for one front, run by a single sub-group.
// SG = sub-group lane count (8 / 16 / 32); SG=32 is the classic one-warp-per-
// front form. `sl` is the lane within the sub-group (0..SG-1) and `mask` its
// active-lane mask. Packing small fronts (fsz <= SG) into SG-lane groups keeps
// every lane busy instead of idling 32-fsz of a warp, raising memory-level
// parallelism on this latency-bound tier.
template <typename FT, int SG>
__device__ __forceinline__ void LuSmallWarp(FT* F, int fsz, int nc, int sl,
                                            unsigned mask, int* sing,
                                            bool static_pivoting,
                                            double pivot_threshold,
                                            double pivot_shift) {
  // Right-looking LU: for each pivot column scale L below the diagonal, then
  // rank-1 update the trailing submatrix.
  for (int k = 0; k < nc; ++k) {
    const long diag = (long)k * fsz + k;
    FT piv = GuardedPivot(F[diag], static_pivoting, pivot_threshold,
                          pivot_shift, sing, sl == 0);
    if (sl == 0) F[diag] = piv;
    for (int i = k + 1 + sl; i < fsz; i += SG) F[(long)i * fsz + k] /= piv;
    __syncwarp(mask);
    const int m = fsz - k - 1;
    for (int e = sl; e < m * m; e += SG) {
      const int ii = k + 1 + e / m, jj = k + 1 + e % m;
      F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
    }
    __syncwarp(mask);
  }
}

// SMALL tier — one sub-group per (front, batch); warps packed per block.
//
// Used when the level's max_fsz <= kSmallFrontMax. Leaf fronts are small
// (fsz <~ 30, nc <~ 8) but numerous, so a full block per front would idle most
// threads and pay a __syncthreads on every rank-1 step. Instead each sub-group
// factors a whole front with __syncwarp, and 32/sub_group_size sub-groups pack
// per warp (kSmallTierWarpsPerBlock warps per block) to amortise launch cost
// and overlap independent fronts' memory traffic on this latency-bound tier.
//
// Per-sub-group flow (numbered in the kernel body):
//   1. copy the fsz×fsz front from global F into shared scratch Fs.
//   2. LuSmallWarp: fused panel LU + trailing on Fs.
//   3. writeback the factored L|U panel to global F (the uc×uc CB stays in Fs).
//   4. ExtendAdd: scatter the CB from shared into the parent front via
//      atomicAdd.
template <typename FrontType, int sub_group_size>
__global__ void FactorSmall(
    int lbegin, int level_size, int B, int front_area,
    const int* __restrict__ plcols, const int* __restrict__ front_off,
    const int* __restrict__ front_ptr, const int* __restrict__ ncols,
    const int* __restrict__ panel_parent, const int* __restrict__ asm_ptr,
    const int* __restrict__ asm_local, FrontType* frontB, long front_total,
    int* sing, int do_extend, bool static_pivoting, double pivot_threshold,
    double pivot_shift) {
  constexpr int fronts_per_warp = 32 / sub_group_size;  // fronts per warp
  extern __shared__ unsigned char smem_sw_raw[];
  FrontType* smem_sw = reinterpret_cast<FrontType*>(smem_sw_raw);

  // Sub-group identity: which front this group of sub_group_size lanes owns,
  // and the lane within it.
  const int warp_in_blk = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sg =
      lane / sub_group_size;  // sub-group id in warp (0..fronts_per_warp-1)
  const int sl =
      lane % sub_group_size;  // lane within sub-group (0..sub_group_size-1)
  const unsigned mask = (sub_group_size == 32) ? 0xffffffffu
                                               : (((1u << sub_group_size) - 1u)
                                                  << (sg * sub_group_size));

  const int warps_per_blk = blockDim.x >> 5;
  const int warp_global = blockIdx.x * warps_per_blk + warp_in_blk;
  const int slot =
      warp_global * fronts_per_warp + sg;  // global (front, batch) index
  if (slot >= level_size * B) return;      // whole sub-group exits together
  const int front_local = slot % level_size;
  const int batch_idx = slot / level_size;

  // Locate the front buffer for (batch batch_idx, panel p).
  FrontType* front = frontB + (long)batch_idx * front_total;
  const int p = plcols[lbegin + front_local];
  const int fsz = front_ptr[p + 1] - front_ptr[p];
  const int nc = ncols[p];
  const int uc = fsz - nc;
  const int fsz2 = fsz * fsz;
  FrontType* F = front + front_off[p];

  // Per-sub-group shared scratch (slot front_area reserved per sub-group).
  FrontType* Fs =
      smem_sw + (long)(warp_in_blk * fronts_per_warp + sg) * front_area;

  // 1. global F → per-sub-group Fs.
  for (int e = sl; e < fsz2; e += sub_group_size) Fs[e] = F[e];
  __syncwarp(mask);

  // 2. fused panel LU + trailing on Fs.
  LuSmallWarp<FrontType, sub_group_size>(Fs, fsz, nc, sl, mask, sing,
                                         static_pivoting, pivot_threshold,
                                         pivot_shift);
  __syncwarp(mask);

  // 3. writeback factored panel.
  WritebackFactored<FrontType, FrontType>(F, Fs, fsz, nc, uc, sl,
                                          sub_group_size);

  // 4. CB extend-add into parent front (skip for roots / when disabled).
  const int par = panel_parent[p];
  if (par < 0 || !do_extend) return;
  FrontType* Fp = front + front_off[par];
  const int pfsz = front_ptr[par + 1] - front_ptr[par];
  const int abase = asm_ptr[p];
  for (int e = sl; e < uc * uc; e += sub_group_size) {
    const int a = e / uc, b = e % uc;
    atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
              Fs[(long)(nc + a) * fsz + (nc + b)]);
  }
}

// Pick the small-tier sub-group size sg in {8,16,32} from max_fsz (mirrors
// Solve/dispatch.cuh SolveSmallSg). Pack (sg<32) only while the packed grid
// still saturates the GPU; otherwise fall back to sg=32 to maximise the grid.
static int FactorSmallSg(int max_fsz, long warps_unpacked) {
  int sg = (max_fsz <= 8) ? 8 : (max_fsz <= 16 ? 16 : 32);
  // Packed warp count = (level_size*B) / (32/sg); revert to sg=32 if that
  // under-fills.
  if (!FactorSaturates(warps_unpacked / (32 / sg))) sg = 32;
  return sg;
}

// Launch helper: instantiates FactorSmall for a concrete (front type,
// sub-group size). Keeps the runtime sub_group_size switch in
// IssueFactorLevelRange compact.
template <typename FrontType, int sub_group_size>
static inline void LaunchFactorSmall(
    int num_blocks, int threads_per_block, size_t shared_bytes,
    cudaStream_t stream, int b, int level_size, int B, int front_area,
    const MultifrontalPlan& plan, const int* d_plc, FrontType* frontB,
    int* sing, int do_extend, bool static_pivoting, double pivot_threshold,
    double pivot_shift) {
  FactorSmall<FrontType, sub_group_size>
      <<<num_blocks, threads_per_block, shared_bytes, stream>>>(
          b, level_size, B, front_area, d_plc, plan.d_front_off,
          plan.d_front_ptr, plan.d_ncols, plan.d_panel_parent, plan.d_asm_ptr,
          plan.d_asm_local, frontB, plan.front_total, sing, do_extend,
          static_pivoting, pivot_threshold, pivot_shift);
}

// Launch the small kernel with (front type fixed, sub_group_size resolved at
// the call site).
template <typename FrontType>
static inline void LaunchFactorSmallT(
    int sub_group_size, int num_blocks, int threads_per_block,
    size_t shared_bytes, cudaStream_t stream, int b, int level_size, int B,
    int front_area, const MultifrontalPlan& plan, const int* d_plc,
    FrontType* frontB, int* sing, int do_extend, bool static_pivoting,
    double pivot_threshold, double pivot_shift) {
  if (sub_group_size == 8)
    LaunchFactorSmall<FrontType, 8>(
        num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B,
        front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting,
        pivot_threshold, pivot_shift);
  else if (sub_group_size == 16)
    LaunchFactorSmall<FrontType, 16>(
        num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B,
        front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting,
        pivot_threshold, pivot_shift);
  else
    LaunchFactorSmall<FrontType, 32>(
        num_blocks, threads_per_block, shared_bytes, stream, b, level_size, B,
        front_area, plan, d_plc, frontB, sing, do_extend, static_pivoting,
        pivot_threshold, pivot_shift);
}

// SMALL tier dispatch: resolve the sub-group size from max_fsz, compute the
// packed grid and per-warp shared budget, and launch FactorSmall for the active
// precision. sub_group_size is a launch-config choice only; the tier itself is
// fixed by front size.
static void DispatchFactorSmall(const MultifrontalPlan& plan, State& st,
                                cudaStream_t stream, int b, int e,
                                const int* d_plc, const FrontRangeCaps& caps) {
  const Precision precision = st.precision;
  const int B = st.batch_count;
  const int level_size = e - b;
  constexpr int do_extend = kFactorDoExtend;

  const long warps_unpacked =
      (long)level_size * B;  // unpacked (one-warp-per-front) count
  const int sub_group_size = FactorSmallSg(caps.max_fsz, warps_unpacked);
  const int fronts_per_warp = kWarpSize / sub_group_size;
  const int warps_per_block = kSmallTierWarpsPerBlock;
  const int threads_per_block = warps_per_block * kWarpSize;
  const int num_blocks =
      (int)(((warps_unpacked + fronts_per_warp - 1) / fronts_per_warp +
             warps_per_block - 1) /
            warps_per_block);
  const int front_area = caps.max_fsz * caps.max_fsz;
  const size_t element_bytes =
      (precision == Precision::FP64) ? sizeof(double) : sizeof(float);
  const size_t shared_bytes =
      (size_t)warps_per_block * fronts_per_warp * front_area * element_bytes;
  if (precision == Precision::FP64)
    LaunchFactorSmallT<double>(
        sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b,
        level_size, B, front_area, plan, d_plc, st.d_front_batch, st.d_sing,
        do_extend, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
  else
    LaunchFactorSmallT<float>(
        sub_group_size, num_blocks, threads_per_block, shared_bytes, stream, b,
        level_size, B, front_area, plan, d_plc, st.d_front_batch_f, st.d_sing,
        do_extend, st.static_pivoting, st.pivot_threshold, st.pivot_shift);
}

}  // namespace
}  // namespace custom_linear_solver
