#include "cuda_backend_impl.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>

#include <cmath>
#include <vector>
#include <algorithm>


// ---------------------------------------------------------------------------
// negate_cast_f64_to_f32_kernel
//
// Converts FP64 src → FP32 dst with negation: dst[i] = -(float)src[i].
// Used to build the cuDSS RHS b = -F (Mixed mode) entirely on the GPU.
// ---------------------------------------------------------------------------
__global__ void negate_cast_f64_to_f32_kernel(
    const double* __restrict__ src,
    float*        __restrict__ dst,
    int32_t n
)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = -static_cast<float>(src[i]);
}

void cuda_negate_cast(const double* src, float* dst, int32_t n)
{
    if (n <= 0) return;
    constexpr int32_t block = 256;
    int32_t grid = (n + block - 1) / block;
    negate_cast_f64_to_f32_kernel<<<grid, block>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// negate_f64_kernel
//
// FP64 negate: dst[i] = -src[i]. Used to build cuDSS RHS for FP64 mode.
// ---------------------------------------------------------------------------
__global__ void negate_f64_kernel(
    const double* __restrict__ src,
    double*       __restrict__ dst,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = -src[i];
}

void cuda_negate_f64(const double* src, double* dst, int32_t n)
{
    if (n <= 0) return;
    constexpr int32_t block = 256;
    int32_t grid = (n + block - 1) / block;
    negate_f64_kernel<<<grid, block>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// mismatch_pack_kernel
//
// Computes power mismatch and packs the result into F:
//   mis[bus] = V[bus] * conj(Ibus[bus]) - Sbus[bus]
//   F[0      : n_pv]         = Re(mis[pv])
//   F[n_pv   : n_pv+n_pq]   = Re(mis[pq])
//   F[n_pv+n_pq : dimF]     = Im(mis[pq])
//
// d_pv and d_pq are already on GPU (uploaded in analyze).
// One thread per output element in F.
// ---------------------------------------------------------------------------
__global__ void mismatch_pack_kernel(
    const cuDoubleComplex* __restrict__ V,
    const cuDoubleComplex* __restrict__ Ibus,
    const cuDoubleComplex* __restrict__ Sbus,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq,
    double* __restrict__ F
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) return;

    auto mis = [&](int32_t bus) {
        double Vre = cuCreal(V[bus]),    Vim = cuCimag(V[bus]);
        double Ire = cuCreal(Ibus[bus]), Iim = cuCimag(Ibus[bus]);
        double re  = Vre * Ire + Vim * Iim - cuCreal(Sbus[bus]);
        double im  = Vim * Ire - Vre * Iim - cuCimag(Sbus[bus]);
        return make_cuDoubleComplex(re, im);
    };

    if (tid < n_pv) {
        F[tid] = cuCreal(mis(pv[tid]));
    } else if (tid < n_pv + n_pq) {
        F[tid] = cuCreal(mis(pq[tid - n_pv]));
    } else {
        F[tid] = cuCimag(mis(pq[tid - n_pv - n_pq]));
    }
}


// ---------------------------------------------------------------------------
// max_abs_kernel: shared-memory reduction → max|F_i| per block
// ---------------------------------------------------------------------------
__global__ void max_abs_kernel(const double* __restrict__ F, int32_t n, double* __restrict__ out)
{
    extern __shared__ double sdata[];
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid]  = (gid < n) ? fabs(F[gid]) : 0.0;
    __syncthreads();

    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}


// ---------------------------------------------------------------------------
// computeMismatch
//
// GPU pipeline:
//   1. cuSPARSE SpMV: Ibus = Ybus * V  (FP64 complex)
//   2. mismatch_pack_kernel: pack mis[] into d_F (GPU buffer)
//   3. max_abs_kernel: compute normF = max|F_i|  → host scalar
//
// d_F stays on GPU; solveLinearSystem() reads it directly via cuda_negate_cast.
// F_out is NOT populated — CUDA solveLinearSystem ignores the host F pointer.
// d_pv and d_pq are used directly from GPU — uploaded once in analyze().
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::computeMismatch(
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    double* /*F_out*/, double& normF)
{
    auto& im = *impl_;
    const int32_t dimF  = n_pv + 2 * n_pq;

    // --- Step 1: SpMV Ibus = Ybus * V ---
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    CUSPARSE_CHECK(cusparseSpMV(
        im.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, im.sp_Ybus, im.sp_V, &beta, im.sp_Ibus,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, im.d_spmv_buf));

    // --- Step 2: Pack mismatch into d_F (GPU) ---
    const int32_t block = 256;
    const int32_t grid  = (dimF + block - 1) / block;
    mismatch_pack_kernel<<<grid, block>>>(
        im.d_V_cd, im.d_Ibus, im.d_Sbus,
        im.d_pv, im.d_pq, n_pv, n_pq,
        im.d_F);
    CUDA_CHECK(cudaGetLastError());

    // --- Step 3: normF via block reduction (only scalar downloaded) ---
    const int32_t n_blocks = grid;
    double* d_block_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_max, n_blocks * sizeof(double)));

    max_abs_kernel<<<n_blocks, block, block * sizeof(double)>>>(im.d_F, dimF, d_block_max);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_max(n_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max, n_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_block_max));

    normF = *std::max_element(h_block_max.begin(), h_block_max.end());
    // d_F remains on GPU for solveLinearSystem() — no host download.
}


// ---------------------------------------------------------------------------
// mismatch_pack_kernel_fp32
//
// FP32 variant: computes power mismatch in FP32 and packs into d_b_f with
// negation so d_b_f = -F is ready for cuDSS solve in one pass.
//
// F layout: F[0:n_pv] = Re(mis[pv]), F[n_pv:n_pv+n_pq] = Re(mis[pq]),
//           F[n_pv+n_pq:dimF] = Im(mis[pq])
// ---------------------------------------------------------------------------
__global__ void mismatch_pack_kernel_fp32(
    const cuFloatComplex* __restrict__ V,
    const cuFloatComplex* __restrict__ Ibus,
    const cuFloatComplex* __restrict__ Sbus,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq,
    float* __restrict__ neg_F   // output: -F[i]
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) return;

    auto mis = [&](int32_t bus) {
        float Vre = cuCrealf(V[bus]),    Vim = cuCimagf(V[bus]);
        float Ire = cuCrealf(Ibus[bus]), Iim = cuCimagf(Ibus[bus]);
        float re  = Vre * Ire + Vim * Iim - cuCrealf(Sbus[bus]);
        float im  = Vim * Ire - Vre * Iim - cuCimagf(Sbus[bus]);
        return make_cuFloatComplex(re, im);
    };

    float val;
    if (tid < n_pv) {
        val = cuCrealf(mis(pv[tid]));
    } else if (tid < n_pv + n_pq) {
        val = cuCrealf(mis(pq[tid - n_pv]));
    } else {
        val = cuCimagf(mis(pq[tid - n_pv - n_pq]));
    }
    neg_F[tid] = -val;   // negate: cuDSS needs b = -F
}


// ---------------------------------------------------------------------------
// max_abs_kernel_fp32: shared-memory reduction → max|neg_F_i| per block.
// Since neg_F = -F, |neg_F| = |F|, so normF is correct.
// ---------------------------------------------------------------------------
__global__ void max_abs_kernel_fp32(const float* __restrict__ neg_F, int32_t n, float* __restrict__ out)
{
    extern __shared__ float sdata_f[];
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata_f[tid] = (gid < n) ? fabsf(neg_F[gid]) : 0.0f;
    __syncthreads();

    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_f[tid] = fmaxf(sdata_f[tid], sdata_f[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata_f[0];
}


// ---------------------------------------------------------------------------
// computeMismatch_f32
//
// FP32 GPU pipeline:
//   1. cuSPARSE SpMV (FP32): Ibus_f = Ybus_f * V_f
//   2. mismatch_pack_kernel_fp32: pack and negate into d_b_f (= -F, FP32)
//   3. max_abs_kernel_fp32: normF = max|F_i|
//
// d_b_f stays on GPU; solveLinearSystem_f32 feeds it directly to cuDSS.
// Host F pointer is unused (CUDA keeps everything on GPU).
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::computeMismatch_f32(
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    float* /*F_out*/, float& normF)
{
    auto& im = *impl_;
    const int32_t dimF = n_pv + 2 * n_pq;

    // --- Step 1: SpMV Ibus_f = Ybus_f * V_f (FP32) ---
    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuFloatComplex beta  = make_cuFloatComplex(0.0f, 0.0f);
    CUSPARSE_CHECK(cusparseSpMV(
        im.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, im.sp_Ybus_f, im.sp_V_f, &beta, im.sp_Ibus_f,
        CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, im.d_spmv_buf_f));

    // --- Step 2: Pack mismatch as -F into d_b_f ---
    const int32_t block = 256;
    const int32_t grid  = (dimF + block - 1) / block;
    mismatch_pack_kernel_fp32<<<grid, block>>>(
        im.d_V_cf, im.d_Ibus_f, im.d_Sbus_f,
        im.d_pv, im.d_pq, n_pv, n_pq,
        im.d_b_f);
    CUDA_CHECK(cudaGetLastError());

    // --- Step 3: normF via block reduction ---
    const int32_t n_blocks = grid;
    float* d_block_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_max, n_blocks * sizeof(float)));

    max_abs_kernel_fp32<<<n_blocks, block, block * sizeof(float)>>>(im.d_b_f, dimF, d_block_max);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_block_max(n_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max, n_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_block_max));

    normF = *std::max_element(h_block_max.begin(), h_block_max.end());
    // d_b_f = -F remains on GPU; solveLinearSystem_f32 uses it directly.
}


// ---------------------------------------------------------------------------
// mismatch_pack_batch_kernel
//
// Batch variant: one thread per (b, tid) element, flattened to 1-D grid.
//
// V/Ibus layout: col-major nbus×n_batch → element [bus, b] = buf[b*nbus + bus]
// Sbus layout:   same as V (uploaded in initialize_batch, contiguous [nb*nbus])
// F_batch layout: row-major [b * dimF + tid]
// ---------------------------------------------------------------------------
__global__ void mismatch_pack_batch_kernel(
    const cuDoubleComplex* __restrict__ V,      // [n_batch * n_bus] col-major
    const cuDoubleComplex* __restrict__ Ibus,   // [n_batch * n_bus] col-major
    const cuDoubleComplex* __restrict__ Sbus,   // [n_batch * n_bus] col-major
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq,
    int32_t n_bus,
    int32_t dimF,
    int32_t n_batch,
    double* __restrict__ F_batch              // [n_batch * dimF]
) {
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_batch * dimF) return;  // guard: ceil-grid produces excess threads

    int32_t b   = gid / dimF;
    int32_t tid = gid % dimF;

    // bounds checked by caller's grid size
    const int32_t base_bus = b * n_bus;

    auto mis = [&](int32_t bus) -> cuDoubleComplex {
        const cuDoubleComplex v  = V[base_bus + bus];
        const cuDoubleComplex ib = Ibus[base_bus + bus];
        const cuDoubleComplex sb = Sbus[base_bus + bus];
        double re = cuCreal(v) * cuCreal(ib) + cuCimag(v) * cuCimag(ib) - cuCreal(sb);
        double im = cuCimag(v) * cuCreal(ib) - cuCreal(v) * cuCimag(ib) - cuCimag(sb);
        return make_cuDoubleComplex(re, im);
    };

    double val;
    if (tid < n_pv) {
        val = cuCreal(mis(pv[tid]));
    } else if (tid < n_pv + n_pq) {
        val = cuCreal(mis(pq[tid - n_pv]));
    } else {
        val = cuCimag(mis(pq[tid - n_pv - n_pq]));
    }
    F_batch[b * dimF + tid] = val;
}


// ---------------------------------------------------------------------------
// max_abs_batch_kernel
//
// Per-batch max |F| via shared-memory reduction.
// Each block handles one contiguous segment of length dimF for batch b.
// Grid: (n_batch * n_blocks_per_batch) blocks; each block reduces one chunk.
// ---------------------------------------------------------------------------
__global__ void max_abs_batch_kernel(
    const double* __restrict__ F_batch,  // [n_batch * dimF]
    int32_t dimF,
    int32_t n_batch,
    double* __restrict__ out             // [n_batch * n_blocks_per_batch]
) {
    extern __shared__ double sdata[];

    // Each block is mapped to a (batch, chunk) pair
    const int32_t n_blocks_per_batch = gridDim.x / n_batch;
    const int32_t b       = blockIdx.x / n_blocks_per_batch;
    const int32_t blk_in  = blockIdx.x % n_blocks_per_batch;
    const int32_t gid     = blk_in * blockDim.x + threadIdx.x;

    const double* F = F_batch + (int64_t)b * dimF;
    sdata[threadIdx.x] = (gid < dimF) ? fabs(F[gid]) : 0.0;
    __syncthreads();

    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}


// ---------------------------------------------------------------------------
// computeMismatch_batch
//
// GPU pipeline:
//   1. cuSPARSE SpMM: Ibus_batch = Ybus × V_batch  (FP64 complex, n_batch cols)
//   2. mismatch_pack_batch_kernel: pack mis[] into d_F_batch (GPU)
//   3. max_abs_batch_kernel: normF_batch[b] → host (n_batch scalars only)
//
// d_F_batch stays on GPU; solveLinearSystem_batch() reads it via cuda_negate_cast.
// F_batch_out is NOT populated — CUDA solveLinearSystem_batch ignores host F.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::computeMismatch_batch(
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    double* /*F_batch_out*/,
    double* normF_batch,
    int32_t n_batch)
{
    auto& im = *impl_;
    const int32_t dimF  = im.dimF;
    const int32_t n_bus = im.n_bus;

    // --- Step 1: SpMM Ibus_batch = Ybus × V_batch ---
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    CUSPARSE_CHECK(cusparseSpMM(
        im.sp_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, im.sp_Ybus, im.sp_V_mat, &beta, im.sp_Ibus_mat,
        CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, im.d_spmm_buf
    ));

    // --- Step 2: Pack mismatch into d_F_batch (GPU) ---
    const int32_t block  = 256;
    const int32_t total  = n_batch * dimF;
    const int32_t grid2  = (total + block - 1) / block;
    
    mismatch_pack_batch_kernel<<<grid2, block>>>(
        im.d_V_cd_batch, im.d_Ibus_batch, im.d_Sbus_batch,
        im.d_pv, im.d_pq, n_pv, n_pq, n_bus, dimF, n_batch,
        im.d_F_batch);
    CUDA_CHECK(cudaGetLastError());

    // --- Step 3: Per-batch max|F| (only n_batch scalars downloaded) ---
    const int32_t n_blocks_per_batch = (dimF + block - 1) / block;
    const int32_t grid3 = n_batch * n_blocks_per_batch;
    double* d_block_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_max, grid3 * sizeof(double)));

    max_abs_batch_kernel<<<grid3, block, block * sizeof(double)>>>(
        im.d_F_batch, dimF, n_batch, d_block_max);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_max(grid3);
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max,
                          grid3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_block_max));

    for (int32_t b = 0; b < n_batch; ++b) {
        double mx = 0.0;
        for (int32_t k = 0; k < n_blocks_per_batch; ++k)
            mx = std::max(mx, h_block_max[b * n_blocks_per_batch + k]);
        normF_batch[b] = mx;
    }
    // d_F_batch remains on GPU — no host download.
}
