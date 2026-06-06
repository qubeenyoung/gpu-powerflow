// Microbench — compare scalar FP32 vs WMMA per-panel vs WMMA packed for many small
// independent trailing-GEMMs of size (uc × uc) -= (uc × nc) * (nc × uc).
//
// Builds standalone. Usage:
//   nvcc -O3 -arch=sm_86 -o /tmp/wmma_pack tests/wmma_pack_microbench.cu
//   /tmp/wmma_pack [uc] [nc] [num_panels]
//
// Reports per-method:
//   - total wall time
//   - effective FLOPS = useful_ops / wall
//   - useful_FMA_count

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(_e)); exit(1);} } while(0)

// ---- (1) Scalar FP32 reference ------------------------------------------------------------
// One block per panel. blockDim.x threads each compute a subset of the uc*uc outputs.
__global__ void trailing_scalar_kernel(int num_panels, int uc, int nc, int ld,
                                        const float* __restrict__ Lall,
                                        const float* __restrict__ Uall,
                                        float* __restrict__ Call)
{
    const int p = blockIdx.x;
    if (p >= num_panels) return;
    const float* L = Lall + (long)p * uc * nc;
    const float* U = Uall + (long)p * nc * uc;
    float* C = Call + (long)p * uc * uc;
    const int t = threadIdx.x, nt = blockDim.x;
    for (int e = t; e < uc * uc; e += nt) {
        const int i = e / uc, j = e % uc;
        float acc = 0.0f;
        for (int k = 0; k < nc; ++k) {
            acc += L[i * nc + k] * U[k * uc + j];
        }
        C[i * uc + j] -= acc;
    }
}

// ---- (2) WMMA per panel (FP16 inputs, FP32 acc) -------------------------------------------
// 1 warp = 1 panel = 1 WMMA mma_sync (16×16×16, K-padded). uc / nc padded to 16 with zeros.
__global__ void trailing_wmma_per_panel_kernel(int num_panels, int uc, int nc,
                                                 const __half* __restrict__ Lall_h,
                                                 const __half* __restrict__ Uall_h,
                                                 float* __restrict__ Call)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_panels) return;
    const __half* L = Lall_h + (long)warp_id * 16 * 16;  // padded 16×16
    const __half* U = Uall_h + (long)warp_id * 16 * 16;
    float* C = Call + (long)warp_id * uc * uc;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    wmma::load_matrix_sync(af, L, 16);
    wmma::load_matrix_sync(bf, U, 16);
    wmma::mma_sync(cf, af, bf, cf);

    __shared__ float Csc[16 * 16];
    wmma::store_matrix_sync(Csc, cf, 16, wmma::mem_row_major);
    __syncwarp();

    // Subtract into C (only the valid uc × uc region)
    const int lane = threadIdx.x & 31;
    for (int e = lane; e < uc * uc; e += 32) {
        const int i = e / uc, j = e % uc;
        C[i * uc + j] -= Csc[i * 16 + j];
    }
}

// ---- (3) WMMA packed: K panels per WMMA tile ----------------------------------------------
// Each WMMA tile processes K_PACK panels stacked along M (and N). Output is block-diagonal —
// off-diagonal blocks of the 16×16 output are wasted compute, but the WMMA call cost is
// amortized over K_PACK panels. Requires (K_PACK * uc <= 16) AND (K_PACK * uc fits in N too).
//
// For uc=4 → pack 4 panels (16 / 4). For uc=8 → pack 2 panels. For uc=16 → 1 panel (no benefit).
__global__ void trailing_wmma_packed_kernel(int num_panels, int uc, int nc, int K_PACK,
                                              const __half* __restrict__ Lall_h,
                                              const __half* __restrict__ Uall_h,
                                              float* __restrict__ Call)
{
    const int group_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int num_groups = (num_panels + K_PACK - 1) / K_PACK;
    if (group_id >= num_groups) return;
    // Each warp processes K_PACK panels packed into one WMMA tile.

    extern __shared__ __half sh[];
    __half* Lp = sh + (threadIdx.x >> 5) * 2 * 16 * 16;   // per-warp scratch for packed L (16×16 padded)
    __half* Up = Lp + 16 * 16;                              // per-warp scratch for packed U

    const int lane = threadIdx.x & 31;
    // Pack K_PACK panels' L into a 16×16 tile: rows [p*uc .. (p+1)*uc) for panel p, cols 0..nc.
    // Rest zero-padded.
    for (int e = lane; e < 16 * 16; e += 32) Lp[e] = __float2half(0.0f);
    for (int e = lane; e < 16 * 16; e += 32) Up[e] = __float2half(0.0f);
    __syncwarp();
    for (int p_idx = 0; p_idx < K_PACK; ++p_idx) {
        const int p = group_id * K_PACK + p_idx;
        if (p >= num_panels) break;
        const __half* L_p = Lall_h + (long)p * 16 * 16;  // padded 16×16 per panel
        const __half* U_p = Uall_h + (long)p * 16 * 16;
        // Pack L_p[uc rows × nc cols] into Lp at rows [p_idx*uc .. +uc), cols [0..nc)
        for (int e = lane; e < uc * nc; e += 32) {
            const int i = e / nc, k = e % nc;
            Lp[(p_idx * uc + i) * 16 + k] = L_p[i * 16 + k];
        }
        // Pack U_p[nc × uc] into Up at rows [0..nc), cols [p_idx*uc .. +uc)
        for (int e = lane; e < nc * uc; e += 32) {
            const int k = e / uc, j = e % uc;
            Up[k * 16 + (p_idx * uc + j)] = U_p[k * 16 + j];
        }
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, Lp, 16);
    wmma::load_matrix_sync(bf, Up, 16);
    wmma::mma_sync(cf, af, bf, cf);

    __shared__ float Csc[16 * 16 * 8];  // 8 warps * 16×16
    float* Csc_w = Csc + (threadIdx.x >> 5) * 16 * 16;
    wmma::store_matrix_sync(Csc_w, cf, 16, wmma::mem_row_major);
    __syncwarp();

    // Distribute diagonal blocks back to each panel's C
    for (int p_idx = 0; p_idx < K_PACK; ++p_idx) {
        const int p = group_id * K_PACK + p_idx;
        if (p >= num_panels) break;
        float* C_p = Call + (long)p * uc * uc;
        for (int e = lane; e < uc * uc; e += 32) {
            const int i = e / uc, j = e % uc;
            // Diagonal block: row [p_idx*uc + i], col [p_idx*uc + j]
            C_p[i * uc + j] -= Csc_w[(p_idx * uc + i) * 16 + (p_idx * uc + j)];
        }
    }
}

// ---------------------------------------------------------------------------------------------
struct Result {
    double wall_us;
    double useful_gflops;
    long long useful_FMAs;
};

double time_kernel(std::function<void(cudaStream_t)> launch, int reps) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    // Warm up
    for (int i = 0; i < 5; ++i) launch(s);
    cudaStreamSynchronize(s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < reps; ++i) launch(s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaStreamDestroy(s);
    return ms * 1000.0 / reps;  // μs per launch
}

int main(int argc, char** argv) {
    int uc = argc > 1 ? atoi(argv[1]) : 4;
    int nc = argc > 2 ? atoi(argv[2]) : 4;
    int num_panels = argc > 3 ? atoi(argv[3]) : 4096;
    int reps = 50;

    printf("\n=== WMMA packing microbench ===\n");
    printf("uc=%d, nc=%d, num_panels=%d, reps=%d\n", uc, nc, num_panels, reps);
    printf("Useful FMAs per panel = uc*uc*nc = %d\n", uc*uc*nc);
    printf("Useful FMAs total     = %lld\n", (long long)num_panels * uc * uc * nc);
    printf("\n");

    // Allocate FP32 L, U, C
    std::vector<float> hL(num_panels * uc * nc), hU(num_panels * nc * uc), hC(num_panels * uc * uc);
    for (auto& v : hL) v = (rand() % 1000) / 500.0f - 1.0f;
    for (auto& v : hU) v = (rand() % 1000) / 500.0f - 1.0f;
    for (auto& v : hC) v = 0.0f;
    float *dL_f, *dU_f, *dC;
    CK(cudaMalloc(&dL_f, hL.size() * sizeof(float)));
    CK(cudaMalloc(&dU_f, hU.size() * sizeof(float)));
    CK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    cudaMemcpy(dL_f, hL.data(), hL.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dU_f, hU.data(), hU.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Also FP16 versions for WMMA, padded to 16×16
    std::vector<__half> hL_h(num_panels * 16 * 16, __float2half(0.0f));
    std::vector<__half> hU_h(num_panels * 16 * 16, __float2half(0.0f));
    for (int p = 0; p < num_panels; ++p) {
        for (int i = 0; i < uc; ++i)
            for (int k = 0; k < nc; ++k)
                hL_h[p * 256 + i * 16 + k] = __float2half(hL[p * uc * nc + i * nc + k]);
        for (int k = 0; k < nc; ++k)
            for (int j = 0; j < uc; ++j)
                hU_h[p * 256 + k * 16 + j] = __float2half(hU[p * nc * uc + k * uc + j]);
    }
    __half *dL_h, *dU_h;
    CK(cudaMalloc(&dL_h, hL_h.size() * sizeof(__half)));
    CK(cudaMalloc(&dU_h, hU_h.size() * sizeof(__half)));
    cudaMemcpy(dL_h, hL_h.data(), hL_h.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dU_h, hU_h.data(), hU_h.size() * sizeof(__half), cudaMemcpyHostToDevice);

    // ---- Method 1: scalar FP32 ----------------------------------------------------------
    {
        cudaMemsetAsync(dC, 0, hC.size() * sizeof(float));
        auto launch = [&](cudaStream_t s) {
            int bs = 64;
            trailing_scalar_kernel<<<num_panels, bs, 0, s>>>(num_panels, uc, nc, uc, dL_f, dU_f, dC);
        };
        double us = time_kernel(launch, reps);
        long long fmas = (long long)num_panels * uc * uc * nc;
        double gflops = fmas / 1e9 / (us / 1e6);
        printf("  (1) Scalar FP32         %8.1f μs   %6.1f GFLOPS useful\n", us, gflops);
    }

    // ---- Method 2: WMMA per panel -------------------------------------------------------
    {
        cudaMemsetAsync(dC, 0, hC.size() * sizeof(float));
        auto launch = [&](cudaStream_t s) {
            int warps_per_block = 8;
            int blk = warps_per_block * 32;
            int gx = (num_panels + warps_per_block - 1) / warps_per_block;
            trailing_wmma_per_panel_kernel<<<gx, blk, 0, s>>>(num_panels, uc, nc, dL_h, dU_h, dC);
        };
        double us = time_kernel(launch, reps);
        long long fmas = (long long)num_panels * uc * uc * nc;
        double gflops = fmas / 1e9 / (us / 1e6);
        // WMMA actual cycles = num_panels × 16*16*16 = 4096 per panel
        long long wmma_fmas = (long long)num_panels * 4096;
        double waste = 1.0 - (double)fmas / wmma_fmas;
        printf("  (2) WMMA per-panel      %8.1f μs   %6.1f GFLOPS useful   (WMMA waste %.0f%%)\n",
               us, gflops, waste * 100);
    }

    // ---- Method 3: WMMA packed K panels per tile ----------------------------------------
    for (int K_PACK : {2, 4}) {
        if (K_PACK * uc > 16) continue;
        cudaMemsetAsync(dC, 0, hC.size() * sizeof(float));
        auto launch = [&](cudaStream_t s) {
            int warps_per_block = 8;
            int blk = warps_per_block * 32;
            int num_groups = (num_panels + K_PACK - 1) / K_PACK;
            int gx = (num_groups + warps_per_block - 1) / warps_per_block;
            size_t shb = warps_per_block * 2 * 16 * 16 * sizeof(__half);
            trailing_wmma_packed_kernel<<<gx, blk, shb, s>>>(num_panels, uc, nc, K_PACK, dL_h, dU_h, dC);
        };
        double us = time_kernel(launch, reps);
        long long fmas = (long long)num_panels * uc * uc * nc;
        double gflops = fmas / 1e9 / (us / 1e6);
        long long wmma_fmas = (long long)((num_panels + K_PACK - 1) / K_PACK) * 4096;
        double waste = 1.0 - (double)fmas / wmma_fmas;
        printf("  (3) WMMA packed K=%d     %8.1f μs   %6.1f GFLOPS useful   (WMMA waste %.0f%%)\n",
               K_PACK, us, gflops, waste * 100);
    }

    cudaFree(dL_f); cudaFree(dU_f); cudaFree(dC);
    cudaFree(dL_h); cudaFree(dU_h);
    return 0;
}
