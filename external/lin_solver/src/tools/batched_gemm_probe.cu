// GO/NO-GO probe for the batched-GEMM factor rewrite (the only remaining path to the
// large-matrix factor gap). cuDSS's speed on the medium-front bulk comes from
// amalgamating fronts and running the trailing updates as batched dense GEMM. This
// asks the core enabling question on THIS hardware: for representative (amalgamated)
// front sizes, is cublasDgemmBatched substantially faster than our one-block-per-front
// in-kernel rank-nc trailing update? If not, the rewrite cannot win even before the
// fill-increase penalty -> NO-GO. Standalone (cuBLAS + CUDA only), touches nothing.
#include <algorithm>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// One block per front: C(uc x uc) -= L(uc x nc) * U(nc x uc), column-major, like the
// MF factor's Phase-3 trailing update. base[f] points each front's L|U|C slab.
__global__ void inkernel_rank_nc(int uc, int nc, int stride, const double* L, const double* U,
                                 double* C)
{
    const int f = blockIdx.x;
    const double* Lf = L + (long)f * uc * nc;
    const double* Uf = U + (long)f * nc * uc;
    double* Cf = C + (long)f * uc * uc;
    for (int e = threadIdx.x; e < uc * uc; e += blockDim.x) {
        const int i = e % uc, j = e / uc;  // column-major (col j, row i)
        double acc = 0.0;
        for (int k = 0; k < nc; ++k) acc += Lf[(long)k * uc + i] * Uf[(long)j * nc + k];
        Cf[e] -= acc;
    }
    (void)stride;
}

static double time_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

int main()
{
    cublasHandle_t h;
    cublasCreate(&h);
    const int nc = 8;
    const int reps = 50;
    std::printf("batched-GEMM probe: rank-%d trailing update, N fronts, cublasDgemmBatched vs "
                "in-kernel (one block/front, 256 thr)\n",
                nc);
    std::printf("%-6s %-8s | %-14s %-14s %-10s\n", "uc", "N", "cublas_ms", "inkernel_ms", "speedup");
    for (int uc : {16, 32, 64, 128, 200}) {
        // Total dense work ~ constant across sizes: scale N so N*uc*uc ~ 60M elements.
        const long target = 60'000'000;
        const int N = (int)std::max<long>(1, target / ((long)uc * uc));
        std::vector<double> hL((long)N * uc * nc, 0.5), hU((long)N * nc * uc, 0.25),
            hC((long)N * uc * uc, 1.0);
        double *dL, *dU, *dC;
        cudaMalloc(&dL, hL.size() * sizeof(double));
        cudaMalloc(&dU, hU.size() * sizeof(double));
        cudaMalloc(&dC, hC.size() * sizeof(double));
        cudaMemcpy(dL, hL.data(), hL.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dU, hU.data(), hU.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hC.data(), hC.size() * sizeof(double), cudaMemcpyHostToDevice);
        // Pointer arrays for the batched call.
        std::vector<const double*> pL(N), pU(N);
        std::vector<double*> pC(N);
        for (int f = 0; f < N; ++f) {
            pL[f] = dL + (long)f * uc * nc;
            pU[f] = dU + (long)f * nc * uc;
            pC[f] = dC + (long)f * uc * uc;
        }
        const double **dpL, **dpU;
        double** dpC;
        cudaMalloc(&dpL, N * sizeof(double*));
        cudaMalloc(&dpU, N * sizeof(double*));
        cudaMalloc(&dpC, N * sizeof(double*));
        cudaMemcpy(dpL, pL.data(), N * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(dpU, pU.data(), N * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(dpC, pC.data(), N * sizeof(double*), cudaMemcpyHostToDevice);
        const double alpha = -1.0, beta = 1.0;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        // cuBLAS batched (warmup + timed).
        cublasDgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, uc, uc, nc, &alpha, dpL, uc, dpU, nc, &beta,
                           dpC, uc, N);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r)
            cublasDgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, uc, uc, nc, &alpha, dpL, uc, dpU, nc,
                               &beta, dpC, uc, N);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        const double cublas_ms = time_ms(e0, e1) / reps;
        // In-kernel (warmup + timed).
        inkernel_rank_nc<<<N, 256>>>(uc, nc, 0, dL, dU, dC);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) inkernel_rank_nc<<<N, 256>>>(uc, nc, 0, dL, dU, dC);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        const double ink_ms = time_ms(e0, e1) / reps;
        std::printf("%-6d %-8d | %-14.4f %-14.4f %-10.2f\n", uc, N, cublas_ms, ink_ms,
                    ink_ms / cublas_ms);
        cudaFree(dL); cudaFree(dU); cudaFree(dC);
        cudaFree(dpL); cudaFree(dpU); cudaFree(dpC);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    cublasDestroy(h);
    return 0;
}
