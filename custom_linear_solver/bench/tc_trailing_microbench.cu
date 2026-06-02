// Isolated batched trailing-update microbench: C(uc x uc) -= L(uc x nc) * U(nc x uc),
// B independent systems (one block per (front,batch)), FP32 FMA vs FP16 WMMA tensor core.
// Answers: at what nc (= GEMM K depth) does the tensor-core trailing beat plain FP32?
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// one block per (front,batch); front stored row-major fsz x fsz (fsz = nc+uc), FP32.
__global__ void fp32_trailing(int B, int fsz, int nc, float* fronts) {
    const int b = blockIdx.y;
    float* F = fronts + (long)b * fsz * fsz;
    const int uc = fsz - nc, t = threadIdx.x, nt = blockDim.x;
    for (int e = t; e < uc * uc; e += nt) {
        const int ii = nc + e / uc, jj = nc + e % uc;
        float acc = 0.f;
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

__global__ void wmma_trailing(int B, int fsz, int nc, float* fronts) {
    const int b = blockIdx.y;
    float* F = fronts + (long)b * fsz * fsz;
    const int uc = fsz - nc, t = threadIdx.x, nt = blockDim.x;
    __shared__ __half Lh[256 * 32];
    __shared__ __half Uh[32 * 256];
    __shared__ float Csc[4 * 256];
    const int UCP = ((uc + 15) / 16) * 16, KP = ((nc + 15) / 16) * 16;
    for (int e = t; e < UCP * KP; e += nt) { int i = e / KP, k = e % KP;
        Lh[e] = (i < uc && k < nc) ? __float2half(F[(long)(nc + i) * fsz + k]) : __float2half(0.f); }
    for (int e = t; e < KP * UCP; e += nt) { int k = e / UCP, j = e % UCP;
        Uh[k * 256 + j] = (k < nc && j < uc) ? __float2half(F[(long)k * fsz + (nc + j)]) : __float2half(0.f); }
    __syncthreads();
    const int ntj = UCP / 16, nks = KP / 16, warp = t >> 5, nwarp = nt >> 5, lane = t & 31;
    for (int ti = warp; ti < ntj; ti += nwarp) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
        for (int kc = 0; kc < nks; ++kc) wmma::load_matrix_sync(af[kc], &Lh[(ti * 16) * KP + kc * 16], KP);
        for (int tj = 0; tj < ntj; ++tj) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
            wmma::fill_fragment(cf, 0.f);
            for (int kc = 0; kc < nks; ++kc) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &Uh[(kc * 16) * 256 + tj * 16], 256);
                wmma::mma_sync(cf, af[kc], bf, cf);
            }
            wmma::store_matrix_sync(&Csc[warp * 256], cf, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) { int r = e >> 4, c = e & 15, ii = ti * 16 + r, jj = tj * 16 + c;
                if (ii < uc && jj < uc) F[(long)(nc + ii) * fsz + (nc + jj)] -= Csc[warp * 256 + e]; }
            __syncwarp();
        }
    }
}

int main() {
    const int B = 64;
    int sizes[][2] = {{4,64},{8,128},{16,128},{32,128},{16,256},{32,256},{8,64},{16,64}};
    printf("%-14s %10s %10s %8s\n", "nc x uc(fsz)", "fp32_us", "wmma_us", "speedup");
    for (auto& s : sizes) {
        int nc = s[0], uc = s[1], fsz = nc + uc;
        std::vector<float> h((long)B * fsz * fsz);
        for (auto& x : h) x = (float)(rand() % 100) / 97.f + 0.5f;
        float* d; cudaMalloc(&d, h.size() * sizeof(float));
        dim3 grid(1, B); int T = 128;
        cudaEvent_t a, bb; cudaEventCreate(&a); cudaEventCreate(&bb);
        int reps = 200;
        // fp32
        cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
        fp32_trailing<<<grid, T>>>(B, fsz, nc, d); cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r) fp32_trailing<<<grid, T>>>(B, fsz, nc, d);
        cudaEventRecord(bb); cudaEventSynchronize(bb);
        float t1; cudaEventElapsedTime(&t1, a, bb);
        // wmma
        cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
        wmma_trailing<<<grid, T>>>(B, fsz, nc, d); cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r) wmma_trailing<<<grid, T>>>(B, fsz, nc, d);
        cudaEventRecord(bb); cudaEventSynchronize(bb);
        float t2; cudaEventElapsedTime(&t2, a, bb);
        char lbl[32]; snprintf(lbl, sizeof lbl, "%dx%d(%d)", nc, uc, fsz);
        printf("%-14s %10.2f %10.2f %7.2fx\n", lbl, t1 / reps * 1000, t2 / reps * 1000, t1 / t2);
        cudaFree(d);
    }
    return 0;
}
