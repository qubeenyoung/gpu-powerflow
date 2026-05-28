// Decisive probe (cy126): is the dense-front factor work FP64-compute-bound on this
// GPU (RTX 3090: FP64 = 1/64 FP32)? If FP32 is much faster for the rank-nc trailing
// update + panel work, a mixed-precision factor (allowed: cuDSS-level accuracy +
// iterative refinement) is a large, MEASURABLE win. In-process FP64-vs-FP32 A/B after
// warmup -> reliable ratio despite the unpinnable GPU clock. Standalone (CUDA only).
#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

// One block per front: full blocked rank-nc dense no-pivot LU (mirrors the real factor
// kernel's big-front path: panel factor + U panel + rank-nc trailing update), templated
// on the front element type. This is the per-front compute that dominates factor time.
template <typename T>
__global__ void front_lu(int fsz, int nc, int stride, T* fronts)
{
    T* F = fronts + (long)blockIdx.x * stride;
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    for (int k = 0; k < nc; ++k) {  // Phase 1: panel
        T piv = F[(long)k * fsz + k];
        if (piv == T(0)) piv = T(1);
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
    for (int k = 1; k < nc; ++k) {  // Phase 2: U panel
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            T v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
    for (int e = t; e < uc * uc; e += nt) {  // Phase 3: trailing rank-nc update
        const int ii = nc + e / uc, jj = nc + e % uc;
        T acc = T(0);
        for (int k = 0; k < nc; ++k) acc += F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= acc;
    }
}

// MIXED: FP64 storage (stable pivots, no overflow/cancellation-to-zero), but the bulk
// trailing rank-nc update computes in FP32 (the 2x-throughput op on RTX 3090). Tests
// whether FP32 compute alone (without FP32 storage) gives the speedup -> a stable
// mixed-precision factor.
__global__ void front_lu_mixed(int fsz, int nc, int stride, double* fronts)
{
    double* F = fronts + (long)blockIdx.x * stride;
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    for (int k = 0; k < nc; ++k) {  // Phase 1: panel in FP64 (precision-critical)
        double piv = F[(long)k * fsz + k];
        if (piv == 0.0) piv = 1.0;
        for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            F[(long)ii * fsz + jj] -= F[(long)ii * fsz + k] * F[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
    for (int k = 1; k < nc; ++k) {  // Phase 2
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            double v = F[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * F[(long)i * fsz + jj];
            F[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
    for (int e = t; e < uc * uc; e += nt) {  // Phase 3: FP32 compute on FP64 storage
        const int ii = nc + e / uc, jj = nc + e % uc;
        float acc = 0.0f;
        for (int k = 0; k < nc; ++k)
            acc += (float)F[(long)ii * fsz + k] * (float)F[(long)k * fsz + jj];
        F[(long)ii * fsz + jj] -= (double)acc;
    }
}

static double bench_mixed(int fsz, int nc, int N, int reps)
{
    const int stride = fsz * fsz;
    std::vector<double> h((long)N * stride);
    for (long i = 0; i < (long)N * stride; ++i) h[i] = 1.0 + (i % 7) * 0.1;
    double* d;
    cudaMalloc(&d, (long)N * stride * sizeof(double));
    cudaMemcpy(d, h.data(), (long)N * stride * sizeof(double), cudaMemcpyHostToDevice);
    front_lu_mixed<<<N, 512>>>(fsz, nc, stride, d);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) front_lu_mixed<<<N, 512>>>(fsz, nc, stride, d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaFree(d);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

// Shared-memory FP64 LU: load global front -> shared, full LU in shared, write back.
// Tests whether staging the front in shared beats the global LU (i.e. is the dense LU
// global-bandwidth-bound such that shared helps -- the FP64 way to get the bandwidth
// win without FP32's accuracy loss). dynamic shared = fsz*fsz doubles.
__global__ void front_lu_shared(int fsz, int nc, int stride, double* fronts)
{
    extern __shared__ double S[];
    double* F = fronts + (long)blockIdx.x * stride;
    const int t = threadIdx.x, nt = blockDim.x;
    const int uc = fsz - nc;
    for (int e = t; e < fsz * fsz; e += nt) S[e] = F[e];  // load global -> shared
    __syncthreads();
    for (int k = 0; k < nc; ++k) {
        double piv = S[(long)k * fsz + k];
        if (piv == 0.0) piv = 1.0;
        for (int i = k + 1 + t; i < fsz; i += nt) S[(long)i * fsz + k] /= piv;
        __syncthreads();
        const int pc = nc - 1 - k;
        for (int e = t; e < (fsz - k - 1) * pc; e += nt) {
            const int ii = k + 1 + e / pc, jj = k + 1 + e % pc;
            S[(long)ii * fsz + jj] -= S[(long)ii * fsz + k] * S[(long)k * fsz + jj];
        }
        if (pc > 0) __syncthreads();
    }
    for (int k = 1; k < nc; ++k) {
        for (int e = t; e < uc; e += nt) {
            const int jj = nc + e;
            double v = S[(long)k * fsz + jj];
            for (int i = 0; i < k; ++i) v -= S[(long)k * fsz + i] * S[(long)i * fsz + jj];
            S[(long)k * fsz + jj] = v;
        }
        __syncthreads();
    }
    for (int e = t; e < uc * uc; e += nt) {
        const int ii = nc + e / uc, jj = nc + e % uc;
        double acc = 0.0;
        for (int k = 0; k < nc; ++k) acc += S[(long)ii * fsz + k] * S[(long)k * fsz + jj];
        S[(long)ii * fsz + jj] -= acc;
    }
    __syncthreads();
    for (int e = t; e < fsz * fsz; e += nt) F[e] = S[e];  // write back shared -> global
}

static double bench_shared(int fsz, int nc, int N, int reps)
{
    const int stride = fsz * fsz;
    const int shbytes = stride * sizeof(double);
    std::vector<double> h((long)N * stride);
    for (long i = 0; i < (long)N * stride; ++i) h[i] = 1.0 + (i % 7) * 0.1;
    double* d;
    cudaMalloc(&d, (long)N * stride * sizeof(double));
    cudaMemcpy(d, h.data(), (long)N * stride * sizeof(double), cudaMemcpyHostToDevice);
    front_lu_shared<<<N, 512, shbytes>>>(fsz, nc, stride, d);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) front_lu_shared<<<N, 512, shbytes>>>(fsz, nc, stride, d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaFree(d);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

template <typename T>
static double bench(int fsz, int nc, int N, int reps)
{
    const int stride = fsz * fsz;
    std::vector<T> h((long)N * stride);
    for (long i = 0; i < (long)N * stride; ++i) h[i] = T(1) + T((i % 7) * 0.1);
    T* d;
    cudaMalloc(&d, (long)N * stride * sizeof(T));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaMemcpy(d, h.data(), (long)N * stride * sizeof(T), cudaMemcpyHostToDevice);
    front_lu<T><<<N, 512>>>(fsz, nc, stride, d);  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r)  // time the KERNEL only (no per-rep memcpy)
        front_lu<T><<<N, 512>>>(fsz, nc, stride, d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaFree(d);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

int main()
{
    cudaFree(0);  // init context
    const int nc = 8, reps = 30;
    std::printf("FP64-vs-FP32 dense-front LU (one block/front, 512 thr), nc=%d. ratio>1 => FP32 "
                "faster => FP64-compute-bound => mixed-precision factor wins.\n",
                nc);
    std::printf("%-6s %-8s | %-12s %-12s %-8s\n", "fsz", "N", "fp64_ms", "fp32_ms", "speedup");
    for (int fsz : {64, 96, 128, 200}) {
        const long target = 40'000'000;
        const int N = (int)std::max<long>(1, target / ((long)fsz * fsz));
        // Warm the GPU to steady boost before the timed A/B (the clock can't be pinned).
        for (int w = 0; w < 3; ++w) {
            bench<double>(fsz, nc, N, 5);
            bench<float>(fsz, nc, N, 5);
            bench_mixed(fsz, nc, N, 5);
        }
        const double f64 = bench<double>(fsz, nc, N, reps);
        const double f32 = bench<float>(fsz, nc, N, reps);
        const double fmx = bench_mixed(fsz, nc, N, reps);
        std::printf("%-6d %-8d | f64=%-9.4f f32=%-9.4f mixed=%-9.4f | f32 %.2fx, mixed %.2fx\n",
                    fsz, N, f64, f32, fmx, f64 / f32, f64 / fmx);
    }
    std::printf("\nshared-mem FP64 LU vs global FP64 (fronts that fit in 48KB shared). "
                "ratio>1 => shared faster => global-bandwidth-bound => FP64 shared-LU helps.\n");
    std::printf("%-6s %-8s | %-12s %-12s %-8s\n", "fsz", "N", "global_ms", "shared_ms", "speedup");
    for (int fsz : {32, 48, 64}) {
        const long target = 40'000'000;
        const int N = (int)std::max<long>(1, target / ((long)fsz * fsz));
        for (int w = 0; w < 3; ++w) { bench<double>(fsz, nc, N, 5); bench_shared(fsz, nc, N, 5); }
        const double g = bench<double>(fsz, nc, N, reps);
        const double sh = bench_shared(fsz, nc, N, reps);
        std::printf("%-6d %-8d | %-12.4f %-12.4f %-8.2f\n", fsz, N, g, sh, g / sh);
    }
    return 0;
}
