#include "cuda_backend_impl.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>


__device__ inline float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ inline double atomic_add_f64_compat(double* address, double val)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}


// ---------------------------------------------------------------------------
// convert_V_cd_to_f_kernel
//
// Mixed mode: FP64 complex V_cd → FP32 interleaved V_f for Jacobian kernel.
// ---------------------------------------------------------------------------
__global__ void convert_V_cd_to_f_kernel(
    const cuDoubleComplex* __restrict__ V_cd,
    float* __restrict__ V_f,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;
    V_f[i * 2]     = static_cast<float>(cuCreal(V_cd[i]));
    V_f[i * 2 + 1] = static_cast<float>(cuCimag(V_cd[i]));
}


// ---------------------------------------------------------------------------
// convert_V_cf_to_f_kernel
//
// FP32 mode: FP32 complex V_cf → FP32 interleaved V_f for Jacobian kernel.
// ---------------------------------------------------------------------------
__global__ void convert_V_cf_to_f_kernel(
    const cuFloatComplex* __restrict__ V_cf,
    float* __restrict__ V_f,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;
    V_f[i * 2]     = cuCrealf(V_cf[i]);
    V_f[i * 2 + 1] = cuCimagf(V_cf[i]);
}


// ---------------------------------------------------------------------------
// update_jacobian_vertex_kernel_fp32
//
// One warp owns one active bus row i (i in pvpq) and iterates over the Ybus
// CSR row directly. This makes the row owner unique, so every off-diagonal
// Jacobian entry is written exactly once and the diagonal can be accumulated
// in registers before a single final store.
//
// Compared to the edge-based kernel this removes the atomicAdd hot spot while
// keeping the same CSR-indexed JacobianMaps.
// ---------------------------------------------------------------------------
__global__ void update_jacobian_vertex_kernel_fp32(
    int32_t n_active_buses,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const float* __restrict__ V_f,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_csr_values)
{
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane             = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block  = blockDim.x / warp_size;
    const int32_t bus_slot         = blockIdx.x * warps_per_block + warp_id_in_block;

    if (bus_slot >= n_active_buses) return;

    const int32_t i = pvpq[bus_slot];
    const int32_t row_begin = y_row_ptr[i];
    const int32_t row_end   = y_row_ptr[i + 1];

    const cuFloatComplex Vi = make_cuFloatComplex(V_f[i * 2], V_f[i * 2 + 1]);
    const cuFloatComplex neg_j_Vi = make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi));
    const float vi_abs = cuCabsf(Vi);
    const bool have_vi_norm = vi_abs > 1e-6f;
    const cuFloatComplex Vi_norm = have_vi_norm
        ? make_cuFloatComplex(cuCrealf(Vi) / vi_abs, cuCimagf(Vi) / vi_abs)
        : make_cuFloatComplex(0.0f, 0.0f);

    float diag11_acc = 0.0f;
    float diag21_acc = 0.0f;
    float diag12_acc = 0.0f;
    float diag22_acc = 0.0f;

    float self11 = 0.0f;
    float self21 = 0.0f;
    float self12 = 0.0f;
    float self22 = 0.0f;

    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t j = y_col[k];

        const cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
        const cuFloatComplex Vj = make_cuFloatComplex(V_f[j * 2], V_f[j * 2 + 1]);
        const cuFloatComplex curr = cuCmulf(y, Vj);
        const cuFloatComplex term_va = cuCmulf(neg_j_Vi, cuConjf(curr));

        const float vj_abs = cuCabsf(Vj);
        const cuFloatComplex term_vm = (vj_abs > 1e-6f)
            ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr) / vj_abs,
                                                      cuCimagf(curr) / vj_abs)))
            : make_cuFloatComplex(0.0f, 0.0f);

        diag11_acc += -cuCrealf(term_va);
        diag21_acc += -cuCimagf(term_va);

        if (have_vi_norm) {
            const cuFloatComplex term_vm2 = cuCmulf(Vi_norm, cuConjf(curr));
            diag12_acc += cuCrealf(term_vm2);
            diag22_acc += cuCimagf(term_vm2);
        }

        if (j == i) {
            self11 = cuCrealf(term_va);
            self21 = cuCimagf(term_va);
            self12 = cuCrealf(term_vm);
            self22 = cuCimagf(term_vm);
            continue;
        }

        if (map11[k] >= 0) J_csr_values[map11[k]] = cuCrealf(term_va);
        if (map21[k] >= 0) J_csr_values[map21[k]] = cuCimagf(term_va);
        if (map12[k] >= 0) J_csr_values[map12[k]] = cuCrealf(term_vm);
        if (map22[k] >= 0) J_csr_values[map22[k]] = cuCimagf(term_vm);
    }

    diag11_acc = warp_sum(diag11_acc);
    diag21_acc = warp_sum(diag21_acc);
    diag12_acc = warp_sum(diag12_acc);
    diag22_acc = warp_sum(diag22_acc);

    self11 = warp_sum(self11);
    self21 = warp_sum(self21);
    self12 = warp_sum(self12);
    self22 = warp_sum(self22);

    if (lane == 0) {
        if (diag11[i] >= 0) J_csr_values[diag11[i]] = self11 + diag11_acc;
        if (diag21[i] >= 0) J_csr_values[diag21[i]] = self21 + diag21_acc;
        if (diag12[i] >= 0) J_csr_values[diag12[i]] = self12 + diag12_acc;
        if (diag22[i] >= 0) J_csr_values[diag22[i]] = self22 + diag22_acc;
    }
}


// ---------------------------------------------------------------------------
// update_jacobian_kernel_fp32
//
// One thread per Ybus non-zero entry k at (Y_i, Y_j).
// Computes the Jacobian contributions of the edge (i→j) in FP32 and writes
// them directly into d_J_csr_f (CSR-ordered, no separate permutation step).
//
// JacobianMaps indices (map11 etc.) are CSR positions — they map directly
// into d_J_csr_f.
//
// Off-diagonal and diagonal contributions use atomicAdd because multiple
// edges can map to the same Jacobian entry.
// ---------------------------------------------------------------------------
__global__ void update_jacobian_kernel_fp32(
    int32_t n_elements,
    const float* __restrict__ G,          // [y_nnz] conductance (FP32)
    const float* __restrict__ B,          // [y_nnz] susceptance (FP32)
    const int32_t* __restrict__ Y_row,    // [y_nnz] from-bus index
    const int32_t* __restrict__ Y_col,    // [y_nnz] to-bus index
    const float* __restrict__ V_f,        // [n_bus*2] interleaved FP32 voltage
    const int32_t* __restrict__ map11,    // [y_nnz] → J CSR value index
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,   // [n_bus] → J CSR diagonal index
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_csr_values      // [j_nnz] output (CSR order)
) {
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) return;

    const int32_t i = Y_row[k];
    const int32_t j = Y_col[k];

    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
    cuFloatComplex Vi = make_cuFloatComplex(V_f[i * 2], V_f[i * 2 + 1]);
    cuFloatComplex Vj = make_cuFloatComplex(V_f[j * 2], V_f[j * 2 + 1]);

    // curr = Yij * Vj
    cuFloatComplex curr = cuCmulf(y, Vj);

    // term_va = -j * Vi * conj(curr)
    //   -j * Vi = (Im(Vi), -Re(Vi))
    cuFloatComplex neg_j_Vi = make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi));
    cuFloatComplex term_va  = cuCmulf(neg_j_Vi, cuConjf(curr));

    // term_vm = Vi * conj(curr / |Vj|)
    float vj_abs = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_abs > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr) / vj_abs,
                                                   cuCimagf(curr) / vj_abs)))
        : make_cuFloatComplex(0.0f, 0.0f);

    // Off-diagonal contributions
    if (map11[k] >= 0) atomicAdd(&J_csr_values[map11[k]], cuCrealf(term_va));
    if (map21[k] >= 0) atomicAdd(&J_csr_values[map21[k]], cuCimagf(term_va));
    if (map12[k] >= 0) atomicAdd(&J_csr_values[map12[k]], cuCrealf(term_vm));
    if (map22[k] >= 0) atomicAdd(&J_csr_values[map22[k]], cuCimagf(term_vm));

    // Diagonal correction at from-bus i:
    //   diag_va = -term_va  (sign flip because diagonal absorbs all edges)
    if (diag11[i] >= 0) atomicAdd(&J_csr_values[diag11[i]], -cuCrealf(term_va));
    if (diag21[i] >= 0) atomicAdd(&J_csr_values[diag21[i]], -cuCimagf(term_va));

    // Diagonal vm correction uses Vi/|Vi| instead of Vj/|Vj|
    float vi_abs = cuCabsf(Vi);
    if (vi_abs > 1e-6f) {
        cuFloatComplex Vi_norm = make_cuFloatComplex(cuCrealf(Vi) / vi_abs,
                                                      cuCimagf(Vi) / vi_abs);
        cuFloatComplex term_vm2 = cuCmulf(Vi_norm, cuConjf(curr));
        if (diag12[i] >= 0) atomicAdd(&J_csr_values[diag12[i]], cuCrealf(term_vm2));
        if (diag22[i] >= 0) atomicAdd(&J_csr_values[diag22[i]], cuCimagf(term_vm2));
    }
}


// ---------------------------------------------------------------------------
// update_jacobian_vertex_kernel_fp64
//
// FP64 variant of the vertex-based Jacobian kernel.
// One warp per active bus (pvpq), iterates over Ybus CSR row.
// All arithmetic in double / cuDoubleComplex.
// ---------------------------------------------------------------------------
__global__ void update_jacobian_vertex_kernel_fp64(
    int32_t n_active_buses,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ y_row_ptr,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ G,
    const double* __restrict__ B,
    const cuDoubleComplex* __restrict__ V_cd,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    double* __restrict__ J_csr_values)
{
    constexpr int32_t warp_size = 32;

    const int32_t warp_id_in_block = threadIdx.x / warp_size;
    const int32_t lane             = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block  = blockDim.x / warp_size;
    const int32_t bus_slot         = blockIdx.x * warps_per_block + warp_id_in_block;

    if (bus_slot >= n_active_buses) return;

    const int32_t i = pvpq[bus_slot];
    const int32_t row_begin = y_row_ptr[i];
    const int32_t row_end   = y_row_ptr[i + 1];

    const cuDoubleComplex Vi     = V_cd[i];
    const cuDoubleComplex neg_j_Vi = make_cuDoubleComplex(cuCimag(Vi), -cuCreal(Vi));
    const double vi_abs = cuCabs(Vi);
    const bool have_vi_norm = vi_abs > 1e-12;
    const cuDoubleComplex Vi_norm = have_vi_norm
        ? make_cuDoubleComplex(cuCreal(Vi) / vi_abs, cuCimag(Vi) / vi_abs)
        : make_cuDoubleComplex(0.0, 0.0);

    double diag11_acc = 0.0, diag21_acc = 0.0;
    double diag12_acc = 0.0, diag22_acc = 0.0;
    double self11 = 0.0, self21 = 0.0, self12 = 0.0, self22 = 0.0;

    for (int32_t k = row_begin + lane; k < row_end; k += warp_size) {
        const int32_t j = y_col[k];

        const cuDoubleComplex y    = make_cuDoubleComplex(G[k], B[k]);
        const cuDoubleComplex Vj   = V_cd[j];
        const cuDoubleComplex curr = cuCmul(y, Vj);
        const cuDoubleComplex term_va = cuCmul(neg_j_Vi, cuConj(curr));

        const double vj_abs = cuCabs(Vj);
        const cuDoubleComplex term_vm = (vj_abs > 1e-12)
            ? cuCmul(Vi, cuConj(make_cuDoubleComplex(cuCreal(curr) / vj_abs,
                                                      cuCimag(curr) / vj_abs)))
            : make_cuDoubleComplex(0.0, 0.0);

        diag11_acc += -cuCreal(term_va);
        diag21_acc += -cuCimag(term_va);

        if (have_vi_norm) {
            const cuDoubleComplex term_vm2 = cuCmul(Vi_norm, cuConj(curr));
            diag12_acc += cuCreal(term_vm2);
            diag22_acc += cuCimag(term_vm2);
        }

        if (j == i) {
            self11 = cuCreal(term_va); self21 = cuCimag(term_va);
            self12 = cuCreal(term_vm); self22 = cuCimag(term_vm);
            continue;
        }

        if (map11[k] >= 0) J_csr_values[map11[k]] = cuCreal(term_va);
        if (map21[k] >= 0) J_csr_values[map21[k]] = cuCimag(term_va);
        if (map12[k] >= 0) J_csr_values[map12[k]] = cuCreal(term_vm);
        if (map22[k] >= 0) J_csr_values[map22[k]] = cuCimag(term_vm);
    }

    // Warp reduction for diagonal accumulators (double)
    for (int offset = 16; offset > 0; offset >>= 1) {
        diag11_acc += __shfl_down_sync(0xffffffffu, diag11_acc, offset);
        diag21_acc += __shfl_down_sync(0xffffffffu, diag21_acc, offset);
        diag12_acc += __shfl_down_sync(0xffffffffu, diag12_acc, offset);
        diag22_acc += __shfl_down_sync(0xffffffffu, diag22_acc, offset);
        self11 += __shfl_down_sync(0xffffffffu, self11, offset);
        self21 += __shfl_down_sync(0xffffffffu, self21, offset);
        self12 += __shfl_down_sync(0xffffffffu, self12, offset);
        self22 += __shfl_down_sync(0xffffffffu, self22, offset);
    }

    if (lane == 0) {
        if (diag11[i] >= 0) J_csr_values[diag11[i]] = self11 + diag11_acc;
        if (diag21[i] >= 0) J_csr_values[diag21[i]] = self21 + diag21_acc;
        if (diag12[i] >= 0) J_csr_values[diag12[i]] = self12 + diag12_acc;
        if (diag22[i] >= 0) J_csr_values[diag22[i]] = self22 + diag22_acc;
    }
}


// ---------------------------------------------------------------------------
// update_jacobian_kernel_fp64
//
// FP64 edge-based Jacobian kernel. One thread per Ybus non-zero entry.
// Uses atomicAdd (double atomics supported on sm_60+).
// ---------------------------------------------------------------------------
__global__ void update_jacobian_kernel_fp64(
    int32_t n_elements,
    const double* __restrict__ G,
    const double* __restrict__ B,
    const int32_t* __restrict__ Y_row,
    const int32_t* __restrict__ Y_col,
    const cuDoubleComplex* __restrict__ V_cd,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    double* __restrict__ J_csr_values)
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) return;

    const int32_t i = Y_row[k];
    const int32_t j = Y_col[k];

    const cuDoubleComplex y    = make_cuDoubleComplex(G[k], B[k]);
    const cuDoubleComplex Vi   = V_cd[i];
    const cuDoubleComplex Vj   = V_cd[j];
    const cuDoubleComplex curr = cuCmul(y, Vj);

    const cuDoubleComplex neg_j_Vi = make_cuDoubleComplex(cuCimag(Vi), -cuCreal(Vi));
    const cuDoubleComplex term_va  = cuCmul(neg_j_Vi, cuConj(curr));

    const double vj_abs = cuCabs(Vj);
    const cuDoubleComplex term_vm = (vj_abs > 1e-12)
        ? cuCmul(Vi, cuConj(make_cuDoubleComplex(cuCreal(curr) / vj_abs,
                                                  cuCimag(curr) / vj_abs)))
        : make_cuDoubleComplex(0.0, 0.0);

    if (map11[k] >= 0) atomic_add_f64_compat(&J_csr_values[map11[k]], cuCreal(term_va));
    if (map21[k] >= 0) atomic_add_f64_compat(&J_csr_values[map21[k]], cuCimag(term_va));
    if (map12[k] >= 0) atomic_add_f64_compat(&J_csr_values[map12[k]], cuCreal(term_vm));
    if (map22[k] >= 0) atomic_add_f64_compat(&J_csr_values[map22[k]], cuCimag(term_vm));

    if (diag11[i] >= 0) atomic_add_f64_compat(&J_csr_values[diag11[i]], -cuCreal(term_va));
    if (diag21[i] >= 0) atomic_add_f64_compat(&J_csr_values[diag21[i]], -cuCimag(term_va));

    const double vi_abs = cuCabs(Vi);
    if (vi_abs > 1e-12) {
        const cuDoubleComplex Vi_norm = make_cuDoubleComplex(cuCreal(Vi) / vi_abs,
                                                              cuCimag(Vi) / vi_abs);
        const cuDoubleComplex term_vm2 = cuCmul(Vi_norm, cuConj(curr));
        if (diag12[i] >= 0) atomic_add_f64_compat(&J_csr_values[diag12[i]], cuCreal(term_vm2));
        if (diag22[i] >= 0) atomic_add_f64_compat(&J_csr_values[diag22[i]], cuCimag(term_vm2));
    }
}


// ---------------------------------------------------------------------------
// updateJacobian
//
// Dispatches to FP32 or FP64 Jacobian kernel based on precision_mode:
//
//   Mixed / FP32 — zero d_J_csr_f, convert V_cd→V_f, run FP32 kernel
//   FP64         — zero d_J_csr_d, run FP64 kernel directly on d_V_cd
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::updateJacobian()
{
    auto& im = *impl_;
    const int32_t y_nnz = im.y_nnz;
    const int32_t j_nnz = im.j_nnz;
    const int32_t n_bus = im.n_bus;
    const int32_t block = 256;

    if (im.precision_mode == PrecisionMode::FP64) {
        // --- FP64 path ---
        CUDA_CHECK(cudaMemset(im.d_J_csr_d, 0, j_nnz * sizeof(double)));

        if (im.jacobian_type == JacobianBuilderType::VertexBased) {
            constexpr int32_t warp_size = 32;
            const int32_t warps_per_block = block / warp_size;
            const int32_t grid = (im.n_pvpq + warps_per_block - 1) / warps_per_block;

            update_jacobian_vertex_kernel_fp64<<<grid, block>>>(
                im.n_pvpq, im.d_pvpq,
                im.d_Ybus_rp, im.d_Ybus_ci,
                im.d_G_d, im.d_B_d,
                im.d_V_cd,
                im.d_mapJ11, im.d_mapJ21, im.d_mapJ12, im.d_mapJ22,
                im.d_diagJ11, im.d_diagJ21, im.d_diagJ12, im.d_diagJ22,
                im.d_J_csr_d);
        } else {
            const int32_t grid = (y_nnz + block - 1) / block;
            update_jacobian_kernel_fp64<<<grid, block>>>(
                y_nnz,
                im.d_G_d, im.d_B_d,
                im.d_Y_row, im.d_Y_col,
                im.d_V_cd,
                im.d_mapJ11, im.d_mapJ21, im.d_mapJ12, im.d_mapJ22,
                im.d_diagJ11, im.d_diagJ21, im.d_diagJ12, im.d_diagJ22,
                im.d_J_csr_d);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // --- FP32 / Mixed path ---
    CUDA_CHECK(cudaMemset(im.d_J_csr_f, 0, j_nnz * sizeof(float)));

    // Convert voltage to FP32 interleaved for Jacobian kernel.
    // Mixed: convert from d_V_cd (FP64 complex state).
    // FP32:  d_V_f is kept up-to-date by updateVoltage_f32, but we
    //        also re-sync from d_V_cf to be safe.
    if (im.precision_mode == PrecisionMode::Mixed) {
        int32_t grid = (n_bus + block - 1) / block;
        convert_V_cd_to_f_kernel<<<grid, block>>>(im.d_V_cd, im.d_V_f, n_bus);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // FP32: d_V_cf → d_V_f (interleaved)
        int32_t grid = (n_bus + block - 1) / block;
        convert_V_cf_to_f_kernel<<<grid, block>>>(im.d_V_cf, im.d_V_f, n_bus);
        CUDA_CHECK(cudaGetLastError());
    }

    if (im.jacobian_type == JacobianBuilderType::VertexBased) {
        constexpr int32_t warp_size = 32;
        const int32_t warps_per_block = block / warp_size;
        const int32_t grid = (im.n_pvpq + warps_per_block - 1) / warps_per_block;

        update_jacobian_vertex_kernel_fp32<<<grid, block>>>(
            im.n_pvpq, im.d_pvpq,
            im.d_Ybus_rp, im.d_Ybus_ci,
            im.d_G_f, im.d_B_f, im.d_V_f,
            im.d_mapJ11, im.d_mapJ21, im.d_mapJ12, im.d_mapJ22,
            im.d_diagJ11, im.d_diagJ21, im.d_diagJ12, im.d_diagJ22,
            im.d_J_csr_f);
    } else {
        const int32_t grid = (y_nnz + block - 1) / block;
        update_jacobian_kernel_fp32<<<grid, block>>>(
            y_nnz,
            im.d_G_f, im.d_B_f,
            im.d_Y_row, im.d_Y_col,
            im.d_V_f,
            im.d_mapJ11, im.d_mapJ21, im.d_mapJ12, im.d_mapJ22,
            im.d_diagJ11, im.d_diagJ21, im.d_diagJ12, im.d_diagJ22,
            im.d_J_csr_f);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// convert_V_cf_batch_kernel
//
// Converts complex FP64 voltage batch to FP32 complex (cuFloatComplex) batch.
// Both buffers layout: [n_batch * n_bus], index = b*n_bus + i.
// ---------------------------------------------------------------------------
__global__ void convert_V_cf_batch_kernel(
    const cuDoubleComplex* __restrict__ V_cd,  // [n_batch * n_bus]
    cuFloatComplex* __restrict__ V_cf,         // [n_batch * n_bus]
    int32_t total)                              // n_batch * n_bus
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    V_cf[k] = make_cuFloatComplex(static_cast<float>(cuCreal(V_cd[k])),
                                   static_cast<float>(cuCimag(V_cd[k])));
}


// ---------------------------------------------------------------------------
// update_jacobian_batch_kernel_fp32
//
// Edge-based batch Jacobian kernel.
// One thread per Ybus nnz entry k; inner loop over batch b.
//
// V_cf layout:   cuFloatComplex [n_batch * n_bus], V[b][i] = V_cf[b*n_bus + i]
// J_csr layout:  float [n_batch * j_nnz],          J[b][p] = J[b*j_nnz + p]
//
// Uses atomicAdd for off-diagonal and diagonal contributions (same as single
// edge-based kernel).
// ---------------------------------------------------------------------------
__global__ void update_jacobian_batch_kernel_fp32(
    int32_t n_elements,
    int32_t n_bus,
    int32_t n_batch,
    int32_t j_nnz,
    const float* __restrict__ G,           // [y_nnz]
    const float* __restrict__ B,           // [y_nnz]
    const int32_t* __restrict__ Y_row,     // [y_nnz]
    const int32_t* __restrict__ Y_col,     // [y_nnz]
    const cuFloatComplex* __restrict__ V,  // [n_batch * n_bus]
    const int32_t* __restrict__ map11,     // [y_nnz]
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,    // [n_bus]
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ J_csr_values       // [n_batch * j_nnz]
) {
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) return;

    const int32_t i   = Y_row[k];
    const int32_t j   = Y_col[k];
    const cuFloatComplex y = make_cuFloatComplex(G[k], B[k]);

    const int32_t p11 = map11[k];
    const int32_t p21 = map21[k];
    const int32_t p12 = map12[k];
    const int32_t p22 = map22[k];
    const int32_t d11 = diag11[i];
    const int32_t d21 = diag21[i];
    const int32_t d12 = diag12[i];
    const int32_t d22 = diag22[i];

    for (int32_t b = 0; b < n_batch; ++b) {
        const cuFloatComplex Vi = V[b * n_bus + i];
        const cuFloatComplex Vj = V[b * n_bus + j];

        const cuFloatComplex curr    = cuCmulf(y, Vj);
        const cuFloatComplex neg_j_Vi = make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi));
        const cuFloatComplex term_va  = cuCmulf(neg_j_Vi, cuConjf(curr));

        const float vj_abs = cuCabsf(Vj);
        const cuFloatComplex term_vm = (vj_abs > 1e-6f)
            ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr) / vj_abs,
                                                       cuCimagf(curr) / vj_abs)))
            : make_cuFloatComplex(0.0f, 0.0f);

        float* Jb = J_csr_values + b * j_nnz;

        if (p11 >= 0) atomicAdd(&Jb[p11], cuCrealf(term_va));
        if (p21 >= 0) atomicAdd(&Jb[p21], cuCimagf(term_va));
        if (p12 >= 0) atomicAdd(&Jb[p12], cuCrealf(term_vm));
        if (p22 >= 0) atomicAdd(&Jb[p22], cuCimagf(term_vm));

        if (d11 >= 0) atomicAdd(&Jb[d11], -cuCrealf(term_va));
        if (d21 >= 0) atomicAdd(&Jb[d21], -cuCimagf(term_va));

        const float vi_abs = cuCabsf(Vi);
        if (vi_abs > 1e-6f) {
            const cuFloatComplex Vi_norm = make_cuFloatComplex(cuCrealf(Vi) / vi_abs,
                                                                cuCimagf(Vi) / vi_abs);
            const cuFloatComplex term_vm2 = cuCmulf(Vi_norm, cuConjf(curr));
            if (d12 >= 0) atomicAdd(&Jb[d12], cuCrealf(term_vm2));
            if (d22 >= 0) atomicAdd(&Jb[d22], cuCimagf(term_vm2));
        }
    }
}


// ---------------------------------------------------------------------------
// updateJacobian_batch
//
// GPU pipeline:
//   1. Zero d_J_csr_f_batch
//   2. FP64 V_cd_batch → FP32 complex V_cf_batch
//   3. update_jacobian_batch_kernel_fp32: fill d_J_csr_f_batch
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::updateJacobian_batch(int32_t n_batch)
{
    auto& im = *impl_;
    const int32_t y_nnz  = im.y_nnz;
    const int32_t j_nnz  = im.j_nnz;
    const int32_t n_bus  = im.n_bus;
    const int32_t block  = 256;
    const int64_t total  = (int64_t)n_batch * j_nnz;

    // Step 1: zero batch Jacobian buffer
    CUDA_CHECK(cudaMemset(im.d_J_csr_f_batch, 0, total * sizeof(float)));

    // Step 2: FP64 → FP32 complex (batch)
    {
        const int32_t tot = n_batch * n_bus;
        int32_t grid = (tot + block - 1) / block;
        convert_V_cf_batch_kernel<<<grid, block>>>(im.d_V_cd_batch, im.d_V_cf_batch, tot);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: fill Jacobian values
    {
        int32_t grid = (y_nnz + block - 1) / block;
        update_jacobian_batch_kernel_fp32<<<grid, block>>>(
            y_nnz, n_bus, n_batch, j_nnz,
            im.d_G_f, im.d_B_f,
            im.d_Y_row, im.d_Y_col,
            im.d_V_cf_batch,
            im.d_mapJ11, im.d_mapJ21, im.d_mapJ12, im.d_mapJ22,
            im.d_diagJ11, im.d_diagJ21, im.d_diagJ12, im.d_diagJ22,
            im.d_J_csr_f_batch);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ---------------------------------------------------------------------------
// update_voltage_batch_kernel_fp64
//
// Applies dx corrections to Va/Vm and reconstructs complex V for all batches.
// dx layout: [n_batch * dimF], V/Va/Vm layout: [n_batch * n_bus]
// One thread per (b, tid) element of dx (flattened).
// ---------------------------------------------------------------------------
__global__ void update_voltage_batch_kernel_fp64(
    double* __restrict__ Va,          // [n_batch * n_bus]
    double* __restrict__ Vm,          // [n_batch * n_bus]
    const float* __restrict__ dx,     // [n_batch * dimF]  FP32 from cuDSS
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq,
    int32_t n_bus, int32_t dimF,
    int32_t n_batch)
{
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_batch * dimF) return;

    const int32_t b   = gid / dimF;
    const int32_t tid = gid % dimF;

    double* Va_b = Va + b * n_bus;
    double* Vm_b = Vm + b * n_bus;
    double  dxv  = static_cast<double>(dx[b * dimF + tid]);

    if (tid < n_pv) {
        Va_b[pv[tid]] += dxv;
    } else if (tid < n_pv + n_pq) {
        Va_b[pq[tid - n_pv]] += dxv;
    } else {
        Vm_b[pq[tid - n_pv - n_pq]] += dxv;
    }
}

__global__ void decompose_V_batch_kernel(
    const cuDoubleComplex* __restrict__ V_cd,
    double* __restrict__ Va,
    double* __restrict__ Vm,
    int32_t total)
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;

    const double re = cuCreal(V_cd[k]);
    const double im = cuCimag(V_cd[k]);
    Va[k] = atan2(im, re);
    Vm[k] = hypot(re, im);
}

__global__ void reconstruct_V_batch_kernel(
    const double* __restrict__ Va,      // [n_batch * n_bus]
    const double* __restrict__ Vm,      // [n_batch * n_bus]
    cuDoubleComplex* __restrict__ V_cd, // [n_batch * n_bus]
    int32_t total)                      // n_batch * n_bus
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    V_cd[k] = make_cuDoubleComplex(Vm[k] * cos(Va[k]), Vm[k] * sin(Va[k]));
}


// ---------------------------------------------------------------------------
// updateVoltage_batch
//
// Applies dx corrections (FP32, from cuDSS solve) for all batches.
// dx_batch is host-side [n_batch * dimF] FP32.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::updateVoltage_batch(
    const double*  dx_batch_d64,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    int32_t        n_batch)
{
    auto& im = *impl_;
    const int32_t dimF  = im.dimF;
    const int32_t n_bus = im.n_bus;
    const int32_t block = 256;

    // dx_batch_d64 is a host double array; convert to float and upload
    // (d_x_f_batch already holds the FP32 solution from cuDSS — reuse directly)
    // The caller (solve_batch NR loop) passes dx from solveLinearSystem_batch
    // which already wrote to d_x_f_batch. We skip the redundant copy and use
    // d_x_f_batch directly.
    (void)dx_batch_d64;  // not used; cuDSS output lives in d_x_f_batch

    {
        const int32_t tot2 = n_batch * n_bus;
        int32_t grid = (tot2 + block - 1) / block;
        decompose_V_batch_kernel<<<grid, block>>>(
            im.d_V_cd_batch, im.d_Va_batch, im.d_Vm_batch, tot2);
        CUDA_CHECK(cudaGetLastError());
    }

    const int32_t total = n_batch * dimF;
    {
        int32_t grid = (total + block - 1) / block;
        update_voltage_batch_kernel_fp64<<<grid, block>>>(
            im.d_Va_batch, im.d_Vm_batch,
            im.d_x_f_batch,
            im.d_pv, im.d_pq, n_pv, n_pq,
            n_bus, dimF, n_batch);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const int32_t tot2 = n_batch * n_bus;
        int32_t grid = (tot2 + block - 1) / block;
        reconstruct_V_batch_kernel<<<grid, block>>>(
            im.d_Va_batch, im.d_Vm_batch, im.d_V_cd_batch, tot2);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ---------------------------------------------------------------------------
// decompose_V_kernel
//
// Matches v1 CPU semantics by recomputing Va/Vm from the current complex V
// before each dx application. This avoids accumulating a stale Vm sign state.
// ---------------------------------------------------------------------------
__global__ void decompose_V_kernel(
    const cuDoubleComplex* __restrict__ V_cd,
    double* __restrict__ Va,
    double* __restrict__ Vm,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;

    const double re = cuCreal(V_cd[i]);
    const double im = cuCimag(V_cd[i]);
    Va[i] = atan2(im, re);
    Vm[i] = hypot(re, im);
}

// ---------------------------------------------------------------------------
// update_voltage_kernel_f32
//
// Applies FP32 dx correction (from cuDSS d_x_f, GPU) to FP64 Va/Vm.
// One thread per element of dx.  Va/Vm accumulation stays FP64 for accuracy.
// ---------------------------------------------------------------------------
__global__ void update_voltage_kernel_f32(
    double* __restrict__ Va,
    double* __restrict__ Vm,
    const float* __restrict__ dx_f,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) return;

    double dxv = static_cast<double>(dx_f[tid]);
    if (tid < n_pv) {
        Va[pv[tid]] += dxv;
    } else if (tid < n_pv + n_pq) {
        Va[pq[tid - n_pv]] += dxv;
    } else {
        Vm[pq[tid - n_pv - n_pq]] += dxv;
    }
}

__global__ void reconstruct_V_kernel(
    const double* __restrict__ Va,
    const double* __restrict__ Vm,
    cuDoubleComplex* __restrict__ V_cd,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;
    V_cd[i] = make_cuDoubleComplex(Vm[i] * cos(Va[i]), Vm[i] * sin(Va[i]));
}


// ---------------------------------------------------------------------------
// update_voltage_kernel_f64
//
// FP64 mode: applies FP64 dx (from cuDSS FP64 solve) to FP64 Va/Vm.
// ---------------------------------------------------------------------------
__global__ void update_voltage_kernel_f64(
    double* __restrict__ Va,
    double* __restrict__ Vm,
    const double* __restrict__ dx_d,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) return;

    if (tid < n_pv) {
        Va[pv[tid]] += dx_d[tid];
    } else if (tid < n_pv + n_pq) {
        Va[pq[tid - n_pv]] += dx_d[tid];
    } else {
        Vm[pq[tid - n_pv - n_pq]] += dx_d[tid];
    }
}


// ---------------------------------------------------------------------------
// decompose_V_f32_kernel
//
// FP32 mode: FP32 complex V_cf → float Va, Vm.
// ---------------------------------------------------------------------------
__global__ void decompose_V_f32_kernel(
    const cuFloatComplex* __restrict__ V_cf,
    float* __restrict__ Va,
    float* __restrict__ Vm,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;
    Va[i] = atan2f(cuCimagf(V_cf[i]), cuCrealf(V_cf[i]));
    Vm[i] = hypotf(cuCrealf(V_cf[i]), cuCimagf(V_cf[i]));
}


// ---------------------------------------------------------------------------
// update_voltage_f32_only_kernel
//
// FP32 mode: applies FP32 dx to FP32 Va/Vm (both in FP32).
// ---------------------------------------------------------------------------
__global__ void update_voltage_f32_only_kernel(
    float* __restrict__ Va,
    float* __restrict__ Vm,
    const float* __restrict__ dx_f,
    const int32_t* __restrict__ pv,
    const int32_t* __restrict__ pq,
    int32_t n_pv, int32_t n_pq)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pv + 2 * n_pq) return;

    if (tid < n_pv) {
        Va[pv[tid]] += dx_f[tid];
    } else if (tid < n_pv + n_pq) {
        Va[pq[tid - n_pv]] += dx_f[tid];
    } else {
        Vm[pq[tid - n_pv - n_pq]] += dx_f[tid];
    }
}


// ---------------------------------------------------------------------------
// reconstruct_V_f32_kernel
//
// FP32 mode: float Va, Vm → FP32 complex V_cf.
// ---------------------------------------------------------------------------
__global__ void reconstruct_V_f32_kernel(
    const float* __restrict__ Va,
    const float* __restrict__ Vm,
    cuFloatComplex* __restrict__ V_cf,
    int32_t n_bus)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bus) return;
    V_cf[i] = make_cuFloatComplex(Vm[i] * cosf(Va[i]), Vm[i] * sinf(Va[i]));
}


// ---------------------------------------------------------------------------
// updateVoltage: apply dx correction on GPU.
//
// Mixed — reads d_x_f (FP32) from cuDSS; accumulates into FP64 Va/Vm.
// FP64  — reads d_x_d (FP64) from cuDSS; accumulates into FP64 Va/Vm.
// FP32  — handled by updateVoltage_f32 (separate method below).
//
// Host dx pointer is intentionally ignored for all CUDA paths.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::updateVoltage(
    const double*  /*dx*/,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    auto& im = *impl_;
    const int32_t dimF  = n_pv + 2 * n_pq;
    const int32_t block = 256;

    {
        int32_t grid = (im.n_bus + block - 1) / block;
        decompose_V_kernel<<<grid, block>>>(im.d_V_cd, im.d_Va, im.d_Vm, im.n_bus);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int32_t grid = (dimF + block - 1) / block;
        if (im.precision_mode == PrecisionMode::FP64) {
            update_voltage_kernel_f64<<<grid, block>>>(
                im.d_Va, im.d_Vm, im.d_x_d,
                im.d_pv, im.d_pq, n_pv, n_pq);
        } else {
            // Mixed: FP32 dx from cuDSS, FP64 Va/Vm accumulation
            update_voltage_kernel_f32<<<grid, block>>>(
                im.d_Va, im.d_Vm, im.d_x_f,
                im.d_pv, im.d_pq, n_pv, n_pq);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int32_t grid = (im.n_bus + block - 1) / block;
        reconstruct_V_kernel<<<grid, block>>>(im.d_Va, im.d_Vm, im.d_V_cd, im.n_bus);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ---------------------------------------------------------------------------
// updateVoltage_f32: FP32 end-to-end voltage update.
//
// Reads d_x_f (FP32) from cuDSS; accumulates into FP32 Va_f/Vm_f;
// reconstructs FP32 complex V_cf. Also updates d_V_f (interleaved) for
// the next Jacobian kernel call.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::updateVoltage_f32(
    const float*   /*dx*/,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    auto& im = *impl_;
    const int32_t dimF  = n_pv + 2 * n_pq;
    const int32_t block = 256;

    {
        int32_t grid = (im.n_bus + block - 1) / block;
        decompose_V_f32_kernel<<<grid, block>>>(im.d_V_cf, im.d_Va_f, im.d_Vm_f, im.n_bus);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int32_t grid = (dimF + block - 1) / block;
        update_voltage_f32_only_kernel<<<grid, block>>>(
            im.d_Va_f, im.d_Vm_f, im.d_x_f,
            im.d_pv, im.d_pq, n_pv, n_pq);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int32_t grid = (im.n_bus + block - 1) / block;
        reconstruct_V_f32_kernel<<<grid, block>>>(im.d_Va_f, im.d_Vm_f, im.d_V_cf, im.n_bus);
        CUDA_CHECK(cudaGetLastError());
    }
}
