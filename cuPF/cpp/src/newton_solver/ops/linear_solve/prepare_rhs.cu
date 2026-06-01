// ---------------------------------------------------------------------------
// prepare_rhs.cu
//
// CUDA support kernels for the linear-solve and torch-bridge paths. Despite the
// historical name, this unit hosts several small kernels (each followed by its
// typed launch_* wrapper declared in cuda_linear_solve_kernels.hpp):
//   - prepare_rhs            : down-cast FP64 RHS -> FP32 working buffer
//   - transpose_csr_values   : scatter J values into J^T order (adjoint)
//   - gather_adjoint_rhs     : pack dL/dVa, dL/dVm into the dimF RHS
//   - project_load_gradients : scatter the adjoint solution onto load grads
//   - set_pf_inputs_from_load: build Sbus / V from base power + load tensors
//   - copy_voltage_outputs   : write Va/Vm out at the caller's precision
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_linear_solve_kernels.hpp"

#include "utils/cuda_utils.hpp"

#include <stdexcept>
#include <type_traits>


// ===========================================================================
// Device kernels
// ===========================================================================
namespace {

// Element-wise FP64 -> FP32 down-cast of the residual into the FP32 RHS buffer.
__global__ void prepare_rhs_kernel(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int32_t count)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }

    dst[tid] = static_cast<float>(src[tid]);  // explicit narrowing double -> float
}

// Move each batched J value to its transposed slot via the precomputed map.
template <typename T>
__global__ void transpose_csr_values_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int32_t* __restrict__ src_to_transpose_pos,
    int32_t nnz,
    int32_t total)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }
    const int32_t batch = tid / nnz;
    const int32_t k = tid - batch * nnz;
    const int32_t dst_pos = src_to_transpose_pos[k];
    dst[batch * nnz + dst_pos] = src[tid];
}

// Pack the dense per-bus gradients into the dimF adjoint RHS (one thread per
// RHS entry). Row layout: [dVa@pv | dVa@pq | dVm@pq] — the device counterpart
// of build_grad_state() in adjoint_math.cpp.
template <typename T>
__global__ void gather_adjoint_rhs_kernel(
    const T* __restrict__ grad_va,
    const T* __restrict__ grad_vm,
    T* __restrict__ grad_state,
    const int32_t* __restrict__ pv,
    int32_t n_pv,
    const int32_t* __restrict__ pq,
    int32_t n_pq,
    int32_t n_bus,
    int32_t dimF,
    int32_t total)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int32_t b = tid / dimF;
    const int32_t row = tid - b * dimF;
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t base_bus = b * n_bus;
    if (row < n_pv) {
        grad_state[tid] = grad_va[base_bus + pv[row]];
    } else if (row < n_pvpq) {
        grad_state[tid] = grad_va[base_bus + pq[row - n_pv]];
    } else {
        grad_state[tid] = grad_vm[base_bus + pq[row - n_pvpq]];
    }
}

// Scatter the adjoint solution lambda onto per-bus load gradients (one thread
// per bus): load gradient = -lambda at pv/pq buses, zero elsewhere. Device
// counterpart of project_load_gradients() in adjoint_math.cpp.
template <typename T>
__global__ void project_load_gradients_kernel(
    const T* __restrict__ lambda,
    T* __restrict__ grad_load_p,
    T* __restrict__ grad_load_q,
    const int32_t* __restrict__ pv,
    int32_t n_pv,
    const int32_t* __restrict__ pq,
    int32_t n_pq,
    int32_t n_bus,
    int32_t dimF,
    int32_t total_bus)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_bus) return;

    // Default to zero; pv/pq buses below overwrite their own entry.
    grad_load_p[tid] = T(0);
    grad_load_q[tid] = T(0);

    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t b = tid / n_bus;            // batch case for this bus
    const int32_t base_bus = b * n_bus;       // start of this case's bus block
    const int32_t base_state = b * dimF;      // start of this case's lambda block
    for (int32_t i = 0; i < n_pv; ++i) {
        if (tid == base_bus + pv[i]) {
            grad_load_p[tid] = -lambda[base_state + i];
        }
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        if (tid == base_bus + pq[i]) {
            grad_load_p[tid] = -lambda[base_state + n_pv + i];
            grad_load_q[tid] = -lambda[base_state + n_pvpq + i];
        }
    }
}

// Build the solver's Sbus and initial voltage from base power, load, and flat-
// start angle/magnitude (one thread per bus). Inputs (InputT) may differ in
// precision from storage (StorageT); static_cast bridges the two. Sbus = base
// injection - load; V = mag * (cos(angle) + i sin(angle)).
template <typename InputT, typename StorageT>
__global__ void set_pf_inputs_from_load_kernel(
    const InputT* __restrict__ sbus_base_re,
    const InputT* __restrict__ sbus_base_im,
    const InputT* __restrict__ load_p,
    const InputT* __restrict__ load_q,
    const InputT* __restrict__ v0_va,
    const InputT* __restrict__ v0_vm,
    StorageT* __restrict__ sbus_re,
    StorageT* __restrict__ sbus_im,
    StorageT* __restrict__ va,
    StorageT* __restrict__ vm,
    StorageT* __restrict__ v_re,
    StorageT* __restrict__ v_im,
    int32_t n_bus,
    int32_t total_bus)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_bus) return;
    const int32_t bus = tid % n_bus;
    const StorageT p = static_cast<StorageT>(sbus_base_re[bus]) - static_cast<StorageT>(load_p[tid]);
    const StorageT q = static_cast<StorageT>(sbus_base_im[bus]) - static_cast<StorageT>(load_q[tid]);
    const StorageT angle = static_cast<StorageT>(v0_va[bus]);
    const StorageT mag = static_cast<StorageT>(v0_vm[bus]);
    // Use the precision-matched sincos intrinsic for the polar -> rect step.
    StorageT s = StorageT(0);
    StorageT c = StorageT(0);
    if constexpr (std::is_same_v<StorageT, double>) {
        sincos(angle, &s, &c);
    } else {
        sincosf(angle, &s, &c);
    }
    sbus_re[tid] = p;
    sbus_im[tid] = q;
    va[tid] = angle;
    vm[tid] = mag;
    v_re[tid] = mag * c;
    v_im[tid] = mag * s;
}

// Copy converged Va/Vm to the output tensors, casting storage -> output type.
template <typename StorageT, typename OutputT>
__global__ void copy_voltage_outputs_kernel(
    const StorageT* __restrict__ va,
    const StorageT* __restrict__ vm,
    OutputT* __restrict__ va_out,
    OutputT* __restrict__ vm_out,
    int32_t total_bus)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_bus) return;
    va_out[tid] = static_cast<OutputT>(va[tid]);  // storage precision -> output precision
    vm_out[tid] = static_cast<OutputT>(vm[tid]);
}

}  // namespace


// ===========================================================================
// Host launch wrappers (validate args, size the grid, launch on the cuPF
// stream, then check for errors). Templated launchers are explicitly
// instantiated at the bottom for the type combinations the pipelines use.
// ===========================================================================

void launch_prepare_rhs(const double* src, float* dst, int32_t count)
{
    if (src == nullptr || dst == nullptr || count < 0) {
        throw std::runtime_error("launch_prepare_rhs: invalid arguments");
    }
    if (count == 0) {
        return;
    }

    constexpr int32_t block = 256;
    const int32_t grid = (count + block - 1) / block;
    prepare_rhs_kernel<<<grid, block, 0, cupf_current_cuda_stream()>>>(src, dst, count);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename T>
void launch_transpose_csr_values_impl(const T* src,
                                      T* dst,
                                      const int32_t* src_to_transpose_pos,
                                      int32_t nnz,
                                      int32_t batch_size)
{
    if (src == nullptr || dst == nullptr || src_to_transpose_pos == nullptr ||
        nnz <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_transpose_csr_values: invalid arguments");
    }
    constexpr int32_t block = 256;
    const int32_t total = nnz * batch_size;
    const int32_t grid = (total + block - 1) / block;
    transpose_csr_values_kernel<T><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        src, dst, src_to_transpose_pos, nnz, total);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

void launch_transpose_csr_values(const double* src,
                                 double* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size)
{
    launch_transpose_csr_values_impl(src, dst, src_to_transpose_pos, nnz, batch_size);
}

void launch_transpose_csr_values(const float* src,
                                 float* dst,
                                 const int32_t* src_to_transpose_pos,
                                 int32_t nnz,
                                 int32_t batch_size)
{
    launch_transpose_csr_values_impl(src, dst, src_to_transpose_pos, nnz, batch_size);
}

template <typename T>
void launch_gather_adjoint_rhs(const T* grad_va,
                               const T* grad_vm,
                               T* grad_state,
                               const int32_t* pv,
                               int32_t n_pv,
                               const int32_t* pq,
                               int32_t n_pq,
                               int32_t n_bus,
                               int32_t batch_size)
{
    if (!grad_va || !grad_vm || !grad_state || !pq || n_pq < 0 || n_pv < 0 ||
        n_bus <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_gather_adjoint_rhs: invalid arguments");
    }
    const int32_t dimF = n_pv + 2 * n_pq;
    constexpr int32_t block = 256;
    const int32_t total = batch_size * dimF;
    const int32_t grid = (total + block - 1) / block;
    gather_adjoint_rhs_kernel<T><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        grad_va, grad_vm, grad_state, pv, n_pv, pq, n_pq, n_bus, dimF, total);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename T>
void launch_project_load_gradients(const T* lambda,
                                   T* grad_load_p,
                                   T* grad_load_q,
                                   const int32_t* pv,
                                   int32_t n_pv,
                                   const int32_t* pq,
                                   int32_t n_pq,
                                   int32_t n_bus,
                                   int32_t batch_size)
{
    if (!lambda || !grad_load_p || !grad_load_q || !pq || n_pq < 0 || n_pv < 0 ||
        n_bus <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_project_load_gradients: invalid arguments");
    }
    const int32_t dimF = n_pv + 2 * n_pq;
    constexpr int32_t block = 256;
    const int32_t total = batch_size * n_bus;
    const int32_t grid = (total + block - 1) / block;
    project_load_gradients_kernel<T><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        lambda, grad_load_p, grad_load_q, pv, n_pv, pq, n_pq, n_bus, dimF, total);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template void launch_gather_adjoint_rhs<double>(const double*, const double*, double*, const int32_t*, int32_t, const int32_t*, int32_t, int32_t, int32_t);
template void launch_gather_adjoint_rhs<float>(const float*, const float*, float*, const int32_t*, int32_t, const int32_t*, int32_t, int32_t, int32_t);
template void launch_project_load_gradients<double>(const double*, double*, double*, const int32_t*, int32_t, const int32_t*, int32_t, int32_t, int32_t);
template void launch_project_load_gradients<float>(const float*, float*, float*, const int32_t*, int32_t, const int32_t*, int32_t, int32_t, int32_t);

template <typename InputT, typename StorageT>
void launch_set_pf_inputs_from_load(const InputT* sbus_base_re,
                                    const InputT* sbus_base_im,
                                    const InputT* load_p,
                                    const InputT* load_q,
                                    const InputT* v0_va,
                                    const InputT* v0_vm,
                                    StorageT* sbus_re,
                                    StorageT* sbus_im,
                                    StorageT* va,
                                    StorageT* vm,
                                    StorageT* v_re,
                                    StorageT* v_im,
                                    int32_t n_bus,
                                    int32_t batch_size)
{
    if (!sbus_base_re || !sbus_base_im || !load_p || !load_q || !v0_va || !v0_vm ||
        !sbus_re || !sbus_im || !va || !vm || !v_re || !v_im ||
        n_bus <= 0 || batch_size <= 0) {
        throw std::runtime_error("launch_set_pf_inputs_from_load: invalid arguments");
    }
    constexpr int32_t block = 256;
    const int32_t total = batch_size * n_bus;
    const int32_t grid = (total + block - 1) / block;
    set_pf_inputs_from_load_kernel<InputT, StorageT><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        sbus_base_re, sbus_base_im, load_p, load_q, v0_va, v0_vm,
        sbus_re, sbus_im, va, vm, v_re, v_im, n_bus, total);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template <typename StorageT, typename OutputT>
void launch_copy_voltage_outputs(const StorageT* va,
                                 const StorageT* vm,
                                 OutputT* va_out,
                                 OutputT* vm_out,
                                 int32_t total_bus)
{
    if (!va || !vm || !va_out || !vm_out || total_bus <= 0) {
        throw std::runtime_error("launch_copy_voltage_outputs: invalid arguments");
    }
    constexpr int32_t block = 256;
    const int32_t grid = (total_bus + block - 1) / block;
    copy_voltage_outputs_kernel<StorageT, OutputT><<<grid, block, 0, cupf_current_cuda_stream()>>>(
        va, vm, va_out, vm_out, total_bus);
    CUDA_CHECK(cudaGetLastError());
    sync_cuda_for_timing();
}

template void launch_set_pf_inputs_from_load<double, double>(const double*, const double*, const double*, const double*, const double*, const double*, double*, double*, double*, double*, double*, double*, int32_t, int32_t);
template void launch_set_pf_inputs_from_load<float, float>(const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, float*, float*, int32_t, int32_t);
template void launch_set_pf_inputs_from_load<float, double>(const float*, const float*, const float*, const float*, const float*, const float*, double*, double*, double*, double*, double*, double*, int32_t, int32_t);
template void launch_copy_voltage_outputs<double, double>(const double*, const double*, double*, double*, int32_t);
template void launch_copy_voltage_outputs<float, float>(const float*, const float*, float*, float*, int32_t);
template void launch_copy_voltage_outputs<double, float>(const double*, const double*, float*, float*, int32_t);

#endif  // CUPF_WITH_CUDA
