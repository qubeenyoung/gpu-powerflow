#include "benchmark_fill.hpp"

#include "benchmark_cuda.hpp"
#include "common/data_types.hpp"

#include <cuda_runtime.h>

__global__ void fill_jacobian_edge(
    const YbusCoo ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values);

__global__ void fill_jacobian_edge_no_atomic(
    const YbusCoo ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values);

__global__ void fill_jacobian_vertex(
    const YbusCsr ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    int32_t n_pv,
    int32_t n_rows,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values);

namespace exp20260426::jac_asm_bench {

namespace {

float runEdgeFillKernel(const Options& options,
                        const CaseData& data,
                        const EdgeYbusMap& edge_map,
                        const JacobianPattern& pattern,
                        const JacobianMap& map,
                        bool no_atomic)
{
    DeviceBuffer<int32_t> d_row(edge_map.row);
    DeviceBuffer<int32_t> d_col(data.col);
    DeviceBuffer<float> d_y_re(data.y_re);
    DeviceBuffer<float> d_y_im(data.y_im);
    DeviceBuffer<float> d_v_re(data.v_re);
    DeviceBuffer<float> d_v_im(data.v_im);
    DeviceBuffer<float> d_v_norm_re(data.v_norm_re);
    DeviceBuffer<float> d_v_norm_im(data.v_norm_im);

    DeviceBuffer<int32_t> d_J11(map.offdiagJ11);
    DeviceBuffer<int32_t> d_J21(map.offdiagJ21);
    DeviceBuffer<int32_t> d_J12(map.offdiagJ12);
    DeviceBuffer<int32_t> d_J22(map.offdiagJ22);
    DeviceBuffer<int32_t> d_diagJ11(map.diagJ11);
    DeviceBuffer<int32_t> d_diagJ21(map.diagJ21);
    DeviceBuffer<int32_t> d_diagJ12(map.diagJ12);
    DeviceBuffer<int32_t> d_diagJ22(map.diagJ22);
    DeviceBuffer<float> d_values(pattern.nnz);

    const YbusCoo ybus = makeDeviceCoo(data, d_row, d_col, d_y_re, d_y_im);
    const int32_t block = 256;
    const int32_t grid = (data.n_edges + block - 1) / block;

    checkCuda(cudaMemset(d_values.ptr, 0, sizeof(float) * pattern.nnz), "edge memset");

    for (int32_t iter = 0; iter < options.warmup; ++iter) {
        if (no_atomic) {
            fill_jacobian_edge_no_atomic<<<grid, block>>>(
                ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
                d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
                d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
                d_values.ptr);
        } else {
            fill_jacobian_edge<<<grid, block>>>(
                ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
                d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
                d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
                d_values.ptr);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "edge warmup");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaEventCreate(&begin), "edge event begin");
    checkCuda(cudaEventCreate(&end), "edge event end");

    checkCuda(cudaEventRecord(begin), "edge record begin");
    for (int32_t iter = 0; iter < options.iters; ++iter) {
        if (no_atomic) {
            fill_jacobian_edge_no_atomic<<<grid, block>>>(
                ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
                d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
                d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
                d_values.ptr);
        } else {
            fill_jacobian_edge<<<grid, block>>>(
                ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
                d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
                d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
                d_values.ptr);
        }
    }
    checkCuda(cudaEventRecord(end), "edge record end");
    checkCuda(cudaEventSynchronize(end), "edge synchronize");
    checkCuda(cudaGetLastError(), "edge kernel");

    const float ms = elapsedKernelMs(begin, end, options.iters);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms;
}

}  // namespace

float runEdgeFill(const Options& options,
                  const CaseData& data,
                  const EdgeYbusMap& edge_map,
                  const JacobianPattern& pattern,
                  const JacobianMap& map)
{
    return runEdgeFillKernel(options, data, edge_map, pattern, map, false);
}

float runEdgeFillNoAtomic(const Options& options,
                          const CaseData& data,
                          const EdgeYbusMap& edge_map,
                          const JacobianPattern& pattern,
                          const JacobianMap& map)
{
    return runEdgeFillKernel(options, data, edge_map, pattern, map, true);
}

float runVertexFill(const Options& options,
                    const CaseData& data,
                    const BusIndexMap& index,
                    const JacobianPattern& pattern,
                    const JacobianMap& map)
{
    DeviceBuffer<int32_t> d_row_ptr(data.row_ptr);
    DeviceBuffer<int32_t> d_col(data.col);
    DeviceBuffer<float> d_y_re(data.y_re);
    DeviceBuffer<float> d_y_im(data.y_im);
    DeviceBuffer<float> d_v_re(data.v_re);
    DeviceBuffer<float> d_v_im(data.v_im);
    DeviceBuffer<float> d_v_norm_re(data.v_norm_re);
    DeviceBuffer<float> d_v_norm_im(data.v_norm_im);

    DeviceBuffer<int32_t> d_pvpq(index.pvpq);
    DeviceBuffer<int32_t> d_J11(map.offdiagJ11);
    DeviceBuffer<int32_t> d_J21(map.offdiagJ21);
    DeviceBuffer<int32_t> d_J12(map.offdiagJ12);
    DeviceBuffer<int32_t> d_J22(map.offdiagJ22);
    DeviceBuffer<int32_t> d_diagJ11(map.diagJ11);
    DeviceBuffer<int32_t> d_diagJ21(map.diagJ21);
    DeviceBuffer<int32_t> d_diagJ12(map.diagJ12);
    DeviceBuffer<int32_t> d_diagJ22(map.diagJ22);
    DeviceBuffer<float> d_values(pattern.nnz);

    const YbusCsr ybus = makeDeviceCsr(data, d_row_ptr, d_col, d_y_re, d_y_im);
    const int32_t block = 256;
    const int32_t n_pv = index.n_pvpq - index.n_pq;
    const int32_t grid = (index.n_pvpq + block - 1) / block;

    checkCuda(cudaMemset(d_values.ptr, 0, sizeof(float) * pattern.nnz), "vertex memset");

    for (int32_t iter = 0; iter < options.warmup; ++iter) {
        fill_jacobian_vertex<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            n_pv, index.n_pvpq, d_pvpq.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaDeviceSynchronize(), "vertex warmup");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaEventCreate(&begin), "vertex event begin");
    checkCuda(cudaEventCreate(&end), "vertex event end");

    checkCuda(cudaEventRecord(begin), "vertex record begin");
    for (int32_t iter = 0; iter < options.iters; ++iter) {
        fill_jacobian_vertex<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            n_pv, index.n_pvpq, d_pvpq.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaEventRecord(end), "vertex record end");
    checkCuda(cudaEventSynchronize(end), "vertex synchronize");
    checkCuda(cudaGetLastError(), "vertex kernel");

    const float ms = elapsedKernelMs(begin, end, options.iters);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms;
}

}  // namespace exp20260426::jac_asm_bench
