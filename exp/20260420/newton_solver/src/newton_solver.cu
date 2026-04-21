#include "newton_solver.hpp"

#include "edge_fill_jac.cuh"
#include "jacobian_build.hpp"
#include "update_voltage.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace exp20260420::newton_solver {

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess) {                                                          \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err));                                         \
        }                                                                                   \
    } while (0)

namespace {

constexpr int32_t kBlock = 256;

}  // namespace

void newtonDestroy(NewtonWorkspace& ws)
{
    exp20260420::mismatch::mismatchDestroy(ws.mismatch);
    exp20260420::linear_solver::cudssLuDestroy(ws.lu);

    if (ws.pv != nullptr) cudaFree(ws.pv);
    if (ws.pq != nullptr) cudaFree(ws.pq);
    if (ws.pvpq != nullptr) cudaFree(ws.pvpq);
    if (ws.J_row_ptr != nullptr) cudaFree(ws.J_row_ptr);
    if (ws.J_col_idx != nullptr) cudaFree(ws.J_col_idx);
    if (ws.J_values != nullptr) cudaFree(ws.J_values);
    if (ws.offdiagJ11 != nullptr) cudaFree(ws.offdiagJ11);
    if (ws.offdiagJ12 != nullptr) cudaFree(ws.offdiagJ12);
    if (ws.offdiagJ21 != nullptr) cudaFree(ws.offdiagJ21);
    if (ws.offdiagJ22 != nullptr) cudaFree(ws.offdiagJ22);
    if (ws.diagJ11 != nullptr) cudaFree(ws.diagJ11);
    if (ws.diagJ12 != nullptr) cudaFree(ws.diagJ12);
    if (ws.diagJ21 != nullptr) cudaFree(ws.diagJ21);
    if (ws.diagJ22 != nullptr) cudaFree(ws.diagJ22);
    if (ws.sbus_re != nullptr) cudaFree(ws.sbus_re);
    if (ws.sbus_im != nullptr) cudaFree(ws.sbus_im);
    if (ws.v_re != nullptr) cudaFree(ws.v_re);
    if (ws.v_im != nullptr) cudaFree(ws.v_im);
    if (ws.F != nullptr) cudaFree(ws.F);
    if (ws.dx != nullptr) cudaFree(ws.dx);

    ws = NewtonWorkspace{};
}

void newtonAnalyze(NewtonWorkspace& ws,
                   const YbusGraph& host_ybus,
                   const YbusGraph& device_ybus,
                   const int32_t* pv,
                   int32_t n_pv,
                   const int32_t* pq,
                   int32_t n_pq,
                   int32_t batch_size,
                   cudaStream_t stream)
{
    if (host_ybus.n_bus <= 0 || host_ybus.n_edges <= 0 ||
        host_ybus.row == nullptr || host_ybus.col == nullptr || host_ybus.row_ptr == nullptr) {
        throw std::invalid_argument("newtonAnalyze: bad host Ybus graph");
    }
    if (device_ybus.n_bus != host_ybus.n_bus ||
        device_ybus.n_edges != host_ybus.n_edges ||
        device_ybus.row == nullptr || device_ybus.col == nullptr ||
        device_ybus.row_ptr == nullptr || device_ybus.real == nullptr || device_ybus.imag == nullptr) {
        throw std::invalid_argument("newtonAnalyze: bad device Ybus graph");
    }
    if (n_pv < 0 || n_pq < 0 || n_pv + n_pq <= 0 || batch_size <= 0) {
        throw std::invalid_argument("newtonAnalyze: bad pv/pq sizes");
    }
    if ((n_pv > 0 && pv == nullptr) || (n_pq > 0 && pq == nullptr)) {
        throw std::invalid_argument("newtonAnalyze: pv/pq pointers are null");
    }

    newtonDestroy(ws);

    JacobianBuild build = buildJacobian(host_ybus, pv, n_pv, pq, n_pq);

    ws.n_bus = host_ybus.n_bus;
    ws.n_edges = host_ybus.n_edges;
    ws.n_pv = n_pv;
    ws.n_pq = n_pq;
    ws.n_pvpq = n_pv + n_pq;
    ws.batch_size = batch_size;
    ws.dim = build.pattern.dim;
    ws.jac_nnz = build.pattern.nnz;
    ws.ybus = device_ybus;

    if (n_pv > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.pv), static_cast<std::size_t>(n_pv) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(ws.pv, pv, static_cast<std::size_t>(n_pv) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    if (n_pq > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.pq), static_cast<std::size_t>(n_pq) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(ws.pq, pq, static_cast<std::size_t>(n_pq) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.pvpq), static_cast<std::size_t>(ws.n_pvpq) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(ws.pvpq, build.index.pvpq.data(), static_cast<std::size_t>(ws.n_pvpq) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.J_row_ptr), static_cast<std::size_t>(ws.dim + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.J_col_idx), static_cast<std::size_t>(ws.jac_nnz) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.J_values),
                          static_cast<std::size_t>(ws.batch_size) *
                              static_cast<std::size_t>(ws.jac_nnz) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(ws.J_row_ptr, build.pattern.row_ptr.data(), static_cast<std::size_t>(ws.dim + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.J_col_idx, build.pattern.col_idx.data(), static_cast<std::size_t>(ws.jac_nnz) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.offdiagJ11), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.offdiagJ12), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.offdiagJ21), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.offdiagJ22), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(ws.offdiagJ11, build.map.offdiagJ11.data(), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.offdiagJ12, build.map.offdiagJ12.data(), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.offdiagJ21, build.map.offdiagJ21.data(), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.offdiagJ22, build.map.offdiagJ22.data(), static_cast<std::size_t>(ws.n_edges) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.diagJ11), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.diagJ12), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.diagJ21), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.diagJ22), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyAsync(ws.diagJ11, build.map.diagJ11.data(), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.diagJ12, build.map.diagJ12.data(), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.diagJ21, build.map.diagJ21.data(), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.diagJ22, build.map.diagJ22.data(), static_cast<std::size_t>(ws.n_bus) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const std::size_t bus_count =
        static_cast<std::size_t>(ws.batch_size) * static_cast<std::size_t>(ws.n_bus);
    const std::size_t dim_count =
        static_cast<std::size_t>(ws.batch_size) * static_cast<std::size_t>(ws.dim);
    const std::size_t jac_count =
        static_cast<std::size_t>(ws.batch_size) * static_cast<std::size_t>(ws.jac_nnz);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.sbus_re), bus_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.sbus_im), bus_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.v_re), bus_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.v_im), bus_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.F), dim_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.dx), dim_count * sizeof(float)));

    CUDA_CHECK(cudaMemsetAsync(ws.J_values, 0, jac_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.F, 0, dim_count * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.dx, 0, dim_count * sizeof(float), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    exp20260420::mismatch::mismatchAnalyze(
        ws.mismatch,
        device_ybus,
        ws.pv,
        n_pv,
        ws.pq,
        n_pq,
        ws.batch_size,
        stream);

    exp20260420::linear_solver::cudssLuAnalyze(
        ws.lu,
        ws.dim,
        ws.batch_size,
        ws.jac_nnz,
        ws.J_row_ptr,
        ws.J_col_idx,
        ws.J_values,
        ws.dx,
        stream);
}

NewtonResult newtonSolve(NewtonWorkspace& ws,
                         const float* sbus_re,
                         const float* sbus_im,
                         const double* v0_re,
                         const double* v0_im,
                         const NewtonOptions& options,
                         double* out_v_re,
                         double* out_v_im,
                         cudaStream_t stream)
{
    if (ws.n_bus <= 0 || ws.dim <= 0 || ws.J_values == nullptr) {
        throw std::runtime_error("newtonSolve: call newtonAnalyze first");
    }
    if (sbus_re == nullptr || sbus_im == nullptr || v0_re == nullptr || v0_im == nullptr) {
        throw std::invalid_argument("newtonSolve: device input pointer is null");
    }
    if (options.max_iter < 0 || options.tolerance <= 0.0) {
        throw std::invalid_argument("newtonSolve: bad options");
    }
    if ((out_v_re == nullptr) != (out_v_im == nullptr)) {
        throw std::invalid_argument("newtonSolve: output voltage pointers must both be null or both be non-null");
    }

    if (options.batch_size != ws.batch_size) {
        throw std::invalid_argument("newtonSolve: options.batch_size must match analyze batch size");
    }

    const std::size_t bus_count =
        static_cast<std::size_t>(ws.n_bus) * static_cast<std::size_t>(ws.batch_size);
    const std::size_t jac_count =
        static_cast<std::size_t>(ws.jac_nnz) * static_cast<std::size_t>(ws.batch_size);

    CUDA_CHECK(cudaMemcpyAsync(ws.sbus_re, sbus_re, bus_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.sbus_im, sbus_im, bus_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    exp20260420::voltage_update::init_voltage(
        v0_re,
        v0_im,
        ws.v_re,
        ws.v_im,
        ws.n_bus,
        ws.batch_size,
        stream);

    NewtonResult result;

    for (int32_t iter = 0; iter <= options.max_iter; ++iter) {
        const double norm = exp20260420::mismatch::mismatchCompute(
            ws.mismatch,
            ws.v_re,
            ws.v_im,
            ws.sbus_re,
            ws.sbus_im,
            ws.F,
            stream);

        result.iterations = iter;
        result.final_mismatch = norm;
        result.converged = norm <= options.tolerance;
        if (result.converged || iter == options.max_iter) {
            break;
        }

        CUDA_CHECK(cudaMemsetAsync(ws.J_values, 0, jac_count * sizeof(float), stream));

        const dim3 edge_grid((ws.n_edges + kBlock - 1) / kBlock, ws.batch_size);
        fill_jacobian_edge_batch<<<edge_grid, kBlock, 0, stream>>>(
            ws.ybus,
            ws.v_re,
            ws.v_im,
            ws.batch_size,
            ws.offdiagJ11,
            ws.offdiagJ21,
            ws.offdiagJ12,
            ws.offdiagJ22,
            ws.diagJ11,
            ws.diagJ21,
            ws.diagJ12,
            ws.diagJ22,
            ws.jac_nnz,
            ws.J_values);
        CUDA_CHECK(cudaGetLastError());

        exp20260420::linear_solver::cudssLuFactorize(ws.lu, stream);
        exp20260420::linear_solver::cudssLuSolve(ws.lu, ws.F, stream);

        exp20260420::voltage_update::update_voltage(
            ws.v_re,
            ws.v_im,
            ws.dx,
            ws.pvpq,
            ws.n_pvpq,
            ws.pq,
            ws.n_pq,
            ws.n_bus,
            ws.batch_size,
            stream);
    }

    if (out_v_re != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(out_v_re, ws.v_re, bus_count * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(out_v_im, ws.v_im, bus_count * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

}  // namespace exp20260420::newton_solver
