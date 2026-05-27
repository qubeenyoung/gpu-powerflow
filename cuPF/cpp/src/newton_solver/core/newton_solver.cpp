// ---------------------------------------------------------------------------
// newton_solver.cpp
//
// NewtonSolver의 구현.
//
// public I/O는 항상 FP64다. backend·precision 결정은 생성자 시점에 완료되며,
// 이후 hot path는 std::visit를 통해 pipeline으로 위임한다.
// ---------------------------------------------------------------------------

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/pipeline.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"
#include "newton_solver/ops/jacobian/fill_jacobian.hpp"
#include "utils/nvtx_trace.hpp"
#include "utils/timer.hpp"

#include <Eigen/KLUSupport>
#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#ifdef CUPF_WITH_CUDA
#include "newton_solver/ops/linear_solve/cuda_linear_solve_kernels.hpp"
#include "newton_solver/ops/linear_solve/cudss_config.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#endif


namespace {

using Clock = std::chrono::steady_clock;

struct StageScope {
    explicit StageScope(const char* label)
        : range(label)
        , timer(label)
    {}

    newton_solver::utils::ScopedNvtxRange range;
    newton_solver::utils::ScopedTimer     timer;
};

void validate_batch_args(int32_t batch_size, int64_t stride, int32_t n_bus, const char* name)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("NewtonSolver::solve_batch(): batch_size must be positive");
    }
    if (stride < n_bus) {
        throw std::invalid_argument(std::string("NewtonSolver::solve_batch(): ") +
                                    name + " stride must be at least n_bus");
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void validate_adjoint_args(int32_t n_bus,
                           int32_t dimF,
                           int32_t stored_batch_size,
                           const double* grad_va,
                           int64_t grad_va_stride,
                           const double* grad_vm,
                           int64_t grad_vm_stride,
                           int32_t batch_size,
                           const int32_t* pv,
                           int32_t n_pv,
                           const int32_t* pq,
                           int32_t n_pq)
{
    if (n_bus <= 0 || dimF <= 0) {
        throw std::runtime_error("NewtonSolver::solve_adjoint(): solver state is not prepared");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): batch_size must be positive");
    }
    if (batch_size != stored_batch_size) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): batch_size does not match the last forward solve");
    }
    if (grad_va == nullptr || grad_vm == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): grad_va/grad_vm must not be null");
    }
    if (grad_va_stride < n_bus || grad_vm_stride < n_bus) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): grad strides must be at least n_bus");
    }
    if (n_pv > 0 && pv == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pv must not be null");
    }
    if (n_pq > 0 && pq == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pq must not be null");
    }
    if (dimF != n_pv + 2 * n_pq) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint(): pv/pq dimensions do not match solver dimF");
    }
}

std::vector<double> build_grad_state(const double* grad_va,
                                     int64_t grad_va_stride,
                                     const double* grad_vm,
                                     int64_t grad_vm_stride,
                                     int32_t batch_size,
                                     const int32_t* pv,
                                     int32_t n_pv,
                                     const int32_t* pq,
                                     int32_t n_pq)
{
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dimF = n_pvpq + n_pq;
    std::vector<double> grad_state(static_cast<std::size_t>(batch_size) *
                                   static_cast<std::size_t>(dimF), 0.0);
    for (int32_t b = 0; b < batch_size; ++b) {
        const double* va = grad_va + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_va_stride);
        const double* vm = grad_vm + static_cast<std::size_t>(b) * static_cast<std::size_t>(grad_vm_stride);
        double* dst = grad_state.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(dimF);
        for (int32_t i = 0; i < n_pv; ++i) {
            dst[i] = va[pv[i]];
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            dst[n_pv + i] = va[pq[i]];
            dst[n_pvpq + i] = vm[pq[i]];
        }
    }
    return grad_state;
}

void project_load_gradients(const std::vector<double>& lambda,
                            int32_t n_bus,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            AdjointResult& result)
{
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dimF = n_pvpq + n_pq;
    result.grad_load_p.assign(static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(n_bus), 0.0);
    result.grad_load_q.assign(result.grad_load_p.size(), 0.0);

    for (int32_t b = 0; b < batch_size; ++b) {
        const double* lam = lambda.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(dimF);
        double* grad_p = result.grad_load_p.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
        double* grad_q = result.grad_load_q.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);

        for (int32_t i = 0; i < n_pv; ++i) {
            grad_p[pv[i]] = -lam[i];
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            grad_p[pq[i]] = -lam[n_pv + i];
            grad_q[pq[i]] = -lam[n_pvpq + i];
        }
    }
}

struct CsrTransposePattern {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<int32_t> src_to_transpose_pos;
};

CsrTransposePattern build_transpose_pattern(const std::vector<int32_t>& row_ptr,
                                            const std::vector<int32_t>& col_idx,
                                            int32_t dim)
{
    CsrTransposePattern out;
    const int32_t nnz = static_cast<int32_t>(col_idx.size());
    out.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    out.col_idx.assign(static_cast<std::size_t>(nnz), 0);
    out.src_to_transpose_pos.assign(static_cast<std::size_t>(nnz), -1);

    for (int32_t k = 0; k < nnz; ++k) {
        ++out.row_ptr[static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)] + 1)];
    }
    for (int32_t row = 0; row < dim; ++row) {
        out.row_ptr[static_cast<std::size_t>(row + 1)] += out.row_ptr[static_cast<std::size_t>(row)];
    }

    std::vector<int32_t> cursor = out.row_ptr;
    for (int32_t row = 0; row < dim; ++row) {
        for (int32_t k = row_ptr[static_cast<std::size_t>(row)];
             k < row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
            const int32_t col = col_idx[static_cast<std::size_t>(k)];
            const int32_t dst = cursor[static_cast<std::size_t>(col)]++;
            out.col_idx[static_cast<std::size_t>(dst)] = row;
            out.src_to_transpose_pos[static_cast<std::size_t>(k)] = dst;
        }
    }
    return out;
}

template <typename T>
std::vector<T> transpose_batched_values(const std::vector<T>& values,
                                        const std::vector<int32_t>& src_to_transpose_pos,
                                        int32_t batch_size,
                                        int32_t nnz)
{
    std::vector<T> transposed(static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(nnz), T(0));
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz);
        for (int32_t k = 0; k < nnz; ++k) {
            const int32_t dst = src_to_transpose_pos[static_cast<std::size_t>(k)];
            transposed[base + static_cast<std::size_t>(dst)] = values[base + static_cast<std::size_t>(k)];
        }
    }
    return transposed;
}

template <typename T>
double relative_residual_norm_csr(const std::vector<int32_t>& row_ptr,
                                  const std::vector<int32_t>& col_idx,
                                  const std::vector<T>& values,
                                  const std::vector<double>& lambda,
                                  const std::vector<double>& rhs,
                                  int32_t batch_size,
                                  int32_t dim,
                                  int32_t nnz)
{
    long double residual_sq = 0.0L;
    long double rhs_sq = 0.0L;
    for (int32_t b = 0; b < batch_size; ++b) {
        const std::size_t dense_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(dim);
        const std::size_t sparse_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz);
        for (int32_t row = 0; row < dim; ++row) {
            long double acc = 0.0L;
            for (int32_t k = row_ptr[static_cast<std::size_t>(row)];
                 k < row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
                const int32_t col = col_idx[static_cast<std::size_t>(k)];
                acc += static_cast<long double>(values[sparse_base + static_cast<std::size_t>(k)]) *
                       static_cast<long double>(lambda[dense_base + static_cast<std::size_t>(col)]);
            }
            const long double diff = acc - static_cast<long double>(rhs[dense_base + static_cast<std::size_t>(row)]);
            residual_sq += diff * diff;
            const long double r = static_cast<long double>(rhs[dense_base + static_cast<std::size_t>(row)]);
            rhs_sq += r * r;
        }
    }
    const long double denom = std::sqrt(std::max(rhs_sq, 1.0e-60L));
    return static_cast<double>(std::sqrt(residual_sq) / denom);
}

double relative_residual_norm_eigen(const CpuJacobianMatrixF64& matrix,
                                    const std::vector<double>& lambda,
                                    const std::vector<double>& rhs)
{
    using CpuRealVectorF64 = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    Eigen::Map<const CpuRealVectorF64> lam(lambda.data(), matrix.cols());
    Eigen::Map<const CpuRealVectorF64> g(rhs.data(), matrix.rows());
    const CpuRealVectorF64 residual = matrix * lam - g;
    return residual.norm() / std::max(g.norm(), 1.0e-30);
}

#ifdef CUPF_WITH_CUDA
int32_t cuda_batch_size(const CudaFp64Buffers&) { return 1; }
int32_t cuda_batch_size(const CudaFp32Buffers& b) { return b.batch_size; }
int32_t cuda_batch_size(const CudaMixedBuffers& b) { return b.batch_size; }

int32_t cuda_nnz_j(const CudaFp64Buffers& b)
{
    return static_cast<int32_t>(b.d_J_values.size());
}
int32_t cuda_nnz_j(const CudaFp32Buffers& b) { return b.nnz_J; }
int32_t cuda_nnz_j(const CudaMixedBuffers& b) { return b.nnz_J; }

void ensure_cuda_tensor_batch(CudaFp64Buffers& buf, int32_t batch_size)
{
    if (batch_size != 1) {
        throw std::runtime_error("CUDA FP64 torch extension path currently supports batch_size=1; use fp32 or mixed for batched runs");
    }
    const std::size_t bus_count = static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count = static_cast<std::size_t>(buf.dimF);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, 1);
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

void ensure_cuda_tensor_batch(CudaFp32Buffers& buf, int32_t batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("torch extension path requires a positive batch size");
    }
    buf.batch_size = batch_size;
    buf.ybus_values_batched = false;
    const std::size_t bus_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.dimF);
    const std::size_t jacobian_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.nnz_J);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_J_values, jacobian_count);
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, static_cast<std::size_t>(batch_size));
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

void ensure_cuda_tensor_batch(CudaMixedBuffers& buf, int32_t batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("torch extension path requires a positive batch size");
    }
    buf.batch_size = batch_size;
    buf.ybus_values_batched = false;
    const std::size_t bus_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.n_bus);
    const std::size_t residual_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.dimF);
    const std::size_t jacobian_count = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(buf.nnz_J);
    const auto ensure_size = [](auto& b, std::size_t count) {
        if (b.size() != count) b.resize(count);
    };
    ensure_size(buf.d_J_values, jacobian_count);
    ensure_size(buf.d_F, residual_count);
    ensure_size(buf.d_normF, static_cast<std::size_t>(batch_size));
    ensure_size(buf.d_dx, residual_count);
    ensure_size(buf.d_Va, bus_count);
    ensure_size(buf.d_Vm, bus_count);
    ensure_size(buf.d_V_re, bus_count);
    ensure_size(buf.d_V_im, bus_count);
    ensure_size(buf.d_Sbus_re, bus_count);
    ensure_size(buf.d_Sbus_im, bus_count);
    ensure_size(buf.d_Ibus_re, bus_count);
    ensure_size(buf.d_Ibus_im, bus_count);
}

template <typename PipelineT>
const char* cuda_pipeline_dtype_name()
{
    if constexpr (std::is_same_v<PipelineT, CudaFp64Pipeline>) {
        return "float64";
    } else {
        return "float32";
    }
}

template <typename PipelineT>
const char* cuda_pipeline_backend_name()
{
    if constexpr (std::is_same_v<PipelineT, CudaFp64Pipeline>) {
        return "cuda_cudss_fp64";
    } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
        return "cuda_cudss_fp32";
    } else {
        return "cuda_cudss_mixed";
    }
}

template <typename PipelineT>
void fill_common_cuda_adjoint_metadata(PipelineT& p,
                                       AdjointResult& result,
                                       bool raw_pointer_api_used)
{
    result.success = true;
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = p.adjoint_cache.has_adjoint_cache;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = p.adjoint_cache.used_explicit_transpose;
    result.used_python_scipy = false;
    result.includes_host_device_transfer = false;
    result.zero_copy = true;
    result.torch_extension_zero_copy = !raw_pointer_api_used;
    result.raw_pointer_api_used = raw_pointer_api_used;
    result.current_stream_integrated = true;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = cuda_batch_size(p.buf);
    result.dimF = p.buf.dimF;
    result.backend = p.adjoint_cache.backend_name.empty()
        ? cuda_pipeline_backend_name<PipelineT>()
        : p.adjoint_cache.backend_name;
    result.transpose_solve_backend = p.adjoint_cache.transpose_solve_backend_name;
}

#ifdef CUPF_ENABLE_CUDSS
template <typename T> cudaDataType_t cudss_value_type_adjoint();
template <> cudaDataType_t cudss_value_type_adjoint<double>() { return CUDA_R_64F; }
template <> cudaDataType_t cudss_value_type_adjoint<float>() { return CUDA_R_32F; }

struct LocalCudssState {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs_matrix = nullptr;
    cudssMatrix_t solution_matrix = nullptr;

    ~LocalCudssState()
    {
        if (matrix) cudssMatrixDestroy(matrix);
        if (rhs_matrix) cudssMatrixDestroy(rhs_matrix);
        if (solution_matrix) cudssMatrixDestroy(solution_matrix);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }
};
#endif

template <typename T>
void solve_cudss_transpose_csr(const std::vector<int32_t>& row_ptr,
                               const std::vector<int32_t>& col_idx,
                               const std::vector<T>& values,
                               const std::vector<double>& rhs_double,
                               int32_t dim,
                               int32_t batch_size,
                               const CuDSSOptions& cudss_options,
                               std::vector<double>& lambda,
                               double& factorization_time_ms,
                               double& solve_time_ms)
{
#ifndef CUPF_ENABLE_CUDSS
    (void)row_ptr;
    (void)col_idx;
    (void)values;
    (void)rhs_double;
    (void)dim;
    (void)batch_size;
    (void)cudss_options;
    (void)lambda;
    (void)factorization_time_ms;
    (void)solve_time_ms;
    throw std::runtime_error("NewtonSolver::solve_adjoint(): cuDSS is not enabled");
#else
    const int32_t nnz = static_cast<int32_t>(col_idx.size());
    std::vector<T> rhs(rhs_double.size(), T(0));
    for (std::size_t i = 0; i < rhs_double.size(); ++i) {
        rhs[i] = static_cast<T>(rhs_double[i]);
    }
    std::vector<T> solution(rhs.size(), T(0));

    DeviceBuffer<int32_t> d_row_ptr;
    DeviceBuffer<int32_t> d_col_idx;
    DeviceBuffer<T> d_values;
    DeviceBuffer<T> d_rhs;
    DeviceBuffer<T> d_solution;

    d_row_ptr.assign(row_ptr.data(), row_ptr.size());
    d_col_idx.assign(col_idx.data(), col_idx.size());
    d_values.assign(values.data(), values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    d_solution.resize(solution.size());
    d_solution.memsetZero();

    LocalCudssState state;
    CUDSS_CHECK(cudssCreate(&state.handle));
    cupf_cudss_detail::configure_handle(state.handle);
    CUDSS_CHECK(cudssConfigCreate(&state.config));
    CUDSS_CHECK(cudssDataCreate(state.handle, &state.data));
    cupf_cudss_detail::configure_solver(state.config, cudss_options, batch_size);

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state.matrix,
        dim, dim, static_cast<int64_t>(nnz),
        d_row_ptr.data(), nullptr, d_col_idx.data(), d_values.data(),
        CUDA_R_32I, cudss_value_type_adjoint<T>(),
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state.rhs_matrix,
        dim, 1, dim, d_rhs.data(),
        cudss_value_type_adjoint<T>(), CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state.solution_matrix,
        dim, 1, dim, d_solution.data(),
        cudss_value_type_adjoint<T>(), CUDSS_LAYOUT_COL_MAJOR));

    auto factor_start = Clock::now();
    CUDSS_CHECK(cudssExecute(
        state.handle, CUDSS_PHASE_ANALYSIS,
        state.config, state.data,
        state.matrix, state.solution_matrix, state.rhs_matrix));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDSS_CHECK(cudssExecute(
        state.handle, CUDSS_PHASE_FACTORIZATION,
        state.config, state.data,
        state.matrix, state.solution_matrix, state.rhs_matrix));
    CUDA_CHECK(cudaDeviceSynchronize());
    factorization_time_ms = elapsed_ms(factor_start, Clock::now());

    auto solve_start = Clock::now();
    CUDSS_CHECK(cudssExecute(
        state.handle, CUDSS_PHASE_SOLVE,
        state.config, state.data,
        state.matrix, state.solution_matrix, state.rhs_matrix));
    CUDA_CHECK(cudaDeviceSynchronize());
    solve_time_ms = elapsed_ms(solve_start, Clock::now());

    d_solution.copyTo(solution.data(), solution.size());
    lambda.assign(solution.size(), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        lambda[i] = static_cast<double>(solution[i]);
    }
#endif
}

template <typename ValueT, typename Buffers>
void solve_cuda_adjoint_from_buffer(Buffers& buf,
                                    const std::vector<double>& grad_state,
                                    int32_t batch_size,
                                    const CuDSSOptions& cudss_options,
                                    AdjointResult& result)
{
    const int32_t dim = buf.dimF;
    const int32_t nnz = cuda_nnz_j(buf);
    std::vector<int32_t> row_ptr(static_cast<std::size_t>(dim + 1));
    std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
    std::vector<ValueT> values(static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(nnz));

    buf.d_J_row_ptr.copyTo(row_ptr.data(), row_ptr.size());
    buf.d_J_col_idx.copyTo(col_idx.data(), col_idx.size());
    buf.d_J_values.copyTo(values.data(), values.size());

    const CsrTransposePattern transpose = build_transpose_pattern(row_ptr, col_idx, dim);
    std::vector<ValueT> jt_values =
        transpose_batched_values(values, transpose.src_to_transpose_pos, batch_size, nnz);

    solve_cudss_transpose_csr(transpose.row_ptr,
                              transpose.col_idx,
                              jt_values,
                              grad_state,
                              dim,
                              batch_size,
                              cudss_options,
                              result.lambda,
                              result.factorization_time_ms,
                              result.solve_time_ms);

    result.jt_residual_norm = relative_residual_norm_csr(
        transpose.row_ptr,
        transpose.col_idx,
        jt_values,
        result.lambda,
        grad_state,
        batch_size,
        dim,
        nnz);
}
#endif  // CUPF_WITH_CUDA

void solve_adjoint_pipeline(CpuFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions&,
                            AdjointResult& result)
{
    validate_adjoint_args(p.buf.n_bus, p.buf.dimF, 1,
                          grad_va, grad_va_stride,
                          grad_vm, grad_vm_stride,
                          batch_size, pv, n_pv, pq, n_pq);
    const auto total_start = Clock::now();
    result.backend = "cpu_klu";
    result.transpose_solve_backend = "cpu_klu_tsolve_cached_factorization";
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = false;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = false;
    result.includes_host_device_transfer = false;
    result.used_python_scipy = false;
    result.zero_copy = false;
    result.torch_extension_zero_copy = false;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = false;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = 1;
    result.dimF = p.buf.dimF;

    const std::vector<double> grad_state =
        build_grad_state(grad_va, grad_va_stride, grad_vm, grad_vm_stride,
                         batch_size, pv, n_pv, pq, n_pq);

    const bool cache_ok =
        p.adjoint_cache.has_adjoint_cache &&
        p.adjoint_cache.adjoint_cache_matches_final_state &&
        p.adjoint_cache.factorization_supports_transpose_solve;
    const bool allow_backward_refactor =
        options.allow_refactorize && options.allow_refactorize_for_backward;
    if (!cache_ok) {
        if (options.require_cached_factorization || !allow_backward_refactor) {
            throw std::runtime_error(
                "NewtonSolver::solve_adjoint(): missing exact final-state adjoint cache");
        }
        NRConfig cfg;
        IterationContext ctx{
            .config = cfg,
            .pv = pv, .n_pv = n_pv,
            .pq = pq, .n_pq = n_pq,
        };
        CpuIbusOp{}.run(p.buf, ctx);
        CpuJacobianOpF64{}.run(p.buf, ctx);
        const auto factor_start = Clock::now();
        p.linear_solve.factorize(p.buf, ctx);
        result.factorization_time_ms = elapsed_ms(factor_start, Clock::now());
        result.refactorized_for_backward = true;
        result.jt_refactorized_during_backward = true;
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.factorization_supports_transpose_solve = true;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.used_explicit_transpose = false;
        p.adjoint_cache.backend_name = "cpu_klu";
        p.adjoint_cache.transpose_solve_backend_name = "cpu_klu_tsolve_cached_factorization";
        p.adjoint_cache.batch_size = 1;
        p.adjoint_cache.dimF = p.buf.dimF;
    } else {
        result.factorization_time_ms = 0.0;
        result.reused_final_state_factorization = true;
    }

    auto solve_start = Clock::now();
    result.lambda.assign(static_cast<std::size_t>(p.buf.dimF), 0.0);
    p.linear_solve.solve_transpose(grad_state.data(), result.lambda.data(), p.buf.dimF, 1);
    result.solve_time_ms = elapsed_ms(solve_start, Clock::now());
    result.transpose_solve_time_ms = result.solve_time_ms;

    result.used_adjoint_cache = cache_ok;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    if (options.check_residual) {
        CpuJacobianMatrixF64 jt = p.buf.J.transpose();
        jt.makeCompressed();
        result.jt_residual_norm = relative_residual_norm_eigen(jt, result.lambda, grad_state);
    }
    if (options.compute_load_gradients) {
        project_load_gradients(result.lambda, p.buf.n_bus, 1, pv, n_pv, pq, n_pq, result);
    }
    result.total_time_ms = elapsed_ms(total_start, Clock::now());
    result.success = true;
}

#ifdef CUPF_WITH_CUDA
template <typename PipelineT, typename ValueT>
void solve_adjoint_cuda_pipeline(PipelineT& p,
                                 const double* grad_va,
                                 int64_t grad_va_stride,
                                 const double* grad_vm,
                                 int64_t grad_vm_stride,
                                 int32_t batch_size,
                                 const int32_t* pv,
                                 int32_t n_pv,
                                 const int32_t* pq,
                                 int32_t n_pq,
                                 const AdjointOptions& options,
                                 const CuDSSOptions& cudss_options,
                                 const char* backend_name,
                                 AdjointResult& result)
{
    validate_adjoint_args(p.buf.n_bus, p.buf.dimF, cuda_batch_size(p.buf),
                          grad_va, grad_va_stride,
                          grad_vm, grad_vm_stride,
                          batch_size, pv, n_pv, pq, n_pq);
    const auto total_start = Clock::now();
    result.backend = backend_name;
    result.transpose_solve_backend = "cuda_cudss_cached_explicit_transpose_factorization";
    result.used_adjoint_cache = p.adjoint_cache.has_adjoint_cache;
    result.adjoint_cache_matches_final_state = p.adjoint_cache.adjoint_cache_matches_final_state;
    result.reused_forward_factorization = p.adjoint_cache.reused_forward_factorization;
    result.reused_final_state_factorization = false;
    result.refactorized_for_backward = false;
    result.used_explicit_transpose = true;
    result.includes_host_device_transfer = true;
    result.used_python_scipy = false;
    result.zero_copy = false;
    result.torch_extension_zero_copy = false;
    result.raw_pointer_api_used = false;
    result.current_stream_integrated = false;
    result.jt_symbolic_analyzed_at_initialize = p.adjoint_cache.jt_symbolic_analyzed_at_initialize;
    result.jt_values_transposed_on_device = p.adjoint_cache.jt_values_transposed_on_device;
    result.jt_factorized_during_forward_cache = p.adjoint_cache.jt_factorized_during_forward_cache;
    result.jt_refactorized_during_backward = false;
    result.host_roundtrip_for_jt_transpose = p.adjoint_cache.host_roundtrip_for_jt_transpose;
    result.n_bus = p.buf.n_bus;
    result.batch_size = batch_size;
    result.dimF = p.buf.dimF;

    const std::vector<double> grad_state =
        build_grad_state(grad_va, grad_va_stride, grad_vm, grad_vm_stride,
                         batch_size, pv, n_pv, pq, n_pq);

    const bool cache_ok =
        p.adjoint_cache.has_adjoint_cache &&
        p.adjoint_cache.adjoint_cache_matches_final_state &&
        p.linear_solve.has_adjoint_cache();
    const bool allow_backward_refactor =
        options.allow_refactorize && options.allow_refactorize_for_backward;
    if (!cache_ok) {
        if (options.require_cached_factorization || !allow_backward_refactor ||
            !options.allow_explicit_transpose_fallback) {
            throw std::runtime_error(
                "NewtonSolver::solve_adjoint(): missing exact CUDA adjoint cache; cuDSS transpose solve is unsupported, and explicit transpose fallback is not enabled");
        }
        NRConfig cfg;
        IterationContext ctx{
            .config = cfg,
            .pv = pv, .n_pv = n_pv,
            .pq = pq, .n_pq = n_pq,
        };
        p.ibus(ctx);
        p.jacobian(ctx);
        p.linear_solve.prepare_adjoint_explicit_transpose_cache(
            p.buf, ctx, result.factorization_time_ms);
        result.refactorized_for_backward = true;
        result.jt_refactorized_during_backward = true;
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.factorization_supports_transpose_solve = false;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.used_explicit_transpose = true;
        p.adjoint_cache.includes_host_device_transfer = false;
        p.adjoint_cache.backend_name = backend_name;
        p.adjoint_cache.transpose_solve_backend_name =
            "cuda_cudss_cached_explicit_transpose_factorization";
        p.adjoint_cache.batch_size = batch_size;
        p.adjoint_cache.dimF = p.buf.dimF;
    } else {
        result.factorization_time_ms = 0.0;
        result.reused_final_state_factorization = true;
    }

    result.lambda.assign(static_cast<std::size_t>(batch_size) *
                         static_cast<std::size_t>(p.buf.dimF), 0.0);
    p.linear_solve.solve_adjoint_explicit_transpose_host(
        grad_state.data(),
        result.lambda.data(),
        batch_size,
        result.solve_time_ms);
    result.transpose_solve_time_ms = result.solve_time_ms;

    if (options.check_residual) {
        const int32_t dim = p.buf.dimF;
        const int32_t nnz = cuda_nnz_j(p.buf);
        std::vector<int32_t> row_ptr(static_cast<std::size_t>(dim + 1));
        std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
        std::vector<ValueT> values(static_cast<std::size_t>(batch_size) *
                                   static_cast<std::size_t>(nnz));
        p.buf.d_J_row_ptr.copyTo(row_ptr.data(), row_ptr.size());
        p.buf.d_J_col_idx.copyTo(col_idx.data(), col_idx.size());
        p.buf.d_J_values.copyTo(values.data(), values.size());
        const CsrTransposePattern transpose = build_transpose_pattern(row_ptr, col_idx, dim);
        const std::vector<ValueT> jt_values =
            transpose_batched_values(values, transpose.src_to_transpose_pos, batch_size, nnz);
        result.jt_residual_norm = relative_residual_norm_csr(
            transpose.row_ptr, transpose.col_idx, jt_values,
            result.lambda, grad_state, batch_size, dim, nnz);
    } else {
        result.jt_residual_norm = 0.0;
    }
    if (options.compute_load_gradients) {
        project_load_gradients(result.lambda, p.buf.n_bus, batch_size, pv, n_pv, pq, n_pq, result);
    }
    result.total_time_ms = elapsed_ms(total_start, Clock::now());
    result.success = true;
}

void solve_adjoint_pipeline(CudaFp64Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaFp64Pipeline, double>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_fp64", result);
}

void solve_adjoint_pipeline(CudaFp32Pipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaFp32Pipeline, float>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_fp32", result);
}

void solve_adjoint_pipeline(CudaMixedPipeline& p,
                            const double* grad_va,
                            int64_t grad_va_stride,
                            const double* grad_vm,
                            int64_t grad_vm_stride,
                            int32_t batch_size,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            const AdjointOptions& options,
                            const CuDSSOptions& cudss_options,
                            AdjointResult& result)
{
    solve_adjoint_cuda_pipeline<CudaMixedPipeline, float>(
        p, grad_va, grad_va_stride, grad_vm, grad_vm_stride,
        batch_size, pv, n_pv, pq, n_pq, options, cudss_options, "cuda_cudss_mixed", result);
}
#endif  // CUPF_WITH_CUDA

}  // namespace


// ---------------------------------------------------------------------------
// Constructor: assemble the pipeline from NewtonOptions.
// ---------------------------------------------------------------------------
NewtonSolver::NewtonSolver(const NewtonOptions& options)
    : options_(options)
{
    if (options.backend == BackendKind::CPU) {
        pipeline_ = std::make_unique<SolverPipeline>(
            SolverPipeline{CpuFp64Pipeline{}});
        return;
    }

#ifdef CUPF_WITH_CUDA
    if (options.backend == BackendKind::CUDA) {
        if (options.compute == ComputePolicy::FP64) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaFp64Pipeline{options.cudss}});
            return;
        }
        if (options.compute == ComputePolicy::FP32) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaFp32Pipeline{options.cudss}});
            return;
        }
        if (options.compute == ComputePolicy::Mixed) {
            pipeline_ = std::make_unique<SolverPipeline>(
                SolverPipeline{CudaMixedPipeline{options.cudss}});
            return;
        }
    }
#else
    if (options.backend == BackendKind::CUDA) {
        throw std::invalid_argument(
            "NewtonSolver: CUDA backend를 요청했지만 cuPF가 CUDA 없이 빌드되었습니다.");
    }
#endif

    throw std::invalid_argument(
        "NewtonSolver: 지원하지 않는 backend/compute 조합입니다.");
}


NewtonSolver::~NewtonSolver() = default;


// ---------------------------------------------------------------------------
// initialize: Jacobian analysis → pipeline::initialize (prepare + KLU/cuDSS)
// ---------------------------------------------------------------------------
void NewtonSolver::initialize(
    const YbusView& ybus,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    StageScope total("NR.initialize.total");

    JacobianPattern pattern;
    JacobianScatterMap scatter_map;
    {
        StageScope stage("NR.initialize.jacobian_analysis");
        const JacobianIndexing indexing =
            make_jacobian_indexing(ybus.rows, pv, n_pv, pq, n_pq);
        pattern = JacobianPatternGenerator().generate(ybus, indexing);
        scatter_map = JacobianMapBuilder().build(ybus, indexing, pattern);
    }

    InitializeContext ctx{
        .ybus  = ybus,
        .maps  = scatter_map,
        .J     = pattern,
        .n_bus = ybus.rows,
        .pv    = pv, .n_pv = n_pv,
        .pq    = pq, .n_pq = n_pq,
    };

    {
        StageScope stage("NR.initialize.pipeline");
        std::visit([&](auto& p) { p.initialize(ctx); }, pipeline_->v);
    }

    initialized_ = true;
}


// ---------------------------------------------------------------------------
// solve: single-case wrapper around solve_batch
// ---------------------------------------------------------------------------
void NewtonSolver::solve(
    const YbusView&          ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    const SolveOptions&         solve_options,
    NRResult&                result)
{
    NRBatchResult batch_result;
    solve_batch(ybus, sbus, ybus.rows, V0, ybus.rows, 1,
                pv, n_pv, pq, n_pq, config, solve_options, batch_result);

    result = {};
    result.V = std::move(batch_result.V);
    result.iterations    = batch_result.iterations.empty()     ? 0    : batch_result.iterations[0];
    result.final_mismatch= batch_result.final_mismatch.empty() ? 0.0  : batch_result.final_mismatch[0];
    result.converged     = !batch_result.converged.empty() && batch_result.converged[0] != 0;
}


void NewtonSolver::solve_batch(
    const YbusView&          ybus,
    const std::complex<double>* sbus,
    int64_t                     sbus_stride,
    const std::complex<double>* V0,
    int64_t                     V0_stride,
    int32_t                     batch_size,
    const int32_t*              pv, int32_t n_pv,
    const int32_t*              pq, int32_t n_pq,
    const NRConfig&             config,
    const SolveOptions&         solve_options,
    NRBatchResult&           result)
{
    StageScope total("NR.solve.total");

    if (!initialized_) {
        throw std::runtime_error(
            "NewtonSolver::solve_batch(): initialize()를 먼저 호출해야 합니다.");
    }

    validate_batch_args(batch_size, sbus_stride, ybus.rows, "sbus");
    validate_batch_args(batch_size, V0_stride,   ybus.rows, "V0");

    if (batch_size != 1) {
        const bool supported = std::visit(
            [](const auto& p) { return p.batch_supported; }, pipeline_->v);
        if (!supported) {
            throw std::runtime_error(
                "NewtonSolver::solve_batch(): batch_size > 1 is currently supported "
                "only by the CUDA Mixed pipeline.");
        }
    }

    result = {};
    result.n_bus       = ybus.rows;
    result.batch_size  = batch_size;

    {
        StageScope stage("NR.solve.upload");
        std::visit([&](auto& p) { p.adjoint_cache = AdjointCache{}; }, pipeline_->v);
        SolveContext upload_ctx{
            .ybus   = &ybus,
            .sbus   = sbus,
            .V0     = V0,
            .config = &config,
            .batch_size          = batch_size,
            .sbus_stride         = sbus_stride,
            .V0_stride           = V0_stride,
            .ybus_values_batched = false,
            .ybus_value_stride   = ybus.nnz,
        };
        std::visit([&](auto& p) { p.upload(upload_ctx); }, pipeline_->v);
    }

    IterationContext iter_ctx{
        .config = config,
        .pv     = pv, .n_pv = n_pv,
        .pq     = pq, .n_pq = n_pq,
    };

    const int32_t iterations = run_iteration_stages(iter_ctx);

    if (solve_options.prepare_adjoint_cache) {
        StageScope stage("NR.solve.prepare_adjoint_cache");
        prepare_adjoint_cache(iter_ctx, solve_options, iter_ctx.normF);
    }

    {
        StageScope stage("NR.solve.download");
        std::visit([&](auto& p) { p.download_batch(result); }, pipeline_->v);
    }

    result.n_bus      = ybus.rows;
    result.batch_size = batch_size;
    result.iterations.assign(static_cast<std::size_t>(batch_size), iterations);

    if (result.final_mismatch.size() != static_cast<std::size_t>(batch_size)) {
        result.final_mismatch.assign(static_cast<std::size_t>(batch_size), iter_ctx.normF);
    }
    result.converged.resize(static_cast<std::size_t>(batch_size));
    for (int32_t b = 0; b < batch_size; ++b) {
        const double norm =
            result.final_mismatch.empty()
                ? iter_ctx.normF
                : result.final_mismatch[static_cast<std::size_t>(b)];
        result.converged[static_cast<std::size_t>(b)] =
            static_cast<uint8_t>(norm <= config.tolerance ? 1 : 0);
    }
}


void NewtonSolver::solve_adjoint(const double*        grad_va,
                                 int64_t              grad_va_stride,
                                 const double*        grad_vm,
                                 int64_t              grad_vm_stride,
                                 int32_t              batch_size,
                                 const int32_t*       pv, int32_t n_pv,
                                 const int32_t*       pq, int32_t n_pq,
                                 const AdjointOptions& options,
                                 AdjointResult&       result)
{
    StageScope total("NR.adjoint.total");

    if (!initialized_) {
        throw std::runtime_error(
            "NewtonSolver::solve_adjoint(): initialize() and solve()/solve_batch() must be called first");
    }
    if (options.reuse_forward_factorization && !options.allow_refactorize) {
        throw std::runtime_error(
            "NewtonSolver::solve_adjoint(): forward factorization reuse is not available for the final converged state; "
            "enable allow_refactorize");
    }

    result = {};
    std::visit([&](auto& p) {
        solve_adjoint_pipeline(p,
                               grad_va,
                               grad_va_stride,
                               grad_vm,
                               grad_vm_stride,
                               batch_size,
                               pv, n_pv,
                               pq, n_pq,
                               options,
                               options_.cudss,
                               result);
    }, pipeline_->v);
}


void NewtonSolver::solve_adjoint_cuda_raw(uintptr_t grad_va_device_ptr,
                                          uintptr_t grad_vm_device_ptr,
                                          uintptr_t grad_load_p_device_ptr,
                                          uintptr_t grad_load_q_device_ptr,
                                          int32_t batch_size,
                                          int32_t n_bus,
                                          const char* dtype,
                                          const AdjointOptions& options,
                                          AdjointResult& result)
{
    solve_adjoint_cuda_raw_unsafe(grad_va_device_ptr,
                                  grad_vm_device_ptr,
                                  grad_load_p_device_ptr,
                                  grad_load_q_device_ptr,
                                  batch_size,
                                  n_bus,
                                  dtype,
                                  options,
                                  result);
}


void NewtonSolver::solve_adjoint_cuda_raw_unsafe(uintptr_t grad_va_device_ptr,
                                                 uintptr_t grad_vm_device_ptr,
                                                 uintptr_t grad_load_p_device_ptr,
                                                 uintptr_t grad_load_q_device_ptr,
                                                 int32_t batch_size,
                                                 int32_t n_bus,
                                                 const char* dtype,
                                                 const AdjointOptions& options,
                                                 AdjointResult& result)
{
    solve_adjoint_cuda_device(reinterpret_cast<const void*>(grad_va_device_ptr),
                              reinterpret_cast<const void*>(grad_vm_device_ptr),
                              reinterpret_cast<void*>(grad_load_p_device_ptr),
                              reinterpret_cast<void*>(grad_load_q_device_ptr),
                              batch_size,
                              n_bus,
                              dtype,
                              options,
                              true,
                              result);
}


void NewtonSolver::solve_adjoint_cuda_device(const void* grad_va_device_ptr,
                                             const void* grad_vm_device_ptr,
                                             void* grad_load_p_device_ptr,
                                             void* grad_load_q_device_ptr,
                                             int32_t batch_size,
                                             int32_t n_bus,
                                             const char* dtype,
                                             const AdjointOptions& options,
                                             bool raw_pointer_api_used,
                                             AdjointResult& result)
{
#ifndef CUPF_WITH_CUDA
    (void)grad_va_device_ptr;
    (void)grad_vm_device_ptr;
    (void)grad_load_p_device_ptr;
    (void)grad_load_q_device_ptr;
    (void)batch_size;
    (void)n_bus;
    (void)dtype;
    (void)options;
    (void)raw_pointer_api_used;
    (void)result;
    throw std::runtime_error("NewtonSolver::solve_adjoint_cuda_device requires a CUDA build");
#else
    if (!initialized_) {
        throw std::runtime_error("NewtonSolver::solve_adjoint_cuda_device(): initialize() and forward solve must be called first");
    }
    if (grad_va_device_ptr == nullptr || grad_vm_device_ptr == nullptr ||
        grad_load_p_device_ptr == nullptr || grad_load_q_device_ptr == nullptr) {
        throw std::invalid_argument("NewtonSolver::solve_adjoint_cuda_device(): null device pointer");
    }
    const std::string dtype_name = dtype ? std::string(dtype) : std::string();

    result = {};
    std::visit([&](auto& p) {
        using PipelineT = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<PipelineT, CpuFp64Pipeline>) {
            throw std::runtime_error("NewtonSolver::solve_adjoint_cuda_device(): CPU pipeline is not supported");
        } else {
            using ValueT = std::conditional_t<
                std::is_same_v<PipelineT, CudaFp64Pipeline>, double, float>;
            const char* expected_dtype = cuda_pipeline_dtype_name<PipelineT>();
            if (dtype_name != expected_dtype) {
                throw std::invalid_argument(
                    std::string("NewtonSolver::solve_adjoint_cuda_device(): dtype must be ") +
                    expected_dtype + " for this cuPF compute policy");
            }
            if (batch_size != cuda_batch_size(p.buf) || n_bus != p.buf.n_bus) {
                throw std::invalid_argument("NewtonSolver::solve_adjoint_cuda_device(): shape does not match cached forward solve");
            }
            const bool cache_ok =
                p.adjoint_cache.has_adjoint_cache &&
                p.adjoint_cache.adjoint_cache_matches_final_state &&
                p.linear_solve.has_adjoint_cache();
            if (!cache_ok) {
                throw std::runtime_error(
                    "NewtonSolver::solve_adjoint_cuda_device(): missing exact cached adjoint factorization");
            }

            const auto total_start = Clock::now();
            const ValueT* grad_va = reinterpret_cast<const ValueT*>(grad_va_device_ptr);
            const ValueT* grad_vm = reinterpret_cast<const ValueT*>(grad_vm_device_ptr);
            ValueT* grad_p = reinterpret_cast<ValueT*>(grad_load_p_device_ptr);
            ValueT* grad_q = reinterpret_cast<ValueT*>(grad_load_q_device_ptr);
            const int32_t n_pv = p.buf.n_pvpq - p.buf.n_pq;

            launch_gather_adjoint_rhs<ValueT>(
                grad_va, grad_vm, p.linear_solve.adjoint_rhs_data(),
                p.buf.d_pv.data(), n_pv,
                p.buf.d_pq.data(), p.buf.n_pq,
                p.buf.n_bus, batch_size);
            p.linear_solve.solve_adjoint_explicit_transpose_cached(result.solve_time_ms);
            result.transpose_solve_time_ms = result.solve_time_ms;
            launch_project_load_gradients<ValueT>(
                p.linear_solve.adjoint_solution_data(),
                grad_p,
                grad_q,
                p.buf.d_pv.data(), n_pv,
                p.buf.d_pq.data(), p.buf.n_pq,
                p.buf.n_bus, batch_size);

            fill_common_cuda_adjoint_metadata(p, result, raw_pointer_api_used);
            result.used_adjoint_cache = true;
            result.adjoint_cache_matches_final_state = true;
            result.reused_final_state_factorization = true;
            result.factorization_time_ms = 0.0;
            result.total_time_ms = elapsed_ms(total_start, Clock::now());
            if (raw_pointer_api_used) {
                result.transpose_solve_backend += "_raw_pointer_unsafe";
            } else {
                result.transpose_solve_backend += "_torch_extension";
            }
        }
    }, pipeline_->v);
#endif
}


void NewtonSolver::solve_cuda_load_pq_device(const void* sbus_base_re_device_ptr,
                                             const void* sbus_base_im_device_ptr,
                                             const void* load_p_device_ptr,
                                             const void* load_q_device_ptr,
                                             const void* v0_va_device_ptr,
                                             const void* v0_vm_device_ptr,
                                             void* va_out_device_ptr,
                                             void* vm_out_device_ptr,
                                             int32_t batch_size,
                                             int32_t n_bus,
                                             const char* dtype,
                                             const NRConfig& config,
                                             const SolveOptions& solve_options,
                                             AdjointResult& result)
{
#ifndef CUPF_WITH_CUDA
    (void)sbus_base_re_device_ptr;
    (void)sbus_base_im_device_ptr;
    (void)load_p_device_ptr;
    (void)load_q_device_ptr;
    (void)v0_va_device_ptr;
    (void)v0_vm_device_ptr;
    (void)va_out_device_ptr;
    (void)vm_out_device_ptr;
    (void)batch_size;
    (void)n_bus;
    (void)dtype;
    (void)config;
    (void)solve_options;
    (void)result;
    throw std::runtime_error("NewtonSolver::solve_cuda_load_pq_device requires a CUDA build");
#else
    if (!initialized_) {
        throw std::runtime_error("NewtonSolver::solve_cuda_load_pq_device(): initialize() must be called first");
    }
    if (!sbus_base_re_device_ptr || !sbus_base_im_device_ptr || !load_p_device_ptr ||
        !load_q_device_ptr || !v0_va_device_ptr || !v0_vm_device_ptr ||
        !va_out_device_ptr || !vm_out_device_ptr) {
        throw std::invalid_argument("NewtonSolver::solve_cuda_load_pq_device(): null device pointer");
    }
    const std::string dtype_name = dtype ? std::string(dtype) : std::string();
    const auto total_start = Clock::now();

    result = {};
    std::visit([&](auto& p) {
        using PipelineT = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<PipelineT, CpuFp64Pipeline>) {
            throw std::runtime_error("NewtonSolver::solve_cuda_load_pq_device(): CPU pipeline is not supported");
        } else {
            const char* expected_dtype = cuda_pipeline_dtype_name<PipelineT>();
            if (dtype_name != expected_dtype) {
                throw std::invalid_argument(
                    std::string("NewtonSolver::solve_cuda_load_pq_device(): dtype must be ") +
                    expected_dtype + " for this cuPF compute policy");
            }
            if (n_bus != p.buf.n_bus) {
                throw std::invalid_argument("NewtonSolver::solve_cuda_load_pq_device(): n_bus does not match initialized solver");
            }
            if (batch_size != 1 && !PipelineT::batch_supported) {
                throw std::runtime_error("NewtonSolver::solve_cuda_load_pq_device(): requested batch size is not supported by this CUDA pipeline");
            }

            p.adjoint_cache = AdjointCache{};
            ensure_cuda_tensor_batch(p.buf, batch_size);
            p.buf.d_F.memsetZero();
            p.buf.d_normF.memsetZero();
            p.buf.d_dx.memsetZero();
            p.buf.d_J_values.memsetZero();
            p.buf.d_Ibus_re.memsetZero();
            p.buf.d_Ibus_im.memsetZero();

            if constexpr (std::is_same_v<PipelineT, CudaFp64Pipeline>) {
                const double* base_re = static_cast<const double*>(sbus_base_re_device_ptr);
                const double* base_im = static_cast<const double*>(sbus_base_im_device_ptr);
                const double* load_p = static_cast<const double*>(load_p_device_ptr);
                const double* load_q = static_cast<const double*>(load_q_device_ptr);
                const double* v0_va = static_cast<const double*>(v0_va_device_ptr);
                const double* v0_vm = static_cast<const double*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<double, double>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
                const float* base_re = static_cast<const float*>(sbus_base_re_device_ptr);
                const float* base_im = static_cast<const float*>(sbus_base_im_device_ptr);
                const float* load_p = static_cast<const float*>(load_p_device_ptr);
                const float* load_q = static_cast<const float*>(load_q_device_ptr);
                const float* v0_va = static_cast<const float*>(v0_va_device_ptr);
                const float* v0_vm = static_cast<const float*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<float, float>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            } else {
                const float* base_re = static_cast<const float*>(sbus_base_re_device_ptr);
                const float* base_im = static_cast<const float*>(sbus_base_im_device_ptr);
                const float* load_p = static_cast<const float*>(load_p_device_ptr);
                const float* load_q = static_cast<const float*>(load_q_device_ptr);
                const float* v0_va = static_cast<const float*>(v0_va_device_ptr);
                const float* v0_vm = static_cast<const float*>(v0_vm_device_ptr);
                launch_set_pf_inputs_from_load<float, double>(
                    base_re, base_im, load_p, load_q, v0_va, v0_vm,
                    p.buf.d_Sbus_re.data(), p.buf.d_Sbus_im.data(),
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    p.buf.d_V_re.data(), p.buf.d_V_im.data(),
                    p.buf.n_bus, batch_size);
            }

            IterationContext iter_ctx{
                .config = config,
                .pv = nullptr, .n_pv = p.buf.n_pvpq - p.buf.n_pq,
                .pq = nullptr, .n_pq = p.buf.n_pq,
            };
            const int32_t iterations = run_iteration_stages(iter_ctx);
            (void)iterations;

            if (solve_options.prepare_adjoint_cache) {
                prepare_adjoint_cache(iter_ctx, solve_options, iter_ctx.normF);
            }

            const int32_t total_bus = batch_size * p.buf.n_bus;
            if constexpr (std::is_same_v<PipelineT, CudaFp64Pipeline>) {
                launch_copy_voltage_outputs<double, double>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<double*>(va_out_device_ptr),
                    static_cast<double*>(vm_out_device_ptr),
                    total_bus);
            } else if constexpr (std::is_same_v<PipelineT, CudaFp32Pipeline>) {
                launch_copy_voltage_outputs<float, float>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<float*>(va_out_device_ptr),
                    static_cast<float*>(vm_out_device_ptr),
                    total_bus);
            } else {
                launch_copy_voltage_outputs<double, float>(
                    p.buf.d_Va.data(), p.buf.d_Vm.data(),
                    static_cast<float*>(va_out_device_ptr),
                    static_cast<float*>(vm_out_device_ptr),
                    total_bus);
            }

            fill_common_cuda_adjoint_metadata(p, result, false);
            result.success = true;
            result.torch_extension_zero_copy = true;
            result.raw_pointer_api_used = false;
            result.current_stream_integrated = true;
            result.n_bus = p.buf.n_bus;
            result.batch_size = batch_size;
            result.dimF = p.buf.dimF;
            result.factorization_time_ms = p.adjoint_cache.factorization_time_ms;
            result.total_time_ms = elapsed_ms(total_start, Clock::now());
        }
    }, pipeline_->v);
#endif
}


void NewtonSolver::prepare_adjoint_cache(IterationContext& ctx,
                                         const SolveOptions& solve_options,
                                         double final_mismatch_norm)
{
    if (solve_options.adjoint_cache_mode == AdjointCacheMode::None) {
        return;
    }
    if (solve_options.adjoint_cache_mode == AdjointCacheMode::ReuseLastNewtonFactorizationIfExact &&
        !ctx.jacobian_updated_this_iter) {
        throw std::runtime_error(
            "NewtonSolver::prepare_adjoint_cache(): last Newton factorization is not known to be exact at final state");
    }

    std::visit([&](auto& p) {
        const auto start = Clock::now();
        p.ibus(ctx);
        p.jacobian(ctx);

        p.adjoint_cache = AdjointCache{};
        p.adjoint_cache.has_adjoint_cache = true;
        p.adjoint_cache.adjoint_cache_matches_final_state = true;
        p.adjoint_cache.final_mismatch_norm = final_mismatch_norm;
        p.adjoint_cache.batch_size = [&]() -> int32_t {
            if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CpuFp64Pipeline>) {
                return 1;
            } else {
#ifdef CUPF_WITH_CUDA
                return cuda_batch_size(p.buf);
#else
                return 1;
#endif
            }
        }();
        p.adjoint_cache.dimF = p.buf.dimF;
        p.adjoint_cache.reused_forward_factorization = false;
        p.adjoint_cache.refactorized_for_adjoint_cache = true;

        if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CpuFp64Pipeline>) {
            p.linear_solve.factorize(p.buf, ctx);
            p.adjoint_cache.factorization_supports_transpose_solve = true;
            p.adjoint_cache.used_explicit_transpose = false;
            p.adjoint_cache.includes_host_device_transfer = false;
            p.adjoint_cache.jt_symbolic_analyzed_at_initialize = true;
            p.adjoint_cache.jt_values_transposed_on_device = false;
            p.adjoint_cache.jt_factorized_during_forward_cache = true;
            p.adjoint_cache.host_roundtrip_for_jt_transpose = false;
            p.adjoint_cache.backend_name = "cpu_klu";
            p.adjoint_cache.transpose_solve_backend_name =
                "cpu_klu_tsolve_cached_factorization";
        }
#ifdef CUPF_WITH_CUDA
        else {
            if (!solve_options.allow_explicit_transpose_fallback) {
                throw std::runtime_error(
                    "NewtonSolver::prepare_adjoint_cache(): cuDSS transpose solve is unsupported; "
                    "enable allow_explicit_transpose_fallback to cache an explicit J^T factorization");
            }
            double factor_ms = 0.0;
            p.linear_solve.prepare_adjoint_explicit_transpose_cache(p.buf, ctx, factor_ms);
            p.adjoint_cache.factorization_time_ms = factor_ms;
            p.adjoint_cache.factorization_supports_transpose_solve = false;
            p.adjoint_cache.used_explicit_transpose = true;
            p.adjoint_cache.includes_host_device_transfer = false;
            p.adjoint_cache.jt_symbolic_analyzed_at_initialize =
                p.linear_solve.has_adjoint_symbolic_analysis();
            p.adjoint_cache.jt_values_transposed_on_device = true;
            p.adjoint_cache.jt_factorized_during_forward_cache = true;
            p.adjoint_cache.host_roundtrip_for_jt_transpose = false;
            if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CudaFp64Pipeline>) {
                p.adjoint_cache.backend_name = "cuda_cudss_fp64";
            } else if constexpr (std::is_same_v<std::decay_t<decltype(p)>, CudaFp32Pipeline>) {
                p.adjoint_cache.backend_name = "cuda_cudss_fp32";
            } else {
                p.adjoint_cache.backend_name = "cuda_cudss_mixed";
            }
            p.adjoint_cache.transpose_solve_backend_name =
                "cuda_cudss_cached_explicit_transpose_factorization";
        }
#endif

        if (p.adjoint_cache.factorization_time_ms == 0.0) {
            p.adjoint_cache.factorization_time_ms = elapsed_ms(start, Clock::now());
        }
    }, pipeline_->v);
}


// ---------------------------------------------------------------------------
// run_iteration_stages: NR loop — ibus → mismatch → norm → jac → solve → update
// ---------------------------------------------------------------------------
int32_t NewtonSolver::run_iteration_stages(IterationContext& ctx)
{
    int32_t completed = 0;

    for (int32_t iter = 0; iter < ctx.config.max_iter; ++iter) {
        StageScope total("NR.iteration.total");
        ctx.iter = iter;
        ctx.jacobian_updated_this_iter = false;
        completed = iter + 1;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.ibus");          p.ibus(ctx); }
            { StageScope s("NR.iteration.mismatch");      p.mismatch(ctx); }
            { StageScope s("NR.iteration.mismatch_norm"); p.mismatch_norm(ctx); }
        }, pipeline_->v);

        if (ctx.converged) break;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.jacobian");       p.jacobian(ctx); }
        }, pipeline_->v);
        ctx.jacobian_updated_this_iter = true;
        ctx.jacobian_age = 0;

        std::visit([&](auto& p) {
            { StageScope s("NR.iteration.prepare_rhs");    p.prepare_rhs(ctx); }
            { StageScope s("NR.iteration.factorize");      p.factorize(ctx); }
            { StageScope s("NR.iteration.solve");          p.solve(ctx); }
            { StageScope s("NR.iteration.voltage_update"); p.voltage_update(ctx); }
        }, pipeline_->v);
    }

    return completed;
}
