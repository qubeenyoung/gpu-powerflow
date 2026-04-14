#include "jfnk_amgx_gpu.hpp"

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/contexts.hpp"

#include <Eigen/Dense>

#include <amgx_c.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp_20260414::newton_krylov {
namespace {

using Clock = std::chrono::steady_clock;

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t status__ = (call);                                                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                            \
            throw std::runtime_error(                                                       \
                std::string("cuBLAS call failed at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + " status=" +                                    \
                std::to_string(static_cast<int>(status__)));                                \
        }                                                                                   \
    } while (0)
#endif

constexpr int32_t kBlockSize = 256;

double elapsed_sec(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

void check_amgx(AMGX_RC code, const char* call)
{
    if (code != AMGX_RC_OK) {
        char message[4096] = {};
        AMGX_get_error_string(code, message, sizeof(message));
        throw std::runtime_error(std::string("AMGX call failed: ") + call + ": " + message);
    }
}

template <typename T>
void ensure_size(DeviceBuffer<T>& buffer, std::size_t count)
{
    if (buffer.size() != count) {
        buffer.resize(count);
    }
}

class CublasHandle {
public:
    CublasHandle()
    {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

class AmgxRuntime {
public:
    AmgxRuntime()
    {
        check_amgx(AMGX_initialize(), "AMGX_initialize");
        AMGX_register_print_callback([](const char*, int) {});
    }

    ~AmgxRuntime()
    {
        AMGX_finalize();
    }
};

void initialize_amgx_runtime()
{
    static AmgxRuntime runtime;
    (void)runtime;
}

const char* amgx_amg_config()
{
    return R"json(
{
  "config_version": 2,
  "solver": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": "BLOCK_JACOBI",
    "presweeps": 0,
    "postsweeps": 3,
    "cycle": "V",
    "coarse_solver": "NOSOLVER",
    "max_iters": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "relaxation_factor": 0.75,
    "tolerance": 0.0,
    "monitor_residual": 0,
    "print_grid_stats": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0
  }
}
)json";
}

class AmgxDevicePreconditioner {
public:
    AmgxDevicePreconditioner() = default;

    ~AmgxDevicePreconditioner()
    {
        destroy();
    }

    AmgxDevicePreconditioner(const AmgxDevicePreconditioner&) = delete;
    AmgxDevicePreconditioner& operator=(const AmgxDevicePreconditioner&) = delete;

    bool compute(int32_t n,
                 int32_t nnz,
                 const int32_t* row_ptr_device,
                 const int32_t* col_idx_device,
                 const double* values_device)
    {
        destroy();
        initialize_amgx_runtime();

        n_ = n;
        nnz_ = nnz;

        check_amgx(AMGX_config_create(&config_, amgx_amg_config()), "AMGX_config_create");
        check_amgx(AMGX_resources_create_simple(&resources_, config_),
                   "AMGX_resources_create_simple");
        check_amgx(AMGX_matrix_create(&matrix_, resources_, AMGX_mode_dDDI),
                   "AMGX_matrix_create");
        check_amgx(AMGX_vector_create(&rhs_, resources_, AMGX_mode_dDDI),
                   "AMGX_vector_create(rhs)");
        check_amgx(AMGX_vector_create(&x_, resources_, AMGX_mode_dDDI),
                   "AMGX_vector_create(x)");
        check_amgx(AMGX_solver_create(&solver_, resources_, AMGX_mode_dDDI, config_),
                   "AMGX_solver_create");

        check_amgx(AMGX_matrix_upload_all(matrix_,
                                          n_,
                                          nnz_,
                                          1,
                                          1,
                                          row_ptr_device,
                                          col_idx_device,
                                          values_device,
                                          nullptr),
                   "AMGX_matrix_upload_all");
        check_amgx(AMGX_vector_set_zero(rhs_, n_, 1), "AMGX_vector_set_zero(rhs)");
        check_amgx(AMGX_vector_set_zero(x_, n_, 1), "AMGX_vector_set_zero(x)");
        check_amgx(AMGX_solver_setup(solver_, matrix_), "AMGX_solver_setup");
        return true;
    }

    void solve(const double* rhs_device, double* x_device, int32_t n) const
    {
        if (solver_ == nullptr || n != n_) {
            throw std::runtime_error("AMGX preconditioner is not ready");
        }
        check_amgx(AMGX_vector_upload(rhs_, n, 1, rhs_device), "AMGX_vector_upload(rhs)");
        check_amgx(AMGX_vector_set_zero(x_, n, 1), "AMGX_vector_set_zero(x)");
        check_amgx(AMGX_solver_solve_with_0_initial_guess(solver_, rhs_, x_),
                   "AMGX_solver_solve_with_0_initial_guess");
        check_amgx(AMGX_vector_download(x_, x_device), "AMGX_vector_download(x)");
    }

private:
    void destroy()
    {
        if (solver_ != nullptr) {
            AMGX_solver_destroy(solver_);
            solver_ = nullptr;
        }
        if (x_ != nullptr) {
            AMGX_vector_destroy(x_);
            x_ = nullptr;
        }
        if (rhs_ != nullptr) {
            AMGX_vector_destroy(rhs_);
            rhs_ = nullptr;
        }
        if (matrix_ != nullptr) {
            AMGX_matrix_destroy(matrix_);
            matrix_ = nullptr;
        }
        if (resources_ != nullptr) {
            AMGX_resources_destroy(resources_);
            resources_ = nullptr;
        }
        if (config_ != nullptr) {
            AMGX_config_destroy(config_);
            config_ = nullptr;
        }
        n_ = 0;
        nnz_ = 0;
    }

    int32_t n_ = 0;
    int32_t nnz_ = 0;
    AMGX_config_handle config_ = nullptr;
    AMGX_resources_handle resources_ = nullptr;
    AMGX_matrix_handle matrix_ = nullptr;
    AMGX_vector_handle rhs_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};

__global__ void negate_kernel(int32_t n, const double* input, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = -input[i];
    }
}

__global__ void scale_copy_kernel(int32_t n, const double* input, double scale, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = scale * input[i];
    }
}

__global__ void subtract_kernel(int32_t n, const double* lhs, const double* rhs, double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = lhs[i] - rhs[i];
    }
}

__global__ void jv_finish_kernel(int32_t n,
                                 const double* perturbed_f,
                                 const double* base_f,
                                 double inv_eps,
                                 double* output)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (perturbed_f[i] - base_f[i]) * inv_eps;
    }
}

__global__ void reduce_absmax_kernel(int32_t n, const double* input, double* partial)
{
    __shared__ double values[kBlockSize];
    const int32_t tid = threadIdx.x;
    const int32_t i = blockIdx.x * blockDim.x + tid;

    values[tid] = (i < n) ? fabs(input[i]) : 0.0;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] = fmax(values[tid], values[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = values[0];
    }
}

__global__ void decompose_voltage_kernel(const double* v_re,
                                         const double* v_im,
                                         double* va,
                                         double* vm,
                                         int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    const double re = v_re[bus];
    const double im = v_im[bus];
    va[bus] = atan2(im, re);
    vm[bus] = hypot(re, im);
}

__global__ void apply_direction_to_voltage_kernel(double* va,
                                                  double* vm,
                                                  const double* direction,
                                                  double eps,
                                                  const int32_t* pv,
                                                  const int32_t* pq,
                                                  int32_t n_pv,
                                                  int32_t n_pq)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pv + 2 * n_pq;
    if (tid >= dim) {
        return;
    }

    const double delta = eps * direction[tid];
    if (tid < n_pv) {
        va[pv[tid]] += delta;
    } else if (tid < n_pv + n_pq) {
        va[pq[tid - n_pv]] += delta;
    } else {
        vm[pq[tid - n_pv - n_pq]] += delta;
    }
}

__global__ void reconstruct_voltage_kernel(const double* va,
                                           const double* vm,
                                           double* v_re,
                                           double* v_im,
                                           int32_t n_bus)
{
    const int32_t bus = blockIdx.x * blockDim.x + threadIdx.x;
    if (bus >= n_bus) {
        return;
    }
    v_re[bus] = vm[bus] * cos(va[bus]);
    v_im[bus] = vm[bus] * sin(va[bus]);
}

__global__ void mismatch_pack_kernel(const double* y_re,
                                     const double* y_im,
                                     const int32_t* y_row_ptr,
                                     const int32_t* y_col,
                                     const double* v_re,
                                     const double* v_im,
                                     const double* sbus_re,
                                     const double* sbus_im,
                                     const int32_t* pv,
                                     const int32_t* pq,
                                     int32_t n_pv,
                                     int32_t n_pq,
                                     double* f)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t dim = n_pv + 2 * n_pq;
    if (tid >= dim) {
        return;
    }

    int32_t bus = 0;
    bool take_imag = false;
    if (tid < n_pv) {
        bus = pv[tid];
    } else if (tid < n_pv + n_pq) {
        bus = pq[tid - n_pv];
    } else {
        bus = pq[tid - n_pv - n_pq];
        take_imag = true;
    }

    double i_re = 0.0;
    double i_im = 0.0;
    for (int32_t k = y_row_ptr[bus]; k < y_row_ptr[bus + 1]; ++k) {
        const int32_t col = y_col[k];
        const double yr = y_re[k];
        const double yi = y_im[k];
        const double vr = v_re[col];
        const double vi = v_im[col];
        i_re += yr * vr - yi * vi;
        i_im += yr * vi + yi * vr;
    }

    const double vr = v_re[bus];
    const double vi = v_im[bus];
    const double mis_re = vr * i_re + vi * i_im - sbus_re[bus];
    const double mis_im = vi * i_re - vr * i_im - sbus_im[bus];
    f[tid] = take_imag ? mis_im : mis_re;
}

__global__ void set_seed_columns_kernel(int32_t count,
                                        const int32_t* columns,
                                        double* seed)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        seed[columns[i]] = 1.0;
    }
}

__global__ void scatter_fd_values_kernel(int32_t count,
                                         const int32_t* rows,
                                         const int32_t* positions,
                                         const double* jv,
                                         double* values)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[positions[i]] = jv[rows[i]];
    }
}

__global__ void add_linear_combination_kernel(int32_t n,
                                              int32_t k,
                                              const double* coeffs,
                                              const double* vectors,
                                              int32_t stride,
                                              double* x)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    double sum = 0.0;
    for (int32_t j = 0; j < k; ++j) {
        sum += coeffs[j] * vectors[static_cast<std::size_t>(j) * stride + row];
    }
    x[row] += sum;
}

std::vector<std::vector<int32_t>> build_jacobian_pattern_by_column(
    int32_t n_bus,
    const std::vector<int32_t>& ybus_indptr,
    const std::vector<int32_t>& ybus_indices,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    const int32_t n_pv = static_cast<int32_t>(pv.size());
    const int32_t n_pq = static_cast<int32_t>(pq.size());
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim = n_pvpq + n_pq;

    std::vector<int32_t> pvpq_index(static_cast<std::size_t>(n_bus), -1);
    std::vector<int32_t> pq_index(static_cast<std::size_t>(n_bus), -1);
    for (int32_t i = 0; i < n_pv; ++i) {
        pvpq_index[static_cast<std::size_t>(pv[static_cast<std::size_t>(i)])] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[static_cast<std::size_t>(i)];
        pvpq_index[static_cast<std::size_t>(bus)] = n_pv + i;
        pq_index[static_cast<std::size_t>(bus)] = i;
    }

    std::vector<std::vector<int32_t>> col_rows(static_cast<std::size_t>(dim));
    auto add = [&](int32_t row, int32_t col) {
        if (row >= 0 && col >= 0) {
            col_rows[static_cast<std::size_t>(col)].push_back(row);
        }
    };

    auto add_bus_pair = [&](int32_t row_bus, int32_t col_bus) {
        const int32_t row_p = pvpq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q_local = pq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q = row_q_local >= 0 ? n_pvpq + row_q_local : -1;
        const int32_t col_theta = pvpq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm_local = pq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm = col_vm_local >= 0 ? n_pvpq + col_vm_local : -1;

        add(row_p, col_theta);
        add(row_p, col_vm);
        add(row_q, col_theta);
        add(row_q, col_vm);
    };

    for (int32_t row_bus = 0; row_bus < n_bus; ++row_bus) {
        for (int32_t k = ybus_indptr[static_cast<std::size_t>(row_bus)];
             k < ybus_indptr[static_cast<std::size_t>(row_bus + 1)];
             ++k) {
            add_bus_pair(row_bus, ybus_indices[static_cast<std::size_t>(k)]);
        }
    }
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        add_bus_pair(bus, bus);
    }

    for (auto& rows : col_rows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    return col_rows;
}

std::vector<int32_t> color_jacobian_columns(
    int32_t dim,
    const std::vector<std::vector<int32_t>>& col_rows)
{
    std::vector<std::vector<int32_t>> row_cols(static_cast<std::size_t>(dim));
    for (int32_t col = 0; col < dim; ++col) {
        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            row_cols[static_cast<std::size_t>(row)].push_back(col);
        }
    }

    std::vector<int32_t> color(static_cast<std::size_t>(dim), -1);
    std::vector<int32_t> forbidden;
    int32_t num_colors = 0;

    for (int32_t col = 0; col < dim; ++col) {
        if (static_cast<int32_t>(forbidden.size()) < num_colors) {
            forbidden.resize(static_cast<std::size_t>(num_colors), -1);
        }

        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            for (int32_t other_col : row_cols[static_cast<std::size_t>(row)]) {
                const int32_t other_color = color[static_cast<std::size_t>(other_col)];
                if (other_color >= 0) {
                    forbidden[static_cast<std::size_t>(other_color)] = col;
                }
            }
        }

        int32_t chosen = 0;
        while (chosen < num_colors &&
               forbidden[static_cast<std::size_t>(chosen)] == col) {
            ++chosen;
        }
        if (chosen == num_colors) {
            ++num_colors;
            forbidden.push_back(-1);
        }
        color[static_cast<std::size_t>(col)] = chosen;
    }

    return color;
}

struct FdPattern {
    int32_t dim = 0;
    int32_t nnz = 0;
    int32_t num_colors = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<int32_t> color_ptr;
    std::vector<int32_t> color_cols;
    std::vector<int32_t> color_nnz_ptr;
    std::vector<int32_t> scatter_rows;
    std::vector<int32_t> scatter_positions;
};

FdPattern build_fd_pattern(int32_t n_bus,
                           const std::vector<int32_t>& ybus_indptr,
                           const std::vector<int32_t>& ybus_indices,
                           const std::vector<int32_t>& pv,
                           const std::vector<int32_t>& pq)
{
    const auto col_rows =
        build_jacobian_pattern_by_column(n_bus, ybus_indptr, ybus_indices, pv, pq);
    const int32_t dim = static_cast<int32_t>(col_rows.size());
    const std::vector<int32_t> colors = color_jacobian_columns(dim, col_rows);
    const int32_t num_colors =
        colors.empty() ? 0 : (*std::max_element(colors.begin(), colors.end()) + 1);

    std::vector<std::vector<int32_t>> row_cols(static_cast<std::size_t>(dim));
    for (int32_t col = 0; col < dim; ++col) {
        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            row_cols[static_cast<std::size_t>(row)].push_back(col);
        }
    }
    for (auto& cols : row_cols) {
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }

    FdPattern pattern;
    pattern.dim = dim;
    pattern.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    for (int32_t row = 0; row < dim; ++row) {
        pattern.row_ptr[static_cast<std::size_t>(row + 1)] =
            pattern.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<int32_t>(row_cols[static_cast<std::size_t>(row)].size());
    }
    pattern.nnz = pattern.row_ptr.back();
    pattern.col_idx.reserve(static_cast<std::size_t>(pattern.nnz));
    for (const auto& cols : row_cols) {
        pattern.col_idx.insert(pattern.col_idx.end(), cols.begin(), cols.end());
    }

    std::vector<std::vector<int32_t>> color_cols(static_cast<std::size_t>(num_colors));
    for (int32_t col = 0; col < dim; ++col) {
        color_cols[static_cast<std::size_t>(colors[static_cast<std::size_t>(col)])].push_back(col);
    }

    pattern.num_colors = num_colors;
    pattern.color_ptr.assign(static_cast<std::size_t>(num_colors + 1), 0);
    for (int32_t color = 0; color < num_colors; ++color) {
        pattern.color_ptr[static_cast<std::size_t>(color + 1)] =
            pattern.color_ptr[static_cast<std::size_t>(color)] +
            static_cast<int32_t>(color_cols[static_cast<std::size_t>(color)].size());
        pattern.color_cols.insert(pattern.color_cols.end(),
                                  color_cols[static_cast<std::size_t>(color)].begin(),
                                  color_cols[static_cast<std::size_t>(color)].end());
    }

    pattern.color_nnz_ptr.assign(static_cast<std::size_t>(num_colors + 1), 0);
    for (int32_t color = 0; color < num_colors; ++color) {
        for (int32_t col : color_cols[static_cast<std::size_t>(color)]) {
            for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
                const auto& cols = row_cols[static_cast<std::size_t>(row)];
                const auto it = std::lower_bound(cols.begin(), cols.end(), col);
                if (it == cols.end() || *it != col) {
                    throw std::runtime_error("FD pattern scatter position not found");
                }
                const int32_t local = static_cast<int32_t>(it - cols.begin());
                pattern.scatter_rows.push_back(row);
                pattern.scatter_positions.push_back(
                    pattern.row_ptr[static_cast<std::size_t>(row)] + local);
            }
        }
        pattern.color_nnz_ptr[static_cast<std::size_t>(color + 1)] =
            static_cast<int32_t>(pattern.scatter_rows.size());
    }

    return pattern;
}

struct DeviceFdPattern {
    DeviceBuffer<int32_t> row_ptr;
    DeviceBuffer<int32_t> col_idx;
    DeviceBuffer<int32_t> color_ptr;
    DeviceBuffer<int32_t> color_cols;
    DeviceBuffer<int32_t> color_nnz_ptr;
    DeviceBuffer<int32_t> scatter_rows;
    DeviceBuffer<int32_t> scatter_positions;
    int32_t dim = 0;
    int32_t nnz = 0;
    int32_t num_colors = 0;
};

DeviceFdPattern upload_fd_pattern(const FdPattern& host)
{
    DeviceFdPattern device;
    device.dim = host.dim;
    device.nnz = host.nnz;
    device.num_colors = host.num_colors;
    device.row_ptr.assign(host.row_ptr.data(), host.row_ptr.size());
    device.col_idx.assign(host.col_idx.data(), host.col_idx.size());
    device.color_ptr.assign(host.color_ptr.data(), host.color_ptr.size());
    device.color_cols.assign(host.color_cols.data(), host.color_cols.size());
    device.color_nnz_ptr.assign(host.color_nnz_ptr.data(), host.color_nnz_ptr.size());
    device.scatter_rows.assign(host.scatter_rows.data(), host.scatter_rows.size());
    device.scatter_positions.assign(host.scatter_positions.data(), host.scatter_positions.size());
    return device;
}

double device_absmax(int32_t n,
                     const double* data,
                     DeviceBuffer<double>& partial,
                     DeviceBuffer<double>& scratch)
{
    if (n <= 0) {
        return 0.0;
    }

    const int32_t grid = (n + kBlockSize - 1) / kBlockSize;
    ensure_size(partial, static_cast<std::size_t>(grid));
    reduce_absmax_kernel<<<grid, kBlockSize>>>(n, data, partial.data());
    CUDA_CHECK(cudaGetLastError());

    int32_t current = grid;
    bool in_partial = true;
    while (current > 1) {
        const int32_t next = (current + kBlockSize - 1) / kBlockSize;
        if (in_partial) {
            ensure_size(scratch, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, partial.data(), scratch.data());
        } else {
            ensure_size(partial, static_cast<std::size_t>(next));
            reduce_absmax_kernel<<<next, kBlockSize>>>(current, scratch.data(), partial.data());
        }
        CUDA_CHECK(cudaGetLastError());
        current = next;
        in_partial = !in_partial;
    }

    double result = 0.0;
    if (in_partial) {
        partial.copyTo(&result, 1);
    } else {
        scratch.copyTo(&result, 1);
    }
    return result;
}

void add_linear_combination(int32_t dim,
                            const Eigen::VectorXd& coeffs,
                            const double* vectors,
                            int32_t stride,
                            double* x,
                            DeviceBuffer<double>& d_coeffs)
{
    const int32_t k = static_cast<int32_t>(coeffs.size());
    if (k <= 0) {
        return;
    }
    d_coeffs.assign(coeffs.data(), static_cast<std::size_t>(k));
    const int32_t grid = (dim + kBlockSize - 1) / kBlockSize;
    add_linear_combination_kernel<<<grid, kBlockSize>>>(
        dim, k, d_coeffs.data(), vectors, stride, x);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

JfnkLinearSolveAmgx::JfnkLinearSolveAmgx(IStorage& storage, JfnkOptions options)
    : storage_(static_cast<CudaFp64Storage&>(storage))
    , options_(std::move(options))
{}

JfnkLinearSolveAmgx::~JfnkLinearSolveAmgx() = default;

void JfnkLinearSolveAmgx::analyze(const AnalyzeContext& ctx)
{
    n_bus_ = ctx.n_bus;
    ybus_indptr_.assign(ctx.ybus.indptr, ctx.ybus.indptr + ctx.ybus.rows + 1);
    ybus_indices_.assign(ctx.ybus.indices, ctx.ybus.indices + ctx.ybus.nnz);
    pv_.assign(ctx.pv, ctx.pv + ctx.n_pv);
    pq_.assign(ctx.pq, ctx.pq + ctx.n_pq);
    analyzed_ = true;
}

void JfnkLinearSolveAmgx::run(IterationContext& ctx)
{
    if (!analyzed_) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: analyze() must be called first");
    }
    if (storage_.dimF <= 0 || storage_.n_bus <= 0) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: storage is not prepared");
    }
    if (options_.solver != "fgmres" && options_.solver != "gmres_none") {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: GPU path supports fgmres only");
    }
    if (options_.preconditioner != "amg_fd" && options_.preconditioner != "none") {
        throw std::runtime_error(
            "JfnkLinearSolveAmgx::run: GPU path supports amg_fd or none preconditioner only");
    }
    if (options_.linear_tolerance <= 0.0 ||
        options_.max_inner_iterations <= 0 ||
        options_.gmres_restart <= 0) {
        throw std::runtime_error("JfnkLinearSolveAmgx::run: invalid linear solver options");
    }

    const auto solve_start = Clock::now();
    const int32_t dim = storage_.dimF;
    const int32_t n_bus = storage_.n_bus;
    const int32_t n_pv = storage_.n_pvpq - storage_.n_pq;
    const int32_t n_pq = storage_.n_pq;
    const int32_t vector_grid = (dim + kBlockSize - 1) / kBlockSize;
    const int32_t bus_grid = (n_bus + kBlockSize - 1) / kBlockSize;

    stats_.last_success = false;
    stats_.last_iterations = 0;
    stats_.last_jv_calls = 0;
    stats_.last_estimated_error = std::numeric_limits<double>::quiet_NaN();
    stats_.last_epsilon = 0.0;
    stats_.last_failure_reason.clear();

    CublasHandle blas;

    DeviceBuffer<double> d_base_f(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_rhs(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_x(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_r(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_w(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_ax(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_seed(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_fd_jv(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_scratch_f(static_cast<std::size_t>(dim));
    DeviceBuffer<double> d_scratch_v_re(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_v_im(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_va(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_scratch_vm(static_cast<std::size_t>(n_bus));
    DeviceBuffer<double> d_reduce;
    DeviceBuffer<double> d_reduce_scratch;
    DeviceBuffer<double> d_coeffs;

    const int32_t restart = options_.gmres_restart;
    DeviceBuffer<double> d_basis(static_cast<std::size_t>(dim) * (restart + 1));
    DeviceBuffer<double> d_z_basis(static_cast<std::size_t>(dim) * restart);

    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, storage_.d_F.data(), 1, d_base_f.data(), 1));
    negate_kernel<<<vector_grid, kBlockSize>>>(dim, d_base_f.data(), d_rhs.data());
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_rhs.data(), 1, d_r.data(), 1));
    d_x.memsetZero();

    auto apply_jv = [&](const double* direction, double* output) {
        const auto jv_start = Clock::now();
        const double v_norm =
            std::max(device_absmax(dim, direction, d_reduce, d_reduce_scratch),
                     std::numeric_limits<double>::min());
        double eps = options_.fixed_epsilon;
        if (options_.auto_epsilon) {
            eps = std::sqrt(std::numeric_limits<double>::epsilon()) * 2.0 / v_norm;
            eps = std::max(eps, 1e-12);
        }
        stats_.last_epsilon = eps;

        const auto update_start = Clock::now();
        decompose_voltage_kernel<<<bus_grid, kBlockSize>>>(
            storage_.d_V_re.data(),
            storage_.d_V_im.data(),
            d_scratch_va.data(),
            d_scratch_vm.data(),
            n_bus);
        CUDA_CHECK(cudaGetLastError());
        apply_direction_to_voltage_kernel<<<vector_grid, kBlockSize>>>(
            d_scratch_va.data(),
            d_scratch_vm.data(),
            direction,
            eps,
            storage_.d_pv.data(),
            storage_.d_pq.data(),
            n_pv,
            n_pq);
        CUDA_CHECK(cudaGetLastError());
        reconstruct_voltage_kernel<<<bus_grid, kBlockSize>>>(
            d_scratch_va.data(),
            d_scratch_vm.data(),
            d_scratch_v_re.data(),
            d_scratch_v_im.data(),
            n_bus);
        CUDA_CHECK(cudaGetLastError());
        const auto update_end = Clock::now();

        const auto mismatch_start = Clock::now();
        mismatch_pack_kernel<<<vector_grid, kBlockSize>>>(
            storage_.d_Ybus_re.data(),
            storage_.d_Ybus_im.data(),
            storage_.d_Ybus_indptr.data(),
            storage_.d_Ybus_indices.data(),
            d_scratch_v_re.data(),
            d_scratch_v_im.data(),
            storage_.d_Sbus_re.data(),
            storage_.d_Sbus_im.data(),
            storage_.d_pv.data(),
            storage_.d_pq.data(),
            n_pv,
            n_pq,
            d_scratch_f.data());
        CUDA_CHECK(cudaGetLastError());
        const auto mismatch_end = Clock::now();

        jv_finish_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_scratch_f.data(), d_base_f.data(), 1.0 / eps, output);
        CUDA_CHECK(cudaGetLastError());

        const auto jv_end = Clock::now();
        ++stats_.last_jv_calls;
        ++stats_.total_jv_calls;
        stats_.total_jv_update_sec += elapsed_sec(update_start, update_end);
        stats_.total_jv_mismatch_sec += elapsed_sec(mismatch_start, mismatch_end);
        stats_.total_jv_sec += elapsed_sec(jv_start, jv_end);
    };

    AmgxDevicePreconditioner amg_preconditioner;
    DeviceBuffer<double> d_fd_values;

    if (options_.preconditioner == "amg_fd") {
        const auto setup_start = Clock::now();
        bool setup_ok = false;

        try {
            const FdPattern host_pattern =
                build_fd_pattern(n_bus_, ybus_indptr_, ybus_indices_, pv_, pq_);
            DeviceFdPattern device_pattern = upload_fd_pattern(host_pattern);
            d_fd_values.resize(static_cast<std::size_t>(host_pattern.nnz));
            d_fd_values.memsetZero();

            for (int32_t color = 0; color < device_pattern.num_colors; ++color) {
                const int32_t col_begin = host_pattern.color_ptr[static_cast<std::size_t>(color)];
                const int32_t col_end = host_pattern.color_ptr[static_cast<std::size_t>(color + 1)];
                const int32_t col_count = col_end - col_begin;
                d_seed.memsetZero();
                if (col_count > 0) {
                    const int32_t seed_grid = (col_count + kBlockSize - 1) / kBlockSize;
                    set_seed_columns_kernel<<<seed_grid, kBlockSize>>>(
                        col_count,
                        device_pattern.color_cols.data() + col_begin,
                        d_seed.data());
                    CUDA_CHECK(cudaGetLastError());
                }

                apply_jv(d_seed.data(), d_fd_jv.data());

                const int32_t nnz_begin =
                    host_pattern.color_nnz_ptr[static_cast<std::size_t>(color)];
                const int32_t nnz_end =
                    host_pattern.color_nnz_ptr[static_cast<std::size_t>(color + 1)];
                const int32_t nnz_count = nnz_end - nnz_begin;
                if (nnz_count > 0) {
                    const int32_t scatter_grid = (nnz_count + kBlockSize - 1) / kBlockSize;
                    scatter_fd_values_kernel<<<scatter_grid, kBlockSize>>>(
                        nnz_count,
                        device_pattern.scatter_rows.data() + nnz_begin,
                        device_pattern.scatter_positions.data() + nnz_begin,
                        d_fd_jv.data(),
                        d_fd_values.data());
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            setup_ok = amg_preconditioner.compute(device_pattern.dim,
                                                  device_pattern.nnz,
                                                  device_pattern.row_ptr.data(),
                                                  device_pattern.col_idx.data(),
                                                  d_fd_values.data());
        } catch (const std::exception& ex) {
            stats_.last_failure_reason = std::string("preconditioner_setup_") + ex.what();
            setup_ok = false;
        }

        const auto setup_end = Clock::now();
        stats_.total_preconditioner_setup_sec += elapsed_sec(setup_start, setup_end);

        if (!setup_ok) {
            if (stats_.last_failure_reason.empty()) {
                stats_.last_failure_reason = "preconditioner_setup";
            }
            ++stats_.linear_failures;
            stats_.total_inner_iterations += stats_.last_iterations;
            stats_.max_inner_iterations =
                std::max(stats_.max_inner_iterations, stats_.last_iterations);
            storage_.d_dx.memsetZero();
            const auto solve_end = Clock::now();
            stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
            return;
        }
    }

    auto apply_preconditioner = [&](const double* input, double* output) {
        if (options_.preconditioner == "amg_fd") {
            amg_preconditioner.solve(input, output, dim);
        } else {
            CUBLAS_CHECK(cublasDcopy(blas.get(), dim, input, 1, output, 1));
        }
    };

    double rhs_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_rhs.data(), 1, &rhs_norm));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    const double atol = options_.linear_tolerance * rhs_norm;

    double residual_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
    stats_.last_estimated_error = residual_norm / rhs_norm;
    if (residual_norm <= atol) {
        stats_.last_success = true;
        CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, storage_.d_dx.data(), 1));
        const auto solve_end = Clock::now();
        stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
        return;
    }

    while (stats_.last_iterations < options_.max_inner_iterations) {
        double beta = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &beta));
        if (!std::isfinite(beta)) {
            stats_.last_failure_reason = "nonfinite_residual";
            break;
        }
        if (beta <= atol) {
            stats_.last_success = true;
            stats_.last_estimated_error = beta / rhs_norm;
            break;
        }

        const int32_t basis_dim = std::min(
            restart,
            options_.max_inner_iterations - stats_.last_iterations);

        Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(basis_dim + 1, basis_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(basis_dim + 1);
        g[0] = beta;

        scale_copy_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_r.data(), 1.0 / beta, d_basis.data());
        CUDA_CHECK(cudaGetLastError());

        Eigen::VectorXd best_y;
        int32_t best_k = 0;
        bool restart_done = true;

        for (int32_t j = 0; j < basis_dim; ++j) {
            double* v_j = d_basis.data() + static_cast<std::size_t>(j) * dim;
            double* z_j = d_z_basis.data() + static_cast<std::size_t>(j) * dim;
            apply_preconditioner(v_j, z_j);
            apply_jv(z_j, d_w.data());

            for (int32_t i = 0; i <= j; ++i) {
                double* v_i = d_basis.data() + static_cast<std::size_t>(i) * dim;
                double dot = 0.0;
                CUBLAS_CHECK(cublasDdot(blas.get(), dim, v_i, 1, d_w.data(), 1, &dot));
                hessenberg(i, j) = dot;
                const double alpha = -dot;
                CUBLAS_CHECK(cublasDaxpy(blas.get(), dim, &alpha, v_i, 1, d_w.data(), 1));
            }

            double h_next = 0.0;
            CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_w.data(), 1, &h_next));
            hessenberg(j + 1, j) = h_next;
            const bool happy_breakdown =
                h_next <= std::numeric_limits<double>::epsilon();
            if (!happy_breakdown) {
                scale_copy_kernel<<<vector_grid, kBlockSize>>>(
                    dim,
                    d_w.data(),
                    1.0 / h_next,
                    d_basis.data() + static_cast<std::size_t>(j + 1) * dim);
                CUDA_CHECK(cudaGetLastError());
            }

            ++stats_.last_iterations;

            const auto h = hessenberg.block(0, 0, j + 2, j + 1);
            const auto g_head = g.head(j + 2);
            const Eigen::VectorXd y = h.colPivHouseholderQr().solve(g_head);
            const double residual_estimate = (g_head - h * y).norm();

            best_y = y;
            best_k = j + 1;
            stats_.last_estimated_error = residual_estimate / rhs_norm;

            if (stats_.last_estimated_error <= options_.linear_tolerance ||
                happy_breakdown) {
                add_linear_combination(dim,
                                       best_y,
                                       d_z_basis.data(),
                                       dim,
                                       d_x.data(),
                                       d_coeffs);
                stats_.last_success = true;
                restart_done = false;
                break;
            }
        }

        if (!restart_done) {
            break;
        }

        if (best_k > 0) {
            add_linear_combination(dim,
                                   best_y,
                                   d_z_basis.data(),
                                   dim,
                                   d_x.data(),
                                   d_coeffs);
        }

        apply_jv(d_x.data(), d_ax.data());
        subtract_kernel<<<vector_grid, kBlockSize>>>(
            dim, d_rhs.data(), d_ax.data(), d_r.data());
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDnrm2(blas.get(), dim, d_r.data(), 1, &residual_norm));
        stats_.last_estimated_error = residual_norm / rhs_norm;
    }

    if (!stats_.last_success && stats_.last_failure_reason.empty()) {
        stats_.last_failure_reason = "max_inner_iterations";
    }
    if (!stats_.last_success) {
        ++stats_.linear_failures;
    }

    stats_.total_inner_iterations += stats_.last_iterations;
    stats_.max_inner_iterations = std::max(stats_.max_inner_iterations, stats_.last_iterations);

    CUBLAS_CHECK(cublasDcopy(blas.get(), dim, d_x.data(), 1, storage_.d_dx.data(), 1));
    const auto solve_end = Clock::now();
    stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
}

}  // namespace exp_20260414::newton_krylov

#endif  // CUPF_WITH_CUDA
