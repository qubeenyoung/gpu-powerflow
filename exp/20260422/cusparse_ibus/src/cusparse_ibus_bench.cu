#include "dump_case_loader.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            throw std::runtime_error(                                                       \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err__));                                        \
        }                                                                                   \
    } while (0)

#define CUSPARSE_CHECK(call)                                                                \
    do {                                                                                    \
        cusparseStatus_t status__ = (call);                                                 \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                                          \
            throw std::runtime_error(                                                       \
                std::string("cuSPARSE error at ") + __FILE__ + ":" +                       \
                std::to_string(__LINE__) + " - " + cusparse_status_text(status__));         \
        }                                                                                   \
    } while (0)

namespace {

struct Options {
    std::filesystem::path dataset_root =
        "/workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets";
    std::filesystem::path output =
        "/workspace/gpu-powerflow/exp/20260422/cusparse_ibus/results/ibus_cusparse_spmm.csv";
    std::vector<std::string> cases = {
        "case_ACTIVSg200",
        "case_ACTIVSg2000",
        "Texas7k_20220923",
        "case_ACTIVSg25k",
        "case_ACTIVSg70k",
    };
    std::vector<int32_t> batches = {1, 4, 8, 16, 64, 256};
    int32_t warmup = 3;
    int32_t repeats = 20;
};

std::string cusparse_status_text(cusparseStatus_t status)
{
    const char* name = cusparseGetErrorName(status);
    const char* detail = cusparseGetErrorString(status);
    std::ostringstream oss;
    oss << (name != nullptr ? name : "CUSPARSE_STATUS_UNKNOWN")
        << "(" << static_cast<int>(status) << ")";
    if (detail != nullptr) {
        oss << ": " << detail;
    }
    return oss.str();
}

std::vector<std::string> split_csv_strings(const std::string& text)
{
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

std::vector<int32_t> split_csv_ints(const std::string& text)
{
    std::vector<int32_t> values;
    for (const std::string& item : split_csv_strings(text)) {
        values.push_back(static_cast<int32_t>(std::stoi(item)));
    }
    return values;
}

Options parse_args(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            opts.dataset_root = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            opts.cases = split_csv_strings(argv[++i]);
        } else if (arg == "--batches" && i + 1 < argc) {
            opts.batches = split_csv_ints(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            opts.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [--dataset-root PATH] [--output PATH]\n"
                << "  [--cases caseA,caseB] [--batches 1,4,8] [--warmup N] [--repeats N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (opts.cases.empty()) {
        throw std::runtime_error("--cases must not be empty");
    }
    if (opts.batches.empty()) {
        throw std::runtime_error("--batches must not be empty");
    }
    if (opts.warmup < 0 || opts.repeats <= 0) {
        throw std::runtime_error("--warmup must be >= 0 and --repeats must be > 0");
    }
    for (int32_t batch : opts.batches) {
        if (batch <= 0) {
            throw std::runtime_error("--batches entries must be positive");
        }
    }
    return opts;
}

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t count) { resize(count); }
    ~DeviceArray() { release(); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void resize(std::size_t count)
    {
        release();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    void assign(const std::vector<T>& values)
    {
        resize(values.size());
        if (!values.empty()) {
            CUDA_CHECK(cudaMemcpy(ptr_, values.data(), values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    void copy_to(std::vector<T>& values) const
    {
        values.resize(count_);
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(values.data(), ptr_, count_ * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
    }

    void memset_zero()
    {
        if (ptr_ != nullptr && count_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

    T* data() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

struct CusparseHandle {
    CusparseHandle() { CUSPARSE_CHECK(cusparseCreate(&handle)); }
    ~CusparseHandle()
    {
        if (handle != nullptr) {
            cusparseDestroy(handle);
        }
    }
    cusparseHandle_t handle = nullptr;
};

struct SpMat {
    ~SpMat()
    {
        if (descr != nullptr) {
            cusparseDestroySpMat(descr);
        }
    }
    cusparseSpMatDescr_t descr = nullptr;
};

struct DnMat {
    ~DnMat()
    {
        if (descr != nullptr) {
            cusparseDestroyDnMat(descr);
        }
    }
    cusparseDnMatDescr_t descr = nullptr;
};

template <typename Fn>
double time_cuda_ms(int32_t warmup, int32_t repeats, Fn&& fn)
{
    for (int32_t i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int32_t i = 0; i < repeats; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / static_cast<double>(repeats);
}

__global__ void compute_ibus_batch_fp32_kernel(
    int32_t total_rows,
    int32_t n_bus,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ y_re,
    const float* __restrict__ y_im,
    const double* __restrict__ v_re,
    const double* __restrict__ v_im,
    double* __restrict__ ibus_re,
    double* __restrict__ ibus_im,
    float* __restrict__ j_ibus_re,
    float* __restrict__ j_ibus_im)
{
    constexpr int32_t warp_size = 32;
    const int32_t warp_id = threadIdx.x / warp_size;
    const int32_t lane = threadIdx.x & (warp_size - 1);
    const int32_t warps_per_block = blockDim.x / warp_size;
    const int32_t row_slot = blockIdx.x * warps_per_block + warp_id;
    if (row_slot >= total_rows) {
        return;
    }

    const int32_t batch = row_slot / n_bus;
    const int32_t bus = row_slot - batch * n_bus;
    const int32_t v_base = batch * n_bus;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int32_t k = row_ptr[bus] + lane; k < row_ptr[bus + 1]; k += warp_size) {
        const int32_t col = col_idx[k];
        const double yr = static_cast<double>(y_re[k]);
        const double yi = static_cast<double>(y_im[k]);
        const double vr = v_re[v_base + col];
        const double vi = v_im[v_base + col];
        acc_re += yr * vr - yi * vi;
        acc_im += yr * vi + yi * vr;
    }

    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        acc_re += __shfl_down_sync(0xffffffffu, acc_re, offset);
        acc_im += __shfl_down_sync(0xffffffffu, acc_im, offset);
    }

    if (lane == 0) {
        const int32_t out = v_base + bus;
        ibus_re[out] = acc_re;
        ibus_im[out] = acc_im;
        j_ibus_re[out] = static_cast<float>(acc_re);
        j_ibus_im[out] = static_cast<float>(acc_im);
    }
}

__global__ void pack_double_to_float_pair_kernel(
    int32_t count,
    const double* __restrict__ in_re,
    const double* __restrict__ in_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    out_re[tid] = static_cast<float>(in_re[tid]);
    out_im[tid] = static_cast<float>(in_im[tid]);
}

__global__ void pack_float_to_double_pair_kernel(
    int32_t count,
    const float* __restrict__ in_re,
    const float* __restrict__ in_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    float* __restrict__ j_out_re,
    float* __restrict__ j_out_im)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    const float re = in_re[tid];
    const float im = in_im[tid];
    out_re[tid] = static_cast<double>(re);
    out_im[tid] = static_cast<double>(im);
    j_out_re[tid] = re;
    j_out_im[tid] = im;
}

void launch_custom_ibus(int32_t n_bus,
                        int32_t batch_size,
                        const DeviceArray<int32_t>& row_ptr,
                        const DeviceArray<int32_t>& col_idx,
                        const DeviceArray<float>& y_re,
                        const DeviceArray<float>& y_im,
                        const DeviceArray<double>& v_re,
                        const DeviceArray<double>& v_im,
                        DeviceArray<double>& out_re,
                        DeviceArray<double>& out_im,
                        DeviceArray<float>& j_out_re,
                        DeviceArray<float>& j_out_im)
{
    constexpr int32_t block = 256;
    constexpr int32_t warps_per_block = block / 32;
    const int32_t total_rows = n_bus * batch_size;
    const int32_t grid = (total_rows + warps_per_block - 1) / warps_per_block;
    compute_ibus_batch_fp32_kernel<<<grid, block>>>(
        total_rows,
        n_bus,
        row_ptr.data(),
        col_idx.data(),
        y_re.data(),
        y_im.data(),
        v_re.data(),
        v_im.data(),
        out_re.data(),
        out_im.data(),
        j_out_re.data(),
        j_out_im.data());
    CUDA_CHECK(cudaGetLastError());
}

void launch_pack_double_to_float(int32_t count,
                                 const DeviceArray<double>& in_re,
                                 const DeviceArray<double>& in_im,
                                 DeviceArray<float>& out_re,
                                 DeviceArray<float>& out_im)
{
    constexpr int32_t block = 256;
    const int32_t grid = (count + block - 1) / block;
    pack_double_to_float_pair_kernel<<<grid, block>>>(
        count, in_re.data(), in_im.data(), out_re.data(), out_im.data());
    CUDA_CHECK(cudaGetLastError());
}

void launch_pack_float_to_double(int32_t count,
                                 const DeviceArray<float>& in_re,
                                 const DeviceArray<float>& in_im,
                                 DeviceArray<double>& out_re,
                                 DeviceArray<double>& out_im,
                                 DeviceArray<float>& j_out_re,
                                 DeviceArray<float>& j_out_im)
{
    constexpr int32_t block = 256;
    const int32_t grid = (count + block - 1) / block;
    pack_float_to_double_pair_kernel<<<grid, block>>>(
        count, in_re.data(), in_im.data(), out_re.data(), out_im.data(),
        j_out_re.data(), j_out_im.data());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
double max_complex_abs_diff(const std::vector<T>& a_re,
                            const std::vector<T>& a_im,
                            const std::vector<T>& b_re,
                            const std::vector<T>& b_im)
{
    if (a_re.size() != b_re.size() || a_im.size() != b_im.size() ||
        a_re.size() != a_im.size()) {
        throw std::runtime_error("max_complex_abs_diff: vector size mismatch");
    }
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a_re.size(); ++i) {
        const double dre = static_cast<double>(a_re[i]) - static_cast<double>(b_re[i]);
        const double dim = static_cast<double>(a_im[i]) - static_cast<double>(b_im[i]);
        max_diff = std::max(max_diff, std::hypot(dre, dim));
    }
    return max_diff;
}

struct SpmmResult {
    bool supported = true;
    std::string status = "SUCCESS";
    double spmm_ms = std::nan("");
    double with_pack_ms = std::nan("");
    double max_abs_diff = std::nan("");
    double max_abs_diff_j = std::nan("");
};

void spmm_buffer_size_or_throw(cusparseHandle_t handle,
                               cusparseSpMatDescr_t mat_a,
                               cusparseDnMatDescr_t mat_b,
                               cusparseDnMatDescr_t mat_c,
                               const void* alpha,
                               const void* beta,
                               cudaDataType compute_type,
                               size_t& max_buffer)
{
    size_t size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_c,
        compute_type,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &size));
    max_buffer = std::max(max_buffer, size);
}

template <typename Scalar>
void create_dense_matrix(DnMat& mat,
                         int32_t n_bus,
                         int32_t batch_size,
                         DeviceArray<Scalar>& values,
                         cudaDataType type)
{
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat.descr,
        n_bus,
        batch_size,
        n_bus,
        values.data(),
        type,
        CUSPARSE_ORDER_COL));
}

SpmmResult benchmark_spmm_fp64(CusparseHandle& handle,
                               int32_t n_bus,
                               int32_t nnz,
                               int32_t batch_size,
                               int32_t warmup,
                               int32_t repeats,
                               DeviceArray<int32_t>& row_ptr,
                               DeviceArray<int32_t>& col_idx,
                               DeviceArray<double>& y_re,
                               DeviceArray<double>& y_im,
                               DeviceArray<double>& v_re,
                               DeviceArray<double>& v_im,
                               DeviceArray<double>& out_re,
                               DeviceArray<double>& out_im,
                               DeviceArray<float>& j_out_re,
                               DeviceArray<float>& j_out_im)
{
    const int32_t count = n_bus * batch_size;
    const double one = 1.0;
    const double zero = 0.0;
    const double neg_one = -1.0;

    SpMat mat_y_re;
    SpMat mat_y_im;
    DnMat mat_v_re;
    DnMat mat_v_im;
    DnMat mat_out_re;
    DnMat mat_out_im;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_y_re.descr, n_bus, n_bus, nnz, row_ptr.data(), col_idx.data(), y_re.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_y_im.descr, n_bus, n_bus, nnz, row_ptr.data(), col_idx.data(), y_im.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    create_dense_matrix(mat_v_re, n_bus, batch_size, v_re, CUDA_R_64F);
    create_dense_matrix(mat_v_im, n_bus, batch_size, v_im, CUDA_R_64F);
    create_dense_matrix(mat_out_re, n_bus, batch_size, out_re, CUDA_R_64F);
    create_dense_matrix(mat_out_im, n_bus, batch_size, out_im, CUDA_R_64F);

    size_t buffer_size = 0;
    spmm_buffer_size_or_throw(handle.handle, mat_y_re.descr, mat_v_re.descr, mat_out_re.descr,
                              &one, &zero, CUDA_R_64F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_im.descr, mat_v_im.descr, mat_out_re.descr,
                              &neg_one, &one, CUDA_R_64F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_re.descr, mat_v_im.descr, mat_out_im.descr,
                              &one, &zero, CUDA_R_64F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_im.descr, mat_v_re.descr, mat_out_im.descr,
                              &one, &one, CUDA_R_64F, buffer_size);
    DeviceArray<char> buffer(buffer_size);

    const auto spmm_only = [&]() {
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_re.descr, mat_v_re.descr, &zero, mat_out_re.descr, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &neg_one, mat_y_im.descr, mat_v_im.descr, &one, mat_out_re.descr, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_re.descr, mat_v_im.descr, &zero, mat_out_im.descr, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_im.descr, mat_v_re.descr, &one, mat_out_im.descr, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
    };

    SpmmResult result;
    result.spmm_ms = time_cuda_ms(warmup, repeats, spmm_only);
    result.with_pack_ms = time_cuda_ms(warmup, repeats, [&]() {
        spmm_only();
        launch_pack_double_to_float(count, out_re, out_im, j_out_re, j_out_im);
    });
    return result;
}

SpmmResult benchmark_spmm_fp32(CusparseHandle& handle,
                               int32_t n_bus,
                               int32_t nnz,
                               int32_t batch_size,
                               int32_t warmup,
                               int32_t repeats,
                               DeviceArray<int32_t>& row_ptr,
                               DeviceArray<int32_t>& col_idx,
                               DeviceArray<float>& y_re,
                               DeviceArray<float>& y_im,
                               DeviceArray<float>& v_re,
                               DeviceArray<float>& v_im,
                               DeviceArray<float>& tmp_re,
                               DeviceArray<float>& tmp_im,
                               DeviceArray<double>& out_re,
                               DeviceArray<double>& out_im,
                               DeviceArray<float>& j_out_re,
                               DeviceArray<float>& j_out_im)
{
    const int32_t count = n_bus * batch_size;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float neg_one = -1.0f;

    SpMat mat_y_re;
    SpMat mat_y_im;
    DnMat mat_v_re;
    DnMat mat_v_im;
    DnMat mat_tmp_re;
    DnMat mat_tmp_im;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_y_re.descr, n_bus, n_bus, nnz, row_ptr.data(), col_idx.data(), y_re.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_y_im.descr, n_bus, n_bus, nnz, row_ptr.data(), col_idx.data(), y_im.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    create_dense_matrix(mat_v_re, n_bus, batch_size, v_re, CUDA_R_32F);
    create_dense_matrix(mat_v_im, n_bus, batch_size, v_im, CUDA_R_32F);
    create_dense_matrix(mat_tmp_re, n_bus, batch_size, tmp_re, CUDA_R_32F);
    create_dense_matrix(mat_tmp_im, n_bus, batch_size, tmp_im, CUDA_R_32F);

    size_t buffer_size = 0;
    spmm_buffer_size_or_throw(handle.handle, mat_y_re.descr, mat_v_re.descr, mat_tmp_re.descr,
                              &one, &zero, CUDA_R_32F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_im.descr, mat_v_im.descr, mat_tmp_re.descr,
                              &neg_one, &one, CUDA_R_32F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_re.descr, mat_v_im.descr, mat_tmp_im.descr,
                              &one, &zero, CUDA_R_32F, buffer_size);
    spmm_buffer_size_or_throw(handle.handle, mat_y_im.descr, mat_v_re.descr, mat_tmp_im.descr,
                              &one, &one, CUDA_R_32F, buffer_size);
    DeviceArray<char> buffer(buffer_size);

    const auto spmm_only = [&]() {
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_re.descr, mat_v_re.descr, &zero, mat_tmp_re.descr, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &neg_one, mat_y_im.descr, mat_v_im.descr, &one, mat_tmp_re.descr, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_re.descr, mat_v_im.descr, &zero, mat_tmp_im.descr, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
        CUSPARSE_CHECK(cusparseSpMM(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_y_im.descr, mat_v_re.descr, &one, mat_tmp_im.descr, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
    };

    SpmmResult result;
    result.spmm_ms = time_cuda_ms(warmup, repeats, spmm_only);
    result.with_pack_ms = time_cuda_ms(warmup, repeats, [&]() {
        spmm_only();
        launch_pack_float_to_double(count, tmp_re, tmp_im, out_re, out_im, j_out_re, j_out_im);
    });
    return result;
}

std::string probe_mixed_spmm(CusparseHandle& handle,
                             int32_t n_bus,
                             int32_t nnz,
                             int32_t batch_size,
                             DeviceArray<int32_t>& row_ptr,
                             DeviceArray<int32_t>& col_idx,
                             DeviceArray<float>& y_re,
                             DeviceArray<double>& v_re,
                             DeviceArray<double>& out_re)
{
    const double one = 1.0;
    const double zero = 0.0;

    SpMat mat_y;
    DnMat mat_v;
    DnMat mat_out;
    cusparseStatus_t status = cusparseCreateCsr(
        &mat_y.descr, n_bus, n_bus, nnz, row_ptr.data(), col_idx.data(), y_re.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_status_text(status);
    }
    status = cusparseCreateDnMat(
        &mat_v.descr, n_bus, batch_size, n_bus, v_re.data(), CUDA_R_64F, CUSPARSE_ORDER_COL);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_status_text(status);
    }
    status = cusparseCreateDnMat(
        &mat_out.descr, n_bus, batch_size, n_bus, out_re.data(), CUDA_R_64F, CUSPARSE_ORDER_COL);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_status_text(status);
    }

    size_t buffer_size = 0;
    status = cusparseSpMM_bufferSize(
        handle.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        mat_y.descr,
        mat_v.descr,
        &zero,
        mat_out.descr,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_status_text(status);
    }

    DeviceArray<char> buffer(buffer_size);
    status = cusparseSpMM(
        handle.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        mat_y.descr,
        mat_v.descr,
        &zero,
        mat_out.descr,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        buffer.data());
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cusparse_status_text(status);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return "SUCCESS";
}

std::string csv_value(double value)
{
    if (std::isnan(value)) {
        return "";
    }
    std::ostringstream oss;
    oss << std::setprecision(10) << value;
    return oss.str();
}

void write_header(std::ofstream& out)
{
    out
        << "case,buses,nnz,batch,warmup,repeats,"
        << "custom_ms,"
        << "cusparse_fp64_spmm_ms,cusparse_fp64_with_pack_ms,"
        << "cusparse_fp64_spmm_speedup,cusparse_fp64_with_pack_speedup,"
        << "cusparse_fp64_max_abs_diff,cusparse_fp64_j_max_abs_diff,"
        << "cusparse_fp32_spmm_ms,cusparse_fp32_with_pack_ms,"
        << "cusparse_fp32_spmm_speedup,cusparse_fp32_with_pack_speedup,"
        << "cusparse_fp32_max_abs_diff,cusparse_fp32_j_max_abs_diff,"
        << "cusparse_mixed_probe_status\n";
}

void write_row(std::ofstream& out,
               const std::string& case_name,
               int32_t buses,
               int32_t nnz,
               int32_t batch,
               int32_t warmup,
               int32_t repeats,
               double custom_ms,
               const SpmmResult& fp64,
               const SpmmResult& fp32,
               const std::string& mixed_probe)
{
    const auto speedup = [&](double variant_ms) -> double {
        if (std::isnan(variant_ms) || variant_ms <= 0.0) {
            return std::nan("");
        }
        return custom_ms / variant_ms;
    };

    out
        << case_name << ','
        << buses << ','
        << nnz << ','
        << batch << ','
        << warmup << ','
        << repeats << ','
        << csv_value(custom_ms) << ','
        << csv_value(fp64.spmm_ms) << ','
        << csv_value(fp64.with_pack_ms) << ','
        << csv_value(speedup(fp64.spmm_ms)) << ','
        << csv_value(speedup(fp64.with_pack_ms)) << ','
        << csv_value(fp64.max_abs_diff) << ','
        << csv_value(fp64.max_abs_diff_j) << ','
        << csv_value(fp32.spmm_ms) << ','
        << csv_value(fp32.with_pack_ms) << ','
        << csv_value(speedup(fp32.spmm_ms)) << ','
        << csv_value(speedup(fp32.with_pack_ms)) << ','
        << csv_value(fp32.max_abs_diff) << ','
        << csv_value(fp32.max_abs_diff_j) << ','
        << '"' << mixed_probe << '"' << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options opts = parse_args(argc, argv);
        std::filesystem::create_directories(opts.output.parent_path());

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device is available");
        }

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << "\n";
        std::cout << "Output: " << opts.output << "\n";

        CusparseHandle cusparse;
        std::ofstream csv(opts.output);
        if (!csv) {
            throw std::runtime_error("Failed to open output CSV: " + opts.output.string());
        }
        write_header(csv);

        for (const std::string& case_name : opts.cases) {
            const auto case_dir = opts.dataset_root / case_name;
            const cupf::tests::DumpCaseData data = cupf::tests::load_dump_case(case_dir);
            const int32_t n_bus = data.rows;
            const int32_t nnz = static_cast<int32_t>(data.ybus_data.size());

            std::vector<float> h_y_re_f(static_cast<std::size_t>(nnz));
            std::vector<float> h_y_im_f(static_cast<std::size_t>(nnz));
            std::vector<double> h_y_re_d(static_cast<std::size_t>(nnz));
            std::vector<double> h_y_im_d(static_cast<std::size_t>(nnz));
            for (int32_t k = 0; k < nnz; ++k) {
                const float re = static_cast<float>(data.ybus_data[static_cast<std::size_t>(k)].real());
                const float im = static_cast<float>(data.ybus_data[static_cast<std::size_t>(k)].imag());
                h_y_re_f[static_cast<std::size_t>(k)] = re;
                h_y_im_f[static_cast<std::size_t>(k)] = im;
                h_y_re_d[static_cast<std::size_t>(k)] = static_cast<double>(re);
                h_y_im_d[static_cast<std::size_t>(k)] = static_cast<double>(im);
            }

            DeviceArray<int32_t> d_row_ptr;
            DeviceArray<int32_t> d_col_idx;
            DeviceArray<float> d_y_re_f;
            DeviceArray<float> d_y_im_f;
            DeviceArray<double> d_y_re_d;
            DeviceArray<double> d_y_im_d;
            d_row_ptr.assign(data.indptr);
            d_col_idx.assign(data.indices);
            d_y_re_f.assign(h_y_re_f);
            d_y_im_f.assign(h_y_im_f);
            d_y_re_d.assign(h_y_re_d);
            d_y_im_d.assign(h_y_im_d);

            for (int32_t batch : opts.batches) {
                const std::size_t count =
                    static_cast<std::size_t>(batch) * static_cast<std::size_t>(n_bus);

                std::vector<double> h_v_re_d(count);
                std::vector<double> h_v_im_d(count);
                std::vector<float> h_v_re_f(count);
                std::vector<float> h_v_im_f(count);
                for (int32_t b = 0; b < batch; ++b) {
                    const std::size_t base =
                        static_cast<std::size_t>(b) * static_cast<std::size_t>(n_bus);
                    for (int32_t bus = 0; bus < n_bus; ++bus) {
                        const auto& v = data.v0[static_cast<std::size_t>(bus)];
                        const std::size_t idx = base + static_cast<std::size_t>(bus);
                        h_v_re_d[idx] = v.real();
                        h_v_im_d[idx] = v.imag();
                        h_v_re_f[idx] = static_cast<float>(v.real());
                        h_v_im_f[idx] = static_cast<float>(v.imag());
                    }
                }

                DeviceArray<double> d_v_re_d;
                DeviceArray<double> d_v_im_d;
                DeviceArray<float> d_v_re_f;
                DeviceArray<float> d_v_im_f;
                d_v_re_d.assign(h_v_re_d);
                d_v_im_d.assign(h_v_im_d);
                d_v_re_f.assign(h_v_re_f);
                d_v_im_f.assign(h_v_im_f);

                DeviceArray<double> d_ref_re(count);
                DeviceArray<double> d_ref_im(count);
                DeviceArray<float> d_ref_j_re(count);
                DeviceArray<float> d_ref_j_im(count);
                DeviceArray<double> d_test_re(count);
                DeviceArray<double> d_test_im(count);
                DeviceArray<float> d_test_j_re(count);
                DeviceArray<float> d_test_j_im(count);
                DeviceArray<float> d_tmp_re(count);
                DeviceArray<float> d_tmp_im(count);

                const double custom_ms = time_cuda_ms(opts.warmup, opts.repeats, [&]() {
                    launch_custom_ibus(n_bus, batch, d_row_ptr, d_col_idx, d_y_re_f, d_y_im_f,
                                       d_v_re_d, d_v_im_d, d_ref_re, d_ref_im,
                                       d_ref_j_re, d_ref_j_im);
                });

                std::vector<double> h_ref_re;
                std::vector<double> h_ref_im;
                std::vector<float> h_ref_j_re;
                std::vector<float> h_ref_j_im;
                d_ref_re.copy_to(h_ref_re);
                d_ref_im.copy_to(h_ref_im);
                d_ref_j_re.copy_to(h_ref_j_re);
                d_ref_j_im.copy_to(h_ref_j_im);

                SpmmResult fp64 = benchmark_spmm_fp64(
                    cusparse, n_bus, nnz, batch, opts.warmup, opts.repeats,
                    d_row_ptr, d_col_idx, d_y_re_d, d_y_im_d, d_v_re_d, d_v_im_d,
                    d_test_re, d_test_im, d_test_j_re, d_test_j_im);
                std::vector<double> h_test_re;
                std::vector<double> h_test_im;
                std::vector<float> h_test_j_re;
                std::vector<float> h_test_j_im;
                d_test_re.copy_to(h_test_re);
                d_test_im.copy_to(h_test_im);
                d_test_j_re.copy_to(h_test_j_re);
                d_test_j_im.copy_to(h_test_j_im);
                fp64.max_abs_diff =
                    max_complex_abs_diff(h_ref_re, h_ref_im, h_test_re, h_test_im);
                fp64.max_abs_diff_j =
                    max_complex_abs_diff(h_ref_j_re, h_ref_j_im, h_test_j_re, h_test_j_im);

                SpmmResult fp32 = benchmark_spmm_fp32(
                    cusparse, n_bus, nnz, batch, opts.warmup, opts.repeats,
                    d_row_ptr, d_col_idx, d_y_re_f, d_y_im_f, d_v_re_f, d_v_im_f,
                    d_tmp_re, d_tmp_im, d_test_re, d_test_im, d_test_j_re, d_test_j_im);
                d_test_re.copy_to(h_test_re);
                d_test_im.copy_to(h_test_im);
                d_test_j_re.copy_to(h_test_j_re);
                d_test_j_im.copy_to(h_test_j_im);
                fp32.max_abs_diff =
                    max_complex_abs_diff(h_ref_re, h_ref_im, h_test_re, h_test_im);
                fp32.max_abs_diff_j =
                    max_complex_abs_diff(h_ref_j_re, h_ref_j_im, h_test_j_re, h_test_j_im);

                const std::string mixed_probe =
                    probe_mixed_spmm(cusparse, n_bus, nnz, batch,
                                     d_row_ptr, d_col_idx, d_y_re_f, d_v_re_d, d_test_re);

                write_row(csv, case_name, n_bus, nnz, batch, opts.warmup, opts.repeats,
                          custom_ms, fp64, fp32, mixed_probe);
                csv.flush();

                std::cout
                    << case_name << " B=" << batch
                    << " custom=" << std::fixed << std::setprecision(4) << custom_ms << " ms"
                    << " fp64_pack=" << fp64.with_pack_ms << " ms"
                    << " fp32_pack=" << fp32.with_pack_ms << " ms"
                    << " mixed=" << mixed_probe << "\n";
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Done.\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
