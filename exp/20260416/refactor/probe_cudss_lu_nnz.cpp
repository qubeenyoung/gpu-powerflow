#include "cupf_dataset_loader.hpp"
#include "powerflow_linear_system.hpp"

#include "newton_solver/core/jacobian_builder.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using exp_20260409::CupfDatasetCase;
using exp_20260409::PowerFlowLinearSystem;

struct CliOptions {
    std::filesystem::path case_dir;
    std::string reordering_alg = "DEFAULT";
    bool use_matching = false;
    std::string matching_alg = "DEFAULT";
};

struct DeviceBuffers {
    int32_t* d_row_ptr = nullptr;
    int32_t* d_col_idx = nullptr;
    float* d_values = nullptr;
    float* d_rhs = nullptr;
    float* d_x = nullptr;

    ~DeviceBuffers()
    {
        if (d_row_ptr) cudaFree(d_row_ptr);
        if (d_col_idx) cudaFree(d_col_idx);
        if (d_values) cudaFree(d_values);
        if (d_rhs) cudaFree(d_rhs);
        if (d_x) cudaFree(d_x);
    }
};

struct CudssObjects {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs = nullptr;
    cudssMatrix_t x = nullptr;

    ~CudssObjects()
    {
        if (matrix) cudssMatrixDestroy(matrix);
        if (rhs) cudssMatrixDestroy(rhs);
        if (x) cudssMatrixDestroy(x);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }
};

void usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " --case-dir PATH --reordering-alg DEFAULT|ALG_1|ALG_2|ALG_3"
        << " [--use-matching 0|1] [--matching-alg DEFAULT|ALG_1|ALG_2|ALG_3|ALG_4|ALG_5]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--reordering-alg" && i + 1 < argc) {
            options.reordering_alg = argv[++i];
        } else if (arg == "--use-matching" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "1" || value == "true" || value == "TRUE" || value == "ON") {
                options.use_matching = true;
            } else if (value == "0" || value == "false" || value == "FALSE" || value == "OFF") {
                options.use_matching = false;
            } else {
                throw std::runtime_error("--use-matching must be 0 or 1");
            }
        } else if (arg == "--matching-alg" && i + 1 < argc) {
            options.matching_alg = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (options.case_dir.empty()) {
        throw std::runtime_error("--case-dir is required");
    }
    return options;
}

cudssAlgType_t parse_reordering_alg(const std::string& value)
{
    if (value == "DEFAULT") return CUDSS_ALG_DEFAULT;
    if (value == "ALG_1") return CUDSS_ALG_1;
    if (value == "ALG_2") return CUDSS_ALG_2;
    if (value == "ALG_3") return CUDSS_ALG_3;
    if (value == "ALG_4") return CUDSS_ALG_4;
    if (value == "ALG_5") return CUDSS_ALG_5;
    throw std::runtime_error("Unsupported cuDSS algorithm: " + value);
}

std::vector<int64_t> get_int_values(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param)
{
    size_t needed = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, nullptr, 0, &needed));
    if (needed == 0) {
        return {};
    }

    std::vector<unsigned char> raw(needed);
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, raw.data(), raw.size(), &written));
    if (written != needed) {
        throw std::runtime_error("cuDSS wrote unexpected byte count");
    }

    if (needed % sizeof(int64_t) == 0) {
        std::vector<int64_t> values(needed / sizeof(int64_t));
        std::memcpy(values.data(), raw.data(), needed);
        return values;
    }
    if (needed % sizeof(int32_t) == 0) {
        const std::size_t count = needed / sizeof(int32_t);
        std::vector<int64_t> values(count);
        const auto* src = reinterpret_cast<const int32_t*>(raw.data());
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = src[i];
        }
        return values;
    }

    throw std::runtime_error("Unexpected byte count: " + std::to_string(needed));
}

std::string join_values(const std::vector<int64_t>& values)
{
    std::string text;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) text += ':';
        text += std::to_string(values[i]);
    }
    return text;
}

void probe_one(const CliOptions& options)
{
    const CupfDatasetCase case_data = exp_20260409::load_cupf_dataset_case(options.case_dir);
    const PowerFlowLinearSystem system =
        exp_20260409::build_linear_system(case_data, JacobianBuilderType::EdgeBased);

    std::vector<float> h_values(system.values.size());
    std::vector<float> h_rhs(system.rhs.size());
    for (std::size_t i = 0; i < system.values.size(); ++i) {
        h_values[i] = static_cast<float>(system.values[i]);
    }
    for (std::size_t i = 0; i < system.rhs.size(); ++i) {
        h_rhs[i] = static_cast<float>(system.rhs[i]);
    }

    DeviceBuffers buffers;
    const auto dim = static_cast<std::size_t>(system.structure.dim);
    const auto nnz = static_cast<std::size_t>(system.structure.nnz);
    CUDA_CHECK(cudaMalloc(&buffers.d_row_ptr, (dim + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.d_col_idx, nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_rhs, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_x, dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(buffers.d_row_ptr, system.structure.row_ptr.data(),
                          (dim + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_col_idx, system.structure.col_idx.data(),
                          nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_values, h_values.data(),
                          nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_rhs, h_rhs.data(),
                          dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(buffers.d_x, 0, dim * sizeof(float)));

    CudssObjects cudss;
    CUDSS_CHECK(cudssCreate(&cudss.handle));
    CUDSS_CHECK(cudssConfigCreate(&cudss.config));
    CUDSS_CHECK(cudssDataCreate(cudss.handle, &cudss.data));

    cudssAlgType_t alg = parse_reordering_alg(options.reordering_alg);
    CUDSS_CHECK(cudssConfigSet(
        cudss.config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));

    int use_matching = options.use_matching ? 1 : 0;
    CUDSS_CHECK(cudssConfigSet(
        cudss.config, CUDSS_CONFIG_USE_MATCHING,
        &use_matching, sizeof(use_matching)));
    if (options.use_matching) {
        cudssAlgType_t matching_alg = parse_reordering_alg(options.matching_alg);
        CUDSS_CHECK(cudssConfigSet(
            cudss.config, CUDSS_CONFIG_MATCHING_ALG,
            &matching_alg, sizeof(matching_alg)));
    }

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &cudss.matrix,
        system.structure.dim, system.structure.dim, system.structure.nnz,
        buffers.d_row_ptr, nullptr, buffers.d_col_idx, buffers.d_values,
        CUDA_R_32I, CUDA_R_32F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &cudss.rhs, system.structure.dim, 1, system.structure.dim,
        buffers.d_rhs, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &cudss.x, system.structure.dim, 1, system.structure.dim,
        buffers.d_x, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

    CUDSS_CHECK(cudssExecute(
        cudss.handle, CUDSS_PHASE_ANALYSIS,
        cudss.config, cudss.data,
        cudss.matrix, cudss.x, cudss.rhs));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDSS_CHECK(cudssExecute(
        cudss.handle, CUDSS_PHASE_FACTORIZATION,
        cudss.config, cudss.data,
        cudss.matrix, cudss.x, cudss.rhs));
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<int64_t> lu_nnz = get_int_values(cudss.handle, cudss.data, CUDSS_DATA_LU_NNZ);
    const std::vector<int64_t> npivots = get_int_values(cudss.handle, cudss.data, CUDSS_DATA_NPIVOTS);
    int64_t lu_nnz_total = 0;
    for (const int64_t value : lu_nnz) {
        lu_nnz_total += value;
    }
    int64_t npivots_total = 0;
    for (const int64_t value : npivots) {
        npivots_total += value;
    }

    std::cout
        << "LU_NNZ"
        << " case=" << options.case_dir.filename().string()
        << " alg=" << options.reordering_alg
        << " use_matching=" << (options.use_matching ? 1 : 0)
        << " matching_alg=" << options.matching_alg
        << " n_bus=" << case_data.rows
        << " jacobian_dim=" << system.structure.dim
        << " jacobian_nnz=" << system.structure.nnz
        << " lu_nnz_total=" << lu_nnz_total
        << " lu_nnz_values=" << join_values(lu_nnz)
        << " npivots_total=" << npivots_total
        << " npivots_values=" << join_values(npivots)
        << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        probe_one(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
