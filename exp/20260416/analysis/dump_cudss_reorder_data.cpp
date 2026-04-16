#include "cupf_dataset_loader.hpp"
#include "powerflow_linear_system.hpp"

#include "newton_solver/core/jacobian_builder.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using exp_20260409::CupfDatasetCase;
using exp_20260409::PowerFlowLinearSystem;

struct CliOptions {
    std::filesystem::path case_dir;
    std::filesystem::path output_dir;
    std::string case_label;
    std::string target_bus_label;
    std::string reordering_alg = "DEFAULT";
    int nd_nlevels = -1;
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
        << " --case-dir PATH --output-dir PATH"
        << " [--case-label NAME] [--target-bus-label INT]"
        << " [--reordering-alg DEFAULT|ALG_1|ALG_2]"
        << " [--nd-nlevels AUTO|INT]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--case-label" && i + 1 < argc) {
            options.case_label = argv[++i];
        } else if (arg == "--target-bus-label" && i + 1 < argc) {
            options.target_bus_label = argv[++i];
        } else if (arg == "--reordering-alg" && i + 1 < argc) {
            options.reordering_alg = argv[++i];
        } else if (arg == "--nd-nlevels" && i + 1 < argc) {
            const std::string value = argv[++i];
            options.nd_nlevels = (value == "AUTO" || value == "auto") ? -1 : std::stoi(value);
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
    if (options.output_dir.empty()) {
        throw std::runtime_error("--output-dir is required");
    }
    if (options.case_label.empty()) {
        options.case_label = options.case_dir.filename().string();
    }
    return options;
}

cudssAlgType_t parse_reordering_alg(const std::string& value)
{
    if (value == "DEFAULT") return CUDSS_ALG_DEFAULT;
    if (value == "ALG_1") return CUDSS_ALG_1;
    if (value == "ALG_2") return CUDSS_ALG_2;
    throw std::runtime_error("Unsupported --reordering-alg: " + value);
}

template <typename T>
void write_vector_txt(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    for (const auto& value : values) {
        out << value << '\n';
    }
}

std::vector<int32_t> get_int32_data(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param)
{
    size_t needed = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, nullptr, 0, &needed));
    if (needed == 0) {
        return {};
    }
    if (needed % sizeof(int32_t) != 0) {
        throw std::runtime_error("cuDSS returned non-int32 byte count: " + std::to_string(needed));
    }

    std::vector<int32_t> values(needed / sizeof(int32_t));
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, values.data(), needed, &written));
    if (written != needed) {
        throw std::runtime_error("cuDSS wrote unexpected byte count");
    }
    return values;
}

void write_metadata_json(const std::filesystem::path& path,
                         const CliOptions& options,
                         const CupfDatasetCase& case_data,
                         const PowerFlowLinearSystem& system,
                         const std::vector<int32_t>& perm_row,
                         const std::vector<int32_t>& perm_col,
                         const std::vector<int32_t>& elimination_tree)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open metadata output: " + path.string());
    }

    out << "{\n";
    out << "  \"case\": \"" << options.case_label << "\",\n";
    out << "  \"target_bus_label\": \"" << options.target_bus_label << "\",\n";
    out << "  \"case_dir\": \"" << options.case_dir.string() << "\",\n";
    out << "  \"matrix_kind\": \"newton_jacobian_edge_based_at_v0\",\n";
    out << "  \"cudss_phase\": \"CUDSS_PHASE_REORDERING\",\n";
    out << "  \"reordering_alg\": \"" << options.reordering_alg << "\",\n";
    out << "  \"nd_nlevels\": ";
    if (options.nd_nlevels < 0) {
        out << "\"AUTO\"";
    } else {
        out << options.nd_nlevels;
    }
    out << ",\n";
    out << "  \"n_bus\": " << case_data.rows << ",\n";
    out << "  \"n_pv\": " << case_data.pv.size() << ",\n";
    out << "  \"n_pq\": " << case_data.pq.size() << ",\n";
    out << "  \"jacobian_dim\": " << system.structure.dim << ",\n";
    out << "  \"jacobian_nnz\": " << system.structure.nnz << ",\n";
    out << "  \"perm_reorder_row_len\": " << perm_row.size() << ",\n";
    out << "  \"perm_reorder_col_len\": " << perm_col.size() << ",\n";
    out << "  \"elimination_tree_len\": " << elimination_tree.size() << "\n";
    out << "}\n";
}

void dump_one_case(const CliOptions& options)
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

    if (options.nd_nlevels >= 0) {
        int nd_nlevels = options.nd_nlevels;
        CUDSS_CHECK(cudssConfigSet(
            cudss.config, CUDSS_CONFIG_ND_NLEVELS,
            &nd_nlevels, sizeof(nd_nlevels)));
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
        cudss.handle, CUDSS_PHASE_REORDERING,
        cudss.config, cudss.data,
        cudss.matrix, cudss.x, cudss.rhs));
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<int32_t> perm_row =
        get_int32_data(cudss.handle, cudss.data, CUDSS_DATA_PERM_REORDER_ROW);
    const std::vector<int32_t> perm_col =
        get_int32_data(cudss.handle, cudss.data, CUDSS_DATA_PERM_REORDER_COL);
    const std::vector<int32_t> elimination_tree =
        get_int32_data(cudss.handle, cudss.data, CUDSS_DATA_ELIMINATION_TREE);

    std::filesystem::create_directories(options.output_dir);
    write_vector_txt(options.output_dir / "perm_reorder_row.txt", perm_row);
    write_vector_txt(options.output_dir / "perm_reorder_col.txt", perm_col);
    write_vector_txt(options.output_dir / "elimination_tree.txt", elimination_tree);
    write_metadata_json(options.output_dir / "metadata.json",
                        options, case_data, system,
                        perm_row, perm_col, elimination_tree);

    std::cout
        << "DUMP case=" << options.case_label
        << " n_bus=" << case_data.rows
        << " jacobian_dim=" << system.structure.dim
        << " jacobian_nnz=" << system.structure.nnz
        << " perm_row=" << perm_row.size()
        << " perm_col=" << perm_col.size()
        << " etree=" << elimination_tree.size()
        << " output=" << options.output_dir << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        dump_one_case(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
