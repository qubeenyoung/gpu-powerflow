#include "dump_case_loader.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/core/jacobian_builder.hpp"
#include "newton_solver/ops/jacobian/cuda_edge_fp32.hpp"
#include "newton_solver/ops/jacobian/cuda_vertex_fp32.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int32_t kBlockSize = 256;

enum class ProbeMode {
    EdgeAtomic,
    EdgeNoAtomic,
    Vertex,
};

struct CliOptions {
    std::filesystem::path case_dir;
    ProbeMode mode = ProbeMode::Vertex;
    int32_t warmup = 5;
    int32_t repeats = 20;
};

const char* mode_name(ProbeMode mode)
{
    switch (mode) {
        case ProbeMode::EdgeAtomic:
            return "edge_atomic";
        case ProbeMode::EdgeNoAtomic:
            return "edge_noatomic";
        case ProbeMode::Vertex:
            return "vertex";
    }
    return "unknown";
}

ProbeMode parse_mode(const std::string& value)
{
    if (value == "edge_atomic") {
        return ProbeMode::EdgeAtomic;
    }
    if (value == "edge_noatomic") {
        return ProbeMode::EdgeNoAtomic;
    }
    if (value == "vertex") {
        return ProbeMode::Vertex;
    }
    throw std::runtime_error("unknown --mode: " + value);
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " --case-dir PATH"
        << " [--mode edge_atomic|edge_noatomic|vertex]"
        << " [--warmup INT] [--repeats INT]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            options.mode = parse_mode(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.case_dir.empty()) {
        throw std::runtime_error("--case-dir is required");
    }
    if (options.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("--repeats must be > 0");
    }
    return options;
}

__device__ inline void add_noatomic(float* address, float value)
{
    *address += value;
}

__global__ void update_jacobian_edge_noatomic_fp32_kernel(
    int32_t n_elements,
    const double* __restrict__ y_re,
    const double* __restrict__ y_im,
    const int32_t* __restrict__ y_row,
    const int32_t* __restrict__ y_col,
    const double* __restrict__ v_re_f64,
    const double* __restrict__ v_im_f64,
    const int32_t* __restrict__ map11,
    const int32_t* __restrict__ map21,
    const int32_t* __restrict__ map12,
    const int32_t* __restrict__ map22,
    const int32_t* __restrict__ diag11,
    const int32_t* __restrict__ diag21,
    const int32_t* __restrict__ diag12,
    const int32_t* __restrict__ diag22,
    float* __restrict__ j_values)
{
    const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) {
        return;
    }

    const int32_t i = y_row[k];
    const int32_t j = y_col[k];

    const float yr = static_cast<float>(y_re[k]);
    const float yi = static_cast<float>(y_im[k]);
    const float vi_re = static_cast<float>(v_re_f64[i]);
    const float vi_im = static_cast<float>(v_im_f64[i]);
    const float vj_re = static_cast<float>(v_re_f64[j]);
    const float vj_im = static_cast<float>(v_im_f64[j]);

    const float curr_re = yr * vj_re - yi * vj_im;
    const float curr_im = yr * vj_im + yi * vj_re;

    const float neg_j_vi_re = vi_im;
    const float neg_j_vi_im = -vi_re;
    const float term_va_re = neg_j_vi_re * curr_re + neg_j_vi_im * curr_im;
    const float term_va_im = neg_j_vi_im * curr_re - neg_j_vi_re * curr_im;

    const float vj_abs = hypotf(vj_re, vj_im);
    float term_vm_re = 0.0f;
    float term_vm_im = 0.0f;
    if (vj_abs > 1e-6f) {
        const float scaled_re = curr_re / vj_abs;
        const float scaled_im = curr_im / vj_abs;
        term_vm_re = vi_re * scaled_re + vi_im * scaled_im;
        term_vm_im = vi_im * scaled_re - vi_re * scaled_im;
    }

    if (map11[k] >= 0) add_noatomic(&j_values[map11[k]], term_va_re);
    if (map21[k] >= 0) add_noatomic(&j_values[map21[k]], term_va_im);
    if (map12[k] >= 0) add_noatomic(&j_values[map12[k]], term_vm_re);
    if (map22[k] >= 0) add_noatomic(&j_values[map22[k]], term_vm_im);

    if (diag11[i] >= 0) add_noatomic(&j_values[diag11[i]], -term_va_re);
    if (diag21[i] >= 0) add_noatomic(&j_values[diag21[i]], -term_va_im);

    const float vi_abs = hypotf(vi_re, vi_im);
    if (vi_abs > 1e-6f) {
        const float vi_norm_re = vi_re / vi_abs;
        const float vi_norm_im = vi_im / vi_abs;
        const float term_vm2_re = vi_norm_re * curr_re + vi_norm_im * curr_im;
        const float term_vm2_im = vi_norm_im * curr_re - vi_norm_re * curr_im;
        if (diag12[i] >= 0) add_noatomic(&j_values[diag12[i]], term_vm2_re);
        if (diag22[i] >= 0) add_noatomic(&j_values[diag22[i]], term_vm2_im);
    }
}

void run_edge_noatomic(CudaMixedStorage& storage)
{
    if (storage.d_Y_row.empty() || storage.d_J_values.empty()) {
        throw std::runtime_error("run_edge_noatomic: storage is not prepared");
    }

    storage.d_J_values.memsetZero();

    const int32_t y_nnz = static_cast<int32_t>(storage.d_Y_row.size());
    const int32_t grid = (y_nnz + kBlockSize - 1) / kBlockSize;
    update_jacobian_edge_noatomic_fp32_kernel<<<grid, kBlockSize>>>(
        y_nnz,
        storage.d_Ybus_re.data(),
        storage.d_Ybus_im.data(),
        storage.d_Y_row.data(),
        storage.d_Ybus_indices.data(),
        storage.d_V_re.data(),
        storage.d_V_im.data(),
        storage.d_mapJ11.data(),
        storage.d_mapJ21.data(),
        storage.d_mapJ12.data(),
        storage.d_mapJ22.data(),
        storage.d_diagJ11.data(),
        storage.d_diagJ21.data(),
        storage.d_diagJ12.data(),
        storage.d_diagJ22.data(),
        storage.d_J_values.data());
    CUDA_CHECK(cudaGetLastError());
}

template <typename Fn>
float time_cuda_ms(Fn&& fn)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions cli = parse_args(argc, argv);
        const cupf::tests::DumpCaseData case_data = cupf::tests::load_dump_case(cli.case_dir);
        const YbusViewF64 ybus = case_data.ybus();
        const int32_t n_pv = static_cast<int32_t>(case_data.pv.size());
        const int32_t n_pq = static_cast<int32_t>(case_data.pq.size());

        const JacobianBuilderType builder_type =
            (cli.mode == ProbeMode::Vertex) ? JacobianBuilderType::VertexBased
                                            : JacobianBuilderType::EdgeBased;
        JacobianBuilder builder(builder_type);
        JacobianBuilder::Result analysis = builder.analyze(
            ybus,
            case_data.pv.data(), n_pv,
            case_data.pq.data(), n_pq);

        CudaMixedStorage storage;
        AnalyzeContext analyze_ctx{
            ybus,
            analysis.maps,
            analysis.J,
            ybus.rows,
            case_data.pv.data(),
            n_pv,
            case_data.pq.data(),
            n_pq,
        };
        storage.prepare(analyze_ctx);

        const NRConfig config{1e-8, 1};
        SolveContext solve_ctx{
            &ybus,
            case_data.sbus.data(),
            case_data.v0.data(),
            &config,
        };
        storage.upload(solve_ctx);

        CudaJacobianOpEdgeFp32 edge_atomic(storage);
        CudaJacobianOpVertexFp32 vertex(storage);
        IterationContext iter_ctx{
            storage,
            config,
            case_data.pv.data(),
            n_pv,
            case_data.pq.data(),
            n_pq,
            0,
            0.0,
            false,
        };

        auto run_selected = [&]() {
            switch (cli.mode) {
                case ProbeMode::EdgeAtomic:
                    edge_atomic.run(iter_ctx);
                    break;
                case ProbeMode::EdgeNoAtomic:
                    run_edge_noatomic(storage);
                    break;
                case ProbeMode::Vertex:
                    vertex.run(iter_ctx);
                    break;
            }
        };

        for (int32_t i = 0; i < cli.warmup; ++i) {
            run_selected();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << std::fixed << std::setprecision(6);
        std::cout
            << "case,mode,repeat,elapsed_ms,n_bus,n_pv,n_pq,ybus_nnz,jacobian_dim,jacobian_nnz\n";
        for (int32_t repeat = 0; repeat < cli.repeats; ++repeat) {
            const float elapsed_ms = time_cuda_ms(run_selected);
            std::cout
                << case_data.case_name << ','
                << mode_name(cli.mode) << ','
                << repeat << ','
                << elapsed_ms << ','
                << case_data.rows << ','
                << case_data.pv.size() << ','
                << case_data.pq.size() << ','
                << ybus.nnz << ','
                << analysis.J.dim << ','
                << analysis.J.nnz
                << '\n';
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "jacobian_operator_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
