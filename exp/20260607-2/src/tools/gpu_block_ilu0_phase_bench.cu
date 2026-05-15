#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/core/csr_matrix.hpp"
#include "gpu_block_ilu0_preconditioner.cuh"

// File responsibility:
//   - CLI parsing
//   - J/F dump loading
//   - benchmark execution
//   - CSV/Markdown output
// The block ILU(0) implementation itself lives in gpu_block_ilu0_preconditioner.cuh.

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

namespace {

namespace block_ilu = gpu_block_ilu0;

using BenchRow = block_ilu::BenchRow;

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

struct CliOptions {
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path output_dir = "results/gpu_block_ilu0";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> block_sizes = {16, 32};
    int32_t iteration = 1;
    double diag_shift_scale = 1.0e-8;
    bool enable_profile = false;
    bool compute_output_norm = false;
    bool run_sanity_tests = true;
    bool allow_missing = false;
};

std::vector<std::string> split_list(const std::string& text)
{
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
    std::vector<int32_t> out;
    for (const std::string& item : split_list(text)) {
        out.push_back(std::stoi(item));
    }
    return out;
}

void usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --jf-root PATH\n"
        << "  --output-dir PATH\n"
        << "  --cases a,b,c\n"
        << "  --block-sizes 16,32\n"
        << "  --iteration 1\n"
        << "  --block-ilu-diag-shift-scale FLOAT\n"
        << "  --enable-profile\n"
        << "  --compute-output-norm\n"
        << "  --skip-sanity-tests\n"
        << "  --allow-missing\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--jf-root" && i + 1 < argc) {
            options.jf_root = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--block-sizes" && i + 1 < argc) {
            options.block_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--iteration" && i + 1 < argc) {
            options.iteration = std::stoi(argv[++i]);
        } else if (arg == "--block-ilu-diag-shift-scale" && i + 1 < argc) {
            options.diag_shift_scale = std::stod(argv[++i]);
        } else if (arg == "--enable-profile") {
            options.enable_profile = true;
        } else if (arg == "--compute-output-norm") {
            options.compute_output_norm = true;
        } else if (arg == "--skip-sanity-tests") {
            options.run_sanity_tests = false;
        } else if (arg == "--allow-missing") {
            options.allow_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

// ---------------------------------------------------------------------------
// J/F dump input
// ---------------------------------------------------------------------------

void expect_token(std::istream& in, const std::string& expected, const std::filesystem::path& path)
{
    std::string token;
    if (!(in >> token) || token != expected) {
        throw std::runtime_error("expected token '" + expected + "' in " + path.string());
    }
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("not a csr_matrix dump: " + path.string());
    }

    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;

    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows + 1));
    matrix.col_idx.resize(static_cast<std::size_t>(nnz));
    matrix.values.resize(static_cast<std::size_t>(nnz));

    expect_token(in, "row_ptr", path);
    for (int32_t i = 0; i <= matrix.rows; ++i) {
        in >> matrix.row_ptr[static_cast<std::size_t>(i)];
    }

    expect_token(in, "col_idx", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.col_idx[static_cast<std::size_t>(i)];
    }

    expect_token(in, "values", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.values[static_cast<std::size_t>(i)];
    }

    if (!in || matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != nnz) {
        throw std::runtime_error("malformed matrix dump: " + path.string());
    }
    return matrix;
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector file: " + path.string());
    }

    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("not a vector dump: " + path.string());
    }

    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;

    expect_token(in, "values", path);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        if (!in || index < 0 || index >= n) {
            throw std::runtime_error("malformed vector dump: " + path.string());
        }
        values[static_cast<std::size_t>(index)] = value;
    }
    return values;
}

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const auto direct = case_dir / ("J" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(direct)) {
        return direct;
    }
    return case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt");
}

std::filesystem::path rhs_path(const std::filesystem::path& jf_root,
                               const std::string& case_name,
                               int32_t iteration)
{
    const auto case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("F" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_iter" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_before_update_iter" + std::to_string(iteration) + ".txt"),
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

std::string format_double(double value)
{
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }

    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

void write_csv(const std::filesystem::path& path, const std::vector<BenchRow>& rows)
{
    std::ofstream out(path);
    out << "case,block_size,n,nnz,num_blocks,block_nnz,max_block_dim,lower_edges,upper_edges,"
           "update_ops,factor_failed,failed_block,setup_ms,dense_scatter_ms,factor_total_ms,"
           "factor_right_ms,factor_update_ms,factor_diag_inv_ms,apply_total_ms,forward_ms,"
           "backward_ms,apply_offdiag_ms,apply_diag_ms,diag_factor_work,offdiag_update_work,"
           "total_factor_work,factor_offdiag_work_share,diag_apply_work,offdiag_apply_work,"
           "total_apply_work,apply_offdiag_work_share,factor_work_relative_to_bj,"
           "apply_work_relative_to_bj,output_norm2\n";

    for (const BenchRow& row : rows) {
        out << row.case_name << ',' << row.block_size << ',' << row.n << ',' << row.nnz << ','
            << row.num_blocks << ',' << row.block_nnz << ',' << row.max_block_dim << ','
            << row.lower_edges << ',' << row.upper_edges << ',' << row.update_ops << ','
            << row.factor_failed << ',' << row.failed_block << ','
            << format_double(row.setup_ms) << ',' << format_double(row.dense_scatter_ms) << ','
            << format_double(row.factor_total_ms) << ',' << format_double(row.factor_right_ms)
            << ',' << format_double(row.factor_update_ms) << ','
            << format_double(row.factor_diag_inv_ms) << ',' << format_double(row.apply_total_ms)
            << ',' << format_double(row.forward_ms) << ',' << format_double(row.backward_ms)
            << ',' << format_double(row.apply_offdiag_ms) << ','
            << format_double(row.apply_diag_ms) << ',' << format_double(row.diag_factor_work)
            << ',' << format_double(row.offdiag_update_work) << ','
            << format_double(row.total_factor_work) << ','
            << format_double(row.factor_offdiag_work_share) << ','
            << format_double(row.diag_apply_work) << ','
            << format_double(row.offdiag_apply_work) << ','
            << format_double(row.total_apply_work) << ','
            << format_double(row.apply_offdiag_work_share) << ','
            << format_double(row.factor_work_relative_to_bj) << ','
            << format_double(row.apply_work_relative_to_bj) << ','
            << format_double(row.output_norm2) << '\n';
    }
}

bool has_profile_columns(const std::vector<BenchRow>& rows)
{
    for (const BenchRow& row : rows) {
        if (row.factor_right_ms > 0.0 || row.factor_update_ms > 0.0 ||
            row.factor_diag_inv_ms > 0.0 || row.apply_offdiag_ms > 0.0 ||
            row.apply_diag_ms > 0.0) {
            return true;
        }
    }
    return false;
}

void write_report(const std::filesystem::path& path, const std::vector<BenchRow>& rows)
{
    std::ofstream out(path);
    const bool profiled = has_profile_columns(rows);

    out << "# GPU Block ILU(0) Phase Pilot\n\n";
    out << "This is a GPU numeric phase pilot, not a production block ILU solver. It runs "
           "block ILU(0) factor/apply on GPU for the block-coloring order and records phase "
           "timings. The kernels are intentionally simple scalar CUDA kernels; no Tensor Core "
           "kernel is used here.\n\n";
    if (!profiled) {
        out << "This run used the fast path, so inner-loop phase timing is disabled. The "
               "`factor ms` and `apply ms` columns are the useful runtime numbers; subphase "
               "columns are zero by design. Re-run with `--enable-profile` for detailed "
               "phase attribution.\n\n";
    }

    out << "## Timing\n\n";
    out << "| case | bs | blocks | block nnz | setup ms | factor ms | right+update ms | "
           "diag inv ms | factor remainder ms | apply ms | offdiag apply ms | diag apply ms | "
           "apply remainder ms | failed |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const BenchRow& row : rows) {
        const double factor_accounted =
            row.factor_right_ms + row.factor_update_ms + row.factor_diag_inv_ms;
        const double apply_accounted = row.apply_offdiag_ms + row.apply_diag_ms;
        out << '|' << row.case_name << '|' << row.block_size << '|' << row.num_blocks << '|'
            << row.block_nnz << '|' << format_double(row.setup_ms) << '|'
            << format_double(row.factor_total_ms) << '|'
            << format_double(row.factor_update_ms + row.factor_right_ms) << '|'
            << format_double(row.factor_diag_inv_ms) << '|'
            << format_double(row.factor_total_ms - factor_accounted) << '|'
            << format_double(row.apply_total_ms) << '|'
            << format_double(row.apply_offdiag_ms) << '|'
            << format_double(row.apply_diag_ms) << '|'
            << format_double(row.apply_total_ms - apply_accounted) << '|'
            << row.factor_failed << "|\n";
    }

    out << "\n## Work shares\n\n";
    out << "| block size | mean factor offdiag work share | mean apply offdiag work share | "
           "mean factor/BJ work | mean apply/BJ work |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    for (int32_t bs : {16, 32}) {
        double factor_share = 0.0;
        double apply_share = 0.0;
        double factor_relative = 0.0;
        double apply_relative = 0.0;
        int32_t count = 0;

        for (const BenchRow& row : rows) {
            if (row.block_size != bs) {
                continue;
            }
            factor_share += row.factor_offdiag_work_share;
            apply_share += row.apply_offdiag_work_share;
            factor_relative += row.factor_work_relative_to_bj;
            apply_relative += row.apply_work_relative_to_bj;
            ++count;
        }

        if (count > 0) {
            out << '|' << bs << '|' << format_double(factor_share / count) << '|'
                << format_double(apply_share / count) << '|'
                << format_double(factor_relative / count) << '|'
                << format_double(apply_relative / count) << "|\n";
        }
    }

    out << "\n## Interpretation\n\n";
    out << "- GPU block ILU(0) exists here only as a phase-timing pilot.\n";
    out << "- The implementation verifies that factor/apply can be moved to GPU, but it is not optimized.\n";
    if (profiled) {
        out << "- The measured pilot is still slow because it launches many small kernels/cuBLAS calls. The remainder columns mostly represent launch/scheduling gaps between the individually timed subphases.\n";
    } else {
        out << "- Fast-path runs intentionally avoid inner-loop event synchronization, so phase columns are not populated.\n";
    }
    out << "- The symbolic work shares show that the mathematical work is dominated by dense off-diagonal block update/apply.\n";
    out << "- Those update/apply phases are the Tensor Core target. The current kernels do not use Tensor Cores, so this report should be read as a bottleneck map and optimization target, not as the final accelerated result.\n";
}

// ---------------------------------------------------------------------------
// Self-check
// ---------------------------------------------------------------------------

void run_sanity_tests()
{
    constexpr int32_t pad = 3;
    constexpr int32_t dim = 2;

    // Nonsymmetric A = [[2, 1], [3, 4]] catches accidental transpose bugs.
    std::vector<float> blocks(static_cast<std::size_t>(2 * pad * pad), 0.0f);
    blocks[0 * pad * pad + 0 * pad + 0] = 2.0f;
    blocks[0 * pad * pad + 0 * pad + 1] = 1.0f;
    blocks[0 * pad * pad + 1 * pad + 0] = 3.0f;
    blocks[0 * pad * pad + 1 * pad + 1] = 4.0f;
    blocks[1 * pad * pad + 0 * pad + 0] = 1.0f;
    blocks[1 * pad * pad + 0 * pad + 1] = 2.0f;
    blocks[1 * pad * pad + 1 * pad + 0] = 0.0f;
    blocks[1 * pad * pad + 1 * pad + 1] = 1.0f;

    cuiter::DeviceBuffer<float> d_blocks;
    cuiter::DeviceBuffer<float> d_diag;
    cuiter::DeviceBuffer<float> d_inv;
    cuiter::DeviceBuffer<int32_t> d_pivots;
    cuiter::DeviceBuffer<int32_t> d_info;
    cuiter::DeviceBuffer<float*> d_diag_ptrs;
    cuiter::DeviceBuffer<float*> d_inv_ptrs;
    cuiter::DeviceBuffer<double> d_z;
    d_blocks.assign(blocks.data(), blocks.size());
    d_diag.resize(static_cast<std::size_t>(pad * pad));
    d_inv.resize(static_cast<std::size_t>(pad * pad));
    d_pivots.resize(static_cast<std::size_t>(pad));
    d_info.resize(1);

    std::vector<float*> diag_ptrs = {d_diag.data()};
    std::vector<float*> inv_ptrs = {d_inv.data()};
    d_diag_ptrs.assign(diag_ptrs.data(), diag_ptrs.size());
    d_inv_ptrs.assign(inv_ptrs.data(), inv_ptrs.size());

    const std::vector<double> rhs = {5.0, 11.0};
    d_z.assign(rhs.data(), rhs.size());

    cublasHandle_t handle = nullptr;
    CUITER_CUBLAS_CHECK(cublasCreate(&handle));

    block_ilu::prepare_diag_for_cublas_kernel<<<1, dim3(pad, pad)>>>(d_blocks.data(),
                                                                      d_diag.data(),
                                                                      0,
                                                                      0,
                                                                      dim,
                                                                      pad,
                                                                      0.0f);
    CUITER_CUDA_CHECK(cudaGetLastError());
    CUITER_CUBLAS_CHECK(cublasSgetrfBatched(handle,
                                            dim,
                                            d_diag_ptrs.data(),
                                            pad,
                                            d_pivots.data(),
                                            d_info.data(),
                                            1));
    CUITER_CUBLAS_CHECK(cublasSgetriBatched(handle,
                                            dim,
                                            d_diag_ptrs.data(),
                                            pad,
                                            d_pivots.data(),
                                            d_inv_ptrs.data(),
                                            pad,
                                            d_info.data(),
                                            1));

    block_ilu::diag_apply_kernel<<<1, pad>>>(d_inv.data(), 0, 0, dim, pad, d_z.data());
    CUITER_CUDA_CHECK(cudaGetLastError());

    std::vector<double> x(rhs.size(), 0.0);
    d_z.copy_to(x.data(), x.size());
    if (std::abs(x[0] - 1.8) > 1.0e-4 || std::abs(x[1] - 1.4) > 1.0e-4) {
        CUITER_CUBLAS_CHECK(cublasDestroy(handle));
        throw std::runtime_error("sanity test failed: cuBLAS inverse apply layout is transposed");
    }

    block_ilu::right_multiply_kernel<<<1, dim3(pad, pad)>>>(d_blocks.data(),
                                                            d_inv.data(),
                                                            1,
                                                            0,
                                                            dim,
                                                            dim,
                                                            pad);
    CUITER_CUDA_CHECK(cudaGetLastError());

    std::vector<float> out(blocks.size(), 0.0f);
    d_blocks.copy_to(out.data(), out.size());
    const float* l_inv = out.data() + pad * pad;
    if (std::abs(l_inv[0 * pad + 0] + 0.4f) > 1.0e-4f ||
        std::abs(l_inv[0 * pad + 1] - 0.6f) > 1.0e-4f ||
        std::abs(l_inv[1 * pad + 0] + 0.6f) > 1.0e-4f ||
        std::abs(l_inv[1 * pad + 1] - 0.4f) > 1.0e-4f) {
        CUITER_CUBLAS_CHECK(cublasDestroy(handle));
        throw std::runtime_error("sanity test failed: right multiply layout mismatch");
    }

    CUITER_CUBLAS_CHECK(cublasDestroy(handle));
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

BenchRow run_one_case_block(const std::string& case_name,
                            const cuiter::CsrMatrix& matrix,
                            const std::vector<double>& rhs,
                            int32_t block_size,
                            const CliOptions& cli_options)
{
    block_ilu::GpuBlockILU0::Options options;
    options.block_size = block_size;
    options.pad = block_size;
    options.enable_profile = cli_options.enable_profile;
    options.compute_output_norm = cli_options.compute_output_norm;
    options.default_shift_scale = cli_options.diag_shift_scale;

    block_ilu::GpuBlockILU0 ilu;
    ilu.setup(matrix, rhs, case_name, options);

    BenchRow row = ilu.pattern().work;

    const block_ilu::GpuBlockILU0::Stats factor_stats =
        ilu.factorize(cli_options.diag_shift_scale);
    row.setup_ms = factor_stats.setup_ms;
    row.dense_scatter_ms = factor_stats.dense_scatter_ms;
    row.factor_total_ms = factor_stats.factor_total_ms;
    row.factor_right_ms = factor_stats.factor_right_ms;
    row.factor_update_ms = factor_stats.factor_update_ms;
    row.factor_diag_inv_ms = factor_stats.factor_diag_inv_ms;
    row.factor_failed = factor_stats.factor_failed;
    row.failed_block = factor_stats.failed_block;

    if (!row.factor_failed) {
        const block_ilu::GpuBlockILU0::Stats apply_stats = ilu.apply();
        row.apply_total_ms = apply_stats.apply_total_ms;
        row.forward_ms = apply_stats.forward_ms;
        row.backward_ms = apply_stats.backward_ms;
        row.apply_offdiag_ms = apply_stats.apply_offdiag_ms;
        row.apply_diag_ms = apply_stats.apply_diag_ms;
        row.output_norm2 = apply_stats.output_norm2;
    }

    return row;
}

void run_benchmark(const CliOptions& options)
{
    std::filesystem::create_directories(options.output_dir);

    std::vector<BenchRow> rows;
    for (const std::string& case_name : options.cases) {
        const auto j_path = jacobian_path(options.jf_root, case_name, options.iteration);
        const auto f_path = rhs_path(options.jf_root, case_name, options.iteration);
        if (!std::filesystem::exists(j_path) || !std::filesystem::exists(f_path)) {
            if (options.allow_missing) {
                std::cerr << "[skip] missing J/F for " << case_name << '\n';
                continue;
            }
            throw std::runtime_error("missing J/F for " + case_name);
        }

        const cuiter::CsrMatrix matrix = load_cupf_csr_dump(j_path);
        const std::vector<double> rhs = load_cupf_vector_dump(f_path);
        std::cout << "[case] " << case_name << " n=" << matrix.rows
                  << " nnz=" << matrix.values.size() << '\n';

        for (int32_t block_size : options.block_sizes) {
            if (block_size > block_ilu::kMaxBlockDim) {
                throw std::runtime_error("block size > 32 is not supported by this pilot");
            }

            std::cout << "  [gpu block ILU0] bs=" << block_size << '\n';
            BenchRow row = run_one_case_block(case_name, matrix, rhs, block_size, options);
            std::cout << "    factor_ms=" << format_double(row.factor_total_ms)
                      << " apply_ms=" << format_double(row.apply_total_ms)
                      << " failed=" << row.factor_failed << '\n';
            rows.push_back(row);
        }
    }

    write_csv(options.output_dir / "gpu_block_ilu0_phase_timing.csv", rows);
    write_report(options.output_dir / "gpu_block_ilu0_phase_report.md", rows);
    std::cout << "[done] " << (options.output_dir / "gpu_block_ilu0_phase_report.md") << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        CUITER_CUDA_CHECK(cudaFree(nullptr));
        CUITER_CUDA_CHECK(cudaDeviceSynchronize());

        if (options.run_sanity_tests) {
            run_sanity_tests();
        }

        run_benchmark(options);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
