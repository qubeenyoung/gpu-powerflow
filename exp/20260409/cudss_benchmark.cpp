#include "cudss_schur_runner.hpp"
#include "cupf_dataset_loader.hpp"
#include "powerflow_linear_system.hpp"

#include "newton_solver/core/jacobian_types.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using PowerFlowLinearSystem = exp_20260409::PowerFlowLinearSystem;

struct CliOptions {
    std::filesystem::path dataset_root = "/workspace/datasets/cuPF_datasets";
    std::filesystem::path case_dir;
    std::string case_name = "case118_ieee";
    std::string jacobian  = "edge_based";
    std::string mode      = "full";  // "full" or "schur_j22"
    int32_t warmup  = 0;
    int32_t repeats = 1;
    bool list_cases = false;
    std::filesystem::path output_json;
};

// Metrics for one measured iteration.
// analysis_sec is filled once by the caller and attached to the first run entry.
// Schur-specific fields are only valid when mode == "schur_j22".
struct RepeatMetrics {
    int32_t repeat_idx = 0;
    double analysis_sec      = 0.0;
    double factorization_sec = 0.0;
    double solve_sec         = 0.0;
    double total_sec         = 0.0;
    double no_analysis_sec   = 0.0;
    double residual_inf          = 0.0;
    double relative_residual_inf = 0.0;
    double solution_inf          = 0.0;
    // Schur-specific
    int32_t schur_dim        = 0;
    double schur_extract_sec = 0.0;
    double fwd_solve_sec     = 0.0;
    double diag_sec          = 0.0;
    double getrf_sec         = 0.0;
    double getrs_sec         = 0.0;
    double schur_solve_sec   = 0.0;
    double bwd_solve_sec     = 0.0;
};

// ---------------------------------------------------------------------------
// CuDssLinearSystemRunner — full sparse direct solve (baseline)
//
// Design:
//   • CUDA events for timing — one cudaEventSynchronize at end of factorize_and_solve()
//   • d_rhs_orig_ holds original RHS on device; reset uses D2D copy (no H2D per iter)
//   • download_solution() is separate — call once after all iterations
// ---------------------------------------------------------------------------

class CuDssLinearSystemRunner {
public:
    explicit CuDssLinearSystemRunner(const PowerFlowLinearSystem& system)
        : system_(system)
        , system_dim_(system.structure.dim)
        , matrix_nnz_(system.structure.nnz)
        , h_values_f_(system.values.size())
        , h_x_(system.structure.dim, 0.0f)
    {
        convert_host_inputs_to_fp32();
        allocate_device_buffers();
        upload_static_inputs();
        create_cudss_objects();
        for (auto& ev : ev_) CUDA_CHECK(cudaEventCreate(&ev));
    }

    ~CuDssLinearSystemRunner()
    {
        for (auto& ev : ev_) if (ev) cudaEventDestroy(ev);
        destroy_cudss_objects();
        free_device_buffers();
    }

    double analyze()
    {
        const auto t0 = Clock::now();
        CUDSS_CHECK(cudssExecute(
            handle_, CUDSS_PHASE_ANALYSIS,
            config_, data_,
            matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }

    // Returns partial RepeatMetrics (no analysis_sec, no residual fields).
    RepeatMetrics factorize_and_solve(int32_t repeat_idx)
    {
        // D2D reset — no H2D in the hot path.
        CUDA_CHECK(cudaMemcpy(d_rhs_, d_rhs_orig_,
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_x_, 0, static_cast<size_t>(system_dim_) * sizeof(float)));

        // ev_[0] = total/factor start
        // ev_[1] = factor end / solve start
        // ev_[2] = solve end / total end
        CUDA_CHECK(cudaEventRecord(ev_[0]));

        CUDSS_CHECK(cudssExecute(
            handle_, CUDSS_PHASE_FACTORIZATION,
            config_, data_,
            matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaEventRecord(ev_[1]));

        CUDSS_CHECK(cudssExecute(
            handle_, CUDSS_PHASE_SOLVE,
            config_, data_,
            matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaEventRecord(ev_[2]));

        CUDA_CHECK(cudaEventSynchronize(ev_[2]));

        RepeatMetrics m;
        m.repeat_idx        = repeat_idx;
        m.factorization_sec = elapsed_ms(ev_[0], ev_[1]) * 1e-3;
        m.solve_sec         = elapsed_ms(ev_[1], ev_[2]) * 1e-3;
        m.total_sec         = elapsed_ms(ev_[0], ev_[2]) * 1e-3;
        return m;
    }

    void download_solution()
    {
        CUDA_CHECK(cudaMemcpy(h_x_.data(), d_x_,
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    const std::vector<float>& solution() const { return h_x_; }

private:
    void convert_host_inputs_to_fp32()
    {
        for (size_t i = 0; i < system_.values.size(); ++i)
            h_values_f_[i] = static_cast<float>(system_.values[i]);
    }

    void allocate_device_buffers()
    {
        CUDA_CHECK(cudaMalloc(&d_row_ptr_,  static_cast<size_t>(system_dim_ + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_col_idx_,  static_cast<size_t>(matrix_nnz_) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_,   static_cast<size_t>(matrix_nnz_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rhs_orig_, static_cast<size_t>(system_dim_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rhs_,      static_cast<size_t>(system_dim_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_,        static_cast<size_t>(system_dim_) * sizeof(float)));
    }

    void upload_static_inputs()
    {
        CUDA_CHECK(cudaMemcpy(d_row_ptr_, system_.structure.row_ptr.data(),
                              static_cast<size_t>(system_dim_ + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx_, system_.structure.col_idx.data(),
                              static_cast<size_t>(matrix_nnz_) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values_, h_values_f_.data(),
                              static_cast<size_t>(matrix_nnz_) * sizeof(float),
                              cudaMemcpyHostToDevice));
        // Upload RHS once to the device-side original; working buffer is reset each iter via D2D.
        std::vector<float> h_rhs_f(system_.rhs.size());
        for (size_t i = 0; i < system_.rhs.size(); ++i)
            h_rhs_f[i] = static_cast<float>(system_.rhs[i]);
        CUDA_CHECK(cudaMemcpy(d_rhs_orig_, h_rhs_f.data(),
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void create_cudss_objects()
    {
        CUDSS_CHECK(cudssCreate(&handle_));
        CUDSS_CHECK(cudssConfigCreate(&config_));
        CUDSS_CHECK(cudssDataCreate(handle_, &data_));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &matrix_,
            system_dim_, system_dim_, matrix_nnz_,
            d_row_ptr_, nullptr, d_col_idx_, d_values_,
            CUDA_R_32I, CUDA_R_32F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        CUDSS_CHECK(cudssMatrixCreateDn(
            &rhs_matrix_, system_dim_, 1, system_dim_,
            d_rhs_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &solution_matrix_, system_dim_, 1, system_dim_,
            d_x_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
    }

    void destroy_cudss_objects()
    {
        if (matrix_)          cudssMatrixDestroy(matrix_);
        if (rhs_matrix_)      cudssMatrixDestroy(rhs_matrix_);
        if (solution_matrix_) cudssMatrixDestroy(solution_matrix_);
        if (data_)            cudssDataDestroy(handle_, data_);
        if (config_)          cudssConfigDestroy(config_);
        if (handle_)          cudssDestroy(handle_);
    }

    void free_device_buffers()
    {
        if (d_row_ptr_)  cudaFree(d_row_ptr_);
        if (d_col_idx_)  cudaFree(d_col_idx_);
        if (d_values_)   cudaFree(d_values_);
        if (d_rhs_orig_) cudaFree(d_rhs_orig_);
        if (d_rhs_)      cudaFree(d_rhs_);
        if (d_x_)        cudaFree(d_x_);
    }

    static float elapsed_ms(cudaEvent_t start, cudaEvent_t end)
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, end);
        return ms;
    }

    const PowerFlowLinearSystem& system_;
    int32_t system_dim_ = 0;
    int32_t matrix_nnz_ = 0;

    std::vector<float> h_values_f_;
    std::vector<float> h_x_;

    int32_t* d_row_ptr_  = nullptr;
    int32_t* d_col_idx_  = nullptr;
    float*   d_values_   = nullptr;
    float*   d_rhs_orig_ = nullptr;
    float*   d_rhs_      = nullptr;
    float*   d_x_        = nullptr;

    cudssHandle_t handle_          = nullptr;
    cudssConfig_t config_          = nullptr;
    cudssData_t   data_            = nullptr;
    cudssMatrix_t matrix_          = nullptr;
    cudssMatrix_t rhs_matrix_      = nullptr;
    cudssMatrix_t solution_matrix_ = nullptr;

    cudaEvent_t ev_[3] = {};  // factor_start, solve_start, solve_end
};

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [--dataset-root PATH] [--case NAME] [--case-dir PATH]"
        << " [--jacobian edge_based|vertex_based]"
        << " [--mode full|schur_j22]"
        << " [--warmup INT] [--repeats INT]"
        << " [--output-json PATH] [--list-cases]\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            options.dataset_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_name = argv[++i];
        } else if (arg == "--case-dir" && i + 1 < argc) {
            options.case_dir = argv[++i];
        } else if (arg == "--jacobian" && i + 1 < argc) {
            options.jacobian = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            options.mode = argv[++i];
            if (options.mode != "full" && options.mode != "schur_j22")
                throw std::runtime_error("mode must be 'full' or 'schur_j22'");
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--output-json" && i + 1 < argc) {
            options.output_json = argv[++i];
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.warmup < 0)  throw std::runtime_error("warmup must be non-negative");
    if (options.repeats <= 0) throw std::runtime_error("repeats must be positive");
    return options;
}

std::filesystem::path resolve_case_dir(const CliOptions& cli)
{
    if (!cli.case_dir.empty()) return cli.case_dir;
    return cli.dataset_root / cli.case_name;
}

JacobianBuilderType parse_jacobian(const std::string& jacobian)
{
    if (jacobian == "edge_based")   return JacobianBuilderType::EdgeBased;
    if (jacobian == "vertex_based") return JacobianBuilderType::VertexBased;
    throw std::runtime_error("jacobian must be 'edge_based' or 'vertex_based'");
}

template <typename Member>
double mean_value(const std::vector<RepeatMetrics>& runs, Member member)
{
    if (runs.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : runs) sum += static_cast<double>(r.*member);
    return sum / static_cast<double>(runs.size());
}

template <typename Member>
double min_value(const std::vector<RepeatMetrics>& runs, Member member)
{
    double best = std::numeric_limits<double>::infinity();
    for (const auto& r : runs) best = std::min(best, static_cast<double>(r.*member));
    return best;
}

template <typename Member>
double max_value(const std::vector<RepeatMetrics>& runs, Member member)
{
    double best = 0.0;
    for (const auto& r : runs) best = std::max(best, static_cast<double>(r.*member));
    return best;
}

std::string json_escape(std::string_view text)
{
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += ch;     break;
        }
    }
    return out;
}

void print_system_summary(const exp_20260409::CupfDatasetCase& case_data,
                          const PowerFlowLinearSystem& system,
                          const CliOptions& cli,
                          double build_system_sec)
{
    std::cout << "[system] case=" << case_data.case_name
              << " mode=" << cli.mode
              << " jacobian=" << cli.jacobian
              << " buses=" << case_data.rows
              << " pv=" << case_data.pv.size()
              << " pq=" << case_data.pq.size()
              << " dim=" << system.structure.dim
              << " nnz=" << system.structure.nnz
              << " mismatch_inf=" << std::scientific << std::setprecision(6) << system.mismatch_inf
              << " rhs_inf=" << exp_20260409::rhs_inf_norm(system.rhs)
              << " build_system_sec=" << std::fixed << std::setprecision(6) << build_system_sec
              << "\n";
}

void print_repeat_summary(const char* tag,
                          const std::string& case_name,
                          const std::string& mode,
                          const RepeatMetrics& m)
{
    std::cout << "[" << tag << "] case=" << case_name
              << " mode=" << mode
              << " repeat=" << m.repeat_idx
              << " analysis_sec=" << std::fixed << std::setprecision(6) << m.analysis_sec
              << " factorization_sec=" << m.factorization_sec;

    if (mode == "schur_j22") {
        std::cout << " schur_extract_sec=" << m.schur_extract_sec
                  << " fwd_solve_sec=" << m.fwd_solve_sec
                  << " diag_sec=" << m.diag_sec
                  << " getrf_sec=" << m.getrf_sec
                  << " getrs_sec=" << m.getrs_sec
                  << " bwd_solve_sec=" << m.bwd_solve_sec;
    }

    std::cout << " solve_sec=" << m.solve_sec
              << " total_sec=" << m.total_sec
              << " residual_inf=" << std::scientific << std::setprecision(6) << m.residual_inf;

    if (std::string_view(tag) == "run") {
        std::cout << " relative_residual_inf=" << m.relative_residual_inf
                  << " x_inf=" << std::fixed << std::setprecision(6) << m.solution_inf;
    }
    std::cout << "\n";
}

void print_benchmark_summary(const std::string& case_name,
                             const std::string& mode,
                             const std::vector<RepeatMetrics>& runs)
{
    std::cout << "[summary] case=" << case_name
              << " mode=" << mode
              << " repeats=" << runs.size()
              << " analysis_sec_mean=" << std::fixed << std::setprecision(6)
              << mean_value(runs, &RepeatMetrics::analysis_sec)
              << " factorization_sec_mean=" << mean_value(runs, &RepeatMetrics::factorization_sec)
              << " no_analysis_sec_mean=" << mean_value(runs, &RepeatMetrics::no_analysis_sec);

    if (mode == "schur_j22") {
        std::cout << " schur_extract_sec_mean=" << mean_value(runs, &RepeatMetrics::schur_extract_sec)
                  << " fwd_solve_sec_mean=" << mean_value(runs, &RepeatMetrics::fwd_solve_sec)
                  << " diag_sec_mean=" << mean_value(runs, &RepeatMetrics::diag_sec)
                  << " getrf_sec_mean=" << mean_value(runs, &RepeatMetrics::getrf_sec)
                  << " getrs_sec_mean=" << mean_value(runs, &RepeatMetrics::getrs_sec)
                  << " bwd_solve_sec_mean=" << mean_value(runs, &RepeatMetrics::bwd_solve_sec);
    }

    std::cout << " solve_sec_mean=" << mean_value(runs, &RepeatMetrics::solve_sec)
              << " total_sec_mean=" << mean_value(runs, &RepeatMetrics::total_sec)
              << " residual_inf_max=" << std::scientific << std::setprecision(6)
              << max_value(runs, &RepeatMetrics::residual_inf)
              << "\n";
}

void write_summary_json(const std::filesystem::path& output_path,
                        const CliOptions& cli,
                        const std::filesystem::path& case_dir,
                        const exp_20260409::CupfDatasetCase& case_data,
                        const PowerFlowLinearSystem& system,
                        double build_system_sec,
                        double analysis_sec,
                        const std::vector<RepeatMetrics>& runs)
{
    const auto parent_dir = output_path.parent_path();
    if (!parent_dir.empty()) std::filesystem::create_directories(parent_dir);

    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("Failed to open output json: " + output_path.string());

    const bool is_schur = (cli.mode == "schur_j22");

    out << "{\n";
    out << "  \"dataset_root\": \""     << json_escape(cli.dataset_root.string()) << "\",\n";
    out << "  \"case_dir\": \""         << json_escape(case_dir.string()) << "\",\n";
    out << "  \"case_name\": \""        << json_escape(case_data.case_name) << "\",\n";
    out << "  \"benchmark_kind\": \"cudss_linear_system\",\n";
    out << "  \"mode\": \""             << json_escape(cli.mode) << "\",\n";
    out << "  \"linear_system_origin\": \"V0 Newton step\",\n";
    out << "  \"rhs_definition\": \"b = -F(V0)\",\n";
    out << "  \"matrix_definition\": \"J = dF/dx at V0\",\n";
    out << "  \"matrix_precision\": \"float32\",\n";
    out << "  \"rhs_precision\": \"float32\",\n";
    out << "  \"jacobian\": \""         << json_escape(cli.jacobian) << "\",\n";
    out << "  \"warmup\": "             << cli.warmup << ",\n";
    out << "  \"repeats\": "            << cli.repeats << ",\n";
    out << "  \"buses\": "              << case_data.rows << ",\n";
    out << "  \"ybus_nnz\": "           << case_data.ybus_data.size() << ",\n";
    out << "  \"pv\": "                 << case_data.pv.size() << ",\n";
    out << "  \"pq\": "                 << case_data.pq.size() << ",\n";
    if (is_schur) out << "  \"schur_dim\": " << case_data.pq.size() << ",\n";
    out << "  \"linear_dim\": "         << system.structure.dim << ",\n";
    out << "  \"linear_nnz\": "         << system.structure.nnz << ",\n";
    out << "  \"build_system_sec\": "   << std::setprecision(17) << build_system_sec << ",\n";
    out << "  \"analysis_sec\": "       << analysis_sec << ",\n";
    out << "  \"no_analysis_sec_mean\": " << mean_value(runs, &RepeatMetrics::no_analysis_sec) << ",\n";
    out << "  \"mismatch_inf\": "       << system.mismatch_inf << ",\n";
    out << "  \"rhs_inf\": "            << exp_20260409::rhs_inf_norm(system.rhs) << ",\n";
    out << "  \"runs\": [\n";

    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        out << "    {\n";
        out << "      \"repeat_idx\": "        << r.repeat_idx << ",\n";
        out << "      \"factorization_sec\": " << r.factorization_sec << ",\n";
        if (is_schur) {
            out << "      \"schur_extract_sec\": " << r.schur_extract_sec << ",\n";
            out << "      \"fwd_solve_sec\": "     << r.fwd_solve_sec << ",\n";
            out << "      \"diag_sec\": "          << r.diag_sec << ",\n";
            out << "      \"getrf_sec\": "         << r.getrf_sec << ",\n";
            out << "      \"getrs_sec\": "         << r.getrs_sec << ",\n";
            out << "      \"schur_solve_sec\": "   << r.schur_solve_sec << ",\n";
            out << "      \"bwd_solve_sec\": "     << r.bwd_solve_sec << ",\n";
        }
        out << "      \"solve_sec\": "         << r.solve_sec << ",\n";
        out << "      \"total_sec\": "         << r.total_sec << ",\n";
        out << "      \"residual_inf\": "      << r.residual_inf << ",\n";
        out << "      \"relative_residual_inf\": " << r.relative_residual_inf << ",\n";
        out << "      \"solution_inf\": "      << r.solution_inf << "\n";
        out << "    }";
        if (i + 1 != runs.size()) out << ",";
        out << "\n";
    }

    out << "  ],\n";
    out << "  \"summary\": {\n";
    out << "    \"analysis_sec\": "             << analysis_sec << ",\n";
    out << "    \"factorization_sec_mean\": "   << mean_value(runs, &RepeatMetrics::factorization_sec) << ",\n";
    out << "    \"no_analysis_sec_mean\": "     << mean_value(runs, &RepeatMetrics::no_analysis_sec) << ",\n";
    if (is_schur) {
        out << "    \"schur_extract_sec_mean\": " << mean_value(runs, &RepeatMetrics::schur_extract_sec) << ",\n";
        out << "    \"fwd_solve_sec_mean\": "     << mean_value(runs, &RepeatMetrics::fwd_solve_sec) << ",\n";
        out << "    \"diag_sec_mean\": "          << mean_value(runs, &RepeatMetrics::diag_sec) << ",\n";
        out << "    \"getrf_sec_mean\": "         << mean_value(runs, &RepeatMetrics::getrf_sec) << ",\n";
        out << "    \"getrs_sec_mean\": "         << mean_value(runs, &RepeatMetrics::getrs_sec) << ",\n";
        out << "    \"schur_solve_sec_mean\": "   << mean_value(runs, &RepeatMetrics::schur_solve_sec) << ",\n";
        out << "    \"bwd_solve_sec_mean\": "     << mean_value(runs, &RepeatMetrics::bwd_solve_sec) << ",\n";
    }
    out << "    \"solve_sec_mean\": "            << mean_value(runs, &RepeatMetrics::solve_sec) << ",\n";
    out << "    \"total_sec_mean\": "            << mean_value(runs, &RepeatMetrics::total_sec) << ",\n";
    out << "    \"factorization_sec_min\": "     << min_value(runs, &RepeatMetrics::factorization_sec) << ",\n";
    out << "    \"solve_sec_min\": "             << min_value(runs, &RepeatMetrics::solve_sec) << ",\n";
    out << "    \"total_sec_min\": "             << min_value(runs, &RepeatMetrics::total_sec) << ",\n";
    out << "    \"residual_inf_max\": "          << max_value(runs, &RepeatMetrics::residual_inf) << ",\n";
    out << "    \"relative_residual_inf_max\": " << max_value(runs, &RepeatMetrics::relative_residual_inf) << "\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions cli = parse_args(argc, argv);

        if (cli.list_cases) {
            for (const auto& name : exp_20260409::list_case_names(cli.dataset_root))
                std::cout << name << "\n";
            return 0;
        }

        const std::filesystem::path case_dir = resolve_case_dir(cli);
        const auto case_data = exp_20260409::load_cupf_dataset_case(case_dir);

        const auto build_start = Clock::now();
        const auto system = exp_20260409::build_linear_system(case_data, parse_jacobian(cli.jacobian));
        const double build_system_sec = std::chrono::duration<double>(Clock::now() - build_start).count();

        print_system_summary(case_data, system, cli, build_system_sec);

        const int32_t n_pq = static_cast<int32_t>(case_data.pq.size());
        const double rhs_inf = exp_20260409::rhs_inf_norm(system.rhs);

        std::vector<RepeatMetrics> runs;
        runs.reserve(cli.repeats);
        double analysis_sec = 0.0;

        // -----------------------------------------------------------------------
        // Full mode: one runner, analyze once, N factorize_and_solve calls,
        //            download + residual once after the loop.
        // -----------------------------------------------------------------------
        if (cli.mode == "full") {
            CuDssLinearSystemRunner runner(system);
            analysis_sec = runner.analyze();

            // Warmup
            for (int32_t i = 0; i < cli.warmup; ++i) {
                auto m = runner.factorize_and_solve(-(i + 1));
                m.analysis_sec    = 0.0;
                m.no_analysis_sec = m.total_sec;
                print_repeat_summary("warmup", case_data.case_name, cli.mode, m);
            }

            // Measured iterations
            for (int32_t i = 0; i < cli.repeats; ++i) {
                auto m = runner.factorize_and_solve(i);
                m.analysis_sec    = (i == 0) ? analysis_sec : 0.0;
                m.no_analysis_sec = m.total_sec;
                runs.push_back(m);
            }

            // Download and verify once (last iteration's solution).
            runner.download_solution();
            const auto& h_x = runner.solution();
            const double residual = exp_20260409::residual_inf_norm(
                system.structure, system.values, system.rhs, h_x);
            const double sol_inf = exp_20260409::solution_inf_norm(h_x);
            runs.back().residual_inf          = residual;
            runs.back().relative_residual_inf = residual / std::max(rhs_inf, 1e-30);
            runs.back().solution_inf          = sol_inf;

        // -----------------------------------------------------------------------
        // Schur mode
        // -----------------------------------------------------------------------
        } else {
            exp_20260409::CuDssSchurRunner runner(system, n_pq);
            analysis_sec = runner.analyze();

            // Warmup
            for (int32_t i = 0; i < cli.warmup; ++i) {
                const auto t = runner.factorize_and_solve(-(i + 1));
                RepeatMetrics m;
                m.repeat_idx       = t.repeat_idx;
                m.schur_dim        = t.schur_dim;
                m.factorization_sec = t.factorization_sec;
                m.schur_extract_sec = t.schur_extract_sec;
                m.fwd_solve_sec     = t.fwd_solve_sec;
                m.diag_sec          = t.diag_sec;
                m.getrf_sec         = t.getrf_sec;
                m.getrs_sec         = t.getrs_sec;
                m.schur_solve_sec   = t.schur_solve_sec;
                m.bwd_solve_sec     = t.bwd_solve_sec;
                m.solve_sec         = t.solve_sec;
                m.total_sec         = t.total_sec;
                m.no_analysis_sec   = t.total_sec;
                print_repeat_summary("warmup", case_data.case_name, cli.mode, m);
            }

            // Measured iterations
            for (int32_t i = 0; i < cli.repeats; ++i) {
                const auto t = runner.factorize_and_solve(i);
                RepeatMetrics m;
                m.repeat_idx        = t.repeat_idx;
                m.schur_dim         = t.schur_dim;
                m.factorization_sec = t.factorization_sec;
                m.schur_extract_sec = t.schur_extract_sec;
                m.fwd_solve_sec     = t.fwd_solve_sec;
                m.diag_sec          = t.diag_sec;
                m.getrf_sec         = t.getrf_sec;
                m.getrs_sec         = t.getrs_sec;
                m.schur_solve_sec   = t.schur_solve_sec;
                m.bwd_solve_sec     = t.bwd_solve_sec;
                m.solve_sec         = t.solve_sec;
                m.total_sec         = t.total_sec;
                m.analysis_sec      = (i == 0) ? analysis_sec : 0.0;
                m.no_analysis_sec   = t.total_sec;
                runs.push_back(m);
            }

            // Download and verify once (last iteration's solution).
            runner.download_solution();
            const auto& h_x = runner.solution();
            const double residual = exp_20260409::residual_inf_norm(
                system.structure, system.values, system.rhs, h_x);
            const double sol_inf = exp_20260409::solution_inf_norm(h_x);
            runs.back().residual_inf          = residual;
            runs.back().relative_residual_inf = residual / std::max(rhs_inf, 1e-30);
            runs.back().solution_inf          = sol_inf;
        }

        for (const auto& m : runs)
            print_repeat_summary("run", case_data.case_name, cli.mode, m);
        print_benchmark_summary(case_data.case_name, cli.mode, runs);

        if (!cli.output_json.empty()) {
            write_summary_json(cli.output_json, cli, case_dir, case_data, system,
                               build_system_sec, analysis_sec, runs);
            std::cout << "[json] wrote " << cli.output_json.string() << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cudss_benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
