#include "dump_case_loader.hpp"
#include "newton_solver/core/newton_solver.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <complex>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CUPF_WITH_CUDA
  #include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    fs::path case_root;
    fs::path output_dir;
    std::vector<std::string> cases;
    std::string backend = "cuda";
    std::string compute = "mixed";
    std::string cpu_linear_solver = "klu";
    std::string cuda_jacobian = "edge";
    std::string cuda_linear_solver = "cudss";
    std::string custom_precision = "fp32";   // custom 백엔드 factor 정밀도: fp64|fp32|tf32
    bool custom_serial_nd = false;           // custom 결정적 serial METIS-ND
    int32_t custom_seed = 42;                // custom nested-dissection seed
    double tolerance = 1e-8;
    int32_t max_iter = 50;
    int32_t warmup = 1;
    int32_t repeats = 5;
};

std::string require_value(int& i, int argc, char** argv, const std::string& name)
{
    if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
    }
    return argv[++i];
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--case-root") args.case_root = require_value(i, argc, argv, key);
        else if (key == "--output-dir") args.output_dir = require_value(i, argc, argv, key);
        else if (key == "--backend") args.backend = require_value(i, argc, argv, key);
        else if (key == "--compute") args.compute = require_value(i, argc, argv, key);
        else if (key == "--cpu-linear-solver") args.cpu_linear_solver = require_value(i, argc, argv, key);
        else if (key == "--cuda-jacobian") args.cuda_jacobian = require_value(i, argc, argv, key);
        else if (key == "--cuda-linear-solver") args.cuda_linear_solver = require_value(i, argc, argv, key);
        else if (key == "--custom-precision") args.custom_precision = require_value(i, argc, argv, key);
        else if (key == "--custom-serial-nd") args.custom_serial_nd = (std::stoi(require_value(i, argc, argv, key)) != 0);
        else if (key == "--custom-seed") args.custom_seed = std::stoi(require_value(i, argc, argv, key));
        else if (key == "--tolerance") args.tolerance = std::stod(require_value(i, argc, argv, key));
        else if (key == "--max-iter") args.max_iter = std::stoi(require_value(i, argc, argv, key));
        else if (key == "--warmup") args.warmup = std::stoi(require_value(i, argc, argv, key));
        else if (key == "--repeats") args.repeats = std::stoi(require_value(i, argc, argv, key));
        else if (key == "--cases") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                args.cases.push_back(argv[++i]);
            }
        } else if (key == "-h" || key == "--help") {
            std::cout << "Usage: cupf_cpp_evaluate --case-root DIR --output-dir DIR [options]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.case_root.empty()) throw std::invalid_argument("--case-root is required");
    if (args.output_dir.empty()) throw std::invalid_argument("--output-dir is required");
    return args;
}

std::string csv_escape(const std::string& input)
{
    const bool quote = input.find(char(44)) != std::string::npos ||
                       input.find(char(34)) != std::string::npos ||
                       input.find(char(10)) != std::string::npos ||
                       input.find(char(13)) != std::string::npos;
    if (!quote) return input;
    std::string out(1, char(34));
    for (char ch : input) {
        if (ch == char(34)) {
            out += char(34);
            out += char(34);
        } else {
            out += ch;
        }
    }
    out += char(34);
    return out;
}

void write_csv_row(std::ofstream& out, const std::vector<std::string>& values)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << char(44);
        out << csv_escape(values[i]);
    }
    out << char(10);
}

template <typename T>
std::string to_string_precise(T value)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

bool is_comment_or_empty(const std::string& line)
{
    for (char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == 35 || ch == 37;
        }
    }
    return true;
}

std::vector<std::complex<double>> load_complex_pairs(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::vector<std::complex<double>> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) continue;
        std::istringstream iss(line);
        double re = 0.0;
        double im = 0.0;
        if (!(iss >> re >> im)) {
            throw std::runtime_error("malformed complex pair in " + path.string());
        }
        values.emplace_back(re, im);
    }
    return values;
}

std::vector<fs::path> case_dirs(const Args& args)
{
    std::vector<fs::path> dirs;
    if (!args.cases.empty()) {
        for (const auto& name : args.cases) {
            fs::path dir = args.case_root / name;
            if (!fs::exists(dir) && name.size() > 2 && name.substr(name.size() - 2) == ".m") {
                dir = args.case_root / name.substr(0, name.size() - 2);
            }
            dirs.push_back(dir);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(args.case_root)) {
            if (entry.is_directory() && fs::exists(entry.path() / "dump_Ybus.mtx")) {
                dirs.push_back(entry.path());
            }
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

NewtonOptions make_options(const Args& args)
{
    NewtonOptions options;
    if (args.backend == "cpu") options.backend = BackendKind::CPU;
    else if (args.backend == "cuda") options.backend = BackendKind::CUDA;
    else throw std::invalid_argument("backend must be cpu or cuda");

    if (args.compute == "fp64") options.compute = ComputePolicy::FP64;
    else if (args.compute == "fp32") options.compute = ComputePolicy::FP32;
    else if (args.compute == "mixed") options.compute = ComputePolicy::Mixed;
    else throw std::invalid_argument("compute must be fp64, fp32, or mixed");

    if (args.cpu_linear_solver == "klu") options.cpu_linear_solver = CpuLinearSolverKind::KLU;
    else if (args.cpu_linear_solver == "umfpack") options.cpu_linear_solver = CpuLinearSolverKind::UMFPACK;
    else throw std::invalid_argument("cpu-linear-solver must be klu or umfpack");

    if (args.cuda_jacobian == "edge") options.cuda_jacobian = CudaJacobianKind::Edge;
    else if (args.cuda_jacobian == "edge_atomic") options.cuda_jacobian = CudaJacobianKind::EdgeAtomic;
    else if (args.cuda_jacobian == "vertex_warp") options.cuda_jacobian = CudaJacobianKind::VertexWarp;
    else throw std::invalid_argument("cuda-jacobian must be edge, edge_atomic, or vertex_warp");

    if (args.cuda_linear_solver == "cudss") options.cuda_linear_solver = CudaLinearSolverKind::CuDSS;
    else if (args.cuda_linear_solver == "custom") options.cuda_linear_solver = CudaLinearSolverKind::Custom;
    else throw std::invalid_argument("cuda-linear-solver must be cudss or custom");

    if (args.custom_precision == "fp64") options.custom.precision = CustomPrecision::FP64;
    else if (args.custom_precision == "fp32") options.custom.precision = CustomPrecision::FP32;
    else if (args.custom_precision == "tf32") options.custom.precision = CustomPrecision::TF32;
    else throw std::invalid_argument("custom-precision must be fp64, fp32, or tf32");
    options.custom.serial_nd  = args.custom_serial_nd;
    options.custom.metis_seed = args.custom_seed;
    return options;
}

void sync_if_needed(const Args& args)
{
#ifdef CUPF_WITH_CUDA
    if (args.backend == "cuda") {
        cudaDeviceSynchronize();
    }
#else
    (void)args;
#endif
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double max_voltage_error(const std::vector<std::complex<double>>& actual,
                         const std::vector<std::complex<double>>& reference)
{
    if (actual.size() != reference.size() || actual.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double max_err = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_err = std::max(max_err, std::abs(actual[i] - reference[i]));
    }
    return max_err;
}

double rms_voltage_error(const std::vector<std::complex<double>>& actual,
                         const std::vector<std::complex<double>>& reference)
{
    if (actual.size() != reference.size() || actual.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double err = std::abs(actual[i] - reference[i]);
        sum += err * err;
    }
    return std::sqrt(sum / static_cast<double>(actual.size()));
}

void write_timing_rows(std::ofstream& timing_csv,
                       const std::string& case_name,
                       int32_t repeat_idx,
                       const std::string& phase,
                       const std::vector<newton_solver::utils::TimingEntry>& entries)
{
    for (const auto& entry : entries) {
        const double avg_us = entry.count ? static_cast<double>(entry.total_us) / entry.count : 0.0;
        write_csv_row(timing_csv,
                      {case_name,
                       std::to_string(repeat_idx),
                       phase,
                       entry.name ? entry.name : "",
                       std::to_string(entry.count),
                       std::to_string(entry.total_us),
                       to_string_precise(avg_us)});
    }
}

struct SolveMeasurement {
    double initialize_ms = 0.0;
    double solve_ms = 0.0;
    NRResult result;
    std::vector<newton_solver::utils::TimingEntry> initialize_timing;
    std::vector<newton_solver::utils::TimingEntry> solve_timing;
};

SolveMeasurement run_once(const Args& args, const cupf::tests::DumpCaseData& data)
{
    const YbusView ybus = data.ybus();
    NRConfig config;
    config.tolerance = args.tolerance;
    config.max_iter = args.max_iter;

    SolveMeasurement measurement;
    NewtonSolver solver(make_options(args));

    newton_solver::utils::resetTimingCollector();
    sync_if_needed(args);
    const auto init_start = Clock::now();
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));
    sync_if_needed(args);
    measurement.initialize_ms = elapsed_ms(init_start, Clock::now());
    measurement.initialize_timing = newton_solver::utils::timingSnapshot();

    newton_solver::utils::resetTimingCollector();
    sync_if_needed(args);
    const auto solve_start = Clock::now();
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 measurement.result);
    sync_if_needed(args);
    measurement.solve_ms = elapsed_ms(solve_start, Clock::now());
    measurement.solve_timing = newton_solver::utils::timingSnapshot();
    return measurement;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_args(argc, argv);
        fs::create_directories(args.output_dir);

        std::ofstream runs_csv(args.output_dir / "runs.csv");
        std::ofstream timing_csv(args.output_dir / "timing.csv");
        write_csv_row(runs_csv,
                      {"mode", "case_name", "case_path", "backend", "compute", "cuda_linear_solver",
                       "repeat_idx", "warmup", "success", "error_message", "n_bus", "ybus_nnz",
                       "n_ref", "n_pv", "n_pq", "initialize_ms", "solve_ms", "device_solve_ms",
                       "cupf_reported_total_ms", "cupf_converged", "cupf_iterations", "cupf_final_mismatch",
                       "output_mismatch", "reference_converged", "reference_iterations", "reference_final_mismatch",
                       "max_abs_v_error", "rms_abs_v_error", "max_abs_vm_error", "max_abs_va_error"});
        write_csv_row(timing_csv,
                      {"case_name", "repeat_idx", "phase", "timer_name", "count", "total_us", "avg_us"});

        for (const auto& dir : case_dirs(args)) {
            const std::string case_name = dir.filename().string();
            try {
                const auto data = cupf::tests::load_dump_case(dir);
                const auto reference = load_complex_pairs(dir / "dump_Vref.txt");

                for (int32_t i = 0; i < args.warmup; ++i) {
                    (void)run_once(args, data);
                }

                for (int32_t repeat = 0; repeat < args.repeats; ++repeat) {
                    SolveMeasurement measurement = run_once(args, data);
                    const double max_err = max_voltage_error(measurement.result.V, reference);
                    const double rms_err = rms_voltage_error(measurement.result.V, reference);
                    write_csv_row(runs_csv,
                                  {"cpp",
                                   case_name,
                                   dir.string(),
                                   args.backend,
                                   args.compute,
                                   args.cuda_linear_solver,
                                   std::to_string(repeat),
                                   std::to_string(args.warmup),
                                   "1",
                                   "",
                                   std::to_string(data.rows),
                                   std::to_string(data.ybus_data.size()),
                                   "",
                                   std::to_string(data.pv.size()),
                                   std::to_string(data.pq.size()),
                                   to_string_precise(measurement.initialize_ms),
                                   to_string_precise(measurement.solve_ms),
                                   "",
                                   "",
                                   measurement.result.converged ? "1" : "0",
                                   std::to_string(measurement.result.iterations),
                                   to_string_precise(measurement.result.final_mismatch),
                                   to_string_precise(measurement.result.final_mismatch),
                                   "",
                                   "",
                                   "",
                                   to_string_precise(max_err),
                                   to_string_precise(rms_err),
                                   "",
                                   ""});
                    write_timing_rows(timing_csv, case_name, repeat, "initialize", measurement.initialize_timing);
                    write_timing_rows(timing_csv, case_name, repeat, "solve", measurement.solve_timing);
                    std::cout << "[cpp][OK] case=" << case_name
                              << " repeat=" << repeat
                              << " init_ms=" << measurement.initialize_ms
                              << " solve_ms=" << measurement.solve_ms
                              << " mismatch=" << measurement.result.final_mismatch
                              << " err=" << max_err << std::endl;
                }
            } catch (const std::exception& exc) {
                write_csv_row(runs_csv,
                              {"cpp", case_name, dir.string(), args.backend, args.compute,
                               args.cuda_linear_solver, "", std::to_string(args.warmup), "0", exc.what()});
                std::cerr << "[cpp][FAIL] case=" << case_name << " error=" << exc.what() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cupf_cpp_evaluate: " << exc.what() << std::endl;
        return 2;
    }
}
