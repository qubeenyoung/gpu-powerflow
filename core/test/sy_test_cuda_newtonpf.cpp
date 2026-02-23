// 소영CUDA: Test program for CUDA-accelerated Newton-Raphson power flow

#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <fstream>  // 소영CUDA: ofstream
#include <nlohmann/json.hpp>  // 소영CUDA: json

#include "nr_data.hpp"
#include "sy_cuda_newtonpf.cuh"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "timer.hpp"
#include "dump.hpp"  // 소영CUDA: dump.hpp already includes io.hpp

// 소영CUDA: Print usage
static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " --case <case_name> --out <result_dir>\n"
              << "Example:\n"
              << "  " << prog << " --case case14 --out runs/cuda_pf_logs\n";
}

// 소영CUDA: Parse command line arguments
static void parse_args(int argc, char** argv, std::string& case_name, std::string& out_root) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--case" && i + 1 < argc)
            case_name = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            out_root = argv[++i];
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    if (case_name.empty() || out_root.empty()) {
        throw std::runtime_error("Missing required arguments (--case, --out)");
    }
}

int main(int argc, char** argv) {
    std::string case_name, save_root;

    try {
        parse_args(argc, argv, case_name, save_root);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    std::string base_path = save_root + "/" + case_name;
    std::filesystem::create_directories(base_path);

    try {
        #ifdef USE_BLOCK_TIMER
        std::string log_path = base_path + "/" + "cuda_timer.log";
        auto logger = spdlog::basic_logger_mt("timer", log_path);
        logger->set_pattern("%v");  // 소영CUDA: Simple format
        logger->flush_on(spdlog::level::info);  // 소영CUDA: Immediate flush
        init_timer_logger(logger);
        #endif

        #ifdef DUMP_DATA
        std::string dump_path = base_path + "/" + "cuda_dump";
        init_dump_path(dump_path);
        #endif

        spdlog::info("=== CUDA NewtonPF start ===");
        spdlog::info("case: {}", case_name);

        // 소영CUDA: Load case data
        nr_data::NRData case_data;
        case_data.load_data(case_name);

        // 소영CUDA: Run CUDA Newton-Raphson
        nr_data::NRResult result;
        {
            BlockTimer timer("CUDA_newtonpf");
            result = cuda_nr::cuda_newtonPF(
                case_data.Ybus,
                case_data.Sbus,
                case_data.V0,
                case_data.pv,
                case_data.pq,
                1e-8, 20
            );
        }

        // 소영CUDA: Save result
        spdlog::info("Result obtained: converged={}, iter={}, normF={}, V.size()={}",
                     result.converged, result.iter, result.normF, result.V.size());

        std::string result_path = std::string("datasets/nr_dataset") + "/" + case_name;
        std::string v_file = result_path + "/V_cuda.npy";

        // 소영수정: Skip V saving for now (cnpy crash issue)
        spdlog::info("Skipping V save (cnpy complex type issue)");

        std::string summary_file = result_path + "/summary_cuda.json";
        std::ofstream ofs(summary_file);
        nlohmann::json summary;
        summary["converged"] = result.converged;
        summary["iter"] = result.iter;
        summary["normF"] = result.normF;
        ofs << summary.dump(4);
        ofs.close();
        spdlog::info("Saved CUDA summary to {}", summary_file);

        // 소영CUDA: Print result to console
        std::cout << "\n";
        if (result.converged) {
            std::cout << "[CUDA] Newton's method power flow converged in "
                      << result.iter << " iterations.\n";
        } else {
            std::cout << "[CUDA] Newton's method power flow did not converge in "
                      << result.iter << " iterations.\n";
        }
        std::cout << "[CUDA] Final mismatch (normF): " << result.normF << "\n";
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}
