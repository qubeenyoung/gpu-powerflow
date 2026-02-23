
#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>

#include "nr_data.hpp"     // NRData, NRResult
#include "newtonpf.hpp"    // nr_data::NRResult newtonpf(const nr_data::NRData&)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "timer.hpp"       
#include "dump.hpp"

// -------------------- Utility --------------------

static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
                << "  " << prog << " --case <case_name> --out <result_dir>\n"
                << "Example:\n"
                << "  " << prog << " --case case9 --out runs/\n";
}

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
        std::string log_path = base_path + "/" + "timer.log";
        auto logger = spdlog::basic_logger_mt("timer", log_path);
        logger->set_pattern("%v");  // 소영수정: 간단한 포맷 (시간 없이)
        logger->flush_on(spdlog::level::info);  // 소영수정: 즉시 flush
        init_timer_logger(logger);
        #endif

        #ifdef DUMP_DATA
        std::string dump_path = base_path + "/" + "dump";
        init_dump_path(dump_path);
        #endif

        spdlog::info("=== NewtonPF start ===");
        spdlog::info("case: {}", case_name);

        nr_data::NRData case_data;
        case_data.load_data(case_name);

        nr_data::NRResult result;
        {
            BlockTimer timer("newtonpf");
            result = newtonPF(
                case_data.Ybus,
                case_data.Sbus,
                case_data.V0,
                case_data.pv,
                case_data.pq,
                1e-8, 10
            );
        }

        result.save_result(case_name);

        // 소영수정: 최종 결과를 콘솔에 출력
        std::cout << "\n";
        if (result.converged) {
            std::cout << "Newton's method power flow converged in "
                      << result.iter << " iterations.\n";
        } else {
            std::cout << "Newton's method power flow did not converge in "
                      << result.iter << " iterations.\n";
        }
        std::cout << "Final mismatch (normF): " << result.normF << "\n";
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}