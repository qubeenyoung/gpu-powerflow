#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <numeric>

#include "nr_data.hpp"
#include "newtonpf.hpp"
#include "timer.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace std;
using namespace std::chrono;
using namespace nr_data;
namespace fs = std::filesystem;

// 크기별 분류
string classify_by_size(int nb) {
    if (nb <= 200) return "≤200";
    else if (nb <= 1000) return "200-1k";
    else if (nb <= 10000) return "1k-10k";
    else if (nb <= 20000) return "10k-20k";
    else if (nb <= 30000) return "20k-30k";
    else if (nb <= 40000) return "30k-40k";
    else if (nb <= 50000) return "40k-50k";
    else return ">50k";
}

// 세부 타이밍 결과 구조체
struct DetailedTimings {
    double analyze_jacobian_ms;    // AnalyzeJacobian (CPU 희소 패턴 분석)
    double cuda_init_ms;           // CUDA_Initialize (GPU 메모리 할당)
    double cudss_analyze_ms;       // cuDSS_AnalyzePattern (Symbolic Factorization)
    double analyze_total_ms;       // 위 3개 합
    double mismatch_avg_ms;
    double update_jacobian_avg_ms;
    double factorize_avg_ms;
    double solve_avg_ms;
    double update_v_avg_ms;
    double total_per_iter_ms;
};

// 벤치마크 결과 구조체
struct BenchmarkResult {
    string case_name;
    int nb;
    int nnz;
    int iterations;
    bool converged;
    double final_normF;
    DetailedTimings timings;
};

// 단일 케이스 벤치마크 (타이머 기록을 외부에서 수집)
BenchmarkResult benchmark_case(const string& case_name, const string& mode) {
    BenchmarkResult result;
    result.case_name = case_name;

    try {
        // 데이터 로드
        NRData case_data;
        case_data.load_data(case_name);

        result.nb = case_data.V0.size();
        result.nnz = case_data.Ybus.nonZeros();

        cout << "  Benchmarking: " << case_name << " (nb=" << result.nb << ")" << endl;

        // Timer 로그 초기화 - 매 케이스마다 새로 시작
#ifdef USE_BLOCK_TIMER
        // 기존 로거 제거 (있으면)
        spdlog::drop("benchmark_timer");
        // 로그 파일 삭제 (truncate 보장)
        std::remove("/tmp/benchmark_timer.log");
        // 새 로거 생성
        auto logger = spdlog::basic_logger_mt("benchmark_timer", "/tmp/benchmark_timer.log", true);
        logger->set_pattern("%v");
        logger->flush_on(spdlog::level::info);
        init_timer_logger(logger);
#endif

        // Newton-Raphson 실행
        auto start = high_resolution_clock::now();
        NRResult nr_result = newtonPF(
            case_data.Ybus,
            case_data.Sbus,
            case_data.V0,
            case_data.pv,
            case_data.pq,
            1e-8, 10
        );
        auto end = high_resolution_clock::now();

        result.iterations = nr_result.iter;
        result.converged = nr_result.converged;
        result.final_normF = nr_result.normF;

        // 수렴 여부와 상관없이 측정 계속
        string status = result.converged ? "converged" : "did not converge";
        cout << "    " << result.iterations << " iterations (" << status << ")" << endl;

        // 타이머 로그 파싱
#ifdef USE_BLOCK_TIMER
        spdlog::drop("benchmark_timer");
#endif

        // 로그에서 타이밍 추출
        ifstream log_file("/tmp/benchmark_timer.log");
        string line;

        double analyze_jacobian = 0, cuda_init = 0, cudss_analyze = 0;
        vector<double> mismatch_times, jacobian_times, permutation_times, factorize_times, solve_times, update_v_times, total_iter_times;

        while (getline(log_file, line)) {
            if (line.find("AnalyzeJacobian") != string::npos) {
                double t;
                sscanf(line.c_str(), "AnalyzeJacobian %lf", &t);
                analyze_jacobian += t;
            }
            else if (line.find("cuDSS_AnalyzePattern") != string::npos) {
                double t;
                sscanf(line.c_str(), "cuDSS_AnalyzePattern %lf", &t);
                cudss_analyze += t;
            }
            else if (line.find("CUDA_Initialize") != string::npos) {
                double t;
                sscanf(line.c_str(), "CUDA_Initialize %lf", &t);
                cuda_init += t;
            }
            else if (line.find("Mismatch_") != string::npos) {
                double t; sscanf(line.c_str(), "Mismatch_%*d %lf", &t);
                mismatch_times.push_back(t);
            }
            else if (line.find("UpdateJacobian_") != string::npos || line.find("CUDA_UpdateJacobian_") != string::npos) {
                double t;
                // FP32 Mixed Precision or FP64
                if (sscanf(line.c_str(), "CUDA_UpdateJacobian_FP32_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "CUDA_UpdateJacobian_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "CPU_UpdateJacobian_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "UpdateJacobian_%*d %lf", &t) == 1) {
                    jacobian_times.push_back(t);
                }
            }
            else if (line.find("Permutation_") != string::npos) {
                // GPU Permutation - 별도 배열로 분리
                double t;
                if (sscanf(line.c_str(), "GPU_Permutation_FP32_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "GPU_Permutation_%*d %lf", &t) == 1) {
                    permutation_times.push_back(t);
                }
            }
            else if (line.find("Factorize_") != string::npos || line.find("cuDSS_Factorize_") != string::npos) {
                double t;
                // FP32 Mixed Precision or FP64
                if (sscanf(line.c_str(), "cuDSS_Factorize_FP32_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "cuDSS_Factorize_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "Factorize_%*d %lf", &t) == 1) {
                    factorize_times.push_back(t);
                }
            }
            else if (line.find("Solve_") != string::npos || line.find("cuDSS_Solve_") != string::npos) {
                double t;
                // FP32 Mixed Precision or FP64
                if (sscanf(line.c_str(), "cuDSS_Solve_FP32_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "cuDSS_Solve_%*d %lf", &t) == 1 ||
                    sscanf(line.c_str(), "Solve_%*d %lf", &t) == 1) {
                    solve_times.push_back(t);
                }
            }
            else if (line.find("UpdateV_") != string::npos) {
                double t; sscanf(line.c_str(), "UpdateV_%*d %lf", &t);
                update_v_times.push_back(t);
            }
        }

        // 평균 계산 (첫 번째 warm-up iteration 제외)
        auto avg_skip_first = [](const vector<double>& v) {
            if (v.size() <= 1) return v.empty() ? 0.0 : v[0];
            // 첫 번째 제외하고 평균
            return std::accumulate(v.begin() + 1, v.end(), 0.0) / (v.size() - 1);
        };

        // 타이머 로그는 초(seconds) 단위이므로 ms로 변환
        result.timings.analyze_jacobian_ms = analyze_jacobian * 1000.0;
        result.timings.cuda_init_ms = cuda_init * 1000.0;
        result.timings.cudss_analyze_ms = cudss_analyze * 1000.0;
        result.timings.analyze_total_ms = (analyze_jacobian + cuda_init + cudss_analyze) * 1000.0;
        result.timings.mismatch_avg_ms = avg_skip_first(mismatch_times) * 1000.0;
        // Update Jacobian = CUDA_UpdateJacobian + GPU_Permutation
        double jacobian_avg = avg_skip_first(jacobian_times) * 1000.0;
        double permutation_avg = avg_skip_first(permutation_times) * 1000.0;
        result.timings.update_jacobian_avg_ms = jacobian_avg + permutation_avg;
        result.timings.factorize_avg_ms = avg_skip_first(factorize_times) * 1000.0;
        result.timings.solve_avg_ms = avg_skip_first(solve_times) * 1000.0;
        result.timings.update_v_avg_ms = avg_skip_first(update_v_times) * 1000.0;

        result.timings.total_per_iter_ms =
            result.timings.mismatch_avg_ms +
            result.timings.update_jacobian_avg_ms +
            result.timings.factorize_avg_ms +
            result.timings.solve_avg_ms +
            result.timings.update_v_avg_ms;

        cout << "    Avg time per iteration: " << fixed << setprecision(3)
             << result.timings.total_per_iter_ms << " ms" << endl;

    } catch (const exception& e) {
        cout << "    ERROR: " << e.what() << endl;
        result.converged = false;
    }

    return result;
}

int main(int argc, char** argv) {
    string mode = "cpu";  // cpu or gpu

    if (argc > 1) {
        mode = argv[1];
    }

#ifdef USE_CUDA
    mode = "gpu";
#endif

    cout << "========================================" << endl;
    cout << "  C++ Detailed Benchmark (" << mode << " mode)" << endl;
    cout << "========================================" << endl;

    string dataset_root = "/workspace/datasets/nr_dataset";

    // 모든 케이스 찾기
    vector<string> cases;
    for (const auto& entry : fs::directory_iterator(dataset_root)) {
        if (entry.is_directory()) {
            string case_name = entry.path().filename().string();
            if (fs::exists(entry.path() / "Ybus.npz")) {
                cases.push_back(case_name);
            }
        }
    }

    sort(cases.begin(), cases.end());

    cout << "Found " << cases.size() << " cases" << endl;
    cout << string(80, '=') << endl;

    // GPU Warm-up: 첫 번째 케이스를 2번 실행해서 GPU를 예열
    if (!cases.empty()) {
        cout << "[WARM-UP] Running first case twice to warm up GPU..." << endl;
        benchmark_case(cases[0], mode);  // 첫 번째 실행 (cold)
        benchmark_case(cases[0], mode);  // 두 번째 실행 (warm)
        cout << "[WARM-UP] Done. Starting actual benchmark..." << endl;
        cout << string(80, '=') << endl;
    }

    vector<BenchmarkResult> results;

    for (const auto& case_name : cases) {
        auto result = benchmark_case(case_name, mode);
        // 수렴 여부와 상관없이 모든 결과 저장
        results.push_back(result);
    }

    // 크기별로 그룹화
    map<string, vector<BenchmarkResult>> by_size;
    for (const auto& r : results) {
        string size_cat = classify_by_size(r.nb);
        by_size[size_cat].push_back(r);
    }

    // 크기별 평균 계산
    cout << "\n" << string(80, '=') << endl;
    cout << "SUMMARY: C++ (" << mode << ") Performance by Size Category" << endl;
    cout << string(80, '=') << endl;

    vector<string> size_order = {"≤200", "200-1k", "1k-10k", "10k-20k", "20k-30k", "30k-40k", "40k-50k", ">50k"};

    // JSON 출력 준비
    ofstream json_out("/workspace/benchmark_cpp_" + mode + "_detailed.json");
    json_out << "{\n";
    json_out << "  \"language\": \"cpp\",\n";
    json_out << "  \"mode\": \"" << mode << "\",\n";
    json_out << "  \"summary_by_size\": {\n";

    bool first_size = true;

    for (const auto& size_cat : size_order) {
        if (by_size.find(size_cat) == by_size.end()) continue;

        const auto& group = by_size[size_cat];
        int n = group.size();

        // 평균 계산
        double avg_nb = 0, avg_analyze_jacobian = 0, avg_cuda_init = 0, avg_cudss_analyze = 0;
        double avg_analyze_total = 0, avg_mismatch = 0, avg_jacobian = 0;
        double avg_factorize = 0, avg_solve = 0, avg_update_v = 0, avg_total = 0;

        for (const auto& r : group) {
            avg_nb += r.nb;
            avg_analyze_jacobian += r.timings.analyze_jacobian_ms;
            avg_cuda_init += r.timings.cuda_init_ms;
            avg_cudss_analyze += r.timings.cudss_analyze_ms;
            avg_analyze_total += r.timings.analyze_total_ms;
            avg_mismatch += r.timings.mismatch_avg_ms;
            avg_jacobian += r.timings.update_jacobian_avg_ms;
            avg_factorize += r.timings.factorize_avg_ms;
            avg_solve += r.timings.solve_avg_ms;
            avg_update_v += r.timings.update_v_avg_ms;
            avg_total += r.timings.total_per_iter_ms;
        }

        avg_nb /= n; avg_analyze_jacobian /= n; avg_cuda_init /= n; avg_cudss_analyze /= n;
        avg_analyze_total /= n; avg_mismatch /= n; avg_jacobian /= n;
        avg_factorize /= n; avg_solve /= n; avg_update_v /= n; avg_total /= n;

        cout << "\n[" << size_cat << "] (" << n << " cases, avg_nb=" << (int)avg_nb << ")" << endl;
        cout << "  AnalyzeJacobian:   " << setw(8) << fixed << setprecision(3) << avg_analyze_jacobian << " ms" << endl;
        cout << "  CUDA_Initialize:   " << setw(8) << avg_cuda_init << " ms" << endl;
        cout << "  cuDSS_Analyze:     " << setw(8) << avg_cudss_analyze << " ms" << endl;
        cout << "  ─────────────────────────────" << endl;
        cout << "  Analyze Total:     " << setw(8) << avg_analyze_total << " ms" << endl;
        cout << "  ─────────────────────────────" << endl;
        cout << "  Mismatch:          " << setw(8) << avg_mismatch << " ms" << endl;
        cout << "  Update Jacobian:   " << setw(8) << avg_jacobian << " ms" << endl;
        cout << "  Factorize (LU):    " << setw(8) << avg_factorize << " ms" << endl;
        cout << "  Solve:             " << setw(8) << avg_solve << " ms" << endl;
        cout << "  Update V:          " << setw(8) << avg_update_v << " ms" << endl;
        cout << "  ─────────────────────────────" << endl;
        cout << "  Total/iter:        " << setw(8) << avg_total << " ms" << endl;

        // JSON 출력
        if (!first_size) json_out << ",\n";
        first_size = false;

        json_out << "    \"" << size_cat << "\": {\n";
        json_out << "      \"count\": " << n << ",\n";
        json_out << "      \"avg_nb\": " << (int)avg_nb << ",\n";
        json_out << "      \"analyze_jacobian\": " << avg_analyze_jacobian << ",\n";
        json_out << "      \"cuda_init\": " << avg_cuda_init << ",\n";
        json_out << "      \"cudss_analyze\": " << avg_cudss_analyze << ",\n";
        json_out << "      \"analyze_total\": " << avg_analyze_total << ",\n";
        json_out << "      \"mismatch\": " << avg_mismatch << ",\n";
        json_out << "      \"update_jacobian\": " << avg_jacobian << ",\n";
        json_out << "      \"factorize\": " << avg_factorize << ",\n";
        json_out << "      \"solve\": " << avg_solve << ",\n";
        json_out << "      \"update_v\": " << avg_update_v << ",\n";
        json_out << "      \"total_per_iter\": " << avg_total << "\n";
        json_out << "    }";
    }

    json_out << "\n  }\n";
    json_out << "}\n";
    json_out.close();

    cout << "\n" << string(80, '=') << endl;
    cout << "Results saved to: /workspace/benchmark_cpp_" << mode << "_detailed.json" << endl;

    return 0;
}
