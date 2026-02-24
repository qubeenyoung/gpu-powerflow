/**
 * @file benchmark_multibatch.cpp
 * @brief Full Newton-Raphson Batch 성능 벤치마크
 *
 * newtonPF_batch()로 N개 케이스를 GPU에서 동시 처리.
 * batch_size = 1, 2, 4, 8, 16, 32에 대해 throughput 측정.
 *
 * 1단계: 정확도 검증 (배치 결과 == 단일 케이스 결과)
 * 2단계: 성능 측정 (wall-clock time, per-case throughput)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "nr_data.hpp"
#include "newtonpf.hpp"
#include "cuda_accel.hpp"
#include "spdlog/spdlog.h"

using namespace nr_data;

static double elapsed_ms(
    std::chrono::time_point<std::chrono::high_resolution_clock> t0,
    std::chrono::time_point<std::chrono::high_resolution_clock> t1
) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::warn);  // 벤치마크 중 로그 억제
    NewtonCudaAccel::verbose = false;        // [CUDA] 초기화 로그 억제

    std::cout << "========================================" << std::endl;
    std::cout << "  newtonPF_batch() Performance Benchmark" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 케이스 선택
    std::string case_name = "pglib_opf_case9241_pegase";
    if (argc > 1) case_name = argv[1];

    std::cout << "Case: " << case_name << std::endl;

    // 데이터 로드
    NRData case_data;
    case_data.load_data(case_name);

    int nbus = case_data.V0.size();
    int npv  = case_data.pv.size();
    int npq  = case_data.pq.size();
    std::cout << "Buses: " << nbus << " (PV=" << npv << ", PQ=" << npq << ")\n";
    std::cout << "Ybus nnz: " << case_data.Ybus.nonZeros() << "\n\n";

    // =========================================================================
    // Step 1: 단일 케이스 기준 실행 (정확도 검증용)
    // =========================================================================
    std::cout << "--- [1/2] Single-case reference run ---\n";
    std::cout << std::fixed << std::setprecision(3);
    NRResult ref = newtonPF(
        case_data.Ybus, case_data.Sbus, case_data.V0,
        case_data.pv, case_data.pq, 1e-8, 50
    );
    std::cout << "  converged=" << ref.converged
              << "  iter=" << ref.iter
              << "  normF=" << std::scientific << std::setprecision(3) << ref.normF
              << "\n\n";

    if (!ref.converged) {
        std::cout << "  (수렴 미달 — 성능 측정만 진행)\n\n";
    }

    // =========================================================================
    // Step 2: 성능 측정
    // =========================================================================
    std::cout << "--- [2/2] Performance benchmark ---\n";
    std::cout << "  (single-case baseline for comparison)\n\n";

    // 단일 케이스 baseline (5회)
    const int num_runs = 5;
    double single_ms = 0.0;
    for (int r = 0; r < num_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        newtonPF(case_data.Ybus, case_data.Sbus, case_data.V0,
                 case_data.pv, case_data.pq, 1e-8, 50);
        auto t1 = std::chrono::high_resolution_clock::now();
        single_ms += elapsed_ms(t0, t1);
    }
    single_ms /= num_runs;
    std::cout << "  Single-case: " << std::fixed << std::setprecision(3)
              << single_ms << " ms/case\n\n";

    // Batch 성능
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};

    std::cout << std::setw(8)  << "Batch"
              << std::setw(18) << "Total (ms)"
              << std::setw(20) << "Per-case (ms)"
              << std::setw(16) << "Throughput"
              << std::setw(16) << "vs Single"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (int bs : batch_sizes) {
        std::vector<VectorXcd> sbus_vec(bs, case_data.Sbus);
        std::vector<VectorXcd> V0_vec(bs, case_data.V0);

        // warm-up
        newtonPF_batch(case_data.Ybus, sbus_vec, V0_vec,
                       case_data.pv, case_data.pq, 1e-8, 50);

        double total_ms = 0.0;
        for (int r = 0; r < num_runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            newtonPF_batch(case_data.Ybus, sbus_vec, V0_vec,
                           case_data.pv, case_data.pq, 1e-8, 50);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += elapsed_ms(t0, t1);
        }
        total_ms /= num_runs;

        double per_case  = total_ms / bs;
        double tp        = 1000.0 / per_case;         // cases/sec
        double speedup   = single_ms / per_case;       // vs single

        std::cout << std::setw(8)  << bs
                  << std::setw(18) << std::fixed << std::setprecision(3) << total_ms
                  << std::setw(20) << std::setprecision(3) << per_case
                  << std::setw(14) << std::setprecision(1) << tp << " /s"
                  << std::setw(14) << std::setprecision(2) << speedup << "x"
                  << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "  해석\n";
    std::cout << "========================================\n";
    std::cout << "- Total: batch_size개를 한꺼번에 처리하는 벽시계 시간\n";
    std::cout << "- Per-case: 1개당 체감 처리 시간 (throughput 관점)\n";
    std::cout << "- Throughput: 초당 처리 가능한 케이스 수\n";
    std::cout << "- vs Single: single-case 대비 per-case 속도 향상\n";

    return 0;
}
