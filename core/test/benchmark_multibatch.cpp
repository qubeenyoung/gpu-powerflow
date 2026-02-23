/**
 * @file benchmark_multibatch.cpp
 * @brief Multi-batch Jacobian 커널 성능 측정
 *
 * 동일한 데이터로 batch size를 1, 2, 4, 8, 16, 32, 64로 늘려가며
 * GPU throughput 향상 효과를 측정합니다.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "nr_data.hpp"
#include "newtonpf.hpp"

#ifdef USE_CUDA
#include "cuda_accel.hpp"
#endif

using namespace nr_data;

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Multi-Batch Jacobian Performance Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

#ifndef USE_CUDA
    std::cerr << "Error: This benchmark requires CUDA support." << std::endl;
    std::cerr << "Please build with -DBUILD_CUDA=ON" << std::endl;
    return 1;
#else

    // 케이스 선택
    std::string case_name = "RT_2024_0420_1932_53535MW_modified_korea_powergrid_zbr";
    if (argc > 1) {
        case_name = argv[1];
    }

    std::cout << "Testing case: " << case_name << std::endl;

    // 데이터 로드
    NRData case_data;
    case_data.load_data(case_name);

    int nbus = case_data.V0.size();
    std::cout << "Loaded: " << nbus << " buses" << std::endl;
    std::cout << "  PV buses: " << case_data.pv.size() << std::endl;
    std::cout << "  PQ buses: " << case_data.pq.size() << std::endl;
    std::cout << "  Ybus nnz: " << case_data.Ybus.nonZeros() << std::endl;

    // Jacobian 구조 분석
    std::cout << "\nAnalyzing Jacobian structure..." << std::endl;
    Jacobian jacobian;
    jacobian.analyze(case_data.Ybus, case_data.pv, case_data.pq);

    int J_nnz = jacobian.J.nonZeros();
    std::cout << "  Jacobian nnz: " << J_nnz << std::endl;

    // CUDA 가속기 초기화
    std::cout << "\nInitializing CUDA accelerator..." << std::endl;
    NewtonCudaAccel cuda_accel;
    cuda_accel.initialize(
        nbus, J_nnz, &case_data.Ybus,
        jacobian.mapJ11, jacobian.mapJ21, jacobian.mapJ12, jacobian.mapJ22,
        jacobian.diagMapJ11, jacobian.diagMapJ21, jacobian.diagMapJ12, jacobian.diagMapJ22
    );

    // V 벡터를 double 배열로 변환 (복소수를 실수 2개로)
    std::vector<double> V_real(nbus * 2);
    for (int i = 0; i < nbus; ++i) {
        V_real[i * 2] = case_data.V0[i].real();
        V_real[i * 2 + 1] = case_data.V0[i].imag();
    }

    // 배치 크기별 성능 측정
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    // 최대 batch_size로 메모리 미리 할당 (성능 측정에서 할당 오버헤드 제거)
    std::cout << "\n[메모리 사전 할당] 최대 batch_size=512로 메모리 할당..." << std::endl;
    cuda_accel.update_jacobian_batch(V_real.data(), 512);

    // Warm-up
    std::cout << "\n[Warm-up] Running batch=1 three times..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        cuda_accel.update_jacobian_batch(V_real.data(), 1);
    }

    // 정확도 검증 (batch=4로 테스트)
    std::cout << "\n[정확도 검증] batch=4로 테스트..." << std::endl;
    cuda_accel.update_jacobian_batch(V_real.data(), 4);
    bool verification_passed = cuda_accel.verify_batch_correctness(4);
    if (!verification_passed) {
        std::cerr << "\n오류: 정확도 검증 실패! 계산 결과가 정확하지 않습니다." << std::endl;
        return 1;
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Multi-Batch Performance Results" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << std::setw(8) << "Batch"
         << std::setw(18) << "Total Time (ms)"
         << std::setw(22) << "Time per Batch (ms)"
         << std::setw(18) << "Speedup vs 1"
         << std::setw(20) << "GPU Efficiency (%)"
         << std::endl;
    std::cout << std::string(86, '-') << std::endl;

    double base_time = 0.0;
    double batch1_time = 0.0;

    for (int batch_size : batch_sizes) {
        // 10회 반복 측정
        const int num_runs = 10;
        double total_ms = 0.0;

        for (int run = 0; run < num_runs; ++run) {
            float kernel_time = cuda_accel.update_jacobian_batch(V_real.data(), batch_size);
            total_ms += kernel_time;
        }

        double avg_ms = total_ms / num_runs;

        if (batch_size == 1) {
            base_time = avg_ms;
            batch1_time = avg_ms;
        }

        // 통계 계산
        double time_per_batch = avg_ms / batch_size;           // 1개 처리 체감 시간
        double speedup = (batch1_time * batch_size) / avg_ms;  // 이상적: batch_size배
        double efficiency = (speedup / batch_size) * 100.0;     // GPU 효율 (%)

        std::cout << std::setw(8) << batch_size
             << std::setw(18) << std::fixed << std::setprecision(4) << avg_ms
             << std::setw(22) << std::setprecision(4) << time_per_batch
             << std::setw(18) << std::setprecision(2) << speedup << "x"
             << std::setw(19) << std::setprecision(1) << efficiency
             << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  해석" << std::endl;
    std::cout << "========================================\n" << std::endl;
    std::cout << "- Total Time: batch_size개를 한꺼번에 처리하는 데 걸린 시간" << std::endl;
    std::cout << "- Time per Batch: 1개당 처리 시간 (throughput 관점)" << std::endl;
    std::cout << "- Speedup: Batch=1 대비 가속비 (이상적으로는 batch_size배)" << std::endl;
    std::cout << "- GPU Efficiency: GPU 활용 효율 (100%에 가까울수록 좋음)" << std::endl;
    std::cout << "\n분석:" << std::endl;
    std::cout << "- Speedup이 batch_size에 비례하면 GPU 활용이 완벽 (효율 100%)" << std::endl;
    std::cout << "- Speedup이 포화되면 GPU가 최대 성능에 도달한 것" << std::endl;
    std::cout << "- Time per Batch가 감소하면 throughput이 증가하는 것" << std::endl;

    return 0;
#endif
}
