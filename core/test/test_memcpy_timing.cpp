/**
 * @file test_memcpy_timing.cpp
 * @brief memcpy vs 커널 시간 분리 측정
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

using namespace std;
using namespace nr_data;

int main() {
    cout << "========================================" << endl;
    cout << "  memcpy vs Kernel 시간 분석" << endl;
    cout << "========================================" << endl;

#ifndef USE_CUDA
    cerr << "Error: This test requires CUDA support." << endl;
    return 1;
#else

    string case_name = "RT_2024_0420_1932_53535MW_modified_korea_powergrid_zbr";

    // 데이터 로드
    NRData case_data;
    case_data.load_data(case_name);

    int nbus = case_data.V0.size();
    cout << "버스 수: " << nbus << endl;
    cout << "Ybus nnz: " << case_data.Ybus.nonZeros() << endl;

    // Jacobian 분석
    Jacobian jacobian;
    jacobian.analyze(case_data.Ybus, case_data.pv, case_data.pq);
    int J_nnz = jacobian.J.nonZeros();
    cout << "Jacobian nnz: " << J_nnz << endl;

    // CUDA 가속기 초기화
    NewtonCudaAccel cuda_accel;
    cuda_accel.initialize(
        nbus, J_nnz, &case_data.Ybus,
        jacobian.mapJ11, jacobian.mapJ21, jacobian.mapJ12, jacobian.mapJ22,
        jacobian.diagMapJ11, jacobian.diagMapJ21, jacobian.diagMapJ12, jacobian.diagMapJ22
    );

    // 전압 데이터 준비
    vector<double> V_real(nbus * 2);
    for (int i = 0; i < nbus; ++i) {
        V_real[i * 2] = case_data.V0[i].real();
        V_real[i * 2 + 1] = case_data.V0[i].imag();
    }

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        cuda_accel.update_jacobian_to_buffer_fp32(V_real.data());
    }

    // 측정
    const int num_runs = 100;
    double total_memcpy_ms = 0.0;

    cout << "\n측정 중... (" << num_runs << "회)" << endl;

    for (int i = 0; i < num_runs; ++i) {
        cuda_accel.update_jacobian_to_buffer_fp32(V_real.data());
        total_memcpy_ms += cuda_accel.getLastMemcpyTime();
    }

    double avg_memcpy_ms = total_memcpy_ms / num_runs;

    // 커널만 측정 (benchmark_kernel_variants 사용)
    double total_kernel_ms = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        total_kernel_ms += cuda_accel.benchmark_kernel_variants(V_real.data(), 1, 0);
    }
    double avg_kernel_ms = total_kernel_ms / num_runs;

    cout << "\n========================================" << endl;
    cout << "  Single Batch 시간 분석 결과" << endl;
    cout << "========================================" << endl;
    cout << fixed << setprecision(4);
    cout << "memcpy (Host→Device): " << avg_memcpy_ms << " ms" << endl;
    cout << "Kernel 실행:          " << avg_kernel_ms << " ms" << endl;
    cout << "----------------------------------------" << endl;

    double total_time = avg_memcpy_ms + avg_kernel_ms;
    double memcpy_ratio = (avg_memcpy_ms / total_time) * 100.0;
    double kernel_ratio = (avg_kernel_ms / total_time) * 100.0;

    cout << "합계:                 " << total_time << " ms" << endl;
    cout << endl;
    cout << "비율 분석:" << endl;
    cout << "  memcpy 비율: " << setprecision(1) << memcpy_ratio << "%" << endl;
    cout << "  Kernel 비율: " << kernel_ratio << "%" << endl;

    // 데이터 크기 분석
    cout << "\n========================================" << endl;
    cout << "  데이터 전송량 분석" << endl;
    cout << "========================================" << endl;

    size_t v_bytes = nbus * 2 * sizeof(float);  // V 벡터 (FP32)
    size_t j_bytes = J_nnz * sizeof(float);     // J 값 (FP32)

    cout << "V 벡터 크기:     " << v_bytes / 1024.0 << " KB" << endl;
    cout << "J 값 크기:       " << j_bytes / 1024.0 << " KB" << endl;
    cout << "합계:            " << (v_bytes + j_bytes) / 1024.0 << " KB" << endl;

    // 대역폭 계산
    double bandwidth_gbps = (v_bytes / 1e9) / (avg_memcpy_ms / 1000.0);
    cout << "\n추정 대역폭:     " << setprecision(2) << bandwidth_gbps << " GB/s" << endl;
    cout << "(PCIe 3.0 x16 이론: ~16 GB/s, PCIe 4.0 x16: ~32 GB/s)" << endl;

    cout << "\n========================================" << endl;
    cout << "  결론" << endl;
    cout << "========================================" << endl;

    if (memcpy_ratio > 50.0) {
        cout << "★ memcpy가 주요 병목! (" << setprecision(1) << memcpy_ratio << "%)" << endl;
        cout << "  → 배치 처리로 memcpy 오버헤드 분산 필요" << endl;
    } else if (memcpy_ratio > 30.0) {
        cout << "★ memcpy가 상당한 비중 (" << memcpy_ratio << "%)" << endl;
        cout << "  → 배치 처리로 개선 가능" << endl;
    } else {
        cout << "★ Kernel이 주요 병목 (" << kernel_ratio << "%)" << endl;
        cout << "  → Kernel 최적화가 더 효과적" << endl;
    }

#endif
    return 0;
}
