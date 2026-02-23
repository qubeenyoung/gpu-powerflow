/**
 * @file benchmark_atomic.cpp
 * @brief atomicAdd 병목 정량화 벤치마크
 *
 * 세 가지 커널 변형 성능 비교:
 *   1. 기본 atomicAdd 버전
 *   2. atomicAdd 제거 버전 (정확도 무시, 순수 성능)
 *   3. Warp-level Reduction 버전
 *
 * 목적: atomicAdd가 얼마나 병목인지 정량화
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

int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "  atomicAdd 병목 정량화 벤치마크" << endl;
    cout << "========================================" << endl;

#ifndef USE_CUDA
    cerr << "Error: This benchmark requires CUDA support." << endl;
    cerr << "Please build with -DBUILD_CUDA=ON" << endl;
    return 1;
#else

    // 1. 케이스 선택
    string case_name = "RT_2024_0420_1932_53535MW_modified_korea_powergrid_zbr";
    if (argc > 1) {
        case_name = argv[1];
    }

    cout << "케이스: " << case_name << endl;

    // 2. 데이터 로드
    NRData case_data;
    case_data.load_data(case_name);

    int nbus = case_data.V0.size();
    cout << "버스 수: " << nbus << endl;
    cout << "  PV buses: " << case_data.pv.size() << endl;
    cout << "  PQ buses: " << case_data.pq.size() << endl;
    cout << "Ybus nnz: " << case_data.Ybus.nonZeros() << endl;

    // 3. Jacobian 초기화 (매핑 테이블 생성)
    cout << "\nAnalyzing Jacobian structure..." << endl;
    Jacobian jacobian;
    jacobian.analyze(case_data.Ybus, case_data.pv, case_data.pq);

    int J_nnz = jacobian.J.nonZeros();
    cout << "Jacobian nnz: " << J_nnz << endl;

    // 4. CUDA 가속기 초기화
    NewtonCudaAccel cuda_accel;
    cuda_accel.initialize(
        nbus, J_nnz, &case_data.Ybus,
        jacobian.mapJ11, jacobian.mapJ21, jacobian.mapJ12, jacobian.mapJ22,
        jacobian.diagMapJ11, jacobian.diagMapJ21, jacobian.diagMapJ12, jacobian.diagMapJ22
    );

    // 5. 전압 데이터 준비
    vector<double> V_real(nbus * 2);
    for (int i = 0; i < nbus; ++i) {
        V_real[i * 2] = case_data.V0[i].real();
        V_real[i * 2 + 1] = case_data.V0[i].imag();
    }

    // 6. Warm-up
    cout << "\nWarm-up 중..." << endl;
    for (int i = 0; i < 5; ++i) {
        cuda_accel.benchmark_kernel_variants(V_real.data(), 1, 0);
    }

    // 7. 벤치마크 설정
    const int num_runs = 20;
    vector<int> batch_sizes = {1, 4, 16, 64};
    const char* variant_names[] = {
        "atomicAdd (기본)",
        "No-Atomic (덮어쓰기)",
        "Warp Reduction"
    };

    cout << "\n========================================" << endl;
    cout << "벤치마크 결과 (" << num_runs << "회 평균)" << endl;
    cout << "========================================" << endl;

    // 8. 벤치마크 실행
    cout << "\n| Batch | " << setw(20) << variant_names[0]
         << " | " << setw(20) << variant_names[1]
         << " | " << setw(20) << variant_names[2]
         << " | Speedup (No-Atomic) |" << endl;
    cout << "|-------|" << string(22, '-') << "|" << string(22, '-')
         << "|" << string(22, '-') << "|---------------------|" << endl;

    for (int batch_size : batch_sizes) {
        double times[3] = {0.0, 0.0, 0.0};

        for (int variant = 0; variant < 3; ++variant) {
            double total_time = 0.0;

            for (int run = 0; run < num_runs; ++run) {
                float kernel_ms = cuda_accel.benchmark_kernel_variants(
                    V_real.data(), batch_size, variant
                );
                total_time += kernel_ms;
            }

            times[variant] = total_time / num_runs;
        }

        // Speedup 계산: atomicAdd 대비 No-Atomic
        double speedup = times[0] / times[1];

        cout << "| " << setw(5) << batch_size << " | "
             << setw(17) << fixed << setprecision(4) << times[0] << " ms | "
             << setw(17) << times[1] << " ms | "
             << setw(17) << times[2] << " ms | "
             << setw(17) << setprecision(2) << speedup << "x |" << endl;
    }

    // 9. 상세 분석 출력
    cout << "\n========================================" << endl;
    cout << "분석 결과" << endl;
    cout << "========================================" << endl;

    // Batch=64에서의 상세 분석
    int analysis_batch = 64;
    double t_atomic = 0.0, t_no_atomic = 0.0, t_warp = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        t_atomic += cuda_accel.benchmark_kernel_variants(V_real.data(), analysis_batch, 0);
        t_no_atomic += cuda_accel.benchmark_kernel_variants(V_real.data(), analysis_batch, 1);
        t_warp += cuda_accel.benchmark_kernel_variants(V_real.data(), analysis_batch, 2);
    }

    t_atomic /= num_runs;
    t_no_atomic /= num_runs;
    t_warp /= num_runs;

    double atomic_overhead = t_atomic - t_no_atomic;
    double atomic_ratio = (atomic_overhead / t_atomic) * 100.0;
    double warp_improvement = ((t_atomic - t_warp) / t_atomic) * 100.0;

    cout << "\nBatch=" << analysis_batch << " 상세 분석:" << endl;
    cout << "  - atomicAdd 버전:     " << fixed << setprecision(4) << t_atomic << " ms" << endl;
    cout << "  - No-Atomic 버전:     " << t_no_atomic << " ms" << endl;
    cout << "  - Warp Reduce 버전:   " << t_warp << " ms" << endl;
    cout << endl;
    cout << "  - atomicAdd 오버헤드: " << setprecision(4) << atomic_overhead << " ms ("
         << setprecision(1) << atomic_ratio << "% of total)" << endl;
    cout << "  - Warp Reduce 개선:   " << setprecision(1) << warp_improvement << "%" << endl;

    if (atomic_ratio > 50.0) {
        cout << "\n★ 결론: atomicAdd가 주요 병목 (오버헤드 " << atomic_ratio << "%)" << endl;
        cout << "  → Graph Coloring 또는 더 정교한 Warp Reduction 필요" << endl;
    } else if (atomic_ratio > 20.0) {
        cout << "\n★ 결론: atomicAdd가 상당한 병목 (오버헤드 " << atomic_ratio << "%)" << endl;
        cout << "  → 부분적 최적화로 개선 가능" << endl;
    } else {
        cout << "\n★ 결론: atomicAdd는 작은 병목 (오버헤드 " << atomic_ratio << "%)" << endl;
        cout << "  → 다른 병목(메모리 대역폭, 연산 등) 확인 필요" << endl;
    }

    cout << "\n========================================" << endl;
    cout << "벤치마크 완료" << endl;
    cout << "========================================" << endl;

#endif

    return 0;
}
