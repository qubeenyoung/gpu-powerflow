/**
 * @file profile_no_atomic.cpp
 * @brief atomicAdd 제거 커널 Nsight Compute 프로파일링용
 *
 * 사용법:
 *   ./profile_no_atomic 0  # atomicAdd 커널
 *   ./profile_no_atomic 1  # no-atomic 커널
 *   ./profile_no_atomic 2  # warp-reduce 커널
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
    cout << "  Kernel Variant Profiling" << endl;
    cout << "========================================" << endl;

#ifndef USE_CUDA
    cerr << "Error: This test requires CUDA support." << endl;
    return 1;
#else

    // variant 선택: 0=atomicAdd, 1=no-atomic, 2=warp-reduce
    int variant = 0;  // 기본값: atomicAdd
    int batch_size = 512;  // 기본값: Multi-batch

    if (argc > 1) {
        variant = atoi(argv[1]);
    }
    if (argc > 2) {
        batch_size = atoi(argv[2]);
    }

    cout << "Kernel variant: " << variant;
    if (variant == 0) cout << " (atomicAdd)";
    else if (variant == 1) cout << " (no-atomic)";
    else if (variant == 2) cout << " (warp-reduce)";
    cout << endl;
    cout << "Batch size: " << batch_size << endl;

    string case_name = "RT_2024_0420_1932_53535MW_modified_korea_powergrid_zbr";

    // 데이터 로드 - benchmark_multibatch와 동일한 방식
    NRData case_data;
    case_data.load_data(case_name);

    int nbus = case_data.V0.size();
    cout << "버스 수: " << nbus << endl;
    cout << "Ybus nnz: " << case_data.Ybus.nonZeros() << endl;

    // Jacobian 구조 분석
    cout << "\nAnalyzing Jacobian structure..." << endl;
    Jacobian jacobian;
    jacobian.analyze(case_data.Ybus, case_data.pv, case_data.pq);

    int J_nnz = jacobian.J.nonZeros();
    cout << "Jacobian nnz: " << J_nnz << endl;

    // CUDA 가속기 초기화
    cout << "\nInitializing CUDA accelerator..." << endl;
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
    cout << "\n[Warm-up] Running 3 times with batch_size=" << batch_size << "..." << endl;
    for (int i = 0; i < 3; ++i) {
        cuda_accel.benchmark_kernel_variants(V_real.data(), batch_size, variant);
    }

    // 프로파일링 대상 실행 (Nsight Compute가 캡처)
    cout << "\n프로파일링 실행 중..." << endl;

    const int num_runs = 5;
    float total_ms = 0.0f;

    for (int i = 0; i < num_runs; ++i) {
        float ms = cuda_accel.benchmark_kernel_variants(V_real.data(), batch_size, variant);
        total_ms += ms;
    }

    cout << fixed << setprecision(4);
    cout << "평균 커널 시간: " << (total_ms / num_runs) << " ms" << endl;

#endif
    return 0;
}
