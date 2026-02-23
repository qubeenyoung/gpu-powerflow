/**
 * @file newtonpf.cpp
 * @brief Newton-Raphson 전력조류 계산 (Power Flow) 구현
 *
 * GPU 가속 파이프라인:
 *   1. Jacobian 업데이트: CUDA 커널 (FP32)
 *   2. Permutation: Eigen 순서 → CSR 순서 (GPU 커널)
 *   3. LU 분해: cuDSS REFACTORIZATION (FP32)
 *   4. Solve: cuDSS (FP32 계산, FP64 결과)
 *
 * A10 GPU 성능: FP32 = 31.2 TFLOPS, FP64 = 0.49 TFLOPS (64배 차이)
 */

#include "newtonpf.hpp"
#include "timer.hpp"
#include "spdlog/spdlog.h"

#include <Eigen/KLUSupport>  // CPU용 희소 행렬 solver (KLU)

#ifdef USE_CUDA
#include "cuda_accel.hpp"    // GPU Jacobian 계산
#endif

#ifdef USE_CUDSS
#include "cudss_solver.hpp"  // GPU LU 분해 및 풀이
#endif

nr_data::NRResult newtonPF(
    const nr_data::YbusType& ybus,
    const nr_data::VectorXcd& sbus,
    const nr_data::VectorXcd& V0,
    const nr_data::VectorXi32& pv,
    const nr_data::VectorXi32& pq,
    const double tolerance,
    const int32_t max_iter
) {
    // =========================================================================
    // 초기화
    // =========================================================================
    const int32_t npv = pv.size();
    const int32_t npq = pq.size();
    const int32_t nb = V0.size();
    const int32_t dimF = npv + 2 * npq;  // 미지수 개수 (위상각 + 전압크기)

    // 전압 벡터 (복소수, 위상, 크기)
    nr_data::VectorXcd V = V0;
    nr_data::VectorXd Va = V.unaryExpr([](const std::complex<double>& z) { return std::arg(z); });
    nr_data::VectorXd Vm = V.cwiseAbs();

    // 작업 벡터
    nr_data::VectorXd F(dimF);     // mismatch 벡터
    nr_data::VectorXd dx(dimF);    // 보정값
    nr_data::VectorXcd Ibus(nb);   // 전류 주입

    double normF = 0.0;
    bool converged = false;
    int iter = 0;

    // =========================================================================
    // Step 0: Jacobian 패턴 분석 (CPU, 1회만 실행)
    // =========================================================================
    Jacobian jacobian;
    {
        BlockTimer timer("AnalyzeJacobian");
        jacobian.analyze(ybus, pv, pq);
    }

    // =========================================================================
    // GPU 초기화 (USE_CUDA && USE_CUDSS)
    // =========================================================================
#if defined(USE_CUDA) && defined(USE_CUDSS)
    // CUDA Jacobian 가속기 초기화
    NewtonCudaAccel cuda_accel;
    {
        int J_nnz = jacobian.J.nonZeros();
        spdlog::info("CUDA 가속기 초기화 (버스: {}, Y_nnz: {}, J_nnz: {})", nb, ybus.nonZeros(), J_nnz);

        BlockTimer timer("CUDA_Initialize");
        cuda_accel.initialize(
            nb, J_nnz, &ybus,
            jacobian.mapJ11, jacobian.mapJ21, jacobian.mapJ12, jacobian.mapJ22,
            jacobian.diagMapJ11, jacobian.diagMapJ21, jacobian.diagMapJ12, jacobian.diagMapJ22
        );
    }

    // cuDSS 솔버 초기화
    CuDSSSolver cudss_solver;
    {
        spdlog::info("cuDSS 솔버 초기화 (행렬 크기: {})", dimF);

        BlockTimer timer("cuDSS_AnalyzePattern");
        cudss_solver.analyzePattern(jacobian.J);
    }

    spdlog::info("GPU 파이프라인 준비 완료 (FP32 Mixed Precision)");

#else
    // CPU 전용 솔버
    Eigen::KLU<nr_data::JacobianType> solver;
    {
        spdlog::info("CPU KLU 솔버 초기화 (행렬 크기: {})", dimF);

        BlockTimer timer("AnalyzeSolver");
        solver.analyzePattern(jacobian.J);
    }
#endif

    // =========================================================================
    // Newton-Raphson Iteration
    // =========================================================================
    while (!converged && iter < max_iter) {
        // -----------------------------------------------------------------
        // Step 1: Mismatch 계산
        // -----------------------------------------------------------------
        {
            BlockTimer timer("Mismatch_" + std::to_string(iter));
            Ibus = ybus * V;
            mismatch(normF, F, V, Ibus, sbus, pv, pq);
        }

        // 수렴 체크
        if (normF < tolerance) {
            converged = true;
            continue;
        }

        ++iter;
        spdlog::info("[ITER {}] normF = {:.6e}", iter, normF);

        // -----------------------------------------------------------------
        // Step 2: Jacobian 업데이트 & LU 분해
        // -----------------------------------------------------------------
#if defined(USE_CUDA) && defined(USE_CUDSS)
        // GPU 파이프라인 (FP32)
        {
            // (1) Jacobian 업데이트 (GPU, FP32)
            {
                BlockTimer timer("CUDA_UpdateJacobian_FP32_" + std::to_string(iter));
                cuda_accel.update_jacobian_to_buffer_fp32(V.data());
            }

            // (2) Permutation: Eigen 순서 → CSR 순서 (GPU)
            {
                BlockTimer timer("GPU_Permutation_FP32_" + std::to_string(iter));
                float* d_eigen_J_f = cuda_accel.getDeviceJacobianBufferFP32();
                cudss_solver.applyPermutationFP32(d_eigen_J_f);
            }

            // (3) LU 분해 (GPU, FP32, REFACTORIZATION)
            {
                BlockTimer timer("cuDSS_Factorize_FP32_" + std::to_string(iter));
                cudss_solver.factorizeDirectGPU_FP32();
            }
        }
#else
        // CPU 경로
        {
            // (1) Jacobian 업데이트 (CPU)
            {
                BlockTimer timer("CPU_UpdateJacobian_" + std::to_string(iter));
                jacobian.update(ybus, V, Ibus);
            }

            // (2) LU 분해 (CPU)
            {
                BlockTimer timer("CPU_Factorize_" + std::to_string(iter));
                solver.factorize(jacobian.J);
            }
        }
#endif

        // -----------------------------------------------------------------
        // Step 3: Solve (Jx = -F)
        // -----------------------------------------------------------------
#if defined(USE_CUDA) && defined(USE_CUDSS)
        {
            BlockTimer timer("cuDSS_Solve_FP32_" + std::to_string(iter));
            dx = cudss_solver.solveFP32(-F);
        }
#else
        {
            BlockTimer timer("CPU_Solve_" + std::to_string(iter));
            dx = solver.solve(-F);
        }
#endif

        // -----------------------------------------------------------------
        // Step 4: 전압 업데이트 (V += dx)
        // -----------------------------------------------------------------
        {
            BlockTimer timer("UpdateV_" + std::to_string(iter));

            int k = 0;
            // PV 버스: 위상각만 업데이트
            for (int32_t i = 0; i < npv; ++i) {
                Va(pv(i)) += dx(k++);
            }
            // PQ 버스: 위상각 업데이트
            for (int32_t i = 0; i < npq; ++i) {
                Va(pq(i)) += dx(k++);
            }
            // PQ 버스: 전압크기 업데이트
            for (int32_t i = 0; i < npq; ++i) {
                Vm(pq(i)) += dx(k++);
            }

            // 복소수 전압 재구성: V = Vm * e^(j*Va)
            V.real() = (Vm.array() * Va.array().cos()).matrix();
            V.imag() = (Vm.array() * Va.array().sin()).matrix();
        }
    }

    // =========================================================================
    // 결과 반환
    // =========================================================================
    nr_data::NRResult result;
    result.V = V;
    result.converged = converged;
    result.iter = iter;
    result.normF = normF;

    return result;
}
