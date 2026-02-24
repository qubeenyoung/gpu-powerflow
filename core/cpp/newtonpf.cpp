/**
 * @file newtonpf.cpp
 * @brief Newton-Raphson 전력조류 계산 (Power Flow) 구현
 *
 * GPU 가속 파이프라인 (V GPU 상주):
 *   0. 초기화: V0, Sbus GPU 업로드 (1회)
 *   1. Mismatch: cuSPARSE SpMV + CUDA 커널 (FP64, F 다운로드)
 *   2. Jacobian 업데이트: CUDA 커널 (FP32, V는 GPU에서 FP32로 이미 존재)
 *   3. Permutation: Eigen 순서 → CSR 순서 (GPU 커널)
 *   4. LU 분해: cuDSS REFACTORIZATION (FP32)
 *   5. Solve: cuDSS (FP64 결과, dx 다운로드)
 *   6. UpdateV: GPU에서 FP64 Va/Vm 업데이트 + V_cd/V_f 재구성
 *
 * CPU↔GPU 전송: F 다운로드 + dx 업로드 (매 iteration, O(dimF))
 * V 전송 제거: V는 GPU에 상주, CPU로 내보내지 않음 (결과 수집 시 1회)
 *
 * A10 GPU 성능: FP32 = 31.2 TFLOPS, FP64 = 0.49 TFLOPS (64배 차이)
 */

#include "newtonpf.hpp"
#include "timer.hpp"
#include "spdlog/spdlog.h"

#include <algorithm>
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

    // 전압 벡터 (CPU, 결과 수집용)
    nr_data::VectorXcd V = V0;

    // 작업 벡터
    nr_data::VectorXd F(dimF);     // mismatch 벡터
    nr_data::VectorXd dx(dimF);    // 보정값
    nr_data::VectorXcd Ibus(nb);   // 전류 주입 (CPU 경로용)

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

    spdlog::info("GPU 파이프라인 준비 완료 (FP32 Mixed Precision, V GPU 상주)");

    // Sbus, V0를 GPU에 업로드 (1회)
    cuda_accel.upload_sbus(sbus.data());
    cuda_accel.upload_V_initial(V0.data());

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
#if defined(USE_CUDA) && defined(USE_CUDSS)
            // GPU: cuSPARSE SpMV + Mismatch 커널 (V는 GPU 상주, 업로드 불필요)
            cuda_accel.compute_mismatch_gpu(
                nullptr,  // V_ptr=nullptr: d_V_cd_ 이미 최신
                pv.data(), pq.data(),
                npv, npq,
                F.data(), normF
            );
#else
            Ibus = ybus * V;
            mismatch(normF, F, V, Ibus, sbus, pv, pq);
#endif
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
        // GPU 파이프라인 (FP32, V는 GPU 상주)
        {
            // (1) Jacobian 업데이트 (GPU, FP32, V 업로드 없음)
            {
                BlockTimer timer("CUDA_UpdateJacobian_FP32_" + std::to_string(iter));
                cuda_accel.update_jacobian_to_buffer_fp32_no_upload();
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
        // Step 4: 전압 업데이트 (GPU에서 FP64 Va/Vm 업데이트 + V 재구성)
        // -----------------------------------------------------------------
        {
            BlockTimer timer("UpdateV_" + std::to_string(iter));
#if defined(USE_CUDA) && defined(USE_CUDSS)
            // GPU에서 Va/Vm FP64 업데이트, V_cd + V_f 재구성 (V 다운로드 없음)
            cuda_accel.update_voltage_gpu(
                dx.data(),
                pv.data(), pq.data(),
                npv, npq
            );
#else
            {
                nr_data::VectorXd Va = V.unaryExpr([](const std::complex<double>& z) { return std::arg(z); });
                nr_data::VectorXd Vm = V.cwiseAbs();

                int k = 0;
                for (int32_t i = 0; i < npv; ++i) Va(pv(i)) += dx(k++);
                for (int32_t i = 0; i < npq; ++i) Va(pq(i)) += dx(k++);
                for (int32_t i = 0; i < npq; ++i) Vm(pq(i)) += dx(k++);

                V.real() = (Vm.array() * Va.array().cos()).matrix();
                V.imag() = (Vm.array() * Va.array().sin()).matrix();
            }
#endif
        }
    }

    // =========================================================================
    // 결과 반환
    // =========================================================================
#if defined(USE_CUDA) && defined(USE_CUDSS)
    // GPU에서 최종 V 다운로드 (1회)
    cuda_accel.download_V(V.data());
#endif

    nr_data::NRResult result;
    result.V = V;
    result.converged = converged;
    result.iter = iter;
    result.normF = normF;

    return result;
}


// =============================================================================
// newtonPF_batch: Batch Newton-Raphson (N개 케이스 GPU 동시 처리)
// =============================================================================


std::vector<nr_data::NRResult>
newtonPF_batch(
    const nr_data::YbusType& ybus,
    const std::vector<nr_data::VectorXcd>& sbus_vec,
    const std::vector<nr_data::VectorXcd>& V0_vec,
    const nr_data::VectorXi32& pv,
    const nr_data::VectorXi32& pq,
    double tolerance,
    int max_iter
) {
    const int batch_size = static_cast<int>(sbus_vec.size());
    const int32_t npv  = pv.size();
    const int32_t npq  = pq.size();
    const int32_t nb   = V0_vec[0].size();
    const int32_t dimF = npv + 2 * npq;

    // 결과 초기화
    std::vector<nr_data::NRResult> results(batch_size);

    // =========================================================================
    // Step 0: Jacobian 패턴 분석 (1회)
    // =========================================================================
    Jacobian jacobian;
    {
        BlockTimer timer("Batch_AnalyzeJacobian");
        jacobian.analyze(ybus, pv, pq);
    }

#if defined(USE_CUDA) && defined(USE_CUDSS)
    // =========================================================================
    // GPU 초기화 (batch)
    // =========================================================================
    NewtonCudaAccel cuda_accel;
    {
        int J_nnz = jacobian.J.nonZeros();
        spdlog::info("CUDA Batch 가속기 초기화 (버스: {}, Y_nnz: {}, J_nnz: {}, batch: {})",
                     nb, ybus.nonZeros(), J_nnz, batch_size);
        BlockTimer timer("Batch_CUDA_Initialize");
        cuda_accel.initialize(
            nb, J_nnz, &ybus,
            jacobian.mapJ11, jacobian.mapJ21, jacobian.mapJ12, jacobian.mapJ22,
            jacobian.diagMapJ11, jacobian.diagMapJ21, jacobian.diagMapJ12, jacobian.diagMapJ22
        );
    }

    CuDSSSolver cudss_solver;
    {
        spdlog::info("cuDSS UBatch 솔버 초기화 (행렬: {}, batch: {})", dimF, batch_size);
        BlockTimer timer("Batch_cuDSS_AnalyzePattern");
        // 선임 방식: 단일 CSR + flat buffer + UBATCH_SIZE 설정
        cudss_solver.analyzePatternUBatch(jacobian.J, batch_size);
    }

    // N개 V0/Sbus → GPU 업로드
    {
        BlockTimer timer("Batch_UploadInitial");
        std::vector<const void*> V_ptrs(batch_size), Sbus_ptrs(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            V_ptrs[b]    = V0_vec[b].data();
            Sbus_ptrs[b] = sbus_vec[b].data();
        }
        cuda_accel.upload_batch_initial(V_ptrs.data(), Sbus_ptrs.data(), batch_size);
    }

    spdlog::info("GPU Batch 파이프라인 준비 완료");

    // =========================================================================
    // Batch Newton-Raphson Iteration
    // =========================================================================

    // 배치별 수렴 상태
    std::vector<bool>   converged(batch_size, false);
    std::vector<int>    iters(batch_size, 0);
    std::vector<double> normFs(batch_size, 0.0);
    bool all_converged = false;

    // F는 GPU 상주 (d_F_batch_), CPU 버퍼 불필요

    // 미수렴 배치 마스크 (전체 배치를 동시에 처리하되 수렴한 것은 무시)
    // (간단화: 모든 케이스가 수렴할 때까지 반복)
    int iter = 0;

    while (!all_converged && iter < max_iter) {
        // ----------------------------------------------------------------
        // Step 1: Batch Mismatch (SpMM + GPU normF reduction, F GPU 상주)
        // ----------------------------------------------------------------
        {
            BlockTimer timer("Batch_Mismatch_" + std::to_string(iter));
            cuda_accel.compute_mismatch_batch(
                pv.data(), pq.data(),
                npv, npq,
                nullptr,        // F_batch_out=nullptr: F는 GPU 상주
                normFs.data()   // normF만 다운로드
            );
        }

        // 수렴 체크
        all_converged = true;
        for (int b = 0; b < batch_size; ++b) {
            if (!converged[b]) {
                if (normFs[b] < tolerance) {
                    converged[b] = true;
                    iters[b] = iter;
                } else {
                    all_converged = false;
                }
            }
        }
        if (all_converged) break;

        ++iter;
        if (iter <= 3 || iter % 5 == 0) {
            double max_norm = *std::max_element(normFs.begin(), normFs.end());
            spdlog::info("[Batch ITER {}] max normF = {:.6e}", iter, max_norm);
        }

        // ----------------------------------------------------------------
        // Step 2: Batch Jacobian (FP32, V GPU 상주)
        // ----------------------------------------------------------------
        float* d_J_batch = nullptr;
        {
            BlockTimer timer("Batch_Jacobian_" + std::to_string(iter));
            d_J_batch = cuda_accel.update_jacobian_batch_no_upload();
        }

        // ----------------------------------------------------------------
        // Step 3: Batch Permutation (Eigen → CSR 순서, N개 병렬)
        // ----------------------------------------------------------------
        {
            BlockTimer timer("Batch_Permutation_" + std::to_string(iter));
            cudss_solver.applyPermutationUBatchFP32(d_J_batch);
        }

        // ----------------------------------------------------------------
        // Step 4+5: UBatch Factorize + Solve (Jx = -F)
        // 선임 방식: 단일 cudssExecute 2번으로 batch_size개 동시 처리
        // ----------------------------------------------------------------
        {
            BlockTimer timer("Batch_FactorizeSolve_" + std::to_string(iter));

            // F(FP64, GPU) → -F(FP32, GPU): CPU 경유 없음
            float* d_b_batch = cudss_solver.getBatchBBuffer();
            cuda_accel.negate_F_to_fp32(d_b_batch, dimF);

            float* d_x_batch = cudss_solver.getBatchXBuffer();
            cudss_solver.factorizeAndSolveUBatchFP32(d_b_batch, d_x_batch);
            cudaDeviceSynchronize();  // cuDSS 비동기 작업 완료 보장 (타이머 경계)
        }

        // ----------------------------------------------------------------
        // Step 6: Batch UpdateV (FP64 Va/Vm 업데이트)
        // cuDSS는 비동기이므로 타이머 전 sync로 FactorizeSolve와 경계 확정
        // d_x_f (device FP32) 직접 사용 — D→H→D round trip 없음
        // ----------------------------------------------------------------
        {
            cudaDeviceSynchronize();  // FactorizeSolve GPU 작업 완료 보장
            BlockTimer timer("Batch_UpdateV_" + std::to_string(iter));

            float* d_x_batch = cudss_solver.getBatchXBuffer();
            cuda_accel.update_voltage_batch_from_fp32_device(
                d_x_batch,
                pv.data(), pq.data(),
                npv, npq
            );
        }
    }

    // 미수렴 케이스에 대해 iter 기록
    for (int b = 0; b < batch_size; ++b) {
        if (!converged[b]) iters[b] = iter;
    }

    // =========================================================================
    // 결과: N개 V 다운로드
    // =========================================================================
    std::vector<std::complex<double>> V_batch_out(nb * batch_size);
    cuda_accel.download_V_batch(V_batch_out.data());

    for (int b = 0; b < batch_size; ++b) {
        nr_data::VectorXcd V(nb);
        for (int i = 0; i < nb; ++i) {
            V(i) = V_batch_out[b * nb + i];
        }
        results[b].V         = V;
        results[b].converged = converged[b];
        results[b].iter      = iters[b];
        results[b].normF     = normFs[b];
    }

    spdlog::info("Batch NR 완료: {}/{} 케이스 수렴, {} iterations",
                 std::count(converged.begin(), converged.end(), true),
                 batch_size, iter);

    return results;

#else
    // GPU 없을 때: 순차적 단일 케이스 실행
    spdlog::warn("USE_CUDA/USE_CUDSS 미설정 — 배치를 순차적으로 실행합니다");
    for (int b = 0; b < batch_size; ++b) {
        results[b] = newtonPF(ybus, sbus_vec[b], V0_vec[b], pv, pq, tolerance, max_iter);
    }
    return results;
#endif
}
