/**
 * @file cudss_solver.hpp
 * @brief cuDSS 기반 GPU 희소 선형 시스템 솔버
 *
 * Newton-Raphson 전력조류 계산을 위한 Jx=b 풀이
 * FP32 Mixed Precision으로 A10 GPU에서 64배 빠른 성능 달성
 *
 * 주요 기능:
 *   - Analyze: 희소 패턴 분석 (Symbolic Factorization) - 1회만 실행
 *   - Factorize: 수치적 LU 분해 - 매 iteration 실행
 *   - Solve: Jx=b 풀이 - 매 iteration 실행
 *   - Refactorization: 2회차부터 빠른 재분해 사용
 */

#pragma once

#include "nr_data.hpp"
#include <Eigen/Sparse>

#ifdef USE_CUDSS

#include <cuda_runtime.h>
#include "cudss.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class CuDSSSolver {
public:
    CuDSSSolver();
    ~CuDSSSolver();

    // =========================================================================
    // 핵심 API (FP32 Mixed Precision)
    // =========================================================================

    /**
     * @brief 희소 패턴 분석 (Symbolic Factorization)
     *
     * 시뮬레이션 시작 시 1회만 호출. Jacobian의 희소 구조를 분석하고
     * fill-in 패턴을 결정함. Eigen → CSR 변환 및 순서 매핑 테이블 생성.
     *
     * @param J Jacobian 행렬 (Eigen CSC 포맷)
     */
    void analyzePattern(const nr_data::JacobianType& J);

    /**
     * @brief Eigen 순서 → CSR 순서 변환 (GPU 커널)
     *
     * GPU에서 계산된 Jacobian 값(Eigen 저장 순서)을 cuDSS가 요구하는
     * CSR 순서로 재배열.
     *
     * @param d_eigen_values_f Jacobian 값 (GPU, Eigen 순서, FP32)
     */
    void applyPermutationFP32(float* d_eigen_values_f);

    /**
     * @brief 수치적 LU 분해 (FP32)
     *
     * 매 iteration마다 호출. 첫 호출은 FACTORIZATION, 이후는
     * REFACTORIZATION을 사용하여 속도 향상.
     */
    void factorizeDirectGPU_FP32();

    /**
     * @brief 선형 시스템 풀이: Jx = b (FP32 계산, FP64 결과)
     *
     * @param b 우변 벡터 (mismatch, FP64)
     * @return x 해 벡터 (dx, FP64)
     */
    nr_data::VectorXd solveFP32(const nr_data::VectorXd& b);

    // =========================================================================
    // 타이밍 및 유틸리티
    // =========================================================================

    float getLastFactorizeTime() const { return last_factorize_ms_; }
    float getLastSolveTime() const { return last_solve_ms_; }
    float getLastMemcpyTime() const { return last_memcpy_ms_; }
    int64_t getNNZ() const { return nnz_; }

private:
    // cuDSS 핸들 (FP32)
    cudssHandle_t handle_;
    cudssConfig_t config_;
    cudssData_t solver_data_f_;      // FP32 솔버 데이터
    cudssMatrix_t mat_A_f_;          // Jacobian 행렬 래퍼
    cudssMatrix_t mat_x_f_;          // 해 벡터 래퍼
    cudssMatrix_t mat_b_f_;          // 우변 벡터 래퍼

    // GPU 메모리 (공통 구조)
    int64_t* d_rowPtr_;              // CSR row pointer
    int64_t* d_colInd_;              // CSR column indices
    int64_t* d_perm_map_;            // Eigen → CSR 순서 매핑

    // GPU 메모리 (FP32)
    float* d_val_f_;                 // Jacobian 값 (CSR 순서)
    float* d_b_f_;                   // 우변 벡터
    float* d_x_f_;                   // 해 벡터

    // 행렬 차원
    int64_t rows_;
    int64_t cols_;
    int64_t nnz_;

    // 호스트 측 CSR 저장소 (Eigen → CSR 변환용)
    std::vector<int64_t> h_rowPtr_;
    std::vector<int64_t> h_colInd_;
    std::vector<double> h_val_;

    // Eigen (row, col) → CSR 인덱스 매핑
    std::vector<std::unordered_map<int, int64_t>> eigen_to_csr_;

    // 타이밍
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    float last_factorize_ms_ = 0.0f;
    float last_solve_ms_ = 0.0f;
    float last_memcpy_ms_ = 0.0f;

    // 상태 플래그
    bool analyzed_ = false;
    bool factorized_f_ = false;
    bool first_factorize_f_ = true;  // 첫 factorization 여부 (REFACTORIZATION 분기용)
    bool allocated_ = false;

    // 헬퍼 함수
    void checkCuda(cudaError_t err, const char* msg);
    void checkCudss(cudssStatus_t status, const char* msg);
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void convertEigenToCSR(const nr_data::JacobianType& J);
};

#endif // USE_CUDSS
