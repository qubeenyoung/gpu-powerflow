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
    // 단일 케이스 API (FP32 Mixed Precision)
    // =========================================================================

    void analyzePattern(const nr_data::JacobianType& J);
    void applyPermutationFP32(float* d_eigen_values_f);
    void factorizeDirectGPU_FP32();
    nr_data::VectorXd solveFP32(const nr_data::VectorXd& b);

    // =========================================================================
    // Uniform Batch API (cuDSS UBATCH, 같은 sparsity 패턴 N개 동시 처리)
    // =========================================================================

    /**
     * @brief Batch 희소 패턴 분석 (1회만)
     *
     * 단일 Jacobian 패턴으로 N개 케이스에 대한 cuDSS Uniform Batch 설정.
     * batch_size개의 J값 배열, b/x 벡터를 GPU에 할당.
     *
     * @param J         Jacobian 행렬 (희소 구조, 값은 무관)
     * @param batch_size 동시 처리할 케이스 수
     */
    void analyzePatternBatch(const nr_data::JacobianType& J, int batch_size);

    /**
     * @brief Batch Permutation: Eigen 순서 → CSR 순서 (N개 동시)
     *
     * @param d_eigen_vals_batch  GPU, [J_nnz * batch_size], FP32
     *                            batch_size개의 Jacobian 값이 연속 저장됨
     */
    void applyPermutationBatchFP32(float* d_eigen_vals_batch);

    /**
     * @brief Batch LU 분해 (N개 동시)
     */
    void factorizeBatchFP32();

    /**
     * @brief Batch Solve: N개 Jx=b 동시 풀이
     *
     * @param d_b_batch  GPU, [rows_ * batch_size], FP32  (b 벡터들)
     * @param d_x_batch  GPU, [cols_ * batch_size], FP32  (x 출력)
     */
    void solveBatchFP32(float* d_b_batch, float* d_x_batch);

    // =========================================================================
    // UBATCH API v2 (선임 방식: 단일 CSR + flat buffer + UBATCH_SIZE)
    // cudssMatrixCreateBatchCsr 대신 cudssMatrixCreateCsr 사용
    // =========================================================================

    /**
     * @brief UBATCH v2 초기화 (선임 방식)
     *
     * 단일 cudssMatrixCreateCsr에 flat [batch_size * nnz] 버퍼를 넘기고
     * UBATCH_SIZE 설정으로 cuDSS가 batch로 처리하게 함.
     */
    void analyzePatternUBatch(const nr_data::JacobianType& J, int batch_size);

    /**
     * @brief UBATCH v2 Permutation (SeqBatch와 동일 커널 재사용)
     */
    void applyPermutationUBatchFP32(float* d_eigen_vals_batch);

    /**
     * @brief UBATCH v2 Factorize + Solve (단일 cudssExecute 2번)
     *
     * @param d_b_batch  GPU, [rows_ * batch_size], FP32
     * @param d_x_batch  GPU, [cols_ * batch_size], FP32 (출력)
     */
    void factorizeAndSolveUBatchFP32(float* d_b_batch, float* d_x_batch);

    // =========================================================================
    // Sequential Batch API (cuDSS UBATCH 미지원 fallback)
    // 같은 symbolic 구조를 재사용하며 N개를 순차 처리
    // =========================================================================

    /**
     * @brief Sequential Batch 초기화
     *
     * analyzePattern()과 동일하나, batch_size개의 val/b/x 버퍼를 추가 할당.
     * 각 배치 항목은 동일한 symbolic 구조를 공유함.
     */
    void analyzePatternSeqBatch(const nr_data::JacobianType& J, int batch_size);

    /**
     * @brief Batch Permutation (Sequential용)
     * d_eigen_vals_batch → d_val_batch_ (nnz * batch_size)
     */
    void applyPermutationSeqBatchFP32(float* d_eigen_vals_batch);

    /**
     * @brief Sequential Batch Factorize+Solve
     *
     * 각 배치 항목에 대해 val을 d_val_f_에 복사 후 REFACTORIZATION → SOLVE.
     * @param d_b_batch  GPU, [rows_ * batch_size], FP32
     * @param d_x_batch  GPU, [cols_ * batch_size], FP32 (출력)
     */
    void factorizeAndSolveSeqBatchFP32(float* d_b_batch, float* d_x_batch);

    // =========================================================================
    // 타이밍 및 유틸리티
    // =========================================================================

    float getLastFactorizeTime() const { return last_factorize_ms_; }
    float getLastSolveTime() const { return last_solve_ms_; }
    float getLastMemcpyTime() const { return last_memcpy_ms_; }
    int64_t getNNZ() const { return nnz_; }
    int getBatchSize() const { return batch_size_; }
    int64_t getRows() const { return rows_; }

    /** Batch b 벡터 GPU 버퍼 포인터 (FP32, [rows_ * batch_size]) */
    float* getBatchBBuffer() { return d_b_batch_; }
    /** Batch x 벡터 GPU 버퍼 포인터 (FP32, [cols_ * batch_size]) */
    float* getBatchXBuffer() { return d_x_batch_; }

private:
    // cuDSS 핸들 (FP32)
    cudssHandle_t handle_;
    cudssConfig_t config_;
    cudssData_t solver_data_f_;      // 단일 케이스 솔버 데이터
    cudssMatrix_t mat_A_f_;          // 단일 Jacobian 행렬 래퍼
    cudssMatrix_t mat_x_f_;          // 단일 해 벡터 래퍼
    cudssMatrix_t mat_b_f_;          // 단일 우변 벡터 래퍼

    // Batch cuDSS 핸들 (구 방식: BatchCsr API)
    cudssData_t solver_data_batch_;  // Batch 솔버 데이터
    cudssMatrix_t mat_A_batch_;      // Batch Jacobian 행렬 (BatchCsr)
    cudssMatrix_t mat_x_batch_;      // Batch 해 벡터 (BatchDn)
    cudssMatrix_t mat_b_batch_;      // Batch 우변 벡터 (BatchDn)

    // UBATCH v2 핸들 (선임 방식: 단일 CSR + flat buffer)
    cudssData_t solver_data_ubatch_; // UBATCH v2 솔버 데이터
    cudssMatrix_t mat_A_ubatch_;     // 단일 CSR (flat batch_size*nnz values)
    cudssMatrix_t mat_x_ubatch_;     // 단일 Dn (flat batch_size*rows)
    cudssMatrix_t mat_b_ubatch_;     // 단일 Dn (flat batch_size*rows)
    bool analyzed_ubatch_ = false;
    bool factorized_ubatch_ = false;

    // GPU 메모리 (공통 구조)
    int64_t* d_rowPtr_;              // CSR row pointer
    int64_t* d_colInd_;              // CSR column indices
    int64_t* d_perm_map_;            // Eigen → CSR 순서 매핑 (단일)

    // GPU 메모리 (단일 FP32)
    float* d_val_f_;                 // Jacobian 값 (단일)
    float* d_b_f_;                   // 우변 벡터 (단일)
    float* d_x_f_;                   // 해 벡터 (단일)

    // GPU 메모리 (Batch FP32) — 연속 할당
    float* d_val_batch_;             // [nnz_ * batch_size] Jacobian 값들
    float* d_b_batch_;               // [rows_ * batch_size] RHS 벡터들
    float* d_x_batch_;               // [cols_ * batch_size] 해 벡터들
    int64_t* d_perm_map_batch_;      // [nnz_ * batch_size] 배치 permutation 맵

    // cuDSS Batch에 필요한 포인터 배열 (GPU에 올릴 포인터 배열)
    void** d_val_ptrs_;              // [batch_size] → 각 J값 배열 포인터
    void** d_b_ptrs_;                // [batch_size] → 각 b 벡터 포인터
    void** d_x_ptrs_;                // [batch_size] → 각 x 벡터 포인터

    // 행렬 차원
    int64_t rows_;
    int64_t cols_;
    int64_t nnz_;
    int batch_size_ = 0;

    // 호스트 측 CSR 저장소
    std::vector<int64_t> h_rowPtr_;
    std::vector<int64_t> h_colInd_;
    std::vector<double> h_val_;
    std::vector<int64_t> h_perm_map_;  // 저장해 놓기

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
    bool analyzed_batch_ = false;
    bool factorized_f_ = false;
    bool first_factorize_f_ = true;
    bool allocated_ = false;
    bool allocated_batch_ = false;

    // 헬퍼 함수
    void checkCuda(cudaError_t err, const char* msg);
    void checkCudss(cudssStatus_t status, const char* msg);
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void allocateBatchDeviceMemory(int batch_size);
    void freeBatchDeviceMemory();
    void convertEigenToCSR(const nr_data::JacobianType& J);
};

#endif // USE_CUDSS
