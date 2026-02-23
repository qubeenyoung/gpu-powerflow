/**
 * @file cuda_accel.hpp
 * @brief Newton-Raphson 전력조류 계산용 GPU 가속 Jacobian 행렬 계산기
 *
 * FP32 (Mixed Precision) 사용으로 NVIDIA A10 GPU에서 최적 성능 달성
 * A10 성능: FP32 = 31.2 TFLOPS, FP64 = 0.49 TFLOPS (64배 차이)
 */

#ifndef CUDA_ACCEL_HPP
#define CUDA_ACCEL_HPP

#include <vector>
#include <cuda_runtime.h>

class NewtonCudaAccel {
public:
    NewtonCudaAccel();
    ~NewtonCudaAccel();

    /**
     * @brief GPU 메모리 초기화 - Ybus 데이터 및 매핑 테이블 업로드
     *
     * 시뮬레이션 시작 시 1회만 호출. Ybus(G, B)와 Jacobian 매핑 테이블을
     * GPU 메모리로 복사함.
     *
     * @param nb 버스 개수
     * @param j_nnz Jacobian 행렬의 Non-zero 개수
     * @param Ybus Ybus 희소행렬 (Eigen CSC 포맷)
     * @param mapJ11~22 Ybus 인덱스 → Jacobian 인덱스 매핑 (비대각 성분)
     * @param diagMapJ11~22 버스 인덱스 → Jacobian 대각 인덱스 매핑
     */
    void initialize(
        int nb, int j_nnz,
        const void* Ybus,
        const std::vector<int>& mapJ11,
        const std::vector<int>& mapJ21,
        const std::vector<int>& mapJ12,
        const std::vector<int>& mapJ22,
        const std::vector<int>& diagMapJ11,
        const std::vector<int>& diagMapJ21,
        const std::vector<int>& diagMapJ12,
        const std::vector<int>& diagMapJ22
    );

    /**
     * @brief FP32 정밀도로 GPU에서 Jacobian 행렬 계산
     *
     * 매 Newton iteration마다 호출. 전압 벡터 업로드 후 CUDA 커널로
     * Jacobian 계산, 결과는 내부 GPU 버퍼에 저장.
     *
     * @param V_ptr 전압 벡터 (complex<double>, CPU 메모리, 크기 = nbus)
     */
    void update_jacobian_to_buffer_fp32(const void* V_ptr);

    /**
     * @brief FP32 Jacobian 값의 GPU 포인터 반환 (Eigen 저장 순서)
     */
    float* getDeviceJacobianBufferFP32() { return d_J_temp_f; }

    /**
     * @brief 마지막 V 벡터 memcpy 시간 반환 (Host→Device, 밀리초)
     */
    float getLastMemcpyTime() const;

    /**
     * @brief Multi-batch Jacobian 계산 (성능 실험용)
     *
     * 동일한 전압 데이터를 batch_size만큼 복제하여 GPU에서 병렬 계산.
     * GPU 활용률 측정 및 throughput 벤치마킹용.
     *
     * @param V_ptr 전압 벡터 (complex<double>, CPU 메모리, 크기 = nbus)
     * @param batch_size 배치 크기 (1, 2, 4, 8, 16, 32, 64 등)
     * @return 커널 실행 시간 (밀리초)
     */
    float update_jacobian_batch(const void* V_ptr, int batch_size);

    /**
     * @brief Multi-batch 결과 정확도 검증
     *
     * 모든 배치가 동일한 입력으로 동일한 결과를 생성하는지 검증.
     *
     * @param batch_size 검증할 배치 크기
     * @return 정확도 검증 통과 여부
     */
    bool verify_batch_correctness(int batch_size);

    /**
     * @brief 커널 변형 벤치마크 (atomicAdd 병목 정량화용)
     *
     * 세 가지 커널 변형의 성능을 비교:
     *   variant=0: 기본 atomicAdd 버전
     *   variant=1: atomicAdd 제거 버전 (결과 틀림, 성능만 측정)
     *   variant=2: Warp-level Reduction 버전 (실험적)
     *
     * @param V_ptr 전압 벡터 (complex<double>, CPU 메모리, 크기 = nbus)
     * @param batch_size 배치 크기
     * @param variant 커널 변형 (0, 1, 2)
     * @return 커널 실행 시간 (밀리초)
     */
    float benchmark_kernel_variants(const void* V_ptr, int batch_size, int variant);

private:
    // Jacobian 매핑 테이블 (GPU, 읽기 전용)
    int *d_mapJ11, *d_mapJ21, *d_mapJ12, *d_mapJ22;         // 비대각 성분 매핑
    int *d_diagMapJ11, *d_diagMapJ21, *d_diagMapJ12, *d_diagMapJ22;  // 대각 성분 매핑

    // Ybus 데이터 (GPU, FP32)
    float *d_G_f, *d_B_f;   // 컨덕턴스(G), 서셉턴스(B)
    int *d_Y_row, *d_Y_col; // CSC row/col 인덱스
    int Y_nnz;              // Ybus Non-zero 개수

    // 전압 및 Jacobian 버퍼 (GPU, FP32)
    float *d_V_f;       // 전압: [V0_re, V0_im, V1_re, V1_im, ...] (크기 = nbus * 2)
    float *d_J_temp_f;  // Jacobian 값, Eigen 저장 순서 (크기 = J_nnz)
    int J_nnz;          // Jacobian Non-zero 개수
    int nbus;           // 버스 개수

    // 시간 측정용
    cudaEvent_t memcpy_start_evt_, memcpy_stop_evt_;
    float last_memcpy_ms_ = 0.0f;

    // Multi-batch용 메모리 (재사용)
    float *d_V_batch_;      // 배치용 전압 버퍼 (크기 = nbus * 2 * MAX_BATCH)
    float *d_J_batch_;      // 배치용 Jacobian 버퍼 (크기 = J_nnz * MAX_BATCH)
    int max_batch_size_;    // 할당된 최대 배치 크기
};

#endif // CUDA_ACCEL_HPP