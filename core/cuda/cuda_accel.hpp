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
#include <complex>
#include <cuda_runtime.h>
#include <cusparse.h>

class NewtonCudaAccel {
public:
    static bool verbose;  // false로 설정하면 [CUDA] 로그 출력 억제

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
     * @brief 초기 전압 벡터를 GPU에 업로드 (1회만 호출, 이후 V는 GPU에 상주)
     *
     * V0를 GPU의 d_V_cd_ (FP64), d_V_f (FP32), d_Va_d_, d_Vm_d_로 분해하여 업로드.
     *
     * @param V_ptr 초기 전압 (complex<double>, CPU, nbus)
     */
    void upload_V_initial(const void* V_ptr);

    /**
     * @brief GPU에서 전압 업데이트 (FP64 정밀도 유지)
     *
     * Va/Vm을 FP64로 GPU에서 업데이트한 후 V_cd (complex128) 및 V_f (FP32) 재구성.
     * CPU ↔ GPU 전송 없음.
     *
     * @param dx_ptr   보정 벡터 (double, CPU, dimF = npv+2*npq)
     * @param pv_ptr   PV 버스 인덱스 (int32, CPU, npv)
     * @param pq_ptr   PQ 버스 인덱스 (int32, CPU, npq)
     * @param npv      PV 버스 수
     * @param npq      PQ 버스 수
     */
    void update_voltage_gpu(
        const double* dx_ptr,
        const int* pv_ptr, const int* pq_ptr,
        int npv, int npq
    );

    /**
     * @brief GPU에 상주하는 현재 전압 벡터를 CPU로 다운로드
     *
     * @param V_out complex<double>*, CPU, nbus
     */
    void download_V(void* V_out) const;

    /**
     * @brief FP32 정밀도로 GPU에서 Jacobian 행렬 계산 (V 업로드 포함, 레거시)
     *
     * @param V_ptr 전압 벡터 (complex<double>, CPU 메모리, 크기 = nbus)
     */
    void update_jacobian_to_buffer_fp32(const void* V_ptr);

    /**
     * @brief FP32 정밀도로 GPU에서 Jacobian 행렬 계산 (V GPU 상주 전제, 빠름)
     *
     * upload_V_initial 또는 update_voltage_gpu 이후에 호출.
     * V CPU→GPU 전송 없음.
     */
    void update_jacobian_to_buffer_fp32_no_upload();

    /**
     * @brief FP32 Jacobian 값의 GPU 포인터 반환 (Eigen 저장 순서)
     */
    float* getDeviceJacobianBufferFP32() { return d_J_temp_f; }

    /**
     * @brief 마지막 V 벡터 memcpy 시간 반환 (Host→Device, 밀리초)
     */
    float getLastMemcpyTime() const;

    /**
     * @brief Sbus를 GPU에 업로드 (초기화 1회, compute_mismatch_gpu 전에 호출)
     * @param Sbus_ptr complex<double>*, CPU, nbus
     */
    void upload_sbus(const void* Sbus_ptr);

    /**
     * @brief GPU에서 Mismatch 계산 (cuSPARSE SpMV + CUDA 커널)
     *
     * Ibus = Ybus * V  (cuSPARSE, FP64 complex)
     * mis  = V ⊙ conj(Ibus) - Sbus
     * F    = [real(mis[pv]), real(mis[pq]), imag(mis[pq])]
     * normF = max(|F|)
     *
     * 사전조건: upload_mismatch_data() 1회 호출 후 사용
     *
     * @param V_ptr  전압 벡터 (complex<double>, CPU, nbus)
     * @param pv_ptr PV 버스 인덱스 (int32, CPU, npv)
     * @param pq_ptr PQ 버스 인덱스 (int32, CPU, npq)
     * @param npv    PV 버스 수
     * @param npq    PQ 버스 수
     * @param F_out  mismatch 벡터 출력 (double, CPU, npv+2*npq)
     * @param normF  수렴 판정값 출력
     */
    void compute_mismatch_gpu(
        const void* V_ptr,
        const int* pv_ptr, const int* pq_ptr,
        int npv, int npq,
        double* F_out, double& normF
    );

    // =========================================================================
    // Batch API (N개 케이스 동시 처리)
    // =========================================================================

    /**
     * @brief N개 케이스의 초기 전압/Sbus를 GPU에 업로드 (1회)
     */
    void upload_batch_initial(
        const void* const* V_ptrs,
        const void* const* Sbus_ptrs,
        int batch_size
    );

    /**
     * @brief N개 케이스 동시 Mismatch 계산 (V GPU 상주)
     *
     * @param pv_ptr, pq_ptr  공통 버스 인덱스 (CPU)
     * @param F_batch_out     [dimF * batch_size] CPU double 출력
     * @param normF_out       [batch_size] CPU double 출력
     */
    void compute_mismatch_batch(
        const int* pv_ptr, const int* pq_ptr,
        int npv, int npq,
        double* F_batch_out, double* normF_out
    );

    /**
     * @brief N개 케이스 동시 Jacobian 계산 (V GPU 상주)
     *
     * @return GPU FP32 Jacobian 버퍼 포인터 [J_nnz * batch_size]
     */
    float* update_jacobian_batch_no_upload();

    /**
     * @brief N개 케이스 동시 전압 업데이트 (FP64, GPU 상주)
     *
     * @param dx_batch  [dimF * batch_size] CPU double
     */
    void update_voltage_batch(
        const double* dx_batch,
        const int* pv_ptr, const int* pq_ptr,
        int npv, int npq
    );

    /**
     * @brief N개 케이스 동시 전압 업데이트 — Device FP32 포인터 직접 수신
     *
     * D→H→D round trip 없이 GPU에서 직접 FP32→FP64 변환 + Va/Vm scatter +
     * V_cd/V_f 재구성까지 처리.
     *
     * @param d_x_f  [dimF * batch_size] device FP32 (cuDSS 해 벡터)
     * @param pv_ptr, pq_ptr  CPU pv/pq 인덱스 (최초 호출 시 GPU로 업로드)
     */
    void update_voltage_batch_from_fp32_device(
        const float* d_x_f,
        const int* pv_ptr, const int* pq_ptr,
        int npv, int npq
    );

    /**
     * @brief GPU F_batch (FP64) → d_b_batch (FP32, 부호 반전) 직접 변환
     *
     * CPU 경유 없이 GPU에서 -F (FP64) → b (FP32) 변환.
     * cuDSS RHS 버퍼에 직접 기록.
     *
     * @param d_b_out  [dimF * batch_size] device FP32 출력 (cuDSS b 버퍼)
     * @param dimF     dimF = npv + 2*npq
     */
    void negate_F_to_fp32(float* d_b_out, int dimF);

    /**
     * @brief N개 케이스의 최종 V를 CPU로 다운로드
     *
     * @param V_batch_out [nbus * batch_size] complex<double> CPU 출력
     */
    void download_V_batch(void* V_batch_out) const;

    int getBatchSizeGPU() const;

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

    // 전압 및 Jacobian 버퍼
    float *d_V_f;       // 전압: [V0_re, V0_im, V1_re, V1_im, ...] FP32 (크기 = nbus * 2)
    float *d_Va_f_;     // 위상각 FP32 (미사용, 예약)
    float *d_Vm_f_;     // 전압크기 FP32 (미사용, 예약)
    double *d_Va_d_;    // 위상각 FP64 (크기 = nbus) — GPU 상주
    double *d_Vm_d_;    // 전압크기 FP64 (크기 = nbus) — GPU 상주
    float *d_J_temp_f;  // Jacobian 값, Eigen 저장 순서 (크기 = J_nnz)
    int J_nnz;          // Jacobian Non-zero 개수
    int nbus;           // 버스 개수

    // UpdateV용 GPU 버퍼 (dx, pv, pq)
    double *d_dx_;      // 보정 벡터 (dimF)
    int *d_pv_upd_;     // PV 버스 인덱스 (npv)
    int *d_pq_upd_;     // PQ 버스 인덱스 (npq)
    int npv_upd_, npq_upd_;
    bool voltage_initialized_;  // upload_V_initial 호출 여부

    // ---- Batch GPU 버퍼 ----
    int batch_size_gpu_ = 0;

    // Batch 전압 (FP64)
    cuDoubleComplex* d_V_cd_batch_;  // [nbus * batch_size] complex128
    double*          d_Va_d_batch_;  // [nbus * batch_size] 위상각
    double*          d_Vm_d_batch_;  // [nbus * batch_size] 전압크기
    float*           d_V_f_batch_;   // [nbus * 2 * batch_size] FP32 인터리브

    // Batch Sbus (FP64)
    cuDoubleComplex* d_Sbus_batch_;  // [nbus * batch_size]

    // Batch Ibus (cuSPARSE SpMV 결과)
    cuDoubleComplex* d_Ibus_batch_;  // [nbus * batch_size]

    // Batch F, pv/pq (mismatch 출력)
    double*          d_F_batch_;     // [dimF * batch_size]

    // Batch dx (updateV 입력)
    double*          d_dx_batch_;    // [dimF * batch_size]

    // Batch Jacobian (FP32)
    float*           d_J_batch_new_; // [J_nnz * batch_size]

    // cuSPARSE batch SpMV 디스크립터들
    // (각 배치마다 별도 디스크립터 필요)
    cuDoubleComplex** d_V_cd_ptrs_;   // GPU 포인터 배열 [batch_size]
    cuDoubleComplex** d_Ibus_ptrs_;   // GPU 포인터 배열 [batch_size]

    // cuSPARSE SpMM (Ybus × V_batch, column-major dense matrix)
    cusparseDnMatDescr_t sp_V_mat_;       // Dense [nbus × batch_size] col-major
    cusparseDnMatDescr_t sp_Ibus_mat_;    // Dense [nbus × batch_size] col-major
    void*   d_spmm_buf_;
    size_t  spmm_buf_size_;
    bool    spmm_initialized_;

    // normF GPU reduction 버퍼
    double* d_normF_batch_;   // [batch_size] GPU

    bool batch_initialized_ = false;

    // 시간 측정용
    cudaEvent_t memcpy_start_evt_, memcpy_stop_evt_;
    float last_memcpy_ms_ = 0.0f;

    // ---- cuSPARSE Mismatch 계산용 ----
    cusparseHandle_t   sp_handle_;    // cuSPARSE 핸들

    // Ybus CSR (complex128, FP64) - GPU
    cuDoubleComplex*   d_Ybus_val_;   // CSR 값 (complex128)
    int*               d_Ybus_row_;   // CSR row pointer (nbus+1)
    int*               d_Ybus_col_;   // CSR col indices (Y_nnz)

    // cuSPARSE 행렬/벡터 디스크립터
    cusparseSpMatDescr_t sp_Ybus_;
    cusparseDnVecDescr_t sp_V_;
    cusparseDnVecDescr_t sp_Ibus_;

    // Mismatch 계산용 GPU 버퍼 (FP64 complex)
    cuDoubleComplex*   d_V_cd_;      // 전압 (complex128, nbus)
    cuDoubleComplex*   d_Ibus_;      // 전류 주입 (complex128, nbus)
    cuDoubleComplex*   d_Sbus_;      // 전력 주입 (complex128, nbus)

    // F 벡터 및 pv/pq 인덱스 (GPU)
    double*            d_F_;         // mismatch 벡터 (double, npv+2*npq)
    int*               d_pv_mis_;    // PV 버스 인덱스
    int*               d_pq_mis_;    // PQ 버스 인덱스
    int                npv_mis_, npq_mis_;

    // SpMV 임시 버퍼
    void*              d_spmv_buf_;
    size_t             spmv_buf_size_;

    bool               mismatch_initialized_;

    // Multi-batch용 메모리 (재사용)
    float *d_V_batch_;      // 배치용 전압 버퍼 (크기 = nbus * 2 * MAX_BATCH)
    float *d_J_batch_;      // 배치용 Jacobian 버퍼 (크기 = J_nnz * MAX_BATCH)
    int max_batch_size_;    // 할당된 최대 배치 크기
};

#endif // CUDA_ACCEL_HPP