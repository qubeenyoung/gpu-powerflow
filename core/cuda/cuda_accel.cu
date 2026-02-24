/**
 * @file cuda_accel.cu
 * @brief Newton-Raphson 전력조류 계산용 GPU Jacobian 커널 구현
 *
 * FP32 (Mixed Precision) 사용으로 A10 GPU에서 64배 빠른 성능 달성
 */

#include "cuda_accel.hpp"
#include <cstdio>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>
#include <cuComplex.h>
#include <cusparse.h>

#define CUSPARSE_CHECK(call) \
    do { cusparseStatus_t st = call; if (st != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error %d at %s:%d\n", (int)st, __FILE__, __LINE__); \
    }} while(0)

using YbusType = Eigen::SparseMatrix<std::complex<double>>;

bool NewtonCudaAccel::verbose = true;  // 기본값: 출력 on
#define CUDA_LOG(...) do { if (NewtonCudaAccel::verbose) printf(__VA_ARGS__); } while(0)


// CUDA 에러 체크 매크로
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    } } while (0)


// ============================================================================
// 생성자 / 소멸자
// ============================================================================

NewtonCudaAccel::NewtonCudaAccel()
    : d_mapJ11(nullptr), d_mapJ21(nullptr), d_mapJ12(nullptr), d_mapJ22(nullptr)
    , d_diagMapJ11(nullptr), d_diagMapJ21(nullptr), d_diagMapJ12(nullptr), d_diagMapJ22(nullptr)
    , d_G_f(nullptr), d_B_f(nullptr), d_Y_row(nullptr), d_Y_col(nullptr)
    , d_V_f(nullptr), d_Va_f_(nullptr), d_Vm_f_(nullptr)
    , d_Va_d_(nullptr), d_Vm_d_(nullptr)
    , d_J_temp_f(nullptr)
    , d_V_batch_(nullptr), d_J_batch_(nullptr)
    , Y_nnz(0), J_nnz(0), nbus(0)
    , max_batch_size_(0)
    , d_dx_(nullptr), d_pv_upd_(nullptr), d_pq_upd_(nullptr)
    , npv_upd_(0), npq_upd_(0)
    , voltage_initialized_(false)
    , batch_size_gpu_(0)
    , d_V_cd_batch_(nullptr), d_Va_d_batch_(nullptr), d_Vm_d_batch_(nullptr), d_V_f_batch_(nullptr)
    , d_Sbus_batch_(nullptr), d_Ibus_batch_(nullptr)
    , d_F_batch_(nullptr), d_dx_batch_(nullptr), d_J_batch_new_(nullptr)
    , d_V_cd_ptrs_(nullptr), d_Ibus_ptrs_(nullptr)
    , sp_V_mat_(nullptr), sp_Ibus_mat_(nullptr)
    , d_spmm_buf_(nullptr), spmm_buf_size_(0), spmm_initialized_(false)
    , d_normF_batch_(nullptr)
    , batch_initialized_(false)
    , last_memcpy_ms_(0.0f)
    // cuSPARSE mismatch
    , sp_handle_(nullptr)
    , d_Ybus_val_(nullptr), d_Ybus_row_(nullptr), d_Ybus_col_(nullptr)
    , sp_Ybus_(nullptr), sp_V_(nullptr), sp_Ibus_(nullptr)
    , d_V_cd_(nullptr), d_Ibus_(nullptr), d_Sbus_(nullptr)
    , d_F_(nullptr), d_pv_mis_(nullptr), d_pq_mis_(nullptr)
    , npv_mis_(0), npq_mis_(0)
    , d_spmv_buf_(nullptr), spmv_buf_size_(0)
    , mismatch_initialized_(false)
{
    cudaEventCreate(&memcpy_start_evt_);
    cudaEventCreate(&memcpy_stop_evt_);
    CUSPARSE_CHECK(cusparseCreate(&sp_handle_));
    CUDA_LOG("[CUDA] Jacobian 가속기 생성 완료 (FP32 Mixed Precision)\n");
}

NewtonCudaAccel::~NewtonCudaAccel() {
    // 매핑 테이블 해제
    if (d_mapJ11) cudaFree(d_mapJ11);
    if (d_mapJ21) cudaFree(d_mapJ21);
    if (d_mapJ12) cudaFree(d_mapJ12);
    if (d_mapJ22) cudaFree(d_mapJ22);
    if (d_diagMapJ11) cudaFree(d_diagMapJ11);
    if (d_diagMapJ21) cudaFree(d_diagMapJ21);
    if (d_diagMapJ12) cudaFree(d_diagMapJ12);
    if (d_diagMapJ22) cudaFree(d_diagMapJ22);

    // Ybus 데이터 해제
    if (d_G_f) cudaFree(d_G_f);
    if (d_B_f) cudaFree(d_B_f);
    if (d_Y_row) cudaFree(d_Y_row);
    if (d_Y_col) cudaFree(d_Y_col);

    // 전압/Jacobian 버퍼 해제
    if (d_V_f) cudaFree(d_V_f);
    if (d_Va_f_) cudaFree(d_Va_f_);
    if (d_Vm_f_) cudaFree(d_Vm_f_);
    if (d_Va_d_) cudaFree(d_Va_d_);
    if (d_Vm_d_) cudaFree(d_Vm_d_);
    if (d_J_temp_f) cudaFree(d_J_temp_f);
    // UpdateV 버퍼 해제
    if (d_dx_) cudaFree(d_dx_);
    if (d_pv_upd_) cudaFree(d_pv_upd_);
    if (d_pq_upd_) cudaFree(d_pq_upd_);

    // Multi-batch 버퍼 해제 (레거시)
    if (d_V_batch_) cudaFree(d_V_batch_);
    if (d_J_batch_) cudaFree(d_J_batch_);

    // Batch 버퍼 해제
    if (d_V_cd_batch_)  cudaFree(d_V_cd_batch_);
    if (d_Va_d_batch_)  cudaFree(d_Va_d_batch_);
    if (d_Vm_d_batch_)  cudaFree(d_Vm_d_batch_);
    if (d_V_f_batch_)   cudaFree(d_V_f_batch_);
    if (d_Sbus_batch_)  cudaFree(d_Sbus_batch_);
    if (d_Ibus_batch_)  cudaFree(d_Ibus_batch_);
    if (d_F_batch_)     cudaFree(d_F_batch_);
    if (d_dx_batch_)    cudaFree(d_dx_batch_);
    if (d_J_batch_new_) cudaFree(d_J_batch_new_);
    if (d_V_cd_ptrs_)   cudaFree(d_V_cd_ptrs_);
    if (d_Ibus_ptrs_)   cudaFree(d_Ibus_ptrs_);

    // SpMM 리소스 해제
    if (sp_V_mat_)      cusparseDestroyDnMat(sp_V_mat_);
    if (sp_Ibus_mat_)   cusparseDestroyDnMat(sp_Ibus_mat_);
    if (d_spmm_buf_)    cudaFree(d_spmm_buf_);
    if (d_normF_batch_) cudaFree(d_normF_batch_);

    // cuSPARSE 리소스 해제
    if (sp_Ybus_)     cusparseDestroySpMat(sp_Ybus_);
    if (sp_V_)        cusparseDestroyDnVec(sp_V_);
    if (sp_Ibus_)     cusparseDestroyDnVec(sp_Ibus_);
    if (d_Ybus_val_)  cudaFree(d_Ybus_val_);
    if (d_Ybus_row_)  cudaFree(d_Ybus_row_);
    if (d_Ybus_col_)  cudaFree(d_Ybus_col_);
    if (d_V_cd_)      cudaFree(d_V_cd_);
    if (d_Ibus_)      cudaFree(d_Ibus_);
    if (d_Sbus_)      cudaFree(d_Sbus_);
    if (d_F_)         cudaFree(d_F_);
    if (d_pv_mis_)    cudaFree(d_pv_mis_);
    if (d_pq_mis_)    cudaFree(d_pq_mis_);
    if (d_spmv_buf_)  cudaFree(d_spmv_buf_);
    if (sp_handle_)   cusparseDestroy(sp_handle_);

    // 이벤트 해제
    cudaEventDestroy(memcpy_start_evt_);
    cudaEventDestroy(memcpy_stop_evt_);

    CUDA_LOG("[CUDA] Jacobian 가속기 해제 완료\n");
}

float NewtonCudaAccel::getLastMemcpyTime() const {
    return last_memcpy_ms_;
}


// ============================================================================
// 초기화: Ybus 및 매핑 테이블을 GPU로 업로드
// ============================================================================

void NewtonCudaAccel::initialize(
    int nb, int j_nnz, const void* Ybus_void,
    const std::vector<int>& mapJ11, const std::vector<int>& mapJ21,
    const std::vector<int>& mapJ12, const std::vector<int>& mapJ22,
    const std::vector<int>& diagMapJ11, const std::vector<int>& diagMapJ21,
    const std::vector<int>& diagMapJ12, const std::vector<int>& diagMapJ22
) {
    this->nbus = nb;
    this->J_nnz = j_nnz;

    const YbusType* Y = static_cast<const YbusType*>(Ybus_void);
    this->Y_nnz = Y->nonZeros();

    CUDA_LOG("[CUDA] GPU 데이터 초기화 (버스: %d, Y_nnz: %d, J_nnz: %d)\n", nb, Y_nnz, J_nnz);

    // --- 매핑 테이블 업로드 ---
    auto upload_int = [&](int*& d_ptr, const std::vector<int>& h_vec, const char* name) {
        if (h_vec.empty()) return;
        size_t bytes = h_vec.size() * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_vec.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_LOG("[CUDA]   %s 업로드: %zu개\n", name, h_vec.size());
    };

    upload_int(d_mapJ11, mapJ11, "mapJ11");
    upload_int(d_mapJ21, mapJ21, "mapJ21");
    upload_int(d_mapJ12, mapJ12, "mapJ12");
    upload_int(d_mapJ22, mapJ22, "mapJ22");
    upload_int(d_diagMapJ11, diagMapJ11, "diagMapJ11");
    upload_int(d_diagMapJ21, diagMapJ21, "diagMapJ21");
    upload_int(d_diagMapJ12, diagMapJ12, "diagMapJ12");
    upload_int(d_diagMapJ22, diagMapJ22, "diagMapJ22");

    // --- Ybus 데이터 추출 (G, B) 및 FP32 변환 ---
    std::vector<float> h_G_f(Y_nnz), h_B_f(Y_nnz);
    std::vector<int> h_Y_col(Y_nnz);

    for (int k = 0; k < Y_nnz; ++k) {
        std::complex<double> v = Y->valuePtr()[k];
        h_G_f[k] = static_cast<float>(v.real());
        h_B_f[k] = static_cast<float>(v.imag());
    }

    // CSC -> COO 변환 (열 인덱스 복원)
    const int* outer = Y->outerIndexPtr();
    for (int i = 0; i < Y->cols(); ++i) {
        for (int k = outer[i]; k < outer[i+1]; ++k) {
            h_Y_col[k] = i;
        }
    }

    // --- GPU 메모리 할당 및 업로드 ---
    size_t fb = Y_nnz * sizeof(float);
    size_t ib = Y_nnz * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_G_f, fb));
    CUDA_CHECK(cudaMemcpy(d_G_f, h_G_f.data(), fb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_B_f, fb));
    CUDA_CHECK(cudaMemcpy(d_B_f, h_B_f.data(), fb, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_Y_row, ib));
    CUDA_CHECK(cudaMemcpy(d_Y_row, Y->innerIndexPtr(), ib, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_Y_col, ib));
    CUDA_CHECK(cudaMemcpy(d_Y_col, h_Y_col.data(), ib, cudaMemcpyHostToDevice));

    // 전압 버퍼: complex → 2 floats per bus (Jacobian 커널용, FP32)
    CUDA_CHECK(cudaMalloc(&d_V_f, nb * sizeof(float) * 2));

    // Va, Vm 버퍼 FP64 (GPU 상주 전압, 정밀도 보전)
    CUDA_CHECK(cudaMalloc(&d_Va_d_, nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Vm_d_, nb * sizeof(double)));

    // Jacobian 출력 버퍼
    CUDA_CHECK(cudaMalloc(&d_J_temp_f, J_nnz * sizeof(float)));

    // ---- Ybus complex CSR을 GPU에 업로드 (cuSPARSE SpMV용, FP64) ----
    {
        // Eigen CSC → CSR 변환 (Ybus는 대칭이므로 transpose = CSR)
        // 실제로 CSC outerIndex = CSR rowPtr, innerIndex = CSR colInd (전치 시)
        // 하지만 Ybus는 대칭행렬이 아닐 수 있으므로 명시적으로 변환
        int n = Y->rows();  // nbus
        std::vector<cuDoubleComplex> h_val(Y_nnz);
        std::vector<int> h_row(n + 1, 0), h_col(Y_nnz);

        // CSC → CSR: row count 먼저 집계
        for (int k = 0; k < Y_nnz; ++k) {
            int r = Y->innerIndexPtr()[k];  // CSC inner = row
            h_row[r + 1]++;
        }
        for (int r = 0; r < n; ++r) h_row[r + 1] += h_row[r];

        // CSC → CSR: 값/col 배치
        std::vector<int> pos(h_row.begin(), h_row.end());
        const int* csc_outer = Y->outerIndexPtr();  // col ptr
        for (int c = 0; c < n; ++c) {
            for (int k = csc_outer[c]; k < csc_outer[c + 1]; ++k) {
                int r = Y->innerIndexPtr()[k];
                int dst = pos[r]++;
                auto v = Y->valuePtr()[k];
                h_val[dst] = make_cuDoubleComplex(v.real(), v.imag());
                h_col[dst] = c;
            }
        }

        CUDA_CHECK(cudaMalloc(&d_Ybus_val_, Y_nnz * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Ybus_row_, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Ybus_col_, Y_nnz   * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_Ybus_val_, h_val.data(), Y_nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ybus_row_, h_row.data(), (n + 1) * sizeof(int),           cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ybus_col_, h_col.data(), Y_nnz   * sizeof(int),           cudaMemcpyHostToDevice));

        // cuSPARSE SpMat 디스크립터
        CUSPARSE_CHECK(cusparseCreateCsr(
            &sp_Ybus_,
            n, n, Y_nnz,
            d_Ybus_row_, d_Ybus_col_, d_Ybus_val_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F
        ));

        // V, Ibus 벡터 버퍼 (complex128)
        CUDA_CHECK(cudaMalloc(&d_V_cd_, n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Ibus_, n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Sbus_, n * sizeof(cuDoubleComplex)));

        // 벡터 디스크립터
        CUSPARSE_CHECK(cusparseCreateDnVec(&sp_V_,    n, d_V_cd_, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&sp_Ibus_, n, d_Ibus_, CUDA_C_64F));

        // SpMV 버퍼 크기 조회
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            sp_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_Ybus_, sp_V_, &beta, sp_Ibus_,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size_
        ));
        if (spmv_buf_size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmv_buf_, spmv_buf_size_));
        }

        CUDA_LOG("[CUDA] cuSPARSE Ybus CSR 업로드 완료 (nbus=%d, Y_nnz=%d)\n", n, Y_nnz);
    }

    CUDA_LOG("[CUDA] 초기화 완료 (FP32 Mixed Precision)\n");
}


// ============================================================================
// FP32 Jacobian 업데이트 커널
// ============================================================================
//
// 입력:
//   - G, B: Ybus의 컨덕턴스(실수부), 서셉턴스(허수부)
//   - row, col: Ybus의 (i, j) 인덱스
//   - V_real: 전압 벡터 [V0_re, V0_im, V1_re, V1_im, ...]
//   - map11~22: Ybus 인덱스 → Jacobian 인덱스 매핑 (비대각)
//   - diag11~22: 버스 인덱스 → Jacobian 대각 인덱스 매핑
//
// 출력:
//   - J_values: Jacobian 행렬 값 (Eigen 저장 순서)
//
// Jacobian 구조:
//   J = [J11  J12]  =  [dP/dθ   dP/d|V|]
//       [J21  J22]     [dQ/dθ   dQ/d|V|]
//
// ============================================================================

__global__ void update_jacobian_kernel_fp32(
    int n_elements,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const int* __restrict__ row,
    const int* __restrict__ col,
    const float* __restrict__ V_real,
    const int* __restrict__ map11,
    const int* __restrict__ map21,
    const int* __restrict__ map12,
    const int* __restrict__ map22,
    const int* __restrict__ diag11,
    const int* __restrict__ diag21,
    const int* __restrict__ diag12,
    const int* __restrict__ diag22,
    float* J_values
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elements) return;

    int i = row[k];  // from bus
    int j = col[k];  // to bus

    // 복소수 연산을 위한 cuComplex 사용
    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);           // Yij
    cuFloatComplex Vi = make_cuFloatComplex(V_real[i*2], V_real[i*2+1]);  // Vi
    cuFloatComplex Vj = make_cuFloatComplex(V_real[j*2], V_real[j*2+1]);  // Vj

    // 전류 주입: curr = Yij * Vj
    cuFloatComplex curr = cuCmulf(y, Vj);

    // --- 미분항 계산 ---

    // 위상각 미분: term_va = -j * Vi * conj(Yij * Vj)
    cuFloatComplex term_va = cuCmulf(
        make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi)),  // -j * Vi
        cuConjf(curr)
    );

    // 전압크기 미분: term_vm = Vi * conj(Yij * Vj / |Vj|)
    float vj_norm = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_norm > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr)/vj_norm, cuCimagf(curr)/vj_norm)))
        : make_cuFloatComplex(0.0f, 0.0f);

    // --- Jacobian에 값 누적 (Atomic Add) ---

    // 비대각 성분 (off-diagonal)
    if (map11[k] >= 0) atomicAdd(&J_values[map11[k]], cuCrealf(term_va));  // dP/dθ
    if (map21[k] >= 0) atomicAdd(&J_values[map21[k]], cuCimagf(term_va));  // dQ/dθ
    if (map12[k] >= 0) atomicAdd(&J_values[map12[k]], cuCrealf(term_vm));  // dP/d|V|
    if (map22[k] >= 0) atomicAdd(&J_values[map22[k]], cuCimagf(term_vm));  // dQ/d|V|

    // 대각 성분 보정 (diagonal correction)
    if (diag11[i] >= 0) atomicAdd(&J_values[diag11[i]], -cuCrealf(term_va));
    if (diag21[i] >= 0) atomicAdd(&J_values[diag21[i]], -cuCimagf(term_va));

    // 전압크기 미분의 대각 보정
    float vi_norm = cuCabsf(Vi);
    if (vi_norm > 1e-6f) {
        cuFloatComplex term_vm2 = cuCmulf(
            make_cuFloatComplex(cuCrealf(Vi)/vi_norm, cuCimagf(Vi)/vi_norm),
            cuConjf(curr)
        );
        if (diag12[i] >= 0) atomicAdd(&J_values[diag12[i]], cuCrealf(term_vm2));
        if (diag22[i] >= 0) atomicAdd(&J_values[diag22[i]], cuCimagf(term_vm2));
    }
}




// ============================================================================
// FP32 Jacobian 업데이트 함수 (V CPU→GPU 업로드 포함, 레거시용)
// ============================================================================

void NewtonCudaAccel::update_jacobian_to_buffer_fp32(const void* V_ptr) {
    // 1. 전압 벡터 double → float 변환 (CPU)
    const double* V_double = static_cast<const double*>(V_ptr);
    std::vector<float> h_V_f(nbus * 2);
    for (int i = 0; i < nbus * 2; ++i) {
        h_V_f[i] = static_cast<float>(V_double[i]);
    }

    // 2. 전압 벡터 GPU로 업로드 (시간 측정)
    cudaEventRecord(memcpy_start_evt_);
    CUDA_CHECK(cudaMemcpy(d_V_f, h_V_f.data(), nbus * sizeof(float) * 2, cudaMemcpyHostToDevice));
    cudaEventRecord(memcpy_stop_evt_);
    cudaEventSynchronize(memcpy_stop_evt_);
    cudaEventElapsedTime(&last_memcpy_ms_, memcpy_start_evt_, memcpy_stop_evt_);

    // 3. Jacobian 버퍼 초기화 + 커널 실행
    update_jacobian_to_buffer_fp32_no_upload();
}


// ============================================================================
// Mismatch 커널: mis = V ⊙ conj(Ibus) - Sbus, F 벡터 패킹
//
// 입력:
//   V    [nbus]: complex128
//   Ibus [nbus]: complex128  (cuSPARSE SpMV 결과)
//   Sbus [nbus]: complex128
//   pv   [npv], pq [npq]: 버스 인덱스
//
// 출력:
//   F    [npv+2*npq]: [real(mis[pv]) | real(mis[pq]) | imag(mis[pq])]
// ============================================================================

__global__ void mismatch_pack_kernel(
    const cuDoubleComplex* __restrict__ V,
    const cuDoubleComplex* __restrict__ Ibus,
    const cuDoubleComplex* __restrict__ Sbus,
    const int* __restrict__ pv,
    const int* __restrict__ pq,
    int npv, int npq,
    double* F  // 크기 = npv + 2*npq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = npv + 2 * npq;
    if (tid >= total) return;

    // mis[bus] = V[bus] * conj(Ibus[bus]) - Sbus[bus]
    auto mis = [&](int bus) -> cuDoubleComplex {
        cuDoubleComplex vi = V[bus];
        cuDoubleComplex ii = Ibus[bus];
        cuDoubleComplex si = Sbus[bus];
        // V * conj(Ibus)
        double re = cuCreal(vi) * cuCreal(ii) + cuCimag(vi) * cuCimag(ii);
        double im = cuCimag(vi) * cuCreal(ii) - cuCreal(vi) * cuCimag(ii);
        return make_cuDoubleComplex(re - cuCreal(si), im - cuCimag(si));
    };

    if (tid < npv) {
        // real(mis[pv[tid]])
        F[tid] = cuCreal(mis(pv[tid]));
    } else if (tid < npv + npq) {
        // real(mis[pq[i]])
        int i = tid - npv;
        F[tid] = cuCreal(mis(pq[i]));
    } else {
        // imag(mis[pq[i]])
        int i = tid - npv - npq;
        F[npv + npq + i] = cuCimag(mis(pq[i]));
    }
}

// normF = max(|F[i]|) — 간단한 reduction (블록 단위, 충분히 작은 F)
__global__ void max_abs_kernel(const double* F, int n, double* out) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? fabs(F[gid]) : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}


// ============================================================================
// upload_sbus: Sbus를 GPU에 업로드 (초기화 1회)
// ============================================================================

void NewtonCudaAccel::upload_sbus(const void* Sbus_ptr) {
    const std::complex<double>* sb = static_cast<const std::complex<double>*>(Sbus_ptr);
    CUDA_CHECK(cudaMemcpy(d_Sbus_, sb, nbus * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}


// ============================================================================
// V 변환 커널: complex128 → FP32 인터리브 (V_cd → V_f)
// ============================================================================

__global__ void convert_V_cd_to_f_kernel(
    const cuDoubleComplex* __restrict__ V_cd,
    float* V_f,
    int nbus
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbus) return;
    V_f[i * 2]     = (float)cuCreal(V_cd[i]);
    V_f[i * 2 + 1] = (float)cuCimag(V_cd[i]);
}

// ============================================================================
// Va/Vm 업데이트 + V 재구성 커널 (FP64)
//
// Va[pv[i]] += dx[i]                 for i in [0, npv)
// Va[pq[i]] += dx[npv + i]           for i in [0, npq)
// Vm[pq[i]] += dx[npv + npq + i]     for i in [0, npq)
// V_cd[bus]  = Vm[bus] * (cos(Va[bus]) + j*sin(Va[bus]))
// V_f[bus*2] = (float)V_cd[bus].re,  V_f[bus*2+1] = (float)V_cd[bus].im
// ============================================================================

__global__ void update_voltage_kernel_fp64(
    double* __restrict__ Va,       // [nbus]
    double* __restrict__ Vm,       // [nbus]
    cuDoubleComplex* __restrict__ V_cd,  // [nbus]
    float*  __restrict__ V_f,      // [nbus*2]
    const double* __restrict__ dx, // [npv+2*npq]
    const int* __restrict__ pv,    // [npv]
    const int* __restrict__ pq,    // [npq]
    int npv, int npq, int nbus
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = npv + 2 * npq;
    if (tid >= total) return;

    if (tid < npv) {
        // PV 버스: 위상각 업데이트
        Va[pv[tid]] += dx[tid];
    } else if (tid < npv + npq) {
        // PQ 버스: 위상각 업데이트
        int i = tid - npv;
        Va[pq[i]] += dx[tid];
    } else {
        // PQ 버스: 전압크기 업데이트
        int i = tid - npv - npq;
        Vm[pq[i]] += dx[tid];
    }
    // V 재구성은 별도 커널에서 (위상각/크기 업데이트가 완료된 후)
}

__global__ void reconstruct_V_kernel(
    const double* __restrict__ Va,
    const double* __restrict__ Vm,
    cuDoubleComplex* __restrict__ V_cd,
    float* __restrict__ V_f,
    int nbus
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbus) return;
    double re = Vm[i] * cos(Va[i]);
    double im = Vm[i] * sin(Va[i]);
    V_cd[i] = make_cuDoubleComplex(re, im);
    V_f[i * 2]     = (float)re;
    V_f[i * 2 + 1] = (float)im;
}


// ============================================================================
// upload_V_initial: 초기 전압 벡터 GPU 업로드 (1회)
// ============================================================================

void NewtonCudaAccel::upload_V_initial(const void* V_ptr) {
    const std::complex<double>* V = static_cast<const std::complex<double>*>(V_ptr);

    // d_V_cd_: complex FP64
    CUDA_CHECK(cudaMemcpy(d_V_cd_, V, nbus * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // d_Va_d_, d_Vm_d_: 분해
    std::vector<double> h_Va(nbus), h_Vm(nbus);
    for (int i = 0; i < nbus; ++i) {
        h_Va[i] = std::arg(V[i]);
        h_Vm[i] = std::abs(V[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_Va_d_, h_Va.data(), nbus * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vm_d_, h_Vm.data(), nbus * sizeof(double), cudaMemcpyHostToDevice));

    // d_V_f: FP32 변환
    int block = 256, grid = (nbus + block - 1) / block;
    convert_V_cd_to_f_kernel<<<grid, block>>>(d_V_cd_, d_V_f, nbus);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    voltage_initialized_ = true;
    CUDA_LOG("[CUDA] 초기 전압 GPU 업로드 완료 (nbus=%d)\n", nbus);
}


// ============================================================================
// update_voltage_gpu: dx로 Va/Vm 업데이트 후 V 재구성 (FP64, GPU 상주)
// ============================================================================

void NewtonCudaAccel::update_voltage_gpu(
    const double* dx_ptr,
    const int* pv_ptr, const int* pq_ptr,
    int npv, int npq
) {
    int dimF = npv + 2 * npq;

    // pv/pq 인덱스 업로드 (변경시 or 최초)
    if (npv != npv_upd_ || npq != npq_upd_) {
        if (d_pv_upd_) cudaFree(d_pv_upd_);
        if (d_pq_upd_) cudaFree(d_pq_upd_);
        if (d_dx_)     cudaFree(d_dx_);
        CUDA_CHECK(cudaMalloc(&d_pv_upd_, npv  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pq_upd_, npq  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dx_,     dimF * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_pv_upd_, pv_ptr, npv * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pq_upd_, pq_ptr, npq * sizeof(int), cudaMemcpyHostToDevice));
        npv_upd_ = npv;
        npq_upd_ = npq;
    }

    // dx 업로드 (매 iteration)
    CUDA_CHECK(cudaMemcpy(d_dx_, dx_ptr, dimF * sizeof(double), cudaMemcpyHostToDevice));

    // Va/Vm 업데이트 커널
    int block = 256, grid = (dimF + block - 1) / block;
    update_voltage_kernel_fp64<<<grid, block>>>(
        d_Va_d_, d_Vm_d_, d_V_cd_, d_V_f,
        d_dx_, d_pv_upd_, d_pq_upd_,
        npv, npq, nbus
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // V 재구성 (Va/Vm → V_cd + V_f)
    int grid2 = (nbus + block - 1) / block;
    reconstruct_V_kernel<<<grid2, block>>>(d_Va_d_, d_Vm_d_, d_V_cd_, d_V_f, nbus);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================================
// download_V: GPU → CPU 전압 다운로드
// ============================================================================

void NewtonCudaAccel::download_V(void* V_out) const {
    CUDA_CHECK(cudaMemcpy(V_out, d_V_cd_, nbus * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
}


// ============================================================================
// update_jacobian_to_buffer_fp32_no_upload: V 업로드 없이 Jacobian 계산
// (d_V_f가 이미 GPU에 있다고 가정)
// ============================================================================

void NewtonCudaAccel::update_jacobian_to_buffer_fp32_no_upload() {
    // Jacobian 버퍼 초기화
    CUDA_CHECK(cudaMemset(d_J_temp_f, 0, J_nnz * sizeof(float)));

    // 커널 실행 (d_V_f 이미 GPU 상주)
    int block = 256;
    int grid = (Y_nnz + block - 1) / block;

    update_jacobian_kernel_fp32<<<grid, block>>>(
        Y_nnz,
        d_G_f, d_B_f,
        d_Y_row, d_Y_col,
        d_V_f,
        d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
        d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
        d_J_temp_f
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================================
// compute_mismatch_gpu: Mismatch 계산 (GPU 전체)
// ============================================================================

void NewtonCudaAccel::compute_mismatch_gpu(
    const void* V_ptr,
    const int* pv_ptr, const int* pq_ptr,
    int npv, int npq,
    double* F_out, double& normF
) {
    int dimF = npv + 2 * npq;

    // pv/pq 인덱스가 변경됐거나 처음이면 재업로드
    if (!mismatch_initialized_ || npv != npv_mis_ || npq != npq_mis_) {
        if (d_pv_mis_) cudaFree(d_pv_mis_);
        if (d_pq_mis_) cudaFree(d_pq_mis_);
        if (d_F_)      cudaFree(d_F_);
        CUDA_CHECK(cudaMalloc(&d_pv_mis_, npv  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pq_mis_, npq  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_F_,      dimF * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_pv_mis_, pv_ptr, npv * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pq_mis_, pq_ptr, npq * sizeof(int), cudaMemcpyHostToDevice));
        npv_mis_ = npv;
        npq_mis_ = npq;
        mismatch_initialized_ = true;
    }

    // 1. V 업로드 (FP64 complex) — V가 GPU에 상주 중이면 스킵
    if (V_ptr != nullptr && !voltage_initialized_) {
        const std::complex<double>* V_cd = static_cast<const std::complex<double>*>(V_ptr);
        CUDA_CHECK(cudaMemcpy(d_V_cd_, V_cd, nbus * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    // voltage_initialized_ = true이면 d_V_cd_는 이미 최신 상태 (update_voltage_gpu가 유지)

    // 2. cuSPARSE SpMV: Ibus = Ybus * V
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    CUSPARSE_CHECK(cusparseSpMV(
        sp_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, sp_Ybus_, sp_V_, &beta, sp_Ibus_,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf_
    ));

    // 3. Mismatch 패킹 커널
    int block = 256;
    int grid  = (dimF + block - 1) / block;
    mismatch_pack_kernel<<<grid, block>>>(
        d_V_cd_, d_Ibus_, d_Sbus_,
        d_pv_mis_, d_pq_mis_,
        npv, npq, d_F_
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. normF: max reduction
    int n_blocks = (dimF + block - 1) / block;
    double* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(double)));
    max_abs_kernel<<<n_blocks, block, block * sizeof(double)>>>(d_F_, dimF, d_partial);
    CUDA_CHECK(cudaGetLastError());
    // 두 번째 reduction (블록 수가 적으므로 CPU에서 처리)
    std::vector<double> h_partial(n_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, n_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    normF = *std::max_element(h_partial.begin(), h_partial.end());
    cudaFree(d_partial);

    // 5. F 다운로드 (수렴 판정 후 Solve에 사용)
    CUDA_CHECK(cudaMemcpy(F_out, d_F_, dimF * sizeof(double), cudaMemcpyDeviceToHost));
}


// ============================================================================
// Multi-Batch FP32 Jacobian 업데이트 커널 (성능 실험용)
// ============================================================================
//
// 입력:
//   - batch_size: 동시에 처리할 배치 개수 (1, 2, 4, 8, 16, 32, 64 등)
//   - V_real: 전압 벡터 [Batch0의 V들 | Batch1의 V들 | ... | BatchN의 V들]
//            크기: nbus * 2 * batch_size
//   - J_values: Jacobian 값 [Batch0의 J들 | Batch1의 J들 | ... | BatchN의 J들]
//               크기: J_nnz * batch_size
//
// 구조(Ybus, 매핑)는 모든 배치가 동일하므로 공유하여 사용 (Uniform Batch)
//
// ============================================================================

__global__ void update_jacobian_batch_kernel_fp32(
    int n_elements,      // Y_nnz (Ybus의 비영점소 개수)
    int batch_size,      // 배치 크기
    int nbus,            // 버스 개수
    int J_nnz,           // Jacobian 비영점소 개수
    const float* __restrict__ G,
    const float* __restrict__ B,
    const int* __restrict__ row,
    const int* __restrict__ col,
    const float* __restrict__ V_real,  // 크기: nbus * 2 * batch_size
    const int* __restrict__ map11,
    const int* __restrict__ map21,
    const int* __restrict__ map12,
    const int* __restrict__ map22,
    const int* __restrict__ diag11,
    const int* __restrict__ diag21,
    const int* __restrict__ diag12,
    const int* __restrict__ diag22,
    float* J_values  // 크기: J_nnz * batch_size
) {
    // 전체 스레드 인덱스 = batch_id * Y_nnz + k
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_elements * batch_size;

    if (idx >= total_elements) return;

    // 내가 몇 번째 배치의, 몇 번째 선로를 담당하는가?
    int batch_id = idx / n_elements;  // 0 ~ (batch_size-1)
    int k = idx % n_elements;         // 0 ~ (Y_nnz-1)

    // 메모리 오프셋 계산
    int v_offset = batch_id * (nbus * 2);  // 전압 배열 시작 위치
    int j_offset = batch_id * J_nnz;       // Jacobian 배열 시작 위치

    // Ybus 인덱스 (모든 배치가 동일)
    int i = row[k];  // from bus
    int j_bus = col[k];  // to bus

    // 복소수 연산
    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
    cuFloatComplex Vi = make_cuFloatComplex(V_real[v_offset + i*2], V_real[v_offset + i*2+1]);
    cuFloatComplex Vj = make_cuFloatComplex(V_real[v_offset + j_bus*2], V_real[v_offset + j_bus*2+1]);

    // 전류 주입: curr = Yij * Vj
    cuFloatComplex curr = cuCmulf(y, Vj);

    // 위상각 미분: term_va = -j * Vi * conj(Yij * Vj)
    cuFloatComplex term_va = cuCmulf(
        make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi)),
        cuConjf(curr)
    );

    // 전압크기 미분: term_vm = Vi * conj(Yij * Vj / |Vj|)
    float vj_norm = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_norm > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr)/vj_norm, cuCimagf(curr)/vj_norm)))
        : make_cuFloatComplex(0.0f, 0.0f);

    // Jacobian에 값 누적 (오프셋 적용, 매핑은 공유)
    if (map11[k] >= 0) atomicAdd(&J_values[j_offset + map11[k]], cuCrealf(term_va));
    if (map21[k] >= 0) atomicAdd(&J_values[j_offset + map21[k]], cuCimagf(term_va));
    if (map12[k] >= 0) atomicAdd(&J_values[j_offset + map12[k]], cuCrealf(term_vm));
    if (map22[k] >= 0) atomicAdd(&J_values[j_offset + map22[k]], cuCimagf(term_vm));

    // 대각 성분 보정
    if (diag11[i] >= 0) atomicAdd(&J_values[j_offset + diag11[i]], -cuCrealf(term_va));
    if (diag21[i] >= 0) atomicAdd(&J_values[j_offset + diag21[i]], -cuCimagf(term_va));

    float vi_norm = cuCabsf(Vi);
    if (vi_norm > 1e-6f) {
        cuFloatComplex term_vm2 = cuCmulf(
            make_cuFloatComplex(cuCrealf(Vi)/vi_norm, cuCimagf(Vi)/vi_norm),
            cuConjf(curr)
        );
        if (diag12[i] >= 0) atomicAdd(&J_values[j_offset + diag12[i]], cuCrealf(term_vm2));
        if (diag22[i] >= 0) atomicAdd(&J_values[j_offset + diag22[i]], cuCimagf(term_vm2));
    }
}


// ============================================================================
// Multi-Batch Jacobian 업데이트 함수 (호스트 측)
// ============================================================================

float NewtonCudaAccel::update_jacobian_batch(const void* V_ptr, int batch_size) {
    // 1. 메모리 재할당 필요 시 (최대 batch_size 증가 시에만)
    if (batch_size > max_batch_size_) {
        // 기존 메모리 해제
        if (d_V_batch_) cudaFree(d_V_batch_);
        if (d_J_batch_) cudaFree(d_J_batch_);

        // 새로 할당
        size_t v_bytes = nbus * 2 * batch_size * sizeof(float);
        size_t j_bytes = J_nnz * batch_size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_V_batch_, v_bytes));
        CUDA_CHECK(cudaMalloc(&d_J_batch_, j_bytes));

        max_batch_size_ = batch_size;
        CUDA_LOG("[CUDA] Multi-batch 메모리 할당: batch_size=%d (V: %.2f MB, J: %.2f MB)\n",
               batch_size, v_bytes / 1024.0 / 1024.0, j_bytes / 1024.0 / 1024.0);
    }

    // 2. 입력 V 준비 (CPU): 동일 데이터를 batch_size만큼 반복
    const double* V_double = static_cast<const double*>(V_ptr);
    std::vector<float> h_V_batch(nbus * 2 * batch_size);

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < nbus * 2; ++i) {
            h_V_batch[b * nbus * 2 + i] = static_cast<float>(V_double[i]);
        }
    }

    // 3. 데이터 업로드 및 초기화
    size_t v_bytes = nbus * 2 * batch_size * sizeof(float);
    size_t j_bytes = J_nnz * batch_size * sizeof(float);

    CUDA_CHECK(cudaMemcpy(d_V_batch_, h_V_batch.data(), v_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J_batch_, 0, j_bytes));

    // 4. 커널 실행 (시간 측정)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_threads = Y_nnz * batch_size;
    int block = 256;
    int grid = (total_threads + block - 1) / block;

    cudaEventRecord(start);
    update_jacobian_batch_kernel_fp32<<<grid, block>>>(
        Y_nnz, batch_size, nbus, J_nnz,
        d_G_f, d_B_f,
        d_Y_row, d_Y_col,
        d_V_batch_,
        d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
        d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
        d_J_batch_
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());

    // 5. 시간 측정
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    // 6. 정리 (이벤트만 해제, 메모리는 재사용)
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return kernel_time_ms;
}


// ============================================================================
// atomicAdd 제거 버전 커널 (병목 정량화용 - 결과는 틀림!)
// ============================================================================
//
// 목적: atomicAdd 오버헤드가 얼마나 되는지 정량화
// 주의: 이 커널의 결과는 정확하지 않음! 순수 성능 측정용
//
// ============================================================================

__global__ void update_jacobian_batch_kernel_no_atomic(
    int n_elements,
    int batch_size,
    int nbus,
    int J_nnz,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const int* __restrict__ row,
    const int* __restrict__ col,
    const float* __restrict__ V_real,
    const int* __restrict__ map11,
    const int* __restrict__ map21,
    const int* __restrict__ map12,
    const int* __restrict__ map22,
    const int* __restrict__ diag11,
    const int* __restrict__ diag21,
    const int* __restrict__ diag12,
    const int* __restrict__ diag22,
    float* J_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_elements * batch_size;

    if (idx >= total_elements) return;

    int batch_id = idx / n_elements;
    int k = idx % n_elements;

    int v_offset = batch_id * (nbus * 2);
    int j_offset = batch_id * J_nnz;

    int i = row[k];
    int j_bus = col[k];

    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
    cuFloatComplex Vi = make_cuFloatComplex(V_real[v_offset + i*2], V_real[v_offset + i*2+1]);
    cuFloatComplex Vj = make_cuFloatComplex(V_real[v_offset + j_bus*2], V_real[v_offset + j_bus*2+1]);

    cuFloatComplex curr = cuCmulf(y, Vj);

    cuFloatComplex term_va = cuCmulf(
        make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi)),
        cuConjf(curr)
    );

    float vj_norm = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_norm > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr)/vj_norm, cuCimagf(curr)/vj_norm)))
        : make_cuFloatComplex(0.0f, 0.0f);

    // ★★★ atomicAdd 대신 일반 대입 (덮어쓰기) ★★★
    // 결과는 틀리지만, 메모리 쓰기 패턴은 동일
    if (map11[k] >= 0) J_values[j_offset + map11[k]] = cuCrealf(term_va);
    if (map21[k] >= 0) J_values[j_offset + map21[k]] = cuCimagf(term_va);
    if (map12[k] >= 0) J_values[j_offset + map12[k]] = cuCrealf(term_vm);
    if (map22[k] >= 0) J_values[j_offset + map22[k]] = cuCimagf(term_vm);

    if (diag11[i] >= 0) J_values[j_offset + diag11[i]] = -cuCrealf(term_va);
    if (diag21[i] >= 0) J_values[j_offset + diag21[i]] = -cuCimagf(term_va);

    float vi_norm = cuCabsf(Vi);
    if (vi_norm > 1e-6f) {
        cuFloatComplex term_vm2 = cuCmulf(
            make_cuFloatComplex(cuCrealf(Vi)/vi_norm, cuCimagf(Vi)/vi_norm),
            cuConjf(curr)
        );
        if (diag12[i] >= 0) J_values[j_offset + diag12[i]] = cuCrealf(term_vm2);
        if (diag22[i] >= 0) J_values[j_offset + diag22[i]] = cuCimagf(term_vm2);
    }
}


// ============================================================================
// Warp-level Reduction 커널 (atomicAdd 횟수 감소)
// ============================================================================
//
// 아이디어: 같은 warp 내의 스레드들 중, 같은 Jacobian 위치에 쓰는 경우
//          warp shuffle로 먼저 합산 후, 대표 스레드만 atomicAdd
//
// 실제로는 같은 map 위치를 가진 스레드가 같은 warp에 있을 확률이 낮지만,
// 적어도 대각 요소(diag)는 같은 버스 i에 대해 여러 스레드가 기여하므로
// 부분적 최적화 가능
//
// ============================================================================

__global__ void update_jacobian_batch_kernel_warp_reduce(
    int n_elements,
    int batch_size,
    int nbus,
    int J_nnz,
    const float* __restrict__ G,
    const float* __restrict__ B,
    const int* __restrict__ row,
    const int* __restrict__ col,
    const float* __restrict__ V_real,
    const int* __restrict__ map11,
    const int* __restrict__ map21,
    const int* __restrict__ map12,
    const int* __restrict__ map22,
    const int* __restrict__ diag11,
    const int* __restrict__ diag21,
    const int* __restrict__ diag12,
    const int* __restrict__ diag22,
    float* J_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_elements * batch_size;

    if (idx >= total_elements) return;

    int batch_id = idx / n_elements;
    int k = idx % n_elements;
    int lane_id = threadIdx.x & 31;  // warp 내 lane (0-31)

    int v_offset = batch_id * (nbus * 2);
    int j_offset = batch_id * J_nnz;

    int i = row[k];
    int j_bus = col[k];

    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
    cuFloatComplex Vi = make_cuFloatComplex(V_real[v_offset + i*2], V_real[v_offset + i*2+1]);
    cuFloatComplex Vj = make_cuFloatComplex(V_real[v_offset + j_bus*2], V_real[v_offset + j_bus*2+1]);

    cuFloatComplex curr = cuCmulf(y, Vj);

    cuFloatComplex term_va = cuCmulf(
        make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi)),
        cuConjf(curr)
    );

    float vj_norm = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_norm > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr)/vj_norm, cuCimagf(curr)/vj_norm)))
        : make_cuFloatComplex(0.0f, 0.0f);

    // 비대각 요소: 그대로 atomicAdd (map이 다르므로 reduction 불가)
    if (map11[k] >= 0) atomicAdd(&J_values[j_offset + map11[k]], cuCrealf(term_va));
    if (map21[k] >= 0) atomicAdd(&J_values[j_offset + map21[k]], cuCimagf(term_va));
    if (map12[k] >= 0) atomicAdd(&J_values[j_offset + map12[k]], cuCrealf(term_vm));
    if (map22[k] >= 0) atomicAdd(&J_values[j_offset + map22[k]], cuCimagf(term_vm));

    // ★★★ 대각 요소: Warp-level reduction 시도 ★★★
    // 같은 warp 내에서 같은 버스 i를 가진 스레드들끼리 합산
    // (실제로는 효과가 제한적이지만, 실험 목적)

    int diag_idx11 = diag11[i];
    float diag_val11 = -cuCrealf(term_va);

    // Warp 내에서 같은 diag_idx를 가진 스레드 찾기
    unsigned int active_mask = __activemask();

    // 각 lane의 diag_idx를 broadcast하여 비교
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_idx = __shfl_down_sync(active_mask, diag_idx11, offset);
        float other_val = __shfl_down_sync(active_mask, diag_val11, offset);

        // 같은 인덱스면 합산
        if (other_idx == diag_idx11 && lane_id + offset < 32) {
            diag_val11 += other_val;
        }
    }

    // 가장 낮은 lane만 atomicAdd (간단한 구현: lane 0만 항상 write)
    // 실제로는 segment 시작점만 write해야 하지만, 복잡해서 일단 fallback
    if (diag_idx11 >= 0) atomicAdd(&J_values[j_offset + diag_idx11], diag_val11);

    // 나머지 대각 요소들은 그냥 atomicAdd (복잡도 때문에)
    if (diag21[i] >= 0) atomicAdd(&J_values[j_offset + diag21[i]], -cuCimagf(term_va));

    float vi_norm = cuCabsf(Vi);
    if (vi_norm > 1e-6f) {
        cuFloatComplex term_vm2 = cuCmulf(
            make_cuFloatComplex(cuCrealf(Vi)/vi_norm, cuCimagf(Vi)/vi_norm),
            cuConjf(curr)
        );
        if (diag12[i] >= 0) atomicAdd(&J_values[j_offset + diag12[i]], cuCrealf(term_vm2));
        if (diag22[i] >= 0) atomicAdd(&J_values[j_offset + diag22[i]], cuCimagf(term_vm2));
    }
}


// ============================================================================
// 벤치마크용 함수: 세 가지 커널 비교
// ============================================================================

float NewtonCudaAccel::benchmark_kernel_variants(const void* V_ptr, int batch_size, int variant) {
    // 메모리 할당 (필요시)
    if (batch_size > max_batch_size_) {
        if (d_V_batch_) cudaFree(d_V_batch_);
        if (d_J_batch_) cudaFree(d_J_batch_);

        size_t v_bytes = nbus * 2 * batch_size * sizeof(float);
        size_t j_bytes = J_nnz * batch_size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_V_batch_, v_bytes));
        CUDA_CHECK(cudaMalloc(&d_J_batch_, j_bytes));

        max_batch_size_ = batch_size;
    }

    // 입력 데이터 준비
    const double* V_double = static_cast<const double*>(V_ptr);
    std::vector<float> h_V_batch(nbus * 2 * batch_size);

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < nbus * 2; ++i) {
            h_V_batch[b * nbus * 2 + i] = static_cast<float>(V_double[i]);
        }
    }

    size_t v_bytes = nbus * 2 * batch_size * sizeof(float);
    size_t j_bytes = J_nnz * batch_size * sizeof(float);

    CUDA_CHECK(cudaMemcpy(d_V_batch_, h_V_batch.data(), v_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J_batch_, 0, j_bytes));

    // 커널 실행 (시간 측정)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_threads = Y_nnz * batch_size;
    int block = 256;
    int grid = (total_threads + block - 1) / block;

    cudaEventRecord(start);

    switch (variant) {
        case 0:  // 기본 atomicAdd 버전
            update_jacobian_batch_kernel_fp32<<<grid, block>>>(
                Y_nnz, batch_size, nbus, J_nnz,
                d_G_f, d_B_f, d_Y_row, d_Y_col, d_V_batch_,
                d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
                d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
                d_J_batch_
            );
            break;

        case 1:  // atomicAdd 제거 버전 (정확도 무시)
            update_jacobian_batch_kernel_no_atomic<<<grid, block>>>(
                Y_nnz, batch_size, nbus, J_nnz,
                d_G_f, d_B_f, d_Y_row, d_Y_col, d_V_batch_,
                d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
                d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
                d_J_batch_
            );
            break;

        case 2:  // Warp-level Reduction 버전
            update_jacobian_batch_kernel_warp_reduce<<<grid, block>>>(
                Y_nnz, batch_size, nbus, J_nnz,
                d_G_f, d_B_f, d_Y_row, d_Y_col, d_V_batch_,
                d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
                d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
                d_J_batch_
            );
            break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return kernel_time_ms;
}


// ============================================================================
// Multi-Batch 정확도 검증
// ============================================================================

bool NewtonCudaAccel::verify_batch_correctness(int batch_size) {
    if (batch_size < 2) {
        printf("[검증] batch_size가 2 미만이므로 검증 불필요\n");
        return true;
    }

    if (max_batch_size_ < batch_size) {
        printf("[검증 실패] 메모리가 할당되지 않음. update_jacobian_batch를 먼저 호출하세요.\n");
        return false;
    }

    // GPU에서 결과 다운로드
    std::vector<float> h_J_batch(J_nnz * batch_size);
    size_t j_bytes = J_nnz * batch_size * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_J_batch.data(), d_J_batch_, j_bytes, cudaMemcpyDeviceToHost));

    // Batch 0의 결과를 기준으로 삼음
    float* batch0 = h_J_batch.data();

    // 각 배치의 결과를 Batch 0과 비교 (상대 오차 기준)
    double max_relative_error = 0.0;
    int diff_count = 0;
    const double tolerance = 0.01;  // 상대 오차 1% 허용 (atomicAdd 비결정성 고려)

    for (int b = 1; b < batch_size; ++b) {
        float* batch_b = h_J_batch.data() + b * J_nnz;

        for (int i = 0; i < J_nnz; ++i) {
            double v0 = batch0[i];
            double vb = batch_b[i];
            double abs_diff = std::abs(vb - v0);

            // 상대 오차 계산 (값이 아주 작으면 절대 오차 사용)
            double reference = std::max(std::abs(v0), std::abs(vb));
            double relative_error = (reference > 1e-6) ? (abs_diff / reference) : abs_diff;

            max_relative_error = std::max(max_relative_error, relative_error);

            if (relative_error > tolerance) {
                diff_count++;
                if (diff_count <= 5) {  // 처음 5개만 출력
                    printf("[검증] Batch %d, idx %d: %.6e vs %.6e (상대오차: %.4f%%)\n",
                           b, i, vb, v0, relative_error * 100.0);
                }
            }
        }
    }

    bool passed = (diff_count == 0);

    if (passed) {
        printf("[검증 성공] 모든 배치 결과 일치 (최대 상대오차: %.4f%%)\n", max_relative_error * 100.0);
    } else {
        printf("[검증 실패] %d개 값이 허용 오차(%.1f%%)를 초과 (최대 상대오차: %.4f%%)\n",
               diff_count, tolerance * 100.0, max_relative_error * 100.0);
    }

    return passed;
}


// ============================================================================
// Batch API 구현
// ============================================================================

int NewtonCudaAccel::getBatchSizeGPU() const { return batch_size_gpu_; }

// ---- Batch 초기화 커널: V_cd → Va_d, Vm_d, V_f ----
__global__ void init_batch_V_kernel(
    const cuDoubleComplex* __restrict__ V_cd,  // [nbus * batch_size]
    double*  __restrict__ Va_d,                // [nbus * batch_size]
    double*  __restrict__ Vm_d,                // [nbus * batch_size]
    float*   __restrict__ V_f,                 // [nbus * 2 * batch_size]
    int nbus, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbus * batch_size;
    if (idx >= total) return;

    cuDoubleComplex v = V_cd[idx];
    double re = cuCreal(v), im = cuCimag(v);
    Va_d[idx] = atan2(im, re);
    Vm_d[idx] = sqrt(re*re + im*im);
    V_f[idx * 2]     = (float)re;
    V_f[idx * 2 + 1] = (float)im;
}


void NewtonCudaAccel::upload_batch_initial(
    const void* const* V_ptrs,
    const void* const* Sbus_ptrs,
    int batch_size
) {
    // 이전 배치 버퍼 해제
    if (d_V_cd_batch_)  { cudaFree(d_V_cd_batch_);  d_V_cd_batch_  = nullptr; }
    if (d_Va_d_batch_)  { cudaFree(d_Va_d_batch_);  d_Va_d_batch_  = nullptr; }
    if (d_Vm_d_batch_)  { cudaFree(d_Vm_d_batch_);  d_Vm_d_batch_  = nullptr; }
    if (d_V_f_batch_)   { cudaFree(d_V_f_batch_);   d_V_f_batch_   = nullptr; }
    if (d_Sbus_batch_)  { cudaFree(d_Sbus_batch_);  d_Sbus_batch_  = nullptr; }
    if (d_Ibus_batch_)  { cudaFree(d_Ibus_batch_);  d_Ibus_batch_  = nullptr; }
    if (d_F_batch_)     { cudaFree(d_F_batch_);      d_F_batch_     = nullptr; }
    if (d_dx_batch_)    { cudaFree(d_dx_batch_);     d_dx_batch_    = nullptr; }
    if (d_J_batch_new_) { cudaFree(d_J_batch_new_); d_J_batch_new_ = nullptr; }
    if (d_V_cd_ptrs_)   { cudaFree(d_V_cd_ptrs_);   d_V_cd_ptrs_   = nullptr; }
    if (d_Ibus_ptrs_)   { cudaFree(d_Ibus_ptrs_);   d_Ibus_ptrs_   = nullptr; }

    batch_size_gpu_ = batch_size;
    size_t nb_b = (size_t)nbus * batch_size;

    CUDA_CHECK(cudaMalloc(&d_V_cd_batch_,  nb_b * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Va_d_batch_,  nb_b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Vm_d_batch_,  nb_b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V_f_batch_,   nb_b * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Sbus_batch_,  nb_b * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Ibus_batch_,  nb_b * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_J_batch_new_, (size_t)J_nnz * batch_size * sizeof(float)));

    // V, Sbus 업로드
    for (int b = 0; b < batch_size; ++b) {
        CUDA_CHECK(cudaMemcpy(
            d_V_cd_batch_ + b * nbus, V_ptrs[b],
            nbus * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_Sbus_batch_ + b * nbus, Sbus_ptrs[b],
            nbus * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice
        ));
    }

    // Va_d, Vm_d, V_f 초기화 커널
    int block = 256, grid = ((int)nb_b + block - 1) / block;
    init_batch_V_kernel<<<grid, block>>>(
        d_V_cd_batch_, d_Va_d_batch_, d_Vm_d_batch_, d_V_f_batch_,
        nbus, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // SpMM 디스크립터 생성 (Ybus × V_batch → Ibus_batch, column-major)
    if (sp_V_mat_)    { cusparseDestroyDnMat(sp_V_mat_);   sp_V_mat_   = nullptr; }
    if (sp_Ibus_mat_) { cusparseDestroyDnMat(sp_Ibus_mat_); sp_Ibus_mat_ = nullptr; }
    if (d_spmm_buf_)  { cudaFree(d_spmm_buf_); d_spmm_buf_ = nullptr; spmm_buf_size_ = 0; }

    // V_batch: [nbus × batch_size] col-major, leading dimension = nbus
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &sp_V_mat_, nbus, batch_size, nbus,
        d_V_cd_batch_, CUDA_C_64F, CUSPARSE_ORDER_COL
    ));
    // Ibus_batch: [nbus × batch_size] col-major
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &sp_Ibus_mat_, nbus, batch_size, nbus,
        d_Ibus_batch_, CUDA_C_64F, CUSPARSE_ORDER_COL
    ));

    // SpMM 버퍼 크기 조회
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sp_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_Ybus_, sp_V_mat_, &beta, sp_Ibus_mat_,
            CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_buf_size_
        ));
        if (spmm_buf_size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmm_buf_, spmm_buf_size_));
        }
    }
    spmm_initialized_ = true;

    // normF GPU 버퍼
    if (d_normF_batch_) { cudaFree(d_normF_batch_); d_normF_batch_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_normF_batch_, batch_size * sizeof(double)));

    batch_initialized_ = true;
    CUDA_LOG("[CUDA] 배치 초기화 완료 (batch_size=%d, nbus=%d)\n", batch_size, nbus);
}


// ---- normF GPU reduction 커널 ----
// 각 배치 b에 대해 F[b*dimF .. (b+1)*dimF) 의 max(|F[i]|) 계산
// 블록 1개 = 배치 1개, blockDim.x >= dimF 가정 (dimF <= 65536)
__global__ void normF_reduction_kernel(
    const double* __restrict__ F,   // [dimF * batch_size]
    double* __restrict__ normF,     // [batch_size]
    int dimF
) {
    extern __shared__ double sdata[];
    int b   = blockIdx.x;  // 배치 인덱스
    int tid = threadIdx.x;

    // 각 스레드가 자신의 원소들의 max abs 계산
    double val = 0.0;
    for (int i = tid; i < dimF; i += blockDim.x) {
        double v = F[b * dimF + i];
        if (v < 0) v = -v;
        if (v > val) val = v;
    }
    sdata[tid] = val;
    __syncthreads();

    // tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) normF[b] = sdata[0];
}

// ---- Batch Mismatch 커널 ----
__global__ void mismatch_batch_kernel(
    const cuDoubleComplex* __restrict__ V,      // [nbus * batch_size]
    const cuDoubleComplex* __restrict__ Ibus,   // [nbus * batch_size]
    const cuDoubleComplex* __restrict__ Sbus,   // [nbus * batch_size]
    const int* __restrict__ pv,
    const int* __restrict__ pq,
    int npv, int npq, int nbus,
    int batch_size,
    double* F   // [(npv+2*npq) * batch_size]
) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int dimF = npv + 2 * npq;
    int total = dimF * batch_size;
    if (tid >= total) return;

    int b   = tid / dimF;
    int pos = tid % dimF;

    int offset = b * nbus;

    auto mis = [&](int bus) -> cuDoubleComplex {
        cuDoubleComplex vi = V[offset + bus];
        cuDoubleComplex ii = Ibus[offset + bus];
        cuDoubleComplex si = Sbus[offset + bus];
        double re = cuCreal(vi)*cuCreal(ii) + cuCimag(vi)*cuCimag(ii);
        double im = cuCimag(vi)*cuCreal(ii) - cuCreal(vi)*cuCimag(ii);
        return make_cuDoubleComplex(re - cuCreal(si), im - cuCimag(si));
    };

    int F_offset = b * dimF;
    if (pos < npv) {
        F[F_offset + pos] = cuCreal(mis(pv[pos]));
    } else if (pos < npv + npq) {
        int i = pos - npv;
        F[F_offset + pos] = cuCreal(mis(pq[i]));
    } else {
        int i = pos - npv - npq;
        F[F_offset + npv + npq + i] = cuCimag(mis(pq[i]));
    }
}


void NewtonCudaAccel::compute_mismatch_batch(
    const int* pv_ptr, const int* pq_ptr,
    int npv, int npq,
    double* F_batch_out, double* normF_out
) {
    int dimF = npv + 2 * npq;
    int bs   = batch_size_gpu_;

    // pv/pq GPU 업로드 (최초 1회)
    if (!mismatch_initialized_ || npv != npv_mis_ || npq != npq_mis_) {
        if (d_pv_mis_) cudaFree(d_pv_mis_);
        if (d_pq_mis_) cudaFree(d_pq_mis_);
        if (d_F_)      cudaFree(d_F_);
        CUDA_CHECK(cudaMalloc(&d_pv_mis_, npv  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pq_mis_, npq  * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_F_,      dimF * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_pv_mis_, pv_ptr, npv * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pq_mis_, pq_ptr, npq * sizeof(int), cudaMemcpyHostToDevice));
        npv_mis_ = npv; npq_mis_ = npq;
        mismatch_initialized_ = true;
    }

    // F 배치 버퍼 (필요시 할당)
    if (!d_F_batch_) {
        CUDA_CHECK(cudaMalloc(&d_F_batch_, (size_t)dimF * bs * sizeof(double)));
    }
    if (!d_dx_batch_) {
        CUDA_CHECK(cudaMalloc(&d_dx_batch_, (size_t)dimF * bs * sizeof(double)));
    }

    // ---- cuSPARSE SpMM: Ibus_batch = Ybus × V_batch (한 번에) ----
    // V_batch, Ibus_batch 디스크립터는 upload_batch_initial에서 생성됨
    // 전압이 update_voltage_batch_from_fp32_device로 갱신되면 d_V_cd_batch_ 포인터는 동일하게 유지됨
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        CUSPARSE_CHECK(cusparseSpMM(
            sp_handle_,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_Ybus_, sp_V_mat_, &beta, sp_Ibus_mat_,
            CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            d_spmm_buf_
        ));
    }

    // ---- Batch Mismatch 커널 (F 계산) ----
    {
        int block = 256, total = dimF * bs;
        int grid = (total + block - 1) / block;
        mismatch_batch_kernel<<<grid, block>>>(
            d_V_cd_batch_, d_Ibus_batch_, d_Sbus_batch_,
            d_pv_mis_, d_pq_mis_,
            npv, npq, nbus, bs,
            d_F_batch_
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- normF GPU reduction (배치당 블록 1개, shared memory 사용) ----
    {
        int block = 256;  // 256 threads per block
        int smem  = block * sizeof(double);
        normF_reduction_kernel<<<bs, block, smem>>>(
            d_F_batch_, d_normF_batch_, dimF
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // normF만 다운로드 (작은 배열, batch_size * 8 bytes)
    CUDA_CHECK(cudaMemcpy(normF_out, d_normF_batch_, bs * sizeof(double), cudaMemcpyDeviceToHost));

    // F_batch_out이 필요한 경우 다운로드 (null이면 스킵)
    if (F_batch_out) {
        CUDA_CHECK(cudaMemcpy(F_batch_out, d_F_batch_, (size_t)dimF * bs * sizeof(double), cudaMemcpyDeviceToHost));
    }
}


// ---- -F (FP64) → b (FP32) 변환 커널 ----
__global__ void negate_and_cast_fp32_kernel(
    const double* __restrict__ F,   // [dimF * batch_size] FP64
    float*        __restrict__ b,   // [dimF * batch_size] FP32 출력
    int total                       // dimF * batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    b[tid] = (float)(-F[tid]);
}

void NewtonCudaAccel::negate_F_to_fp32(float* d_b_out, int dimF) {
    int bs    = batch_size_gpu_;
    int total = dimF * bs;
    int block = 256;
    int grid  = (total + block - 1) / block;
    negate_and_cast_fp32_kernel<<<grid, block>>>(d_F_batch_, d_b_out, total);
    CUDA_CHECK(cudaGetLastError());
    // 동기화는 호출자 측에서 처리
}


// ---- Batch Jacobian 업데이트 커널 (V 배치 버전) ----
__global__ void update_jacobian_batch_v2_kernel(
    int Y_nnz, int batch_size, int nbus, int J_nnz,
    const float* __restrict__ G, const float* __restrict__ B,
    const int* __restrict__ row, const int* __restrict__ col,
    const float* __restrict__ V_batch,   // [nbus*2 * batch_size]
    const int* __restrict__ map11, const int* __restrict__ map21,
    const int* __restrict__ map12, const int* __restrict__ map22,
    const int* __restrict__ diag11, const int* __restrict__ diag21,
    const int* __restrict__ diag12, const int* __restrict__ diag22,
    float* J_batch  // [J_nnz * batch_size]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Y_nnz * batch_size) return;

    int b = idx / Y_nnz;
    int k = idx % Y_nnz;

    int v_off = b * nbus * 2;
    int j_off = b * J_nnz;

    int i = row[k], j_bus = col[k];

    cuFloatComplex y  = make_cuFloatComplex(G[k], B[k]);
    cuFloatComplex Vi = make_cuFloatComplex(V_batch[v_off + i*2], V_batch[v_off + i*2+1]);
    cuFloatComplex Vj = make_cuFloatComplex(V_batch[v_off + j_bus*2], V_batch[v_off + j_bus*2+1]);

    cuFloatComplex curr = cuCmulf(y, Vj);
    cuFloatComplex term_va = cuCmulf(
        make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi)), cuConjf(curr)
    );
    float vj_norm = cuCabsf(Vj);
    cuFloatComplex term_vm = (vj_norm > 1e-6f)
        ? cuCmulf(Vi, cuConjf(make_cuFloatComplex(cuCrealf(curr)/vj_norm, cuCimagf(curr)/vj_norm)))
        : make_cuFloatComplex(0.0f, 0.0f);

    if (map11[k] >= 0) atomicAdd(&J_batch[j_off + map11[k]], cuCrealf(term_va));
    if (map21[k] >= 0) atomicAdd(&J_batch[j_off + map21[k]], cuCimagf(term_va));
    if (map12[k] >= 0) atomicAdd(&J_batch[j_off + map12[k]], cuCrealf(term_vm));
    if (map22[k] >= 0) atomicAdd(&J_batch[j_off + map22[k]], cuCimagf(term_vm));
    if (diag11[i] >= 0) atomicAdd(&J_batch[j_off + diag11[i]], -cuCrealf(term_va));
    if (diag21[i] >= 0) atomicAdd(&J_batch[j_off + diag21[i]], -cuCimagf(term_va));

    float vi_norm = cuCabsf(Vi);
    if (vi_norm > 1e-6f) {
        cuFloatComplex term_vm2 = cuCmulf(
            make_cuFloatComplex(cuCrealf(Vi)/vi_norm, cuCimagf(Vi)/vi_norm), cuConjf(curr)
        );
        if (diag12[i] >= 0) atomicAdd(&J_batch[j_off + diag12[i]], cuCrealf(term_vm2));
        if (diag22[i] >= 0) atomicAdd(&J_batch[j_off + diag22[i]], cuCimagf(term_vm2));
    }
}


float* NewtonCudaAccel::update_jacobian_batch_no_upload() {
    int bs = batch_size_gpu_;
    CUDA_CHECK(cudaMemset(d_J_batch_new_, 0, (size_t)J_nnz * bs * sizeof(float)));

    int total = Y_nnz * bs;
    int block = 256, grid = (total + block - 1) / block;
    update_jacobian_batch_v2_kernel<<<grid, block>>>(
        Y_nnz, bs, nbus, J_nnz,
        d_G_f, d_B_f, d_Y_row, d_Y_col,
        d_V_f_batch_,
        d_mapJ11, d_mapJ21, d_mapJ12, d_mapJ22,
        d_diagMapJ11, d_diagMapJ21, d_diagMapJ12, d_diagMapJ22,
        d_J_batch_new_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return d_J_batch_new_;
}


// ---- Batch UpdateV 커널 ----
__global__ void update_voltage_batch_kernel(
    double* __restrict__ Va,   // [nbus * batch_size]
    double* __restrict__ Vm,   // [nbus * batch_size]
    cuDoubleComplex* __restrict__ V_cd,  // [nbus * batch_size]
    float*  __restrict__ V_f,  // [nbus*2 * batch_size]
    const double* __restrict__ dx,  // [dimF * batch_size]
    const int* __restrict__ pv, const int* __restrict__ pq,
    int npv, int npq, int nbus, int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dimF = npv + 2 * npq;
    if (tid >= dimF * batch_size) return;

    int b   = tid / dimF;
    int pos = tid % dimF;
    int v_off = b * nbus;

    if (pos < npv) {
        Va[v_off + pv[pos]] += dx[b * dimF + pos];
    } else if (pos < npv + npq) {
        int i = pos - npv;
        Va[v_off + pq[i]] += dx[b * dimF + pos];
    } else {
        int i = pos - npv - npq;
        Vm[v_off + pq[i]] += dx[b * dimF + pos];
    }
}

__global__ void reconstruct_V_batch_kernel(
    const double* __restrict__ Va,
    const double* __restrict__ Vm,
    cuDoubleComplex* __restrict__ V_cd,
    float* __restrict__ V_f,
    int nbus, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbus * batch_size) return;

    double re = Vm[idx] * cos(Va[idx]);
    double im = Vm[idx] * sin(Va[idx]);
    V_cd[idx] = make_cuDoubleComplex(re, im);
    V_f[idx * 2]     = (float)re;
    V_f[idx * 2 + 1] = (float)im;
}


void NewtonCudaAccel::update_voltage_batch(
    const double* dx_batch,
    const int* pv_ptr, const int* pq_ptr,
    int npv, int npq
) {
    int bs   = batch_size_gpu_;
    int dimF = npv + 2 * npq;

    // pv/pq 인덱스 (공유)
    if (npv != npv_upd_ || npq != npq_upd_) {
        if (d_pv_upd_) cudaFree(d_pv_upd_);
        if (d_pq_upd_) cudaFree(d_pq_upd_);
        CUDA_CHECK(cudaMalloc(&d_pv_upd_, npv * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pq_upd_, npq * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_pv_upd_, pv_ptr, npv * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pq_upd_, pq_ptr, npq * sizeof(int), cudaMemcpyHostToDevice));
        npv_upd_ = npv; npq_upd_ = npq;
    }

    // dx 배치 업로드
    CUDA_CHECK(cudaMemcpy(d_dx_batch_, dx_batch, (size_t)dimF * bs * sizeof(double), cudaMemcpyHostToDevice));

    // Va/Vm 업데이트
    int total = dimF * bs, block = 256, grid = (total + block - 1) / block;
    update_voltage_batch_kernel<<<grid, block>>>(
        d_Va_d_batch_, d_Vm_d_batch_, d_V_cd_batch_, d_V_f_batch_,
        d_dx_batch_, d_pv_upd_, d_pq_upd_,
        npv, npq, nbus, bs
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // V 재구성
    int total2 = nbus * bs, grid2 = (total2 + block - 1) / block;
    reconstruct_V_batch_kernel<<<grid2, block>>>(
        d_Va_d_batch_, d_Vm_d_batch_, d_V_cd_batch_, d_V_f_batch_,
        nbus, bs
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ---- Batch UpdateV (Device FP32 직접 수신) — 2-pass 커널 ----
// Pass 1: FP32 dx scatter → Va/Vm (dimF * batch_size 스레드)
// Pass 2: Va/Vm → V_cd/V_f reconstruct (nbus * batch_size 스레드)
// 두 pass를 독립 커널으로 유지하되 sync는 pass 2 후 1회만

__global__ void update_V_scatter_fp32_kernel(
    double* __restrict__ Va,
    double* __restrict__ Vm,
    const float*  __restrict__ dx_f,
    const int* __restrict__ pv, const int* __restrict__ pq,
    int npv, int npq, int nbus, int batch_size
) {
    int dimF = npv + 2 * npq;
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dimF * batch_size) return;

    int b   = tid / dimF;
    int pos = tid % dimF;
    int v_off = b * nbus;
    double delta = (double)dx_f[tid];

    if (pos < npv) {
        Va[v_off + pv[pos]] += delta;
    } else if (pos < npv + npq) {
        Va[v_off + pq[pos - npv]] += delta;
    } else {
        Vm[v_off + pq[pos - npv - npq]] += delta;
    }
}

void NewtonCudaAccel::update_voltage_batch_from_fp32_device(
    const float* d_x_f,
    const int* pv_ptr, const int* pq_ptr,
    int npv, int npq
) {
    int bs   = batch_size_gpu_;
    int dimF = npv + 2 * npq;

    // pv/pq 인덱스 (최초 호출 또는 크기 변경 시 GPU 업로드)
    if (npv != npv_upd_ || npq != npq_upd_) {
        if (d_pv_upd_) cudaFree(d_pv_upd_);
        if (d_pq_upd_) cudaFree(d_pq_upd_);
        CUDA_CHECK(cudaMalloc(&d_pv_upd_, npv * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pq_upd_, npq * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_pv_upd_, pv_ptr, npv * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pq_upd_, pq_ptr, npq * sizeof(int), cudaMemcpyHostToDevice));
        npv_upd_ = npv; npq_upd_ = npq;
    }

    // Pass 1: FP32 dx → Va/Vm scatter (비동기, sync 없음)
    {
        int total = dimF * bs, block = 256, grid = (total + block - 1) / block;
        update_V_scatter_fp32_kernel<<<grid, block>>>(
            d_Va_d_batch_, d_Vm_d_batch_,
            d_x_f, d_pv_upd_, d_pq_upd_,
            npv, npq, nbus, bs
        );
        CUDA_CHECK(cudaGetLastError());
        // sync 없음 — pass 2가 같은 default stream에서 순서 보장
    }

    // Pass 2: Va/Vm → V_cd / V_f 재구성 + sync 1회
    {
        int total = nbus * bs, block = 256, grid = (total + block - 1) / block;
        reconstruct_V_batch_kernel<<<grid, block>>>(
            d_Va_d_batch_, d_Vm_d_batch_, d_V_cd_batch_, d_V_f_batch_,
            nbus, bs
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}


void NewtonCudaAccel::download_V_batch(void* V_batch_out) const {
    CUDA_CHECK(cudaMemcpy(
        V_batch_out, d_V_cd_batch_,
        (size_t)nbus * batch_size_gpu_ * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost
    ));
}
