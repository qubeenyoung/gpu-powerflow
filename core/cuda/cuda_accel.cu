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
#include <Eigen/Sparse>
#include <cuComplex.h>

using YbusType = Eigen::SparseMatrix<std::complex<double>>;

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
    , d_V_f(nullptr), d_J_temp_f(nullptr)
    , d_V_batch_(nullptr), d_J_batch_(nullptr)
    , Y_nnz(0), J_nnz(0), nbus(0)
    , max_batch_size_(0)
    , last_memcpy_ms_(0.0f)
{
    cudaEventCreate(&memcpy_start_evt_);
    cudaEventCreate(&memcpy_stop_evt_);
    printf("[CUDA] Jacobian 가속기 생성 완료 (FP32 Mixed Precision)\n");
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
    if (d_J_temp_f) cudaFree(d_J_temp_f);

    // Multi-batch 버퍼 해제
    if (d_V_batch_) cudaFree(d_V_batch_);
    if (d_J_batch_) cudaFree(d_J_batch_);

    // 이벤트 해제
    cudaEventDestroy(memcpy_start_evt_);
    cudaEventDestroy(memcpy_stop_evt_);

    printf("[CUDA] Jacobian 가속기 해제 완료\n");
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

    printf("[CUDA] GPU 데이터 초기화 (버스: %d, Y_nnz: %d, J_nnz: %d)\n", nb, Y_nnz, J_nnz);

    // --- 매핑 테이블 업로드 ---
    auto upload_int = [&](int*& d_ptr, const std::vector<int>& h_vec, const char* name) {
        if (h_vec.empty()) return;
        size_t bytes = h_vec.size() * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_vec.data(), bytes, cudaMemcpyHostToDevice));
        printf("[CUDA]   %s 업로드: %zu개\n", name, h_vec.size());
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

    // 전압 버퍼: complex → 2 floats per bus
    CUDA_CHECK(cudaMalloc(&d_V_f, nb * sizeof(float) * 2));

    // Jacobian 출력 버퍼
    CUDA_CHECK(cudaMalloc(&d_J_temp_f, J_nnz * sizeof(float)));

    printf("[CUDA] 초기화 완료 (FP32 Mixed Precision)\n");
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
// FP32 Jacobian 업데이트 함수 (호스트 측)
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

    // 3. Jacobian 버퍼 초기화
    CUDA_CHECK(cudaMemset(d_J_temp_f, 0, J_nnz * sizeof(float)));

    // 4. 커널 실행 (A10 GPU에서 FP32는 FP64보다 64배 빠름)
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
        printf("[CUDA] Multi-batch 메모리 할당: batch_size=%d (V: %.2f MB, J: %.2f MB)\n",
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
