/**
 * @file cudss_solver.cpp
 * @brief cuDSS 기반 GPU 희소 선형 시스템 솔버 구현
 *
 * Newton-Raphson 전력조류 계산을 위한 Jx=b 풀이
 * FP32 Mixed Precision으로 A10 GPU에서 64배 빠른 성능 달성
 */

#include "cudss_solver.hpp"

#ifdef USE_CUDSS

#include "spdlog/spdlog.h"
#include <algorithm>
#include <utility>

// ============================================================================
// 생성자 / 소멸자
// ============================================================================

CuDSSSolver::CuDSSSolver()
    : handle_(nullptr)
    , config_(nullptr)
    , solver_data_f_(nullptr)
    , mat_A_f_(nullptr)
    , mat_x_f_(nullptr)
    , mat_b_f_(nullptr)
    , solver_data_batch_(nullptr)
    , mat_A_batch_(nullptr)
    , mat_x_batch_(nullptr)
    , mat_b_batch_(nullptr)
    , solver_data_ubatch_(nullptr)
    , mat_A_ubatch_(nullptr)
    , mat_x_ubatch_(nullptr)
    , mat_b_ubatch_(nullptr)
    , d_rowPtr_(nullptr)
    , d_colInd_(nullptr)
    , d_perm_map_(nullptr)
    , d_val_f_(nullptr)
    , d_b_f_(nullptr)
    , d_x_f_(nullptr)
    , d_val_batch_(nullptr)
    , d_b_batch_(nullptr)
    , d_x_batch_(nullptr)
    , d_perm_map_batch_(nullptr)
    , d_val_ptrs_(nullptr)
    , d_b_ptrs_(nullptr)
    , d_x_ptrs_(nullptr)
    , rows_(0)
    , cols_(0)
    , nnz_(0)
    , batch_size_(0)
{
    // CUDA 이벤트 생성 (타이밍용)
    checkCuda(cudaEventCreate(&start_event_), "이벤트 생성 (start)");
    checkCuda(cudaEventCreate(&stop_event_), "이벤트 생성 (stop)");

    // cuDSS 핸들 생성
    checkCudss(cudssCreate(&handle_), "cudssCreate");

    // cuDSS 버전 출력
    int maj = 0, min = 0, patch = 0;
    checkCudss(cudssGetProperty(MAJOR_VERSION, &maj), "버전 조회");
    checkCudss(cudssGetProperty(MINOR_VERSION, &min), "버전 조회");
    checkCudss(cudssGetProperty(PATCH_LEVEL, &patch), "버전 조회");
    spdlog::info("cuDSS 초기화: 버전 {}.{}.{}", maj, min, patch);

    // cuDSS 설정 생성
    checkCudss(cudssConfigCreate(&config_), "cudssConfigCreate");

    // Reordering 알고리즘 설정 (DEFAULT가 가장 빠름)
    // DEFAULT: 224ms, ALG_2(Metis): 706ms - 3배 느림
    cudssAlgType_t reorder = CUDSS_ALG_DEFAULT;
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_REORDERING_ALG, &reorder, sizeof(reorder)),
        "Reordering 알고리즘 설정"
    );
}

CuDSSSolver::~CuDSSSolver() {
    freeBatchDeviceMemory();
    freeDeviceMemory();

    if (mat_A_batch_) cudssMatrixDestroy(mat_A_batch_);
    if (mat_x_batch_) cudssMatrixDestroy(mat_x_batch_);
    if (mat_b_batch_) cudssMatrixDestroy(mat_b_batch_);
    if (solver_data_batch_) cudssDataDestroy(handle_, solver_data_batch_);

    if (mat_A_ubatch_) cudssMatrixDestroy(mat_A_ubatch_);
    if (mat_x_ubatch_) cudssMatrixDestroy(mat_x_ubatch_);
    if (mat_b_ubatch_) cudssMatrixDestroy(mat_b_ubatch_);
    if (solver_data_ubatch_) cudssDataDestroy(handle_, solver_data_ubatch_);

    if (mat_A_f_) cudssMatrixDestroy(mat_A_f_);
    if (mat_x_f_) cudssMatrixDestroy(mat_x_f_);
    if (mat_b_f_) cudssMatrixDestroy(mat_b_f_);
    if (solver_data_f_) cudssDataDestroy(handle_, solver_data_f_);

    if (config_) cudssConfigDestroy(config_);
    if (handle_) cudssDestroy(handle_);

    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}


// ============================================================================
// 에러 체크
// ============================================================================

void CuDSSSolver::checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::string error_msg = std::string(msg) + ": " + cudaGetErrorString(err);
        spdlog::error("CUDA 에러: {}", error_msg);
        throw std::runtime_error(error_msg);
    }
}

void CuDSSSolver::checkCudss(cudssStatus_t status, const char* msg) {
    if (status != CUDSS_STATUS_SUCCESS) {
        std::string error_msg = std::string(msg) + ": status=" + std::to_string(status);
        spdlog::error("cuDSS 에러: {}", error_msg);
        throw std::runtime_error(error_msg);
    }
}


// ============================================================================
// GPU 메모리 관리
// ============================================================================

void CuDSSSolver::allocateDeviceMemory() {
    if (allocated_) {
        freeDeviceMemory();
    }

    spdlog::info("GPU 메모리 할당: 값 {}B, 인덱스 {}B",
                 nnz_ * sizeof(float), (rows_ + 1 + nnz_) * sizeof(int64_t));

    // CSR 구조 (int64_t)
    checkCuda(cudaMalloc(&d_rowPtr_, (rows_ + 1) * sizeof(int64_t)), "d_rowPtr 할당");
    checkCuda(cudaMalloc(&d_colInd_, nnz_ * sizeof(int64_t)), "d_colInd 할당");
    checkCuda(cudaMalloc(&d_perm_map_, nnz_ * sizeof(int64_t)), "d_perm_map 할당");

    // FP32 버퍼
    checkCuda(cudaMalloc(&d_val_f_, nnz_ * sizeof(float)), "d_val_f 할당");
    checkCuda(cudaMalloc(&d_b_f_, rows_ * sizeof(float)), "d_b_f 할당");
    checkCuda(cudaMalloc(&d_x_f_, cols_ * sizeof(float)), "d_x_f 할당");

    allocated_ = true;
}

void CuDSSSolver::freeDeviceMemory() {
    if (!allocated_) return;

    if (d_rowPtr_) { cudaFree(d_rowPtr_); d_rowPtr_ = nullptr; }
    if (d_colInd_) { cudaFree(d_colInd_); d_colInd_ = nullptr; }
    if (d_perm_map_) { cudaFree(d_perm_map_); d_perm_map_ = nullptr; }
    if (d_val_f_) { cudaFree(d_val_f_); d_val_f_ = nullptr; }
    if (d_b_f_) { cudaFree(d_b_f_); d_b_f_ = nullptr; }
    if (d_x_f_) { cudaFree(d_x_f_); d_x_f_ = nullptr; }

    allocated_ = false;
}

void CuDSSSolver::allocateBatchDeviceMemory(int bs) {
    freeBatchDeviceMemory();
    batch_size_ = bs;

    size_t val_bytes = nnz_  * bs * sizeof(float);
    size_t vec_bytes = rows_ * bs * sizeof(float);
    size_t perm_bytes= nnz_  * bs * sizeof(int64_t);

    checkCuda(cudaMalloc(&d_val_batch_,      val_bytes),  "d_val_batch 할당");
    checkCuda(cudaMalloc(&d_b_batch_,        vec_bytes),  "d_b_batch 할당");
    checkCuda(cudaMalloc(&d_x_batch_,        vec_bytes),  "d_x_batch 할당");
    checkCuda(cudaMalloc(&d_perm_map_batch_, perm_bytes), "d_perm_map_batch 할당");

    // 배치 permutation 맵 초기화: 각 배치 b의 d_perm_map_batch_[b*nnz_+i] = h_perm_map_[i]
    std::vector<int64_t> h_perm_batch(nnz_ * bs);
    for (int b = 0; b < bs; ++b) {
        for (int64_t i = 0; i < nnz_; ++i) {
            h_perm_batch[b * nnz_ + i] = h_perm_map_[i] + b * nnz_;
        }
    }
    checkCuda(cudaMemcpy(d_perm_map_batch_, h_perm_batch.data(), perm_bytes, cudaMemcpyHostToDevice),
              "perm_map_batch 업로드");

    // 포인터 배열 (GPU 측)
    checkCuda(cudaMalloc(&d_val_ptrs_, bs * sizeof(void*)), "d_val_ptrs 할당");
    checkCuda(cudaMalloc(&d_b_ptrs_,   bs * sizeof(void*)), "d_b_ptrs 할당");
    checkCuda(cudaMalloc(&d_x_ptrs_,   bs * sizeof(void*)), "d_x_ptrs 할당");

    // 호스트에서 포인터 배열 구성 후 업로드
    std::vector<void*> h_val_ptrs(bs), h_b_ptrs(bs), h_x_ptrs(bs);
    for (int b = 0; b < bs; ++b) {
        h_val_ptrs[b] = d_val_batch_ + b * nnz_;
        h_b_ptrs[b]   = d_b_batch_   + b * rows_;
        h_x_ptrs[b]   = d_x_batch_   + b * cols_;
    }
    checkCuda(cudaMemcpy(d_val_ptrs_, h_val_ptrs.data(), bs * sizeof(void*), cudaMemcpyHostToDevice), "val_ptrs 업로드");
    checkCuda(cudaMemcpy(d_b_ptrs_,   h_b_ptrs.data(),   bs * sizeof(void*), cudaMemcpyHostToDevice), "b_ptrs 업로드");
    checkCuda(cudaMemcpy(d_x_ptrs_,   h_x_ptrs.data(),   bs * sizeof(void*), cudaMemcpyHostToDevice), "x_ptrs 업로드");

    allocated_batch_ = true;
    spdlog::info("Batch GPU 메모리 할당 완료 (batch_size={}, val={:.2f}MB, vec={:.2f}MB)",
                 bs, val_bytes/1e6, vec_bytes/1e6);
}

void CuDSSSolver::freeBatchDeviceMemory() {
    if (!allocated_batch_) return;

    if (d_val_batch_)      { cudaFree(d_val_batch_);      d_val_batch_      = nullptr; }
    if (d_b_batch_)        { cudaFree(d_b_batch_);        d_b_batch_        = nullptr; }
    if (d_x_batch_)        { cudaFree(d_x_batch_);        d_x_batch_        = nullptr; }
    if (d_perm_map_batch_) { cudaFree(d_perm_map_batch_); d_perm_map_batch_ = nullptr; }
    if (d_val_ptrs_)       { cudaFree(d_val_ptrs_);       d_val_ptrs_       = nullptr; }
    if (d_b_ptrs_)         { cudaFree(d_b_ptrs_);         d_b_ptrs_         = nullptr; }
    if (d_x_ptrs_)         { cudaFree(d_x_ptrs_);         d_x_ptrs_         = nullptr; }

    allocated_batch_ = false;
    batch_size_ = 0;
}


// ============================================================================
// Eigen → CSR 변환
// ============================================================================
//
// Eigen은 기본적으로 CSC (Column-major) 포맷 사용
// cuDSS는 CSR (Row-major) 포맷 필요
// 변환 과정:
//   1. 각 행의 non-zero 개수 카운트
//   2. rowPtr 구축 (누적합)
//   3. colInd, val 채우기 (열 인덱스 정렬)
//   4. Eigen 순서 → CSR 순서 매핑 테이블 생성
//
// ============================================================================

void CuDSSSolver::convertEigenToCSR(const nr_data::JacobianType& J) {
    rows_ = J.rows();
    cols_ = J.cols();
    nnz_ = J.nonZeros();

    spdlog::info("Eigen → CSR 변환: {}x{}, nnz={}", rows_, cols_, nnz_);

    h_rowPtr_.resize(rows_ + 1);
    h_colInd_.resize(nnz_);
    h_val_.resize(nnz_);

    // 각 행의 non-zero 개수 카운트
    std::vector<int> row_counts(rows_, 0);
    for (int k = 0; k < J.outerSize(); ++k) {
        for (nr_data::JacobianType::InnerIterator it(J, k); it; ++it) {
            row_counts[it.row()]++;
        }
    }

    // rowPtr 구축 (누적합)
    h_rowPtr_[0] = 0;
    for (int64_t i = 0; i < rows_; ++i) {
        h_rowPtr_[i + 1] = h_rowPtr_[i] + row_counts[i];
    }

    // colInd, val 채우기
    std::fill(row_counts.begin(), row_counts.end(), 0);
    for (int k = 0; k < J.outerSize(); ++k) {
        for (nr_data::JacobianType::InnerIterator it(J, k); it; ++it) {
            int row = it.row();
            int64_t pos = h_rowPtr_[row] + row_counts[row];
            h_colInd_[pos] = it.col();
            h_val_[pos] = it.value();
            row_counts[row]++;
        }
    }

    // 각 행 내에서 열 인덱스 정렬 + 매핑 테이블 생성
    eigen_to_csr_.clear();
    eigen_to_csr_.resize(rows_);

    for (int64_t i = 0; i < rows_; ++i) {
        int64_t row_start = h_rowPtr_[i];
        int64_t row_end = h_rowPtr_[i + 1];
        int64_t row_len = row_end - row_start;

        if (row_len > 1) {
            // (col, val) 쌍으로 정렬
            std::vector<std::pair<int64_t, double>> row_data(row_len);
            for (int64_t j = 0; j < row_len; ++j) {
                row_data[j] = {h_colInd_[row_start + j], h_val_[row_start + j]};
            }

            std::sort(row_data.begin(), row_data.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

            for (int64_t j = 0; j < row_len; ++j) {
                h_colInd_[row_start + j] = row_data[j].first;
                h_val_[row_start + j] = row_data[j].second;
                eigen_to_csr_[i][static_cast<int>(row_data[j].first)] = row_start + j;
            }
        } else if (row_len == 1) {
            eigen_to_csr_[i][static_cast<int>(h_colInd_[row_start])] = row_start;
        }
    }

    spdlog::info("CSR 변환 완료 (열 인덱스 정렬됨)");
}


// ============================================================================
// analyzePattern: 희소 패턴 분석 (1회만 실행)
// ============================================================================

void CuDSSSolver::analyzePattern(const nr_data::JacobianType& J) {
    spdlog::info("cuDSS analyzePattern 시작");

    // Eigen → CSR 변환
    convertEigenToCSR(J);

    // GPU 메모리 할당
    allocateDeviceMemory();

    // CSR 구조 업로드
    checkCuda(
        cudaMemcpy(d_rowPtr_, h_rowPtr_.data(), (rows_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice),
        "rowPtr 업로드"
    );
    checkCuda(
        cudaMemcpy(d_colInd_, h_colInd_.data(), nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice),
        "colInd 업로드"
    );

    // Permutation 맵 생성 및 업로드
    // h_perm_map[eigen_idx] = csr_idx
    std::vector<int64_t> h_perm_map(nnz_);
    int64_t eigen_idx = 0;
    for (int k = 0; k < J.outerSize(); ++k) {
        for (nr_data::JacobianType::InnerIterator it(J, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            auto& row_map = eigen_to_csr_[row];
            auto it_map = row_map.find(col);
            if (it_map != row_map.end()) {
                h_perm_map[eigen_idx] = it_map->second;
            } else {
                h_perm_map[eigen_idx] = -1;  // 발생하면 안됨
            }
            ++eigen_idx;
        }
    }

    checkCuda(
        cudaMemcpy(d_perm_map_, h_perm_map.data(), nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice),
        "permutation 맵 업로드"
    );
    h_perm_map_ = h_perm_map;  // 배치 초기화 시 재사용
    spdlog::info("Permutation 맵 업로드 완료 ({}개)", nnz_);

    // =========================================================================
    // FP32 cuDSS 설정
    // =========================================================================

    spdlog::info("FP32 Mixed Precision 솔버 설정 (A10: FP32=31.2 TFLOPS, FP64=0.49 TFLOPS)");

    // cuDSS 데이터 객체 생성
    checkCudss(cudssDataCreate(handle_, &solver_data_f_), "cudssDataCreate (FP32)");

    // 행렬 래퍼 생성 (CSR 포맷, FP32 값)
    checkCudss(
        cudssMatrixCreateCsr(
            &mat_A_f_,
            rows_, cols_, nnz_,
            d_rowPtr_, nullptr, d_colInd_, d_val_f_,
            CUDA_R_64I,   // 인덱스: int64_t
            CUDA_R_32F,   // 값: float (FP32)
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO
        ),
        "행렬 A 래퍼 생성 (FP32)"
    );

    // 해 벡터 래퍼
    checkCudss(
        cudssMatrixCreateDn(&mat_x_f_, cols_, 1, cols_, d_x_f_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR),
        "해 벡터 x 래퍼 생성 (FP32)"
    );

    // 우변 벡터 래퍼
    checkCudss(
        cudssMatrixCreateDn(&mat_b_f_, rows_, 1, rows_, d_b_f_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR),
        "우변 벡터 b 래퍼 생성 (FP32)"
    );

    // Symbolic Factorization (Analysis)
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, solver_data_f_, mat_A_f_, mat_x_f_, mat_b_f_),
        "cudssExecute ANALYSIS (FP32)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);

    float analysis_ms = 0.0f;
    cudaEventElapsedTime(&analysis_ms, start_event_, stop_event_);
    spdlog::info("cuDSS FP32 Analysis 완료: {:.3f} ms", analysis_ms);

    analyzed_ = true;
    first_factorize_f_ = true;  // 새 패턴 분석 시 리셋
}


// ============================================================================
// analyzePatternBatch: Uniform Batch cuDSS 설정 (1회만)
// ============================================================================

void CuDSSSolver::analyzePatternBatch(const nr_data::JacobianType& J, int batch_size) {
    // 단일 패턴 분석이 안 됐으면 먼저 실행
    if (!analyzed_) {
        analyzePattern(J);
    }

    // 배치 GPU 메모리 할당
    allocateBatchDeviceMemory(batch_size);

    // =========================================================================
    // cuDSS Uniform Batch 설정
    // =========================================================================

    // 1. Batch 솔버 데이터 생성
    checkCudss(cudssDataCreate(handle_, &solver_data_batch_), "cudssDataCreate (Batch)");

    // 2. UBATCH_SIZE 설정 (int형 사용 — int64_t는 INVALID_VALUE 오류 발생)
    int bs_int = batch_size;
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)),
        "UBATCH_SIZE 설정"
    );

    // 3. Batch CSR 행렬 래퍼 생성
    // rowStart/rowEnd/colIndices는 모든 배치가 공유 (Uniform Batch)
    // values는 배치별로 다름 → d_val_ptrs_ 배열 사용
    checkCudss(
        cudssMatrixCreateBatchCsr(
            &mat_A_batch_,
            batch_size,           // batchCount
            &rows_,               // nrows (공유)
            &cols_,               // ncols (공유)
            &nnz_,                // nnz (공유)
            (void**)&d_rowPtr_,   // rowStart (단일, 공유)
            nullptr,              // rowEnd (nullptr → rowStart[i+1] 사용)
            (void**)&d_colInd_,   // colIndices (단일, 공유)
            d_val_ptrs_,          // values (배치별 포인터 배열)
            CUDA_R_64I,
            CUDA_R_32F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO
        ),
        "Batch CSR 행렬 생성"
    );

    // 4. Batch Dense 벡터 래퍼 (b, x)
    checkCudss(
        cudssMatrixCreateBatchDn(
            &mat_b_batch_,
            batch_size,
            &rows_,               // nrows
            nullptr,              // ncols (1로 간주)
            &rows_,               // ld
            d_b_ptrs_,            // values 배열
            CUDA_R_64I,
            CUDA_R_32F,
            CUDSS_LAYOUT_COL_MAJOR
        ),
        "Batch Dense b 벡터 생성"
    );

    checkCudss(
        cudssMatrixCreateBatchDn(
            &mat_x_batch_,
            batch_size,
            &cols_,
            nullptr,
            &cols_,
            d_x_ptrs_,
            CUDA_R_64I,
            CUDA_R_32F,
            CUDSS_LAYOUT_COL_MAJOR
        ),
        "Batch Dense x 벡터 생성"
    );

    // 5. Batch Analysis (Symbolic Factorization)
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, solver_data_batch_,
                     mat_A_batch_, mat_x_batch_, mat_b_batch_),
        "cudssExecute ANALYSIS (Batch)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event_, stop_event_);
    spdlog::info("cuDSS Batch Analysis 완료 (batch_size={}, {:.3f} ms)", batch_size, ms);

    // UBATCH_SIZE를 다시 0으로 리셋 (단일 케이스 동작에 영향 주지 않도록)
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));

    analyzed_batch_ = true;
}


// ============================================================================
// FP32 Permutation 커널 (외부 정의)
// ============================================================================

extern "C" void launchPermutationKernelFP32(
    const float* d_src,
    float* d_dst,
    const int64_t* d_perm,
    int64_t nnz
);

extern "C" void launchPermutationBatchKernelFP32(
    const float* d_src,
    float* d_dst,
    const int64_t* d_perm_batch,
    int64_t nnz,
    int batch_size
);

void CuDSSSolver::applyPermutationFP32(float* d_eigen_values_f) {
    if (!analyzed_) {
        throw std::runtime_error("analyzePattern()을 먼저 호출해야 합니다");
    }

    // CSR 버퍼 초기화
    checkCuda(cudaMemset(d_val_f_, 0, nnz_ * sizeof(float)), "d_val_f 초기화");

    // Eigen 순서 → CSR 순서 재배열 (GPU 커널)
    launchPermutationKernelFP32(d_eigen_values_f, d_val_f_, d_perm_map_, nnz_);
}


// ============================================================================
// applyPermutationBatchFP32: Batch Permutation
// ============================================================================

void CuDSSSolver::applyPermutationBatchFP32(float* d_eigen_vals_batch) {
    if (!analyzed_batch_) {
        throw std::runtime_error("analyzePatternBatch()를 먼저 호출해야 합니다");
    }
    // d_val_batch_ 초기화
    checkCuda(cudaMemset(d_val_batch_, 0, nnz_ * batch_size_ * sizeof(float)), "d_val_batch 초기화");

    // batch permutation 커널 실행
    launchPermutationBatchKernelFP32(
        d_eigen_vals_batch, d_val_batch_,
        d_perm_map_batch_,
        nnz_, batch_size_
    );
}


// ============================================================================
// factorizeBatchFP32: Batch LU 분해
// ============================================================================

void CuDSSSolver::factorizeBatchFP32() {
    if (!analyzed_batch_) {
        throw std::runtime_error("analyzePatternBatch()를 먼저 호출해야 합니다");
    }

    // UBATCH_SIZE 설정 (int형 사용)
    int bs_int = batch_size_;
    checkCudss(cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)), "UBATCH_SIZE 설정");

    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, solver_data_batch_,
                     mat_A_batch_, mat_x_batch_, mat_b_batch_),
        "cudssExecute FACTORIZATION (Batch)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&last_factorize_ms_, start_event_, stop_event_);

    // UBATCH_SIZE 리셋
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));
}


// ============================================================================
// solveBatchFP32: Batch Solve
// ============================================================================

void CuDSSSolver::solveBatchFP32(float* d_b_batch_in, float* d_x_batch_out) {
    if (!analyzed_batch_) {
        throw std::runtime_error("analyzePatternBatch()를 먼저 호출해야 합니다");
    }

    // b 벡터 복사 (d_b_batch_ 내부 버퍼로)
    if (d_b_batch_in != d_b_batch_) {
        checkCuda(cudaMemcpy(d_b_batch_, d_b_batch_in,
                             rows_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "b_batch 복사");
    }

    // UBATCH_SIZE 설정 (int형 사용)
    int bs_int = batch_size_;
    checkCudss(cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)), "UBATCH_SIZE 설정");

    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, solver_data_batch_,
                     mat_A_batch_, mat_x_batch_, mat_b_batch_),
        "cudssExecute SOLVE (Batch)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&last_solve_ms_, start_event_, stop_event_);

    // x 출력 복사
    if (d_x_batch_out != d_x_batch_) {
        checkCuda(cudaMemcpy(d_x_batch_out, d_x_batch_,
                             cols_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "x_batch 복사");
    }

    // UBATCH_SIZE 리셋
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));
}


// ============================================================================
// factorizeDirectGPU_FP32: 수치적 LU 분해
// ============================================================================
//
// 첫 호출: FACTORIZATION (symbolic + numeric)
// 이후:   REFACTORIZATION (numeric only, 더 빠름)
//
// ============================================================================

void CuDSSSolver::factorizeDirectGPU_FP32() {
    if (!analyzed_) {
        throw std::runtime_error("analyzePattern()을 먼저 호출해야 합니다");
    }

    // 원래 코드: 항상 FACTORIZATION만 사용 (REFACTORIZATION 없음)
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, solver_data_f_, mat_A_f_, mat_x_f_, mat_b_f_),
        "cudssExecute FACTORIZATION (FP32)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);

    cudaEventElapsedTime(&last_factorize_ms_, start_event_, stop_event_);
    spdlog::debug("cuDSS factorization (FP32): {:.3f} ms", last_factorize_ms_);

    factorized_f_ = true;
}


// ============================================================================
// solveFP32: 선형 시스템 풀이 Jx = b
// ============================================================================
//
// 입력: b (mismatch, FP64)
// 내부 처리:
//   1. b → FP32 변환
//   2. GPU로 업로드 (memcpy 시간 측정)
//   3. cuDSS Solve (FP32)
//   4. GPU에서 다운로드
//   5. FP32 → FP64 변환
// 출력: x (dx, FP64)
//
// ============================================================================

nr_data::VectorXd CuDSSSolver::solveFP32(const nr_data::VectorXd& b) {
    if (!factorized_f_) {
        throw std::runtime_error("factorizeDirectGPU_FP32()를 먼저 호출해야 합니다");
    }

    // FP64 → FP32 변환
    std::vector<float> h_b_f(rows_);
    for (int64_t i = 0; i < rows_; ++i) {
        h_b_f[i] = static_cast<float>(b(i));
    }

    // GPU로 업로드 (시간 측정)
    cudaEvent_t memcpy_start, memcpy_stop;
    cudaEventCreate(&memcpy_start);
    cudaEventCreate(&memcpy_stop);

    cudaEventRecord(memcpy_start);
    checkCuda(
        cudaMemcpy(d_b_f_, h_b_f.data(), rows_ * sizeof(float), cudaMemcpyHostToDevice),
        "RHS 벡터 업로드 (FP32)"
    );
    cudaEventRecord(memcpy_stop);
    cudaEventSynchronize(memcpy_stop);
    cudaEventElapsedTime(&last_memcpy_ms_, memcpy_start, memcpy_stop);

    cudaEventDestroy(memcpy_start);
    cudaEventDestroy(memcpy_stop);

    // Solve (FP32)
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, solver_data_f_, mat_A_f_, mat_x_f_, mat_b_f_),
        "cudssExecute SOLVE (FP32)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);

    cudaEventElapsedTime(&last_solve_ms_, start_event_, stop_event_);
    spdlog::debug("cuDSS solve (FP32): {:.3f} ms", last_solve_ms_);

    // GPU에서 다운로드
    std::vector<float> h_x_f(cols_);
    checkCuda(
        cudaMemcpy(h_x_f.data(), d_x_f_, cols_ * sizeof(float), cudaMemcpyDeviceToHost),
        "해 벡터 다운로드 (FP32)"
    );

    // FP32 → FP64 변환
    nr_data::VectorXd x(cols_);
    for (int64_t i = 0; i < cols_; ++i) {
        x(i) = static_cast<double>(h_x_f[i]);
    }

    return x;
}


// ============================================================================
// UBATCH v2: 선임 방식 (단일 CSR + flat buffer + UBATCH_SIZE)
// cudssMatrixCreateBatchCsr 대신 cudssMatrixCreateCsr 사용.
// d_val_batch_ (flat [batch_size * nnz]) 버퍼를 그대로 넘기고
// UBATCH_SIZE 설정만으로 cuDSS가 batch 처리.
// ============================================================================

void CuDSSSolver::analyzePatternUBatch(const nr_data::JacobianType& J, int batch_size) {
    if (!analyzed_) {
        analyzePattern(J);
    }
    allocateBatchDeviceMemory(batch_size);

    // 솔버 데이터 생성
    checkCudss(cudssDataCreate(handle_, &solver_data_ubatch_), "cudssDataCreate (UBatch)");

    // UBATCH_SIZE 설정
    int bs_int = batch_size;
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)),
        "UBATCH_SIZE 설정 (UBatch)"
    );

    // 단일 CSR 행렬 래퍼 — values는 flat [batch_size * nnz] 버퍼
    // cuDSS는 UBATCH_SIZE가 설정된 경우 이 버퍼를 batch_size개의 연속된 블록으로 해석
    checkCudss(
        cudssMatrixCreateCsr(
            &mat_A_ubatch_,
            rows_, cols_, nnz_,
            d_rowPtr_, nullptr, d_colInd_,
            (void*)d_val_batch_,  // flat [batch_size * nnz] FP32 버퍼
            CUDA_R_64I,
            CUDA_R_32F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO
        ),
        "UBatch CSR 행렬 생성"
    );

    // 단일 Dense 래퍼 — flat [rows_ * batch_size] 버퍼
    checkCudss(
        cudssMatrixCreateDn(
            &mat_b_ubatch_,
            rows_, 1, rows_,
            (void*)d_b_batch_,
            CUDA_R_32F,
            CUDSS_LAYOUT_COL_MAJOR
        ),
        "UBatch Dense b 생성"
    );
    checkCudss(
        cudssMatrixCreateDn(
            &mat_x_ubatch_,
            rows_, 1, rows_,
            (void*)d_x_batch_,
            CUDA_R_32F,
            CUDSS_LAYOUT_COL_MAJOR
        ),
        "UBatch Dense x 생성"
    );

    // Analysis (symbolic factorization)
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute ANALYSIS (UBatch)"
    );
    cudaDeviceSynchronize();

    // 첫 번째 Factorization (symbolic + numeric, 더미 값으로)
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute FACTORIZATION init (UBatch)"
    );
    cudaDeviceSynchronize();

    // UBATCH_SIZE 리셋 (다른 솔버에 영향 없도록)
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));

    analyzed_ubatch_ = true;
    factorized_ubatch_ = true;
    spdlog::info("UBatch 초기화 완료 (batch_size={}, 단일 CSR + flat buffer)", batch_size);
}


void CuDSSSolver::applyPermutationUBatchFP32(float* d_eigen_vals_batch) {
    if (!analyzed_ubatch_) {
        throw std::runtime_error("analyzePatternUBatch()를 먼저 호출해야 합니다");
    }
    // SeqBatch와 동일: d_val_batch_ (flat [batch_size * nnz]) 에 permutation 적용
    checkCuda(cudaMemset(d_val_batch_, 0, nnz_ * batch_size_ * sizeof(float)), "d_val_batch 초기화 (UBatch)");
    launchPermutationBatchKernelFP32(
        d_eigen_vals_batch, d_val_batch_,
        d_perm_map_batch_,
        nnz_, batch_size_
    );
}


void CuDSSSolver::factorizeAndSolveUBatchFP32(float* d_b_batch, float* d_x_batch) {
    if (!analyzed_ubatch_) {
        throw std::runtime_error("analyzePatternUBatch()를 먼저 호출해야 합니다");
    }

    // b 복사 (외부 버퍼 → 내부 d_b_batch_)
    if (d_b_batch != d_b_batch_) {
        checkCuda(cudaMemcpy(d_b_batch_, d_b_batch,
                             rows_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "b_batch 복사 (UBatch)");
    }

    // UBATCH_SIZE 설정
    int bs_int = batch_size_;
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)),
        "UBATCH_SIZE 설정 (UBatch factorize)"
    );

    // REFACTORIZATION: d_val_batch_ (flat) 기준으로 batch_size개 동시 처리
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_REFACTORIZATION, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute REFACTORIZATION (UBatch)"
    );

    // SOLVE
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute SOLVE (UBatch)"
    );

    // x 출력 복사 (내부 d_x_batch_ → 외부)
    if (d_x_batch != d_x_batch_) {
        checkCuda(cudaMemcpy(d_x_batch, d_x_batch_,
                             cols_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "x_batch 복사 (UBatch)");
    }

    // UBATCH_SIZE 리셋
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));
}


// ============================================================================
// Sequential Batch API
// cuDSS UBATCH(0.7.1)가 Factorization 상태를 올바르게 업데이트하지 못하는
// 버그가 있어, Sequential Batch로 fallback.
// 동일한 symbolic 구조(LU sparsity pattern)를 재사용하면서 N개를 순차 처리.
// ============================================================================

void CuDSSSolver::analyzePatternSeqBatch(const nr_data::JacobianType& J, int batch_size) {
    // 단일 패턴 분석 (symbolic 구조 확정)
    if (!analyzed_) {
        analyzePattern(J);
    }
    batch_size_ = batch_size;

    // 배치 val/b/x 버퍼 할당
    allocateBatchDeviceMemory(batch_size);

    analyzed_batch_ = true;
    spdlog::info("Sequential Batch 설정 완료 (batch_size={}, shared symbolic structure)", batch_size);
}

void CuDSSSolver::applyPermutationSeqBatchFP32(float* d_eigen_vals_batch) {
    if (!analyzed_batch_) {
        throw std::runtime_error("analyzePatternSeqBatch()를 먼저 호출해야 합니다");
    }
    // d_val_batch_ 초기화
    checkCuda(cudaMemset(d_val_batch_, 0, nnz_ * batch_size_ * sizeof(float)), "d_val_batch 초기화");

    // 배치 permutation 커널 실행
    launchPermutationBatchKernelFP32(
        d_eigen_vals_batch, d_val_batch_,
        d_perm_map_batch_,
        nnz_, batch_size_
    );
}

void CuDSSSolver::factorizeAndSolveSeqBatchFP32(float* d_b_batch, float* d_x_batch) {
    if (!analyzed_batch_) {
        throw std::runtime_error("analyzePatternSeqBatch()를 먼저 호출해야 합니다");
    }

    for (int b = 0; b < batch_size_; ++b) {
        // val[b] → d_val_f_ (단일 케이스 val 버퍼)
        checkCuda(
            cudaMemcpy(d_val_f_, d_val_batch_ + b * nnz_,
                       nnz_ * sizeof(float), cudaMemcpyDeviceToDevice),
            "val 배치→단일 복사"
        );

        // Factorize (REFACTORIZATION if already factorized once)
        cudssPhase_t phase = factorized_f_ ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
        checkCudss(
            cudssExecute(handle_, phase, config_, solver_data_f_,
                         mat_A_f_, mat_x_f_, mat_b_f_),
            "cudssExecute FACTORIZATION (SeqBatch)"
        );
        factorized_f_ = true;

        // b[b] → d_b_f_ (단일 케이스 b 버퍼)
        checkCuda(
            cudaMemcpy(d_b_f_, d_b_batch + b * rows_,
                       rows_ * sizeof(float), cudaMemcpyDeviceToDevice),
            "b 배치→단일 복사"
        );

        // Solve
        checkCudss(
            cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, solver_data_f_,
                         mat_A_f_, mat_x_f_, mat_b_f_),
            "cudssExecute SOLVE (SeqBatch)"
        );

        // d_x_f_ → x_batch[b]
        checkCuda(
            cudaMemcpy(d_x_batch + b * cols_, d_x_f_,
                       cols_ * sizeof(float), cudaMemcpyDeviceToDevice),
            "x 단일→배치 복사"
        );
    }
}

// ============================================================================
// 진단용 API 구현
// ============================================================================

void CuDSSSolver::setIRSteps(int n) {
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_IR_N_STEPS, &n, sizeof(n)),
        "IR_N_STEPS 설정"
    );
}

void CuDSSSolver::factorizeAndSolveUBatchFP32_timed(
    float* d_b_batch, float* d_x_batch,
    float& refact_ms, float& solve_ms)
{
    if (!analyzed_ubatch_) {
        throw std::runtime_error("analyzePatternUBatch()를 먼저 호출해야 합니다");
    }

    // b 복사 (외부 → 내부)
    if (d_b_batch != d_b_batch_) {
        checkCuda(cudaMemcpy(d_b_batch_, d_b_batch,
                             rows_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "b_batch 복사 (timed)");
    }

    int bs_int = batch_size_;
    checkCudss(
        cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &bs_int, sizeof(bs_int)),
        "UBATCH_SIZE 설정 (timed)"
    );

    // REFACTORIZATION 타이밍
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_REFACTORIZATION, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute REFACTORIZATION (timed)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&refact_ms, start_event_, stop_event_);

    // SOLVE 타이밍
    cudaEventRecord(start_event_);
    checkCudss(
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, solver_data_ubatch_,
                     mat_A_ubatch_, mat_x_ubatch_, mat_b_ubatch_),
        "cudssExecute SOLVE (timed)"
    );
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&solve_ms, start_event_, stop_event_);

    // x 출력 복사
    if (d_x_batch != d_x_batch_) {
        checkCuda(cudaMemcpy(d_x_batch, d_x_batch_,
                             cols_ * batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "x_batch 복사 (timed)");
    }

    // UBATCH_SIZE 리셋
    int zero_int = 0;
    cudssConfigSet(config_, CUDSS_CONFIG_UBATCH_SIZE, &zero_int, sizeof(zero_int));
}

#endif // USE_CUDSS
