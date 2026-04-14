#pragma once

#include "powerflow_linear_system.hpp"
#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cudss.h>
#include <cusolverDn.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t _st = (call); \
        if (_st != CUSOLVER_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuSOLVER error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - status=" + std::to_string(static_cast<int>(_st))); \
        } \
    } while (0)
#endif

namespace exp_20260409 {

// Per-iteration GPU timings returned by CuDssSchurRunner::factorize_and_solve().
// All *_sec fields are measured with CUDA events; a single cudaEventSynchronize
// is issued at the end of each call.  Residual / accuracy fields are NOT included
// — call download_solution() and compute accuracy separately after the loop.
struct SchurTimings {
    int32_t repeat_idx       = 0;
    int32_t schur_dim        = 0;
    double factorization_sec = 0.0;  // CUDSS_PHASE_FACTORIZATION (partial LU + Schur build)
    double schur_extract_sec = 0.0;  // cudssDataGet(CUDSS_DATA_SCHUR_MATRIX)
    double fwd_solve_sec     = 0.0;  // SOLVE_FWD_PERM | SOLVE_FWD
    double diag_sec          = 0.0;  // cudaMemcpyAsync x→b  (D=I transfer, SOLVE_DIAG equiv.)
    double getrf_sec         = 0.0;  // cuSOLVER Xgetrf — dense LU of Schur matrix
    double getrs_sec         = 0.0;  // cuSOLVER Xgetrs — dense triangular solve
    double schur_solve_sec   = 0.0;  // diag + getrf + getrs
    double bwd_solve_sec     = 0.0;  // SOLVE_BWD | SOLVE_BWD_PERM
    double solve_sec         = 0.0;  // fwd + schur_solve + bwd
    double total_sec         = 0.0;  // factorization + extract + solve
};

// CuDssSchurRunner — Schur complement mode for power-flow Newton linear systems.
//
// Solve sequence (GENERAL / LU matrix):
//   1. FACTORIZATION        — partial LU (J11) + build dense Schur complement S
//   2. cudssDataGet         — retrieve S into d_schur_mat_
//   3. SOLVE_FWD_PERM|FWD  — writes y1 and Schur RHS into d_x_
//   4. cudaMemcpyAsync      — d_x_ → d_rhs_  (SOLVE_DIAG equivalent, D=I)
//   5. cuSOLVER Xgetrf      — dense LU of S in-place in d_schur_mat_
//   6. cuSOLVER Xgetrs      — d_rhs_[n_pvpq:] = y2
//   7. SOLVE_BWD|BWD_PERM  — reads [y1;y2] from d_rhs_, writes full x to d_x_
//
// Design notes:
//   • Timing: CUDA events — one cudaEventSynchronize at end of factorize_and_solve().
//   • RHS reset: D2D copy from d_rhs_orig_ per iteration — no H2D transfer in the loop.
//   • download_solution() is separate — call once after all iterations.
class CuDssSchurRunner {
public:
    explicit CuDssSchurRunner(const PowerFlowLinearSystem& system, int32_t n_pq)
        : system_(system)
        , system_dim_(system.structure.dim)
        , matrix_nnz_(system.structure.nnz)
        , schur_dim_(n_pq)
        , h_values_f_(system.values.size())
        , h_x_(system.structure.dim, 0.0f)
    {
        convert_host_inputs_to_fp32();
        build_schur_index_vector();
        allocate_device_buffers();
        upload_static_inputs();
        create_cudss_objects();
        create_events();
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
    }

    ~CuDssSchurRunner()
    {
        destroy_events();
        if (cusolver_params_) cusolverDnDestroyParams(cusolver_params_);
        if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
        if (schur_mat_obj_)   cudssMatrixDestroy(schur_mat_obj_);
        if (d_schur_mat_)     cudaFree(d_schur_mat_);
        if (d_ipiv_)          cudaFree(d_ipiv_);
        if (d_info_)          cudaFree(d_info_);
        if (d_work_)          cudaFree(d_work_);
        if (h_work_)          std::free(h_work_);
        destroy_cudss_objects();
        free_device_buffers();
    }

    // Run ANALYSIS once.  Returns wall-clock seconds (includes one DeviceSynchronize).
    // Queries Schur shape, allocates d_schur_mat_, and sets up cuSOLVER workspace.
    // Must be called exactly once before factorize_and_solve().
    double analyze()
    {
        const auto t0 = std::chrono::steady_clock::now();
        CUDSS_CHECK(cudssExecute(
            handle_, CUDSS_PHASE_ANALYSIS,
            config_, data_,
            matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaDeviceSynchronize());
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        int64_t schur_shape[3] = {0};
        size_t size_written = 0;
        CUDSS_CHECK(cudssDataGet(handle_, data_, CUDSS_DATA_SCHUR_SHAPE,
                                 &schur_shape, sizeof(schur_shape), &size_written));
        if (schur_shape[0] != static_cast<int64_t>(schur_dim_)) {
            throw std::runtime_error(
                "cuDSS Schur shape mismatch: expected " + std::to_string(schur_dim_) +
                " but got " + std::to_string(schur_shape[0]));
        }

        CUDA_CHECK(cudaMalloc(&d_schur_mat_,
                              static_cast<size_t>(schur_dim_) * schur_dim_ * sizeof(float)));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &schur_mat_obj_,
            schur_dim_, schur_dim_, schur_dim_,
            d_schur_mat_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params_));
        size_t work_dev = 0, work_host = 0;
        CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
            cusolver_handle_, cusolver_params_,
            schur_dim_, schur_dim_,
            CUDA_R_32F, d_schur_mat_, schur_dim_,
            CUDA_R_32F, &work_dev, &work_host));

        work_device_bytes_ = work_dev;
        work_host_bytes_   = work_host;
        CUDA_CHECK(cudaMalloc(&d_work_, std::max(work_device_bytes_, size_t(1))));
        h_work_ = std::malloc(std::max(work_host_bytes_, size_t(1)));
        if (!h_work_) throw std::runtime_error("malloc for cuSOLVER host workspace failed");

        CUDA_CHECK(cudaMalloc(&d_ipiv_, static_cast<size_t>(schur_dim_) * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_info_, sizeof(int)));

        return sec;
    }

    // Run one factorize + 3-phase solve.  Timings via CUDA events; one
    // cudaEventSynchronize at end.  Solution stays on device.
    SchurTimings factorize_and_solve(int32_t repeat_idx)
    {
        // D2D reset — no H2D copy in the hot path.
        CUDA_CHECK(cudaMemcpy(d_rhs_, d_rhs_orig_,
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_x_, 0, static_cast<size_t>(system_dim_) * sizeof(float)));

        // Event layout:
        //   ev_[0]  factor start   ev_[1]  extract start
        //   ev_[2]  fwd start      ev_[3]  diag start
        //   ev_[4]  getrf start    ev_[5]  getrs start
        //   ev_[6]  bwd start      ev_[7]  total end
        CUDA_CHECK(cudaEventRecord(ev_[0]));

        CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION,
                                 config_, data_, matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaEventRecord(ev_[1]));

        size_t size_written = 0;
        CUDSS_CHECK(cudssDataGet(handle_, data_, CUDSS_DATA_SCHUR_MATRIX,
                                 &schur_mat_obj_, sizeof(cudssMatrix_t), &size_written));
        CUDA_CHECK(cudaEventRecord(ev_[2]));

        CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
                                 config_, data_, matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaEventRecord(ev_[3]));

        // SOLVE_DIAG equivalent for GENERAL (D=I): move FWD output (d_x_) into
        // the b buffer (d_rhs_) so BWD reads [y1; y2] from d_rhs_.
        CUDA_CHECK(cudaMemcpyAsync(d_rhs_, d_x_,
                                   static_cast<size_t>(system_dim_) * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(ev_[4]));

        CUSOLVER_CHECK(cusolverDnXgetrf(
            cusolver_handle_, cusolver_params_,
            schur_dim_, schur_dim_,
            CUDA_R_32F, d_schur_mat_, schur_dim_,
            d_ipiv_,
            CUDA_R_32F, d_work_, work_device_bytes_,
            h_work_, work_host_bytes_,
            d_info_));
        CUDA_CHECK(cudaEventRecord(ev_[5]));

        CUSOLVER_CHECK(cusolverDnXgetrs(
            cusolver_handle_, cusolver_params_,
            CUBLAS_OP_N, schur_dim_, 1,
            CUDA_R_32F, d_schur_mat_, schur_dim_, d_ipiv_,
            CUDA_R_32F, d_rhs_ + (system_dim_ - schur_dim_), schur_dim_,
            d_info_));
        CUDA_CHECK(cudaEventRecord(ev_[6]));

        CUDSS_CHECK(cudssExecute(handle_, CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
                                 config_, data_, matrix_, solution_matrix_, rhs_matrix_));
        CUDA_CHECK(cudaEventRecord(ev_[7]));

        CUDA_CHECK(cudaEventSynchronize(ev_[7]));

        SchurTimings t;
        t.repeat_idx         = repeat_idx;
        t.schur_dim          = schur_dim_;
        t.factorization_sec  = elapsed_ms(ev_[0], ev_[1]) * 1e-3;
        t.schur_extract_sec  = elapsed_ms(ev_[1], ev_[2]) * 1e-3;
        t.fwd_solve_sec      = elapsed_ms(ev_[2], ev_[3]) * 1e-3;
        t.diag_sec           = elapsed_ms(ev_[3], ev_[4]) * 1e-3;
        t.getrf_sec          = elapsed_ms(ev_[4], ev_[5]) * 1e-3;
        t.getrs_sec          = elapsed_ms(ev_[5], ev_[6]) * 1e-3;
        t.schur_solve_sec    = t.diag_sec + t.getrf_sec + t.getrs_sec;
        t.bwd_solve_sec      = elapsed_ms(ev_[6], ev_[7]) * 1e-3;
        t.solve_sec          = t.fwd_solve_sec + t.schur_solve_sec + t.bwd_solve_sec;
        t.total_sec          = elapsed_ms(ev_[0], ev_[7]) * 1e-3;
        return t;
    }

    // Copy solution from device to host.  Call once after all iterations.
    void download_solution()
    {
        CUDA_CHECK(cudaMemcpy(h_x_.data(), d_x_,
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    const std::vector<float>& solution() const { return h_x_; }
    int32_t schur_dim() const { return schur_dim_; }

private:
    void convert_host_inputs_to_fp32()
    {
        h_values_f_.resize(system_.values.size());
        for (size_t i = 0; i < system_.values.size(); ++i)
            h_values_f_[i] = static_cast<float>(system_.values[i]);
    }

    void build_schur_index_vector()
    {
        const int32_t n_pvpq = system_dim_ - schur_dim_;
        h_schur_indices_.assign(system_dim_, 0);
        for (int32_t i = n_pvpq; i < system_dim_; ++i)
            h_schur_indices_[i] = 1;
    }

    void allocate_device_buffers()
    {
        CUDA_CHECK(cudaMalloc(&d_row_ptr_,  static_cast<size_t>(system_dim_ + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_col_idx_,  static_cast<size_t>(matrix_nnz_) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_,   static_cast<size_t>(matrix_nnz_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rhs_orig_, static_cast<size_t>(system_dim_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rhs_,      static_cast<size_t>(system_dim_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_,        static_cast<size_t>(system_dim_) * sizeof(float)));
    }

    void upload_static_inputs()
    {
        CUDA_CHECK(cudaMemcpy(d_row_ptr_, system_.structure.row_ptr.data(),
                              static_cast<size_t>(system_dim_ + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx_, system_.structure.col_idx.data(),
                              static_cast<size_t>(matrix_nnz_) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values_, h_values_f_.data(),
                              static_cast<size_t>(matrix_nnz_) * sizeof(float),
                              cudaMemcpyHostToDevice));
        // Upload RHS once to the device-side original buffer.
        std::vector<float> h_rhs_f(system_.rhs.size());
        for (size_t i = 0; i < system_.rhs.size(); ++i)
            h_rhs_f[i] = static_cast<float>(system_.rhs[i]);
        CUDA_CHECK(cudaMemcpy(d_rhs_orig_, h_rhs_f.data(),
                              static_cast<size_t>(system_dim_) * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void create_cudss_objects()
    {
        CUDSS_CHECK(cudssCreate(&handle_));
        CUDSS_CHECK(cudssConfigCreate(&config_));
        CUDSS_CHECK(cudssDataCreate(handle_, &data_));

        int compute_schur = 1;
        CUDSS_CHECK(cudssConfigSet(config_, CUDSS_CONFIG_SCHUR_MODE,
                                   &compute_schur, sizeof(int)));
        CUDSS_CHECK(cudssDataSet(handle_, data_, CUDSS_DATA_USER_SCHUR_INDICES,
                                 h_schur_indices_.data(),
                                 static_cast<size_t>(system_dim_) * sizeof(int32_t)));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &matrix_,
            system_dim_, system_dim_, matrix_nnz_,
            d_row_ptr_, nullptr, d_col_idx_, d_values_,
            CUDA_R_32I, CUDA_R_32F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        CUDSS_CHECK(cudssMatrixCreateDn(
            &rhs_matrix_, system_dim_, 1, system_dim_,
            d_rhs_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &solution_matrix_, system_dim_, 1, system_dim_,
            d_x_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
    }

    void create_events()
    {
        for (auto& ev : ev_)
            CUDA_CHECK(cudaEventCreate(&ev));
    }

    void destroy_events()
    {
        for (auto& ev : ev_)
            if (ev) cudaEventDestroy(ev);
    }

    void destroy_cudss_objects()
    {
        if (matrix_)          cudssMatrixDestroy(matrix_);
        if (rhs_matrix_)      cudssMatrixDestroy(rhs_matrix_);
        if (solution_matrix_) cudssMatrixDestroy(solution_matrix_);
        if (data_)            cudssDataDestroy(handle_, data_);
        if (config_)          cudssConfigDestroy(config_);
        if (handle_)          cudssDestroy(handle_);
    }

    void free_device_buffers()
    {
        if (d_row_ptr_)  cudaFree(d_row_ptr_);
        if (d_col_idx_)  cudaFree(d_col_idx_);
        if (d_values_)   cudaFree(d_values_);
        if (d_rhs_orig_) cudaFree(d_rhs_orig_);
        if (d_rhs_)      cudaFree(d_rhs_);
        if (d_x_)        cudaFree(d_x_);
    }

    static float elapsed_ms(cudaEvent_t start, cudaEvent_t end)
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, end);
        return ms;
    }

    const PowerFlowLinearSystem& system_;
    int32_t system_dim_ = 0;
    int32_t matrix_nnz_ = 0;
    int32_t schur_dim_  = 0;

    std::vector<float>   h_values_f_;
    std::vector<float>   h_x_;
    std::vector<int32_t> h_schur_indices_;

    int32_t* d_row_ptr_  = nullptr;
    int32_t* d_col_idx_  = nullptr;
    float*   d_values_   = nullptr;
    float*   d_rhs_orig_ = nullptr;  // original RHS on device (never overwritten)
    float*   d_rhs_      = nullptr;  // working b buffer (reset each iteration via D2D)
    float*   d_x_        = nullptr;
    float*   d_schur_mat_ = nullptr;
    int64_t* d_ipiv_      = nullptr;
    int*     d_info_      = nullptr;
    void*    d_work_      = nullptr;
    void*    h_work_      = nullptr;
    size_t   work_device_bytes_ = 0;
    size_t   work_host_bytes_   = 0;

    cudssHandle_t  handle_          = nullptr;
    cudssConfig_t  config_          = nullptr;
    cudssData_t    data_            = nullptr;
    cudssMatrix_t  matrix_          = nullptr;
    cudssMatrix_t  rhs_matrix_      = nullptr;
    cudssMatrix_t  solution_matrix_ = nullptr;
    cudssMatrix_t  schur_mat_obj_   = nullptr;

    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cusolverDnParams_t cusolver_params_ = nullptr;

    cudaEvent_t ev_[8] = {};  // 8 timestamp events per factorize_and_solve call
};

}  // namespace exp_20260409
