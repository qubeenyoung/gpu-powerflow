#pragma once

#include "newton_solver/backend/cuda_backend.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cudss.h>
#include <cuComplex.h>

#include <complex>
#include <vector>

// GPU helper: dst[i] = -(float)src[i]   (FP64→FP32 cast + negate)
void cuda_negate_cast(const double* src, float* dst, int32_t n);

// GPU helper: dst[i] = -src[i]           (FP64 negate in-place to separate buf)
void cuda_negate_f64(const double* src, double* dst, int32_t n);


struct CudaNewtonSolverBackend::Impl {
    // --- Precision mode (set at construction, immutable after analyze) ---
    PrecisionMode precision_mode = PrecisionMode::Mixed;

    // --- Dimensions (set in analyze) ---
    int32_t n_bus  = 0;
    int32_t y_nnz  = 0;
    int32_t j_nnz  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t n_pv   = 0;
    JacobianBuilderType jacobian_type = JacobianBuilderType::EdgeBased;

    // --- Ybus FP32 G/B for Jacobian kernel (Mixed and FP32 modes) ---
    float*   d_G_f   = nullptr;
    float*   d_B_f   = nullptr;
    int32_t* d_Y_row = nullptr;
    int32_t* d_Y_col = nullptr;

    // --- Ybus FP64 G/B for Jacobian kernel (FP64 mode only) ---
    double* d_G_d = nullptr;
    double* d_B_d = nullptr;

    // --- Ybus FP64 complex CSR for cuSPARSE SpMV (Mixed and FP64 modes) ---
    cuDoubleComplex* d_Ybus_val = nullptr;
    int32_t*         d_Ybus_rp  = nullptr;
    int32_t*         d_Ybus_ci  = nullptr;

    // --- Ybus FP32 complex CSR for cuSPARSE SpMV (FP32 mode only) ---
    cuFloatComplex* d_Ybus_val_f = nullptr;
    // d_Ybus_rp and d_Ybus_ci are shared (same int32 structure, allocated once)

    // --- JacobianMaps on GPU (CSR-indexed) ---
    int32_t* d_mapJ11  = nullptr;
    int32_t* d_mapJ12  = nullptr;
    int32_t* d_mapJ21  = nullptr;
    int32_t* d_mapJ22  = nullptr;
    int32_t* d_diagJ11 = nullptr;
    int32_t* d_diagJ12 = nullptr;
    int32_t* d_diagJ21 = nullptr;
    int32_t* d_diagJ22 = nullptr;

    // --- Bus type index arrays on GPU ---
    int32_t* d_pvpq = nullptr;
    int32_t* d_pv   = nullptr;
    int32_t* d_pq   = nullptr;

    // --- FP64 voltage state (Mixed and FP64 modes) ---
    cuDoubleComplex* d_V_cd = nullptr;
    double*          d_Va   = nullptr;
    double*          d_Vm   = nullptr;
    float*           d_V_f  = nullptr;   // FP32 interleaved (Mixed Jacobian kernel)

    // --- FP32 voltage state (FP32 mode only) ---
    cuFloatComplex* d_V_cf  = nullptr;   // FP32 complex voltage state
    float*          d_Va_f  = nullptr;
    float*          d_Vm_f  = nullptr;

    // --- FP64 auxiliary buffers (Mixed and FP64 modes) ---
    cuDoubleComplex* d_Sbus = nullptr;
    cuDoubleComplex* d_Ibus = nullptr;
    double*          d_F    = nullptr;   // mismatch vector FP64
    double*          d_dx   = nullptr;   // pre-allocated; avoids per-iteration cudaMalloc

    // --- FP32 auxiliary buffers (FP32 mode only) ---
    cuFloatComplex* d_Sbus_f = nullptr;
    cuFloatComplex* d_Ibus_f = nullptr;
    // d_b_f serves as both mismatch (negated) and cuDSS RHS in FP32 mode

    // --- FP32 Jacobian / solve buffers (Mixed and FP32 modes) ---
    float* d_J_csr_f = nullptr;
    float* d_b_f     = nullptr;
    float* d_x_f     = nullptr;

    // --- FP64 Jacobian / solve buffers (FP64 mode only) ---
    double* d_J_csr_d = nullptr;
    double* d_b_d     = nullptr;
    double* d_x_d     = nullptr;

    // --- Jacobian CSR structure ---
    int32_t* d_J_csr_rp = nullptr;
    int32_t* d_J_csr_ci = nullptr;

    int32_t dimF    = 0;
    int32_t n_batch = 1;

    // --- cuSPARSE: FP64 SpMV (Mixed and FP64 modes) ---
    cusparseHandle_t     sp_handle  = nullptr;
    cusparseSpMatDescr_t sp_Ybus    = nullptr;   // FP64 CSR descriptor
    cusparseDnVecDescr_t sp_V       = nullptr;   // FP64 dense vector
    cusparseDnVecDescr_t sp_Ibus    = nullptr;   // FP64 dense vector
    void*  d_spmv_buf  = nullptr;
    size_t spmv_buf_sz = 0;

    // --- cuSPARSE: FP32 SpMV (FP32 mode only) ---
    cusparseSpMatDescr_t sp_Ybus_f    = nullptr;  // FP32 CSR descriptor
    cusparseDnVecDescr_t sp_V_f       = nullptr;  // FP32 dense vector
    cusparseDnVecDescr_t sp_Ibus_f    = nullptr;  // FP32 dense vector
    void*  d_spmv_buf_f  = nullptr;
    size_t spmv_buf_f_sz = 0;

    // --- cuDSS: FP32 single-case (Mixed and FP32 modes) ---
    cudssHandle_t dss_handle = nullptr;
    cudssConfig_t dss_config = nullptr;
    cudssData_t   dss_data   = nullptr;
    cudssMatrix_t dss_J      = nullptr;
    cudssMatrix_t dss_b      = nullptr;
    cudssMatrix_t dss_x      = nullptr;

    // --- cuDSS: FP64 single-case (FP64 mode only) ---
    cudssConfig_t dss_config_d64 = nullptr;
    cudssData_t   dss_data_d64   = nullptr;
    cudssMatrix_t dss_J_d64      = nullptr;
    cudssMatrix_t dss_b_d64      = nullptr;
    cudssMatrix_t dss_x_d64      = nullptr;

    // -----------------------------------------------------------------------
    // Batch-mode buffers (allocated in analyze when n_batch > 1)
    // Layout convention:
    //   V/Ibus (SpMM inputs): col-major nbus×n_batch, element [bus,b] = buf[b*nbus+bus]
    //   Va/Vm/Sbus/F/dx:      row-major n_batch×n_bus, element [b,bus] = buf[b*nbus+bus]
    //   J_csr_f:              [n_batch * j_nnz],  batch b at offset b*j_nnz
    //   b_f / x_f:            [n_batch * dimF],   batch b at offset b*dimF
    // -----------------------------------------------------------------------
    cuDoubleComplex* d_V_cd_batch    = nullptr;  // [n_batch * n_bus] col-major (SpMM input)
    cuDoubleComplex* d_Ibus_batch    = nullptr;  // [n_batch * n_bus] col-major (SpMM output)
    double*          d_Va_batch      = nullptr;  // [n_batch * n_bus]
    double*          d_Vm_batch      = nullptr;  // [n_batch * n_bus]
    cuFloatComplex*  d_V_cf_batch    = nullptr;  // [n_batch * n_bus] FP32 complex for Jacobian
    cuDoubleComplex* d_Sbus_batch    = nullptr;  // [n_batch * n_bus]
    double*          d_F_batch       = nullptr;  // [n_batch * dimF]
    float*           d_b_f_batch     = nullptr;  // [n_batch * dimF]
    float*           d_x_f_batch     = nullptr;  // [n_batch * dimF]
    float*           d_J_csr_f_batch = nullptr;  // [n_batch * j_nnz]

    // cuSPARSE SpMM (batch)
    cusparseDnMatDescr_t sp_V_mat     = nullptr;  // nbus × n_batch dense
    cusparseDnMatDescr_t sp_Ibus_mat  = nullptr;  // nbus × n_batch dense
    void*   d_spmm_buf  = nullptr;
    size_t  spmm_buf_sz = 0;

    // cuDSS batch (separate config+data so UBATCH doesn't pollute single-case)
    cudssConfig_t dss_config_batch = nullptr;
    cudssData_t   dss_data_batch   = nullptr;
    cudssMatrix_t dss_J_batch      = nullptr;  // same sparsity, values = d_J_csr_f_batch
    cudssMatrix_t dss_b_batch      = nullptr;  // nrows=dimF, buf=[n_batch*dimF]
    cudssMatrix_t dss_x_batch      = nullptr;

    ~Impl()
    {
        auto safe_free = [](void* ptr) {
            if (ptr) cudaFree(ptr);
        };

        // FP32/Mixed Jacobian G,B
        safe_free(d_G_f);
        safe_free(d_B_f);
        safe_free(d_Y_row);
        safe_free(d_Y_col);
        // FP64 Jacobian G,B
        safe_free(d_G_d);
        safe_free(d_B_d);
        // Ybus CSR structure
        safe_free(d_Ybus_val);
        safe_free(d_Ybus_val_f);
        safe_free(d_Ybus_rp);
        safe_free(d_Ybus_ci);
        // JacobianMaps
        safe_free(d_mapJ11);
        safe_free(d_mapJ12);
        safe_free(d_mapJ21);
        safe_free(d_mapJ22);
        safe_free(d_diagJ11);
        safe_free(d_diagJ12);
        safe_free(d_diagJ21);
        safe_free(d_diagJ22);
        // Bus indices
        safe_free(d_pvpq);
        safe_free(d_pv);
        safe_free(d_pq);
        // FP64 voltage state
        safe_free(d_V_cd);
        safe_free(d_Va);
        safe_free(d_Vm);
        safe_free(d_V_f);
        // FP32 voltage state
        safe_free(d_V_cf);
        safe_free(d_Va_f);
        safe_free(d_Vm_f);
        // FP64 auxiliary
        safe_free(d_Sbus);
        safe_free(d_Ibus);
        safe_free(d_F);
        safe_free(d_dx);
        // FP32 auxiliary
        safe_free(d_Sbus_f);
        safe_free(d_Ibus_f);
        // Jacobian value arrays
        safe_free(d_J_csr_f);
        safe_free(d_b_f);
        safe_free(d_x_f);
        safe_free(d_J_csr_d);
        safe_free(d_b_d);
        safe_free(d_x_d);
        // Jacobian CSR structure
        safe_free(d_spmv_buf);
        safe_free(d_spmv_buf_f);
        safe_free(d_J_csr_rp);
        safe_free(d_J_csr_ci);

        // Batch buffers
        safe_free(d_V_cd_batch);
        safe_free(d_Ibus_batch);
        safe_free(d_Va_batch);
        safe_free(d_Vm_batch);
        safe_free(d_V_cf_batch);
        safe_free(d_Sbus_batch);
        safe_free(d_F_batch);
        safe_free(d_b_f_batch);
        safe_free(d_x_f_batch);
        safe_free(d_J_csr_f_batch);
        safe_free(d_spmm_buf);

        if (sp_Ybus)     cusparseDestroySpMat(sp_Ybus);
        if (sp_V)        cusparseDestroyDnVec(sp_V);
        if (sp_Ibus)     cusparseDestroyDnVec(sp_Ibus);
        if (sp_Ybus_f)   cusparseDestroySpMat(sp_Ybus_f);
        if (sp_V_f)      cusparseDestroyDnVec(sp_V_f);
        if (sp_Ibus_f)   cusparseDestroyDnVec(sp_Ibus_f);
        if (sp_V_mat)    cusparseDestroyDnMat(sp_V_mat);
        if (sp_Ibus_mat) cusparseDestroyDnMat(sp_Ibus_mat);
        if (sp_handle)   cusparseDestroy(sp_handle);

        // FP32 cuDSS (Mixed / FP32)
        if (dss_J)      cudssMatrixDestroy(dss_J);
        if (dss_b)      cudssMatrixDestroy(dss_b);
        if (dss_x)      cudssMatrixDestroy(dss_x);
        if (dss_data)   cudssDataDestroy(dss_handle, dss_data);
        if (dss_config) cudssConfigDestroy(dss_config);

        // FP64 cuDSS
        if (dss_J_d64)      cudssMatrixDestroy(dss_J_d64);
        if (dss_b_d64)      cudssMatrixDestroy(dss_b_d64);
        if (dss_x_d64)      cudssMatrixDestroy(dss_x_d64);
        if (dss_data_d64)   cudssDataDestroy(dss_handle, dss_data_d64);
        if (dss_config_d64) cudssConfigDestroy(dss_config_d64);

        if (dss_J_batch)      cudssMatrixDestroy(dss_J_batch);
        if (dss_b_batch)      cudssMatrixDestroy(dss_b_batch);
        if (dss_x_batch)      cudssMatrixDestroy(dss_x_batch);
        if (dss_data_batch)   cudssDataDestroy(dss_handle, dss_data_batch);
        if (dss_config_batch) cudssConfigDestroy(dss_config_batch);

        if (dss_handle) cudssDestroy(dss_handle);
    }
};
