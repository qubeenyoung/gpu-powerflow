#include "cuda_backend_impl.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/timer.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cudss.h>
#include <cuComplex.h>

#include <vector>
#include <complex>
#include <cmath>


// ---------------------------------------------------------------------------
// Constructor / Destructor / Move
// ---------------------------------------------------------------------------
CudaNewtonSolverBackend::CudaNewtonSolverBackend(int n_batch, PrecisionMode precision)
    : impl_(std::make_unique<Impl>())
    , n_batch_(n_batch)
    , precision_mode_(precision)
{
    impl_->n_batch        = n_batch;
    impl_->precision_mode = precision;
}

CudaNewtonSolverBackend::~CudaNewtonSolverBackend() = default;

CudaNewtonSolverBackend::CudaNewtonSolverBackend(CudaNewtonSolverBackend&&) noexcept = default;
CudaNewtonSolverBackend& CudaNewtonSolverBackend::operator=(CudaNewtonSolverBackend&&) noexcept = default;

void CudaNewtonSolverBackend::synchronizeForTiming()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ---------------------------------------------------------------------------
// Helper: upload a host int32 vector to newly allocated GPU memory.
// ---------------------------------------------------------------------------
static void upload_i32(int32_t*& d_ptr, const int32_t* h_ptr, int32_t n)
{
    size_t bytes = n * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
}

static void upload_i32(int32_t*& d_ptr, const std::vector<int32_t>& v)
{
    upload_i32(d_ptr, v.data(), static_cast<int32_t>(v.size()));
}

static void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}


// ---------------------------------------------------------------------------
// analyze: upload Ybus and JacobianMaps to GPU, build cuSPARSE/cuDSS handles.
//
// Called for PrecisionMode::Mixed and PrecisionMode::FP64.
// PrecisionMode::FP32 uses analyze_f32() instead.
//
// Mixed  — FP32 G/B for Jacobian, FP64 complex Ybus for SpMV, FP32 cuDSS.
// FP64   — FP64 G/B for Jacobian, FP64 complex Ybus for SpMV, FP64 cuDSS.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::analyze(
    const YbusViewF64&       ybus,
    const JacobianMaps&      maps,
    const JacobianStructure& J,
    int32_t                  n_bus)
{
    auto& im = *impl_;
    im.n_bus  = n_bus;
    im.y_nnz  = ybus.nnz;
    im.j_nnz  = J.nnz;
    im.n_pvpq = maps.n_pvpq;
    im.n_pq   = maps.n_pq;
    im.n_pv   = maps.n_pvpq - maps.n_pq;
    im.dimF   = maps.n_pvpq + maps.n_pq;
    im.jacobian_type = maps.builder_type;

    const bool is_mixed = (im.precision_mode == PrecisionMode::Mixed);
    const bool is_fp64  = (im.precision_mode == PrecisionMode::FP64);

    // ------------------------------------------------------------------
    // 1. Ybus G/B for Jacobian kernel + row/col index arrays
    // ------------------------------------------------------------------
    std::vector<int32_t> h_Y_row(im.y_nnz), h_Y_col(im.y_nnz);
    for (int32_t row = 0; row < ybus.rows; ++row)
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            h_Y_row[k] = row;
            h_Y_col[k] = ybus.indices[k];
        }

    if (is_mixed) {
        // FP32 G, B for Mixed Jacobian kernel
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadYbusGBFp32");
        std::vector<float> h_G(im.y_nnz), h_B(im.y_nnz);
        for (int32_t k = 0; k < im.y_nnz; ++k) {
            h_G[k] = static_cast<float>(ybus.data[k].real());
            h_B[k] = static_cast<float>(ybus.data[k].imag());
        }
        CUDA_CHECK(cudaMalloc(&im.d_G_f, im.y_nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&im.d_B_f, im.y_nnz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(im.d_G_f, h_G.data(), im.y_nnz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_B_f, h_B.data(), im.y_nnz * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (is_fp64) {
        // FP64 G, B for FP64 Jacobian kernel
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadYbusGBFp64");
        std::vector<double> h_G(im.y_nnz), h_B(im.y_nnz);
        for (int32_t k = 0; k < im.y_nnz; ++k) {
            h_G[k] = ybus.data[k].real();
            h_B[k] = ybus.data[k].imag();
        }
        CUDA_CHECK(cudaMalloc(&im.d_G_d, im.y_nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&im.d_B_d, im.y_nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(im.d_G_d, h_G.data(), im.y_nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_B_d, h_B.data(), im.y_nnz * sizeof(double), cudaMemcpyHostToDevice));
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadYbusIndices");
        upload_i32(im.d_Y_row, h_Y_row);
        upload_i32(im.d_Y_col, h_Y_col);
    }

    // ------------------------------------------------------------------
    // 2. Ybus FP64 complex CSR for cuSPARSE SpMV (Mixed and FP64)
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadYbusSpmvFp64");
        std::vector<cuDoubleComplex> h_Yval(im.y_nnz);
        for (int32_t k = 0; k < im.y_nnz; ++k)
            h_Yval[k] = make_cuDoubleComplex(ybus.data[k].real(), ybus.data[k].imag());

        CUDA_CHECK(cudaMalloc(&im.d_Ybus_val, im.y_nnz * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_Ybus_rp,  (n_bus + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&im.d_Ybus_ci,  im.y_nnz * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_val, h_Yval.data(), im.y_nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_rp,  ybus.indptr,   (n_bus + 1) * sizeof(int32_t),      cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_ci,  ybus.indices,  im.y_nnz * sizeof(int32_t),         cudaMemcpyHostToDevice));
    }
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cusparseSetupFp64");
        CUSPARSE_CHECK(cusparseCreate(&im.sp_handle));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &im.sp_Ybus,
            n_bus, n_bus, im.y_nnz,
            im.d_Ybus_rp, im.d_Ybus_ci, im.d_Ybus_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

        CUDA_CHECK(cudaMalloc(&im.d_V_cd, n_bus * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_Ibus, n_bus * sizeof(cuDoubleComplex)));

        CUSPARSE_CHECK(cusparseCreateDnVec(&im.sp_V,    n_bus, im.d_V_cd, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&im.sp_Ibus, n_bus, im.d_Ibus, CUDA_C_64F));

        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            im.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, im.sp_Ybus, im.sp_V, &beta, im.sp_Ibus,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &im.spmv_buf_sz));
        if (im.spmv_buf_sz > 0)
            CUDA_CHECK(cudaMalloc(&im.d_spmv_buf, im.spmv_buf_sz));
    }

    // ------------------------------------------------------------------
    // 3. Upload JacobianMaps to GPU.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadJacobianMaps");
        upload_i32(im.d_mapJ11,  maps.mapJ11);
        upload_i32(im.d_mapJ12,  maps.mapJ12);
        upload_i32(im.d_mapJ21,  maps.mapJ21);
        upload_i32(im.d_mapJ22,  maps.mapJ22);
        upload_i32(im.d_diagJ11, maps.diagJ11);
        upload_i32(im.d_diagJ12, maps.diagJ12);
        upload_i32(im.d_diagJ21, maps.diagJ21);
        upload_i32(im.d_diagJ22, maps.diagJ22);
    }

    // ------------------------------------------------------------------
    // 4. Upload pv/pq bus indices.
    // ------------------------------------------------------------------
    const int32_t n_pv = im.n_pv;
    const int32_t n_pq = im.n_pq;
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadBusIndices");
        upload_i32(im.d_pvpq, maps.pvpq);
        upload_i32(im.d_pv, maps.pvpq.data(),        n_pv);
        upload_i32(im.d_pq, maps.pvpq.data() + n_pv, n_pq);
    }

    // ------------------------------------------------------------------
    // 5. Allocate work buffers (precision-dependent)
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudaMallocWorkBuffers");

        // Shared: FP64 voltage state + mismatch vector
        CUDA_CHECK(cudaMalloc(&im.d_Va,   n_bus   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&im.d_Vm,   n_bus   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&im.d_Sbus, n_bus   * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_F,    im.dimF * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&im.d_dx,   im.dimF * sizeof(double)));

        if (is_mixed) {
            // FP32 interleaved V for Jacobian kernel
            CUDA_CHECK(cudaMalloc(&im.d_V_f,     n_bus    * 2 * sizeof(float)));
            // FP32 Jacobian / solve buffers
            CUDA_CHECK(cudaMalloc(&im.d_J_csr_f, im.j_nnz * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&im.d_b_f,     im.dimF  * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&im.d_x_f,     im.dimF  * sizeof(float)));
        }
        if (is_fp64) {
            // FP64 Jacobian / solve buffers
            CUDA_CHECK(cudaMalloc(&im.d_J_csr_d, im.j_nnz * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im.d_b_d,     im.dimF  * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im.d_x_d,     im.dimF  * sizeof(double)));
        }
    }

    // ------------------------------------------------------------------
    // 6. Upload Jacobian CSR structure.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.uploadJacobianStructure");
        upload_i32(im.d_J_csr_rp, J.row_ptr);
        upload_i32(im.d_J_csr_ci, J.col_idx);
    }

    // ------------------------------------------------------------------
    // 7. cuDSS: create handle, run ANALYSIS + initial FACTORIZATION.
    //    Mixed → CUDA_R_32F; FP64 → CUDA_R_64F.
    // ------------------------------------------------------------------
    CUDSS_CHECK(cudssCreate(&im.dss_handle));

    if (is_mixed) {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudssSetupFp32");
        CUDSS_CHECK(cudssConfigCreate(&im.dss_config));
        CUDSS_CHECK(cudssDataCreate(im.dss_handle, &im.dss_data));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &im.dss_J,
            im.dimF, im.dimF, im.j_nnz,
            im.d_J_csr_rp, nullptr, im.d_J_csr_ci, im.d_J_csr_f,
            CUDA_R_32I, CUDA_R_32F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_b, im.dimF, 1, im.dimF, im.d_b_f, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_x, im.dimF, 1, im.dimF, im.d_x_f, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_ANALYSIS,
            im.dss_config, im.dss_data, im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_FACTORIZATION,
            im.dss_config, im.dss_data, im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }

    if (is_fp64) {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudssSetupFp64");
        CUDSS_CHECK(cudssConfigCreate(&im.dss_config_d64));
        CUDSS_CHECK(cudssDataCreate(im.dss_handle, &im.dss_data_d64));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &im.dss_J_d64,
            im.dimF, im.dimF, im.j_nnz,
            im.d_J_csr_rp, nullptr, im.d_J_csr_ci, im.d_J_csr_d,
            CUDA_R_32I, CUDA_R_64F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_b_d64, im.dimF, 1, im.dimF, im.d_b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_x_d64, im.dimF, 1, im.dimF, im.d_x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_ANALYSIS,
            im.dss_config_d64, im.dss_data_d64, im.dss_J_d64, im.dss_x_d64, im.dss_b_d64));
        sync_cuda_for_timing();
        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_FACTORIZATION,
            im.dss_config_d64, im.dss_data_d64, im.dss_J_d64, im.dss_x_d64, im.dss_b_d64));
        sync_cuda_for_timing();
    }

    // ------------------------------------------------------------------
    // 8. (Optional) Batch path: SpMM descriptors + cuDSS UBATCH setup
    //    Only when n_batch > 1.
    // ------------------------------------------------------------------
    if (im.n_batch > 1) {
        const int32_t nb = im.n_batch;

        {
            newton_solver::utils::ScopedTimer timer("CUDA.batch.analyze.cudaMallocBuffers");
            CUDA_CHECK(cudaMalloc(&im.d_V_cd_batch,    (size_t)nb * n_bus * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&im.d_Ibus_batch,    (size_t)nb * n_bus * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&im.d_Va_batch,      (size_t)nb * n_bus * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im.d_Vm_batch,      (size_t)nb * n_bus * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im.d_V_cf_batch,    (size_t)nb * n_bus * sizeof(cuFloatComplex)));
            CUDA_CHECK(cudaMalloc(&im.d_Sbus_batch,    (size_t)nb * n_bus * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMalloc(&im.d_F_batch,       (size_t)nb * im.dimF * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im.d_b_f_batch,     (size_t)nb * im.dimF * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&im.d_x_f_batch,     (size_t)nb * im.dimF * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&im.d_J_csr_f_batch, (size_t)nb * im.j_nnz * sizeof(float)));
        }

        {
            newton_solver::utils::ScopedTimer timer("CUDA.batch.analyze.cusparseSetup");
            // cuSPARSE SpMM: Ybus (n_bus × n_bus) × V_batch (n_bus × nb) → Ibus_batch
            // V/Ibus matrices: col-major, leading dim = n_bus
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &im.sp_V_mat,
                n_bus, nb, n_bus,          // rows, cols, ld
                im.d_V_cd_batch,
                CUDA_C_64F, CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseCreateDnMat(
                &im.sp_Ibus_mat,
                n_bus, nb, n_bus,
                im.d_Ibus_batch,
                CUDA_C_64F, CUSPARSE_ORDER_COL));

            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                im.sp_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, im.sp_Ybus, im.sp_V_mat, &beta, im.sp_Ibus_mat,
                CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT,
                &im.spmm_buf_sz));
            if (im.spmm_buf_sz > 0)
                CUDA_CHECK(cudaMalloc(&im.d_spmm_buf, im.spmm_buf_sz));
        }

        {
            newton_solver::utils::ScopedTimer timer("CUDA.batch.analyze.cudssCreate");
            // cuDSS UBATCH: separate config+data so single-case path is unaffected
            CUDSS_CHECK(cudssConfigCreate(&im.dss_config_batch));
            int32_t ubatch_size = nb;
            CUDSS_CHECK(cudssConfigSet(
                im.dss_config_batch,
                CUDSS_CONFIG_UBATCH_SIZE,
                &ubatch_size,
                sizeof(ubatch_size)));

            CUDSS_CHECK(cudssDataCreate(im.dss_handle, &im.dss_data_batch));

            // Jacobian CSR: same sparsity, values buffer = d_J_csr_f_batch [nb * j_nnz]
            CUDSS_CHECK(cudssMatrixCreateCsr(
                &im.dss_J_batch,
                im.dimF, im.dimF, im.j_nnz,
                im.d_J_csr_rp, nullptr, im.d_J_csr_ci, im.d_J_csr_f_batch,
                CUDA_R_32I, CUDA_R_32F,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

            // RHS/sol: declared as (dimF × 1), buffer is [nb * dimF] — UBATCH interprets it as nb copies
            CUDSS_CHECK(cudssMatrixCreateDn(
                &im.dss_b_batch,
                im.dimF, 1, im.dimF,
                im.d_b_f_batch, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
            CUDSS_CHECK(cudssMatrixCreateDn(
                &im.dss_x_batch,
                im.dimF, 1, im.dimF,
                im.d_x_f_batch, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        }

        {
            newton_solver::utils::ScopedTimer timer("CUDA.batch.analyze.cudssAnalysis");
            CUDSS_CHECK(cudssExecute(
                im.dss_handle, CUDSS_PHASE_ANALYSIS,
                im.dss_config_batch, im.dss_data_batch,
                im.dss_J_batch, im.dss_x_batch, im.dss_b_batch));
            sync_cuda_for_timing();
        }

        {
            newton_solver::utils::ScopedTimer timer("CUDA.batch.analyze.cudssFactorization");
            CUDSS_CHECK(cudssExecute(
                im.dss_handle, CUDSS_PHASE_FACTORIZATION,
                im.dss_config_batch, im.dss_data_batch,
                im.dss_J_batch, im.dss_x_batch, im.dss_b_batch));
            sync_cuda_for_timing();
        }
    }
}


// ---------------------------------------------------------------------------
// analyze_f32: FP32 end-to-end path — FP32 SpMV + FP32 Jacobian + FP32 cuDSS.
//
// Called when PrecisionMode::FP32 is active (CUDA, n_batch == 1).
// Ybus values are complex<float>; G/B extracted directly without downcast.
// d_Ybus_rp / d_Ybus_ci are the same int32 structure used for FP64 path.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::analyze_f32(
    const YbusViewF32&       ybus,
    const JacobianMaps&      maps,
    const JacobianStructure& J,
    int32_t                  n_bus)
{
    auto& im = *impl_;
    im.n_bus  = n_bus;
    im.y_nnz  = ybus.nnz;
    im.j_nnz  = J.nnz;
    im.n_pvpq = maps.n_pvpq;
    im.n_pq   = maps.n_pq;
    im.n_pv   = maps.n_pvpq - maps.n_pq;
    im.dimF   = maps.n_pvpq + maps.n_pq;
    im.jacobian_type = maps.builder_type;

    // ------------------------------------------------------------------
    // 1. FP32 G/B for Jacobian kernel + row/col indices
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.uploadYbusGB");
        std::vector<float>   h_G(im.y_nnz), h_B(im.y_nnz);
        std::vector<int32_t> h_Y_row(im.y_nnz), h_Y_col(im.y_nnz);
        for (int32_t row = 0; row < ybus.rows; ++row)
            for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
                h_G[k]     = ybus.data[k].real();
                h_B[k]     = ybus.data[k].imag();
                h_Y_row[k] = row;
                h_Y_col[k] = ybus.indices[k];
            }
        CUDA_CHECK(cudaMalloc(&im.d_G_f, im.y_nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&im.d_B_f, im.y_nnz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(im.d_G_f, h_G.data(), im.y_nnz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_B_f, h_B.data(), im.y_nnz * sizeof(float), cudaMemcpyHostToDevice));
        upload_i32(im.d_Y_row, h_Y_row);
        upload_i32(im.d_Y_col, h_Y_col);
    }

    // ------------------------------------------------------------------
    // 2. FP32 complex Ybus CSR for cuSPARSE SpMV
    //    d_Ybus_rp / d_Ybus_ci allocated once; shared with FP64 path if needed.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.uploadYbusSpmvFp32");
        std::vector<cuFloatComplex> h_Yval_f(im.y_nnz);
        for (int32_t k = 0; k < im.y_nnz; ++k)
            h_Yval_f[k] = make_cuFloatComplex(ybus.data[k].real(), ybus.data[k].imag());

        CUDA_CHECK(cudaMalloc(&im.d_Ybus_val_f, im.y_nnz * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_Ybus_rp,    (n_bus + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&im.d_Ybus_ci,    im.y_nnz * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_val_f, h_Yval_f.data(), im.y_nnz * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_rp,    ybus.indptr,     (n_bus + 1) * sizeof(int32_t),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(im.d_Ybus_ci,    ybus.indices,    im.y_nnz * sizeof(int32_t),        cudaMemcpyHostToDevice));
    }
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.cusparseSetupFp32");
        CUSPARSE_CHECK(cusparseCreate(&im.sp_handle));

        CUDA_CHECK(cudaMalloc(&im.d_V_cf,   n_bus * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_Ibus_f, n_bus * sizeof(cuFloatComplex)));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &im.sp_Ybus_f,
            n_bus, n_bus, im.y_nnz,
            im.d_Ybus_rp, im.d_Ybus_ci, im.d_Ybus_val_f,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&im.sp_V_f,    n_bus, im.d_V_cf,   CUDA_C_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&im.sp_Ibus_f, n_bus, im.d_Ibus_f, CUDA_C_32F));

        cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
        cuFloatComplex beta  = make_cuFloatComplex(0.0f, 0.0f);
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            im.sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, im.sp_Ybus_f, im.sp_V_f, &beta, im.sp_Ibus_f,
            CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, &im.spmv_buf_f_sz));
        if (im.spmv_buf_f_sz > 0)
            CUDA_CHECK(cudaMalloc(&im.d_spmv_buf_f, im.spmv_buf_f_sz));
    }

    // ------------------------------------------------------------------
    // 3. Upload JacobianMaps
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.uploadJacobianMaps");
        upload_i32(im.d_mapJ11,  maps.mapJ11);
        upload_i32(im.d_mapJ12,  maps.mapJ12);
        upload_i32(im.d_mapJ21,  maps.mapJ21);
        upload_i32(im.d_mapJ22,  maps.mapJ22);
        upload_i32(im.d_diagJ11, maps.diagJ11);
        upload_i32(im.d_diagJ12, maps.diagJ12);
        upload_i32(im.d_diagJ21, maps.diagJ21);
        upload_i32(im.d_diagJ22, maps.diagJ22);
    }

    // ------------------------------------------------------------------
    // 4. Upload bus indices
    // ------------------------------------------------------------------
    const int32_t n_pv = im.n_pv;
    const int32_t n_pq = im.n_pq;
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.uploadBusIndices");
        upload_i32(im.d_pvpq, maps.pvpq);
        upload_i32(im.d_pv, maps.pvpq.data(),        n_pv);
        upload_i32(im.d_pq, maps.pvpq.data() + n_pv, n_pq);
    }

    // ------------------------------------------------------------------
    // 5. FP32 work buffers
    //    d_b_f serves as the mismatch buffer (already negated) and cuDSS RHS.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.cudaMallocWorkBuffers");
        CUDA_CHECK(cudaMalloc(&im.d_Va_f,    n_bus    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&im.d_Vm_f,    n_bus    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&im.d_Sbus_f,  n_bus    * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMalloc(&im.d_V_f,     n_bus    * 2 * sizeof(float)));  // interleaved, Jacobian kernel
        CUDA_CHECK(cudaMalloc(&im.d_J_csr_f, im.j_nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&im.d_b_f,     im.dimF  * sizeof(float)));      // mismatch + cuDSS RHS
        CUDA_CHECK(cudaMalloc(&im.d_x_f,     im.dimF  * sizeof(float)));
    }

    // ------------------------------------------------------------------
    // 6. Upload Jacobian CSR structure
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.uploadJacobianStructure");
        upload_i32(im.d_J_csr_rp, J.row_ptr);
        upload_i32(im.d_J_csr_ci, J.col_idx);
    }

    // ------------------------------------------------------------------
    // 7. cuDSS: FP32 setup + ANALYSIS + initial FACTORIZATION
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze_f32.cudssSetupFp32");
        CUDSS_CHECK(cudssCreate(&im.dss_handle));
        CUDSS_CHECK(cudssConfigCreate(&im.dss_config));
        CUDSS_CHECK(cudssDataCreate(im.dss_handle, &im.dss_data));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &im.dss_J,
            im.dimF, im.dimF, im.j_nnz,
            im.d_J_csr_rp, nullptr, im.d_J_csr_ci, im.d_J_csr_f,
            CUDA_R_32I, CUDA_R_32F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_b, im.dimF, 1, im.dimF, im.d_b_f, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &im.dss_x, im.dimF, 1, im.dimF, im.d_x_f, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_ANALYSIS,
            im.dss_config, im.dss_data, im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
        CUDSS_CHECK(cudssExecute(im.dss_handle, CUDSS_PHASE_FACTORIZATION,
            im.dss_config, im.dss_data, im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }
}


// ---------------------------------------------------------------------------
// initialize: upload V0 and Sbus for a new solve.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::initialize(
    const YbusView&             ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0)
{
    auto& im = *impl_;
    const int32_t n = im.n_bus;
    (void)ybus;

    CUDA_CHECK(cudaMemcpy(im.d_V_cd, V0,   n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Sbus, sbus, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Decompose V0 into Va/Vm on host, then upload
    std::vector<double> h_Va(n), h_Vm(n);
    for (int32_t i = 0; i < n; ++i) {
        h_Va[i] = std::arg(V0[i]);
        h_Vm[i] = std::abs(V0[i]);
    }
    CUDA_CHECK(cudaMemcpy(im.d_Va, h_Va.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Vm, h_Vm.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Mixed mode needs an FP32 interleaved voltage buffer for the Jacobian
    // kernel. FP64 mode uses d_V_cd directly and never allocates d_V_f.
    if (im.precision_mode == PrecisionMode::Mixed) {
        std::vector<float> h_V_f(n * 2);
        for (int32_t i = 0; i < n; ++i) {
            h_V_f[i * 2]     = static_cast<float>(V0[i].real());
            h_V_f[i * 2 + 1] = static_cast<float>(V0[i].imag());
        }
        CUDA_CHECK(cudaMemcpy(im.d_V_f, h_V_f.data(), n * 2 * sizeof(float), cudaMemcpyHostToDevice));
    }
}


// ---------------------------------------------------------------------------
// initialize_f32: upload V0 and Sbus for a new FP32 solve.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::initialize_f32(
    const YbusViewF32&        /*ybus*/,
    const std::complex<float>* sbus,
    const std::complex<float>* V0)
{
    auto& im = *impl_;
    const int32_t n = im.n_bus;

    CUDA_CHECK(cudaMemcpy(im.d_V_cf,   V0,   n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Sbus_f, sbus, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // Decompose V0 into Va/Vm (FP32) and upload interleaved V_f for Jacobian
    std::vector<float> h_Va(n), h_Vm(n), h_V_f(n * 2);
    for (int32_t i = 0; i < n; ++i) {
        h_Va[i]       = std::arg(V0[i]);
        h_Vm[i]       = std::abs(V0[i]);
        h_V_f[i * 2]     = V0[i].real();
        h_V_f[i * 2 + 1] = V0[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(im.d_Va_f, h_Va.data(),   n * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Vm_f, h_Vm.data(),   n * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_V_f,  h_V_f.data(),  n * 2 * sizeof(float), cudaMemcpyHostToDevice));
}


// ---------------------------------------------------------------------------
// downloadV: copy final voltage (FP64) from GPU to caller.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::downloadV(std::complex<double>* V_out, int32_t n_bus)
{
    CUDA_CHECK(cudaMemcpy(V_out, impl_->d_V_cd,
                          n_bus * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
}


// ---------------------------------------------------------------------------
// downloadV_f32: copy final voltage (FP32) from GPU to caller.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::downloadV_f32(std::complex<float>* V_out, int32_t n_bus)
{
    CUDA_CHECK(cudaMemcpy(V_out, impl_->d_V_cf,
                          n_bus * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
}


// ---------------------------------------------------------------------------
// initialize_batch: upload V0 and Sbus for all n_batch cases.
//
// Layout: V0_batch[b * n_bus + bus], sbus_batch[b * n_bus + bus]
// GPU buffers are col-major nbus×n_batch (same linear index b*nbus+bus).
// Also decomposes V0 into Va/Vm and converts to FP32 complex for Jacobian.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::initialize_batch(
    const YbusView&             /*ybus*/,
    const std::complex<double>* sbus_batch,
    const std::complex<double>* V0_batch,
    int32_t                     n_batch)
{
    auto& im = *impl_;
    const int32_t n     = im.n_bus;
    const int64_t total = (int64_t)n_batch * n;

    // Upload V0 and Sbus (contiguous [n_batch * n_bus])
    CUDA_CHECK(cudaMemcpy(im.d_V_cd_batch, V0_batch,
                          total * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Sbus_batch, sbus_batch,
                          total * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Va, Vm, V_cf on host then upload
    std::vector<double>       h_Va(total), h_Vm(total);
    std::vector<cuFloatComplex> h_Vcf(total);
    for (int64_t k = 0; k < total; ++k) {
        const std::complex<double>& v = V0_batch[k];
        h_Va[k]  = std::arg(v);
        h_Vm[k]  = std::abs(v);
        h_Vcf[k] = make_cuFloatComplex(static_cast<float>(v.real()),
                                        static_cast<float>(v.imag()));
    }
    CUDA_CHECK(cudaMemcpy(im.d_Va_batch,   h_Va.data(),  total * sizeof(double),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_Vm_batch,   h_Vm.data(),  total * sizeof(double),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_V_cf_batch, h_Vcf.data(), total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
}


// ---------------------------------------------------------------------------
// downloadV_batch: copy all batch voltages from GPU to caller.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::downloadV_batch(
    std::complex<double>* V_out,
    int32_t               n_bus,
    int32_t               n_batch)
{
    CUDA_CHECK(cudaMemcpy(V_out, impl_->d_V_cd_batch,
                          (size_t)n_batch * n_bus * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
}
