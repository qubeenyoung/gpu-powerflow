#include "cupf_minimal/hybrid_nr.hpp"

// experimental minimal cuPF NR port

#include "cupf_minimal/direct_cudss_solver.hpp"
#include "cupf_minimal/jacobian_analysis.hpp"
#include "cupf_minimal/nr_kernels.hpp"

#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/kernels/gmres_kernels.hpp"
#include "cuiter/solver/cpu_block_ilu0_pilot.hpp"
#include "cuiter/solver/gmres_solver.hpp"

#include <cublas_v2.h>
#include <cusparse.h>

#if defined(CUITER_WITH_GINKGO)
#include <ginkgo/ginkgo.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cupf_minimal {
namespace {

#define CUPF_MINIMAL_CUSPARSE_CHECK(call)                                            \
    do {                                                                             \
        const cusparseStatus_t status__ = (call);                                    \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(std::string("cuSPARSE error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " in " +     \
                                     #call + " status=" +                          \
                                     std::to_string(static_cast<int>(status__)));     \
        }                                                                            \
    } while (0)

template <typename Fn>
double timed_with_sync(Fn&& fn)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    fn();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <typename Fn>
double timed_on_stream(cudaStream_t stream, Fn&& fn)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&start));
    CUITER_CUDA_CHECK(cudaEventCreate(&stop));
    CUITER_CUDA_CHECK(cudaEventRecord(start, stream));
    fn();
    CUITER_CUDA_CHECK(cudaEventRecord(stop, stream));
    CUITER_CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0F;
    CUITER_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUITER_CUDA_CHECK(cudaEventDestroy(start));
    CUITER_CUDA_CHECK(cudaEventDestroy(stop));
    return 0.001 * static_cast<double>(milliseconds);
}

double elapsed_seconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::vector<double> real_parts(const std::vector<std::complex<double>>& values)
{
    std::vector<double> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [](const auto& value) {
        return value.real();
    });
    return out;
}

std::vector<double> imag_parts(const std::vector<std::complex<double>>& values)
{
    std::vector<double> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [](const auto& value) {
        return value.imag();
    });
    return out;
}

struct MismatchNorms {
    double inf = 0.0;
    double two = 0.0;
    double p_inf = 0.0;
    double p_two = 0.0;
    double q_inf = 0.0;
    double q_two = 0.0;
};

struct DxComparisonMetrics {
    double dx_norm_ratio = 0.0;
    double dx_cosine = 0.0;
    double dx_projection = 0.0;
    double dx_orth_error = 0.0;
    double theta_norm_ratio = 0.0;
    double theta_cosine = 0.0;
    double vmag_norm_ratio = 0.0;
    double vmag_cosine = 0.0;
    double max_abs_dx_gmres = 0.0;
    double max_abs_dx_cudss = 0.0;
    double max_abs_dx_diff = 0.0;
};

struct MinimalNrDeviceContext {
    int32_t n_bus = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t n_pvpq = 0;
    int32_t dimF = 0;
    int32_t nnz_ybus = 0;

    cuiter::CsrMatrix j_pattern;
    std::vector<int32_t> index_to_bus;
    std::vector<int32_t> index_field;

    cuiter::DeviceBuffer<int32_t> d_y_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_y_col;
    cuiter::DeviceBuffer<int32_t> d_y_row;
    cuiter::DeviceBuffer<double> d_y_re;
    cuiter::DeviceBuffer<double> d_y_im;

    cuiter::DeviceBuffer<int32_t> d_pv;
    cuiter::DeviceBuffer<int32_t> d_pq;
    cuiter::DeviceBuffer<double> d_sbus_re;
    cuiter::DeviceBuffer<double> d_sbus_im;

    cuiter::DeviceBuffer<double> d_v_re;
    cuiter::DeviceBuffer<double> d_v_im;
    cuiter::DeviceBuffer<double> d_va;
    cuiter::DeviceBuffer<double> d_vm;
    cuiter::DeviceBuffer<double> d_v_re_backup;
    cuiter::DeviceBuffer<double> d_v_im_backup;
    cuiter::DeviceBuffer<double> d_va_backup;
    cuiter::DeviceBuffer<double> d_vm_backup;

    cuiter::DeviceBuffer<double> d_ibus_re;
    cuiter::DeviceBuffer<double> d_ibus_im;
    cuiter::DeviceBuffer<double> d_F;
    cuiter::DeviceBuffer<double> d_F_backup;
    cuiter::DeviceBuffer<double> d_norm_inf;
    cuiter::DeviceBuffer<double> d_dx;
    cuiter::DeviceBuffer<double> d_prev_dx;
    cuiter::DeviceBuffer<double> d_dx_gmres_shadow;
    cuiter::DeviceBuffer<double> d_dx_cudss_shadow;
    cuiter::DeviceBuffer<double> d_dx_diff_shadow;
    cuiter::DeviceBuffer<double> d_ax;
    cuiter::DeviceBuffer<double> d_linear_residual;
    cuiter::DeviceBuffer<int32_t> d_bad_count;

    cuiter::DeviceBuffer<int32_t> d_J_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_J_col_idx;
    cuiter::DeviceBuffer<double> d_J_values;
    cuiter::DeviceBuffer<int32_t> d_mapJ11;
    cuiter::DeviceBuffer<int32_t> d_mapJ12;
    cuiter::DeviceBuffer<int32_t> d_mapJ21;
    cuiter::DeviceBuffer<int32_t> d_mapJ22;
    cuiter::DeviceBuffer<int32_t> d_diagJ11;
    cuiter::DeviceBuffer<int32_t> d_diagJ12;
    cuiter::DeviceBuffer<int32_t> d_diagJ21;
    cuiter::DeviceBuffer<int32_t> d_diagJ22;

    explicit MinimalNrDeviceContext(const DumpCaseData& data)
    {
        n_bus = data.rows;
        n_pv = static_cast<int32_t>(data.pv.size());
        n_pq = static_cast<int32_t>(data.pq.size());
        n_pvpq = n_pv + n_pq;
        dimF = n_pvpq + n_pq;
        nnz_ybus = static_cast<int32_t>(data.ybus_data.size());
        index_to_bus.assign(static_cast<std::size_t>(dimF), -1);
        index_field.assign(static_cast<std::size_t>(dimF), -1);
        for (int32_t i = 0; i < n_pv; ++i) {
            index_to_bus[static_cast<std::size_t>(i)] = data.pv[static_cast<std::size_t>(i)];
            index_field[static_cast<std::size_t>(i)] = 0;
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            const int32_t theta_index = n_pv + i;
            const int32_t vmag_index = n_pvpq + i;
            const int32_t bus = data.pq[static_cast<std::size_t>(i)];
            index_to_bus[static_cast<std::size_t>(theta_index)] = bus;
            index_field[static_cast<std::size_t>(theta_index)] = 0;
            index_to_bus[static_cast<std::size_t>(vmag_index)] = bus;
            index_field[static_cast<std::size_t>(vmag_index)] = 1;
        }

        std::vector<int32_t> y_row(static_cast<std::size_t>(nnz_ybus), 0);
        for (int32_t row = 0; row < n_bus; ++row) {
            for (int32_t pos = data.indptr[static_cast<std::size_t>(row)];
                 pos < data.indptr[static_cast<std::size_t>(row + 1)];
                 ++pos) {
                y_row[static_cast<std::size_t>(pos)] = row;
            }
        }

        std::vector<double> va(static_cast<std::size_t>(n_bus));
        std::vector<double> vm(static_cast<std::size_t>(n_bus));
        for (int32_t bus = 0; bus < n_bus; ++bus) {
            va[static_cast<std::size_t>(bus)] = std::arg(data.v0[static_cast<std::size_t>(bus)]);
            vm[static_cast<std::size_t>(bus)] = std::abs(data.v0[static_cast<std::size_t>(bus)]);
        }

        const std::vector<double> y_re = real_parts(data.ybus_data);
        const std::vector<double> y_im = imag_parts(data.ybus_data);
        const std::vector<double> sbus_re = real_parts(data.sbus);
        const std::vector<double> sbus_im = imag_parts(data.sbus);
        const std::vector<double> v_re = real_parts(data.v0);
        const std::vector<double> v_im = imag_parts(data.v0);

        d_y_row_ptr.assign(data.indptr.data(), data.indptr.size());
        d_y_col.assign(data.indices.data(), data.indices.size());
        d_y_row.assign(y_row.data(), y_row.size());
        d_y_re.assign(y_re.data(), y_re.size());
        d_y_im.assign(y_im.data(), y_im.size());
        d_pv.assign(data.pv.data(), data.pv.size());
        d_pq.assign(data.pq.data(), data.pq.size());
        d_sbus_re.assign(sbus_re.data(), sbus_re.size());
        d_sbus_im.assign(sbus_im.data(), sbus_im.size());
        d_v_re.assign(v_re.data(), v_re.size());
        d_v_im.assign(v_im.data(), v_im.size());
        d_va.assign(va.data(), va.size());
        d_vm.assign(vm.data(), vm.size());
        d_v_re_backup.resize(static_cast<std::size_t>(n_bus));
        d_v_im_backup.resize(static_cast<std::size_t>(n_bus));
        d_va_backup.resize(static_cast<std::size_t>(n_bus));
        d_vm_backup.resize(static_cast<std::size_t>(n_bus));
        d_ibus_re.resize(static_cast<std::size_t>(n_bus));
        d_ibus_im.resize(static_cast<std::size_t>(n_bus));
        d_F.resize(static_cast<std::size_t>(dimF));
        d_F_backup.resize(static_cast<std::size_t>(dimF));
        d_norm_inf.resize(1);
        d_dx.resize(static_cast<std::size_t>(dimF));
        d_prev_dx.resize(static_cast<std::size_t>(dimF));
        d_dx_gmres_shadow.resize(static_cast<std::size_t>(dimF));
        d_dx_cudss_shadow.resize(static_cast<std::size_t>(dimF));
        d_dx_diff_shadow.resize(static_cast<std::size_t>(dimF));
        d_ax.resize(static_cast<std::size_t>(dimF));
        d_linear_residual.resize(static_cast<std::size_t>(dimF));
        d_bad_count.resize(1);

        const YbusView ybus = data.ybus();
        const JacobianIndexing indexing =
            make_jacobian_indexing(n_bus, data.pv.data(), n_pv, data.pq.data(), n_pq);
        const JacobianPattern pattern = JacobianPatternGenerator().generate(ybus, indexing);
        const JacobianScatterMap maps = JacobianMapBuilder().build(ybus, indexing, pattern);

        j_pattern.rows = pattern.dim;
        j_pattern.cols = pattern.dim;
        j_pattern.row_ptr = pattern.row_ptr;
        j_pattern.col_idx = pattern.col_idx;
        j_pattern.values.assign(static_cast<std::size_t>(pattern.nnz), 0.0);

        d_J_row_ptr.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
        d_J_col_idx.assign(pattern.col_idx.data(), pattern.col_idx.size());
        d_J_values.resize(static_cast<std::size_t>(pattern.nnz));
        d_mapJ11.assign(maps.mapJ11.data(), maps.mapJ11.size());
        d_mapJ12.assign(maps.mapJ12.data(), maps.mapJ12.size());
        d_mapJ21.assign(maps.mapJ21.data(), maps.mapJ21.size());
        d_mapJ22.assign(maps.mapJ22.data(), maps.mapJ22.size());
        d_diagJ11.assign(maps.diagJ11.data(), maps.diagJ11.size());
        d_diagJ12.assign(maps.diagJ12.data(), maps.diagJ12.size());
        d_diagJ21.assign(maps.diagJ21.data(), maps.diagJ21.size());
        d_diagJ22.assign(maps.diagJ22.data(), maps.diagJ22.size());
    }

    void backup_state_and_rhs()
    {
        CUITER_CUDA_CHECK(cudaMemcpy(d_v_re_backup.data(),
                                     d_v_re.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_v_im_backup.data(),
                                     d_v_im.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_va_backup.data(),
                                     d_va.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_vm_backup.data(),
                                     d_vm.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_F_backup.data(),
                                     d_F.data(),
                                     static_cast<std::size_t>(dimF) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }

    void restore_state_and_rhs()
    {
        CUITER_CUDA_CHECK(cudaMemcpy(d_v_re.data(),
                                     d_v_re_backup.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_v_im.data(),
                                     d_v_im_backup.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_va.data(),
                                     d_va_backup.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_vm.data(),
                                     d_vm_backup.data(),
                                     static_cast<std::size_t>(n_bus) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        CUITER_CUDA_CHECK(cudaMemcpy(d_F.data(),
                                     d_F_backup.data(),
                                     static_cast<std::size_t>(dimF) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }
};

void validate_options(const HybridNrOptions& options)
{
    if (options.max_nr_iters <= 0 || options.cudss_bootstrap_iters < 0 ||
        options.gmres_restart <= 0 || options.gmres_max_iters <= 0) {
        throw std::runtime_error("invalid hybrid NR iteration options");
    }
    if (options.solver != "pure_cudss" && options.solver != "hybrid") {
        throw std::runtime_error("--solver must be pure_cudss or hybrid");
    }
    if (options.middle_solver != "gmres_block_jacobi" &&
        options.middle_solver != "mr1_block_jacobi" &&
        options.middle_solver != "mr1_block_jacobi_coarse" &&
        options.middle_solver != "mr2_block_jacobi_coarse" &&
        options.middle_solver != "bicgstab_block_jacobi" &&
        options.middle_solver != "bicgstab_block_jacobi_a0" &&
        options.middle_solver != "bicgstab_block_jacobi_a1" &&
        options.middle_solver != "bicgstab_block_jacobi_a0_device" &&
        options.middle_solver != "bicgstab_block_jacobi_a1_device" &&
        options.middle_solver != "bicgstab_block_jacobi_j11_device" &&
        options.middle_solver != "bicgstab_block_jacobi_bpbpp_refine" &&
        options.middle_solver != "bicgstab_block_ilu0" &&
        options.middle_solver != "gmres_block_ilu0" &&
        options.middle_solver != "ginkgo_parilut_bicgstab" &&
        options.middle_solver != "fdlf_bpbpp_2round" &&
        options.middle_solver != "stale_R0" &&
        options.middle_solver != "stale_R1_Richardson" &&
        options.middle_solver != "stale_R2_Richardson" &&
        options.middle_solver != "stale_R1_BiCGSTAB1_no_prec" &&
        options.middle_solver != "stale_R1_BiCGSTAB2_no_prec" &&
        options.middle_solver != "stale_R1_BiCGSTAB1_stale_prec" &&
        options.middle_solver != "stale_R1_BiCGSTAB2_stale_prec" &&
        options.middle_solver != "stale_R1_GMRES1_stale_prec" &&
        options.middle_solver != "stale_R1_GMRES2_stale_prec" &&
        options.middle_solver != "stale_GMRES1" &&
        options.middle_solver != "stale_BJ1" &&
        options.middle_solver != "stale_GMRES1_refresh") {
        throw std::runtime_error("unsupported middle_solver: " + options.middle_solver);
    }
    if (options.preconditioner != "metis_block_jacobi" &&
        options.preconditioner != "metis_block_jacobi_coarse" &&
        options.preconditioner != "ras_overlap1" &&
        options.preconditioner != "block_ilu0" &&
        options.preconditioner != "ginkgo_parilut" &&
        options.preconditioner != "none") {
        throw std::runtime_error("unsupported preconditioner: " + options.preconditioner);
    }
    if (options.block_size != 4 && options.block_size != 8 &&
        options.block_size != 16 && options.block_size != 32 && options.block_size != 64) {
        throw std::runtime_error("block_size must be 4, 8, 16, 32, or 64");
    }
    if (options.block_precision != "fp32" && options.block_precision != "fp64") {
        throw std::runtime_error("block_precision must be fp32 or fp64");
    }
    if (options.coarse_vars_per_block != 1 && options.coarse_vars_per_block != 2) {
        throw std::runtime_error("coarse_vars_per_block must be 1 or 2");
    }
    if (options.coarse_refresh != "bootstrap_only" &&
        options.coarse_refresh != "after_cudss_fallback" &&
        options.coarse_refresh != "every_iter") {
        throw std::runtime_error("unsupported coarse refresh policy: " + options.coarse_refresh);
    }
    if (options.dx_safety_check != "full" &&
        options.dx_safety_check != "nonfinite" &&
        options.dx_safety_check != "off") {
        throw std::runtime_error("unsupported dx_safety_check: " + options.dx_safety_check);
    }
    if (options.coarse_precision != "fp32" && options.coarse_precision != "fp64") {
        throw std::runtime_error("coarse_precision must be fp32 or fp64");
    }
    if (options.linear_scaling != "none" &&
        options.linear_scaling != "ruiz" &&
        options.linear_scaling != "ruiz_row_col" &&
        options.linear_scaling != "field" &&
        options.linear_scaling != "field_wise") {
        throw std::runtime_error("linear_scaling must be none, ruiz, or field");
    }
    if (options.partition_mode != "unknown_metis" &&
        options.partition_mode != "bus_weighted_metis") {
        throw std::runtime_error("partition_mode must be unknown_metis or bus_weighted_metis");
    }
    if (options.partition_mode == "bus_weighted_metis") {
        if (options.middle_solver != "mr1_block_jacobi" ||
            options.preconditioner != "metis_block_jacobi") {
            throw std::runtime_error("bus weighted partition experiment requires MR1 block-Jacobi without coarse");
        }
        if (options.bus_edge_weight != "jacobian_frobenius" ||
            options.bus_edge_weight_scale <= 0.0 ||
            options.bus_edge_weight_clamp <= 0 ||
            options.target_block_unknowns <= 0) {
            throw std::runtime_error("invalid bus weighted partition options");
        }
    }
    if (options.linear_scaling == "ruiz" || options.linear_scaling == "ruiz_row_col") {
        if (options.middle_solver != "mr1_block_jacobi" ||
            options.preconditioner != "metis_block_jacobi") {
            throw std::runtime_error("linear scaling experiment requires MR1 block-Jacobi without coarse");
        }
        if (options.scaling_iters <= 0 || options.scaling_clamp < 1.0 ||
            options.scaling_eps <= 0.0 || !std::isfinite(options.scaling_clamp) ||
            !std::isfinite(options.scaling_eps)) {
            throw std::runtime_error("invalid linear scaling options");
        }
        if (options.scaling_norm != "l2") {
            throw std::runtime_error("only scaling_norm=l2 is implemented");
        }
    }
    if (options.linear_scaling == "field" || options.linear_scaling == "field_wise") {
        if (options.middle_solver != "bicgstab_block_jacobi" ||
            options.preconditioner != "metis_block_jacobi") {
            throw std::runtime_error("field scaling experiment requires BiCGSTAB block-Jacobi without coarse");
        }
        if (options.scaling_iters <= 0 || options.scaling_clamp < 1.0 ||
            options.scaling_eps <= 0.0 || !std::isfinite(options.scaling_clamp) ||
            !std::isfinite(options.scaling_eps)) {
            throw std::runtime_error("invalid field scaling options");
        }
    }
    if (options.previous_dx_warm_start) {
        if (options.middle_solver != "bicgstab_block_jacobi" ||
            options.preconditioner != "metis_block_jacobi" ||
            options.linear_scaling != "none") {
            throw std::runtime_error(
                "previous dx warm start requires unscaled BiCGSTAB block-Jacobi without coarse");
        }
    }
    if (options.fallback_policy != "off" && options.fallback_policy != "immediate" &&
        options.fallback_policy != "after_two_failures") {
        throw std::runtime_error("fallback_policy must be off, immediate, or after_two_failures");
    }
    if (options.bj_setup != "every_middle" &&
        options.bj_setup != "value_update_only" &&
        options.bj_setup != "numeric_reuse_after_full_cudss" &&
        options.bj_setup != "reuse_after_full_cudss" &&
        options.bj_setup != "reuse_for_2_middle_steps") {
        throw std::runtime_error(
            "bj_setup must be every_middle, value_update_only, numeric_reuse_after_full_cudss, "
            "reuse_after_full_cudss, or reuse_for_2_middle_steps");
    }
    if (options.max_middle_accepts < -1 || options.max_a1_middle_accepts < -1) {
        throw std::runtime_error("middle accept caps must be -1 or non-negative");
    }
    if (options.force_gmres_min_steps < 0) {
        throw std::runtime_error("force_gmres_min_steps must be nonnegative");
    }
    if (options.middle_solver == "bicgstab_block_jacobi" ||
        options.middle_solver == "bicgstab_block_jacobi_a0" ||
        options.middle_solver == "bicgstab_block_jacobi_a1" ||
        options.middle_solver == "bicgstab_block_jacobi_a0_device" ||
        options.middle_solver == "bicgstab_block_jacobi_a1_device" ||
        options.middle_solver == "bicgstab_block_jacobi_j11_device") {
        if (options.preconditioner != "metis_block_jacobi" &&
            options.preconditioner != "metis_block_jacobi_coarse" &&
            options.preconditioner != "ras_overlap1") {
            throw std::runtime_error(
                "BiCGSTAB experiment requires metis_block_jacobi, metis_block_jacobi_coarse, or ras_overlap1");
        }
        if (options.bicgstab_iters <= 0) {
            throw std::runtime_error("bicgstab_iters must be positive");
        }
    }
    if (options.middle_solver == "bicgstab_block_ilu0" ||
        options.middle_solver == "gmres_block_ilu0") {
        if (options.preconditioner != "block_ilu0") {
            throw std::runtime_error("block ILU0 pilot requires preconditioner=block_ilu0");
        }
        if (options.middle_solver == "bicgstab_block_ilu0" && options.bicgstab_iters <= 0) {
            throw std::runtime_error("bicgstab_iters must be positive");
        }
        if (options.middle_solver == "gmres_block_ilu0" && options.gmres_max_iters <= 0) {
            throw std::runtime_error("gmres_max_iters must be positive");
        }
        if (options.linear_scaling != "none" || options.previous_dx_warm_start) {
            throw std::runtime_error("block ILU0 pilot does not mix scaling or previous dx warm start");
        }
    }
#if !defined(CUITER_WITH_GINKGO)
    if (options.middle_solver == "ginkgo_parilut_bicgstab") {
        throw std::runtime_error("Ginkgo was not available at build time");
    }
#endif
    if (options.middle_solver == "ginkgo_parilut_bicgstab") {
        if (options.preconditioner != "ginkgo_parilut") {
            throw std::runtime_error("ginkgo_parilut_bicgstab requires preconditioner=ginkgo_parilut");
        }
        if (options.bicgstab_iters <= 0 || options.ginkgo_parilut_iters <= 0 ||
            options.ginkgo_parilut_fill <= 0.0 ||
            !std::isfinite(options.ginkgo_parilut_fill)) {
            throw std::runtime_error("invalid Ginkgo ParILUT options");
        }
    }
    for (double factor : options.damping_factors) {
        if (!std::isfinite(factor) || factor <= 0.0 || factor > 1.0) {
            throw std::runtime_error("damping factors must be finite values in (0, 1]");
        }
    }
    if (options.enable_scaled_mr1_step) {
        if (options.middle_solver != "mr1_block_jacobi_coarse") {
            throw std::runtime_error("scaled MR1 step requires middle_solver=mr1_block_jacobi_coarse");
        }
        if (options.scaled_mr1_gamma_candidates.empty()) {
            throw std::runtime_error("scaled MR1 gamma candidate list must not be empty");
        }
        for (double gamma : options.scaled_mr1_gamma_candidates) {
            if (!std::isfinite(gamma) || gamma <= 0.0) {
                throw std::runtime_error("scaled MR1 gamma candidates must be finite positive values");
            }
        }
    }
    if (options.global_correction != "none" && options.global_correction != "post") {
        throw std::runtime_error("global_correction must be none or post");
    }
    if (options.global_basis_source != "fallback" &&
        options.global_basis_source != "diagnostic") {
        throw std::runtime_error("global_basis_source must be fallback or diagnostic");
    }
    if (options.global_correction_acceptance != "always" &&
        options.global_correction_acceptance != "residual") {
        throw std::runtime_error("global_correction_acceptance must be always or residual");
    }
    if (options.global_rank < 0 || options.global_orth_tol <= 0.0 ||
        !std::isfinite(options.global_orth_tol)) {
        throw std::runtime_error("invalid global correction rank or orthogonalization tolerance");
    }
    if (options.global_correction == "post" && options.middle_solver != "gmres_block_ilu0") {
        throw std::runtime_error("global post-correction is currently implemented for gmres_block_ilu0");
    }
    if (options.field_gain_correction != "none" &&
        options.field_gain_correction != "ls2") {
        throw std::runtime_error("field_gain_correction must be none or ls2");
    }
    if (options.theta_j11_correction != "none" &&
        options.theta_j11_correction != "scalar" &&
        options.theta_j11_correction != "gmres") {
        throw std::runtime_error("theta_j11_correction must be none, scalar, or gmres");
    }
    if (options.field_gain_theta_max <= 0.0 || options.field_gain_vmax <= 0.0 ||
        options.field_gain_trust_ratio <= 0.0 ||
        options.theta_j11_correction_trust_ratio <= 0.0 ||
        options.theta_j11_gmres_maxit <= 0) {
        throw std::runtime_error("invalid field gain or theta correction options");
    }
    if ((options.field_gain_correction != "none" ||
         options.theta_j11_correction != "none") &&
        options.middle_solver != "gmres_block_ilu0") {
        throw std::runtime_error("field gain and theta correction are currently implemented for gmres_block_ilu0");
    }
}

void compute_segment_norms(MinimalNrDeviceContext& ctx,
                           cublasHandle_t cublas,
                           int32_t n,
                           const double* d_values,
                           double& norm_inf,
                           double& norm_two)
{
    if (n <= 0) {
        norm_inf = 0.0;
        norm_two = 0.0;
        return;
    }
    launch_reduce_abs_max(n, d_values, ctx.d_norm_inf.data());
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, n, d_values, 1, &norm_two));
    CUITER_CUDA_CHECK(cudaMemcpy(&norm_inf, ctx.d_norm_inf.data(), sizeof(double), cudaMemcpyDeviceToHost));
}

MismatchNorms compute_mismatch(MinimalNrDeviceContext& ctx,
                               cublasHandle_t cublas,
                               bool compute_components = false)
{
    MismatchNorms norms;
    launch_compute_ibus(ctx.n_bus,
                        ctx.d_y_row_ptr.data(),
                        ctx.d_y_col.data(),
                        ctx.d_y_re.data(),
                        ctx.d_y_im.data(),
                        ctx.d_v_re.data(),
                        ctx.d_v_im.data(),
                        ctx.d_ibus_re.data(),
                        ctx.d_ibus_im.data());
    launch_compute_mismatch_from_ibus(ctx.dimF,
                                      ctx.n_bus,
                                      ctx.n_pv,
                                      ctx.n_pq,
                                      ctx.d_v_re.data(),
                                      ctx.d_v_im.data(),
                                      ctx.d_ibus_re.data(),
                                      ctx.d_ibus_im.data(),
                                      ctx.d_sbus_re.data(),
                                      ctx.d_sbus_im.data(),
                                      ctx.d_pv.data(),
                                      ctx.d_pq.data(),
                                      ctx.d_F.data());
    compute_segment_norms(ctx, cublas, ctx.dimF, ctx.d_F.data(), norms.inf, norms.two);
    if (compute_components) {
        compute_segment_norms(ctx, cublas, ctx.n_pvpq, ctx.d_F.data(), norms.p_inf, norms.p_two);
        compute_segment_norms(ctx,
                              cublas,
                              ctx.n_pq,
                              ctx.d_F.data() + ctx.n_pvpq,
                              norms.q_inf,
                              norms.q_two);
    }
    return norms;
}

void fill_jacobian(MinimalNrDeviceContext& ctx)
{
    launch_fill_jacobian(ctx.nnz_ybus,
                         ctx.j_pattern.nnz(),
                         ctx.n_bus,
                         ctx.d_y_re.data(),
                         ctx.d_y_im.data(),
                         ctx.d_y_row.data(),
                         ctx.d_y_col.data(),
                         ctx.d_y_row_ptr.data(),
                         ctx.d_v_re.data(),
                         ctx.d_v_im.data(),
                         ctx.d_vm.data(),
                         ctx.d_ibus_re.data(),
                         ctx.d_ibus_im.data(),
                         ctx.d_mapJ11.data(),
                         ctx.d_mapJ21.data(),
                         ctx.d_mapJ12.data(),
                         ctx.d_mapJ22.data(),
                         ctx.d_diagJ11.data(),
                         ctx.d_diagJ21.data(),
                         ctx.d_diagJ12.data(),
                         ctx.d_diagJ22.data(),
                         ctx.d_J_values.data());
}

double compute_linear_residual_for_dx(MinimalNrDeviceContext& ctx,
                                      cublasHandle_t cublas,
                                      const double* d_dx,
                                      double rhs_norm,
                                      double& relative)
{
    double absolute = 0.0;
    cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                     ctx.d_J_row_ptr.data(),
                                     ctx.d_J_col_idx.data(),
                                     ctx.d_J_values.data(),
                                     d_dx,
                                     ctx.d_ax.data());
    cuiter::kernels::launch_residual(ctx.dimF, ctx.d_F.data(), ctx.d_ax.data(), ctx.d_linear_residual.data());
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, ctx.dimF, ctx.d_linear_residual.data(), 1, &absolute));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    relative = absolute / rhs_norm;
    return absolute;
}

double compute_linear_residual(MinimalNrDeviceContext& ctx,
                               cublasHandle_t cublas,
                               double rhs_norm,
                               double& relative)
{
    return compute_linear_residual_for_dx(ctx, cublas, ctx.d_dx.data(), rhs_norm, relative);
}

bool dx_is_bad(MinimalNrDeviceContext& ctx,
               cublasHandle_t cublas,
               double mismatch_inf,
               const std::string& safety_mode)
{
    if (safety_mode == "off") {
        return false;
    }
    launch_count_nonfinite(ctx.dimF, ctx.d_dx.data(), ctx.d_bad_count.data());
    int32_t bad_count = 0;
    CUITER_CUDA_CHECK(cudaMemcpy(&bad_count, ctx.d_bad_count.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (bad_count > 0) {
        return true;
    }
    if (safety_mode == "nonfinite") {
        return false;
    }
    double dx_norm = 0.0;
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, ctx.dimF, ctx.d_dx.data(), 1, &dx_norm));
    return mismatch_inf > 1.0e-10 && dx_norm <= 1.0e-30;
}

double vector_norm2(cublasHandle_t cublas, int32_t n, const double* d_x)
{
    if (n <= 0) {
        return 0.0;
    }
    double norm = 0.0;
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, n, d_x, 1, &norm));
    return norm;
}

bool is_stale_refinement_solver(const std::string& middle_solver)
{
    return middle_solver == "stale_R0" ||
           middle_solver == "stale_R1_Richardson" ||
           middle_solver == "stale_R2_Richardson" ||
           middle_solver == "stale_R1_BiCGSTAB1_no_prec" ||
           middle_solver == "stale_R1_BiCGSTAB2_no_prec" ||
           middle_solver == "stale_R1_BiCGSTAB1_stale_prec" ||
           middle_solver == "stale_R1_BiCGSTAB2_stale_prec" ||
           middle_solver == "stale_R1_GMRES1_stale_prec" ||
           middle_solver == "stale_R1_GMRES2_stale_prec" ||
           middle_solver == "stale_GMRES1" ||
           middle_solver == "stale_BJ1" ||
           middle_solver == "stale_GMRES1_refresh";
}

int32_t stale_bicgstab_iters(const std::string& middle_solver)
{
    if (middle_solver == "stale_R1_BiCGSTAB1_no_prec" ||
        middle_solver == "stale_R1_BiCGSTAB1_stale_prec") {
        return 1;
    }
    if (middle_solver == "stale_R1_BiCGSTAB2_no_prec" ||
        middle_solver == "stale_R1_BiCGSTAB2_stale_prec") {
        return 2;
    }
    return 0;
}

bool uses_stale_prec_bicgstab(const std::string& middle_solver)
{
    return middle_solver == "stale_R1_BiCGSTAB1_stale_prec" ||
           middle_solver == "stale_R1_BiCGSTAB2_stale_prec";
}

int32_t stale_gmres_iters(const std::string& middle_solver)
{
    if (middle_solver == "stale_R1_GMRES1_stale_prec" ||
        middle_solver == "stale_GMRES1" ||
        middle_solver == "stale_GMRES1_refresh") {
        return 1;
    }
    if (middle_solver == "stale_R1_GMRES2_stale_prec") {
        return 2;
    }
    return 0;
}

cuiter::GmresSolverOptions make_stale_bicgstab_options(const HybridNrOptions& options,
                                                       int32_t iterations)
{
    cuiter::GmresSolverOptions solver_options;
    solver_options.max_iters = iterations;
    solver_options.restart = 1;
    solver_options.rel_tolerance = 0.0;
    solver_options.abs_tolerance = 0.0;
    solver_options.preconditioner = "none";
    solver_options.use_right_preconditioning = true;
    solver_options.compute_true_residual = true;
    solver_options.minimize_host_sync = true;
    solver_options.use_bicgstab_fixed_path = true;
    solver_options.use_bicgstab_fused_fixed2 =
        options.bicgstab_fused_fixed2 && iterations == 2;
    return solver_options;
}

cuiter::GmresSolverOptions make_stale_bj_mr1_options(const HybridNrOptions& options,
                                                     const MinimalNrDeviceContext& ctx)
{
    cuiter::GmresSolverOptions solver_options;
    solver_options.max_iters = 1;
    solver_options.restart = 1;
    solver_options.rel_tolerance = 0.0;
    solver_options.abs_tolerance = 0.0;
    solver_options.preconditioner = "metis_block_jacobi";
    solver_options.block_size = options.block_size;
    solver_options.use_fp32_preconditioner = options.block_precision == "fp32";
    solver_options.use_right_preconditioning = true;
    solver_options.compute_true_residual = true;
    solver_options.minimize_host_sync = true;
    solver_options.block_jacobi_apply = cuiter::parse_block_jacobi_apply_mode(options.block_apply);
    solver_options.use_mr1_fast_path = true;
    solver_options.partition_mode = options.partition_mode;
    solver_options.bus_edge_weight = options.bus_edge_weight;
    solver_options.bus_edge_weight_scale = options.bus_edge_weight_scale;
    solver_options.bus_edge_weight_clamp = options.bus_edge_weight_clamp;
    solver_options.target_block_unknowns = options.target_block_unknowns;
    solver_options.n_bus = ctx.n_bus;
    solver_options.index_to_bus = ctx.index_to_bus;
    solver_options.index_field = ctx.index_field;
    return solver_options;
}

double compute_residual_into_linear_buffer(MinimalNrDeviceContext& ctx,
                                           cublasHandle_t cublas,
                                           const double* d_x,
                                           double rhs_norm,
                                           double& relative,
                                           double& spmv_seconds,
                                           double& residual_seconds)
{
    spmv_seconds += timed_with_sync([&] {
        cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                         ctx.d_J_row_ptr.data(),
                                         ctx.d_J_col_idx.data(),
                                         ctx.d_J_values.data(),
                                         d_x,
                                         ctx.d_ax.data());
    });
    residual_seconds += timed_with_sync([&] {
        cuiter::kernels::launch_residual(ctx.dimF,
                                         ctx.d_F.data(),
                                         ctx.d_ax.data(),
                                         ctx.d_linear_residual.data());
    });
    double absolute = 0.0;
    CUITER_CUBLAS_CHECK(cublasDnrm2(cublas,
                                    ctx.dimF,
                                    ctx.d_linear_residual.data(),
                                    1,
                                    &absolute));
    rhs_norm = std::max(rhs_norm, std::numeric_limits<double>::min());
    relative = absolute / rhs_norm;
    return absolute;
}

struct StalePrecBicgstabWorkspace {
    cuiter::DeviceBuffer<double> d_r_hat;
    cuiter::DeviceBuffer<double> d_r;
    cuiter::DeviceBuffer<double> d_p;
    cuiter::DeviceBuffer<double> d_p_hat;
    cuiter::DeviceBuffer<double> d_s;
    cuiter::DeviceBuffer<double> d_s_hat;
    cuiter::DeviceBuffer<double> d_v;
    cuiter::DeviceBuffer<double> d_t;
    cuiter::DeviceBuffer<double> d_delta;

    void ensure(int32_t n)
    {
        d_r_hat.resize(static_cast<std::size_t>(n));
        d_r.resize(static_cast<std::size_t>(n));
        d_p.resize(static_cast<std::size_t>(n));
        d_p_hat.resize(static_cast<std::size_t>(n));
        d_s.resize(static_cast<std::size_t>(n));
        d_s_hat.resize(static_cast<std::size_t>(n));
        d_v.resize(static_cast<std::size_t>(n));
        d_t.resize(static_cast<std::size_t>(n));
        d_delta.resize(static_cast<std::size_t>(n));
    }
};

struct StalePrecGmresWorkspace {
    cuiter::DeviceBuffer<double> d_v_basis;
    cuiter::DeviceBuffer<double> d_z_basis;
    cuiter::DeviceBuffer<double> d_w;
    cuiter::DeviceBuffer<double> d_dots;

    void ensure(int32_t n, int32_t max_iters)
    {
        max_iters = std::max(1, std::min(max_iters, 2));
        d_v_basis.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(max_iters + 1));
        d_z_basis.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(max_iters));
        d_w.resize(static_cast<std::size_t>(n));
        d_dots.resize(2);
    }

    double* v_col(int32_t n, int32_t col)
    {
        return d_v_basis.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(col);
    }

    double* z_col(int32_t n, int32_t col)
    {
        return d_z_basis.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(col);
    }
};

double apply_stale_preconditioner(MinimalNrDeviceContext& ctx,
                                  DirectCudssSolver& stale_cudss,
                                  const double* d_rhs,
                                  double* d_out)
{
    const double seconds = timed_with_sync([&] {
        if (d_rhs != ctx.d_linear_residual.data()) {
            CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_linear_residual.data(),
                                         d_rhs,
                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        }
    }) + stale_cudss.solve() +
        timed_with_sync([&] {
            CUITER_CUDA_CHECK(cudaMemcpy(d_out,
                                         ctx.d_dx_cudss_shadow.data(),
                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        });
    return seconds;
}

void run_stale_preconditioned_bicgstab(MinimalNrDeviceContext& ctx,
                                       cublasHandle_t cublas,
                                       DirectCudssSolver& stale_cudss,
                                       int32_t max_iters,
                                       StalePrecBicgstabWorkspace& work,
                                       HybridNrIterationLog& log)
{
    const int32_t n = ctx.dimF;
    work.ensure(n);
    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    bool breakdown = false;

    log.bicgstab_update_seconds += timed_with_sync([&] {
        cuiter::kernels::launch_copy(n, ctx.d_linear_residual.data(), work.d_r.data());
        cuiter::kernels::launch_copy(n, ctx.d_linear_residual.data(), work.d_r_hat.data());
        cuiter::kernels::launch_set_zero(n, work.d_p.data());
        cuiter::kernels::launch_set_zero(n, work.d_v.data());
        cuiter::kernels::launch_set_zero(n, work.d_delta.data());
    });

    for (int32_t iter = 0; iter < max_iters; ++iter) {
        double rho = 0.0;
        log.bicgstab_dot_reduction_seconds += timed_with_sync([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas,
                                           n,
                                           work.d_r_hat.data(),
                                           1,
                                           work.d_r.data(),
                                           1,
                                           &rho));
        });
        if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
            log.stop_reason = "stale_prec_bicgstab_rho_breakdown";
            breakdown = true;
            break;
        }

        log.bicgstab_update_seconds += timed_with_sync([&] {
            if (iter == 0) {
                cuiter::kernels::launch_copy(n, work.d_r.data(), work.d_p.data());
            } else {
                const double beta = (rho / rho_old) * (alpha / omega);
                cuiter::kernels::launch_bicgstab_update_p(
                    n, work.d_r.data(), beta, omega, work.d_v.data(), work.d_p.data());
            }
        });

        const double p_solve_seconds =
            apply_stale_preconditioner(ctx, stale_cudss, work.d_p.data(), work.d_p_hat.data());
        log.stale_prec_solve_seconds += p_solve_seconds;
        log.stale_prec_solve_count += 1;
        log.stale_solve_calls += 1;

        log.bicgstab_spmv_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                             ctx.d_J_row_ptr.data(),
                                             ctx.d_J_col_idx.data(),
                                             ctx.d_J_values.data(),
                                             work.d_p_hat.data(),
                                             work.d_v.data());
        });
        log.current_j_spmv_calls += 1;

        double rhat_v = 0.0;
        log.bicgstab_dot_reduction_seconds += timed_with_sync([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas,
                                           n,
                                           work.d_r_hat.data(),
                                           1,
                                           work.d_v.data(),
                                           1,
                                           &rhat_v));
        });
        if (!std::isfinite(rhat_v) || std::abs(rhat_v) <= std::numeric_limits<double>::min()) {
            log.stop_reason = "stale_prec_bicgstab_alpha_breakdown";
            breakdown = true;
            break;
        }
        alpha = rho / rhat_v;
        if (!std::isfinite(alpha)) {
            log.stop_reason = "stale_prec_bicgstab_alpha_nan";
            breakdown = true;
            break;
        }

        log.bicgstab_update_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_residual_scaled(
                n, work.d_r.data(), work.d_v.data(), alpha, work.d_s.data());
        });

        const double s_solve_seconds =
            apply_stale_preconditioner(ctx, stale_cudss, work.d_s.data(), work.d_s_hat.data());
        log.stale_prec_solve_seconds += s_solve_seconds;
        log.stale_prec_solve_count += 1;
        log.stale_solve_calls += 1;

        log.bicgstab_spmv_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                             ctx.d_J_row_ptr.data(),
                                             ctx.d_J_col_idx.data(),
                                             ctx.d_J_values.data(),
                                             work.d_s_hat.data(),
                                             work.d_t.data());
        });
        log.current_j_spmv_calls += 1;

        double ts = 0.0;
        double tt = 0.0;
        log.bicgstab_dot_reduction_seconds += timed_with_sync([&] {
            CUITER_CUBLAS_CHECK(cublasDdot(cublas,
                                           n,
                                           work.d_t.data(),
                                           1,
                                           work.d_s.data(),
                                           1,
                                           &ts));
            CUITER_CUBLAS_CHECK(cublasDdot(cublas,
                                           n,
                                           work.d_t.data(),
                                           1,
                                           work.d_t.data(),
                                           1,
                                           &tt));
        });
        if (!std::isfinite(ts) || !std::isfinite(tt) ||
            tt <= std::numeric_limits<double>::min()) {
            log.stop_reason = "stale_prec_bicgstab_omega_breakdown";
            breakdown = true;
            break;
        }
        omega = ts / tt;
        if (!std::isfinite(omega)) {
            log.stop_reason = "stale_prec_bicgstab_omega_nan";
            breakdown = true;
            break;
        }

        log.bicgstab_update_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_bicgstab_update_x_r(n,
                                                        alpha,
                                                        work.d_p_hat.data(),
                                                        omega,
                                                        work.d_s_hat.data(),
                                                        work.d_s.data(),
                                                        work.d_t.data(),
                                                        work.d_delta.data(),
                                                        work.d_r.data());
        });

        rho_old = rho;
        log.bicgstab_refinement_iters = iter + 1;
        if (std::abs(omega) <= std::numeric_limits<double>::min()) {
            log.stop_reason = "stale_prec_bicgstab_omega_zero";
            breakdown = true;
            break;
        }
    }

    if (!breakdown && log.stop_reason.empty()) {
        log.stop_reason = "stale_prec_bicgstab_fixed_iter";
    }
    log.bicgstab_update_seconds += timed_with_sync([&] {
        const double one = 1.0;
        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                        n,
                                        &one,
                                        work.d_delta.data(),
                                        1,
                                        ctx.d_dx.data(),
                                        1));
    });
    log.bicgstab_total_seconds +=
        log.bicgstab_spmv_seconds + log.bicgstab_dot_reduction_seconds +
        log.bicgstab_update_seconds + log.stale_prec_solve_seconds;
}

void run_stale_preconditioned_gmres(MinimalNrDeviceContext& ctx,
                                    cublasHandle_t cublas,
                                    DirectCudssSolver& stale_cudss,
                                    int32_t max_iters,
                                    StalePrecGmresWorkspace& work,
                                    HybridNrIterationLog& log)
{
    const int32_t n = ctx.dimF;
    max_iters = std::max(1, std::min(max_iters, 2));
    work.ensure(n, max_iters);
    const double total_spmv_before = log.gmres_spmv_seconds;
    const double total_dot_before = log.gmres_dot_seconds;
    const double total_orth_before = log.gmres_orthogonalization_seconds;
    const double total_update_before = log.gmres_update_seconds;
    const double total_prec_before = log.stale_prec_solve_seconds;

    if (max_iters == 1) {
        double* z = work.z_col(n, 0);
        const double solve_seconds =
            apply_stale_preconditioner(ctx, stale_cudss, ctx.d_linear_residual.data(), z);
        log.stale_prec_solve_seconds += solve_seconds;
        log.stale_prec_solve_count += 1;
        log.stale_solve_calls += 1;

        log.gmres_spmv_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                             ctx.d_J_row_ptr.data(),
                                             ctx.d_J_col_idx.data(),
                                             ctx.d_J_values.data(),
                                             z,
                                             work.d_w.data());
        });
        log.current_j_spmv_calls += 1;

        const double dot_before = log.gmres_dot_seconds;
        log.gmres_dot_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_mr1_two_dot_reduction(
                n, work.d_w.data(), ctx.d_linear_residual.data(), work.d_dots.data());
        });
        double dots[2] = {0.0, 0.0};
        const auto copy_start = std::chrono::steady_clock::now();
        work.d_dots.copy_to(dots, 2);
        const double copy_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_start).count();
        log.gmres_dot_seconds += copy_seconds;
        log.gmres_scalar_sync_seconds += log.gmres_dot_seconds - dot_before;

        if (!std::isfinite(dots[0]) || !std::isfinite(dots[1]) ||
            dots[1] <= std::numeric_limits<double>::min()) {
            log.stop_reason = "stale_prec_gmres_mr1_breakdown";
            return;
        }
        const double alpha = dots[0] / dots[1];
        if (!std::isfinite(alpha)) {
            log.stop_reason = "stale_prec_gmres_mr1_alpha_nan";
            return;
        }
        log.gmres_update_seconds += timed_with_sync([&] {
            CUITER_CUBLAS_CHECK(cublasDaxpy(cublas, n, &alpha, z, 1, ctx.d_dx.data(), 1));
        });
        log.gmres_refinement_iters = 1;
        log.stop_reason = "stale_prec_gmres1_mr_fast";
        log.gmres_total_seconds +=
            (log.gmres_spmv_seconds - total_spmv_before) +
            (log.gmres_dot_seconds - total_dot_before) +
            (log.gmres_orthogonalization_seconds - total_orth_before) +
            (log.gmres_update_seconds - total_update_before) +
            (log.stale_prec_solve_seconds - total_prec_before);
        log.host_sync_seconds += log.gmres_scalar_sync_seconds;
        return;
    }

    double beta = 0.0;
    double reduction_before = log.gmres_dot_seconds;
    log.gmres_dot_seconds += timed_with_sync([&] {
        CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, n, ctx.d_linear_residual.data(), 1, &beta));
    });
    log.gmres_scalar_sync_seconds += log.gmres_dot_seconds - reduction_before;
    if (!std::isfinite(beta) || beta <= std::numeric_limits<double>::min()) {
        log.stop_reason = "stale_prec_gmres_zero_residual";
        return;
    }

    log.gmres_update_seconds += timed_with_sync([&] {
        cuiter::kernels::launch_scale_copy(n, 1.0 / beta, ctx.d_linear_residual.data(), work.v_col(n, 0));
        cuiter::kernels::launch_set_zero(n, work.z_col(n, 0));
        if (max_iters > 1) {
            cuiter::kernels::launch_set_zero(n, work.z_col(n, 1));
        }
    });

    double h00 = 0.0;
    double h10 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    double h21 = 0.0;
    int32_t basis_count = 0;

    for (int32_t j = 0; j < max_iters; ++j) {
        double* z_j = work.z_col(n, j);
        double* v_j = work.v_col(n, j);

        const double solve_seconds = apply_stale_preconditioner(ctx, stale_cudss, v_j, z_j);
        log.stale_prec_solve_seconds += solve_seconds;
        log.stale_prec_solve_count += 1;
        log.stale_solve_calls += 1;

        log.gmres_spmv_seconds += timed_with_sync([&] {
            cuiter::kernels::launch_csr_spmv(ctx.j_pattern.rows,
                                             ctx.d_J_row_ptr.data(),
                                             ctx.d_J_col_idx.data(),
                                             ctx.d_J_values.data(),
                                             z_j,
                                             work.d_w.data());
        });
        log.current_j_spmv_calls += 1;

        for (int32_t i = 0; i <= j; ++i) {
            double h = 0.0;
            const double dot_before = log.gmres_dot_seconds;
            log.gmres_dot_seconds += timed_with_sync([&] {
                CUITER_CUBLAS_CHECK(cublasDdot(cublas,
                                               n,
                                               work.d_w.data(),
                                               1,
                                               work.v_col(n, i),
                                               1,
                                               &h));
            });
            log.gmres_scalar_sync_seconds += log.gmres_dot_seconds - dot_before;
            if (i == 0 && j == 0) {
                h00 = h;
            } else if (i == 0 && j == 1) {
                h01 = h;
            } else if (i == 1 && j == 1) {
                h11 = h;
            }
            log.gmres_orthogonalization_seconds += timed_with_sync([&] {
                const double neg_h = -h;
                CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                                n,
                                                &neg_h,
                                                work.v_col(n, i),
                                                1,
                                                work.d_w.data(),
                                                1));
            });
        }

        double h_next = 0.0;
        const double norm_before = log.gmres_dot_seconds;
        log.gmres_dot_seconds += timed_with_sync([&] {
            CUITER_CUBLAS_CHECK(cublasDnrm2(cublas, n, work.d_w.data(), 1, &h_next));
        });
        log.gmres_scalar_sync_seconds += log.gmres_dot_seconds - norm_before;
        if (j == 0) {
            h10 = h_next;
        } else {
            h21 = h_next;
        }
        basis_count = j + 1;
        if (j + 1 < max_iters) {
            if (!std::isfinite(h_next) || h_next <= std::numeric_limits<double>::min()) {
                break;
            }
            log.gmres_update_seconds += timed_with_sync([&] {
                cuiter::kernels::launch_scale_copy(n, 1.0 / h_next, work.d_w.data(), work.v_col(n, j + 1));
            });
        }
    }

    double y0 = 0.0;
    double y1 = 0.0;
    if (basis_count == 1) {
        const double den = h00 * h00 + h10 * h10;
        if (den <= std::numeric_limits<double>::min() || !std::isfinite(den)) {
            log.stop_reason = "stale_prec_gmres_ls_breakdown";
            return;
        }
        y0 = beta * h00 / den;
    } else {
        const double g00 = h00 * h00 + h10 * h10;
        const double g01 = h00 * h01 + h10 * h11;
        const double g11 = h01 * h01 + h11 * h11 + h21 * h21;
        const double b0 = beta * h00;
        const double b1 = beta * h01;
        const double det = g00 * g11 - g01 * g01;
        if (std::abs(det) <= std::numeric_limits<double>::min() || !std::isfinite(det)) {
            log.stop_reason = "stale_prec_gmres_ls_breakdown";
            return;
        }
        y0 = (b0 * g11 - b1 * g01) / det;
        y1 = (g00 * b1 - g01 * b0) / det;
    }

    if (!std::isfinite(y0) || !std::isfinite(y1)) {
        log.stop_reason = "stale_prec_gmres_y_nan";
        return;
    }
    log.gmres_update_seconds += timed_with_sync([&] {
        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas, n, &y0, work.z_col(n, 0), 1, ctx.d_dx.data(), 1));
        if (basis_count > 1) {
            CUITER_CUBLAS_CHECK(cublasDaxpy(cublas, n, &y1, work.z_col(n, 1), 1, ctx.d_dx.data(), 1));
        }
    });
    log.gmres_refinement_iters = basis_count;
    if (log.stop_reason.empty()) {
        log.stop_reason = "stale_prec_gmres_fixed_iter";
    }
    log.gmres_total_seconds +=
        (log.gmres_spmv_seconds - total_spmv_before) +
        (log.gmres_dot_seconds - total_dot_before) +
        (log.gmres_orthogonalization_seconds - total_orth_before) +
        (log.gmres_update_seconds - total_update_before) +
        (log.stale_prec_solve_seconds - total_prec_before);
    log.host_sync_seconds += log.gmres_scalar_sync_seconds;
}

double vector_dot(cublasHandle_t cublas, int32_t n, const double* d_x, const double* d_y)
{
    if (n <= 0) {
        return 0.0;
    }
    double dot = 0.0;
    CUITER_CUBLAS_CHECK(cublasDdot(cublas, n, d_x, 1, d_y, 1, &dot));
    return dot;
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > std::numeric_limits<double>::min() ? numerator / denominator : 0.0;
}

double cosine_from_dot(double dot, double x_norm, double y_norm)
{
    const double denom = x_norm * y_norm;
    if (denom <= std::numeric_limits<double>::min()) {
        return 0.0;
    }
    return std::clamp(dot / denom, -1.0, 1.0);
}

double max_abs_values(MinimalNrDeviceContext& ctx, int32_t n, const double* d_values)
{
    double value = 0.0;
    if (n <= 0) {
        return value;
    }
    launch_reduce_abs_max(n, d_values, ctx.d_norm_inf.data());
    CUITER_CUDA_CHECK(cudaMemcpy(&value, ctx.d_norm_inf.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return value;
}

std::string sanitize_filename_token(std::string value)
{
    for (char& ch : value) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
        if (!ok) {
            ch = '_';
        }
    }
    return value;
}

double dump_device_vector(const std::filesystem::path& path,
                          int32_t n,
                          const double* d_values)
{
    if (n < 0 || d_values == nullptr) {
        throw std::runtime_error("dump_device_vector: invalid input");
    }
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    CUITER_CUDA_CHECK(cudaMemcpy(values.data(),
                                 d_values,
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open vector dump file: " + path.string());
    }
    out << std::setprecision(17);
    out << "type vector\n";
    out << "size " << n << "\n";
    out << "values\n";
    for (int32_t i = 0; i < n; ++i) {
        out << i << ' ' << values[static_cast<std::size_t>(i)] << '\n';
    }
    out.close();
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    return elapsed_seconds(start);
}

double dump_iteration_f(const HybridNrOptions& options,
                        const MinimalNrDeviceContext& ctx,
                        const std::string& case_name,
                        int32_t iter,
                        const std::string& stage)
{
    if (options.iteration_f_dump_dir.empty()) {
        return 0.0;
    }
    const std::string mode = sanitize_filename_token(options.middle_solver);
    const std::string stage_token = sanitize_filename_token(stage);
    const std::filesystem::path dir =
        std::filesystem::path(options.iteration_f_dump_dir) /
        sanitize_filename_token(case_name) /
        (mode + "_bs" + std::to_string(options.block_size));
    const std::filesystem::path path =
        dir / ("iter_" + std::to_string(iter) + "_F_" + stage_token + ".txt");
    return dump_device_vector(path, ctx.dimF, ctx.d_F.data());
}

DxComparisonMetrics compare_shadow_dx(MinimalNrDeviceContext& ctx, cublasHandle_t cublas)
{
    DxComparisonMetrics metrics;
    const double* d_g = ctx.d_dx_gmres_shadow.data();
    const double* d_d = ctx.d_dx_cudss_shadow.data();
    const double dx_g_norm = vector_norm2(cublas, ctx.dimF, d_g);
    const double dx_d_norm = vector_norm2(cublas, ctx.dimF, d_d);
    const double dx_dot = vector_dot(cublas, ctx.dimF, d_g, d_d);

    metrics.dx_norm_ratio = safe_ratio(dx_g_norm, dx_d_norm);
    metrics.dx_cosine = cosine_from_dot(dx_dot, dx_g_norm, dx_d_norm);
    metrics.dx_projection = safe_ratio(dx_dot, dx_d_norm * dx_d_norm);
    const double orth_sq = std::max(
        0.0, dx_g_norm * dx_g_norm - safe_ratio(dx_dot * dx_dot, dx_d_norm * dx_d_norm));
    metrics.dx_orth_error = safe_ratio(std::sqrt(orth_sq), dx_g_norm);

    const int32_t theta_n = ctx.n_pvpq;
    const int32_t vmag_n = ctx.n_pq;
    const double theta_g_norm = vector_norm2(cublas, theta_n, d_g);
    const double theta_d_norm = vector_norm2(cublas, theta_n, d_d);
    const double theta_dot = vector_dot(cublas, theta_n, d_g, d_d);
    metrics.theta_norm_ratio = safe_ratio(theta_g_norm, theta_d_norm);
    metrics.theta_cosine = cosine_from_dot(theta_dot, theta_g_norm, theta_d_norm);

    const double* d_g_vmag = d_g + ctx.n_pvpq;
    const double* d_d_vmag = d_d + ctx.n_pvpq;
    const double vmag_g_norm = vector_norm2(cublas, vmag_n, d_g_vmag);
    const double vmag_d_norm = vector_norm2(cublas, vmag_n, d_d_vmag);
    const double vmag_dot = vector_dot(cublas, vmag_n, d_g_vmag, d_d_vmag);
    metrics.vmag_norm_ratio = safe_ratio(vmag_g_norm, vmag_d_norm);
    metrics.vmag_cosine = cosine_from_dot(vmag_dot, vmag_g_norm, vmag_d_norm);

    metrics.max_abs_dx_gmres = max_abs_values(ctx, ctx.dimF, d_g);
    metrics.max_abs_dx_cudss = max_abs_values(ctx, ctx.dimF, d_d);
    CUITER_CUBLAS_CHECK(cublasDcopy(cublas, ctx.dimF, d_g, 1, ctx.d_dx_diff_shadow.data(), 1));
    const double minus_one = -1.0;
    CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                    ctx.dimF,
                                    &minus_one,
                                    d_d,
                                    1,
                                    ctx.d_dx_diff_shadow.data(),
                                    1));
    metrics.max_abs_dx_diff = max_abs_values(ctx, ctx.dimF, ctx.d_dx_diff_shadow.data());
    return metrics;
}

cuiter::GmresSolverOptions make_gmres_options(const HybridNrOptions& options,
                                              const MinimalNrDeviceContext& ctx)
{
    cuiter::GmresSolverOptions gmres_options;
    const bool use_mr1 = options.middle_solver == "mr1_block_jacobi" ||
                         options.middle_solver == "mr1_block_jacobi_coarse";
    const bool use_mr2 = options.middle_solver == "mr2_block_jacobi_coarse";
    const bool use_bicgstab = options.middle_solver == "bicgstab_block_jacobi" ||
                              options.middle_solver == "bicgstab_block_jacobi_a0" ||
                              options.middle_solver == "bicgstab_block_jacobi_a1" ||
                              options.middle_solver == "bicgstab_block_jacobi_a0_device" ||
                              options.middle_solver == "bicgstab_block_jacobi_a1_device" ||
                              options.middle_solver == "bicgstab_block_jacobi_j11_device" ||
                              options.middle_solver == "bicgstab_block_jacobi_bpbpp_refine";
    const bool use_cpu_block_ilu0 = options.middle_solver == "bicgstab_block_ilu0" ||
                                    options.middle_solver == "gmres_block_ilu0";
    const bool use_cpu_gmres_block_ilu0 = options.middle_solver == "gmres_block_ilu0";
    const bool use_ginkgo_parilut = options.middle_solver == "ginkgo_parilut_bicgstab";
    gmres_options.max_iters =
        use_bicgstab ? options.bicgstab_iters
                     : ((use_mr1 || use_mr2) ? 1 : options.gmres_max_iters);
    gmres_options.restart = (use_mr1 || use_mr2 || use_bicgstab || use_cpu_block_ilu0)
                                 ? (use_cpu_gmres_block_ilu0 ? options.gmres_restart : 1)
                                 : options.gmres_restart;
    gmres_options.rel_tolerance = options.gmres_fixed_iter_mode ? 0.0 : options.gmres_rtol;
    gmres_options.abs_tolerance = options.gmres_fixed_iter_mode ? 0.0 : options.gmres_atol;
    if (options.middle_solver == "mr1_block_jacobi_coarse" ||
        options.middle_solver == "mr2_block_jacobi_coarse") {
        gmres_options.preconditioner = "metis_block_jacobi_coarse";
    } else if (use_cpu_block_ilu0) {
        gmres_options.preconditioner = "metis_block_jacobi";
    } else if (use_ginkgo_parilut) {
        // Ginkgo middle solves bypass the in-house GMRES object. Keep this
        // placeholder valid so the common NR setup path can still construct it.
        gmres_options.preconditioner = "none";
    } else {
        gmres_options.preconditioner = options.preconditioner;
    }
    gmres_options.block_size = options.block_size;
    gmres_options.use_fp32_preconditioner = options.block_precision == "fp32";
    gmres_options.use_right_preconditioning = true;
    gmres_options.compute_true_residual = true;
    gmres_options.minimize_host_sync = true;
    gmres_options.block_jacobi_apply = cuiter::parse_block_jacobi_apply_mode(options.block_apply);
    gmres_options.use_mr1_fast_path = use_mr1;
    gmres_options.use_mr2_fast_path = use_mr2;
    gmres_options.use_bicgstab_fixed_path = use_bicgstab;
    gmres_options.use_bicgstab_fused_fixed2 = options.bicgstab_fused_fixed2;
    gmres_options.coarse_vars_per_block = options.coarse_vars_per_block;
    gmres_options.coarse_refresh = options.coarse_refresh;
    gmres_options.coarse_precision = options.coarse_precision;
    gmres_options.coarse_diag_shift_scale = options.coarse_diag_shift_scale;
    gmres_options.linear_scaling = options.linear_scaling;
    gmres_options.scaling_iters = options.scaling_iters;
    gmres_options.scaling_norm = options.scaling_norm;
    gmres_options.scaling_clamp = options.scaling_clamp;
    gmres_options.scaling_eps = options.scaling_eps;
    gmres_options.log_scaling_stats = options.log_scaling_stats;
    gmres_options.use_initial_guess = options.previous_dx_warm_start;
    gmres_options.partition_mode = options.partition_mode;
    gmres_options.bus_edge_weight = options.bus_edge_weight;
    gmres_options.bus_edge_weight_scale = options.bus_edge_weight_scale;
    gmres_options.bus_edge_weight_clamp = options.bus_edge_weight_clamp;
    gmres_options.target_block_unknowns = options.target_block_unknowns;
    gmres_options.n_bus = ctx.n_bus;
    gmres_options.index_to_bus = ctx.index_to_bus;
    gmres_options.index_field = ctx.index_field;
    return gmres_options;
}

bool should_use_cudss(const HybridNrOptions& options,
                      int32_t nr_iter,
                      double mismatch_inf,
                      int32_t gmres_calls)
{
    if (options.solver == "pure_cudss") {
        return true;
    }
    if (options.middle_solver == "stale_GMRES1_refresh") {
        return false;
    }
    if (options.middle_solver == "fdlf_bpbpp_2round") {
        return false;
    }
    if (nr_iter < options.cudss_bootstrap_iters) {
        return true;
    }
    if (options.iterative_start_mismatch_threshold >= 0.0 &&
        mismatch_inf > options.iterative_start_mismatch_threshold) {
        return true;
    }
    if (gmres_calls < options.force_gmres_min_steps) {
        return false;
    }
    if (options.cudss_polish_threshold < 0.0) {
        return false;
    }
    return mismatch_inf <= options.cudss_polish_threshold;
}

std::string cudss_solver_name(const HybridNrOptions& options,
                              int32_t nr_iter,
                              double mismatch_inf)
{
    if (options.solver == "pure_cudss") {
        return "cudss_pure";
    }
    if (options.middle_solver == "stale_GMRES1_refresh") {
        return "cudss_disabled_for_stale_gmres_refresh";
    }
    if (nr_iter < options.cudss_bootstrap_iters) {
        return "cudss_bootstrap";
    }
    if (options.iterative_start_mismatch_threshold >= 0.0 &&
        mismatch_inf > options.iterative_start_mismatch_threshold) {
        return "cudss_until_iterative_threshold";
    }
    if (options.cudss_polish_threshold >= 0.0 && mismatch_inf <= options.cudss_polish_threshold) {
        return "cudss_polish";
    }
    return "cudss_direct";
}

void apply_voltage_update_for_dx(MinimalNrDeviceContext& ctx,
                                 const double* d_dx,
                                 double damping_factor = 1.0)
{
    launch_voltage_update(ctx.n_bus,
                          ctx.dimF,
                          ctx.n_pv,
                          ctx.n_pq,
                          ctx.d_va.data(),
                          ctx.d_vm.data(),
                          ctx.d_v_re.data(),
                          ctx.d_v_im.data(),
                          d_dx,
                          ctx.d_pv.data(),
                          ctx.d_pq.data(),
                          damping_factor);
}

void apply_voltage_update(MinimalNrDeviceContext& ctx, double damping_factor = 1.0)
{
    apply_voltage_update_for_dx(ctx, ctx.d_dx.data(), damping_factor);
}

double ratio_to_before(double after, double before)
{
    return before > 0.0 && std::isfinite(after) ? after / before : 0.0;
}

bool gamma_matches(double gamma, double target)
{
    return std::abs(gamma - target) <= 1.0e-12 * std::max(1.0, std::abs(target));
}

void record_scaled_gamma_ratio(HybridNrIterationLog& log, double gamma, double ratio)
{
    if (gamma_matches(gamma, 4.0)) {
        log.mismatch_ratio_gamma_4 = ratio;
    } else if (gamma_matches(gamma, 2.0)) {
        log.mismatch_ratio_gamma_2 = ratio;
    } else if (gamma_matches(gamma, 1.0)) {
        log.mismatch_ratio_gamma_1 = ratio;
    }
}

void update_accepted_linear_stats(const HybridNrIterationLog& log,
                                  double& max_rel,
                                  double& sum_rel,
                                  double& sum_ratio,
                                  int32_t& count)
{
    const bool iterative_solver = log.solver_used == "gmres_middle" ||
                                  log.solver_used == "mr1_middle" ||
                                  log.solver_used == "mr2_middle" ||
                                  log.solver_used == "bicgstab_middle" ||
                                  log.solver_used == "bicgstab_a0_middle" ||
                                  log.solver_used == "bicgstab_a1_middle" ||
                                  log.solver_used == "bicgstab_a0_device_middle" ||
                                  log.solver_used == "bicgstab_a1_device_middle" ||
                                  log.solver_used == "bicgstab_j11_device_middle";
    if (log.step_accepted && iterative_solver && std::isfinite(log.linear_rel_res)) {
        max_rel = std::max(max_rel, log.linear_rel_res);
        sum_rel += log.linear_rel_res;
        if (log.mismatch_inf_before > 0.0 && std::isfinite(log.mismatch_inf_after)) {
            sum_ratio += log.mismatch_inf_after / log.mismatch_inf_before;
        }
        ++count;
    }
}

cuiter::cpu_pilot::CpuBlockIlu0Result solve_cpu_block_ilu0_middle(
    MinimalNrDeviceContext& ctx,
    const HybridNrOptions& options,
    cuiter::CsrMatrix* host_matrix_out,
    std::vector<double>* rhs_out);

double host_dot(const std::vector<double>& a, const std::vector<double>& b);
double host_norm2(const std::vector<double>& values);

double run_shadow_dx_diagnostic(MinimalNrDeviceContext& ctx,
                                cublasHandle_t cublas,
                                cuiter::GmresSolver& shadow_gmres,
                                DirectCudssSolver& shadow_cudss,
                                bool& shadow_cudss_analyzed,
                                double rhs_norm,
                                const HybridNrOptions& options,
                                HybridNrIterationLog& log)
{
    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    log.shadow_dx_diagnostic = true;

    ctx.backup_state_and_rhs();

    if (options.middle_solver == "gmres_block_ilu0" ||
        options.middle_solver == "bicgstab_block_ilu0") {
        const cuiter::cpu_pilot::CpuBlockIlu0Result block_ilu_result =
            solve_cpu_block_ilu0_middle(ctx, options, nullptr, nullptr);
        CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx_gmres_shadow.data(),
                                     block_ilu_result.solution.data(),
                                     static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                     cudaMemcpyHostToDevice));
        log.shadow_gmres_setup_seconds =
            block_ilu_result.setup_seconds + block_ilu_result.factor_seconds;
        log.shadow_gmres_solve_seconds =
            options.middle_solver == "gmres_block_ilu0"
                ? block_ilu_result.gmres_total_seconds
                : block_ilu_result.bicgstab_total_seconds;
        log.shadow_linear_abs_res_gmres = block_ilu_result.residual_norm2;
        log.shadow_linear_rel_res_gmres = block_ilu_result.relative_residual_norm2;
    } else {
        log.shadow_gmres_setup_seconds = timed_with_sync([&] {
            shadow_gmres.setup(ctx.d_J_values.data());
        });
        const cuiter::LinearSolveResult gmres_result =
            shadow_gmres.solve_device(ctx.d_J_values.data(),
                                      ctx.d_F.data(),
                                      ctx.d_dx_gmres_shadow.data());
        log.shadow_gmres_solve_seconds = gmres_result.timings.solve_total_seconds;
        log.shadow_linear_abs_res_gmres = gmres_result.residual_norm2;
        log.shadow_linear_rel_res_gmres = gmres_result.relative_residual_norm2;
    }
    log.shadow_dx_gmres_norm2 =
        vector_norm2(cublas, ctx.dimF, ctx.d_dx_gmres_shadow.data());

    log.shadow_mismatch_eval_seconds += timed_with_sync([&] {
        apply_voltage_update_for_dx(ctx, ctx.d_dx_gmres_shadow.data());
    });
    MismatchNorms gmres_after;
    log.shadow_mismatch_eval_seconds += timed_with_sync([&] {
        gmres_after = compute_mismatch(ctx, cublas, true);
    });
    log.shadow_mismatch_after_gmres_inf = gmres_after.inf;
    log.shadow_mismatch_after_gmres_2 = gmres_after.two;
    log.shadow_mismatch_after_gmres_p_inf = gmres_after.p_inf;
    log.shadow_mismatch_after_gmres_p_2 = gmres_after.p_two;
    log.shadow_mismatch_after_gmres_q_inf = gmres_after.q_inf;
    log.shadow_mismatch_after_gmres_q_2 = gmres_after.q_two;

    ctx.restore_state_and_rhs();

    if (!shadow_cudss_analyzed) {
        log.shadow_cudss_analyze_seconds = shadow_cudss.analyze();
        shadow_cudss_analyzed = true;
    }
    log.shadow_cudss_factorize_seconds = shadow_cudss.factorize();
    log.shadow_cudss_solve_seconds = shadow_cudss.solve();
    log.shadow_linear_abs_res_cudss =
        compute_linear_residual_for_dx(ctx,
                                       cublas,
                                       ctx.d_dx_cudss_shadow.data(),
                                       rhs_norm,
                                       log.shadow_linear_rel_res_cudss);
    log.shadow_dx_cudss_norm2 =
        vector_norm2(cublas, ctx.dimF, ctx.d_dx_cudss_shadow.data());

    log.shadow_mismatch_eval_seconds += timed_with_sync([&] {
        apply_voltage_update_for_dx(ctx, ctx.d_dx_cudss_shadow.data());
    });
    MismatchNorms cudss_after;
    log.shadow_mismatch_eval_seconds += timed_with_sync([&] {
        cudss_after = compute_mismatch(ctx, cublas, true);
    });
    log.shadow_mismatch_after_cudss_inf = cudss_after.inf;
    log.shadow_mismatch_after_cudss_2 = cudss_after.two;
    log.shadow_mismatch_after_cudss_p_inf = cudss_after.p_inf;
    log.shadow_mismatch_after_cudss_p_2 = cudss_after.p_two;
    log.shadow_mismatch_after_cudss_q_inf = cudss_after.q_inf;
    log.shadow_mismatch_after_cudss_q_2 = cudss_after.q_two;

    ctx.restore_state_and_rhs();

    const DxComparisonMetrics dx_metrics = compare_shadow_dx(ctx, cublas);
    log.shadow_dx_norm_ratio = dx_metrics.dx_norm_ratio;
    log.shadow_dx_cosine = dx_metrics.dx_cosine;
    log.shadow_dx_projection = dx_metrics.dx_projection;
    log.shadow_dx_orth_error = dx_metrics.dx_orth_error;
    log.shadow_theta_norm_ratio = dx_metrics.theta_norm_ratio;
    log.shadow_theta_cosine = dx_metrics.theta_cosine;
    log.shadow_vmag_norm_ratio = dx_metrics.vmag_norm_ratio;
    log.shadow_vmag_cosine = dx_metrics.vmag_cosine;
    log.shadow_max_abs_dx_gmres = dx_metrics.max_abs_dx_gmres;
    log.shadow_max_abs_dx_cudss = dx_metrics.max_abs_dx_cudss;
    log.shadow_max_abs_dx_diff = dx_metrics.max_abs_dx_diff;

    std::vector<double> h_g(static_cast<std::size_t>(ctx.dimF), 0.0);
    std::vector<double> h_d(static_cast<std::size_t>(ctx.dimF), 0.0);
    std::vector<double> h_j(static_cast<std::size_t>(ctx.j_pattern.nnz()), 0.0);
    ctx.d_dx_gmres_shadow.copy_to(h_g.data(), h_g.size());
    ctx.d_dx_cudss_shadow.copy_to(h_d.data(), h_d.size());
    ctx.d_J_values.copy_to(h_j.data(), h_j.size());
    const int32_t n_theta = ctx.n_pvpq;
    const int32_t n_v = ctx.n_pq;
    const double theta_iter_dot =
        host_dot(std::vector<double>(h_g.begin(), h_g.begin() + n_theta),
                 std::vector<double>(h_g.begin(), h_g.begin() + n_theta));
    const double v_iter_dot =
        host_dot(std::vector<double>(h_g.begin() + n_theta, h_g.end()),
                 std::vector<double>(h_g.begin() + n_theta, h_g.end()));
    double theta_cross = 0.0;
    double v_cross = 0.0;
    for (int32_t i = 0; i < n_theta; ++i) {
        theta_cross += h_d[static_cast<std::size_t>(i)] * h_g[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < n_v; ++i) {
        const std::size_t idx = static_cast<std::size_t>(n_theta + i);
        v_cross += h_d[idx] * h_g[idx];
    }
    log.alpha_theta_oracle = theta_cross / std::max(theta_iter_dot, 1.0e-300);
    log.alpha_v_oracle = v_cross / std::max(v_iter_dot, 1.0e-300);

    std::vector<double> d(static_cast<std::size_t>(ctx.dimF), 0.0);
    for (int32_t i = 0; i < ctx.dimF; ++i) {
        d[static_cast<std::size_t>(i)] =
            h_d[static_cast<std::size_t>(i)] - h_g[static_cast<std::size_t>(i)];
    }
    std::vector<double> m11(static_cast<std::size_t>(n_theta), 0.0);
    std::vector<double> m12(static_cast<std::size_t>(n_theta), 0.0);
    std::vector<double> m21(static_cast<std::size_t>(n_v), 0.0);
    std::vector<double> m22(static_cast<std::size_t>(n_v), 0.0);
    for (int32_t row = 0; row < ctx.dimF; ++row) {
        for (int32_t pos = ctx.j_pattern.row_ptr[static_cast<std::size_t>(row)];
             pos < ctx.j_pattern.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = ctx.j_pattern.col_idx[static_cast<std::size_t>(pos)];
            const double contrib = h_j[static_cast<std::size_t>(pos)] *
                                   d[static_cast<std::size_t>(col)];
            if (row < n_theta && col < n_theta) {
                m11[static_cast<std::size_t>(row)] += contrib;
            } else if (row < n_theta && col >= n_theta) {
                m12[static_cast<std::size_t>(row)] += contrib;
            } else if (row >= n_theta && col < n_theta) {
                m21[static_cast<std::size_t>(row - n_theta)] += contrib;
            } else {
                m22[static_cast<std::size_t>(row - n_theta)] += contrib;
            }
        }
    }
    std::vector<double> p_missing(static_cast<std::size_t>(n_theta), 0.0);
    std::vector<double> q_missing(static_cast<std::size_t>(n_v), 0.0);
    for (int32_t i = 0; i < n_theta; ++i) {
        p_missing[static_cast<std::size_t>(i)] =
            m11[static_cast<std::size_t>(i)] + m12[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < n_v; ++i) {
        q_missing[static_cast<std::size_t>(i)] =
            m21[static_cast<std::size_t>(i)] + m22[static_cast<std::size_t>(i)];
    }
    log.norm_m11 = host_norm2(m11);
    log.norm_m12 = host_norm2(m12);
    log.norm_m21 = host_norm2(m21);
    log.norm_m22 = host_norm2(m22);
    log.norm_p_missing = host_norm2(p_missing);
    log.norm_q_missing = host_norm2(q_missing);
    log.norm_ad = std::sqrt(log.norm_p_missing * log.norm_p_missing +
                            log.norm_q_missing * log.norm_q_missing);
    const double denom = std::max(log.norm_ad, 1.0e-300);
    log.frac_m11 = log.norm_m11 / denom;
    log.frac_m12 = log.norm_m12 / denom;
    log.frac_m21 = log.norm_m21 / denom;
    log.frac_m22 = log.norm_m22 / denom;

    CUITER_CUDA_CHECK(cudaDeviceSynchronize());
    log.shadow_dx_diagnostic_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return log.shadow_dx_diagnostic_seconds;
}

cuiter::cpu_pilot::CpuBlockIlu0Result solve_cpu_block_ilu0_middle(
    MinimalNrDeviceContext& ctx,
    const HybridNrOptions& options,
    cuiter::CsrMatrix* host_matrix_out = nullptr,
    std::vector<double>* rhs_out = nullptr)
{
    cuiter::CsrMatrix host_matrix = ctx.j_pattern;
    host_matrix.values.assign(static_cast<std::size_t>(ctx.j_pattern.nnz()), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(ctx.dimF), 0.0);
    ctx.d_J_values.copy_to(host_matrix.values.data(), host_matrix.values.size());
    ctx.d_F.copy_to(rhs.data(), rhs.size());
    if (host_matrix_out != nullptr) {
        *host_matrix_out = host_matrix;
    }
    if (rhs_out != nullptr) {
        *rhs_out = rhs;
    }

    cuiter::cpu_pilot::CpuBlockIlu0Options cpu_options;
    cpu_options.block_size = options.block_size;
    cpu_options.bicgstab_iters = options.bicgstab_iters;
    cpu_options.gmres_iters = options.gmres_max_iters;
    cpu_options.diag_shift_scale = 1.0e-8;
    cpu_options.use_block_ilu0 = true;
    cpu_options.use_block_coloring_order = true;
    cpu_options.use_gmres = options.middle_solver == "gmres_block_ilu0";
    return cuiter::cpu_pilot::solve(host_matrix, rhs, cpu_options);
}

std::vector<double> host_spmv(const cuiter::CsrMatrix& matrix, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            sum += matrix.values[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(pos)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

std::vector<double> host_residual(const cuiter::CsrMatrix& matrix,
                                  const std::vector<double>& rhs,
                                  const std::vector<double>& x)
{
    std::vector<double> out = rhs;
    const std::vector<double> ax = host_spmv(matrix, x);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] -= ax[i];
    }
    return out;
}

double host_dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double host_norm2(const std::vector<double>& values)
{
    return std::sqrt(std::max(0.0, host_dot(values, values)));
}

#if defined(CUITER_WITH_GINKGO)
using GkoValue = double;
using GkoIndex = int32_t;
using GkoDense = gko::matrix::Dense<GkoValue>;
using GkoCsr = gko::matrix::Csr<GkoValue, GkoIndex>;
using GkoBicgstab = gko::solver::Bicgstab<GkoValue>;
using GkoParIlut = gko::factorization::ParIlut<GkoValue, GkoIndex>;
using GkoIlu = gko::preconditioner::Ilu<GkoValue, false, GkoIndex>;

std::shared_ptr<gko::Executor> ginkgo_cuda_executor()
{
    static std::shared_ptr<gko::Executor> executor =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    return executor;
}

template <typename Fn>
double timed_ginkgo_seconds(const std::shared_ptr<gko::Executor>& exec, Fn&& fn)
{
    exec->synchronize();
    const auto start = std::chrono::steady_clock::now();
    fn();
    exec->synchronize();
    return elapsed_seconds(start);
}

std::shared_ptr<const GkoCsr> make_ginkgo_csr(const cuiter::CsrMatrix& matrix,
                                             const std::shared_ptr<gko::Executor>& exec)
{
    gko::matrix_data<GkoValue, GkoIndex> data(
        gko::dim<2>{static_cast<gko::size_type>(matrix.rows),
                    static_cast<gko::size_type>(matrix.cols)});
    data.nonzeros.reserve(matrix.values.size());
    for (int32_t row = 0; row < matrix.rows; ++row) {
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            data.nonzeros.emplace_back(row,
                                       matrix.col_idx[static_cast<std::size_t>(pos)],
                                       matrix.values[static_cast<std::size_t>(pos)]);
        }
    }
    auto out = GkoCsr::create(exec);
    out->read(data);
    exec->synchronize();
    return gko::share(std::move(out));
}

std::shared_ptr<GkoDense> make_ginkgo_vector(const std::vector<double>& values,
                                             const std::shared_ptr<gko::Executor>& exec)
{
    auto host = GkoDense::create(exec->get_master(),
                                 gko::dim<2>{static_cast<gko::size_type>(values.size()), 1});
    for (gko::size_type i = 0; i < values.size(); ++i) {
        host->at(i, 0) = values[static_cast<std::size_t>(i)];
    }
    return gko::clone(exec, host);
}

std::shared_ptr<GkoDense> make_ginkgo_zero_vector(gko::size_type n,
                                                  const std::shared_ptr<gko::Executor>& exec)
{
    std::vector<double> zero(static_cast<std::size_t>(n), 0.0);
    return make_ginkgo_vector(zero, exec);
}

std::vector<double> copy_ginkgo_vector_to_host(const std::shared_ptr<GkoDense>& vector)
{
    auto host = gko::clone(vector->get_executor()->get_master(), vector);
    std::vector<double> out(static_cast<std::size_t>(vector->get_size()[0]), 0.0);
    for (gko::size_type i = 0; i < vector->get_size()[0]; ++i) {
        out[static_cast<std::size_t>(i)] = host->at(i, 0);
    }
    return out;
}

cuiter::cpu_pilot::CpuBlockIlu0Result solve_ginkgo_parilut_middle(
    MinimalNrDeviceContext& ctx,
    const HybridNrOptions& options)
{
    cuiter::cpu_pilot::CpuBlockIlu0Result result;
    cuiter::CsrMatrix host_matrix = ctx.j_pattern;
    host_matrix.values.assign(static_cast<std::size_t>(ctx.j_pattern.nnz()), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(ctx.dimF), 0.0);
    const double copy_seconds = timed_with_sync([&] {
        ctx.d_J_values.copy_to(host_matrix.values.data(), host_matrix.values.size());
        ctx.d_F.copy_to(rhs.data(), rhs.size());
    });

    try {
        auto exec = ginkgo_cuda_executor();
        std::shared_ptr<const GkoCsr> matrix;
        std::shared_ptr<GkoDense> b;
        result.setup_seconds += copy_seconds;
        result.setup_seconds += timed_ginkgo_seconds(exec, [&] {
            matrix = make_ginkgo_csr(host_matrix, exec);
            b = make_ginkgo_vector(rhs, exec);
        });

        std::shared_ptr<const gko::LinOp> parilut;
        result.factor_seconds += timed_ginkgo_seconds(exec, [&] {
            auto factory =
                GkoParIlut::build()
                    .with_iterations(static_cast<gko::size_type>(options.ginkgo_parilut_iters))
                    .with_fill_in_limit(options.ginkgo_parilut_fill)
                    .on(exec);
            parilut = gko::share(factory->generate(matrix));
        });

        std::shared_ptr<const gko::LinOp> preconditioner;
        result.factor_seconds += timed_ginkgo_seconds(exec, [&] {
            preconditioner = gko::share(GkoIlu::build().on(exec)->generate(parilut));
        });

        std::unique_ptr<GkoBicgstab> solver;
        result.setup_seconds += timed_ginkgo_seconds(exec, [&] {
            auto factory =
                GkoBicgstab::build()
                    .with_criteria(gko::stop::Iteration::build().with_max_iters(
                        static_cast<gko::size_type>(options.bicgstab_iters)))
                    .with_generated_preconditioner(preconditioner)
                    .on(exec);
            solver = factory->generate(matrix);
        });

        auto x = make_ginkgo_zero_vector(static_cast<gko::size_type>(host_matrix.cols), exec);
        result.bicgstab_total_seconds += timed_ginkgo_seconds(exec, [&] { solver->apply(b, x); });
        result.solution = copy_ginkgo_vector_to_host(x);
        result.iterations = options.bicgstab_iters;
        result.stop_reason = "ginkgo_parilut_bicgstab";
    } catch (const std::exception& ex) {
        result.factor_failed = true;
        result.stop_reason = std::string("ginkgo_parilut_failed:") + ex.what();
        result.solution.assign(static_cast<std::size_t>(ctx.dimF), 0.0);
    }

    if (!result.solution.empty()) {
        const std::vector<double> residual = host_residual(host_matrix, rhs, result.solution);
        result.residual_norm2 = host_norm2(residual);
        result.relative_residual_norm2 =
            result.residual_norm2 / std::max(host_norm2(rhs), std::numeric_limits<double>::min());
    }
    return result;
}
#endif

bool solve_dense_system(std::vector<double> a, std::vector<double> b, int32_t n, std::vector<double>& x)
{
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int32_t k = 0; k < n; ++k) {
        int32_t pivot = k;
        double pivot_abs = std::abs(a[static_cast<std::size_t>(k) * n + k]);
        for (int32_t i = k + 1; i < n; ++i) {
            const double candidate = std::abs(a[static_cast<std::size_t>(i) * n + k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = i;
            }
        }
        if (pivot_abs <= 1.0e-30 || !std::isfinite(pivot_abs)) {
            return false;
        }
        if (pivot != k) {
            for (int32_t j = k; j < n; ++j) {
                std::swap(a[static_cast<std::size_t>(k) * n + j],
                          a[static_cast<std::size_t>(pivot) * n + j]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot)]);
        }
        const double diag = a[static_cast<std::size_t>(k) * n + k];
        for (int32_t i = k + 1; i < n; ++i) {
            const double factor = a[static_cast<std::size_t>(i) * n + k] / diag;
            a[static_cast<std::size_t>(i) * n + k] = 0.0;
            for (int32_t j = k + 1; j < n; ++j) {
                a[static_cast<std::size_t>(i) * n + j] -=
                    factor * a[static_cast<std::size_t>(k) * n + j];
            }
            b[static_cast<std::size_t>(i)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }
    for (int32_t i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<std::size_t>(i)];
        for (int32_t j = i + 1; j < n; ++j) {
            sum -= a[static_cast<std::size_t>(i) * n + j] * x[static_cast<std::size_t>(j)];
        }
        const double diag = a[static_cast<std::size_t>(i) * n + i];
        if (std::abs(diag) <= 1.0e-30 || !std::isfinite(diag)) {
            return false;
        }
        x[static_cast<std::size_t>(i)] = sum / diag;
    }
    return true;
}

struct GlobalPostCorrectionBasis {
    int32_t max_rank = 0;
    double orth_tol = 1.0e-6;
    std::vector<std::vector<double>> columns;

    int32_t rank() const
    {
        return static_cast<int32_t>(columns.size());
    }

    void reset()
    {
        columns.clear();
    }

    bool add(std::vector<double> z_candidate, const std::vector<double>& reference)
    {
        if (max_rank <= 0 || z_candidate.empty()) {
            return false;
        }
        const double reference_norm = std::max(host_norm2(reference), 1.0e-300);
        if (host_norm2(z_candidate) / reference_norm < 1.0e-8) {
            return false;
        }
        for (const auto& basis : columns) {
            const double alpha = host_dot(basis, z_candidate);
            for (std::size_t i = 0; i < z_candidate.size(); ++i) {
                z_candidate[i] -= alpha * basis[i];
            }
        }
        const double norm = host_norm2(z_candidate);
        if (norm < orth_tol || !std::isfinite(norm)) {
            return false;
        }
        for (double& value : z_candidate) {
            value /= norm;
        }
        if (rank() >= max_rank) {
            columns.erase(columns.begin());
        }
        columns.push_back(std::move(z_candidate));
        return true;
    }
};

struct GlobalPostCorrectionResult {
    bool attempted = false;
    bool used = false;
    std::string skipped_reason;
    int32_t rank_before = 0;
    double linear_before = 0.0;
    double linear_after = 0.0;
    double correction_gain = 0.0;
    double correction_norm_ratio = 0.0;
    double total_seconds = 0.0;
    double az_seconds = 0.0;
    double dense_ls_seconds = 0.0;
    std::vector<double> corrected_step;
};

GlobalPostCorrectionResult apply_global_post_correction(
    const cuiter::CsrMatrix& matrix,
    const std::vector<double>& rhs,
    const std::vector<double>& p_gmres,
    const GlobalPostCorrectionBasis& basis)
{
    const auto start = std::chrono::steady_clock::now();
    GlobalPostCorrectionResult out;
    out.rank_before = basis.rank();
    if (basis.rank() <= 0) {
        out.skipped_reason = "empty_basis";
        return out;
    }
    out.attempted = true;
    const std::vector<double> residual = host_residual(matrix, rhs, p_gmres);
    out.linear_before = host_norm2(residual);
    const int32_t r = basis.rank();
    std::vector<std::vector<double>> az(static_cast<std::size_t>(r));
    const auto az_start = std::chrono::steady_clock::now();
    for (int32_t j = 0; j < r; ++j) {
        az[static_cast<std::size_t>(j)] = host_spmv(matrix, basis.columns[static_cast<std::size_t>(j)]);
    }
    out.az_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - az_start).count();

    const auto ls_start = std::chrono::steady_clock::now();
    std::vector<double> gram(static_cast<std::size_t>(r) * r, 0.0);
    std::vector<double> h(static_cast<std::size_t>(r), 0.0);
    double trace = 0.0;
    for (int32_t i = 0; i < r; ++i) {
        h[static_cast<std::size_t>(i)] = host_dot(az[static_cast<std::size_t>(i)], residual);
        for (int32_t j = 0; j < r; ++j) {
            gram[static_cast<std::size_t>(i) * r + j] =
                host_dot(az[static_cast<std::size_t>(i)], az[static_cast<std::size_t>(j)]);
        }
        trace += gram[static_cast<std::size_t>(i) * r + i];
    }
    const double lambda = (r > 0 && trace > 0.0) ? 1.0e-12 * trace / r : 1.0e-12;
    for (int32_t i = 0; i < r; ++i) {
        gram[static_cast<std::size_t>(i) * r + i] += lambda;
    }
    std::vector<double> coeff;
    if (!solve_dense_system(gram, h, r, coeff)) {
        out.skipped_reason = "dense_ls_failed";
        out.dense_ls_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ls_start).count();
        out.total_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return out;
    }
    out.dense_ls_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - ls_start).count();

    std::vector<double> correction(p_gmres.size(), 0.0);
    for (int32_t j = 0; j < r; ++j) {
        const auto& z = basis.columns[static_cast<std::size_t>(j)];
        const double c = coeff[static_cast<std::size_t>(j)];
        for (std::size_t i = 0; i < correction.size(); ++i) {
            correction[i] += c * z[i];
        }
    }
    const double p_norm = std::max(host_norm2(p_gmres), 1.0e-300);
    out.correction_norm_ratio = host_norm2(correction) / p_norm;
    if (!std::isfinite(out.correction_norm_ratio) || out.correction_norm_ratio > 2.0) {
        out.skipped_reason = "correction_norm_too_large";
        out.total_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return out;
    }
    out.corrected_step = p_gmres;
    for (std::size_t i = 0; i < out.corrected_step.size(); ++i) {
        out.corrected_step[i] += correction[i];
    }
    out.linear_after = host_norm2(host_residual(matrix, rhs, out.corrected_step));
    out.correction_gain = out.linear_before > 0.0 ? out.linear_after / out.linear_before : 1.0;
    if (!std::isfinite(out.linear_after) || out.linear_after > 1.1 * out.linear_before) {
        out.skipped_reason = "linear_residual_worse";
        out.total_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return out;
    }
    out.used = true;
    out.skipped_reason = "used";
    out.total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

std::vector<double> host_field_step(const std::vector<double>& step,
                                    int32_t n_theta,
                                    bool theta_field)
{
    std::vector<double> out(step.size(), 0.0);
    if (theta_field) {
        std::copy(step.begin(), step.begin() + n_theta, out.begin());
    } else {
        std::copy(step.begin() + n_theta, step.end(), out.begin() + n_theta);
    }
    return out;
}

double host_norm2_range(const std::vector<double>& values, int32_t begin, int32_t count)
{
    double sum = 0.0;
    for (int32_t i = 0; i < count; ++i) {
        const double value = values[static_cast<std::size_t>(begin + i)];
        sum += value * value;
    }
    return std::sqrt(std::max(0.0, sum));
}

struct FieldGainResult {
    bool attempted = false;
    bool accepted = false;
    std::string skipped_reason;
    double gamma_theta = 1.0;
    double gamma_v = 1.0;
    double linear_before = 0.0;
    double linear_after = 0.0;
    double step_norm_ratio = 1.0;
    double seconds = 0.0;
    std::vector<double> step;
};

FieldGainResult apply_field_gain_ls2(const cuiter::CsrMatrix& matrix,
                                     const std::vector<double>& rhs,
                                     const std::vector<double>& p_iter,
                                     int32_t n_theta,
                                     const HybridNrOptions& options)
{
    const auto start = std::chrono::steady_clock::now();
    FieldGainResult out;
    out.attempted = true;
    out.step = p_iter;
    const std::vector<double> p_theta = host_field_step(p_iter, n_theta, true);
    const std::vector<double> p_v = host_field_step(p_iter, n_theta, false);
    const std::vector<double> u_theta = host_spmv(matrix, p_theta);
    const std::vector<double> u_v = host_spmv(matrix, p_v);
    const double g00 = host_dot(u_theta, u_theta);
    const double g01 = host_dot(u_theta, u_v);
    const double g11 = host_dot(u_v, u_v);
    const double h0 = host_dot(u_theta, rhs);
    const double h1 = host_dot(u_v, rhs);
    const double det = g00 * g11 - g01 * g01;
    if (std::abs(det) <= 1.0e-30 || !std::isfinite(det)) {
        out.skipped_reason = "singular_ls2";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    double gamma_theta = (h0 * g11 - h1 * g01) / det;
    double gamma_v = (g00 * h1 - g01 * h0) / det;
    if (!std::isfinite(gamma_theta) || !std::isfinite(gamma_v)) {
        out.skipped_reason = "nan_gamma";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    if (options.field_gain_nonnegative) {
        gamma_theta = std::max(0.0, gamma_theta);
        gamma_v = std::max(0.0, gamma_v);
    }
    gamma_theta = std::min(gamma_theta, options.field_gain_theta_max);
    gamma_v = std::min(gamma_v, options.field_gain_vmax);
    out.gamma_theta = gamma_theta;
    out.gamma_v = gamma_v;
    for (int32_t i = 0; i < n_theta; ++i) {
        out.step[static_cast<std::size_t>(i)] =
            gamma_theta * p_iter[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = static_cast<std::size_t>(n_theta); i < out.step.size(); ++i) {
        out.step[i] = gamma_v * p_iter[i];
    }
    const double p_norm = std::max(host_norm2(p_iter), 1.0e-300);
    out.step_norm_ratio = host_norm2(out.step) / p_norm;
    if (out.step_norm_ratio > options.field_gain_trust_ratio) {
        out.skipped_reason = "trust_ratio";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    out.linear_before = host_norm2(host_residual(matrix, rhs, p_iter));
    out.linear_after = host_norm2(host_residual(matrix, rhs, out.step));
    if (!std::isfinite(out.linear_after) || out.linear_after > 1.1 * out.linear_before) {
        out.skipped_reason = "linear_residual_worse";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    out.accepted = true;
    out.skipped_reason = "used";
    out.seconds = elapsed_seconds(start);
    return out;
}

std::vector<double> host_j11_spmv(const cuiter::CsrMatrix& matrix,
                                  const std::vector<double>& theta,
                                  int32_t n_theta)
{
    std::vector<double> out(static_cast<std::size_t>(n_theta), 0.0);
    for (int32_t row = 0; row < n_theta; ++row) {
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            if (col < n_theta) {
                sum += matrix.values[static_cast<std::size_t>(pos)] *
                       theta[static_cast<std::size_t>(col)];
            }
        }
        out[static_cast<std::size_t>(row)] = sum;
    }
    return out;
}

std::vector<double> gmres_j11_correction(const cuiter::CsrMatrix& matrix,
                                         const std::vector<double>& rhs_p,
                                         int32_t n_theta,
                                         int32_t max_iters)
{
    const int32_t m = std::max(1, std::min(max_iters, n_theta));
    const double beta = host_norm2(rhs_p);
    if (beta <= 0.0 || !std::isfinite(beta)) {
        return std::vector<double>(static_cast<std::size_t>(n_theta), 0.0);
    }
    std::vector<std::vector<double>> v(static_cast<std::size_t>(m + 1),
                                       std::vector<double>(static_cast<std::size_t>(n_theta), 0.0));
    std::vector<std::vector<double>> h(static_cast<std::size_t>(m + 1),
                                       std::vector<double>(static_cast<std::size_t>(m), 0.0));
    for (int32_t i = 0; i < n_theta; ++i) {
        v[0][static_cast<std::size_t>(i)] = rhs_p[static_cast<std::size_t>(i)] / beta;
    }
    int32_t used = 0;
    for (int32_t j = 0; j < m; ++j) {
        std::vector<double> w = host_j11_spmv(matrix, v[static_cast<std::size_t>(j)], n_theta);
        for (int32_t i = 0; i <= j; ++i) {
            h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                host_dot(w, v[static_cast<std::size_t>(i)]);
            const double hij = h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            for (int32_t row = 0; row < n_theta; ++row) {
                w[static_cast<std::size_t>(row)] -=
                    hij * v[static_cast<std::size_t>(i)][static_cast<std::size_t>(row)];
            }
        }
        const double hn = host_norm2(w);
        h[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] = hn;
        used = j + 1;
        if (hn <= 1.0e-14) {
            break;
        }
        for (int32_t row = 0; row < n_theta; ++row) {
            v[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(row)] =
                w[static_cast<std::size_t>(row)] / hn;
        }
    }
    std::vector<double> normal(static_cast<std::size_t>(used) * used, 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(used), 0.0);
    for (int32_t i = 0; i < used; ++i) {
        rhs[static_cast<std::size_t>(i)] = beta * h[0][static_cast<std::size_t>(i)];
        for (int32_t j = 0; j < used; ++j) {
            double sum = 0.0;
            for (int32_t row = 0; row <= used; ++row) {
                sum += h[static_cast<std::size_t>(row)][static_cast<std::size_t>(i)] *
                       h[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
            }
            normal[static_cast<std::size_t>(i) * used + j] = sum;
        }
    }
    std::vector<double> y;
    if (!solve_dense_system(normal, rhs, used, y)) {
        return std::vector<double>(static_cast<std::size_t>(n_theta), 0.0);
    }
    std::vector<double> delta(static_cast<std::size_t>(n_theta), 0.0);
    for (int32_t j = 0; j < used; ++j) {
        for (int32_t row = 0; row < n_theta; ++row) {
            delta[static_cast<std::size_t>(row)] +=
                y[static_cast<std::size_t>(j)] *
                v[static_cast<std::size_t>(j)][static_cast<std::size_t>(row)];
        }
    }
    return delta;
}

struct ThetaCorrectionResult {
    bool attempted = false;
    bool accepted = false;
    std::string skipped_reason;
    double beta = 0.0;
    double corr_norm = 0.0;
    double j11_res_before = 0.0;
    double j11_res_after = 0.0;
    double p_res_before = 0.0;
    double p_res_after = 0.0;
    double q_res_before = 0.0;
    double q_res_after = 0.0;
    double seconds = 0.0;
    std::vector<double> step;
};

ThetaCorrectionResult apply_theta_correction(const cuiter::CsrMatrix& matrix,
                                             const std::vector<double>& rhs,
                                             const std::vector<double>& step,
                                             int32_t n_theta,
                                             const HybridNrOptions& options)
{
    const auto start = std::chrono::steady_clock::now();
    ThetaCorrectionResult out;
    out.attempted = true;
    out.step = step;
    const int32_t n_v = static_cast<int32_t>(step.size()) - n_theta;
    const std::vector<double> residual = host_residual(matrix, rhs, step);
    std::vector<double> r_p(residual.begin(), residual.begin() + n_theta);
    out.p_res_before = host_norm2(r_p);
    out.q_res_before = n_v > 0 ? host_norm2_range(residual, n_theta, n_v) : 0.0;
    std::vector<double> delta_theta(static_cast<std::size_t>(n_theta), 0.0);
    if (options.theta_j11_correction == "scalar") {
        std::vector<double> p_theta(step.begin(), step.begin() + n_theta);
        std::vector<double> j11_p = host_j11_spmv(matrix, p_theta, n_theta);
        const double denom = std::max(host_dot(j11_p, j11_p), 1.0e-300);
        double beta = host_dot(r_p, j11_p) / denom;
        beta = std::max(-0.5, std::min(8.0, beta));
        out.beta = beta;
        for (int32_t i = 0; i < n_theta; ++i) {
            delta_theta[static_cast<std::size_t>(i)] =
                beta * p_theta[static_cast<std::size_t>(i)];
        }
        out.j11_res_before = out.p_res_before;
    } else {
        delta_theta = gmres_j11_correction(matrix, r_p, n_theta, options.theta_j11_gmres_maxit);
        out.j11_res_before = out.p_res_before;
    }
    out.corr_norm = host_norm2(delta_theta);
    const double step_theta_norm = std::max(host_norm2_range(step, 0, n_theta), 1.0e-300);
    if (!std::isfinite(out.corr_norm) ||
        out.corr_norm / step_theta_norm > options.theta_j11_correction_trust_ratio) {
        out.skipped_reason = "trust_ratio";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    for (int32_t i = 0; i < n_theta; ++i) {
        out.step[static_cast<std::size_t>(i)] += delta_theta[static_cast<std::size_t>(i)];
    }
    const std::vector<double> residual_after = host_residual(matrix, rhs, out.step);
    out.p_res_after = host_norm2_range(residual_after, 0, n_theta);
    out.q_res_after = n_v > 0 ? host_norm2_range(residual_after, n_theta, n_v) : 0.0;
    out.j11_res_after = out.p_res_after;
    if (!std::isfinite(out.p_res_after) || out.p_res_after > 1.1 * out.p_res_before) {
        out.skipped_reason = "p_residual_worse";
        out.seconds = elapsed_seconds(start);
        return out;
    }
    out.accepted = true;
    out.skipped_reason = "used";
    out.seconds = elapsed_seconds(start);
    return out;
}

std::vector<double> host_slice(const std::vector<double>& values, int32_t begin, int32_t count)
{
    return std::vector<double>(values.begin() + begin, values.begin() + begin + count);
}

cuiter::CsrMatrix extract_field_submatrix(const cuiter::CsrMatrix& full,
                                          int32_t row_begin,
                                          int32_t row_count,
                                          int32_t col_begin,
                                          int32_t col_count)
{
    cuiter::CsrMatrix out;
    out.rows = row_count;
    out.cols = col_count;
    out.row_ptr.assign(static_cast<std::size_t>(row_count + 1), 0);
    for (int32_t local_row = 0; local_row < row_count; ++local_row) {
        const int32_t row = row_begin + local_row;
        for (int32_t pos = full.row_ptr[static_cast<std::size_t>(row)];
             pos < full.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = full.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_begin + col_count) {
                out.col_idx.push_back(col - col_begin);
                out.values.push_back(full.values[static_cast<std::size_t>(pos)]);
            }
        }
        out.row_ptr[static_cast<std::size_t>(local_row + 1)] =
            static_cast<int32_t>(out.values.size());
    }
    return out;
}

std::vector<double> host_submatrix_vector_product(const cuiter::CsrMatrix& matrix,
                                                  int32_t row_begin,
                                                  int32_t row_count,
                                                  int32_t col_begin,
                                                  int32_t col_count,
                                                  const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(row_count), 0.0);
    for (int32_t local_row = 0; local_row < row_count; ++local_row) {
        const int32_t row = row_begin + local_row;
        double sum = 0.0;
        for (int32_t pos = matrix.row_ptr[static_cast<std::size_t>(row)];
             pos < matrix.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_begin + col_count) {
                sum += matrix.values[static_cast<std::size_t>(pos)] *
                       x[static_cast<std::size_t>(col - col_begin)];
            }
        }
        y[static_cast<std::size_t>(local_row)] = sum;
    }
    return y;
}

struct FieldSolveStats {
    std::vector<double> x;
    double factor_seconds = 0.0;
    double solve_seconds = 0.0;
};

struct ParallelPhaseStats {
    double wall_seconds = 0.0;
    double left_seconds = 0.0;
    double right_seconds = 0.0;
    double wait_seconds = 0.0;
};

struct FieldCudssCache {
    cuiter::CsrMatrix pattern;
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    DirectCudssSolver solver;
    cudaStream_t stream = nullptr;
    bool initialized = false;
    double analyze_seconds = 0.0;

    void initialize(const cuiter::CsrMatrix& matrix)
    {
        pattern = matrix;
        d_row_ptr.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
        d_col_idx.assign(pattern.col_idx.data(), pattern.col_idx.size());
        d_values.assign(pattern.values.data(), pattern.values.size());
        std::vector<double> zeros(static_cast<std::size_t>(pattern.rows), 0.0);
        d_rhs.assign(zeros.data(), zeros.size());
        d_x.resize(zeros.size());
        solver.initialize(pattern,
                          d_row_ptr.data(),
                          d_col_idx.data(),
                          d_values.data(),
                          d_rhs.data(),
                          d_x.data());
        analyze_seconds = solver.analyze();
        initialized = true;
    }

    void set_stream(cudaStream_t new_stream)
    {
        stream = new_stream;
        solver.set_stream(stream);
    }

    void assign_values_and_rhs(const std::vector<double>& values,
                               const std::vector<double>& rhs)
    {
        if (!initialized) {
            throw std::runtime_error("field cuDSS cache was not initialized");
        }
        d_values.assign(values.data(), values.size());
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
    }

    void assign_rhs(const std::vector<double>& rhs)
    {
        if (!initialized) {
            throw std::runtime_error("field cuDSS cache was not initialized");
        }
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
    }

    void factorize_async()
    {
        solver.factorize_async();
    }

    void solve_async()
    {
        solver.solve_async();
    }

    std::vector<double> copy_solution_to_host() const
    {
        std::vector<double> x(static_cast<std::size_t>(pattern.rows), 0.0);
        d_x.copy_to(x.data(), x.size());
        return x;
    }

    FieldSolveStats factor_and_solve(const std::vector<double>& values,
                                     const std::vector<double>& rhs)
    {
        if (!initialized) {
            throw std::runtime_error("field cuDSS cache was not initialized");
        }
        d_values.assign(values.data(), values.size());
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
        FieldSolveStats stats;
        stats.factor_seconds = solver.factorize();
        stats.solve_seconds = solver.solve();
        stats.x.assign(rhs.size(), 0.0);
        d_x.copy_to(stats.x.data(), stats.x.size());
        return stats;
    }

    FieldSolveStats solve_again(const std::vector<double>& rhs)
    {
        if (!initialized) {
            throw std::runtime_error("field cuDSS cache was not initialized");
        }
        d_rhs.assign(rhs.data(), rhs.size());
        d_x.memset_zero();
        FieldSolveStats stats;
        stats.solve_seconds = solver.solve();
        stats.x.assign(rhs.size(), 0.0);
        d_x.copy_to(stats.x.data(), stats.x.size());
        return stats;
    }
};

template <typename LeftFn, typename RightFn>
ParallelPhaseStats run_two_stream_phase(cudaStream_t left_stream,
                                        LeftFn&& left_fn,
                                        cudaStream_t right_stream,
                                        RightFn&& right_fn)
{
    cudaEvent_t left_start = nullptr;
    cudaEvent_t left_stop = nullptr;
    cudaEvent_t right_start = nullptr;
    cudaEvent_t right_stop = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&left_start));
    CUITER_CUDA_CHECK(cudaEventCreate(&left_stop));
    CUITER_CUDA_CHECK(cudaEventCreate(&right_start));
    CUITER_CUDA_CHECK(cudaEventCreate(&right_stop));

    const auto wall_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaEventRecord(left_start, left_stream));
    left_fn();
    CUITER_CUDA_CHECK(cudaEventRecord(left_stop, left_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(right_start, right_stream));
    right_fn();
    CUITER_CUDA_CHECK(cudaEventRecord(right_stop, right_stream));
    const auto wait_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaEventSynchronize(left_stop));
    CUITER_CUDA_CHECK(cudaEventSynchronize(right_stop));
    const double wait_seconds = elapsed_seconds(wait_start);
    const double wall_seconds = elapsed_seconds(wall_start);

    float left_ms = 0.0F;
    float right_ms = 0.0F;
    CUITER_CUDA_CHECK(cudaEventElapsedTime(&left_ms, left_start, left_stop));
    CUITER_CUDA_CHECK(cudaEventElapsedTime(&right_ms, right_start, right_stop));
    CUITER_CUDA_CHECK(cudaEventDestroy(left_start));
    CUITER_CUDA_CHECK(cudaEventDestroy(left_stop));
    CUITER_CUDA_CHECK(cudaEventDestroy(right_start));
    CUITER_CUDA_CHECK(cudaEventDestroy(right_stop));

    ParallelPhaseStats stats;
    stats.wall_seconds = wall_seconds;
    stats.left_seconds = 0.001 * static_cast<double>(left_ms);
    stats.right_seconds = 0.001 * static_cast<double>(right_ms);
    stats.wait_seconds = wait_seconds;
    return stats;
}

struct FieldCsrPattern {
    cuiter::CsrMatrix matrix;
    std::vector<int32_t> full_positions;
};

FieldCsrPattern extract_field_submatrix_with_positions(const cuiter::CsrMatrix& full,
                                                       int32_t row_begin,
                                                       int32_t row_count,
                                                       int32_t col_begin,
                                                       int32_t col_count)
{
    FieldCsrPattern out;
    out.matrix.rows = row_count;
    out.matrix.cols = col_count;
    out.matrix.row_ptr.assign(static_cast<std::size_t>(row_count + 1), 0);
    for (int32_t local_row = 0; local_row < row_count; ++local_row) {
        const int32_t row = row_begin + local_row;
        for (int32_t pos = full.row_ptr[static_cast<std::size_t>(row)];
             pos < full.row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = full.col_idx[static_cast<std::size_t>(pos)];
            if (col >= col_begin && col < col_begin + col_count) {
                out.matrix.col_idx.push_back(col - col_begin);
                out.matrix.values.push_back(0.0);
                out.full_positions.push_back(pos);
            }
        }
        out.matrix.row_ptr[static_cast<std::size_t>(local_row + 1)] =
            static_cast<int32_t>(out.matrix.col_idx.size());
    }
    return out;
}

struct DeviceFieldCsr {
    cuiter::CsrMatrix pattern;
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<int32_t> d_full_positions;
    cuiter::DeviceBuffer<double> d_values;

    void initialize(const FieldCsrPattern& host)
    {
        pattern = host.matrix;
        d_row_ptr.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
        d_col_idx.assign(pattern.col_idx.data(), pattern.col_idx.size());
        d_full_positions.assign(host.full_positions.data(), host.full_positions.size());
        d_values.resize(static_cast<std::size_t>(pattern.nnz()));
    }
};

struct CusparseSpmvPlan {
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t x_vec = nullptr;
    cusparseDnVecDescr_t y_vec = nullptr;
    cuiter::DeviceBuffer<unsigned char> d_buffer;
    int32_t rows = 0;
    int32_t cols = 0;

    void initialize(cusparseHandle_t handle,
                    const cuiter::CsrMatrix& matrix,
                    const int32_t* d_row_ptr,
                    const int32_t* d_col_idx,
                    const double* d_values,
                    double* d_x,
                    double* d_y)
    {
        rows = matrix.rows;
        cols = matrix.cols;
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreateCsr(&mat,
                                                      rows,
                                                      cols,
                                                      static_cast<int64_t>(matrix.nnz()),
                                                      const_cast<int32_t*>(d_row_ptr),
                                                      const_cast<int32_t*>(d_col_idx),
                                                      const_cast<double*>(d_values),
                                                      CUSPARSE_INDEX_32I,
                                                      CUSPARSE_INDEX_32I,
                                                      CUSPARSE_INDEX_BASE_ZERO,
                                                      CUDA_R_64F));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreateDnVec(&x_vec, cols, d_x, CUDA_R_64F));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreateDnVec(&y_vec, rows, d_y, CUDA_R_64F));
        double alpha = 1.0;
        double beta = 0.0;
        std::size_t buffer_size = 0;
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                            &alpha,
                                                            mat,
                                                            x_vec,
                                                            &beta,
                                                            y_vec,
                                                            CUDA_R_64F,
                                                            CUSPARSE_SPMV_ALG_DEFAULT,
                                                            &buffer_size));
        d_buffer.resize(buffer_size);
    }

    void destroy()
    {
        if (mat != nullptr) {
            cusparseDestroySpMat(mat);
            mat = nullptr;
        }
        if (x_vec != nullptr) {
            cusparseDestroyDnVec(x_vec);
            x_vec = nullptr;
        }
        if (y_vec != nullptr) {
            cusparseDestroyDnVec(y_vec);
            y_vec = nullptr;
        }
    }

    void run(cusparseHandle_t handle, const double* d_x, double* d_y, double alpha, double beta)
    {
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseDnVecSetValues(x_vec, const_cast<double*>(d_x)));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseDnVecSetValues(y_vec, d_y));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSpMV(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 mat,
                                                 x_vec,
                                                 &beta,
                                                 y_vec,
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                                 d_buffer.data()));
    }
};

struct StreamEventTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;

    explicit StreamEventTimer(cudaStream_t new_stream)
        : stream(new_stream)
    {
        CUITER_CUDA_CHECK(cudaEventCreate(&start));
        CUITER_CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~StreamEventTimer()
    {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }

    void begin()
    {
        CUITER_CUDA_CHECK(cudaEventRecord(start, stream));
    }

    void end()
    {
        CUITER_CUDA_CHECK(cudaEventRecord(stop, stream));
    }

    double seconds() const
    {
        float milliseconds = 0.0F;
        CUITER_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return 0.001 * static_cast<double>(milliseconds);
    }
};

class ScopedCublasStream {
public:
    ScopedCublasStream(cublasHandle_t handle, cudaStream_t stream)
        : handle_(handle)
    {
        CUITER_CUBLAS_CHECK(cublasGetStream(handle_, &previous_));
        CUITER_CUBLAS_CHECK(cublasSetStream(handle_, stream));
    }

    ~ScopedCublasStream()
    {
        (void)cublasSetStream(handle_, previous_);
    }

    ScopedCublasStream(const ScopedCublasStream&) = delete;
    ScopedCublasStream& operator=(const ScopedCublasStream&) = delete;

private:
    cublasHandle_t handle_ = nullptr;
    cudaStream_t previous_ = nullptr;
};

struct DeviceFieldCudssCache {
    DeviceFieldCsr csr;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    DirectCudssSolver solver;
    cudaStream_t stream = nullptr;
    double analyze_seconds = 0.0;

    void initialize(const FieldCsrPattern& host, cudaStream_t new_stream)
    {
        stream = new_stream;
        csr.initialize(host);
        d_rhs.resize(static_cast<std::size_t>(csr.pattern.rows));
        d_x.resize(static_cast<std::size_t>(csr.pattern.rows));
        solver.initialize(csr.pattern,
                          csr.d_row_ptr.data(),
                          csr.d_col_idx.data(),
                          csr.d_values.data(),
                          d_rhs.data(),
                          d_x.data());
        solver.set_stream(stream);
        analyze_seconds = solver.analyze();
    }

    void set_stream(cudaStream_t new_stream)
    {
        stream = new_stream;
        solver.set_stream(stream);
    }
};

struct DeviceFieldCorrectionCache {
    DeviceFieldCudssCache j11;
    DeviceFieldCudssCache j22;
    DeviceFieldCsr j12;
    DeviceFieldCsr j21;
    cuiter::DeviceBuffer<double> d_rp1;
    cuiter::DeviceBuffer<double> d_rq1;
    cuiter::DeviceBuffer<double> d_dtheta0;
    cuiter::DeviceBuffer<double> d_dvm0;
    cuiter::DeviceBuffer<double> d_dtheta1;
    cuiter::DeviceBuffer<double> d_dvm1;
    CusparseSpmvPlan full_spmv;
    CusparseSpmvPlan j12_spmv;
    CusparseSpmvPlan j21_spmv;
    cudaStream_t j11_stream = nullptr;
    cudaStream_t j22_stream = nullptr;
    cudaStream_t cross_p_stream = nullptr;
    cudaStream_t cross_q_stream = nullptr;
    cusparseHandle_t cusparse_cross_p = nullptr;
    cusparseHandle_t cusparse_cross_q = nullptr;
    bool initialized = false;

    void initialize(MinimalNrDeviceContext& ctx, cusparseHandle_t cusparse)
    {
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&j11_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&j22_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&cross_p_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&cross_q_stream, cudaStreamNonBlocking));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreate(&cusparse_cross_p));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreate(&cusparse_cross_q));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSetStream(cusparse_cross_p, cross_p_stream));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSetStream(cusparse_cross_q, cross_q_stream));
        const FieldCsrPattern host_j11 =
            extract_field_submatrix_with_positions(ctx.j_pattern, 0, ctx.n_pvpq, 0, ctx.n_pvpq);
        const FieldCsrPattern host_j12 =
            extract_field_submatrix_with_positions(ctx.j_pattern, 0, ctx.n_pvpq, ctx.n_pvpq, ctx.n_pq);
        const FieldCsrPattern host_j21 =
            extract_field_submatrix_with_positions(ctx.j_pattern, ctx.n_pvpq, ctx.n_pq, 0, ctx.n_pvpq);
        const FieldCsrPattern host_j22 =
            extract_field_submatrix_with_positions(ctx.j_pattern,
                                                   ctx.n_pvpq,
                                                   ctx.n_pq,
                                                   ctx.n_pvpq,
                                                   ctx.n_pq);
        j11.initialize(host_j11, j11_stream);
        j22.initialize(host_j22, j22_stream);
        j12.initialize(host_j12);
        j21.initialize(host_j21);
        d_rp1.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_rq1.resize(static_cast<std::size_t>(ctx.n_pq));
        d_dtheta0.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_dvm0.resize(static_cast<std::size_t>(ctx.n_pq));
        d_dtheta1.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_dvm1.resize(static_cast<std::size_t>(ctx.n_pq));
        full_spmv.initialize(cusparse,
                             ctx.j_pattern,
                             ctx.d_J_row_ptr.data(),
                             ctx.d_J_col_idx.data(),
                             ctx.d_J_values.data(),
                             ctx.d_dx.data(),
                             ctx.d_ax.data());
        j12_spmv.initialize(cusparse_cross_p,
                            j12.pattern,
                            j12.d_row_ptr.data(),
                            j12.d_col_idx.data(),
                            j12.d_values.data(),
                            j22.d_x.data(),
                            d_rp1.data());
        j21_spmv.initialize(cusparse_cross_q,
                            j21.pattern,
                            j21.d_row_ptr.data(),
                            j21.d_col_idx.data(),
                            j21.d_values.data(),
                            j11.d_x.data(),
                            d_rq1.data());
        initialized = true;
    }

    void destroy()
    {
        full_spmv.destroy();
        j12_spmv.destroy();
        j21_spmv.destroy();
        if (j11_stream != nullptr) {
            j11.set_stream(nullptr);
            CUITER_CUDA_CHECK(cudaStreamDestroy(j11_stream));
            j11_stream = nullptr;
        }
        if (j22_stream != nullptr) {
            j22.set_stream(nullptr);
            CUITER_CUDA_CHECK(cudaStreamDestroy(j22_stream));
            j22_stream = nullptr;
        }
        if (cusparse_cross_p != nullptr) {
            cusparseDestroy(cusparse_cross_p);
            cusparse_cross_p = nullptr;
        }
        if (cusparse_cross_q != nullptr) {
            cusparseDestroy(cusparse_cross_q);
            cusparse_cross_q = nullptr;
        }
        if (cross_p_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(cross_p_stream));
            cross_p_stream = nullptr;
        }
        if (cross_q_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(cross_q_stream));
            cross_q_stream = nullptr;
        }
    }
};

struct FdlfConvention {
    double p_sign = 1.0;
    double q_sign = 1.0;
    bool p_scale_by_v = false;
    bool q_scale_by_v = false;
    std::string p_name = "r";
    std::string q_name = "r";
};

FdlfConvention parse_fdlf_convention(const HybridNrOptions& options)
{
    auto parse_one = [](const std::string& value, double& sign, bool& scale, std::string& name) {
        if (value == "auto") {
            sign = 1.0;
            scale = true;
            name = "r_over_v";
        } else if (value == "r") {
            sign = 1.0;
            scale = false;
            name = value;
        } else if (value == "-r") {
            sign = -1.0;
            scale = false;
            name = value;
        } else if (value == "r_over_v") {
            sign = 1.0;
            scale = true;
            name = value;
        } else if (value == "-r_over_v") {
            sign = -1.0;
            scale = true;
            name = value;
        } else if (value != "auto") {
            throw std::runtime_error("unsupported FDLF RHS convention: " + value);
        }
    };
    FdlfConvention convention;
    parse_one(options.fdlf_p_rhs, convention.p_sign, convention.p_scale_by_v, convention.p_name);
    parse_one(options.fdlf_q_rhs, convention.q_sign, convention.q_scale_by_v, convention.q_name);
    return convention;
}

cuiter::CsrMatrix build_bprime_matrix(const DumpCaseData& data,
                                      const std::vector<int32_t>& buses,
                                      double diag_shift = 0.0)
{
    std::vector<int32_t> bus_to_local(static_cast<std::size_t>(data.rows), -1);
    for (int32_t i = 0; i < static_cast<int32_t>(buses.size()); ++i) {
        bus_to_local[static_cast<std::size_t>(buses[static_cast<std::size_t>(i)])] = i;
    }

    cuiter::CsrMatrix matrix;
    matrix.rows = static_cast<int32_t>(buses.size());
    matrix.cols = matrix.rows;
    matrix.row_ptr.assign(static_cast<std::size_t>(matrix.rows + 1), 0);
    for (int32_t local_row = 0; local_row < matrix.rows; ++local_row) {
        const int32_t bus = buses[static_cast<std::size_t>(local_row)];
        bool has_diag = false;
        for (int32_t pos = data.indptr[static_cast<std::size_t>(bus)];
             pos < data.indptr[static_cast<std::size_t>(bus + 1)];
             ++pos) {
            const int32_t col_bus = data.indices[static_cast<std::size_t>(pos)];
            const int32_t local_col = bus_to_local[static_cast<std::size_t>(col_bus)];
            if (local_col < 0) {
                continue;
            }
            double value = -std::imag(data.ybus_data[static_cast<std::size_t>(pos)]);
            if (local_col == local_row) {
                value += diag_shift;
                has_diag = true;
            }
            matrix.col_idx.push_back(local_col);
            matrix.values.push_back(value);
        }
        if (!has_diag) {
            matrix.col_idx.push_back(local_row);
            matrix.values.push_back(diag_shift);
        }
        matrix.row_ptr[static_cast<std::size_t>(local_row + 1)] =
            static_cast<int32_t>(matrix.col_idx.size());
    }
    return matrix;
}

struct FdlfCudssCache {
    cuiter::CsrMatrix pattern;
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    DirectCudssSolver solver;
    cudaStream_t stream = nullptr;
    double analyze_seconds = 0.0;
    double factor_seconds = 0.0;

    void initialize(const cuiter::CsrMatrix& matrix, cudaStream_t new_stream)
    {
        stream = new_stream;
        pattern = matrix;
        d_row_ptr.assign(pattern.row_ptr.data(), pattern.row_ptr.size());
        d_col_idx.assign(pattern.col_idx.data(), pattern.col_idx.size());
        d_values.assign(pattern.values.data(), pattern.values.size());
        d_rhs.resize(static_cast<std::size_t>(pattern.rows));
        d_x.resize(static_cast<std::size_t>(pattern.rows));
        solver.initialize(pattern,
                          d_row_ptr.data(),
                          d_col_idx.data(),
                          d_values.data(),
                          d_rhs.data(),
                          d_x.data());
        solver.set_stream(stream);
        analyze_seconds = solver.analyze();
        factor_seconds = solver.factorize();
    }
};

struct FdlfBpBppCache {
    FdlfCudssCache bp;
    FdlfCudssCache bpp;
    DeviceFieldCsr j12;
    DeviceFieldCsr j21;
    CusparseSpmvPlan full_spmv;
    CusparseSpmvPlan j12_spmv;
    CusparseSpmvPlan j21_spmv;
    cuiter::DeviceBuffer<double> d_rp1;
    cuiter::DeviceBuffer<double> d_rq1;
    cuiter::DeviceBuffer<double> d_dtheta0;
    cuiter::DeviceBuffer<double> d_dvm0;
    cuiter::DeviceBuffer<double> d_dtheta1;
    cuiter::DeviceBuffer<double> d_dvm1;
    cudaStream_t theta_stream = nullptr;
    cudaStream_t v_stream = nullptr;
    cudaStream_t cross_p_stream = nullptr;
    cudaStream_t cross_q_stream = nullptr;
    cusparseHandle_t cusparse_cross_p = nullptr;
    cusparseHandle_t cusparse_cross_q = nullptr;
    FdlfConvention convention;
    bool initialized = false;

    void initialize(MinimalNrDeviceContext& ctx,
                    const DumpCaseData& data,
                    cusparseHandle_t)
    {
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&theta_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&v_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&cross_p_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&cross_q_stream, cudaStreamNonBlocking));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreate(&cusparse_cross_p));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreate(&cusparse_cross_q));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSetStream(cusparse_cross_p, cross_p_stream));
        CUPF_MINIMAL_CUSPARSE_CHECK(cusparseSetStream(cusparse_cross_q, cross_q_stream));

        std::vector<int32_t> pvpq;
        pvpq.reserve(static_cast<std::size_t>(ctx.n_pvpq));
        pvpq.insert(pvpq.end(), data.pv.begin(), data.pv.end());
        pvpq.insert(pvpq.end(), data.pq.begin(), data.pq.end());
        bp.initialize(build_bprime_matrix(data, pvpq), theta_stream);
        bpp.initialize(build_bprime_matrix(data, data.pq), v_stream);

        const FieldCsrPattern host_j12 =
            extract_field_submatrix_with_positions(ctx.j_pattern, 0, ctx.n_pvpq, ctx.n_pvpq, ctx.n_pq);
        const FieldCsrPattern host_j21 =
            extract_field_submatrix_with_positions(ctx.j_pattern, ctx.n_pvpq, ctx.n_pq, 0, ctx.n_pvpq);
        j12.initialize(host_j12);
        j21.initialize(host_j21);
        d_rp1.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_rq1.resize(static_cast<std::size_t>(ctx.n_pq));
        d_dtheta0.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_dvm0.resize(static_cast<std::size_t>(ctx.n_pq));
        d_dtheta1.resize(static_cast<std::size_t>(ctx.n_pvpq));
        d_dvm1.resize(static_cast<std::size_t>(ctx.n_pq));
        full_spmv.initialize(cusparse_cross_p,
                             ctx.j_pattern,
                             ctx.d_J_row_ptr.data(),
                             ctx.d_J_col_idx.data(),
                             ctx.d_J_values.data(),
                             ctx.d_dx.data(),
                             ctx.d_ax.data());
        j12_spmv.initialize(cusparse_cross_p,
                            j12.pattern,
                            j12.d_row_ptr.data(),
                            j12.d_col_idx.data(),
                            j12.d_values.data(),
                            d_dvm0.data(),
                            d_rp1.data());
        j21_spmv.initialize(cusparse_cross_q,
                            j21.pattern,
                            j21.d_row_ptr.data(),
                            j21.d_col_idx.data(),
                            j21.d_values.data(),
                            d_dtheta0.data(),
                            d_rq1.data());
        initialized = true;
    }

    void destroy()
    {
        bp.solver.set_stream(nullptr);
        bpp.solver.set_stream(nullptr);
        full_spmv.destroy();
        j12_spmv.destroy();
        j21_spmv.destroy();
        if (cusparse_cross_p != nullptr) {
            cusparseDestroy(cusparse_cross_p);
            cusparse_cross_p = nullptr;
        }
        if (cusparse_cross_q != nullptr) {
            cusparseDestroy(cusparse_cross_q);
            cusparse_cross_q = nullptr;
        }
        if (theta_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(theta_stream));
            theta_stream = nullptr;
        }
        if (v_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(v_stream));
            v_stream = nullptr;
        }
        if (cross_p_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(cross_p_stream));
            cross_p_stream = nullptr;
        }
        if (cross_q_stream != nullptr) {
            CUITER_CUDA_CHECK(cudaStreamDestroy(cross_q_stream));
            cross_q_stream = nullptr;
        }
    }
};

void apply_fdlf_bpbpp_2round(MinimalNrDeviceContext& ctx,
                             FdlfBpBppCache& cache,
                             HybridNrIterationLog& log)
{
    if (!cache.initialized) {
        throw std::runtime_error("FDLF B'/B'' cache was not initialized");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const double minus_one = -1.0;
    const double zero = 0.0;

    cudaEvent_t rhs0_ready = nullptr;
    cudaEvent_t theta0_ready = nullptr;
    cudaEvent_t v0_ready = nullptr;
    cudaEvent_t rp1_ready = nullptr;
    cudaEvent_t rq1_ready = nullptr;
    cudaEvent_t theta1_ready = nullptr;
    cudaEvent_t v1_ready = nullptr;
    cudaEvent_t dx_done = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&rhs0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rp1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rq1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&dx_done));

    StreamEventTimer rhs0_timer(cache.cross_p_stream);
    StreamEventTimer bp_solve0_timer(cache.theta_stream);
    StreamEventTimer bpp_solve0_timer(cache.v_stream);
    StreamEventTimer j12_value_timer(cache.cross_p_stream);
    StreamEventTimer j21_value_timer(cache.cross_q_stream);
    StreamEventTimer j12_spmv_timer(cache.cross_p_stream);
    StreamEventTimer j21_spmv_timer(cache.cross_q_stream);
    StreamEventTimer rhs1_p_timer(cache.cross_p_stream);
    StreamEventTimer rhs1_q_timer(cache.cross_q_stream);
    StreamEventTimer bp_solve1_timer(cache.theta_stream);
    StreamEventTimer bpp_solve1_timer(cache.v_stream);
    StreamEventTimer dx_accum_timer(cache.cross_p_stream);

    rhs0_timer.begin();
    launch_build_fdlf_rhs(ctx.n_pv,
                          ctx.n_pq,
                          ctx.d_F.data(),
                          ctx.d_F.data() + ctx.n_pvpq,
                          ctx.d_vm.data(),
                          ctx.d_pv.data(),
                          ctx.d_pq.data(),
                          cache.convention.p_sign,
                          cache.convention.q_sign,
                          cache.convention.p_scale_by_v,
                          cache.convention.q_scale_by_v,
                          cache.bp.d_rhs.data(),
                          cache.bpp.d_rhs.data(),
                          cache.cross_p_stream);
    rhs0_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rhs0_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.theta_stream, rhs0_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.theta_stream));
    bp_solve0_timer.begin();
    cache.bp.solver.solve_async();
    bp_solve0_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta0.data(),
                                      cache.bp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.theta_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(theta0_ready, cache.theta_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.v_stream, rhs0_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bpp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.v_stream));
    bpp_solve0_timer.begin();
    cache.bpp.solver.solve_async();
    bpp_solve0_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm0.data(),
                                      cache.bpp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.v_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(v0_ready, cache.v_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v0_ready, 0));
    j12_value_timer.begin();
    launch_scatter_field_values(cache.j12.pattern.nnz(),
                                cache.j12.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j12.d_values.data(),
                                cache.cross_p_stream);
    j12_value_timer.end();
    j12_spmv_timer.begin();
    cache.j12_spmv.run(cache.cusparse_cross_p, cache.d_dvm0.data(), cache.d_rp1.data(), minus_one, zero);
    j12_spmv_timer.end();
    rhs1_p_timer.begin();
    launch_build_fdlf_rhs_p(ctx.n_pv,
                            ctx.n_pq,
                            cache.d_rp1.data(),
                            ctx.d_vm.data(),
                            ctx.d_pv.data(),
                            ctx.d_pq.data(),
                            cache.convention.p_sign,
                            cache.convention.p_scale_by_v,
                            cache.bp.d_rhs.data(),
                            cache.cross_p_stream);
    rhs1_p_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rp1_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_q_stream, theta0_ready, 0));
    j21_value_timer.begin();
    launch_scatter_field_values(cache.j21.pattern.nnz(),
                                cache.j21.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j21.d_values.data(),
                                cache.cross_q_stream);
    j21_value_timer.end();
    j21_spmv_timer.begin();
    cache.j21_spmv.run(cache.cusparse_cross_q, cache.d_dtheta0.data(), cache.d_rq1.data(), minus_one, zero);
    j21_spmv_timer.end();
    rhs1_q_timer.begin();
    launch_build_fdlf_rhs_q(ctx.n_pq,
                            cache.d_rq1.data(),
                            ctx.d_vm.data(),
                            ctx.d_pq.data(),
                            cache.convention.q_sign,
                            cache.convention.q_scale_by_v,
                            cache.bpp.d_rhs.data(),
                            cache.cross_q_stream);
    rhs1_q_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rq1_ready, cache.cross_q_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.theta_stream, rp1_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.theta_stream));
    bp_solve1_timer.begin();
    cache.bp.solver.solve_async();
    bp_solve1_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta1.data(),
                                      cache.bp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.theta_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(theta1_ready, cache.theta_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.v_stream, rq1_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bpp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.v_stream));
    bpp_solve1_timer.begin();
    cache.bpp.solver.solve_async();
    bpp_solve1_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm1.data(),
                                      cache.bpp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.v_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(v1_ready, cache.v_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, theta1_ready, 0));
    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v1_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(ctx.d_dx.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                      cache.cross_p_stream));
    dx_accum_timer.begin();
    launch_accumulate_field_dx(ctx.n_pvpq,
                               ctx.n_pq,
                               cache.d_dtheta0.data(),
                               cache.d_dvm0.data(),
                               cache.d_dtheta1.data(),
                               cache.d_dvm1.data(),
                               true,
                               ctx.d_dx.data(),
                               cache.cross_p_stream);
    dx_accum_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(dx_done, cache.cross_p_stream));
    CUITER_CUDA_CHECK(cudaEventSynchronize(dx_done));

    const double bp_solve_seconds = bp_solve0_timer.seconds() + bp_solve1_timer.seconds();
    const double bpp_solve_seconds = bpp_solve0_timer.seconds() + bpp_solve1_timer.seconds();
    const double cross_seconds = j12_spmv_timer.seconds() + j21_spmv_timer.seconds();
    log.fdlf_bp_solve_seconds = bp_solve_seconds;
    log.fdlf_bpp_solve_seconds = bpp_solve_seconds;
    log.fdlf_cross_spmv_seconds = cross_seconds;
    log.fdlf_round0_wall_seconds = std::max(bp_solve0_timer.seconds(), bpp_solve0_timer.seconds());
    log.fdlf_round1_wall_seconds = std::max(bp_solve1_timer.seconds(), bpp_solve1_timer.seconds());
    log.fdlf_2round_wall_seconds = elapsed_seconds(wall_start);
    log.j12_value_update_seconds = j12_value_timer.seconds();
    log.j21_value_update_seconds = j21_value_timer.seconds();
    log.j12_spmv_seconds = j12_spmv_timer.seconds();
    log.j21_spmv_seconds = j21_spmv_timer.seconds();
    log.rhs_gather_seconds = rhs0_timer.seconds() + rhs1_p_timer.seconds() + rhs1_q_timer.seconds();
    log.dx_accum_seconds = dx_accum_timer.seconds();
    log.field_correction_wall_seconds = log.fdlf_2round_wall_seconds;
    log.field_correction_serial_sum_seconds = bp_solve_seconds + bpp_solve_seconds;
    log.linear_solve_seconds += log.fdlf_2round_wall_seconds;
    log.middle_solver_total_seconds += log.fdlf_2round_wall_seconds;
    log.linear_iters = 1;
    log.stop_reason += ":fdlf_bpbpp_2round";

    CUITER_CUDA_CHECK(cudaEventDestroy(rhs0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rp1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rq1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(dx_done));
}

void apply_bpbpp_residual_refinement(MinimalNrDeviceContext& ctx,
                                     cublasHandle_t cublas,
                                     FdlfBpBppCache& cache,
                                     HybridNrIterationLog& log)
{
    if (!cache.initialized) {
        throw std::runtime_error("B'/B'' refinement cache was not initialized");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const double one = 1.0;
    const double minus_one = -1.0;
    const double zero = 0.0;

    cudaEvent_t residual_ready = nullptr;
    cudaEvent_t theta0_ready = nullptr;
    cudaEvent_t v0_ready = nullptr;
    cudaEvent_t rp1_ready = nullptr;
    cudaEvent_t rq1_ready = nullptr;
    cudaEvent_t theta1_ready = nullptr;
    cudaEvent_t v1_ready = nullptr;
    cudaEvent_t dx_done = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&residual_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rp1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rq1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&dx_done));

    StreamEventTimer full_spmv_timer(cache.cross_p_stream);
    StreamEventTimer residual_axpy_timer(cache.cross_p_stream);
    StreamEventTimer rhs0_timer(cache.cross_p_stream);
    StreamEventTimer bp_solve0_timer(cache.theta_stream);
    StreamEventTimer bpp_solve0_timer(cache.v_stream);
    StreamEventTimer j12_value_timer(cache.cross_p_stream);
    StreamEventTimer j21_value_timer(cache.cross_q_stream);
    StreamEventTimer j12_spmv_timer(cache.cross_p_stream);
    StreamEventTimer j21_spmv_timer(cache.cross_q_stream);
    StreamEventTimer rhs1_p_timer(cache.cross_p_stream);
    StreamEventTimer rhs1_q_timer(cache.cross_q_stream);
    StreamEventTimer bp_solve1_timer(cache.theta_stream);
    StreamEventTimer bpp_solve1_timer(cache.v_stream);
    StreamEventTimer dx_accum_timer(cache.cross_p_stream);

    full_spmv_timer.begin();
    cache.full_spmv.run(cache.cusparse_cross_p, ctx.d_dx.data(), ctx.d_ax.data(), one, zero);
    full_spmv_timer.end();

    residual_axpy_timer.begin();
    {
        ScopedCublasStream scoped_stream(cublas, cache.cross_p_stream);
        CUITER_CUBLAS_CHECK(cublasDcopy(cublas,
                                        ctx.dimF,
                                        ctx.d_F.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                        ctx.dimF,
                                        &minus_one,
                                        ctx.d_ax.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
    }
    residual_axpy_timer.end();

    rhs0_timer.begin();
    launch_build_fdlf_rhs(ctx.n_pv,
                          ctx.n_pq,
                          ctx.d_linear_residual.data(),
                          ctx.d_linear_residual.data() + ctx.n_pvpq,
                          ctx.d_vm.data(),
                          ctx.d_pv.data(),
                          ctx.d_pq.data(),
                          cache.convention.p_sign,
                          cache.convention.q_sign,
                          cache.convention.p_scale_by_v,
                          cache.convention.q_scale_by_v,
                          cache.bp.d_rhs.data(),
                          cache.bpp.d_rhs.data(),
                          cache.cross_p_stream);
    rhs0_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(residual_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.theta_stream, residual_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.theta_stream));
    bp_solve0_timer.begin();
    cache.bp.solver.solve_async();
    bp_solve0_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta0.data(),
                                      cache.bp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.theta_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(theta0_ready, cache.theta_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.v_stream, residual_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bpp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.v_stream));
    bpp_solve0_timer.begin();
    cache.bpp.solver.solve_async();
    bpp_solve0_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm0.data(),
                                      cache.bpp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.v_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(v0_ready, cache.v_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v0_ready, 0));
    j12_value_timer.begin();
    launch_scatter_field_values(cache.j12.pattern.nnz(),
                                cache.j12.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j12.d_values.data(),
                                cache.cross_p_stream);
    j12_value_timer.end();
    j12_spmv_timer.begin();
    cache.j12_spmv.run(cache.cusparse_cross_p, cache.d_dvm0.data(), cache.d_rp1.data(), minus_one, zero);
    j12_spmv_timer.end();
    rhs1_p_timer.begin();
    launch_build_fdlf_rhs_p(ctx.n_pv,
                            ctx.n_pq,
                            cache.d_rp1.data(),
                            ctx.d_vm.data(),
                            ctx.d_pv.data(),
                            ctx.d_pq.data(),
                            cache.convention.p_sign,
                            cache.convention.p_scale_by_v,
                            cache.bp.d_rhs.data(),
                            cache.cross_p_stream);
    rhs1_p_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rp1_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_q_stream, theta0_ready, 0));
    j21_value_timer.begin();
    launch_scatter_field_values(cache.j21.pattern.nnz(),
                                cache.j21.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j21.d_values.data(),
                                cache.cross_q_stream);
    j21_value_timer.end();
    j21_spmv_timer.begin();
    cache.j21_spmv.run(cache.cusparse_cross_q, cache.d_dtheta0.data(), cache.d_rq1.data(), minus_one, zero);
    j21_spmv_timer.end();
    rhs1_q_timer.begin();
    launch_build_fdlf_rhs_q(ctx.n_pq,
                            cache.d_rq1.data(),
                            ctx.d_vm.data(),
                            ctx.d_pq.data(),
                            cache.convention.q_sign,
                            cache.convention.q_scale_by_v,
                            cache.bpp.d_rhs.data(),
                            cache.cross_q_stream);
    rhs1_q_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rq1_ready, cache.cross_q_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.theta_stream, rp1_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.theta_stream));
    bp_solve1_timer.begin();
    cache.bp.solver.solve_async();
    bp_solve1_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta1.data(),
                                      cache.bp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.theta_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(theta1_ready, cache.theta_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.v_stream, rq1_ready, 0));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.bpp.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.v_stream));
    bpp_solve1_timer.begin();
    cache.bpp.solver.solve_async();
    bpp_solve1_timer.end();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm1.data(),
                                      cache.bpp.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.v_stream));
    CUITER_CUDA_CHECK(cudaEventRecord(v1_ready, cache.v_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, theta1_ready, 0));
    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v1_ready, 0));
    dx_accum_timer.begin();
    launch_accumulate_field_dx(ctx.n_pvpq,
                               ctx.n_pq,
                               cache.d_dtheta0.data(),
                               cache.d_dvm0.data(),
                               cache.d_dtheta1.data(),
                               cache.d_dvm1.data(),
                               true,
                               ctx.d_dx.data(),
                               cache.cross_p_stream);
    dx_accum_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(dx_done, cache.cross_p_stream));
    CUITER_CUDA_CHECK(cudaEventSynchronize(dx_done));

    const double bp_solve_seconds = bp_solve0_timer.seconds() + bp_solve1_timer.seconds();
    const double bpp_solve_seconds = bpp_solve0_timer.seconds() + bpp_solve1_timer.seconds();
    const double cross_seconds = j12_spmv_timer.seconds() + j21_spmv_timer.seconds();
    log.full_residual_spmv_seconds = full_spmv_timer.seconds();
    log.residual_axpy_seconds = residual_axpy_timer.seconds();
    log.fdlf_bp_solve_seconds = bp_solve_seconds;
    log.fdlf_bpp_solve_seconds = bpp_solve_seconds;
    log.fdlf_cross_spmv_seconds = cross_seconds;
    log.fdlf_round0_wall_seconds = std::max(bp_solve0_timer.seconds(), bpp_solve0_timer.seconds());
    log.fdlf_round1_wall_seconds = std::max(bp_solve1_timer.seconds(), bpp_solve1_timer.seconds());
    log.fdlf_2round_wall_seconds = elapsed_seconds(wall_start);
    log.j12_value_update_seconds = j12_value_timer.seconds();
    log.j21_value_update_seconds = j21_value_timer.seconds();
    log.j12_spmv_seconds = j12_spmv_timer.seconds();
    log.j21_spmv_seconds = j21_spmv_timer.seconds();
    log.rhs_gather_seconds = rhs0_timer.seconds() + rhs1_p_timer.seconds() + rhs1_q_timer.seconds();
    log.dx_accum_seconds = dx_accum_timer.seconds();
    log.field_correction_wall_seconds = log.fdlf_2round_wall_seconds;
    log.field_correction_serial_sum_seconds = bp_solve_seconds + bpp_solve_seconds;
    log.linear_solve_seconds += log.fdlf_2round_wall_seconds;
    log.middle_solver_total_seconds += log.fdlf_2round_wall_seconds;
    log.stop_reason += ":bpbpp_residual_refinement";

    CUITER_CUDA_CHECK(cudaEventDestroy(residual_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rp1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rq1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(dx_done));
}

void apply_device_field_correction_dag(MinimalNrDeviceContext& ctx,
                                       cusparseHandle_t,
                                       cublasHandle_t cublas,
                                       DeviceFieldCorrectionCache& cache,
                                       HybridNrIterationLog& log)
{
    const auto wall_start = std::chrono::steady_clock::now();
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    cudaEvent_t residual_ready = nullptr;
    cudaEvent_t theta0_ready = nullptr;
    cudaEvent_t v0_ready = nullptr;
    cudaEvent_t rp1_ready = nullptr;
    cudaEvent_t rq1_ready = nullptr;
    cudaEvent_t theta1_ready = nullptr;
    cudaEvent_t v1_ready = nullptr;
    cudaEvent_t dx_done = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&residual_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v0_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rp1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&rq1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&theta1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&v1_ready));
    CUITER_CUDA_CHECK(cudaEventCreate(&dx_done));

    StreamEventTimer full_spmv_timer(cache.cross_p_stream);
    StreamEventTimer residual_axpy_timer(cache.cross_p_stream);
    StreamEventTimer rhs_timer(cache.cross_p_stream);
    StreamEventTimer j11_value_timer(cache.j11.stream);
    StreamEventTimer j22_value_timer(cache.j22.stream);
    StreamEventTimer j12_value_timer(cache.cross_p_stream);
    StreamEventTimer j21_value_timer(cache.cross_q_stream);
    StreamEventTimer j11_factor_timer(cache.j11.stream);
    StreamEventTimer j22_factor_timer(cache.j22.stream);
    StreamEventTimer j11_solve0_timer(cache.j11.stream);
    StreamEventTimer j22_solve0_timer(cache.j22.stream);
    StreamEventTimer j12_spmv_timer(cache.cross_p_stream);
    StreamEventTimer j21_spmv_timer(cache.cross_q_stream);
    StreamEventTimer j11_solve1_timer(cache.j11.stream);
    StreamEventTimer j22_solve1_timer(cache.j22.stream);
    StreamEventTimer dx_accum_timer(cache.cross_p_stream);
    StreamEventTimer copy_theta0_timer(cache.j11.stream);
    StreamEventTimer copy_v0_timer(cache.j22.stream);
    StreamEventTimer copy_rhs1_theta_timer(cache.j11.stream);
    StreamEventTimer copy_rhs1_v_timer(cache.j22.stream);
    StreamEventTimer copy_theta1_timer(cache.j11.stream);
    StreamEventTimer copy_v1_timer(cache.j22.stream);

    full_spmv_timer.begin();
    cache.full_spmv.run(cache.cusparse_cross_p, ctx.d_dx.data(), ctx.d_ax.data(), one, zero);
    full_spmv_timer.end();

    residual_axpy_timer.begin();
    {
        ScopedCublasStream scoped_stream(cublas, cache.cross_p_stream);
        CUITER_CUBLAS_CHECK(cublasDcopy(cublas,
                                        ctx.dimF,
                                        ctx.d_F.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                        ctx.dimF,
                                        &minus_one,
                                        ctx.d_ax.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
    }
    residual_axpy_timer.end();

    rhs_timer.begin();
    launch_copy_field_rhs(ctx.n_pvpq,
                          ctx.n_pq,
                          ctx.d_linear_residual.data(),
                          cache.j11.d_rhs.data(),
                          cache.j22.d_rhs.data(),
                          cache.cross_p_stream);
    rhs_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(residual_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j11.stream, residual_ready, 0));
    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j22.stream, residual_ready, 0));

    j11_value_timer.begin();
    launch_scatter_field_values(cache.j11.csr.pattern.nnz(),
                                cache.j11.csr.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j11.csr.d_values.data(),
                                cache.j11.stream);
    j11_value_timer.end();
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.j11.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.j11.stream));
    j11_factor_timer.begin();
    cache.j11.solver.factorize_async();
    j11_factor_timer.end();
    j11_solve0_timer.begin();
    cache.j11.solver.solve_async();
    j11_solve0_timer.end();
    copy_theta0_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta0.data(),
                                      cache.j11.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j11.stream));
    copy_theta0_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(theta0_ready, cache.j11.stream));

    j22_value_timer.begin();
    launch_scatter_field_values(cache.j22.csr.pattern.nnz(),
                                cache.j22.csr.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j22.csr.d_values.data(),
                                cache.j22.stream);
    j22_value_timer.end();
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.j22.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.j22.stream));
    j22_factor_timer.begin();
    cache.j22.solver.factorize_async();
    j22_factor_timer.end();
    j22_solve0_timer.begin();
    cache.j22.solver.solve_async();
    j22_solve0_timer.end();
    copy_v0_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm0.data(),
                                      cache.j22.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j22.stream));
    copy_v0_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(v0_ready, cache.j22.stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v0_ready, 0));
    j12_value_timer.begin();
    launch_scatter_field_values(cache.j12.pattern.nnz(),
                                cache.j12.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j12.d_values.data(),
                                cache.cross_p_stream);
    j12_value_timer.end();
    j12_spmv_timer.begin();
    cache.j12_spmv.run(cache.cusparse_cross_p, cache.d_dvm0.data(), cache.d_rp1.data(), minus_one, zero);
    j12_spmv_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rp1_ready, cache.cross_p_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_q_stream, theta0_ready, 0));
    j21_value_timer.begin();
    launch_scatter_field_values(cache.j21.pattern.nnz(),
                                cache.j21.d_full_positions.data(),
                                ctx.d_J_values.data(),
                                cache.j21.d_values.data(),
                                cache.cross_q_stream);
    j21_value_timer.end();
    j21_spmv_timer.begin();
    cache.j21_spmv.run(cache.cusparse_cross_q, cache.d_dtheta0.data(), cache.d_rq1.data(), minus_one, zero);
    j21_spmv_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(rq1_ready, cache.cross_q_stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j11.stream, rp1_ready, 0));
    copy_rhs1_theta_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.j11.d_rhs.data(),
                                      cache.d_rp1.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j11.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.j11.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cache.j11.stream));
    copy_rhs1_theta_timer.end();
    j11_solve1_timer.begin();
    cache.j11.solver.solve_async();
    j11_solve1_timer.end();
    copy_theta1_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dtheta1.data(),
                                      cache.j11.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j11.stream));
    copy_theta1_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(theta1_ready, cache.j11.stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j22.stream, rq1_ready, 0));
    copy_rhs1_v_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.j22.d_rhs.data(),
                                      cache.d_rq1.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j22.stream));
    CUITER_CUDA_CHECK(cudaMemsetAsync(cache.j22.d_x.data(),
                                      0,
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cache.j22.stream));
    copy_rhs1_v_timer.end();
    j22_solve1_timer.begin();
    cache.j22.solver.solve_async();
    j22_solve1_timer.end();
    copy_v1_timer.begin();
    CUITER_CUDA_CHECK(cudaMemcpyAsync(cache.d_dvm1.data(),
                                      cache.j22.d_x.data(),
                                      static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      cache.j22.stream));
    copy_v1_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(v1_ready, cache.j22.stream));

    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, theta1_ready, 0));
    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.cross_p_stream, v1_ready, 0));
    dx_accum_timer.begin();
    launch_accumulate_field_dx(ctx.n_pvpq,
                               ctx.n_pq,
                               cache.d_dtheta0.data(),
                               cache.d_dvm0.data(),
                               cache.d_dtheta1.data(),
                               cache.d_dvm1.data(),
                               true,
                               ctx.d_dx.data(),
                               cache.cross_p_stream);
    dx_accum_timer.end();
    CUITER_CUDA_CHECK(cudaEventRecord(dx_done, cache.cross_p_stream));

    const auto wait_start = std::chrono::steady_clock::now();
    CUITER_CUDA_CHECK(cudaEventSynchronize(dx_done));
    log.a1_event_wait_seconds = elapsed_seconds(wait_start);
    log.field_correction_wall_seconds = elapsed_seconds(wall_start);

    log.full_residual_spmv_seconds = full_spmv_timer.seconds();
    log.residual_axpy_seconds = residual_axpy_timer.seconds();
    log.rhs_gather_seconds = rhs_timer.seconds() +
                             copy_theta0_timer.seconds() +
                             copy_v0_timer.seconds() +
                             copy_rhs1_theta_timer.seconds() +
                             copy_rhs1_v_timer.seconds() +
                             copy_theta1_timer.seconds() +
                             copy_v1_timer.seconds();
    log.j11_value_update_seconds = j11_value_timer.seconds();
    log.j22_value_update_seconds = j22_value_timer.seconds();
    log.j12_value_update_seconds = j12_value_timer.seconds();
    log.j21_value_update_seconds = j21_value_timer.seconds();
    log.j11_factor_seconds = j11_factor_timer.seconds();
    log.j22_factor_seconds = j22_factor_timer.seconds();
    log.j11_solve_seconds = j11_solve0_timer.seconds() + j11_solve1_timer.seconds();
    log.j22_solve_seconds = j22_solve0_timer.seconds() + j22_solve1_timer.seconds();
    log.j11_solve_round1_seconds = j11_solve1_timer.seconds();
    log.j22_solve_round1_seconds = j22_solve1_timer.seconds();
    log.j12_spmv_seconds = j12_spmv_timer.seconds();
    log.j21_spmv_seconds = j21_spmv_timer.seconds();
    log.dx_accum_seconds = dx_accum_timer.seconds();

    log.field_correction_serial_sum_seconds =
        log.j11_factor_seconds + log.j11_solve_seconds +
        log.j22_factor_seconds + log.j22_solve_seconds;
    const double cudss_critical_path =
        std::max(log.j11_factor_seconds + log.j11_solve_seconds,
                 log.j22_factor_seconds + log.j22_solve_seconds);
    log.non_cudss_overhead_seconds =
        std::max(0.0, log.field_correction_wall_seconds - cudss_critical_path);
    const double accounted_seconds =
        log.full_residual_spmv_seconds + log.residual_axpy_seconds + log.rhs_gather_seconds +
        log.j11_value_update_seconds + log.j22_value_update_seconds +
        log.j12_value_update_seconds + log.j21_value_update_seconds +
        log.j11_factor_seconds + log.j22_factor_seconds +
        log.j11_solve_seconds + log.j22_solve_seconds +
        log.j12_spmv_seconds + log.j21_spmv_seconds + log.dx_accum_seconds;
    log.a1_unaccounted_seconds =
        std::max(0.0, log.field_correction_wall_seconds - accounted_seconds);

    const double value_update_seconds =
        log.j11_value_update_seconds + log.j22_value_update_seconds +
        log.j12_value_update_seconds + log.j21_value_update_seconds;
    const double field_factor_seconds = log.j11_factor_seconds + log.j22_factor_seconds;
    const double field_solve_seconds = log.j11_solve_seconds + log.j22_solve_seconds;
    log.linear_setup_seconds += value_update_seconds + field_factor_seconds;
    log.linear_solve_seconds += log.full_residual_spmv_seconds + log.residual_axpy_seconds +
                                log.rhs_gather_seconds + field_solve_seconds +
                                log.j12_spmv_seconds + log.j21_spmv_seconds +
                                log.dx_accum_seconds;
    log.middle_solver_total_seconds += log.field_correction_wall_seconds;
    log.stop_reason += ":device_a1_field_correction_dag";

    CUITER_CUDA_CHECK(cudaEventDestroy(residual_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v0_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rp1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(rq1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(theta1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(v1_ready));
    CUITER_CUDA_CHECK(cudaEventDestroy(dx_done));
}

void apply_device_field_correction(MinimalNrDeviceContext& ctx,
                                   const HybridNrOptions& options,
                                   cusparseHandle_t cusparse,
                                   cublasHandle_t cublas,
                                   DeviceFieldCorrectionCache& cache,
                                   HybridNrIterationLog& log)
{
    if (!cache.initialized) {
        throw std::runtime_error("device field correction cache was not initialized");
    }
    const bool use_j11_only = options.middle_solver == "bicgstab_block_jacobi_j11_device";
    const bool use_a1 = options.middle_solver == "bicgstab_block_jacobi_a1_device";
    if (use_a1 && options.a1_event_dag) {
        apply_device_field_correction_dag(ctx, cusparse, cublas, cache, log);
        return;
    }
    const auto wall_start = std::chrono::steady_clock::now();
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    log.full_residual_spmv_seconds = timed_on_stream(nullptr, [&] {
        cache.full_spmv.run(cusparse, ctx.d_dx.data(), ctx.d_ax.data(), one, zero);
    });
    log.residual_axpy_seconds = timed_on_stream(nullptr, [&] {
        CUITER_CUBLAS_CHECK(cublasDcopy(cublas,
                                        ctx.dimF,
                                        ctx.d_F.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                        ctx.dimF,
                                        &minus_one,
                                        ctx.d_ax.data(),
                                        1,
                                        ctx.d_linear_residual.data(),
                                        1));
    });
    log.rhs_gather_seconds = timed_on_stream(nullptr, [&] {
        launch_copy_field_rhs(ctx.n_pvpq,
                              ctx.n_pq,
                              ctx.d_linear_residual.data(),
                              cache.j11.d_rhs.data(),
                              cache.j22.d_rhs.data());
    });

    log.j11_value_update_seconds = timed_on_stream(nullptr, [&] {
        launch_scatter_field_values(cache.j11.csr.pattern.nnz(),
                                    cache.j11.csr.d_full_positions.data(),
                                    ctx.d_J_values.data(),
                                    cache.j11.csr.d_values.data());
    });
    if (!use_j11_only) {
        log.j22_value_update_seconds = timed_on_stream(nullptr, [&] {
            launch_scatter_field_values(cache.j22.csr.pattern.nnz(),
                                        cache.j22.csr.d_full_positions.data(),
                                        ctx.d_J_values.data(),
                                        cache.j22.csr.d_values.data());
        });
    }
    if (use_a1) {
        log.j12_value_update_seconds = timed_on_stream(nullptr, [&] {
            launch_scatter_field_values(cache.j12.pattern.nnz(),
                                        cache.j12.d_full_positions.data(),
                                        ctx.d_J_values.data(),
                                        cache.j12.d_values.data());
        });
        log.j21_value_update_seconds = timed_on_stream(nullptr, [&] {
            launch_scatter_field_values(cache.j21.pattern.nnz(),
                                        cache.j21.d_full_positions.data(),
                                        ctx.d_J_values.data(),
                                        cache.j21.d_values.data());
        });
    }

    cache.j11.d_x.memset_zero();
    if (!use_j11_only) {
        cache.j22.d_x.memset_zero();
    }
    cudaEvent_t round0_inputs_ready = nullptr;
    CUITER_CUDA_CHECK(cudaEventCreate(&round0_inputs_ready));
    CUITER_CUDA_CHECK(cudaEventRecord(round0_inputs_ready, nullptr));
    CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j11.stream, round0_inputs_ready, 0));
    if (!use_j11_only) {
        CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j22.stream, round0_inputs_ready, 0));
    }
    double cudss_overlapped_seconds = 0.0;
    if (use_j11_only) {
        log.j11_factor_seconds = timed_on_stream(cache.j11.stream, [&] {
            cache.j11.solver.factorize_async();
        });
        cudss_overlapped_seconds += log.j11_factor_seconds;
    } else {
        const ParallelPhaseStats factor0 = run_two_stream_phase(
            cache.j11.stream,
            [&] { cache.j11.solver.factorize_async(); },
            cache.j22.stream,
            [&] { cache.j22.solver.factorize_async(); });
        cudss_overlapped_seconds += factor0.wall_seconds;
        log.a1_event_wait_seconds += factor0.wait_seconds;
        log.j11_factor_seconds += factor0.left_seconds;
        log.j22_factor_seconds += factor0.right_seconds;
    }
    CUITER_CUDA_CHECK(cudaEventDestroy(round0_inputs_ready));
    if (use_j11_only) {
        log.j11_solve_seconds = timed_on_stream(cache.j11.stream, [&] {
            cache.j11.solver.solve_async();
        });
        cudss_overlapped_seconds += log.j11_solve_seconds;
    } else {
        const ParallelPhaseStats solve0 = run_two_stream_phase(
            cache.j11.stream,
            [&] { cache.j11.solver.solve_async(); },
            cache.j22.stream,
            [&] { cache.j22.solver.solve_async(); });
        cudss_overlapped_seconds += solve0.wall_seconds;
        log.a1_event_wait_seconds += solve0.wait_seconds;
        log.j11_solve_seconds += solve0.left_seconds;
        log.j22_solve_seconds += solve0.right_seconds;
    }

    const double copy_round0_seconds = timed_on_stream(nullptr, [&] {
        CUITER_CUDA_CHECK(cudaMemcpy(cache.d_dtheta0.data(),
                                     cache.j11.d_x.data(),
                                     static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        if (use_j11_only) {
            CUITER_CUDA_CHECK(cudaMemset(cache.d_dvm0.data(),
                                         0,
                                         static_cast<std::size_t>(ctx.n_pq) * sizeof(double)));
        } else {
            CUITER_CUDA_CHECK(cudaMemcpy(cache.d_dvm0.data(),
                                         cache.j22.d_x.data(),
                                         static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        }
    });

    double copy_round1_rhs_seconds = 0.0;
    double copy_round1_solution_seconds = 0.0;
    if (use_a1) {
        log.j12_spmv_seconds = timed_on_stream(nullptr, [&] {
            cache.j12_spmv.run(cusparse, cache.d_dvm0.data(), cache.d_rp1.data(), one, zero);
            CUITER_CUBLAS_CHECK(cublasDscal(cublas, ctx.n_pvpq, &minus_one, cache.d_rp1.data(), 1));
        });
        log.j21_spmv_seconds = timed_on_stream(nullptr, [&] {
            cache.j21_spmv.run(cusparse, cache.d_dtheta0.data(), cache.d_rq1.data(), one, zero);
            CUITER_CUBLAS_CHECK(cublasDscal(cublas, ctx.n_pq, &minus_one, cache.d_rq1.data(), 1));
        });
        copy_round1_rhs_seconds = timed_on_stream(nullptr, [&] {
            CUITER_CUDA_CHECK(cudaMemcpy(cache.j11.d_rhs.data(),
                                         cache.d_rp1.data(),
                                         static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
            CUITER_CUDA_CHECK(cudaMemcpy(cache.j22.d_rhs.data(),
                                         cache.d_rq1.data(),
                                         static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
            cache.j11.d_x.memset_zero();
            cache.j22.d_x.memset_zero();
        });
        cudaEvent_t round1_inputs_ready = nullptr;
        CUITER_CUDA_CHECK(cudaEventCreate(&round1_inputs_ready));
        CUITER_CUDA_CHECK(cudaEventRecord(round1_inputs_ready, nullptr));
        CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j11.stream, round1_inputs_ready, 0));
        CUITER_CUDA_CHECK(cudaStreamWaitEvent(cache.j22.stream, round1_inputs_ready, 0));

        const ParallelPhaseStats solve1 = run_two_stream_phase(
            cache.j11.stream,
            [&] { cache.j11.solver.solve_async(); },
            cache.j22.stream,
            [&] { cache.j22.solver.solve_async(); });
        CUITER_CUDA_CHECK(cudaEventDestroy(round1_inputs_ready));
        cudss_overlapped_seconds += solve1.wall_seconds;
        log.a1_event_wait_seconds += solve1.wait_seconds;
        log.j11_solve_round1_seconds = solve1.left_seconds;
        log.j22_solve_round1_seconds = solve1.right_seconds;
        log.j11_solve_seconds += solve1.left_seconds;
        log.j22_solve_seconds += solve1.right_seconds;

        copy_round1_solution_seconds = timed_on_stream(nullptr, [&] {
            CUITER_CUDA_CHECK(cudaMemcpy(cache.d_dtheta1.data(),
                                         cache.j11.d_x.data(),
                                         static_cast<std::size_t>(ctx.n_pvpq) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
            CUITER_CUDA_CHECK(cudaMemcpy(cache.d_dvm1.data(),
                                         cache.j22.d_x.data(),
                                         static_cast<std::size_t>(ctx.n_pq) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        });
    }

    log.dx_accum_seconds = timed_on_stream(nullptr, [&] {
        launch_accumulate_field_dx(ctx.n_pvpq,
                                   ctx.n_pq,
                                   cache.d_dtheta0.data(),
                                   cache.d_dvm0.data(),
                                   cache.d_dtheta1.data(),
                                   cache.d_dvm1.data(),
                                   use_a1,
                                   ctx.d_dx.data());
    });

    log.field_correction_wall_seconds = elapsed_seconds(wall_start);
    log.field_correction_serial_sum_seconds =
        log.j11_factor_seconds + log.j11_solve_seconds +
        log.j22_factor_seconds + log.j22_solve_seconds;
    log.non_cudss_overhead_seconds =
        std::max(0.0, log.field_correction_wall_seconds - cudss_overlapped_seconds);
    log.rhs_gather_seconds += copy_round0_seconds + copy_round1_rhs_seconds +
                              copy_round1_solution_seconds;
    const double accounted_seconds =
        cudss_overlapped_seconds + log.full_residual_spmv_seconds +
        log.residual_axpy_seconds + log.rhs_gather_seconds +
        log.j11_value_update_seconds + log.j22_value_update_seconds +
        log.j12_value_update_seconds + log.j21_value_update_seconds +
        log.j12_spmv_seconds + log.j21_spmv_seconds + log.dx_accum_seconds;
    log.a1_unaccounted_seconds =
        std::max(0.0, log.field_correction_wall_seconds - accounted_seconds);

    const double value_update_seconds =
        log.j11_value_update_seconds + log.j22_value_update_seconds +
        log.j12_value_update_seconds + log.j21_value_update_seconds;
    const double field_factor_seconds = log.j11_factor_seconds + log.j22_factor_seconds;
    const double field_solve_seconds = log.j11_solve_seconds + log.j22_solve_seconds;
    log.linear_setup_seconds += value_update_seconds + field_factor_seconds;
    log.linear_solve_seconds += log.full_residual_spmv_seconds + log.residual_axpy_seconds +
                                log.rhs_gather_seconds + field_solve_seconds +
                                log.j12_spmv_seconds + log.j21_spmv_seconds +
                                log.dx_accum_seconds;
    log.middle_solver_total_seconds += log.field_correction_wall_seconds;
    log.stop_reason += use_j11_only ? ":device_j11_field_correction" :
                       (use_a1 ? ":device_a1_field_correction" : ":device_a0_field_correction");
}

void apply_field_correction(MinimalNrDeviceContext& ctx,
                            const HybridNrOptions& options,
                            FieldCudssCache& j11_cache,
                            FieldCudssCache& j22_cache,
                            HybridNrIterationLog& log)
{
    const auto wall_start = std::chrono::steady_clock::now();

    cuiter::CsrMatrix host_matrix = ctx.j_pattern;
    host_matrix.values.assign(static_cast<std::size_t>(ctx.j_pattern.nnz()), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(ctx.dimF), 0.0);
    std::vector<double> dx0(static_cast<std::size_t>(ctx.dimF), 0.0);
    ctx.d_J_values.copy_to(host_matrix.values.data(), host_matrix.values.size());
    ctx.d_F.copy_to(rhs.data(), rhs.size());
    ctx.d_dx.copy_to(dx0.data(), dx0.size());

    const std::vector<double> residual = host_residual(host_matrix, rhs, dx0);
    const std::vector<double> r_p = host_slice(residual, 0, ctx.n_pvpq);
    const std::vector<double> r_q = host_slice(residual, ctx.n_pvpq, ctx.n_pq);
    const cuiter::CsrMatrix j11_values =
        extract_field_submatrix(host_matrix, 0, ctx.n_pvpq, 0, ctx.n_pvpq);
    const cuiter::CsrMatrix j22_values =
        extract_field_submatrix(host_matrix, ctx.n_pvpq, ctx.n_pq, ctx.n_pvpq, ctx.n_pq);

    j11_cache.assign_values_and_rhs(j11_values.values, r_p);
    j22_cache.assign_values_and_rhs(j22_values.values, r_q);
    const ParallelPhaseStats factor0 = run_two_stream_phase(
        j11_cache.stream,
        [&] { j11_cache.factorize_async(); },
        j22_cache.stream,
        [&] { j22_cache.factorize_async(); });
    const ParallelPhaseStats solve0 = run_two_stream_phase(
        j11_cache.stream,
        [&] { j11_cache.solve_async(); },
        j22_cache.stream,
        [&] { j22_cache.solve_async(); });

    std::vector<double> dtheta = j11_cache.copy_solution_to_host();
    std::vector<double> dvm = j22_cache.copy_solution_to_host();

    log.j11_factor_seconds += factor0.left_seconds;
    log.j11_solve_seconds += solve0.left_seconds;
    log.j22_factor_seconds += factor0.right_seconds;
    log.j22_solve_seconds += solve0.right_seconds;

    if (options.middle_solver == "bicgstab_block_jacobi_a1") {
        const std::vector<double> j12_dv =
            host_submatrix_vector_product(host_matrix, 0, ctx.n_pvpq, ctx.n_pvpq, ctx.n_pq, dvm);
        const std::vector<double> j21_dtheta =
            host_submatrix_vector_product(host_matrix, ctx.n_pvpq, ctx.n_pq, 0, ctx.n_pvpq, dtheta);
        std::vector<double> r_p1(j12_dv.size(), 0.0);
        std::vector<double> r_q1(j21_dtheta.size(), 0.0);
        for (std::size_t i = 0; i < j12_dv.size(); ++i) {
            r_p1[i] = -j12_dv[i];
        }
        for (std::size_t i = 0; i < j21_dtheta.size(); ++i) {
            r_q1[i] = -j21_dtheta[i];
        }
        j11_cache.assign_rhs(r_p1);
        j22_cache.assign_rhs(r_q1);
        const ParallelPhaseStats solve1 = run_two_stream_phase(
            j11_cache.stream,
            [&] { j11_cache.solve_async(); },
            j22_cache.stream,
            [&] { j22_cache.solve_async(); });
        const std::vector<double> dtheta1 = j11_cache.copy_solution_to_host();
        const std::vector<double> dvm1 = j22_cache.copy_solution_to_host();
        for (std::size_t i = 0; i < dtheta.size(); ++i) {
            dtheta[i] += dtheta1[i];
        }
        for (std::size_t i = 0; i < dvm.size(); ++i) {
            dvm[i] += dvm1[i];
        }
        log.j11_solve_seconds += solve1.left_seconds;
        log.j22_solve_seconds += solve1.right_seconds;
    }

    std::vector<double> dx = dx0;
    for (int32_t i = 0; i < ctx.n_pvpq; ++i) {
        dx[static_cast<std::size_t>(i)] += dtheta[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        dx[static_cast<std::size_t>(ctx.n_pvpq + i)] += dvm[static_cast<std::size_t>(i)];
    }
    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                 dx.data(),
                                 dx.size() * sizeof(double),
                                 cudaMemcpyHostToDevice));

    log.field_correction_wall_seconds = elapsed_seconds(wall_start);
    log.field_correction_serial_sum_seconds =
        log.j11_factor_seconds + log.j11_solve_seconds +
        log.j22_factor_seconds + log.j22_solve_seconds;
    log.linear_setup_seconds += log.j11_factor_seconds + log.j22_factor_seconds;
    log.linear_solve_seconds += log.j11_solve_seconds + log.j22_solve_seconds;
    log.middle_solver_total_seconds += log.field_correction_wall_seconds;
}

}  // namespace

HybridNrResult run_hybrid_nr_case(const DumpCaseData& data,
                                  const HybridNrOptions& options,
                                  double pure_cudss_total_seconds,
                                  int32_t pure_full_cudss_calls,
                                  int32_t case_id)
{
    validate_options(options);
    MinimalNrDeviceContext ctx(data);

    cublasHandle_t cublas = nullptr;
    CUITER_CUBLAS_CHECK(cublasCreate(&cublas));
    CUITER_CUBLAS_CHECK(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
    cusparseHandle_t cusparse = nullptr;
    CUPF_MINIMAL_CUSPARSE_CHECK(cusparseCreate(&cusparse));

    DirectCudssSolver cudss;
    cudss.initialize(ctx.j_pattern,
                     ctx.d_J_row_ptr.data(),
                     ctx.d_J_col_idx.data(),
                     ctx.d_J_values.data(),
                     ctx.d_F.data(),
                     ctx.d_dx.data());
    bool cudss_analyzed = false;
    const bool use_stale_refinement = is_stale_refinement_solver(options.middle_solver);
    cuiter::DeviceBuffer<double> d_stale_J_values;
    d_stale_J_values.resize(static_cast<std::size_t>(ctx.j_pattern.nnz()));
    DirectCudssSolver stale_cudss;
    stale_cudss.initialize(ctx.j_pattern,
                           ctx.d_J_row_ptr.data(),
                           ctx.d_J_col_idx.data(),
                           d_stale_J_values.data(),
                           ctx.d_linear_residual.data(),
                           ctx.d_dx_cudss_shadow.data());
    bool stale_cudss_analyzed = false;
    bool stale_factor_valid = false;
    int32_t stale_factor_age = 0;

    const bool use_fdlf_bpbpp = options.middle_solver == "fdlf_bpbpp_2round";
    cuiter::GmresSolver gmres(make_gmres_options(options, ctx));
    double middle_solver_analyze_setup_seconds = 0.0;
    if (!use_fdlf_bpbpp && !use_stale_refinement && options.solver != "pure_cudss") {
        const auto analyze_start = std::chrono::steady_clock::now();
        gmres.analyze(ctx.j_pattern);
        middle_solver_analyze_setup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - analyze_start).count();
    }
    const int32_t stale_correction_bicgstab_iters = stale_bicgstab_iters(options.middle_solver);
    const bool use_stale_prec_bicgstab = uses_stale_prec_bicgstab(options.middle_solver);
    const int32_t stale_correction_gmres_iters = stale_gmres_iters(options.middle_solver);
    const bool use_stale_bj_correction = options.middle_solver == "stale_BJ1";
    cuiter::GmresSolver stale_bicgstab(
        make_stale_bicgstab_options(options, std::max(stale_correction_bicgstab_iters, 1)));
    if (stale_correction_bicgstab_iters > 0 && !use_stale_prec_bicgstab &&
        options.solver != "pure_cudss") {
        const auto analyze_start = std::chrono::steady_clock::now();
        stale_bicgstab.analyze(ctx.j_pattern);
        middle_solver_analyze_setup_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - analyze_start).count();
    }
    cuiter::GmresSolver stale_bj(make_stale_bj_mr1_options(options, ctx));
    if (use_stale_bj_correction && options.solver != "pure_cudss") {
        const auto analyze_start = std::chrono::steady_clock::now();
        stale_bj.analyze(ctx.j_pattern);
        middle_solver_analyze_setup_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - analyze_start).count();
    }
    StalePrecBicgstabWorkspace stale_prec_workspace;
    StalePrecGmresWorkspace stale_gmres_workspace;
    cuiter::GmresSolver shadow_gmres(make_gmres_options(options, ctx));
    DirectCudssSolver shadow_cudss;
    bool shadow_cudss_analyzed = false;
    if (options.enable_shadow_dx_diagnostic) {
        shadow_gmres.analyze(ctx.j_pattern);
        shadow_cudss.initialize(ctx.j_pattern,
                                ctx.d_J_row_ptr.data(),
                                ctx.d_J_col_idx.data(),
                                ctx.d_J_values.data(),
                                ctx.d_F.data(),
                                ctx.d_dx_cudss_shadow.data());
    }

    const bool use_field_correction =
        options.middle_solver == "bicgstab_block_jacobi_a0" ||
        options.middle_solver == "bicgstab_block_jacobi_a1";
    const bool use_device_field_correction =
        options.middle_solver == "bicgstab_block_jacobi_a0_device" ||
        options.middle_solver == "bicgstab_block_jacobi_a1_device" ||
        options.middle_solver == "bicgstab_block_jacobi_j11_device";
    const bool use_bpbpp_refinement =
        options.middle_solver == "bicgstab_block_jacobi_bpbpp_refine";
    FieldCudssCache j11_cache;
    FieldCudssCache j22_cache;
    cudaStream_t j11_stream = nullptr;
    cudaStream_t j22_stream = nullptr;
    if (use_field_correction) {
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&j11_stream, cudaStreamNonBlocking));
        CUITER_CUDA_CHECK(cudaStreamCreateWithFlags(&j22_stream, cudaStreamNonBlocking));
        cuiter::CsrMatrix host_pattern = ctx.j_pattern;
        host_pattern.values.assign(static_cast<std::size_t>(ctx.j_pattern.nnz()), 0.0);
        j11_cache.initialize(extract_field_submatrix(host_pattern,
                                                     0,
                                                     ctx.n_pvpq,
                                                     0,
                                                     ctx.n_pvpq));
        j22_cache.initialize(extract_field_submatrix(host_pattern,
                                                     ctx.n_pvpq,
                                                     ctx.n_pq,
                                                     ctx.n_pvpq,
                                                     ctx.n_pq));
        j11_cache.set_stream(j11_stream);
        j22_cache.set_stream(j22_stream);
    }
    DeviceFieldCorrectionCache device_field_cache;
    if (use_device_field_correction) {
        device_field_cache.initialize(ctx, cusparse);
    }
    FdlfBpBppCache fdlf_cache;
    if (use_fdlf_bpbpp || use_bpbpp_refinement) {
        fdlf_cache.initialize(ctx, data, cusparse);
        fdlf_cache.convention = parse_fdlf_convention(options);
    }

    HybridNrResult result;
    result.case_name = data.case_name;
    result.buses = data.rows;
    result.n = ctx.dimF;
    result.nnz = ctx.j_pattern.nnz();
    result.pure_cudss_total_seconds = pure_cudss_total_seconds;
    result.pure_full_cudss_calls = pure_full_cudss_calls;
    result.gmres_block_size = options.block_size;
    result.gmres_restart = options.gmres_restart;
    result.gmres_max_iters = options.gmres_max_iters;
    result.bicgstab_iters = options.bicgstab_iters;
    result.gmres_rtol = options.gmres_rtol;
    result.gmres_fixed_iter_mode = options.gmres_fixed_iter_mode;
    result.middle_solver_analyze_setup_seconds = middle_solver_analyze_setup_seconds;
    if (use_device_field_correction) {
        result.field_correction_analyze_setup_seconds =
            device_field_cache.j11.analyze_seconds + device_field_cache.j22.analyze_seconds;
    }
    if (use_fdlf_bpbpp || use_bpbpp_refinement) {
        result.fdlf_bp_analyze_seconds = fdlf_cache.bp.analyze_seconds;
        result.fdlf_bp_factor_seconds = fdlf_cache.bp.factor_seconds;
        result.fdlf_bpp_analyze_seconds = fdlf_cache.bpp.analyze_seconds;
        result.fdlf_bpp_factor_seconds = fdlf_cache.bpp.factor_seconds;
        result.fdlf_p_rhs = fdlf_cache.convention.p_name;
        result.fdlf_q_rhs = fdlf_cache.convention.q_name;
    }

    if (options.full_cudss_analyze_before_loop && !use_fdlf_bpbpp) {
        if (use_stale_refinement) {
            result.full_cudss_analyze_setup_seconds = stale_cudss.analyze();
            stale_cudss_analyzed = true;
        } else {
            result.full_cudss_analyze_setup_seconds = cudss.analyze();
            cudss_analyzed = true;
        }
    }

    auto run_current_full_factor_solve = [&](HybridNrIterationLog& log) {
        if (use_stale_refinement) {
            log.linear_setup_seconds += timed_with_sync([&] {
                CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_linear_residual.data(),
                                             ctx.d_F.data(),
                                             static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                             cudaMemcpyDeviceToDevice));
            });
            if (!stale_cudss_analyzed) {
                log.linear_setup_seconds += stale_cudss.analyze();
                stale_cudss_analyzed = true;
            }
            log.linear_setup_seconds += timed_with_sync([&] {
                CUITER_CUDA_CHECK(cudaMemcpy(d_stale_J_values.data(),
                                             ctx.d_J_values.data(),
                                             static_cast<std::size_t>(ctx.j_pattern.nnz()) * sizeof(double),
                                             cudaMemcpyDeviceToDevice));
            });
            const double factorize_seconds = stale_cudss.factorize();
            const double solve_seconds = stale_cudss.solve();
            log.linear_setup_seconds += factorize_seconds;
            log.linear_solve_seconds += solve_seconds;
            log.fallback_cudss_setup_seconds +=
                log.solver_used == "cudss_fallback" ? factorize_seconds : 0.0;
            log.fallback_cudss_solve_seconds +=
                log.solver_used == "cudss_fallback" ? solve_seconds : 0.0;
            log.stale_solve_f_seconds += solve_seconds;
            log.stale_solve_calls += 1;
            timed_with_sync([&] {
                CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                             ctx.d_dx_cudss_shadow.data(),
                                             static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                             cudaMemcpyDeviceToDevice));
            });
            stale_factor_valid = true;
            stale_factor_age = 0;
        } else {
            if (!cudss_analyzed) {
                log.linear_setup_seconds += cudss.analyze();
                cudss_analyzed = true;
            }
            const double factorize_seconds = cudss.factorize();
            const double solve_seconds = cudss.solve();
            log.linear_setup_seconds += factorize_seconds;
            log.linear_solve_seconds += solve_seconds;
            if (log.solver_used == "cudss_fallback") {
                log.fallback_cudss_setup_seconds += factorize_seconds;
                log.fallback_cudss_solve_seconds += solve_seconds;
            }
        }
        ++result.cudss_calls;
    };

    auto run_diagnostic_full_solve = [&](HybridNrIterationLog& log, std::vector<double>& p_full) {
        if (use_stale_refinement) {
            throw std::runtime_error("diagnostic global basis is not implemented for stale solvers");
        }
        if (!cudss_analyzed) {
            log.linear_setup_seconds += cudss.analyze();
            cudss_analyzed = true;
        }
        const double factorize_seconds = cudss.factorize();
        const double solve_seconds = cudss.solve();
        log.shadow_cudss_factorize_seconds += factorize_seconds;
        log.shadow_cudss_solve_seconds += solve_seconds;
        ++result.diagnostic_full_cudss_calls;
        log.used_diagnostic_full_cudss = true;
        p_full.assign(static_cast<std::size_t>(ctx.dimF), 0.0);
        ctx.d_dx.copy_to(p_full.data(), p_full.size());
    };

    auto refresh_stale_factor_only = [&](HybridNrIterationLog& log) {
        if (!stale_cudss_analyzed) {
            log.linear_setup_seconds += stale_cudss.analyze();
            stale_cudss_analyzed = true;
        }
        log.linear_setup_seconds += timed_with_sync([&] {
            CUITER_CUDA_CHECK(cudaMemcpy(d_stale_J_values.data(),
                                         ctx.d_J_values.data(),
                                         static_cast<std::size_t>(ctx.j_pattern.nnz()) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        });
        const double factorize_seconds = stale_cudss.factorize();
        log.linear_setup_seconds += factorize_seconds;
        log.fallback_cudss_setup_seconds += factorize_seconds;
        stale_factor_valid = true;
        stale_factor_age = 0;
        ++result.cudss_calls;
    };

    auto run_stale_gmres1_candidate = [&](HybridNrIterationLog& log,
                                          const MismatchNorms& norm_reference) {
        log.linear_solve_seconds += timed_with_sync([&] {
            CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_linear_residual.data(),
                                         ctx.d_F.data(),
                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        });
        log.stale_solve_f_seconds += stale_cudss.solve();
        log.stale_solve_calls += 1;
        log.linear_solve_seconds += log.stale_solve_f_seconds;
        timed_with_sync([&] {
            CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                         ctx.d_dx_cudss_shadow.data(),
                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
        });
        double residual_rel = 0.0;
        double residual_spmv_seconds = 0.0;
        double residual_axpy_seconds = 0.0;
        log.linear_residual_after_r0 =
            compute_residual_into_linear_buffer(ctx,
                                                cublas,
                                                ctx.d_dx.data(),
                                                norm_reference.two,
                                                residual_rel,
                                                residual_spmv_seconds,
                                                residual_axpy_seconds);
        log.current_j_spmv_r0_seconds += residual_spmv_seconds;
        log.residual_axpy_seconds += residual_axpy_seconds;
        log.current_j_spmv_calls += 1;
        log.linear_abs_res = log.linear_residual_after_r0;
        log.linear_rel_res = residual_rel;

        const double gmres_total_before = log.gmres_total_seconds;
        run_stale_preconditioned_gmres(ctx,
                                       cublas,
                                       stale_cudss,
                                       1,
                                       stale_gmres_workspace,
                                       log);
        const double gmres_total_delta = log.gmres_total_seconds - gmres_total_before;
        log.linear_solve_seconds += gmres_total_delta;
        log.middle_solver_total_seconds += gmres_total_delta;

        residual_spmv_seconds = 0.0;
        residual_axpy_seconds = 0.0;
        log.linear_residual_after_gmres =
            compute_residual_into_linear_buffer(ctx,
                                                cublas,
                                                ctx.d_dx.data(),
                                                norm_reference.two,
                                                residual_rel,
                                                residual_spmv_seconds,
                                                residual_axpy_seconds);
        log.current_j_spmv_r1_seconds += residual_spmv_seconds;
        log.residual_axpy_seconds += residual_axpy_seconds;
        log.current_j_spmv_calls += 1;
        log.linear_abs_res = log.linear_residual_after_gmres;
        log.linear_rel_res = residual_rel;
        log.middle_solver_total_seconds +=
            log.stale_solve_f_seconds + log.current_j_spmv_r0_seconds +
            log.current_j_spmv_r1_seconds + log.residual_axpy_seconds;
    };

    const auto total_start = std::chrono::steady_clock::now();
    double diagnostic_seconds_total = 0.0;
    int32_t consecutive_iterative_failures = 0;
    double accepted_max_rel = 0.0;
    double accepted_sum_rel = 0.0;
    double accepted_sum_ratio = 0.0;
    int32_t accepted_count = 0;
    bool has_previous_dx = false;
    bool bj_numeric_setup_valid = false;
    int32_t bj_numeric_setup_uses = 0;
    MismatchNorms current_norms{};
    GlobalPostCorrectionBasis global_basis;
    global_basis.max_rank = options.global_rank;
    global_basis.orth_tol = options.global_orth_tol;
    const bool use_global_post_correction =
        options.global_correction == "post" && options.global_rank > 0 &&
        options.middle_solver == "gmres_block_ilu0";

    for (int32_t iter = 0; iter < options.max_nr_iters; ++iter) {
        const double initial_mismatch_seconds = timed_with_sync([&] {
            current_norms = compute_mismatch(ctx, cublas);
        });
        if (current_norms.inf <= options.nr_mismatch_inf_tol ||
            (options.nr_mismatch_2_tol > 0.0 && current_norms.two <= options.nr_mismatch_2_tol)) {
            result.converged = true;
            result.nr_iters = iter;
            result.final_mismatch_inf = current_norms.inf;
            result.final_mismatch_2 = current_norms.two;
            result.stop_reason = "nr_converged";
            (void)initial_mismatch_seconds;
            break;
        }

        HybridNrIterationLog log;
        log.case_id = case_id;
        log.case_name = data.case_name;
        log.nr_iter = iter;
        log.mismatch_inf_before = current_norms.inf;
        log.mismatch_2_before = current_norms.two;
        diagnostic_seconds_total += dump_iteration_f(options, ctx, data.case_name, iter, "before");
        const auto iter_start = std::chrono::steady_clock::now();

        log.jacobian_seconds = timed_with_sync([&] {
            fill_jacobian(ctx);
        });

        const bool middle_accept_limit_reached =
            options.max_middle_accepts >= 0 &&
            result.accepted_gmres_steps >= options.max_middle_accepts;
        const bool a1_accept_limit_reached =
            options.max_a1_middle_accepts >= 0 &&
            (options.middle_solver == "bicgstab_block_jacobi_a1_device" ||
             options.middle_solver == "bicgstab_block_jacobi_bpbpp_refine") &&
            result.accepted_gmres_steps >= options.max_a1_middle_accepts;
        const bool use_cudss =
            middle_accept_limit_reached ||
            a1_accept_limit_reached ||
            should_use_cudss(options, iter, current_norms.inf, result.gmres_calls);
        if (!use_cudss && options.enable_shadow_dx_diagnostic) {
            diagnostic_seconds_total += run_shadow_dx_diagnostic(ctx,
                                                                 cublas,
                                                                 shadow_gmres,
                                                                 shadow_cudss,
                                                                 shadow_cudss_analyzed,
                                                                 current_norms.two,
                                                                 options,
                                                                 log);
        }
        if (use_cudss) {
            log.solver_used = cudss_solver_name(options, iter, current_norms.inf);
            if (log.solver_used == "cudss_polish") {
                ++result.polish_calls;
            }
            log.factor_age = 0;
            run_current_full_factor_solve(log);

            timed_with_sync([&] {
                log.linear_abs_res =
                    compute_linear_residual(ctx, cublas, current_norms.two, log.linear_rel_res);
            });
            log.scaled_linear_abs_res = log.linear_abs_res;
            log.scaled_linear_rel_res = log.linear_rel_res;
            log.unscaled_linear_abs_res = log.linear_abs_res;
            log.unscaled_linear_rel_res = log.linear_rel_res;
            log.voltage_update_seconds = timed_with_sync([&] {
                apply_voltage_update(ctx);
            });
            log.mismatch_recompute_seconds = timed_with_sync([&] {
                current_norms = compute_mismatch(ctx, cublas);
            });
            log.mismatch_inf_after = current_norms.inf;
            log.mismatch_2_after = current_norms.two;
            diagnostic_seconds_total += dump_iteration_f(options, ctx, data.case_name, iter, "after");
            log.step_accepted = true;
            log.stop_reason = "cudss_step";
            consecutive_iterative_failures = 0;
            bj_numeric_setup_valid = false;
            bj_numeric_setup_uses = 0;
            const bool converged_after_cudss =
                current_norms.inf <= options.nr_mismatch_inf_tol ||
                (options.nr_mismatch_2_tol > 0.0 &&
                 current_norms.two <= options.nr_mismatch_2_tol);
            const bool next_step_would_polish =
                options.cudss_polish_threshold >= 0.0 &&
                current_norms.inf <= options.cudss_polish_threshold;
            const bool should_prebuild_bj =
                options.solver != "pure_cudss" &&
                !converged_after_cudss &&
                !next_step_would_polish &&
                (options.bj_setup == "numeric_reuse_after_full_cudss" ||
                 options.bj_setup == "reuse_after_full_cudss");
            if (should_prebuild_bj) {
                const double bj_prebuild_seconds = timed_with_sync([&] {
                    if (use_stale_bj_correction) {
                        stale_bj.setup(ctx.d_J_values.data());
                    } else {
                        gmres.setup(ctx.d_J_values.data());
                    }
                });
                bj_numeric_setup_valid = true;
                bj_numeric_setup_uses = 0;
                log.linear_setup_seconds += bj_prebuild_seconds;
                log.bj_setup_total_seconds = bj_prebuild_seconds;
                const auto& setup_timings = use_stale_bj_correction
                                                ? stale_bj.preconditioner().timings()
                                                : gmres.preconditioner().timings();
                log.bj_metadata_setup_seconds = 0.0;
                log.bj_value_update_seconds = setup_timings.block_extract_seconds;
                log.bj_inverse_build_seconds = setup_timings.block_lu_seconds;
            }
        } else if (use_fdlf_bpbpp) {
            log.solver_used = "fdlf_bpbpp_2round";
            ++result.gmres_calls;

            apply_fdlf_bpbpp_2round(ctx, fdlf_cache, log);
            timed_with_sync([&] {
                log.linear_abs_res =
                    compute_linear_residual(ctx, cublas, current_norms.two, log.linear_rel_res);
            });
            log.scaled_linear_abs_res = log.linear_abs_res;
            log.scaled_linear_rel_res = log.linear_rel_res;
            log.unscaled_linear_abs_res = log.linear_abs_res;
            log.unscaled_linear_rel_res = log.linear_rel_res;

            if (dx_is_bad(ctx, cublas, current_norms.inf, options.dx_safety_check)) {
                log.step_accepted = false;
                log.stop_reason = "fdlf_bad_dx:" + log.stop_reason;
                log.nr_iter_total_seconds =
                    std::max(0.0, elapsed_seconds(iter_start) - log.shadow_dx_diagnostic_seconds);
                result.iteration_logs.push_back(log);
                result.nr_iters = iter + 1;
                result.stop_reason = "fdlf_bad_dx";
                break;
            }

            log.voltage_update_seconds = timed_with_sync([&] {
                apply_voltage_update(ctx);
            });
            log.mismatch_recompute_seconds = timed_with_sync([&] {
                current_norms = compute_mismatch(ctx, cublas);
            });
            log.mismatch_inf_after = current_norms.inf;
            log.mismatch_2_after = current_norms.two;
            diagnostic_seconds_total += dump_iteration_f(options, ctx, data.case_name, iter, "after");
            log.step_accepted = true;
            ++result.accepted_gmres_steps;
            consecutive_iterative_failures = 0;
        } else {
            const bool use_mr1 = options.middle_solver == "mr1_block_jacobi" ||
                                 options.middle_solver == "mr1_block_jacobi_coarse";
            const bool use_mr2 = options.middle_solver == "mr2_block_jacobi_coarse";
            const bool use_bicgstab = options.middle_solver == "bicgstab_block_jacobi" ||
                                      use_field_correction ||
                                      use_device_field_correction ||
                                      use_bpbpp_refinement;
            const bool use_block_ilu0 = options.middle_solver == "bicgstab_block_ilu0" ||
                                        options.middle_solver == "gmres_block_ilu0";
            const bool use_gmres_block_ilu0 = options.middle_solver == "gmres_block_ilu0";
            const bool use_ginkgo_parilut =
                options.middle_solver == "ginkgo_parilut_bicgstab";
            log.solver_used = use_mr2 ? "mr2_middle" :
                              (use_mr1 ? "mr1_middle" : "gmres_middle");
            if (use_gmres_block_ilu0) {
                log.solver_used = "gmres_middle";
            } else if (use_bicgstab || use_block_ilu0) {
                log.solver_used = "bicgstab_middle";
            } else if (use_ginkgo_parilut) {
                log.solver_used = "ginkgo_parilut_middle";
            }
            if (options.middle_solver == "bicgstab_block_jacobi_a0") {
                log.solver_used = "bicgstab_a0_middle";
            } else if (options.middle_solver == "bicgstab_block_jacobi_a1") {
                log.solver_used = "bicgstab_a1_middle";
            } else if (options.middle_solver == "bicgstab_block_jacobi_a0_device") {
                log.solver_used = "bicgstab_a0_device_middle";
            } else if (options.middle_solver == "bicgstab_block_jacobi_a1_device") {
                log.solver_used = "bicgstab_a1_device_middle";
            } else if (options.middle_solver == "bicgstab_block_jacobi_j11_device") {
                log.solver_used = "bicgstab_j11_device_middle";
            } else if (options.middle_solver == "bicgstab_block_jacobi_bpbpp_refine") {
                log.solver_used = "bicgstab_bpbpp_refine_middle";
            } else if (use_stale_refinement) {
                log.solver_used = options.middle_solver;
            }
            ++result.gmres_calls;
            // Experimental fast path for always-accept middle-solver timing.
            // Normal hybrid policies need the backup for fallback, damping,
            // scaled trial steps, and residual-based post-correction trials.
            const bool skip_middle_backup =
                options.skip_middle_backup &&
                !options.enable_shadow_dx_diagnostic &&
                !options.enable_damped_iterative_step &&
                !options.enable_scaled_mr1_step &&
                options.global_correction == "none" &&
                !options.enable_cudss_fallback &&
                options.fallback_policy == "off";
            if (!skip_middle_backup) {
                ctx.backup_state_and_rhs();
            }
            cuiter::CsrMatrix global_host_matrix;
            std::vector<double> global_host_rhs;
            std::vector<double> global_p_gmres_host;
            std::vector<double> global_p_selected_host;
            bool global_basis_candidate_available = false;

            if (use_stale_refinement) {
                log.factor_age = stale_factor_age;
                if (!stale_factor_valid) {
                    if (options.middle_solver == "stale_GMRES1_refresh") {
                        refresh_stale_factor_only(log);
                        log.stop_reason = "stale_factor_initial_refresh";
                    } else {
                        log.stop_reason = "stale_factor_missing";
                    }
                }
                log.linear_solve_seconds += timed_with_sync([&] {
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_linear_residual.data(),
                                                 ctx.d_F.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyDeviceToDevice));
                });
                log.stale_solve_f_seconds = stale_cudss.solve();
                log.stale_solve_calls += 1;
                log.linear_solve_seconds += log.stale_solve_f_seconds;
                timed_with_sync([&] {
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                 ctx.d_dx_cudss_shadow.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyDeviceToDevice));
                });
                double residual_rel = 0.0;
                double residual_spmv_seconds = 0.0;
                double residual_axpy_seconds = 0.0;
                log.linear_residual_after_r0 =
                    compute_residual_into_linear_buffer(ctx,
                                                        cublas,
                                                        ctx.d_dx.data(),
                                                        current_norms.two,
                                                        residual_rel,
                                                        residual_spmv_seconds,
                                                        residual_axpy_seconds);
                log.current_j_spmv_r0_seconds += residual_spmv_seconds;
                log.residual_axpy_seconds += residual_axpy_seconds;
                log.current_j_spmv_calls += 1;
                log.linear_abs_res = log.linear_residual_after_r0;
                log.linear_rel_res = residual_rel;

                if (options.middle_solver == "stale_R1_Richardson" ||
                    options.middle_solver == "stale_R2_Richardson") {
                    log.stale_solve_r0_seconds = stale_cudss.solve();
                    log.stale_solve_calls += 1;
                    log.linear_solve_seconds += log.stale_solve_r0_seconds;
                    timed_with_sync([&] {
                        const double one = 1.0;
                        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                                        ctx.dimF,
                                                        &one,
                                                        ctx.d_dx_cudss_shadow.data(),
                                                        1,
                                                        ctx.d_dx.data(),
                                                        1));
                    });
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_r1_richardson =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_r1_richardson;
                    log.linear_rel_res = residual_rel;
                }

                if (options.middle_solver == "stale_R2_Richardson") {
                    log.stale_solve_r1_seconds = stale_cudss.solve();
                    log.stale_solve_calls += 1;
                    log.linear_solve_seconds += log.stale_solve_r1_seconds;
                    timed_with_sync([&] {
                        const double one = 1.0;
                        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                                        ctx.dimF,
                                                        &one,
                                                        ctx.d_dx_cudss_shadow.data(),
                                                        1,
                                                        ctx.d_dx.data(),
                                                        1));
                    });
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_r2_richardson =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_r2_richardson;
                    log.linear_rel_res = residual_rel;
                }

                if (use_stale_prec_bicgstab) {
                    run_stale_preconditioned_bicgstab(ctx,
                                                      cublas,
                                                      stale_cudss,
                                                      stale_correction_bicgstab_iters,
                                                      stale_prec_workspace,
                                                      log);
                    log.linear_solve_seconds += log.bicgstab_total_seconds;
                    log.middle_solver_total_seconds += log.bicgstab_total_seconds;
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_bicgstab =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_bicgstab;
                    log.linear_rel_res = residual_rel;
                } else if (stale_correction_gmres_iters > 0) {
                    run_stale_preconditioned_gmres(ctx,
                                                   cublas,
                                                   stale_cudss,
                                                   stale_correction_gmres_iters,
                                                   stale_gmres_workspace,
                                                   log);
                    log.linear_solve_seconds += log.gmres_total_seconds;
                    log.middle_solver_total_seconds += log.gmres_total_seconds;
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_gmres =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_gmres;
                    log.linear_rel_res = residual_rel;
                } else if (use_stale_bj_correction) {
                    bool current_bj_cache_reused = false;
                    log.gmres_trial_setup_seconds = timed_with_sync([&] {
                        bool rebuild_bj = options.bj_setup == "every_middle" ||
                                          options.bj_setup == "value_update_only" ||
                                          !bj_numeric_setup_valid;
                        if (options.bj_setup == "reuse_for_2_middle_steps" &&
                            bj_numeric_setup_valid && bj_numeric_setup_uses >= 2) {
                            rebuild_bj = true;
                        }
                        if (rebuild_bj) {
                            stale_bj.setup(ctx.d_J_values.data());
                            bj_numeric_setup_valid = true;
                            bj_numeric_setup_uses = 0;
                        } else {
                            stale_bj.refresh_matrix_values(ctx.d_J_values.data());
                            current_bj_cache_reused = true;
                        }
                    });
                    log.bj_cache_reused = current_bj_cache_reused;
                    const cuiter::LinearSolveResult correction_result =
                        stale_bj.solve_device(ctx.d_J_values.data(),
                                              ctx.d_linear_residual.data(),
                                              ctx.d_dx_diff_shadow.data());
                    ++bj_numeric_setup_uses;
                    timed_with_sync([&] {
                        const double one = 1.0;
                        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                                        ctx.dimF,
                                                        &one,
                                                        ctx.d_dx_diff_shadow.data(),
                                                        1,
                                                        ctx.d_dx.data(),
                                                        1));
                    });
                    log.gmres_trial_solve_seconds = correction_result.timings.solve_total_seconds;
                    log.linear_setup_seconds += log.gmres_trial_setup_seconds;
                    log.linear_solve_seconds +=
                        log.gmres_trial_setup_seconds + log.gmres_trial_solve_seconds;
                    log.middle_solver_total_seconds +=
                        log.gmres_trial_setup_seconds + log.gmres_trial_solve_seconds;
                    log.linear_iters = correction_result.iterations;
                    log.gmres_refinement_iters = correction_result.iterations;
                    log.block_jacobi_apply_seconds =
                        correction_result.timings.block_jacobi_apply_seconds;
                    log.preconditioner_total_seconds =
                        correction_result.timings.preconditioner_total_seconds;
                    log.mr1_spmv_seconds = correction_result.timings.mr1_spmv_seconds;
                    log.mr1_fused_dot_seconds = correction_result.timings.mr1_fused_dot_seconds;
                    log.mr1_update_seconds = correction_result.timings.mr1_update_seconds;
                    log.gmres_spmv_seconds = correction_result.timings.mr1_spmv_seconds;
                    log.gmres_dot_seconds = correction_result.timings.mr1_fused_dot_seconds;
                    log.gmres_update_seconds = correction_result.timings.mr1_update_seconds;
                    log.gmres_total_seconds = correction_result.timings.middle_solver_total_seconds;
                    log.current_j_spmv_calls += 1;
                    log.block_extract_seconds = correction_result.timings.block_extract_seconds;
                    log.block_inverse_seconds = correction_result.timings.block_lu_seconds;
                    log.bj_metadata_setup_seconds = 0.0;
                    log.bj_value_update_seconds =
                        current_bj_cache_reused ? 0.0 : correction_result.timings.block_extract_seconds;
                    log.bj_inverse_build_seconds =
                        current_bj_cache_reused ? 0.0 : correction_result.timings.block_lu_seconds;
                    log.bj_setup_total_seconds = log.gmres_trial_setup_seconds;
                    log.partition_mode = correction_result.block_stats.partition_mode;
                    log.num_bus_partitions = correction_result.block_stats.num_blocks;
                    log.min_block_unknowns = correction_result.block_stats.min_block_size;
                    log.max_block_unknowns = correction_result.block_stats.max_block_size;
                    log.avg_block_unknowns = correction_result.block_stats.avg_block_size;
                    log.std_block_unknowns = correction_result.block_stats.std_block_size;
                    log.diagonal_block_nnz_ratio =
                        correction_result.block_stats.diagonal_block_nnz_ratio;
                    log.offblock_nnz_ratio = correction_result.block_stats.offblock_nnz_ratio;
                    log.diagonal_weighted_coupling_ratio =
                        correction_result.block_stats.diagonal_weighted_coupling_ratio;
                    log.offblock_weighted_coupling_ratio =
                        correction_result.block_stats.offblock_weighted_coupling_ratio;
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_gmres =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_gmres;
                    log.linear_rel_res = residual_rel;
                    log.stop_reason = correction_result.stop_reason;
                } else if (stale_correction_bicgstab_iters > 0) {
                    const double bicgstab_setup_seconds = timed_with_sync([&] {
                        stale_bicgstab.setup(ctx.d_J_values.data());
                    });
                    const cuiter::LinearSolveResult correction_result =
                        stale_bicgstab.solve_device(ctx.d_J_values.data(),
                                                    ctx.d_linear_residual.data(),
                                                    ctx.d_dx_diff_shadow.data());
                    timed_with_sync([&] {
                        const double one = 1.0;
                        CUITER_CUBLAS_CHECK(cublasDaxpy(cublas,
                                                        ctx.dimF,
                                                        &one,
                                                        ctx.d_dx_diff_shadow.data(),
                                                        1,
                                                        ctx.d_dx.data(),
                                                        1));
                    });
                    log.bicgstab_refinement_iters = correction_result.iterations;
                    log.bicgstab_total_seconds =
                        bicgstab_setup_seconds + correction_result.timings.bicgstab_total_seconds;
                    log.bicgstab_spmv_seconds = correction_result.timings.bicgstab_spmv_seconds;
                    log.bicgstab_dot_reduction_seconds =
                        correction_result.timings.bicgstab_dot_reduction_seconds;
                    log.bicgstab_update_seconds =
                        bicgstab_setup_seconds + correction_result.timings.bicgstab_update_seconds;
                    log.linear_solve_seconds += log.bicgstab_total_seconds;
                    log.middle_solver_total_seconds += log.bicgstab_total_seconds;
                    residual_spmv_seconds = 0.0;
                    residual_axpy_seconds = 0.0;
                    log.linear_residual_after_bicgstab =
                        compute_residual_into_linear_buffer(ctx,
                                                            cublas,
                                                            ctx.d_dx.data(),
                                                            current_norms.two,
                                                            residual_rel,
                                                            residual_spmv_seconds,
                                                            residual_axpy_seconds);
                    log.current_j_spmv_r1_seconds += residual_spmv_seconds;
                    log.residual_axpy_seconds += residual_axpy_seconds;
                    log.current_j_spmv_calls += 1;
                    log.linear_abs_res = log.linear_residual_after_bicgstab;
                    log.linear_rel_res = residual_rel;
                    log.stop_reason = correction_result.stop_reason;
                } else if (log.stop_reason.empty()) {
                    log.stop_reason = options.middle_solver;
                }
                log.middle_solver_total_seconds +=
                    log.stale_solve_f_seconds + log.stale_solve_r0_seconds +
                    log.stale_solve_r1_seconds + log.current_j_spmv_r0_seconds +
                    log.current_j_spmv_r1_seconds + log.residual_axpy_seconds;
                log.scaled_linear_abs_res = log.linear_abs_res;
                log.scaled_linear_rel_res = log.linear_rel_res;
                log.unscaled_linear_abs_res = log.linear_abs_res;
                log.unscaled_linear_rel_res = log.linear_rel_res;
            } else if (use_ginkgo_parilut) {
#if defined(CUITER_WITH_GINKGO)
                const cuiter::cpu_pilot::CpuBlockIlu0Result linear_result =
                    solve_ginkgo_parilut_middle(ctx, options);
                CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                             linear_result.solution.data(),
                                             static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                             cudaMemcpyHostToDevice));
                log.gmres_trial_setup_seconds =
                    linear_result.setup_seconds + linear_result.factor_seconds;
                log.gmres_trial_solve_seconds = linear_result.bicgstab_total_seconds;
                log.linear_setup_seconds = log.gmres_trial_setup_seconds;
                log.linear_solve_seconds = log.gmres_trial_solve_seconds;
                log.linear_iters = linear_result.iterations;
                log.linear_abs_res = linear_result.residual_norm2;
                log.linear_rel_res = linear_result.relative_residual_norm2;
                log.scaled_linear_abs_res = log.linear_abs_res;
                log.scaled_linear_rel_res = log.linear_rel_res;
                log.unscaled_linear_abs_res = log.linear_abs_res;
                log.unscaled_linear_rel_res = log.linear_rel_res;
                log.block_ilu_factor_seconds = linear_result.factor_seconds;
                log.preconditioner_total_seconds = linear_result.factor_seconds;
                log.bicgstab_total_seconds = linear_result.bicgstab_total_seconds;
                log.bicgstab_refinement_iters = linear_result.iterations;
                log.middle_solver_total_seconds =
                    linear_result.setup_seconds + linear_result.factor_seconds +
                    linear_result.bicgstab_total_seconds;
                log.block_ilu_failed = linear_result.factor_failed;
                log.stop_reason = linear_result.stop_reason;
#else
                log.stop_reason = "ginkgo_not_built";
                log.block_ilu_failed = true;
#endif
            } else if (use_block_ilu0) {
                const cuiter::cpu_pilot::CpuBlockIlu0Result linear_result =
                    solve_cpu_block_ilu0_middle(ctx,
                                                options,
                                                &global_host_matrix,
                                                &global_host_rhs);
                std::vector<double> selected_step = linear_result.solution;
                global_p_gmres_host = linear_result.solution;
                global_basis_candidate_available = true;
                if (use_global_post_correction && use_gmres_block_ilu0) {
                    GlobalPostCorrectionResult correction =
                        apply_global_post_correction(global_host_matrix,
                                                     global_host_rhs,
                                                     global_p_gmres_host,
                                                     global_basis);
                    log.global_correction_attempted = correction.attempted;
                    log.global_correction_used = correction.used;
                    log.global_correction_skipped_reason = correction.skipped_reason;
                    log.global_basis_rank_before = correction.rank_before;
                    log.global_basis_rank_after = global_basis.rank();
                    log.global_linear_res_before = correction.linear_before;
                    log.global_linear_res_after = correction.linear_after;
                    log.global_correction_gain = correction.correction_gain;
                    log.global_correction_norm_ratio = correction.correction_norm_ratio;
                    log.global_correction_seconds = correction.total_seconds;
                    log.global_az_seconds = correction.az_seconds;
                    log.global_dense_ls_seconds = correction.dense_ls_seconds;
                    result.global_correction_seconds += correction.total_seconds;
                    result.global_az_seconds += correction.az_seconds;
                    result.global_dense_ls_seconds += correction.dense_ls_seconds;
                    if (correction.attempted) {
                        ++result.corrections_attempted;
                    }
                    if (correction.used) {
                        selected_step = correction.corrected_step;
                        ++result.corrections_accepted;
                        log.stop_reason = "global_post_corrected";
                    } else if (correction.attempted) {
                        ++result.corrections_skipped;
                    }
                }
                if (options.field_gain_correction == "ls2" && use_gmres_block_ilu0) {
                    const FieldGainResult gain =
                        apply_field_gain_ls2(global_host_matrix,
                                             global_host_rhs,
                                             selected_step,
                                             ctx.n_pvpq,
                                             options);
                    log.field_gain_attempted = gain.attempted;
                    log.field_gain_accepted = gain.accepted;
                    log.field_gain_skipped_reason = gain.skipped_reason;
                    log.gamma_theta = gain.gamma_theta;
                    log.gamma_v = gain.gamma_v;
                    log.lin_res_before_gain = gain.linear_before;
                    log.lin_res_after_gain = gain.linear_after;
                    log.gain_step_norm_ratio = gain.step_norm_ratio;
                    log.field_gain_seconds = gain.seconds;
                    if (gain.accepted) {
                        selected_step = gain.step;
                    }
                }
                if (options.theta_j11_correction != "none" && use_gmres_block_ilu0) {
                    const ThetaCorrectionResult theta_corr =
                        apply_theta_correction(global_host_matrix,
                                               global_host_rhs,
                                               selected_step,
                                               ctx.n_pvpq,
                                               options);
                    log.theta_corr_attempted = theta_corr.attempted;
                    log.theta_corr_accepted = theta_corr.accepted;
                    log.theta_corr_skipped_reason = theta_corr.skipped_reason;
                    log.theta_p_scalar_beta = theta_corr.beta;
                    log.j11_corr_norm = theta_corr.corr_norm;
                    log.j11_corr_res_before = theta_corr.j11_res_before;
                    log.j11_corr_res_after = theta_corr.j11_res_after;
                    log.p_res_before = theta_corr.p_res_before;
                    log.p_res_after = theta_corr.p_res_after;
                    log.q_res_before = theta_corr.q_res_before;
                    log.q_res_after = theta_corr.q_res_after;
                    log.theta_corr_seconds = theta_corr.seconds;
                    if (theta_corr.accepted) {
                        selected_step = theta_corr.step;
                    }
                }
                CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                             selected_step.data(),
                                             static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                             cudaMemcpyHostToDevice));
                global_p_selected_host = selected_step;
                log.gmres_trial_setup_seconds =
                    linear_result.setup_seconds + linear_result.factor_seconds;
                log.gmres_trial_solve_seconds = linear_result.bicgstab_total_seconds;
                log.linear_setup_seconds = log.gmres_trial_setup_seconds;
                log.linear_solve_seconds = log.gmres_trial_solve_seconds;
                log.linear_iters = linear_result.iterations;
                log.linear_abs_res = linear_result.residual_norm2;
                log.linear_rel_res = linear_result.relative_residual_norm2;
                log.scaled_linear_abs_res = log.linear_abs_res;
                log.scaled_linear_rel_res = log.linear_rel_res;
                log.unscaled_linear_abs_res = log.linear_abs_res;
                log.unscaled_linear_rel_res = log.linear_rel_res;
                log.block_ilu_factor_seconds = linear_result.factor_seconds;
                log.block_ilu_apply_seconds = linear_result.apply_seconds;
                log.block_ilu_forward_seconds = linear_result.forward_seconds;
                log.block_ilu_backward_seconds = linear_result.backward_seconds;
                log.block_ilu_l_levels = linear_result.l_levels;
                log.block_ilu_u_levels = linear_result.u_levels;
                log.block_ilu_avg_level_width = linear_result.avg_level_width;
                log.block_ilu_max_level_width = linear_result.max_level_width;
                log.block_ilu_block_nnz = linear_result.block_nnz;
                log.block_ilu_failed = linear_result.factor_failed;
                log.preconditioner_total_seconds = linear_result.apply_seconds;
                if (options.middle_solver == "gmres_block_ilu0") {
                    log.gmres_total_seconds = linear_result.gmres_total_seconds;
                    log.gmres_spmv_seconds = linear_result.spmv_seconds;
                    log.gmres_dot_seconds = linear_result.dot_seconds;
                    log.gmres_orthogonalization_seconds =
                        linear_result.gmres_orthogonalization_seconds;
                    log.gmres_update_seconds = linear_result.update_seconds;
                    log.gmres_refinement_iters = linear_result.iterations;
                } else {
                    log.bicgstab_total_seconds = linear_result.bicgstab_total_seconds;
                    log.bicgstab_spmv_seconds = linear_result.spmv_seconds;
                    log.bicgstab_dot_reduction_seconds = linear_result.dot_seconds;
                    log.bicgstab_update_seconds = linear_result.update_seconds;
                    log.bicgstab_refinement_iters = linear_result.iterations;
                }
                log.middle_solver_total_seconds =
                    linear_result.setup_seconds + linear_result.factor_seconds +
                    linear_result.bicgstab_total_seconds;
                log.partition_mode = "unknown_metis_block_coloring";
                log.num_bus_partitions = linear_result.num_blocks;
                log.stop_reason = linear_result.stop_reason +
                                  (log.global_correction_used ? ":global_post_corrected" : "");
            } else {
                bool current_bj_cache_reused = false;
                log.gmres_trial_setup_seconds = timed_with_sync([&] {
                    bool rebuild_bj = options.bj_setup == "every_middle" ||
                                      options.bj_setup == "value_update_only" ||
                                      !bj_numeric_setup_valid;
                    if (options.bj_setup == "reuse_for_2_middle_steps" &&
                        bj_numeric_setup_valid && bj_numeric_setup_uses >= 2) {
                        rebuild_bj = true;
                    }
                    if (rebuild_bj) {
                        gmres.setup(ctx.d_J_values.data());
                        bj_numeric_setup_valid = true;
                        bj_numeric_setup_uses = 0;
                    } else {
                        gmres.refresh_matrix_values(ctx.d_J_values.data());
                        current_bj_cache_reused = true;
                    }
                });
                log.linear_setup_seconds = log.gmres_trial_setup_seconds;
                log.bj_cache_reused = current_bj_cache_reused;
                if (options.previous_dx_warm_start) {
                    timed_with_sync([&] {
                        if (has_previous_dx) {
                            CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                         ctx.d_prev_dx.data(),
                                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                         cudaMemcpyDeviceToDevice));
                        } else {
                            CUITER_CUDA_CHECK(cudaMemset(ctx.d_dx.data(),
                                                         0,
                                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double)));
                        }
                    });
                }
                const cuiter::LinearSolveResult linear_result =
                    gmres.solve_device(ctx.d_J_values.data(), ctx.d_F.data(), ctx.d_dx.data());
                ++bj_numeric_setup_uses;
                log.gmres_trial_solve_seconds = linear_result.timings.solve_total_seconds;
                log.linear_solve_seconds = log.gmres_trial_solve_seconds;
                log.linear_iters = linear_result.iterations;
                log.linear_abs_res = linear_result.residual_norm2;
                log.linear_rel_res = linear_result.relative_residual_norm2;
                log.scaled_linear_abs_res = linear_result.scaled_residual_norm2;
                log.scaled_linear_rel_res = linear_result.scaled_relative_residual_norm2;
                log.unscaled_linear_abs_res = linear_result.unscaled_residual_norm2;
                log.unscaled_linear_rel_res = linear_result.unscaled_relative_residual_norm2;
                log.block_jacobi_apply_seconds = linear_result.timings.block_jacobi_apply_seconds;
                log.ras_setup_seconds = linear_result.timings.ras_setup_seconds;
                log.ras_apply_seconds = linear_result.timings.ras_apply_seconds;
                log.ras_gather_seconds = linear_result.timings.ras_gather_seconds;
                log.ras_local_gemv_seconds = linear_result.timings.ras_local_gemv_seconds;
                log.ras_scatter_seconds = linear_result.timings.ras_scatter_seconds;
                log.coarse_az0_spmv_seconds = linear_result.timings.coarse_az0_spmv_seconds;
                log.coarse_compress_seconds = linear_result.timings.coarse_compress_seconds;
                log.coarse_solve_seconds = linear_result.timings.coarse_solve_seconds;
                log.coarse_expand_seconds = linear_result.timings.coarse_expand_seconds;
                log.coarse_total_seconds = linear_result.timings.coarse_total_seconds;
                log.preconditioner_total_seconds = linear_result.timings.preconditioner_total_seconds;
                log.mr1_spmv_seconds = linear_result.timings.mr1_spmv_seconds;
                log.mr1_fused_dot_seconds = linear_result.timings.mr1_fused_dot_seconds;
                log.mr1_update_seconds = linear_result.timings.mr1_update_seconds;
                log.mr2_w1_spmv_seconds = linear_result.timings.mr2_w1_spmv_seconds;
                log.bicgstab_total_seconds = linear_result.timings.bicgstab_total_seconds;
                log.bicgstab_spmv_seconds = linear_result.timings.bicgstab_spmv_seconds;
                log.bicgstab_dot_reduction_seconds =
                    linear_result.timings.bicgstab_dot_reduction_seconds;
                log.bicgstab_update_seconds = linear_result.timings.bicgstab_update_seconds;
                log.bicgstab_scalar_sync_seconds = linear_result.timings.bicgstab_scalar_sync_seconds;
                log.middle_solver_total_seconds = linear_result.timings.middle_solver_total_seconds;
                log.coarse_failed = linear_result.timings.coarse_failed;
                log.scaling_row_norm_seconds = linear_result.timings.scaling_row_norm_seconds;
                log.scaling_col_norm_seconds = linear_result.timings.scaling_col_norm_seconds;
                log.scaling_apply_values_seconds = linear_result.timings.scaling_apply_values_seconds;
                log.scaling_apply_rhs_seconds = linear_result.timings.scaling_apply_rhs_seconds;
                log.scaling_total_seconds = linear_result.timings.scaling_total_seconds;
                log.weighted_graph_build_seconds = linear_result.timings.weighted_graph_build_seconds;
                log.metis_partition_seconds = linear_result.timings.metis_partition_seconds;
                log.permutation_build_seconds = linear_result.timings.permutation_build_seconds;
                log.block_extract_seconds = linear_result.timings.block_extract_seconds;
                log.block_inverse_seconds = linear_result.timings.block_lu_seconds;
                log.bj_metadata_setup_seconds = 0.0;
                log.bj_value_update_seconds =
                    current_bj_cache_reused ? 0.0 : linear_result.timings.block_extract_seconds;
                log.bj_inverse_build_seconds =
                    current_bj_cache_reused ? 0.0 : linear_result.timings.block_lu_seconds;
                log.bj_setup_total_seconds = log.gmres_trial_setup_seconds;
                log.dr_min = linear_result.dr_min;
                log.dr_max = linear_result.dr_max;
                log.dr_geomean = linear_result.dr_geomean;
                log.dc_min = linear_result.dc_min;
                log.dc_max = linear_result.dc_max;
                log.dc_geomean = linear_result.dc_geomean;
                log.row_norm_cv_before = linear_result.row_norm_cv_before;
                log.row_norm_cv_after = linear_result.row_norm_cv_after;
                log.col_norm_cv_before = linear_result.col_norm_cv_before;
                log.col_norm_cv_after = linear_result.col_norm_cv_after;
                log.partition_mode = linear_result.block_stats.partition_mode;
                log.num_bus_partitions = linear_result.block_stats.num_blocks;
                log.min_block_unknowns = linear_result.block_stats.min_block_size;
                log.max_block_unknowns = linear_result.block_stats.max_block_size;
                log.avg_block_unknowns = linear_result.block_stats.avg_block_size;
                log.std_block_unknowns = linear_result.block_stats.std_block_size;
                log.diagonal_block_nnz_ratio = linear_result.block_stats.diagonal_block_nnz_ratio;
                log.offblock_nnz_ratio = linear_result.block_stats.offblock_nnz_ratio;
                log.total_weighted_coupling = linear_result.block_stats.total_weighted_coupling;
                log.diagonal_weighted_coupling_ratio =
                    linear_result.block_stats.diagonal_weighted_coupling_ratio;
                log.offblock_weighted_coupling_ratio =
                    linear_result.block_stats.offblock_weighted_coupling_ratio;
                log.j11_diagonal_weighted_ratio =
                    linear_result.block_stats.j11_diagonal_weighted_ratio;
                log.j12_diagonal_weighted_ratio =
                    linear_result.block_stats.j12_diagonal_weighted_ratio;
                log.j21_diagonal_weighted_ratio =
                    linear_result.block_stats.j21_diagonal_weighted_ratio;
                log.j22_diagonal_weighted_ratio =
                    linear_result.block_stats.j22_diagonal_weighted_ratio;
                log.theta_vmag_split_count = linear_result.block_stats.theta_vmag_split_count;
                log.pq_split_count = linear_result.block_stats.pq_split_count;
                log.stop_reason = linear_result.stop_reason;
                if (use_field_correction) {
                    apply_field_correction(ctx, options, j11_cache, j22_cache, log);
                    timed_with_sync([&] {
                        log.linear_abs_res =
                            compute_linear_residual(ctx, cublas, current_norms.two, log.linear_rel_res);
                    });
                    log.scaled_linear_abs_res = log.linear_abs_res;
                    log.scaled_linear_rel_res = log.linear_rel_res;
                    log.unscaled_linear_abs_res = log.linear_abs_res;
                    log.unscaled_linear_rel_res = log.linear_rel_res;
                } else if (use_device_field_correction) {
                    apply_device_field_correction(ctx,
                                                  options,
                                                  cusparse,
                                                  cublas,
                                                  device_field_cache,
                                                  log);
                    timed_with_sync([&] {
                        log.linear_abs_res =
                            compute_linear_residual(ctx, cublas, current_norms.two, log.linear_rel_res);
                    });
                    log.scaled_linear_abs_res = log.linear_abs_res;
                    log.scaled_linear_rel_res = log.linear_rel_res;
                    log.unscaled_linear_abs_res = log.linear_abs_res;
                    log.unscaled_linear_rel_res = log.linear_rel_res;
                } else if (use_bpbpp_refinement) {
                    apply_bpbpp_residual_refinement(ctx, cublas, fdlf_cache, log);
                    timed_with_sync([&] {
                        log.linear_abs_res =
                            compute_linear_residual(ctx, cublas, current_norms.two, log.linear_rel_res);
                    });
                    log.scaled_linear_abs_res = log.linear_abs_res;
                    log.scaled_linear_rel_res = log.linear_rel_res;
                    log.unscaled_linear_abs_res = log.linear_abs_res;
                    log.unscaled_linear_rel_res = log.linear_rel_res;
                }
            }

            const bool host_post_correction_used =
                log.global_correction_used || log.field_gain_accepted || log.theta_corr_accepted;
            if (host_post_correction_used &&
                options.global_correction_acceptance == "residual" &&
                !global_p_gmres_host.empty() &&
                !global_p_selected_host.empty()) {
                MismatchNorms gmres_trial{};
                MismatchNorms corrected_trial{};
                const double compare_seconds = timed_with_sync([&] {
                    ctx.restore_state_and_rhs();
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                 global_p_gmres_host.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyHostToDevice));
                    apply_voltage_update(ctx);
                    gmres_trial = compute_mismatch(ctx, cublas);

                    ctx.restore_state_and_rhs();
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                 global_p_selected_host.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyHostToDevice));
                    apply_voltage_update(ctx);
                    corrected_trial = compute_mismatch(ctx, cublas);

                    ctx.restore_state_and_rhs();
                });
                log.mismatch_recompute_seconds += compare_seconds;
                log.nonlinear_res_iter_trial = gmres_trial.inf;
                log.nonlinear_res_gain_trial = corrected_trial.inf;
                log.nonlinear_res_after_theta_corr = corrected_trial.inf;
                if (!std::isfinite(corrected_trial.inf) ||
                    corrected_trial.inf > gmres_trial.inf) {
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                 global_p_gmres_host.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyHostToDevice));
                    log.global_correction_used = false;
                    if (log.global_correction_attempted) {
                        log.global_correction_skipped_reason = "nonlinear_residual_worse";
                    }
                    if (log.field_gain_accepted) {
                        log.field_gain_accepted = false;
                        log.field_gain_skipped_reason = "nonlinear_residual_worse";
                    }
                    if (log.theta_corr_accepted) {
                        log.theta_corr_accepted = false;
                        log.theta_corr_skipped_reason = "nonlinear_residual_worse";
                    }
                    if (result.corrections_accepted > 0) {
                        --result.corrections_accepted;
                    }
                    ++result.corrections_skipped;
                } else {
                    CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                                 global_p_selected_host.data(),
                                                 static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                                 cudaMemcpyHostToDevice));
                }
            }

            if (use_global_post_correction &&
                options.global_basis_source == "diagnostic" &&
                options.global_diagnostic_full_step &&
                global_basis_candidate_available) {
                std::vector<double> p_full;
                run_diagnostic_full_solve(log, p_full);
                std::vector<double> candidate = p_full;
                for (std::size_t i = 0; i < candidate.size(); ++i) {
                    candidate[i] -= global_p_gmres_host[i];
                }
                log.global_diagnostic_basis_added =
                    global_basis.add(std::move(candidate), p_full);
                log.global_basis_rank_after = global_basis.rank();
                const double full_norm = std::max(host_norm2(p_full), 1.0e-300);
                const double gmres_norm = std::max(host_norm2(global_p_gmres_host), 1.0e-300);
                const std::vector<double>& direction_for_corr =
                    log.global_correction_used && !global_p_selected_host.empty()
                        ? global_p_selected_host
                        : global_p_gmres_host;
                const double corr_norm = std::max(host_norm2(direction_for_corr), 1.0e-300);
                log.cos_theta_gmres =
                    std::abs(host_dot(p_full, global_p_gmres_host)) / (full_norm * gmres_norm);
                log.cos_theta_corr =
                    std::abs(host_dot(p_full, direction_for_corr)) / (full_norm * corr_norm);
                log.norm_ratio_gmres = gmres_norm / full_norm;
                log.norm_ratio_corr = corr_norm / full_norm;
                const std::vector<double>& restore_step =
                    log.global_correction_used && !global_p_selected_host.empty()
                        ? global_p_selected_host
                        : global_p_gmres_host;
                CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_dx.data(),
                                             restore_step.data(),
                                             static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                             cudaMemcpyHostToDevice));
            }

            const bool fast_stale_gmres1_always_accept =
                options.middle_solver == "stale_GMRES1" &&
                options.skip_middle_backup &&
                options.dx_safety_check == "off" &&
                !options.accept_iterative_by_mismatch &&
                !options.enable_cudss_fallback &&
                options.fallback_policy == "off" &&
                !options.enable_shadow_dx_diagnostic &&
                !options.enable_damped_iterative_step &&
                !options.enable_scaled_mr1_step &&
                options.global_correction == "none";
            const bool bad_dx = fast_stale_gmres1_always_accept
                                    ? false
                                    : dx_is_bad(ctx, cublas, current_norms.inf, options.dx_safety_check);
            const bool gmres_breakdown =
                log.stop_reason.find("breakdown") != std::string::npos ||
                log.stop_reason.find("nan") != std::string::npos ||
                log.stop_reason.find("inf") != std::string::npos ||
                log.block_ilu_failed;
            const bool use_scaled_mr1 =
                options.enable_scaled_mr1_step &&
                options.middle_solver == "mr1_block_jacobi_coarse";

            if (fast_stale_gmres1_always_accept && !gmres_breakdown) {
                log.voltage_update_seconds = timed_with_sync([&] {
                    apply_voltage_update(ctx);
                });
                log.mismatch_recompute_seconds = timed_with_sync([&] {
                    current_norms = compute_mismatch(ctx, cublas);
                });
                log.mismatch_inf_after = current_norms.inf;
                log.mismatch_2_after = current_norms.two;
                diagnostic_seconds_total += dump_iteration_f(options, ctx, data.case_name, iter, "after");
                log.step_accepted = true;
                ++result.accepted_gmres_steps;
                consecutive_iterative_failures = 0;
            } else if (use_scaled_mr1) {
                MismatchNorms best_norms{};
                best_norms.inf = std::numeric_limits<double>::infinity();
                best_norms.two = std::numeric_limits<double>::infinity();
                double best_gamma = options.scaled_mr1_gamma_candidates.front();

                if (!bad_dx && !gmres_breakdown) {
                    for (double gamma : options.scaled_mr1_gamma_candidates) {
                        MismatchNorms trial_norms{};
                        const double trial_seconds = timed_with_sync([&] {
                            ctx.restore_state_and_rhs();
                            apply_voltage_update(ctx, gamma);
                            trial_norms = compute_mismatch(ctx, cublas);
                        });
                        log.extra_mismatch_eval_seconds += trial_seconds;
                        log.mismatch_recompute_seconds += trial_seconds;

                        const double ratio_inf =
                            ratio_to_before(trial_norms.inf, log.mismatch_inf_before);
                        record_scaled_gamma_ratio(log, gamma, ratio_inf);
                        if (std::isfinite(trial_norms.inf) && trial_norms.inf < best_norms.inf) {
                            best_norms = trial_norms;
                            best_gamma = gamma;
                        }
                    }
                    log.chosen_gamma = best_gamma;
                    log.scaled_total_middle_seconds =
                        log.gmres_trial_setup_seconds +
                        log.gmres_trial_solve_seconds +
                        log.extra_mismatch_eval_seconds;
                    if (std::isfinite(best_norms.inf)) {
                        log.voltage_update_seconds += timed_with_sync([&] {
                            ctx.restore_state_and_rhs();
                            apply_voltage_update(ctx, best_gamma);
                        });
                        current_norms = best_norms;
                        log.mismatch_inf_after = best_norms.inf;
                        log.mismatch_2_after = best_norms.two;
                        diagnostic_seconds_total += dump_iteration_f(options,
                                                                     ctx,
                                                                     data.case_name,
                                                                     iter,
                                                                     "after");
                    }
                }

                const bool scaled_accepted =
                    !bad_dx && !gmres_breakdown && std::isfinite(best_norms.inf) &&
                    best_norms.inf < options.accept_mismatch_ratio * log.mismatch_inf_before;
                if (scaled_accepted) {
                    log.step_accepted = true;
                    log.stop_reason += ":scaled_gamma_" + std::to_string(best_gamma);
                    ++result.accepted_gmres_steps;
                    consecutive_iterative_failures = 0;
                } else {
                    ++result.rejected_gmres_steps;
                    ++consecutive_iterative_failures;
                    const bool fallback_allowed =
                        options.enable_cudss_fallback && options.fallback_policy != "off" &&
                        (options.fallback_policy == "immediate" ||
                         consecutive_iterative_failures >= 2);
                    if (!fallback_allowed && options.fallback_policy == "after_two_failures" &&
                        options.enable_cudss_fallback) {
                        timed_with_sync([&] {
                            ctx.restore_state_and_rhs();
                            current_norms.inf = log.mismatch_inf_before;
                            current_norms.two = log.mismatch_2_before;
                        });
                        log.step_accepted = false;
                        log.fallback_used = false;
                        log.stop_reason = "scaled_mr1_rejected_waiting_for_second_failure:" +
                                          log.stop_reason;
                    } else if (fallback_allowed) {
                        ++result.fallback_calls;
                        log.fallback_used = true;
                        log.step_accepted = true;
                        log.solver_used = "cudss_fallback";
                        log.stop_reason = "scaled_mr1_rejected:" + log.stop_reason;
                        consecutive_iterative_failures = 0;
                        bj_numeric_setup_valid = false;
                        bj_numeric_setup_uses = 0;

                        timed_with_sync([&] {
                            ctx.restore_state_and_rhs();
                        });
                        run_current_full_factor_solve(log);
                        if (use_global_post_correction &&
                            options.global_basis_source == "fallback" &&
                            global_basis_candidate_available) {
                            std::vector<double> p_full(static_cast<std::size_t>(ctx.dimF), 0.0);
                            ctx.d_dx.copy_to(p_full.data(), p_full.size());
                            std::vector<double> candidate = p_full;
                            for (std::size_t i = 0; i < candidate.size(); ++i) {
                                candidate[i] -= global_p_gmres_host[i];
                            }
                            log.global_required_basis_added =
                                global_basis.add(std::move(candidate), p_full);
                            log.global_basis_rank_after = global_basis.rank();
                        }
                        log.voltage_update_seconds += timed_with_sync([&] {
                            apply_voltage_update(ctx);
                        });
                        log.mismatch_recompute_seconds += timed_with_sync([&] {
                            current_norms = compute_mismatch(ctx, cublas);
                        });
                        log.mismatch_inf_after = current_norms.inf;
                        log.mismatch_2_after = current_norms.two;
                        diagnostic_seconds_total += dump_iteration_f(options,
                                                                     ctx,
                                                                     data.case_name,
                                                                     iter,
                                                                     "after");
                    } else {
                        timed_with_sync([&] {
                            ctx.restore_state_and_rhs();
                        });
                        log.step_accepted = false;
                        result.stop_reason =
                            bad_dx ? "scaled_mr1_bad_dx" : "scaled_mr1_step_rejected";
                        result.iteration_logs.push_back(log);
                        result.nr_iters = iter + 1;
                        break;
                    }
                }
            } else {
                log.voltage_update_seconds = timed_with_sync([&] {
                    apply_voltage_update(ctx);
                });
                log.mismatch_recompute_seconds = timed_with_sync([&] {
                    current_norms = compute_mismatch(ctx, cublas);
                });
                log.mismatch_inf_after = current_norms.inf;
                log.mismatch_2_after = current_norms.two;
                diagnostic_seconds_total += dump_iteration_f(options, ctx, data.case_name, iter, "after");

                auto mismatch_accept = [&]() {
                    return current_norms.inf < options.accept_mismatch_ratio * log.mismatch_inf_before ||
                           current_norms.two < options.accept_mismatch_ratio * log.mismatch_2_before;
                };
                auto mismatch_reject = [&]() {
                    return current_norms.inf > options.reject_mismatch_ratio * log.mismatch_inf_before ||
                           current_norms.two > options.reject_mismatch_ratio * log.mismatch_2_before;
                };

                if (!bad_dx && !gmres_breakdown &&
                    (!options.accept_iterative_by_mismatch || mismatch_accept()) && !mismatch_reject()) {
                    log.step_accepted = true;
                    ++result.accepted_gmres_steps;
                    consecutive_iterative_failures = 0;
                } else {
                    bool damped_accepted = false;
                    if (!bad_dx && !gmres_breakdown && options.enable_damped_iterative_step) {
                        for (double factor : options.damping_factors) {
                            if (factor >= 1.0) {
                                continue;
                            }
                            timed_with_sync([&] {
                                ctx.restore_state_and_rhs();
                                apply_voltage_update(ctx, factor);
                                current_norms = compute_mismatch(ctx, cublas);
                            });
                            log.mismatch_inf_after = current_norms.inf;
                            log.mismatch_2_after = current_norms.two;
                            if ((!options.accept_iterative_by_mismatch || mismatch_accept()) &&
                                !mismatch_reject()) {
                                damped_accepted = true;
                                log.step_accepted = true;
                                log.stop_reason += ":damped_factor_" + std::to_string(factor);
                                ++result.accepted_gmres_steps;
                                consecutive_iterative_failures = 0;
                                break;
                            }
                        }
                    }

                    if (!damped_accepted) {
                        ++result.rejected_gmres_steps;
                        ++consecutive_iterative_failures;
                        bool refresh_retry_accepted = false;
                        if (options.middle_solver == "stale_GMRES1_refresh" &&
                            !bad_dx && !gmres_breakdown) {
                            ++result.fallback_calls;
                            timed_with_sync([&] {
                                ctx.restore_state_and_rhs();
                                current_norms.inf = log.mismatch_inf_before;
                                current_norms.two = log.mismatch_2_before;
                            });
                            refresh_stale_factor_only(log);
                            run_stale_gmres1_candidate(log, current_norms);
                            log.voltage_update_seconds += timed_with_sync([&] {
                                apply_voltage_update(ctx);
                            });
                            log.mismatch_recompute_seconds += timed_with_sync([&] {
                                current_norms = compute_mismatch(ctx, cublas);
                            });
                            log.mismatch_inf_after = current_norms.inf;
                            log.mismatch_2_after = current_norms.two;
                            refresh_retry_accepted =
                                (!options.accept_iterative_by_mismatch || mismatch_accept()) &&
                                !mismatch_reject();
                            if (refresh_retry_accepted) {
                                log.step_accepted = true;
                                log.fallback_used = true;
                                log.stop_reason += ":refresh_factor_retry_accepted";
                                ++result.accepted_gmres_steps;
                                consecutive_iterative_failures = 0;
                            }
                        }
                        if (refresh_retry_accepted) {
                            // The retry replaced the rejected candidate and is now the accepted step.
                        } else {
                        const bool fallback_allowed =
                            options.enable_cudss_fallback && options.fallback_policy != "off" &&
                            (options.fallback_policy == "immediate" ||
                             consecutive_iterative_failures >= 2);
                        if (!fallback_allowed && options.fallback_policy == "after_two_failures" &&
                            options.enable_cudss_fallback) {
                            timed_with_sync([&] {
                                ctx.restore_state_and_rhs();
                                current_norms.inf = log.mismatch_inf_before;
                                current_norms.two = log.mismatch_2_before;
                            });
                            log.step_accepted = false;
                            log.fallback_used = false;
                            log.stop_reason = "gmres_rejected_waiting_for_second_failure:" + log.stop_reason;
                        } else if (fallback_allowed) {
                            ++result.fallback_calls;
                            log.fallback_used = true;
                            log.step_accepted = true;
                            log.solver_used = "cudss_fallback";
                            log.stop_reason = "gmres_rejected:" + log.stop_reason;
                            consecutive_iterative_failures = 0;
                            bj_numeric_setup_valid = false;
                            bj_numeric_setup_uses = 0;

                            timed_with_sync([&] {
                                ctx.restore_state_and_rhs();
                            });
                            run_current_full_factor_solve(log);
                            if (use_global_post_correction &&
                                options.global_basis_source == "fallback" &&
                                global_basis_candidate_available) {
                                std::vector<double> p_full(static_cast<std::size_t>(ctx.dimF), 0.0);
                                ctx.d_dx.copy_to(p_full.data(), p_full.size());
                                std::vector<double> candidate = p_full;
                                for (std::size_t i = 0; i < candidate.size(); ++i) {
                                    candidate[i] -= global_p_gmres_host[i];
                                }
                                log.global_required_basis_added =
                                    global_basis.add(std::move(candidate), p_full);
                                log.global_basis_rank_after = global_basis.rank();
                            }
                            log.voltage_update_seconds += timed_with_sync([&] {
                                apply_voltage_update(ctx);
                            });
                            log.mismatch_recompute_seconds += timed_with_sync([&] {
                                current_norms = compute_mismatch(ctx, cublas);
                            });
                            log.mismatch_inf_after = current_norms.inf;
                            log.mismatch_2_after = current_norms.two;
                        } else {
                            log.step_accepted = false;
                            result.stop_reason = bad_dx ? "gmres_bad_dx" : "gmres_step_rejected";
                            result.iteration_logs.push_back(log);
                            result.nr_iters = iter + 1;
                            break;
                        }
                        }
                    }
                }
            }
        }

        log.nr_iter_total_seconds =
            std::max(0.0, elapsed_seconds(iter_start) - log.shadow_dx_diagnostic_seconds);
        const double linear_accounted_seconds =
            log.middle_solver_total_seconds > 0.0
                ? log.middle_solver_total_seconds +
                      (log.fallback_used ? log.fallback_cudss_setup_seconds +
                                               log.fallback_cudss_solve_seconds
                                         : 0.0)
                : log.linear_setup_seconds + log.linear_solve_seconds;
        const double accounted_seconds =
            log.jacobian_seconds + linear_accounted_seconds +
            log.voltage_update_seconds + log.mismatch_recompute_seconds;
        log.unaccounted_seconds = std::max(0.0, log.nr_iter_total_seconds - accounted_seconds);
        update_accepted_linear_stats(
            log, accepted_max_rel, accepted_sum_rel, accepted_sum_ratio, accepted_count);
        if (log.step_accepted) {
            if (use_stale_refinement &&
                log.solver_used != "cudss_fallback" &&
                log.solver_used.rfind("cudss_", 0) != 0) {
                ++stale_factor_age;
            }
            CUITER_CUDA_CHECK(cudaMemcpy(ctx.d_prev_dx.data(),
                                         ctx.d_dx.data(),
                                         static_cast<std::size_t>(ctx.dimF) * sizeof(double),
                                         cudaMemcpyDeviceToDevice));
            has_previous_dx = true;
        }
        result.iteration_logs.push_back(log);
        result.nr_iters = iter + 1;
        result.final_mismatch_inf = current_norms.inf;
        result.final_mismatch_2 = current_norms.two;

        if (current_norms.inf <= options.nr_mismatch_inf_tol ||
            (options.nr_mismatch_2_tol > 0.0 && current_norms.two <= options.nr_mismatch_2_tol)) {
            result.converged = true;
            result.stop_reason = "nr_converged";
            break;
        }
        if (consecutive_iterative_failures > options.max_consecutive_iterative_failures) {
            result.stop_reason = "too_many_iterative_failures";
            if (!options.enable_cudss_fallback) {
                break;
            }
        }
    }

    if (result.stop_reason.empty()) {
        result.stop_reason = result.converged ? "nr_converged" : "max_nr_iters";
    }
    result.shadow_dx_diagnostic_seconds = diagnostic_seconds_total;
    result.total_seconds = std::max(0.0, elapsed_seconds(total_start) - diagnostic_seconds_total);
    result.max_linear_rel_res_accepted = accepted_max_rel;
    result.avg_linear_rel_res_accepted =
        accepted_count > 0 ? accepted_sum_rel / static_cast<double>(accepted_count) : 0.0;
    result.accepted_gmres_mismatch_reduction_ratio_mean =
        accepted_count > 0 ? accepted_sum_ratio / static_cast<double>(accepted_count) : 0.0;
    result.final_basis_rank = global_basis.rank();
    if (pure_cudss_total_seconds > 0.0 && result.total_seconds > 0.0) {
        result.speedup_vs_pure_cudss = pure_cudss_total_seconds / result.total_seconds;
    }

    if (use_field_correction) {
        j11_cache.set_stream(nullptr);
        j22_cache.set_stream(nullptr);
        CUITER_CUDA_CHECK(cudaStreamDestroy(j11_stream));
        CUITER_CUDA_CHECK(cudaStreamDestroy(j22_stream));
    }
    if (use_device_field_correction) {
        device_field_cache.destroy();
    }
    if (use_fdlf_bpbpp || use_bpbpp_refinement) {
        fdlf_cache.destroy();
    }
    CUPF_MINIMAL_CUSPARSE_CHECK(cusparseDestroy(cusparse));
    cublasDestroy(cublas);
    return result;
}

}  // namespace cupf_minimal
