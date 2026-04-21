#pragma once

#include "newton_solver/core/contexts.hpp"
#include "utils/dump.hpp"

#ifdef CUPF_ENABLE_CUDSS
  #include "utils/cuda_utils.hpp"

  #include <cstring>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace newton_solver::linear_diagnostics {

struct LinearResidualStats {
    double norm_f_inf = 0.0;
    double dx_inf = 0.0;
    double linear_residual_inf = 0.0;
    double linear_relres = 0.0;
};

inline std::string join_int64_values(const std::vector<int64_t>& values)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ':';
        }
        out << values[i];
    }
    return out.str();
}

template <typename IndexType, typename ValueType, typename DxType>
LinearResidualStats compute_and_dump_linear_residual(
    const std::vector<IndexType>& row_ptr,
    const std::vector<IndexType>& col_idx,
    const std::vector<ValueType>& jacobian_values,
    const std::vector<double>& f,
    const std::vector<DxType>& dx,
    int32_t iteration)
{
    const int32_t dim = static_cast<int32_t>(f.size());
    if (dim < 0 ||
        row_ptr.size() != static_cast<std::size_t>(dim + 1) ||
        dx.size() != f.size() ||
        col_idx.size() != jacobian_values.size()) {
        throw std::runtime_error("linear diagnostics: inconsistent linear system dimensions");
    }

    std::vector<double> residual(static_cast<std::size_t>(dim), 0.0);
    LinearResidualStats stats;
    for (int32_t row = 0; row < dim; ++row) {
        double value = -f[static_cast<std::size_t>(row)];
        for (IndexType pos = row_ptr[static_cast<std::size_t>(row)];
             pos < row_ptr[static_cast<std::size_t>(row + 1)];
             ++pos) {
            const int32_t col = static_cast<int32_t>(col_idx[static_cast<std::size_t>(pos)]);
            value += static_cast<double>(jacobian_values[static_cast<std::size_t>(pos)]) *
                     static_cast<double>(dx[static_cast<std::size_t>(col)]);
        }
        residual[static_cast<std::size_t>(row)] = value;
        stats.linear_residual_inf = std::max(stats.linear_residual_inf, std::abs(value));
    }

    for (double value : f) {
        stats.norm_f_inf = std::max(stats.norm_f_inf, std::abs(value));
    }
    for (DxType value : dx) {
        stats.dx_inf = std::max(stats.dx_inf, std::abs(static_cast<double>(value)));
    }
    stats.linear_relres = stats.linear_residual_inf / std::max(stats.norm_f_inf, 1.0e-300);

    newton_solver::utils::dumpVector("linear_residual", iteration, residual);
    return stats;
}

inline void append_linear_trace(const IterationContext& ctx,
                                const char* backend,
                                const char* precision,
                                const char* factorization_phase,
                                int64_t dim,
                                int64_t nnz,
                                const LinearResidualStats& stats,
                                const std::vector<int64_t>& npivots)
{
    namespace dump = newton_solver::utils;

    if (!dump::isDumpEnabled()) {
        return;
    }

    const std::filesystem::path path =
        std::filesystem::path(dump::getDumpDirectory()) / "linear_diagnostics.csv";
    std::filesystem::create_directories(path.parent_path());

    const bool write_header = !std::filesystem::exists(path) ||
                              std::filesystem::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open linear diagnostics file: " + path.string());
    }

    if (write_header) {
        out << "iter,backend,precision,factorization_phase,dim,nnz,"
            << "jacobian_updated,jacobian_age,"
            << "norm_f_inf,dx_inf,linear_residual_inf,linear_relres,"
            << "npivots_total,npivots_values\n";
    }

    const int64_t npivots_total = std::accumulate(npivots.begin(), npivots.end(), int64_t{0});

    out << ctx.iter << ','
        << backend << ','
        << precision << ','
        << factorization_phase << ','
        << dim << ','
        << nnz << ','
        << (ctx.jacobian_updated_this_iter ? 1 : 0) << ','
        << ctx.jacobian_age << ','
        << std::setprecision(17)
        << stats.norm_f_inf << ','
        << stats.dx_inf << ','
        << stats.linear_residual_inf << ','
        << stats.linear_relres << ','
        << npivots_total << ','
        << join_int64_values(npivots)
        << '\n';
}

#ifdef CUPF_ENABLE_CUDSS
inline std::vector<int64_t> try_get_cudss_int_values(cudssHandle_t handle,
                                                     cudssData_t data,
                                                     cudssDataParam_t param)
{
    try {
        size_t needed = 0;
        CUDSS_CHECK(cudssDataGet(handle, data, param, nullptr, 0, &needed));
        if (needed == 0) {
            return {};
        }

        std::vector<unsigned char> raw(needed);
        size_t written = 0;
        CUDSS_CHECK(cudssDataGet(handle, data, param, raw.data(), raw.size(), &written));
        if (written != needed) {
            return {};
        }

        if (needed % sizeof(int64_t) == 0) {
            std::vector<int64_t> values(needed / sizeof(int64_t));
            std::memcpy(values.data(), raw.data(), needed);
            return values;
        }
        if (needed % sizeof(int32_t) == 0) {
            const std::size_t count = needed / sizeof(int32_t);
            std::vector<int64_t> values(count);
            const auto* src = reinterpret_cast<const int32_t*>(raw.data());
            for (std::size_t i = 0; i < count; ++i) {
                values[i] = src[i];
            }
            return values;
        }
    } catch (const std::exception&) {
        return {};
    }
    return {};
}
#endif

template <typename IndexType, typename ValueType, typename DxType>
void dump_linear_system(const IterationContext& ctx,
                        const char* backend,
                        const char* precision,
                        const char* factorization_phase,
                        const std::vector<IndexType>& row_ptr,
                        const std::vector<IndexType>& col_idx,
                        const std::vector<ValueType>& jacobian_values,
                        const std::vector<double>& f,
                        const std::vector<DxType>& dx,
                        const std::vector<int64_t>& npivots = {})
{
    namespace dump = newton_solver::utils;

    if (!dump::isDumpEnabled()) {
        return;
    }

    const int32_t dim = static_cast<int32_t>(f.size());
    if (ctx.iter == 0) {
        dump::dumpVector("jacobian_row_ptr", 0, row_ptr);
        dump::dumpVector("jacobian_col_idx", 0, col_idx);
    }
    dump::dumpVector("residual_solve", ctx.iter, f);
    dump::dumpVector("jacobian_values_used", ctx.iter, jacobian_values);
    dump::dumpVector("dx", ctx.iter, dx);

    const LinearResidualStats stats =
        compute_and_dump_linear_residual(row_ptr, col_idx, jacobian_values, f, dx, ctx.iter);
    append_linear_trace(
        ctx,
        backend,
        precision,
        factorization_phase,
        dim,
        static_cast<int64_t>(jacobian_values.size()),
        stats,
        npivots);
}

}  // namespace newton_solver::linear_diagnostics
