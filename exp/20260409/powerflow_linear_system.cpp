#include "powerflow_linear_system.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace exp_20260409 {
namespace {

// This file turns one shared cuPF case into the fixed linear system used by
// the cuDSS benchmark:
//
//   J = dF/dx evaluated at the initial state V0
//   b = -F(V0)
//
// The goal is to benchmark sparse linear algebra only, not the full Newton loop.
using Complex = std::complex<double>;

// Compute Ibus = Ybus * V0 once. The benchmark fixes the linear system at V0,
// so every later quantity (mismatch, Jacobian diagonals) reuses this vector.
std::vector<std::complex<double>> compute_ibus(const CupfDatasetCase& case_data)
{
    std::vector<Complex> ibus(case_data.rows, Complex(0.0, 0.0));

    for (int32_t row = 0; row < case_data.rows; ++row) {
        for (int32_t k = case_data.indptr[row]; k < case_data.indptr[row + 1]; ++k) {
            ibus[row] += case_data.ybus_data[k] * case_data.v0[case_data.indices[k]];
        }
    }

    return ibus;
}

// V / |V| appears in the dF/d|V| blocks. Clamp the denominator so pathological
// inputs cannot produce NaNs during benchmark setup.
std::vector<Complex> compute_voltage_direction(const CupfDatasetCase& case_data)
{
    std::vector<Complex> vnorm(case_data.rows);

    for (int32_t bus = 0; bus < case_data.rows; ++bus) {
        const double vm = std::max(std::abs(case_data.v0[bus]), 1e-8);
        vnorm[bus] = case_data.v0[bus] / vm;
    }

    return vnorm;
}

Complex bus_mismatch(const CupfDatasetCase& case_data,
                     const std::vector<Complex>& ibus,
                     int32_t bus)
{
    return case_data.v0[bus] * std::conj(ibus[bus]) - case_data.sbus[bus];
}

// Fill the off-diagonal Jacobian contributions in the same Ybus CSR order used
// by JacobianBuilder::analyze(). The map arrays convert each Ybus entry into the
// correct position inside the final CSR value buffer.
void fill_jacobian_off_diagonal_values(const CupfDatasetCase& case_data,
                                       const std::vector<Complex>& vnorm,
                                       const JacobianMaps& maps,
                                       std::vector<double>& jacobian_values)
{
    static constexpr Complex j(0.0, 1.0);

    int32_t ybus_entry_idx = 0;
    for (int32_t row_bus = 0; row_bus < case_data.rows; ++row_bus) {
        for (int32_t k = case_data.indptr[row_bus];
             k < case_data.indptr[row_bus + 1];
             ++k, ++ybus_entry_idx) {
            const int32_t col_bus = case_data.indices[k];
            const Complex y_ij = case_data.ybus_data[k];

            const Complex d_angle =
                -j * case_data.v0[row_bus] * std::conj(y_ij * case_data.v0[col_bus]);
            const Complex d_magnitude =
                case_data.v0[row_bus] * std::conj(y_ij * vnorm[col_bus]);

            const int32_t j11_pos = maps.mapJ11[ybus_entry_idx];
            const int32_t j21_pos = maps.mapJ21[ybus_entry_idx];
            const int32_t j12_pos = maps.mapJ12[ybus_entry_idx];
            const int32_t j22_pos = maps.mapJ22[ybus_entry_idx];

            if (j11_pos >= 0) jacobian_values[j11_pos] = d_angle.real();
            if (j21_pos >= 0) jacobian_values[j21_pos] = d_angle.imag();
            if (j12_pos >= 0) jacobian_values[j12_pos] = d_magnitude.real();
            if (j22_pos >= 0) jacobian_values[j22_pos] = d_magnitude.imag();
        }
    }
}

// Add the diagonal Jacobian terms, which depend on Ibus at the current state.
void accumulate_jacobian_diagonal_values(const CupfDatasetCase& case_data,
                                         const std::vector<Complex>& ibus,
                                         const std::vector<Complex>& vnorm,
                                         const JacobianMaps& maps,
                                         std::vector<double>& jacobian_values)
{
    static constexpr Complex j(0.0, 1.0);

    for (int32_t bus = 0; bus < case_data.rows; ++bus) {
        const Complex d_angle_diag = j * (case_data.v0[bus] * std::conj(ibus[bus]));
        const Complex d_magnitude_diag = std::conj(ibus[bus]) * vnorm[bus];

        const int32_t j11_pos = maps.diagJ11[bus];
        const int32_t j21_pos = maps.diagJ21[bus];
        const int32_t j12_pos = maps.diagJ12[bus];
        const int32_t j22_pos = maps.diagJ22[bus];

        if (j11_pos >= 0) jacobian_values[j11_pos] += d_angle_diag.real();
        if (j21_pos >= 0) jacobian_values[j21_pos] += d_angle_diag.imag();
        if (j12_pos >= 0) jacobian_values[j12_pos] += d_magnitude_diag.real();
        if (j22_pos >= 0) jacobian_values[j22_pos] += d_magnitude_diag.imag();
    }
}

// Pack the Newton mismatch vector in the benchmark's linear-system layout:
//   [ Re(mis[pv]), Re(mis[pq]), Im(mis[pq]) ]
std::vector<double> build_newton_mismatch_vector(const CupfDatasetCase& case_data,
                                                 const std::vector<Complex>& ibus)
{
    const int32_t dim = static_cast<int32_t>(case_data.pv.size() + 2 * case_data.pq.size());
    std::vector<double> mismatch(dim, 0.0);

    int32_t out = 0;
    for (int32_t bus : case_data.pv) {
        mismatch[out++] = bus_mismatch(case_data, ibus, bus).real();
    }
    for (int32_t bus : case_data.pq) {
        mismatch[out++] = bus_mismatch(case_data, ibus, bus).real();
    }
    for (int32_t bus : case_data.pq) {
        mismatch[out++] = bus_mismatch(case_data, ibus, bus).imag();
    }

    return mismatch;
}

}  // namespace

PowerFlowLinearSystem build_linear_system(const CupfDatasetCase& case_data,
                                          JacobianBuilderType jacobian_type)
{
    if (case_data.rows <= 0) {
        throw std::runtime_error("Cannot build linear system from an empty case");
    }

    JacobianBuilder builder(jacobian_type);
    auto analysis = builder.analyze(case_data.ybus(),
                                    case_data.pv.data(), static_cast<int32_t>(case_data.pv.size()),
                                    case_data.pq.data(), static_cast<int32_t>(case_data.pq.size()));

    const auto& maps = analysis.maps;

    PowerFlowLinearSystem system;
    system.jacobian_type = jacobian_type;
    system.structure = std::move(analysis.J);
    system.values.assign(system.structure.nnz, 0.0);

    const std::vector<Complex> ibus = compute_ibus(case_data);
    const std::vector<Complex> vnorm = compute_voltage_direction(case_data);

    fill_jacobian_off_diagonal_values(case_data, vnorm, maps, system.values);
    accumulate_jacobian_diagonal_values(case_data, ibus, vnorm, maps, system.values);

    const std::vector<double> mismatch = build_newton_mismatch_vector(case_data, ibus);

    system.mismatch_inf = 0.0;
    system.rhs.resize(mismatch.size());
    for (size_t i = 0; i < mismatch.size(); ++i) {
        system.mismatch_inf = std::max(system.mismatch_inf, std::abs(mismatch[i]));
        system.rhs[i] = -mismatch[i];
    }

    return system;
}

double rhs_inf_norm(const std::vector<double>& rhs)
{
    double inf_norm = 0.0;
    for (double value : rhs) {
        inf_norm = std::max(inf_norm, std::abs(value));
    }
    return inf_norm;
}

double residual_inf_norm(const JacobianStructure& structure,
                         const std::vector<double>& values,
                         const std::vector<double>& rhs,
                         const std::vector<float>& x)
{
    if (values.size() != static_cast<size_t>(structure.nnz)) {
        throw std::runtime_error("Jacobian values size does not match Jacobian structure");
    }
    if (rhs.size() != static_cast<size_t>(structure.dim)) {
        throw std::runtime_error("RHS size does not match Jacobian dimension");
    }
    if (x.size() != static_cast<size_t>(structure.dim)) {
        throw std::runtime_error("Solution size does not match Jacobian dimension");
    }

    double inf_norm = 0.0;
    for (int32_t row = 0; row < structure.dim; ++row) {
        double sum = 0.0;
        for (int32_t k = structure.row_ptr[row]; k < structure.row_ptr[row + 1]; ++k) {
            sum += values[k] * static_cast<double>(x[structure.col_idx[k]]);
        }
        inf_norm = std::max(inf_norm, std::abs(sum - rhs[row]));
    }
    return inf_norm;
}

double solution_inf_norm(const std::vector<float>& x)
{
    double inf_norm = 0.0;
    for (float value : x) {
        inf_norm = std::max(inf_norm, std::abs(static_cast<double>(value)));
    }
    return inf_norm;
}

}  // namespace exp_20260409
