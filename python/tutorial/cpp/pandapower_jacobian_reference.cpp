// Tutorial-only reference for a pandapower/PYPOWER-style Newton-Raphson solve.
//
// This file is intentionally outside cuPF.  It shows the full baseline loop:
//   1. evaluate S_calc(V) = V * conj(Ybus * V),
//   2. build the reduced mismatch [P(PV,PQ), Q(PQ)],
//   3. assemble a pandapower-style reduced Jacobian from dS_dVa/dS_dVm,
//   4. solve J dx = mismatch,
//   5. update Va(PV,PQ) and Vm(PQ).
//
// cuPF does not use this path in production.  cuPF precomputes the reduced
// Jacobian sparsity pattern and writes numeric values directly into that fixed
// pattern every Newton iteration.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
constexpr Complex j{0.0, 1.0};

struct CsrComplex {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<Complex> data;
};

struct CsrReal {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<double> data;
};

struct Triplet {
    int32_t row = 0;
    int32_t col = 0;
    double value = 0.0;
};

std::vector<int32_t> concatenate(const std::vector<int32_t>& a, const std::vector<int32_t>& b)
{
    std::vector<int32_t> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

std::vector<Complex> compute_ibus(const CsrComplex& ybus, const std::vector<Complex>& v)
{
    std::vector<Complex> ibus(ybus.rows, Complex{0.0, 0.0});
    for (int32_t row = 0; row < ybus.rows; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            ibus[row] += ybus.data[k] * v[ybus.indices[k]];
        }
    }
    return ibus;
}

std::vector<Complex> compute_power(const CsrComplex& ybus, const std::vector<Complex>& v)
{
    const std::vector<Complex> ibus = compute_ibus(ybus, v);
    std::vector<Complex> s_calc(v.size());
    for (int32_t bus = 0; bus < ybus.rows; ++bus) {
        s_calc[bus] = v[bus] * std::conj(ibus[bus]);
    }
    return s_calc;
}

std::vector<Complex> voltage_norm(const std::vector<Complex>& v)
{
    std::vector<Complex> out(v.size());
    for (int32_t i = 0; i < static_cast<int32_t>(v.size()); ++i) {
        const double mag = std::abs(v[i]);
        if (mag == 0.0) {
            throw std::runtime_error("zero voltage magnitude");
        }
        out[i] = v[i] / mag;
    }
    return out;
}

void add_triplet(std::map<std::pair<int32_t, int32_t>, double>& values,
                 int32_t row,
                 int32_t col,
                 double value)
{
    values[{row, col}] += value;
}

CsrReal compress_triplets(int32_t rows, int32_t cols, const std::vector<Triplet>& trips)
{
    std::map<std::pair<int32_t, int32_t>, double> values;
    for (const Triplet& trip : trips) {
        add_triplet(values, trip.row, trip.col, trip.value);
    }

    CsrReal out;
    out.rows = rows;
    out.cols = cols;
    out.indptr.assign(rows + 1, 0);
    for (const auto& [rc, value] : values) {
        (void)value;
        ++out.indptr[rc.first + 1];
    }
    for (int32_t row = 0; row < rows; ++row) {
        out.indptr[row + 1] += out.indptr[row];
    }
    out.indices.reserve(values.size());
    out.data.reserve(values.size());
    for (const auto& [rc, value] : values) {
        out.indices.push_back(rc.second);
        out.data.push_back(value);
    }
    return out;
}

CsrReal build_pandapower_style_jacobian(
    const CsrComplex& ybus,
    const std::vector<Complex>& v,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    const int32_t n_bus = ybus.rows;
    const std::vector<int32_t> pvpq = concatenate(pv, pq);
    const int32_t n_pvpq = static_cast<int32_t>(pvpq.size());
    const int32_t n_pq = static_cast<int32_t>(pq.size());
    const int32_t dim = n_pvpq + n_pq;

    std::vector<int32_t> pvpq_pos(n_bus, -1);
    std::vector<int32_t> pq_pos(n_bus, -1);
    for (int32_t i = 0; i < n_pvpq; ++i) {
        pvpq_pos[pvpq[i]] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        pq_pos[pq[i]] = i;
    }

    const std::vector<Complex> ibus = compute_ibus(ybus, v);
    const std::vector<Complex> vnorm = voltage_norm(v);
    std::vector<Triplet> trips;
    trips.reserve(4 * ybus.data.size());

    auto scatter_blocks = [&](int32_t bus_i, int32_t bus_j, Complex dS_dVa, Complex dS_dVm) {
        const int32_t ri_p = pvpq_pos[bus_i];
        const int32_t ri_q = pq_pos[bus_i];
        const int32_t cj_p = pvpq_pos[bus_j];
        const int32_t cj_q = pq_pos[bus_j];

        if (ri_p >= 0 && cj_p >= 0) trips.push_back({ri_p, cj_p, dS_dVa.real()});
        if (ri_p >= 0 && cj_q >= 0) trips.push_back({ri_p, n_pvpq + cj_q, dS_dVm.real()});
        if (ri_q >= 0 && cj_p >= 0) trips.push_back({n_pvpq + ri_q, cj_p, dS_dVa.imag()});
        if (ri_q >= 0 && cj_q >= 0) trips.push_back({n_pvpq + ri_q, n_pvpq + cj_q, dS_dVm.imag()});
    };

    for (int32_t row = 0; row < ybus.rows; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            const int32_t col = ybus.indices[k];
            const Complex y = ybus.data[k];
            const Complex dS_dVa = -j * v[row] * std::conj(y * v[col]);
            const Complex dS_dVm = v[row] * std::conj(y * vnorm[col]);
            scatter_blocks(row, col, dS_dVa, dS_dVm);
        }
    }

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const Complex dS_dVa = j * v[bus] * std::conj(ibus[bus]);
        const Complex dS_dVm = std::conj(ibus[bus]) * vnorm[bus];
        scatter_blocks(bus, bus, dS_dVa, dS_dVm);
    }

    return compress_triplets(dim, dim, trips);
}

std::vector<double> build_reduced_mismatch(
    const std::vector<Complex>& s_spec,
    const std::vector<Complex>& s_calc,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    const std::vector<int32_t> pvpq = concatenate(pv, pq);
    std::vector<double> mismatch;
    mismatch.reserve(pvpq.size() + pq.size());

    for (int32_t bus : pvpq) {
        mismatch.push_back((s_spec[bus] - s_calc[bus]).real());
    }
    for (int32_t bus : pq) {
        mismatch.push_back((s_spec[bus] - s_calc[bus]).imag());
    }
    return mismatch;
}

double max_abs(const std::vector<double>& values)
{
    double out = 0.0;
    for (double value : values) {
        out = std::max(out, std::abs(value));
    }
    return out;
}

std::vector<std::vector<double>> to_dense(const CsrReal& csr)
{
    std::vector<std::vector<double>> dense(
        csr.rows, std::vector<double>(csr.cols, 0.0));
    for (int32_t row = 0; row < csr.rows; ++row) {
        for (int32_t k = csr.indptr[row]; k < csr.indptr[row + 1]; ++k) {
            dense[row][csr.indices[k]] += csr.data[k];
        }
    }
    return dense;
}

std::vector<double> solve_dense_linear_system(CsrReal csr, std::vector<double> rhs)
{
    std::vector<std::vector<double>> a = to_dense(csr);
    const int32_t n = csr.rows;

    for (int32_t col = 0; col < n; ++col) {
        int32_t pivot = col;
        for (int32_t row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(a[pivot][col]) < 1e-14) {
            throw std::runtime_error("singular tutorial Jacobian");
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(rhs[pivot], rhs[col]);
        }

        const double diag = a[col][col];
        for (int32_t k = col; k < n; ++k) {
            a[col][k] /= diag;
        }
        rhs[col] /= diag;

        for (int32_t row = 0; row < n; ++row) {
            if (row == col) continue;
            const double factor = a[row][col];
            for (int32_t k = col; k < n; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    return rhs;
}

void update_voltage(
    std::vector<Complex>& v,
    const std::vector<double>& dx,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    const std::vector<int32_t> pvpq = concatenate(pv, pq);
    const int32_t n_pvpq = static_cast<int32_t>(pvpq.size());

    for (int32_t i = 0; i < n_pvpq; ++i) {
        const int32_t bus = pvpq[i];
        const double va = std::arg(v[bus]) + dx[i];
        const double vm = std::abs(v[bus]);
        v[bus] = std::polar(vm, va);
    }
    for (int32_t i = 0; i < static_cast<int32_t>(pq.size()); ++i) {
        const int32_t bus = pq[i];
        const double va = std::arg(v[bus]);
        const double vm = std::abs(v[bus]) + dx[n_pvpq + i];
        v[bus] = std::polar(vm, va);
    }
}

void run_newton_reference(
    const CsrComplex& ybus,
    const std::vector<Complex>& s_spec,
    std::vector<Complex> v,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq)
{
    constexpr int32_t max_iter = 10;
    constexpr double tolerance = 1e-10;

    std::cout << "iter mismatch_inf jac_nnz dx_inf\n";
    for (int32_t iter = 0; iter <= max_iter; ++iter) {
        const std::vector<Complex> s_calc = compute_power(ybus, v);
        const std::vector<double> mismatch =
            build_reduced_mismatch(s_spec, s_calc, pv, pq);
        const double mismatch_norm = max_abs(mismatch);

        CsrReal jacobian = build_pandapower_style_jacobian(ybus, v, pv, pq);
        if (mismatch_norm < tolerance) {
            std::cout << iter << " " << mismatch_norm
                      << " " << jacobian.data.size() << " 0\n";
            break;
        }

        const std::vector<double> dx = solve_dense_linear_system(jacobian, mismatch);
        std::cout << iter << " " << mismatch_norm
                  << " " << jacobian.data.size()
                  << " " << max_abs(dx) << "\n";
        update_voltage(v, dx, pv, pq);
    }

    std::cout << "final_voltage\n";
    for (int32_t bus = 0; bus < static_cast<int32_t>(v.size()); ++bus) {
        std::cout << bus
                  << " Vm=" << std::abs(v[bus])
                  << " Va=" << std::arg(v[bus])
                  << " V=(" << v[bus].real() << ", " << v[bus].imag() << ")\n";
    }
}

}  // namespace

int main()
{
    const CsrComplex ybus{
        3,
        3,
        {0, 2, 5, 7},
        {0, 1, 0, 1, 2, 1, 2},
        {
            {0.0, -10.0}, {0.0, 10.0},
            {0.0, 10.0}, {0.0, -20.0}, {0.0, 10.0},
            {0.0, 10.0}, {0.0, -10.0},
        },
    };
    const std::vector<int32_t> pv{1};
    const std::vector<int32_t> pq{2};

    const std::vector<Complex> target_v{
        {1.0, 0.0},
        {0.98, -0.04},
        {0.96, -0.08},
    };
    const std::vector<Complex> s_spec = compute_power(ybus, target_v);

    std::vector<Complex> v0{
        target_v[0],
        std::polar(std::abs(target_v[1]), 0.0),
        {1.0, 0.0},
    };

    std::cout << std::setprecision(12);
    run_newton_reference(ybus, s_spec, v0, pv, pq);
}
