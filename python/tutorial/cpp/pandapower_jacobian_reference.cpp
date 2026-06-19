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
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <suitesparse/klu.h>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
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

struct CscReal {
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

struct DumpCase {
    std::string case_name;
    CsrComplex ybus;
    std::vector<Complex> sbus;
    std::vector<Complex> v0;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct StageTiming {
    double init_ms = 0.0;
    double solve_total_ms = 0.0;
    double ibus_ms = 0.0;
    double mismatch_ms = 0.0;
    double mnorm_ms = 0.0;
    double jacobian_ms = 0.0;
    double prepare_rhs_ms = 0.0;
    double factorize_ms = 0.0;
    double triangular_solve_ms = 0.0;
    double voltage_update_ms = 0.0;
};

struct BenchmarkResult {
    StageTiming timing;
    int32_t iterations = 0;
    double final_mismatch = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
};

struct BenchmarkOptions {
    std::vector<fs::path> case_dirs;
    int32_t max_iter = 10;
    int32_t warmup = 1;
    int32_t repeats = 5;
    double tolerance = 1e-8;
    double dense_memory_limit_gb = 8.0;
    bool use_klu = true;
};

std::vector<int32_t> concatenate(const std::vector<int32_t>& a, const std::vector<int32_t>& b)
{
    std::vector<int32_t> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string csv_escape(const std::string& input)
{
    const bool quote = input.find(',') != std::string::npos ||
                       input.find('"') != std::string::npos ||
                       input.find('\n') != std::string::npos ||
                       input.find('\r') != std::string::npos;
    if (!quote) return input;
    std::string out = "\"";
    for (char ch : input) {
        if (ch == '"') out += '"';
        out += ch;
    }
    out += '"';
    return out;
}

bool is_comment_or_empty(const std::string& line)
{
    for (char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == '%' || ch == '#';
        }
    }
    return true;
}

std::vector<Complex> load_complex_pairs(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open " + path.string());
    std::vector<Complex> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) continue;
        std::istringstream iss(line);
        double re = 0.0;
        double im = 0.0;
        if (!(iss >> re >> im)) {
            throw std::runtime_error("malformed complex pair in " + path.string());
        }
        values.emplace_back(re, im);
    }
    return values;
}

std::vector<int32_t> load_int_values(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open " + path.string());
    std::vector<int32_t> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) continue;
        std::istringstream iss(line);
        int32_t value = 0;
        if (!(iss >> value)) {
            throw std::runtime_error("malformed integer in " + path.string());
        }
        values.push_back(value);
    }
    return values;
}

CsrComplex load_matrix_market_complex_csr(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open " + path.string());

    std::string header;
    std::getline(in, header);
    std::string lowered = header;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const bool symmetric = lowered.find("symmetric") != std::string::npos;
    if (lowered.find("matrixmarket") == std::string::npos ||
        lowered.find("coordinate") == std::string::npos ||
        lowered.find("complex") == std::string::npos) {
        throw std::runtime_error("unsupported MatrixMarket header in " + path.string());
    }

    std::string line;
    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("missing MatrixMarket dimensions in " + path.string());
        }
    } while (is_comment_or_empty(line));

    std::istringstream dims(line);
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    if (!(dims >> rows >> cols >> nnz)) {
        throw std::runtime_error("malformed MatrixMarket dimensions in " + path.string());
    }

    struct Entry {
        int32_t row = 0;
        int32_t col = 0;
        Complex value{};
    };
    std::vector<Entry> entries;
    entries.reserve(symmetric ? static_cast<std::size_t>(nnz) * 2 : static_cast<std::size_t>(nnz));
    for (int32_t k = 0; k < nnz; ++k) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("unexpected end of MatrixMarket file " + path.string());
        }
        if (is_comment_or_empty(line)) {
            --k;
            continue;
        }
        std::istringstream iss(line);
        int32_t row = 0;
        int32_t col = 0;
        double re = 0.0;
        double im = 0.0;
        if (!(iss >> row >> col >> re >> im)) {
            throw std::runtime_error("malformed MatrixMarket entry in " + path.string());
        }
        const Entry entry{row - 1, col - 1, Complex{re, im}};
        entries.push_back(entry);
        if (symmetric && entry.row != entry.col) {
            entries.push_back({entry.col, entry.row, entry.value});
        }
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.row == b.row ? a.col < b.col : a.row < b.row;
    });

    std::vector<Entry> merged;
    merged.reserve(entries.size());
    for (const Entry& entry : entries) {
        if (!merged.empty() &&
            merged.back().row == entry.row &&
            merged.back().col == entry.col) {
            merged.back().value += entry.value;
        } else {
            merged.push_back(entry);
        }
    }

    CsrComplex csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.indptr.assign(rows + 1, 0);
    for (const Entry& entry : merged) {
        if (entry.row < 0 || entry.row >= rows || entry.col < 0 || entry.col >= cols) {
            throw std::runtime_error("MatrixMarket entry out of bounds in " + path.string());
        }
        ++csr.indptr[entry.row + 1];
    }
    for (int32_t row = 0; row < rows; ++row) {
        csr.indptr[row + 1] += csr.indptr[row];
    }
    csr.indices.reserve(merged.size());
    csr.data.reserve(merged.size());
    for (const Entry& entry : merged) {
        csr.indices.push_back(entry.col);
        csr.data.push_back(entry.value);
    }
    return csr;
}

DumpCase load_dump_case(const fs::path& case_dir)
{
    DumpCase out;
    out.case_name = case_dir.filename().string();
    out.ybus = load_matrix_market_complex_csr(case_dir / "dump_Ybus.mtx");
    out.sbus = load_complex_pairs(case_dir / "dump_Sbus.txt");
    out.v0 = load_complex_pairs(case_dir / "dump_V.txt");
    out.pv = load_int_values(case_dir / "dump_pv.txt");
    out.pq = load_int_values(case_dir / "dump_pq.txt");
    if (static_cast<int32_t>(out.sbus.size()) != out.ybus.rows ||
        static_cast<int32_t>(out.v0.size()) != out.ybus.rows) {
        throw std::runtime_error("dump vector sizes do not match Ybus rows: " + case_dir.string());
    }
    return out;
}

std::vector<Complex> compute_power_from_ibus(
    const std::vector<Complex>& ibus,
    const std::vector<Complex>& v)
{
    std::vector<Complex> s_calc(v.size());
    for (int32_t bus = 0; bus < static_cast<int32_t>(v.size()); ++bus) {
        s_calc[bus] = v[bus] * std::conj(ibus[bus]);
    }
    return s_calc;
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

CscReal to_csc(const CsrReal& csr)
{
    CscReal csc;
    csc.rows = csr.rows;
    csc.cols = csr.cols;
    csc.indptr.assign(csr.cols + 1, 0);
    csc.indices.assign(csr.data.size(), 0);
    csc.data.assign(csr.data.size(), 0.0);

    for (int32_t row = 0; row < csr.rows; ++row) {
        for (int32_t k = csr.indptr[row]; k < csr.indptr[row + 1]; ++k) {
            ++csc.indptr[csr.indices[k] + 1];
        }
    }
    for (int32_t col = 0; col < csr.cols; ++col) {
        csc.indptr[col + 1] += csc.indptr[col];
    }

    std::vector<int32_t> next = csc.indptr;
    for (int32_t row = 0; row < csr.rows; ++row) {
        for (int32_t k = csr.indptr[row]; k < csr.indptr[row + 1]; ++k) {
            const int32_t col = csr.indices[k];
            const int32_t dst = next[col]++;
            csc.indices[dst] = row;
            csc.data[dst] = csr.data[k];
        }
    }
    return csc;
}

class KluLinearSolver {
public:
    KluLinearSolver()
    {
        klu_defaults(&common_);
    }

    ~KluLinearSolver()
    {
        release();
    }

    KluLinearSolver(const KluLinearSolver&) = delete;
    KluLinearSolver& operator=(const KluLinearSolver&) = delete;

    void analyze(const CsrReal& matrix)
    {
        release();
        n_ = matrix.rows;
        csc_ = to_csc(matrix);
        symbolic_ = klu_analyze(n_, csc_.indptr.data(), csc_.indices.data(), &common_);
        if (symbolic_ == nullptr) {
            throw std::runtime_error("KLU symbolic analysis failed");
        }
    }

    void factorize(const CsrReal& matrix)
    {
        if (symbolic_ == nullptr) {
            analyze(matrix);
        } else if (matrix.rows != n_ || matrix.cols != n_) {
            throw std::runtime_error("KLU factorize shape mismatch");
        } else {
            csc_ = to_csc(matrix);
        }
        if (numeric_ != nullptr) {
            klu_free_numeric(&numeric_, &common_);
        }
        numeric_ = klu_factor(csc_.indptr.data(), csc_.indices.data(), csc_.data.data(),
                              symbolic_, &common_);
        if (numeric_ == nullptr) {
            throw std::runtime_error("KLU numeric factorization failed");
        }
    }

    void solve(std::vector<double>& rhs)
    {
        if (symbolic_ == nullptr || numeric_ == nullptr) {
            throw std::runtime_error("KLU solve before factorize");
        }
        if (static_cast<int32_t>(rhs.size()) != n_) {
            throw std::runtime_error("KLU rhs size mismatch");
        }
        if (!klu_solve(symbolic_, numeric_, n_, 1, rhs.data(), &common_)) {
            throw std::runtime_error("KLU solve failed");
        }
    }

private:
    void release()
    {
        if (numeric_ != nullptr) {
            klu_free_numeric(&numeric_, &common_);
            numeric_ = nullptr;
        }
        if (symbolic_ != nullptr) {
            klu_free_symbolic(&symbolic_, &common_);
            symbolic_ = nullptr;
        }
    }

    int32_t n_ = 0;
    CscReal csc_;
    klu_common common_{};
    klu_symbolic* symbolic_ = nullptr;
    klu_numeric* numeric_ = nullptr;
};

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

BenchmarkResult run_newton_reference_timed(
    const CsrComplex& ybus,
    const std::vector<Complex>& s_spec,
    std::vector<Complex> v,
    const std::vector<int32_t>& pv,
    const std::vector<int32_t>& pq,
    const BenchmarkOptions& options)
{
    BenchmarkResult result;
    KluLinearSolver linear_solver;

    {
        const auto start = Clock::now();
        CsrReal initial_jacobian = build_pandapower_style_jacobian(ybus, v, pv, pq);
        linear_solver.analyze(initial_jacobian);
        result.timing.init_ms = elapsed_ms(start, Clock::now());
    }

    const auto solve_start = Clock::now();
    for (int32_t iter = 0; iter < options.max_iter; ++iter) {
        result.iterations = iter + 1;

        std::vector<Complex> ibus;
        {
            const auto start = Clock::now();
            ibus = compute_ibus(ybus, v);
            result.timing.ibus_ms += elapsed_ms(start, Clock::now());
        }

        std::vector<double> mismatch;
        {
            const auto start = Clock::now();
            const std::vector<Complex> s_calc = compute_power_from_ibus(ibus, v);
            mismatch = build_reduced_mismatch(s_spec, s_calc, pv, pq);
            result.timing.mismatch_ms += elapsed_ms(start, Clock::now());
        }

        {
            const auto start = Clock::now();
            result.final_mismatch = max_abs(mismatch);
            result.timing.mnorm_ms += elapsed_ms(start, Clock::now());
        }
        if (result.final_mismatch < options.tolerance) {
            result.converged = true;
            break;
        }

        CsrReal jacobian;
        {
            const auto start = Clock::now();
            jacobian = build_pandapower_style_jacobian(ybus, v, pv, pq);
            result.timing.jacobian_ms += elapsed_ms(start, Clock::now());
        }

        std::vector<double> rhs;
        {
            const auto start = Clock::now();
            rhs = mismatch;
            result.timing.prepare_rhs_ms += elapsed_ms(start, Clock::now());
        }

        {
            const auto start = Clock::now();
            linear_solver.factorize(jacobian);
            result.timing.factorize_ms += elapsed_ms(start, Clock::now());
        }

        {
            const auto start = Clock::now();
            linear_solver.solve(rhs);
            result.timing.triangular_solve_ms += elapsed_ms(start, Clock::now());
        }

        {
            const auto start = Clock::now();
            update_voltage(v, rhs, pv, pq);
            result.timing.voltage_update_ms += elapsed_ms(start, Clock::now());
        }
    }
    result.timing.solve_total_ms = elapsed_ms(solve_start, Clock::now());
    return result;
}

std::string require_value(int& i, int argc, char** argv, const std::string& key)
{
    if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + key);
    }
    return argv[++i];
}

BenchmarkOptions parse_benchmark_options(int argc, char** argv)
{
    BenchmarkOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--max-iter") {
            options.max_iter = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--tolerance") {
            options.tolerance = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--warmup") {
            options.warmup = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--repeats") {
            options.repeats = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--dense-memory-limit-gb") {
            options.dense_memory_limit_gb = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--dense") {
            options.use_klu = false;
        } else if (key == "--klu") {
            options.use_klu = true;
        } else if (key == "--case-dir") {
            options.case_dirs.push_back(require_value(i, argc, argv, key));
        } else if (key == "-h" || key == "--help") {
            std::cout
                << "Usage: pandapower_jacobian_reference [options] CASE_DIR...\n"
                << "Options:\n"
                << "  --case-dir DIR              add one dump case directory\n"
                << "  --repeats N                 measured repeats (default 5)\n"
                << "  --warmup N                  warmup repeats (default 1)\n"
                << "  --max-iter N                max Newton iterations (default 10)\n"
                << "  --tolerance VALUE           convergence tolerance (default 1e-8)\n"
                << "  --klu                       use KLU sparse factor/solve (default)\n"
                << "  --dense                     reserved for tiny cases; dense built-in demo remains no-arg\n";
            std::exit(0);
        } else if (!key.empty() && key[0] == '-') {
            throw std::invalid_argument("unknown argument: " + key);
        } else {
            options.case_dirs.push_back(key);
        }
    }
    return options;
}

void write_benchmark_header()
{
    std::cout
        << "case_name,n_bus,ybus_nnz,repeat_idx,init_ms,solve_total_ms,"
        << "ibus_ms,mismatch_ms,mnorm_ms,jacobian_ms,prepare_rhs_ms,"
        << "factorize_ms,triangular_solve_ms,voltage_update_ms,"
        << "iterations,final_mismatch,converged\n";
}

void write_benchmark_row(
    const DumpCase& data,
    int32_t repeat,
    const BenchmarkResult& result)
{
    const StageTiming& t = result.timing;
    std::cout << csv_escape(data.case_name) << ","
              << data.ybus.rows << ","
              << data.ybus.data.size() << ","
              << repeat << ","
              << t.init_ms << ","
              << t.solve_total_ms << ","
              << t.ibus_ms << ","
              << t.mismatch_ms << ","
              << t.mnorm_ms << ","
              << t.jacobian_ms << ","
              << t.prepare_rhs_ms << ","
              << t.factorize_ms << ","
              << t.triangular_solve_ms << ","
              << t.voltage_update_ms << ","
              << result.iterations << ","
              << result.final_mismatch << ","
              << (result.converged ? 1 : 0) << "\n";
}

void run_benchmark(const BenchmarkOptions& options)
{
    if (!options.use_klu) {
        throw std::invalid_argument("--dense benchmark mode is not enabled for dump cases; use no args for the dense 3-bus demo");
    }
    std::cout << std::setprecision(17);
    write_benchmark_header();
    for (const fs::path& case_dir : options.case_dirs) {
        try {
            const DumpCase data = load_dump_case(case_dir);
            for (int32_t i = 0; i < options.warmup; ++i) {
                (void)run_newton_reference_timed(data.ybus, data.sbus, data.v0, data.pv, data.pq, options);
            }
            for (int32_t repeat = 0; repeat < options.repeats; ++repeat) {
                const BenchmarkResult result =
                    run_newton_reference_timed(data.ybus, data.sbus, data.v0, data.pv, data.pq, options);
                write_benchmark_row(data, repeat, result);
            }
        } catch (const std::bad_alloc& exc) {
            std::cerr << "[tutorial-cpp][SKIP] case=" << case_dir.filename().string()
                      << " error=out_of_memory: " << exc.what() << "\n";
        } catch (const std::exception& exc) {
            std::cerr << "[tutorial-cpp][SKIP] case=" << case_dir.filename().string()
                      << " error=" << exc.what() << "\n";
        }
    }
}

void run_builtin_example()
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

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 1) {
            run_builtin_example();
        } else {
            const BenchmarkOptions options = parse_benchmark_options(argc, argv);
            run_benchmark(options);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "pandapower_jacobian_reference: " << exc.what() << "\n";
        return 2;
    }
}
