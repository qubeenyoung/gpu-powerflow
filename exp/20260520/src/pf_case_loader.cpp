#include "pf_case_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace exp_20260520 {
namespace {

struct JacobianIndex {
    int32_t n_bus = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq = 0;
    std::vector<int32_t> row_pvpq;
    std::vector<int32_t> row_pq;
    std::vector<int32_t> col_pvpq;
    std::vector<int32_t> col_pq;
    std::vector<int32_t> pvpq;
};

struct ScatterMap {
    std::vector<int32_t> map_j11;
    std::vector<int32_t> map_j21;
    std::vector<int32_t> map_j12;
    std::vector<int32_t> map_j22;
    std::vector<int32_t> diag_j11;
    std::vector<int32_t> diag_j21;
    std::vector<int32_t> diag_j12;
    std::vector<int32_t> diag_j22;
};

bool is_comment_or_empty(const std::string& line)
{
    for (const char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == '%' || ch == '#';
        }
    }
    return true;
}

std::string next_payload_line(std::ifstream& in, const std::filesystem::path& path)
{
    std::string line;
    while (std::getline(in, line)) {
        if (!is_comment_or_empty(line)) {
            return line;
        }
    }
    throw std::runtime_error("unexpected end of file while reading " + path.string());
}

std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::vector<Complex> load_complex_pairs(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }

    std::vector<Complex> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) {
            continue;
        }
        std::istringstream iss(line);
        double real = 0.0;
        double imag = 0.0;
        if (!(iss >> real >> imag)) {
            throw std::runtime_error("malformed complex vector entry in " + path.string());
        }
        values.emplace_back(real, imag);
    }
    return values;
}

std::vector<int32_t> load_int_values(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }

    std::vector<int32_t> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) {
            continue;
        }
        std::istringstream iss(line);
        int32_t value = 0;
        if (!(iss >> value)) {
            throw std::runtime_error("malformed integer entry in " + path.string());
        }
        values.push_back(value);
    }
    return values;
}

YbusMatrix load_ybus_matrix_market(const std::filesystem::path& path)
{
    struct Entry {
        int32_t row = 0;
        int32_t col = 0;
        Complex value;
    };

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("missing MatrixMarket header in " + path.string());
    }
    header = lowercase(header);
    if (header.find("matrixmarket") == std::string::npos ||
        header.find("coordinate") == std::string::npos) {
        throw std::runtime_error("unsupported MatrixMarket format in " + path.string());
    }

    const bool symmetric = header.find("symmetric") != std::string::npos;
    const bool complex_field = header.find("complex") != std::string::npos;
    const bool real_field = header.find("real") != std::string::npos ||
                            header.find("integer") != std::string::npos ||
                            header.find("pattern") != std::string::npos;
    if (!complex_field && !real_field) {
        throw std::runtime_error("unsupported MatrixMarket field in " + path.string());
    }

    std::istringstream dims(next_payload_line(in, path));
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
    if (!(dims >> rows >> cols >> nnz) || rows <= 0 || cols <= 0 || nnz < 0) {
        throw std::runtime_error("malformed MatrixMarket dimensions in " + path.string());
    }

    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(nnz) * (symmetric ? 2u : 1u));
    for (int32_t k = 0; k < nnz; ++k) {
        std::istringstream iss(next_payload_line(in, path));
        int32_t row = 0;
        int32_t col = 0;
        double real = 1.0;
        double imag = 0.0;
        if (complex_field) {
            if (!(iss >> row >> col >> real >> imag)) {
                throw std::runtime_error("malformed complex MatrixMarket entry in " + path.string());
            }
        } else {
            if (!(iss >> row >> col)) {
                throw std::runtime_error("malformed MatrixMarket entry in " + path.string());
            }
            if (!(iss >> real)) {
                real = 1.0;
            }
        }

        Entry entry{row - 1, col - 1, {real, imag}};
        entries.push_back(entry);
        if (symmetric && entry.row != entry.col) {
            entries.push_back(Entry{entry.col, entry.row, entry.value});
        }
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
    });

    std::vector<Entry> merged;
    merged.reserve(entries.size());
    for (const Entry& entry : entries) {
        if (entry.row < 0 || entry.row >= rows || entry.col < 0 || entry.col >= cols) {
            throw std::runtime_error("MatrixMarket entry out of bounds in " + path.string());
        }
        if (!merged.empty() && merged.back().row == entry.row && merged.back().col == entry.col) {
            merged.back().value += entry.value;
        } else {
            merged.push_back(entry);
        }
    }

    YbusMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    matrix.col_idx.resize(merged.size());
    matrix.values.resize(merged.size());

    for (const Entry& entry : merged) {
        ++matrix.row_ptr[static_cast<std::size_t>(entry.row + 1)];
    }
    for (int32_t row = 0; row < rows; ++row) {
        matrix.row_ptr[static_cast<std::size_t>(row + 1)] +=
            matrix.row_ptr[static_cast<std::size_t>(row)];
    }

    std::vector<int32_t> cursor = matrix.row_ptr;
    for (const Entry& entry : merged) {
        const int32_t dst = cursor[static_cast<std::size_t>(entry.row)]++;
        matrix.col_idx[static_cast<std::size_t>(dst)] = entry.col;
        matrix.values[static_cast<std::size_t>(dst)] = entry.value;
    }
    return matrix;
}

JacobianIndex make_jacobian_index(const CaseData& data)
{
    JacobianIndex index;
    index.n_bus = data.ybus.rows;
    index.n_pvpq = static_cast<int32_t>(data.pv.size() + data.pq.size());
    index.n_pq = static_cast<int32_t>(data.pq.size());
    index.row_pvpq.assign(static_cast<std::size_t>(index.n_bus), -1);
    index.row_pq.assign(static_cast<std::size_t>(index.n_bus), -1);
    index.col_pvpq.assign(static_cast<std::size_t>(index.n_bus), -1);
    index.col_pq.assign(static_cast<std::size_t>(index.n_bus), -1);
    index.pvpq.reserve(static_cast<std::size_t>(index.n_pvpq));

    for (int32_t i = 0; i < static_cast<int32_t>(data.pv.size()); ++i) {
        const int32_t bus = data.pv[static_cast<std::size_t>(i)];
        if (bus < 0 || bus >= index.n_bus || index.row_pvpq[static_cast<std::size_t>(bus)] >= 0) {
            throw std::runtime_error("invalid or duplicate PV bus index");
        }
        index.pvpq.push_back(bus);
        index.row_pvpq[static_cast<std::size_t>(bus)] = i;
        index.col_pvpq[static_cast<std::size_t>(bus)] = i;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(data.pq.size()); ++i) {
        const int32_t bus = data.pq[static_cast<std::size_t>(i)];
        if (bus < 0 || bus >= index.n_bus || index.row_pvpq[static_cast<std::size_t>(bus)] >= 0) {
            throw std::runtime_error("invalid, duplicate, or overlapping PQ bus index");
        }
        const int32_t pvpq_pos = static_cast<int32_t>(data.pv.size()) + i;
        const int32_t pq_pos = index.n_pvpq + i;
        index.pvpq.push_back(bus);
        index.row_pvpq[static_cast<std::size_t>(bus)] = pvpq_pos;
        index.col_pvpq[static_cast<std::size_t>(bus)] = pvpq_pos;
        index.row_pq[static_cast<std::size_t>(bus)] = pq_pos;
        index.col_pq[static_cast<std::size_t>(bus)] = pq_pos;
    }
    return index;
}

void append_jacobian_columns(std::vector<std::vector<int32_t>>& rows,
                             const JacobianIndex& index,
                             int32_t row_bus,
                             int32_t col_bus)
{
    const int32_t row_p = index.row_pvpq[static_cast<std::size_t>(row_bus)];
    const int32_t row_q = index.row_pq[static_cast<std::size_t>(row_bus)];
    const int32_t col_va = index.col_pvpq[static_cast<std::size_t>(col_bus)];
    const int32_t col_vm = index.col_pq[static_cast<std::size_t>(col_bus)];
    if (row_p >= 0 && col_va >= 0) {
        rows[static_cast<std::size_t>(row_p)].push_back(col_va);
    }
    if (row_q >= 0 && col_va >= 0) {
        rows[static_cast<std::size_t>(row_q)].push_back(col_va);
    }
    if (row_p >= 0 && col_vm >= 0) {
        rows[static_cast<std::size_t>(row_p)].push_back(col_vm);
    }
    if (row_q >= 0 && col_vm >= 0) {
        rows[static_cast<std::size_t>(row_q)].push_back(col_vm);
    }
}

CsrMatrix build_jacobian_pattern(const CaseData& data, const JacobianIndex& index)
{
    const int32_t dim = index.n_pvpq + index.n_pq;
    std::vector<std::vector<int32_t>> rows(static_cast<std::size_t>(dim));
    for (int32_t row_bus = 0; row_bus < data.ybus.rows; ++row_bus) {
        for (int32_t p = data.ybus.row_ptr[static_cast<std::size_t>(row_bus)];
             p < data.ybus.row_ptr[static_cast<std::size_t>(row_bus + 1)]; ++p) {
            const int32_t col_bus = data.ybus.col_idx[static_cast<std::size_t>(p)];
            append_jacobian_columns(rows, index, row_bus, col_bus);
        }
    }
    for (int32_t bus = 0; bus < data.ybus.rows; ++bus) {
        append_jacobian_columns(rows, index, bus, bus);
    }

    CsrMatrix jacobian;
    jacobian.rows = dim;
    jacobian.cols = dim;
    jacobian.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    int64_t nnz = 0;
    for (int32_t row = 0; row < dim; ++row) {
        auto& cols = rows[static_cast<std::size_t>(row)];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        nnz += static_cast<int64_t>(cols.size());
        if (nnz > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Jacobian nnz exceeds int32 range");
        }
        jacobian.row_ptr[static_cast<std::size_t>(row + 1)] = static_cast<int32_t>(nnz);
    }
    jacobian.col_idx.reserve(static_cast<std::size_t>(nnz));
    for (const auto& cols : rows) {
        jacobian.col_idx.insert(jacobian.col_idx.end(), cols.begin(), cols.end());
    }
    jacobian.values.assign(static_cast<std::size_t>(nnz), 0.0);
    return jacobian;
}

int32_t find_position(const CsrMatrix& matrix, int32_t row, int32_t col)
{
    if (row < 0 || row >= matrix.rows || col < 0 || col >= matrix.cols) {
        return -1;
    }
    const auto begin = matrix.col_idx.begin() + matrix.row_ptr[static_cast<std::size_t>(row)];
    const auto end = matrix.col_idx.begin() + matrix.row_ptr[static_cast<std::size_t>(row + 1)];
    const auto it = std::lower_bound(begin, end, col);
    if (it == end || *it != col) {
        return -1;
    }
    return static_cast<int32_t>(it - matrix.col_idx.begin());
}

ScatterMap build_scatter_map(const CaseData& data, const JacobianIndex& index, const CsrMatrix& jacobian)
{
    const std::size_t y_nnz = data.ybus.values.size();
    ScatterMap map;
    map.map_j11.assign(y_nnz, -1);
    map.map_j21.assign(y_nnz, -1);
    map.map_j12.assign(y_nnz, -1);
    map.map_j22.assign(y_nnz, -1);
    map.diag_j11.assign(static_cast<std::size_t>(data.ybus.rows), -1);
    map.diag_j21.assign(static_cast<std::size_t>(data.ybus.rows), -1);
    map.diag_j12.assign(static_cast<std::size_t>(data.ybus.rows), -1);
    map.diag_j22.assign(static_cast<std::size_t>(data.ybus.rows), -1);

    for (int32_t row_bus = 0; row_bus < data.ybus.rows; ++row_bus) {
        const int32_t row_p = index.row_pvpq[static_cast<std::size_t>(row_bus)];
        const int32_t row_q = index.row_pq[static_cast<std::size_t>(row_bus)];
        for (int32_t p = data.ybus.row_ptr[static_cast<std::size_t>(row_bus)];
             p < data.ybus.row_ptr[static_cast<std::size_t>(row_bus + 1)]; ++p) {
            const int32_t col_bus = data.ybus.col_idx[static_cast<std::size_t>(p)];
            const int32_t col_va = index.col_pvpq[static_cast<std::size_t>(col_bus)];
            const int32_t col_vm = index.col_pq[static_cast<std::size_t>(col_bus)];
            if (row_p >= 0 && col_va >= 0) {
                map.map_j11[static_cast<std::size_t>(p)] = find_position(jacobian, row_p, col_va);
            }
            if (row_q >= 0 && col_va >= 0) {
                map.map_j21[static_cast<std::size_t>(p)] = find_position(jacobian, row_q, col_va);
            }
            if (row_p >= 0 && col_vm >= 0) {
                map.map_j12[static_cast<std::size_t>(p)] = find_position(jacobian, row_p, col_vm);
            }
            if (row_q >= 0 && col_vm >= 0) {
                map.map_j22[static_cast<std::size_t>(p)] = find_position(jacobian, row_q, col_vm);
            }
            if (row_bus == col_bus) {
                map.diag_j11[static_cast<std::size_t>(row_bus)] = map.map_j11[static_cast<std::size_t>(p)];
                map.diag_j21[static_cast<std::size_t>(row_bus)] = map.map_j21[static_cast<std::size_t>(p)];
                map.diag_j12[static_cast<std::size_t>(row_bus)] = map.map_j12[static_cast<std::size_t>(p)];
                map.diag_j22[static_cast<std::size_t>(row_bus)] = map.map_j22[static_cast<std::size_t>(p)];
            }
        }
    }
    return map;
}

std::vector<Complex> compute_ibus(const CaseData& data)
{
    std::vector<Complex> ibus(static_cast<std::size_t>(data.ybus.rows), Complex(0.0, 0.0));
    for (int32_t row = 0; row < data.ybus.rows; ++row) {
        for (int32_t p = data.ybus.row_ptr[static_cast<std::size_t>(row)];
             p < data.ybus.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            const int32_t col = data.ybus.col_idx[static_cast<std::size_t>(p)];
            ibus[static_cast<std::size_t>(row)] +=
                data.ybus.values[static_cast<std::size_t>(p)] * data.v0[static_cast<std::size_t>(col)];
        }
    }
    return ibus;
}

std::vector<Complex> compute_vnorm(const CaseData& data)
{
    std::vector<Complex> vnorm(static_cast<std::size_t>(data.ybus.rows));
    for (int32_t bus = 0; bus < data.ybus.rows; ++bus) {
        const double vm = std::max(std::abs(data.v0[static_cast<std::size_t>(bus)]), 1.0e-8);
        vnorm[static_cast<std::size_t>(bus)] = data.v0[static_cast<std::size_t>(bus)] / vm;
    }
    return vnorm;
}

void set_if_valid(std::vector<double>& values, int32_t pos, double value)
{
    if (pos >= 0) {
        values[static_cast<std::size_t>(pos)] = value;
    }
}

void add_if_valid(std::vector<double>& values, int32_t pos, double value)
{
    if (pos >= 0) {
        values[static_cast<std::size_t>(pos)] += value;
    }
}

std::vector<double> build_mismatch_rhs(const CaseData& data,
                                       const JacobianIndex& index,
                                       const std::vector<Complex>& ibus)
{
    std::vector<double> rhs(static_cast<std::size_t>(index.n_pvpq + index.n_pq), 0.0);
    int32_t out = 0;
    for (int32_t bus : data.pv) {
        const Complex mismatch =
            data.v0[static_cast<std::size_t>(bus)] * std::conj(ibus[static_cast<std::size_t>(bus)]) -
            data.sbus[static_cast<std::size_t>(bus)];
        rhs[static_cast<std::size_t>(out++)] = -mismatch.real();
    }
    for (int32_t bus : data.pq) {
        const Complex mismatch =
            data.v0[static_cast<std::size_t>(bus)] * std::conj(ibus[static_cast<std::size_t>(bus)]) -
            data.sbus[static_cast<std::size_t>(bus)];
        rhs[static_cast<std::size_t>(out++)] = -mismatch.real();
    }
    for (int32_t bus : data.pq) {
        const Complex mismatch =
            data.v0[static_cast<std::size_t>(bus)] * std::conj(ibus[static_cast<std::size_t>(bus)]) -
            data.sbus[static_cast<std::size_t>(bus)];
        rhs[static_cast<std::size_t>(out++)] = -mismatch.imag();
    }
    return rhs;
}

std::vector<double> make_synthetic_x_ref(int32_t n)
{
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] =
            1.0 + 0.125 * std::sin(0.017 * static_cast<double>(i)) +
            0.05 * std::cos(0.031 * static_cast<double>(i));
    }
    return x;
}

}  // namespace

CaseData load_case(const std::filesystem::path& case_dir, const std::string& case_name)
{
    CaseData data;
    data.case_name = case_name.empty() ? case_dir.filename().string() : case_name;
    data.ybus = load_ybus_matrix_market(case_dir / "dump_Ybus.mtx");
    data.v0 = load_complex_pairs(case_dir / "dump_V.txt");
    data.sbus = load_complex_pairs(case_dir / "dump_Sbus.txt");
    data.pv = load_int_values(case_dir / "dump_pv.txt");
    data.pq = load_int_values(case_dir / "dump_pq.txt");

    if (data.ybus.rows != data.ybus.cols) {
        throw std::runtime_error("Ybus must be square");
    }
    if (static_cast<int32_t>(data.v0.size()) != data.ybus.rows ||
        static_cast<int32_t>(data.sbus.size()) != data.ybus.rows) {
        throw std::runtime_error("V0/Sbus size does not match Ybus");
    }
    if (data.pv.empty() && data.pq.empty()) {
        throw std::runtime_error("case has no PV/PQ buses");
    }
    return data;
}

LinearSystem build_linear_system(const CaseData& data, const std::string& rhs_mode)
{
    if (rhs_mode != "synthetic" && rhs_mode != "mismatch") {
        throw std::runtime_error("rhs_mode must be synthetic or mismatch");
    }

    static constexpr Complex j(0.0, 1.0);

    const JacobianIndex index = make_jacobian_index(data);
    CsrMatrix jacobian = build_jacobian_pattern(data, index);
    const ScatterMap map = build_scatter_map(data, index, jacobian);
    const std::vector<Complex> ibus = compute_ibus(data);
    const std::vector<Complex> vnorm = compute_vnorm(data);

    for (int32_t row_bus = 0; row_bus < data.ybus.rows; ++row_bus) {
        for (int32_t p = data.ybus.row_ptr[static_cast<std::size_t>(row_bus)];
             p < data.ybus.row_ptr[static_cast<std::size_t>(row_bus + 1)]; ++p) {
            const int32_t col_bus = data.ybus.col_idx[static_cast<std::size_t>(p)];
            const Complex y = data.ybus.values[static_cast<std::size_t>(p)];
            const Complex d_angle =
                -j * data.v0[static_cast<std::size_t>(row_bus)] *
                std::conj(y * data.v0[static_cast<std::size_t>(col_bus)]);
            const Complex d_magnitude =
                data.v0[static_cast<std::size_t>(row_bus)] *
                std::conj(y * vnorm[static_cast<std::size_t>(col_bus)]);

            set_if_valid(jacobian.values, map.map_j11[static_cast<std::size_t>(p)], d_angle.real());
            set_if_valid(jacobian.values, map.map_j21[static_cast<std::size_t>(p)], d_angle.imag());
            set_if_valid(jacobian.values, map.map_j12[static_cast<std::size_t>(p)], d_magnitude.real());
            set_if_valid(jacobian.values, map.map_j22[static_cast<std::size_t>(p)], d_magnitude.imag());
        }
    }

    for (int32_t bus = 0; bus < data.ybus.rows; ++bus) {
        const Complex d_angle_diag =
            j * (data.v0[static_cast<std::size_t>(bus)] * std::conj(ibus[static_cast<std::size_t>(bus)]));
        const Complex d_magnitude_diag =
            std::conj(ibus[static_cast<std::size_t>(bus)]) * vnorm[static_cast<std::size_t>(bus)];
        add_if_valid(jacobian.values, map.diag_j11[static_cast<std::size_t>(bus)], d_angle_diag.real());
        add_if_valid(jacobian.values, map.diag_j21[static_cast<std::size_t>(bus)], d_angle_diag.imag());
        add_if_valid(jacobian.values, map.diag_j12[static_cast<std::size_t>(bus)], d_magnitude_diag.real());
        add_if_valid(jacobian.values, map.diag_j22[static_cast<std::size_t>(bus)], d_magnitude_diag.imag());
    }

    LinearSystem system;
    system.matrix = std::move(jacobian);
    system.rhs_mode = rhs_mode;
    if (rhs_mode == "synthetic") {
        system.x_ref = make_synthetic_x_ref(system.matrix.rows);
        system.rhs = matvec(system.matrix, system.x_ref);
    } else {
        system.rhs = build_mismatch_rhs(data, index, ibus);
        system.x_ref.assign(static_cast<std::size_t>(system.matrix.rows), 0.0);
    }
    return system;
}

std::vector<double> matvec(const CsrMatrix& matrix, const std::vector<double>& x)
{
    std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
    for (int32_t row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int32_t p = matrix.row_ptr[static_cast<std::size_t>(row)];
             p < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            sum += matrix.values[static_cast<std::size_t>(p)] *
                   x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(p)])];
        }
        y[static_cast<std::size_t>(row)] = sum;
    }
    return y;
}

double norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

double relative_error(const std::vector<double>& x, const std::vector<double>& ref)
{
    if (x.size() != ref.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> diff(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        diff[i] = x[i] - ref[i];
    }
    const double denom = std::max(norm2(ref), std::numeric_limits<double>::min());
    return norm2(diff) / denom;
}

}  // namespace exp_20260520
