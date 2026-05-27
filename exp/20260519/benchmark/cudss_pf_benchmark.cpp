#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cudss.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"
#include "numeric/numeric_factorization_cuda.h"
#include "reordering/metis_nd.h"
#include "symbolic/symbolic_factorization.h"
}

namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;

struct YbusMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<Complex> values;
};

struct CaseData {
    std::string case_name;
    YbusMatrix ybus;
    std::vector<Complex> v0;
    std::vector<Complex> sbus;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct CsrMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;

    int32_t nnz() const
    {
        return static_cast<int32_t>(values.size());
    }
};

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

struct LinearSystem {
    CsrMatrix matrix;
    std::vector<double> rhs;
    std::vector<double> x_ref;
    std::string rhs_mode;
};

struct CliOptions {
    std::filesystem::path case_dir;
    std::filesystem::path dump_reorder_path;
    std::string case_name;
    std::string precision = "fp64";
    std::string rhs_mode = "synthetic";
    std::string cudss_perm = "default";
    int32_t warmup = 1;
    int32_t repeats = 5;
    bool csv = false;
    bool enable_mt = false;
    bool metis_symbolic = false;
    bool custom_numeric = false;
    bool custom_progress = false;
    bool cuda_profiler_capture = false;
    bool split_cudss_phases = false;
    std::string threading_lib;
};

struct Stats {
    double mean = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
};

struct TimingSample {
    double analysis_ms = 0.0;
    double analysis_reordering_ms = std::numeric_limits<double>::quiet_NaN();
    double analysis_symbolic_factorization_ms = std::numeric_limits<double>::quiet_NaN();
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    double solve_forward_ms = std::numeric_limits<double>::quiet_NaN();
    double solve_forward_to_rhs_ms = std::numeric_limits<double>::quiet_NaN();
    double solve_backward_ms = std::numeric_limits<double>::quiet_NaN();
    int64_t analysis_lu_nnz = -1;
    int64_t factor_lu_nnz = -1;
};

struct BenchmarkResult {
    Stats analysis_ms;
    Stats analysis_reordering_ms;
    Stats analysis_symbolic_factorization_ms;
    Stats factor_ms;
    Stats solve_ms;
    Stats solve_forward_ms;
    Stats solve_forward_to_rhs_ms;
    Stats solve_backward_ms;
    Stats total_ms;
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    double relative_residual = std::numeric_limits<double>::quiet_NaN();
    double relative_error = std::numeric_limits<double>::quiet_NaN();
    int64_t cudss_analysis_lu_nnz = -1;
    int64_t cudss_factor_lu_nnz = -1;
    double cudss_factor_lu_nnz_per_nnz = std::numeric_limits<double>::quiet_NaN();
};

struct MetisSymbolicResult {
    int status = SDS_OK;
    double ordering_ms = std::numeric_limits<double>::quiet_NaN();
    double symbolic_ms = std::numeric_limits<double>::quiet_NaN();
    int num_fronts = 0;
    int num_levels = 0;
    int max_front_size = 0;
    std::size_t total_dense_entries = 0;
    std::size_t total_dense_bytes = 0;
    double dense_entries_per_nnz = std::numeric_limits<double>::quiet_NaN();
    std::vector<int32_t> perm;
    std::vector<int32_t> inv_perm;
};

struct CustomNumericResult {
    int status = SDS_OK;
    int num_fronts = 0;
    int num_levels = 0;
    int max_front_size = 0;
    std::size_t total_dense_entries = 0;
    std::size_t total_dense_bytes = 0;
    double ordering_ms = std::numeric_limits<double>::quiet_NaN();
    double symbolic_ms = std::numeric_limits<double>::quiet_NaN();
    double numeric_create_ms = std::numeric_limits<double>::quiet_NaN();
    Stats factor_ms;
};

void cuda_check(cudaError_t status, const char* expr, const char* file, int line)
{
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ':' << line << " in " << expr << ": "
            << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

void cudss_check(cudssStatus_t status, const char* expr, const char* file, int line)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuDSS error at " << file << ':' << line << " in " << expr << ": status "
            << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUDSS_CHECK(expr) cudss_check((expr), #expr, __FILE__, __LINE__)

struct NvtxRange {
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

template <typename Func>
double time_ms(Func&& func);

template <typename Func>
double time_ms_nvtx(const char* range_name, Func&& func);

std::string cudss_version();

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --case-dir PATH [options]\n\n"
        << "Options:\n"
        << "  --case NAME\n"
        << "  --precision fp64|fp32\n"
        << "  --rhs-mode synthetic|mismatch\n"
        << "  --cudss-perm default|metis\n"
        << "  --warmup INT\n"
        << "  --repeats INT\n"
        << "  --dump-reorder PATH\n"
        << "  --csv\n"
        << "  --metis-symbolic\n"
        << "  --custom-numeric\n"
        << "  --custom-progress\n"
        << "  --enable-mt\n"
        << "  --threading-lib PATH\n"
        << "  --cuda-profiler-capture\n"
        << "  --split-cudss-phases (alias: --split-cudss-analysis)\n";
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--case-dir") {
            options.case_dir = need_value("--case-dir");
        } else if (arg == "--case") {
            options.case_name = need_value("--case");
        } else if (arg == "--precision") {
            options.precision = need_value("--precision");
        } else if (arg == "--rhs-mode") {
            options.rhs_mode = need_value("--rhs-mode");
        } else if (arg == "--cudss-perm") {
            options.cudss_perm = need_value("--cudss-perm");
        } else if (arg == "--dump-reorder") {
            options.dump_reorder_path = need_value("--dump-reorder");
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int32_t>(std::stoi(need_value("--warmup")));
        } else if (arg == "--repeats") {
            options.repeats = static_cast<int32_t>(std::stoi(need_value("--repeats")));
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--metis-symbolic") {
            options.metis_symbolic = true;
        } else if (arg == "--custom-numeric") {
            options.custom_numeric = true;
        } else if (arg == "--custom-progress") {
            options.custom_progress = true;
        } else if (arg == "--enable-mt") {
            options.enable_mt = true;
        } else if (arg == "--threading-lib") {
            options.threading_lib = need_value("--threading-lib");
        } else if (arg == "--cuda-profiler-capture") {
            options.cuda_profiler_capture = true;
        } else if (arg == "--split-cudss-analysis" || arg == "--split-cudss-phases") {
            options.split_cudss_phases = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.case_dir.empty()) {
        throw std::runtime_error("--case-dir is required");
    }
    if (options.case_name.empty()) {
        options.case_name = options.case_dir.filename().string();
    }
    if (options.precision != "fp64" && options.precision != "fp32") {
        throw std::runtime_error("--precision must be fp64 or fp32");
    }
    if (options.rhs_mode != "synthetic" && options.rhs_mode != "mismatch") {
        throw std::runtime_error("--rhs-mode must be synthetic or mismatch");
    }
    if (options.cudss_perm != "default" && options.cudss_perm != "metis") {
        throw std::runtime_error("--cudss-perm must be default or metis");
    }
    if (options.warmup < 0 || options.repeats <= 0) {
        throw std::runtime_error("--warmup must be nonnegative and --repeats must be positive");
    }
    return options;
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

std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
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

CaseData load_case(const CliOptions& options)
{
    CaseData data;
    data.case_name = options.case_name;
    data.ybus = load_ybus_matrix_market(options.case_dir / "dump_Ybus.mtx");
    data.v0 = load_complex_pairs(options.case_dir / "dump_V.txt");
    data.sbus = load_complex_pairs(options.case_dir / "dump_Sbus.txt");
    data.pv = load_int_values(options.case_dir / "dump_pv.txt");
    data.pq = load_int_values(options.case_dir / "dump_pq.txt");

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

    CsrMatrix J;
    J.rows = dim;
    J.cols = dim;
    J.row_ptr.assign(static_cast<std::size_t>(dim + 1), 0);
    int64_t nnz = 0;
    for (int32_t row = 0; row < dim; ++row) {
        auto& cols = rows[static_cast<std::size_t>(row)];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        nnz += static_cast<int64_t>(cols.size());
        if (nnz > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Jacobian nnz exceeds int32 range");
        }
        J.row_ptr[static_cast<std::size_t>(row + 1)] = static_cast<int32_t>(nnz);
    }
    J.col_idx.reserve(static_cast<std::size_t>(nnz));
    for (const auto& cols : rows) {
        J.col_idx.insert(J.col_idx.end(), cols.begin(), cols.end());
    }
    J.values.assign(static_cast<std::size_t>(nnz), 0.0);
    return J;
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

ScatterMap build_scatter_map(const CaseData& data, const JacobianIndex& index, const CsrMatrix& J)
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
                map.map_j11[static_cast<std::size_t>(p)] = find_position(J, row_p, col_va);
            }
            if (row_q >= 0 && col_va >= 0) {
                map.map_j21[static_cast<std::size_t>(p)] = find_position(J, row_q, col_va);
            }
            if (row_p >= 0 && col_vm >= 0) {
                map.map_j12[static_cast<std::size_t>(p)] = find_position(J, row_p, col_vm);
            }
            if (row_q >= 0 && col_vm >= 0) {
                map.map_j22[static_cast<std::size_t>(p)] = find_position(J, row_q, col_vm);
            }
            if (row_bus == col_bus) {
                map.diag_j11[static_cast<std::size_t>(row_bus)] =
                    map.map_j11[static_cast<std::size_t>(p)];
                map.diag_j21[static_cast<std::size_t>(row_bus)] =
                    map.map_j21[static_cast<std::size_t>(p)];
                map.diag_j12[static_cast<std::size_t>(row_bus)] =
                    map.map_j12[static_cast<std::size_t>(p)];
                map.diag_j22[static_cast<std::size_t>(row_bus)] =
                    map.map_j22[static_cast<std::size_t>(p)];
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

LinearSystem build_linear_system(const CaseData& data, const CliOptions& options)
{
    static constexpr Complex j(0.0, 1.0);

    const JacobianIndex index = make_jacobian_index(data);
    CsrMatrix J = build_jacobian_pattern(data, index);
    const ScatterMap map = build_scatter_map(data, index, J);
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

            set_if_valid(J.values, map.map_j11[static_cast<std::size_t>(p)], d_angle.real());
            set_if_valid(J.values, map.map_j21[static_cast<std::size_t>(p)], d_angle.imag());
            set_if_valid(J.values, map.map_j12[static_cast<std::size_t>(p)], d_magnitude.real());
            set_if_valid(J.values, map.map_j22[static_cast<std::size_t>(p)], d_magnitude.imag());
        }
    }

    for (int32_t bus = 0; bus < data.ybus.rows; ++bus) {
        const Complex d_angle_diag = j * (data.v0[static_cast<std::size_t>(bus)] *
                                          std::conj(ibus[static_cast<std::size_t>(bus)]));
        const Complex d_magnitude_diag =
            std::conj(ibus[static_cast<std::size_t>(bus)]) * vnorm[static_cast<std::size_t>(bus)];
        add_if_valid(J.values, map.diag_j11[static_cast<std::size_t>(bus)], d_angle_diag.real());
        add_if_valid(J.values, map.diag_j21[static_cast<std::size_t>(bus)], d_angle_diag.imag());
        add_if_valid(J.values, map.diag_j12[static_cast<std::size_t>(bus)], d_magnitude_diag.real());
        add_if_valid(J.values, map.diag_j22[static_cast<std::size_t>(bus)], d_magnitude_diag.imag());
    }

    LinearSystem system;
    system.matrix = std::move(J);
    system.rhs_mode = options.rhs_mode;
    if (options.rhs_mode == "synthetic") {
        system.x_ref = make_synthetic_x_ref(system.matrix.rows);
        system.rhs = matvec(system.matrix, system.x_ref);
    } else {
        system.rhs = build_mismatch_rhs(data, index, ibus);
        system.x_ref.assign(static_cast<std::size_t>(system.matrix.rows), 0.0);
    }
    return system;
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

Stats make_stats(std::vector<double> values)
{
    Stats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    stats.min = values.front();
    stats.max = values.back();
    stats.median = values.size() % 2 == 0
        ? 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2])
        : values[values.size() / 2];
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());
    double accum = 0.0;
    for (double value : values) {
        const double diff = value - stats.mean;
        accum += diff * diff;
    }
    stats.stddev = std::sqrt(accum / static_cast<double>(values.size()));
    return stats;
}

void check_sds(int status, const char* expr)
{
    if (status != SDS_OK) {
        std::ostringstream oss;
        oss << expr << " failed with status " << status;
        throw std::runtime_error(oss.str());
    }
}

CSRMatrix make_project_csr(const CsrMatrix& matrix)
{
    CSRMatrix out;
    std::memset(&out, 0, sizeof(out));
    check_sds(csr_create(&out, matrix.rows, matrix.cols, matrix.nnz()), "csr_create");
    std::memcpy(out.rowptr,
                matrix.row_ptr.data(),
                static_cast<std::size_t>(matrix.rows + 1) * sizeof(int));
    std::memcpy(out.colind,
                matrix.col_idx.data(),
                static_cast<std::size_t>(matrix.nnz()) * sizeof(int));
    std::memcpy(out.values,
                matrix.values.data(),
                static_cast<std::size_t>(matrix.nnz()) * sizeof(double));
    return out;
}

int max_front_size(const SymbolicFactorization& symbolic)
{
    int max_size = 0;
    for (int i = 0; i < symbolic.storage.num_fronts; ++i) {
        max_size = std::max(max_size, symbolic.storage.nfront[i]);
    }
    return max_size;
}

MetisSymbolicResult run_metis_symbolic_analysis(const LinearSystem& system)
{
    MetisSymbolicResult result;
    CSRMatrix matrix = make_project_csr(system.matrix);
    CSRMatrix graph;
    CSRMatrix matrix_perm;
    CSCMatrix matrix_perm_csc;
    SymbolicFactorization symbolic;
    std::vector<int> perm(static_cast<std::size_t>(matrix.nrows), 0);
    std::vector<int> inv_perm(static_cast<std::size_t>(matrix.nrows), 0);

    std::memset(&graph, 0, sizeof(graph));
    std::memset(&matrix_perm, 0, sizeof(matrix_perm));
    std::memset(&matrix_perm_csc, 0, sizeof(matrix_perm_csc));
    std::memset(&symbolic, 0, sizeof(symbolic));

    try {
        result.ordering_ms = time_ms([&]() {
            check_sds(build_metis_graph(&matrix, &graph, 1), "build_metis_graph");
            check_sds(compute_metis_nd_ordering(&graph, perm.data(), inv_perm.data()),
                      "compute_metis_nd_ordering");
            check_sds(apply_symmetric_permutation(&matrix, perm.data(), &matrix_perm),
                      "apply_symmetric_permutation");
            check_sds(csr_to_csc(&matrix_perm, &matrix_perm_csc), "csr_to_csc");
        });

        result.symbolic_ms = time_ms([&]() {
            check_sds(symbolic_factorization_analyze(&matrix_perm_csc,
                                                     perm.data(),
                                                     inv_perm.data(),
                                                     &symbolic),
                      "symbolic_factorization_analyze");
        });

        result.num_fronts = symbolic.num_fronts;
        result.num_levels = symbolic.schedule.num_levels;
        result.max_front_size = max_front_size(symbolic);
        result.total_dense_entries = symbolic.storage.total_dense_entries;
        result.total_dense_bytes = symbolic.storage.total_dense_bytes;
        result.dense_entries_per_nnz =
            static_cast<double>(result.total_dense_entries) /
            std::max(1.0, static_cast<double>(system.matrix.nnz()));
        result.perm.assign(perm.begin(), perm.end());
        result.inv_perm.assign(inv_perm.begin(), inv_perm.end());
    } catch (...) {
        symbolic_factorization_destroy(&symbolic);
        csc_destroy(&matrix_perm_csc);
        csr_destroy(&matrix_perm);
        csr_destroy(&graph);
        csr_destroy(&matrix);
        throw;
    }

    symbolic_factorization_destroy(&symbolic);
    csc_destroy(&matrix_perm_csc);
    csr_destroy(&matrix_perm);
    csr_destroy(&graph);
    csr_destroy(&matrix);
    return result;
}

std::vector<int32_t> compute_metis_permutation(const LinearSystem& system)
{
    CSRMatrix matrix = make_project_csr(system.matrix);
    CSRMatrix graph;
    std::vector<int> perm(static_cast<std::size_t>(matrix.nrows), 0);
    std::vector<int> inv_perm(static_cast<std::size_t>(matrix.nrows), 0);

    std::memset(&graph, 0, sizeof(graph));

    try {
        check_sds(build_metis_graph(&matrix, &graph, 1), "build_metis_graph");
        check_sds(compute_metis_nd_ordering(&graph, perm.data(), inv_perm.data()),
                  "compute_metis_nd_ordering");
    } catch (...) {
        csr_destroy(&graph);
        csr_destroy(&matrix);
        throw;
    }

    csr_destroy(&graph);
    csr_destroy(&matrix);
    return std::vector<int32_t>(perm.begin(), perm.end());
}

CustomNumericResult run_custom_numeric_factorization(const LinearSystem& system,
                                                     const CliOptions& options)
{
    CustomNumericResult result;
    CSRMatrix matrix = make_project_csr(system.matrix);
    CSRMatrix graph;
    CSRMatrix matrix_perm;
    CSCMatrix matrix_perm_csc;
    SymbolicFactorization symbolic;
    NumericFactorization numeric;
    std::vector<int> perm(static_cast<std::size_t>(matrix.nrows), 0);
    std::vector<int> inv_perm(static_cast<std::size_t>(matrix.nrows), 0);
    std::vector<double> samples;
    auto log = [&](const char* text) {
        if (options.custom_progress) {
            std::cerr << "[custom-numeric] " << options.case_name
                      << ": " << text << std::endl;
        }
    };

    std::memset(&graph, 0, sizeof(graph));
    std::memset(&matrix_perm, 0, sizeof(matrix_perm));
    std::memset(&matrix_perm_csc, 0, sizeof(matrix_perm_csc));
    std::memset(&symbolic, 0, sizeof(symbolic));
    std::memset(&numeric, 0, sizeof(numeric));

    try {
        log("ordering/permutation/csr_to_csc begin");
        result.ordering_ms = time_ms([&]() {
            check_sds(build_metis_graph(&matrix, &graph, 1), "build_metis_graph");
            check_sds(compute_metis_nd_ordering(&graph, perm.data(), inv_perm.data()),
                      "compute_metis_nd_ordering");
            check_sds(apply_symmetric_permutation(&matrix, perm.data(), &matrix_perm),
                      "apply_symmetric_permutation");
            check_sds(csr_to_csc(&matrix_perm, &matrix_perm_csc), "csr_to_csc");
        });
        log("ordering/permutation/csr_to_csc done");

        log("symbolic begin");
        result.symbolic_ms = time_ms([&]() {
            check_sds(symbolic_factorization_analyze(&matrix_perm_csc,
                                                     perm.data(),
                                                     inv_perm.data(),
                                                     &symbolic),
                      "symbolic_factorization_analyze");
        });
        result.num_fronts = symbolic.num_fronts;
        result.num_levels = symbolic.schedule.num_levels;
        result.max_front_size = max_front_size(symbolic);
        result.total_dense_entries = symbolic.storage.total_dense_entries;
        if (options.custom_progress) {
            std::cerr << "[custom-numeric] " << options.case_name
                      << ": symbolic done fronts=" << symbolic.num_fronts
                      << " levels=" << symbolic.schedule.num_levels
                      << " max_front=" << result.max_front_size
                      << " storage_bytes=" << symbolic.storage.total_dense_bytes
                      << " entry_plan=" << symbolic.entry_assembly.num_entries
                      << " contrib_children="
                      << symbolic.contribution_assembly.num_child_fronts
                      << " contrib_update_indices="
                      << symbolic.contribution_assembly.total_update_indices
                      << std::endl;
        }

        log("numeric create begin");
        result.numeric_create_ms = time_ms([&]() {
            check_sds(numeric_factorization_create_cuda(&symbolic, &numeric),
                      "numeric_factorization_create_cuda");
        });
        result.total_dense_bytes = numeric.total_dense_bytes;
        if (options.custom_progress) {
            std::cerr << "[custom-numeric] " << options.case_name
                      << ": numeric create done total_dense_bytes="
                      << numeric.total_dense_bytes << std::endl;
        }

        NumericFactorizationOptions numeric_options;
        numeric_options.pivot_tol = 1e-12;
        numeric_options.enable_diagonal_perturbation = 0;
        numeric_options.perturb_value = 1e-12;
        numeric_options.enable_debug_print = 0;
        numeric_options.enable_timing = 1;
        numeric_options.use_cublas = 1;

        for (int32_t i = 0; i < options.warmup; ++i) {
            if (options.custom_progress) {
                std::cerr << "[custom-numeric] " << options.case_name
                          << ": warmup factorize " << i << " begin" << std::endl;
            }
            const int status = numeric_factorization_factorize_cuda(&matrix_perm_csc,
                                                                    &symbolic,
                                                                    &numeric_options,
                                                                    &numeric);
            if (status != SDS_OK) {
                result.status = status;
                if (options.custom_progress) {
                    std::cerr << "[custom-numeric] " << options.case_name
                              << ": warmup factorize failed status=" << status
                              << " first_failed_front=" << numeric.first_failed_front
                              << " first_failed_pivot=" << numeric.first_failed_pivot
                              << std::endl;
                }
                break;
            }
            if (options.custom_progress) {
                std::cerr << "[custom-numeric] " << options.case_name
                          << ": warmup factorize " << i
                          << " done factor_ms=" << numeric.factorization_ms
                          << std::endl;
            }
        }

        if (result.status == SDS_OK) {
            samples.reserve(static_cast<std::size_t>(options.repeats));
            for (int32_t i = 0; i < options.repeats; ++i) {
                if (options.custom_progress) {
                    std::cerr << "[custom-numeric] " << options.case_name
                              << ": repeat factorize " << i << " begin" << std::endl;
                }
                const int status = numeric_factorization_factorize_cuda(&matrix_perm_csc,
                                                                        &symbolic,
                                                                        &numeric_options,
                                                                        &numeric);
                if (status != SDS_OK) {
                    result.status = status;
                    if (options.custom_progress) {
                        std::cerr << "[custom-numeric] " << options.case_name
                                  << ": repeat factorize failed status=" << status
                                  << " first_failed_front="
                                  << numeric.first_failed_front
                                  << " first_failed_pivot="
                                  << numeric.first_failed_pivot << std::endl;
                    }
                    break;
                }
                samples.push_back(numeric.factorization_ms);
                if (options.custom_progress) {
                    std::cerr << "[custom-numeric] " << options.case_name
                              << ": repeat factorize " << i
                              << " done factor_ms=" << numeric.factorization_ms
                              << std::endl;
                }
            }
            result.factor_ms = make_stats(samples);
        }
    } catch (...) {
        numeric_factorization_destroy_cuda(&numeric);
        symbolic_factorization_destroy(&symbolic);
        csc_destroy(&matrix_perm_csc);
        csr_destroy(&matrix_perm);
        csr_destroy(&graph);
        csr_destroy(&matrix);
        throw;
    }

    numeric_factorization_destroy_cuda(&numeric);
    symbolic_factorization_destroy(&symbolic);
    csc_destroy(&matrix_perm_csc);
    csr_destroy(&matrix_perm);
    csr_destroy(&graph);
    csr_destroy(&matrix);
    return result;
}

std::vector<int32_t> get_cudss_int32_data(cudssHandle_t handle,
                                          cudssData_t data,
                                          cudssDataParam_t param)
{
    size_t needed = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, nullptr, 0, &needed));
    if (needed == 0) {
        return {};
    }
    if (needed % sizeof(int32_t) != 0) {
        throw std::runtime_error("cuDSS returned non-int32 byte count");
    }
    std::vector<int32_t> values(needed / sizeof(int32_t));
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, values.data(), needed, &written));
    if (written != needed) {
        throw std::runtime_error("cuDSS wrote an unexpected byte count");
    }
    return values;
}

std::vector<int64_t> get_cudss_int_values(cudssHandle_t handle,
                                          cudssData_t data,
                                          cudssDataParam_t param)
{
    size_t needed = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, nullptr, 0, &needed));
    if (needed == 0) {
        return {};
    }

    std::vector<unsigned char> raw(needed);
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(handle, data, param, raw.data(), raw.size(), &written));
    if (written != needed) {
        throw std::runtime_error("cuDSS wrote an unexpected byte count");
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

    throw std::runtime_error("cuDSS returned an unexpected integer byte count");
}

int64_t try_sum_cudss_int_data(cudssHandle_t handle,
                               cudssData_t data,
                               cudssDataParam_t param)
{
    size_t needed = 0;
    cudssStatus_t status = cudssDataGet(handle, data, param, nullptr, 0, &needed);
    if (status != CUDSS_STATUS_SUCCESS || needed == 0) {
        return -1;
    }

    std::vector<unsigned char> raw(needed);
    size_t written = 0;
    status = cudssDataGet(handle, data, param, raw.data(), raw.size(), &written);
    if (status != CUDSS_STATUS_SUCCESS || written != needed) {
        return -1;
    }

    int64_t total = 0;
    if (needed % sizeof(int64_t) == 0) {
        const auto* values = reinterpret_cast<const int64_t*>(raw.data());
        for (std::size_t i = 0; i < needed / sizeof(int64_t); ++i) {
            total += values[i];
        }
        return total;
    }
    if (needed % sizeof(int32_t) == 0) {
        const auto* values = reinterpret_cast<const int32_t*>(raw.data());
        for (std::size_t i = 0; i < needed / sizeof(int32_t); ++i) {
            total += values[i];
        }
        return total;
    }
    return -1;
}

void set_cudss_user_perm(cudssHandle_t handle,
                         cudssData_t data,
                         const std::vector<int32_t>& perm)
{
    cudssStatus_t status = cudssDataSet(handle,
                                        data,
                                        CUDSS_DATA_USER_PERM,
                                        const_cast<int32_t*>(perm.data()),
                                        perm.size() * sizeof(int32_t));
    if (status == CUDSS_STATUS_SUCCESS) {
        return;
    }

    std::vector<int64_t> perm64(perm.begin(), perm.end());
    CUDSS_CHECK(cudssDataSet(handle,
                             data,
                             CUDSS_DATA_USER_PERM,
                             perm64.data(),
                             perm64.size() * sizeof(int64_t)));
}

void write_json_int_array(std::ostream& out, const char* name, const std::vector<int32_t>& values)
{
    out << "  \"" << name << "\": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_reorder_json(const std::filesystem::path& path,
                        const CaseData& data,
                        const LinearSystem& system,
                        const CliOptions& options,
                        const std::vector<int32_t>& perm_row,
                        const std::vector<int32_t>& perm_col)
{
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open reorder JSON: " + path.string());
    }
    out << "{\n";
    out << "  \"case_name\": \"" << data.case_name << "\",\n";
    out << "  \"matrix_kind\": \"newton_jacobian_at_v0\",\n";
    out << "  \"cudss_phase\": \"CUDSS_PHASE_ANALYSIS\",\n";
    out << "  \"cudss_version\": \"" << cudss_version() << "\",\n";
    out << "  \"precision\": \"" << options.precision << "\",\n";
    out << "  \"rhs_mode\": \"" << system.rhs_mode << "\",\n";
    out << "  \"n_bus\": " << data.ybus.rows << ",\n";
    out << "  \"n_pv\": " << data.pv.size() << ",\n";
    out << "  \"n_pq\": " << data.pq.size() << ",\n";
    out << "  \"linear_dim\": " << system.matrix.rows << ",\n";
    out << "  \"linear_nnz\": " << system.matrix.nnz() << ",\n";
    out << "  \"perm_reorder_row_len\": " << perm_row.size() << ",\n";
    out << "  \"perm_reorder_col_len\": " << perm_col.size() << ",\n";
    write_json_int_array(out, "perm_reorder_row", perm_row);
    out << ",\n";
    write_json_int_array(out, "perm_reorder_col", perm_col);
    out << "\n}\n";
}

template <typename Func>
double time_ms(Func&& func)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    func();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename Func>
double time_ms_nvtx(const char* range_name, Func&& func)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    NvtxRange range(range_name);
    const auto start = Clock::now();
    func();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename T>
cudaDataType_t cuda_value_type();

template <>
cudaDataType_t cuda_value_type<double>()
{
    return CUDA_R_64F;
}

template <>
cudaDataType_t cuda_value_type<float>()
{
    return CUDA_R_32F;
}

template <typename T>
std::vector<T> convert_values(const std::vector<double>& values)
{
    std::vector<T> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), [](double value) {
        return static_cast<T>(value);
    });
    return converted;
}

struct CudssObjects {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr;
    cudssMatrix_t rhs = nullptr;
    cudssMatrix_t solution = nullptr;

    ~CudssObjects()
    {
        if (matrix) {
            (void)cudssMatrixDestroy(matrix);
        }
        if (rhs) {
            (void)cudssMatrixDestroy(rhs);
        }
        if (solution) {
            (void)cudssMatrixDestroy(solution);
        }
        if (data) {
            (void)cudssDataDestroy(handle, data);
        }
        if (config) {
            (void)cudssConfigDestroy(config);
        }
        if (handle) {
            (void)cudssDestroy(handle);
        }
    }
};

template <typename T>
TimingSample run_cudss_sample(const CsrMatrix& matrix,
                              int32_t* d_row_ptr,
                              int32_t* d_col_idx,
                              T* d_values,
                              const T* d_rhs_original,
                              T* d_rhs,
                              T* d_x,
                              const CliOptions& options,
                              const std::vector<int32_t>* user_perm)
{
    CUDA_CHECK(cudaMemcpy(d_rhs,
                          d_rhs_original,
                          static_cast<std::size_t>(matrix.rows) * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(matrix.cols) * sizeof(T)));

    CudssObjects cudss;
    CUDSS_CHECK(cudssCreate(&cudss.handle));
    if (options.enable_mt) {
        const char* threading_lib = options.threading_lib.empty() ? nullptr : options.threading_lib.c_str();
        CUDSS_CHECK(cudssSetThreadingLayer(cudss.handle, threading_lib));
    }
    CUDSS_CHECK(cudssConfigCreate(&cudss.config));
    CUDSS_CHECK(cudssDataCreate(cudss.handle, &cudss.data));
    if (options.cudss_perm == "metis") {
        if (!user_perm || static_cast<int32_t>(user_perm->size()) != matrix.rows) {
            throw std::runtime_error("METIS user permutation is missing or has the wrong size");
        }
        set_cudss_user_perm(cudss.handle, cudss.data, *user_perm);
    }
    CUDSS_CHECK(cudssMatrixCreateCsr(&cudss.matrix,
                                     matrix.rows,
                                     matrix.cols,
                                     static_cast<int64_t>(matrix.nnz()),
                                     d_row_ptr,
                                     nullptr,
                                     d_col_idx,
                                     d_values,
                                     CUDA_R_32I,
                                     cuda_value_type<T>(),
                                     CUDSS_MTYPE_GENERAL,
                                     CUDSS_MVIEW_FULL,
                                     CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&cudss.rhs,
                                    matrix.rows,
                                    1,
                                    matrix.rows,
                                    d_rhs,
                                    cuda_value_type<T>(),
                                    CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&cudss.solution,
                                    matrix.cols,
                                    1,
                                    matrix.cols,
                                    d_x,
                                    cuda_value_type<T>(),
                                    CUDSS_LAYOUT_COL_MAJOR));

    TimingSample sample;
    if (options.split_cudss_phases) {
        sample.analysis_ms = time_ms_nvtx("cudss.analysis", [&]() {
            sample.analysis_reordering_ms = time_ms_nvtx("cudss.analysis.reordering", [&]() {
                CUDSS_CHECK(cudssExecute(cudss.handle,
                                         CUDSS_PHASE_REORDERING,
                                         cudss.config,
                                         cudss.data,
                                         cudss.matrix,
                                         cudss.solution,
                                         cudss.rhs));
            });
            sample.analysis_symbolic_factorization_ms =
                time_ms_nvtx("cudss.analysis.symbolic_factorization", [&]() {
                CUDSS_CHECK(cudssExecute(cudss.handle,
                                         CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                                         cudss.config,
                                         cudss.data,
                                         cudss.matrix,
                                         cudss.solution,
                                         cudss.rhs));
            });
        });
    } else {
        sample.analysis_ms = time_ms_nvtx("cudss.analysis", [&]() {
            CUDSS_CHECK(cudssExecute(cudss.handle,
                                     CUDSS_PHASE_ANALYSIS,
                                     cudss.config,
                                     cudss.data,
                                     cudss.matrix,
                                     cudss.solution,
                                     cudss.rhs));
        });
    }
    sample.analysis_lu_nnz =
        try_sum_cudss_int_data(cudss.handle, cudss.data, CUDSS_DATA_LU_NNZ);
    sample.factor_ms = time_ms_nvtx("cudss.factorize", [&]() {
        CUDSS_CHECK(cudssExecute(cudss.handle,
                                 CUDSS_PHASE_FACTORIZATION,
                                 cudss.config,
                                 cudss.data,
                                 cudss.matrix,
                                 cudss.solution,
                                 cudss.rhs));
    });
    sample.factor_lu_nnz =
        try_sum_cudss_int_data(cudss.handle, cudss.data, CUDSS_DATA_LU_NNZ);
    if (options.split_cudss_phases) {
        sample.solve_ms = time_ms_nvtx("cudss.solve", [&]() {
            sample.solve_forward_ms = time_ms_nvtx("cudss.solve.forward", [&]() {
                CUDSS_CHECK(cudssExecute(cudss.handle,
                                         CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
                                         cudss.config,
                                         cudss.data,
                                         cudss.matrix,
                                         cudss.solution,
                                         cudss.rhs));
            });
            sample.solve_forward_to_rhs_ms = time_ms_nvtx("cudss.solve.forward_to_rhs", [&]() {
                CUDA_CHECK(cudaMemcpy(d_rhs,
                                      d_x,
                                      static_cast<std::size_t>(matrix.rows) * sizeof(T),
                                      cudaMemcpyDeviceToDevice));
            });
            sample.solve_backward_ms = time_ms_nvtx("cudss.solve.backward", [&]() {
                CUDSS_CHECK(cudssExecute(cudss.handle,
                                         CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
                                         cudss.config,
                                         cudss.data,
                                         cudss.matrix,
                                         cudss.solution,
                                         cudss.rhs));
            });
        });
    } else {
        sample.solve_ms = time_ms_nvtx("cudss.solve", [&]() {
            CUDSS_CHECK(cudssExecute(cudss.handle,
                                     CUDSS_PHASE_SOLVE,
                                     cudss.config,
                                     cudss.data,
                                     cudss.matrix,
                                     cudss.solution,
                                     cudss.rhs));
        });
    }
    return sample;
}

template <typename T>
BenchmarkResult run_benchmark_typed(const LinearSystem& system,
                                    const CliOptions& options,
                                    const std::vector<int32_t>* user_perm)
{
    const CsrMatrix& matrix = system.matrix;
    std::vector<T> values = convert_values<T>(matrix.values);
    std::vector<T> rhs = convert_values<T>(system.rhs);

    int32_t* d_row_ptr = nullptr;
    int32_t* d_col_idx = nullptr;
    T* d_values = nullptr;
    T* d_rhs_original = nullptr;
    T* d_rhs = nullptr;
    T* d_x = nullptr;

    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, static_cast<std::size_t>(matrix.rows + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, static_cast<std::size_t>(matrix.nnz()) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, static_cast<std::size_t>(matrix.nnz()) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs_original, static_cast<std::size_t>(matrix.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs, static_cast<std::size_t>(matrix.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(matrix.cols) * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr,
                          matrix.row_ptr.data(),
                          static_cast<std::size_t>(matrix.rows + 1) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx,
                          matrix.col_idx.data(),
                          static_cast<std::size_t>(matrix.nnz()) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,
                          values.data(),
                          static_cast<std::size_t>(matrix.nnz()) * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs_original,
                          rhs.data(),
                          static_cast<std::size_t>(matrix.rows) * sizeof(T),
                          cudaMemcpyHostToDevice));

    for (int32_t i = 0; i < options.warmup; ++i) {
        (void)run_cudss_sample<T>(matrix,
                                  d_row_ptr,
                                  d_col_idx,
                                  d_values,
                                  d_rhs_original,
                                  d_rhs,
                                  d_x,
                                  options,
                                  user_perm);
    }

    if (options.cuda_profiler_capture) {
        CUDA_CHECK(cudaProfilerStart());
    }

    std::vector<double> analysis_ms;
    std::vector<double> analysis_reordering_ms;
    std::vector<double> analysis_symbolic_factorization_ms;
    std::vector<double> factor_ms;
    std::vector<double> solve_ms;
    std::vector<double> solve_forward_ms;
    std::vector<double> solve_forward_to_rhs_ms;
    std::vector<double> solve_backward_ms;
    std::vector<double> total_ms;
    analysis_ms.reserve(static_cast<std::size_t>(options.repeats));
    analysis_reordering_ms.reserve(static_cast<std::size_t>(options.repeats));
    analysis_symbolic_factorization_ms.reserve(static_cast<std::size_t>(options.repeats));
    factor_ms.reserve(static_cast<std::size_t>(options.repeats));
    solve_ms.reserve(static_cast<std::size_t>(options.repeats));
    solve_forward_ms.reserve(static_cast<std::size_t>(options.repeats));
    solve_forward_to_rhs_ms.reserve(static_cast<std::size_t>(options.repeats));
    solve_backward_ms.reserve(static_cast<std::size_t>(options.repeats));
    total_ms.reserve(static_cast<std::size_t>(options.repeats));
    int64_t analysis_lu_nnz = -1;
    int64_t factor_lu_nnz = -1;
    for (int32_t i = 0; i < options.repeats; ++i) {
        const TimingSample sample = run_cudss_sample<T>(matrix,
                                                        d_row_ptr,
                                                        d_col_idx,
                                                        d_values,
                                                        d_rhs_original,
                                                        d_rhs,
                                                        d_x,
                                                        options,
                                                        user_perm);
        analysis_ms.push_back(sample.analysis_ms);
        factor_ms.push_back(sample.factor_ms);
        solve_ms.push_back(sample.solve_ms);
        total_ms.push_back(sample.analysis_ms + sample.factor_ms + sample.solve_ms);
        if (sample.analysis_lu_nnz >= 0) {
            analysis_lu_nnz = sample.analysis_lu_nnz;
        }
        if (sample.factor_lu_nnz >= 0) {
            factor_lu_nnz = sample.factor_lu_nnz;
        }
        if (std::isfinite(sample.analysis_reordering_ms)) {
            analysis_reordering_ms.push_back(sample.analysis_reordering_ms);
        }
        if (std::isfinite(sample.analysis_symbolic_factorization_ms)) {
            analysis_symbolic_factorization_ms.push_back(sample.analysis_symbolic_factorization_ms);
        }
        if (std::isfinite(sample.solve_forward_ms)) {
            solve_forward_ms.push_back(sample.solve_forward_ms);
        }
        if (std::isfinite(sample.solve_forward_to_rhs_ms)) {
            solve_forward_to_rhs_ms.push_back(sample.solve_forward_to_rhs_ms);
        }
        if (std::isfinite(sample.solve_backward_ms)) {
            solve_backward_ms.push_back(sample.solve_backward_ms);
        }
    }

    if (options.cuda_profiler_capture) {
        CUDA_CHECK(cudaProfilerStop());
    }

    std::vector<T> x_t(static_cast<std::size_t>(matrix.cols));
    CUDA_CHECK(cudaMemcpy(x_t.data(),
                          d_x,
                          static_cast<std::size_t>(matrix.cols) * sizeof(T),
                          cudaMemcpyDeviceToHost));

    std::vector<double> x(x_t.size());
    std::transform(x_t.begin(), x_t.end(), x.begin(), [](T value) {
        return static_cast<double>(value);
    });
    const std::vector<double> ax = matvec(matrix, x);
    std::vector<double> residual(ax.size(), 0.0);
    for (std::size_t i = 0; i < ax.size(); ++i) {
        residual[i] = ax[i] - system.rhs[i];
    }

    BenchmarkResult result;
    result.analysis_ms = make_stats(analysis_ms);
    result.analysis_reordering_ms = make_stats(analysis_reordering_ms);
    result.analysis_symbolic_factorization_ms =
        make_stats(analysis_symbolic_factorization_ms);
    result.factor_ms = make_stats(factor_ms);
    result.solve_ms = make_stats(solve_ms);
    result.solve_forward_ms = make_stats(solve_forward_ms);
    result.solve_forward_to_rhs_ms = make_stats(solve_forward_to_rhs_ms);
    result.solve_backward_ms = make_stats(solve_backward_ms);
    result.total_ms = make_stats(total_ms);
    result.residual_norm = norm2(residual);
    result.relative_residual =
        result.residual_norm / std::max(norm2(system.rhs), std::numeric_limits<double>::min());
    result.relative_error = options.rhs_mode == "synthetic"
        ? relative_error(x, system.x_ref)
        : std::numeric_limits<double>::quiet_NaN();
    result.cudss_analysis_lu_nnz = analysis_lu_nnz;
    result.cudss_factor_lu_nnz = factor_lu_nnz;
    if (factor_lu_nnz >= 0) {
        result.cudss_factor_lu_nnz_per_nnz =
            static_cast<double>(factor_lu_nnz) /
            std::max(1.0, static_cast<double>(matrix.nnz()));
    }

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_rhs_original));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));

    return result;
}

template <typename T>
void dump_cudss_reorder_typed(const CaseData& data,
                              const LinearSystem& system,
                              const CliOptions& options)
{
    const CsrMatrix& matrix = system.matrix;
    std::vector<T> values = convert_values<T>(matrix.values);
    std::vector<T> rhs = convert_values<T>(system.rhs);

    int32_t* d_row_ptr = nullptr;
    int32_t* d_col_idx = nullptr;
    T* d_values = nullptr;
    T* d_rhs = nullptr;
    T* d_x = nullptr;

    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, static_cast<std::size_t>(matrix.rows + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, static_cast<std::size_t>(matrix.nnz()) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, static_cast<std::size_t>(matrix.nnz()) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_rhs, static_cast<std::size_t>(matrix.rows) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(matrix.cols) * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr,
                          matrix.row_ptr.data(),
                          static_cast<std::size_t>(matrix.rows + 1) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx,
                          matrix.col_idx.data(),
                          static_cast<std::size_t>(matrix.nnz()) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,
                          values.data(),
                          static_cast<std::size_t>(matrix.nnz()) * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs,
                          rhs.data(),
                          static_cast<std::size_t>(matrix.rows) * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<std::size_t>(matrix.cols) * sizeof(T)));

    CudssObjects cudss;
    CUDSS_CHECK(cudssCreate(&cudss.handle));
    if (options.enable_mt) {
        const char* threading_lib = options.threading_lib.empty() ? nullptr : options.threading_lib.c_str();
        CUDSS_CHECK(cudssSetThreadingLayer(cudss.handle, threading_lib));
    }
    CUDSS_CHECK(cudssConfigCreate(&cudss.config));
    CUDSS_CHECK(cudssDataCreate(cudss.handle, &cudss.data));
    CUDSS_CHECK(cudssMatrixCreateCsr(&cudss.matrix,
                                     matrix.rows,
                                     matrix.cols,
                                     static_cast<int64_t>(matrix.nnz()),
                                     d_row_ptr,
                                     nullptr,
                                     d_col_idx,
                                     d_values,
                                     CUDA_R_32I,
                                     cuda_value_type<T>(),
                                     CUDSS_MTYPE_GENERAL,
                                     CUDSS_MVIEW_FULL,
                                     CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&cudss.rhs,
                                    matrix.rows,
                                    1,
                                    matrix.rows,
                                    d_rhs,
                                    cuda_value_type<T>(),
                                    CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&cudss.solution,
                                    matrix.cols,
                                    1,
                                    matrix.cols,
                                    d_x,
                                    cuda_value_type<T>(),
                                    CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssExecute(cudss.handle,
                             CUDSS_PHASE_ANALYSIS,
                             cudss.config,
                             cudss.data,
                             cudss.matrix,
                             cudss.solution,
                             cudss.rhs));
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<int32_t> perm_row =
        get_cudss_int32_data(cudss.handle, cudss.data, CUDSS_DATA_PERM_REORDER_ROW);
    const std::vector<int32_t> perm_col =
        get_cudss_int32_data(cudss.handle, cudss.data, CUDSS_DATA_PERM_REORDER_COL);
    write_reorder_json(options.dump_reorder_path, data, system, options, perm_row, perm_col);

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));
}

void dump_cudss_reorder(const CaseData& data,
                        const LinearSystem& system,
                        const CliOptions& options)
{
    if (options.precision == "fp64") {
        dump_cudss_reorder_typed<double>(data, system, options);
    } else {
        dump_cudss_reorder_typed<float>(data, system, options);
    }
}

BenchmarkResult run_benchmark(const LinearSystem& system,
                              const CliOptions& options,
                              const std::vector<int32_t>* user_perm)
{
    if (options.precision == "fp64") {
        return run_benchmark_typed<double>(system, options, user_perm);
    }
    return run_benchmark_typed<float>(system, options, user_perm);
}

std::string cudss_version()
{
    std::ostringstream oss;
    oss << CUDSS_VERSION_MAJOR << '.' << CUDSS_VERSION_MINOR << '.' << CUDSS_VERSION_PATCH;
    return oss.str();
}

void write_stats_csv(const Stats& stats)
{
    std::cout << stats.mean << ',' << stats.median << ',' << stats.min << ','
              << stats.max << ',' << stats.stddev;
}

void print_csv_header(bool include_symbolic, bool include_custom, bool include_cudss_subphases)
{
    std::cout
        << "case_name,precision,rhs_mode,cudss_version,n_bus,n_pv,n_pq,linear_dim,"
        << "linear_nnz,cudss_perm_mode,cudss_analysis_lu_nnz,cudss_factor_lu_nnz,"
        << "cudss_factor_lu_nnz_per_nnz,warmup,repeats,analysis_ms_mean,analysis_ms_median,"
        << "analysis_ms_min,analysis_ms_max,analysis_ms_stddev,factor_ms_mean,"
        << "factor_ms_median,factor_ms_min,factor_ms_max,factor_ms_stddev,"
        << "solve_ms_mean,solve_ms_median,solve_ms_min,solve_ms_max,solve_ms_stddev,"
        << "total_ms_mean,total_ms_median,total_ms_min,total_ms_max,total_ms_stddev,"
        << "residual_norm,relative_residual,relative_error";
    if (include_cudss_subphases) {
        std::cout
            << ",analysis_reordering_ms_mean,analysis_reordering_ms_median,"
            << "analysis_reordering_ms_min,analysis_reordering_ms_max,"
            << "analysis_reordering_ms_stddev,analysis_symbolic_factorization_ms_mean,"
            << "analysis_symbolic_factorization_ms_median,"
            << "analysis_symbolic_factorization_ms_min,"
            << "analysis_symbolic_factorization_ms_max,"
            << "analysis_symbolic_factorization_ms_stddev,solve_forward_ms_mean,"
            << "solve_forward_ms_median,solve_forward_ms_min,solve_forward_ms_max,"
            << "solve_forward_ms_stddev,solve_forward_to_rhs_ms_mean,"
            << "solve_forward_to_rhs_ms_median,solve_forward_to_rhs_ms_min,"
            << "solve_forward_to_rhs_ms_max,solve_forward_to_rhs_ms_stddev,"
            << "solve_backward_ms_mean,solve_backward_ms_median,"
            << "solve_backward_ms_min,solve_backward_ms_max,solve_backward_ms_stddev";
    }
    if (include_symbolic) {
        std::cout
            << ",metis_symbolic_status,metis_ordering_ms,metis_symbolic_ms,"
            << "metis_num_fronts,metis_num_levels,metis_max_front_size,"
            << "metis_total_dense_entries,metis_total_dense_bytes,"
            << "metis_dense_entries_per_nnz";
    }
    if (include_custom) {
        std::cout
            << ",custom_status,custom_ordering_ms,custom_symbolic_ms,"
            << "custom_numeric_create_ms,custom_factor_ms_mean,"
            << "custom_factor_ms_median,custom_factor_ms_min,custom_factor_ms_max,"
            << "custom_factor_ms_stddev,custom_num_fronts,custom_num_levels,"
            << "custom_max_front_size,custom_total_dense_entries,custom_total_dense_bytes";
    }
    std::cout << "\n";
}

void print_csv_row(const CaseData& data,
                   const LinearSystem& system,
                   const CliOptions& options,
                   const BenchmarkResult& result,
                   const MetisSymbolicResult* symbolic,
                   const CustomNumericResult* custom)
{
    std::cout << std::setprecision(12);
    std::cout << data.case_name << ','
              << options.precision << ','
              << system.rhs_mode << ','
              << cudss_version() << ','
              << data.ybus.rows << ','
              << data.pv.size() << ','
              << data.pq.size() << ','
              << system.matrix.rows << ','
              << system.matrix.nnz() << ','
              << options.cudss_perm << ','
              << result.cudss_analysis_lu_nnz << ','
              << result.cudss_factor_lu_nnz << ','
              << result.cudss_factor_lu_nnz_per_nnz << ','
              << options.warmup << ','
              << options.repeats << ',';
    write_stats_csv(result.analysis_ms);
    std::cout << ',';
    write_stats_csv(result.factor_ms);
    std::cout << ',';
    write_stats_csv(result.solve_ms);
    std::cout << ',';
    write_stats_csv(result.total_ms);
    std::cout << ','
              << result.residual_norm << ','
              << result.relative_residual << ','
              << result.relative_error;
    if (options.split_cudss_phases) {
        std::cout << ',';
        write_stats_csv(result.analysis_reordering_ms);
        std::cout << ',';
        write_stats_csv(result.analysis_symbolic_factorization_ms);
        std::cout << ',';
        write_stats_csv(result.solve_forward_ms);
        std::cout << ',';
        write_stats_csv(result.solve_forward_to_rhs_ms);
        std::cout << ',';
        write_stats_csv(result.solve_backward_ms);
    }
    if (symbolic) {
        std::cout << ','
                  << symbolic->status << ','
                  << symbolic->ordering_ms << ','
                  << symbolic->symbolic_ms << ','
                  << symbolic->num_fronts << ','
                  << symbolic->num_levels << ','
                  << symbolic->max_front_size << ','
                  << symbolic->total_dense_entries << ','
                  << symbolic->total_dense_bytes << ','
                  << symbolic->dense_entries_per_nnz;
    }
    if (custom) {
        std::cout << ','
                  << custom->status << ','
                  << custom->ordering_ms << ','
                  << custom->symbolic_ms << ','
                  << custom->numeric_create_ms << ',';
        write_stats_csv(custom->factor_ms);
        std::cout << ','
                  << custom->num_fronts << ','
                  << custom->num_levels << ','
                  << custom->max_front_size << ','
                  << custom->total_dense_entries << ','
                  << custom->total_dense_bytes;
    }
    std::cout << '\n';
}

void print_markdown(const CaseData& data,
                    const LinearSystem& system,
                    const CliOptions& options,
                    const BenchmarkResult& result,
                    const MetisSymbolicResult* symbolic,
                    const CustomNumericResult* custom)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "# cuDSS Power Flow Jacobian Benchmark\n\n";
    std::cout << "- case: " << data.case_name << "\n";
    std::cout << "- precision: " << options.precision << "\n";
    std::cout << "- rhs_mode: " << system.rhs_mode << "\n";
    std::cout << "- cudss_perm_mode: " << options.cudss_perm << "\n";
    std::cout << "- n_bus: " << data.ybus.rows << "\n";
    std::cout << "- linear_dim: " << system.matrix.rows << "\n";
    std::cout << "- linear_nnz: " << system.matrix.nnz() << "\n";
    std::cout << "- warmup/repeats: " << options.warmup << "/" << options.repeats << "\n\n";
    std::cout << "| phase | mean_ms | median_ms | min_ms | max_ms | stddev_ms |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|\n";
    auto row = [](const char* name, const Stats& stats) {
        std::cout << "| " << name << " | " << stats.mean << " | " << stats.median << " | "
                  << stats.min << " | " << stats.max << " | " << stats.stddev << " |\n";
    };
    row("analysis", result.analysis_ms);
    row("factorization", result.factor_ms);
    row("solve", result.solve_ms);
    row("total", result.total_ms);
    std::cout << "\n";
    if (options.split_cudss_phases) {
        std::cout << "## cuDSS Subphase Timings\n\n";
        std::cout << "| subphase | mean_ms | median_ms | min_ms | max_ms | stddev_ms |\n";
        std::cout << "|---|---:|---:|---:|---:|---:|\n";
        row("analysis.reordering", result.analysis_reordering_ms);
        row("analysis.symbolic_factorization", result.analysis_symbolic_factorization_ms);
        row("solve.forward", result.solve_forward_ms);
        row("solve.forward_to_rhs", result.solve_forward_to_rhs_ms);
        row("solve.backward", result.solve_backward_ms);
        std::cout << "\n";
    }
    std::cout << "| residual | value |\n|---|---:|\n";
    std::cout << "| norm2 | " << std::scientific << result.residual_norm << " |\n";
    std::cout << "| relative_norm2 | " << result.relative_residual << " |\n";
    std::cout << "| relative_error | " << result.relative_error << " |\n";
    std::cout << "\n";
    std::cout << "| cuDSS fill metric | value |\n|---|---:|\n";
    std::cout << "| analysis_lu_nnz | " << result.cudss_analysis_lu_nnz << " |\n";
    std::cout << "| factor_lu_nnz | " << result.cudss_factor_lu_nnz << " |\n";
    std::cout << "| factor_lu_nnz_per_matrix_nnz | " << std::fixed
              << result.cudss_factor_lu_nnz_per_nnz << " |\n";

    if (symbolic) {
        std::cout << "\n";
        std::cout << "## METIS Symbolic Estimate\n\n";
        std::cout << "| item | value |\n|---|---:|\n";
        std::cout << "| status | " << symbolic->status << " |\n";
        std::cout << "| ordering_ms | " << symbolic->ordering_ms << " |\n";
        std::cout << "| symbolic_ms | " << symbolic->symbolic_ms << " |\n";
        std::cout << "| num_fronts | " << symbolic->num_fronts << " |\n";
        std::cout << "| num_levels | " << symbolic->num_levels << " |\n";
        std::cout << "| max_front_size | " << symbolic->max_front_size << " |\n";
        std::cout << "| total_dense_entries | " << symbolic->total_dense_entries << " |\n";
        std::cout << "| total_dense_bytes | " << symbolic->total_dense_bytes << " |\n";
        std::cout << "| dense_entries_per_matrix_nnz | "
                  << symbolic->dense_entries_per_nnz << " |\n";
    }

    if (custom) {
        std::cout << "\n";
        std::cout << "## Current Numeric Factorization Path\n\n";
        std::cout << "| item | value |\n|---|---:|\n";
        std::cout << "| status | " << custom->status << " |\n";
        std::cout << "| ordering_ms | " << std::fixed << custom->ordering_ms << " |\n";
        std::cout << "| symbolic_ms | " << custom->symbolic_ms << " |\n";
        std::cout << "| numeric_create_ms | " << custom->numeric_create_ms << " |\n";
        std::cout << "| factor_ms_mean | " << custom->factor_ms.mean << " |\n";
        std::cout << "| factor_ms_median | " << custom->factor_ms.median << " |\n";
        std::cout << "| num_fronts | " << custom->num_fronts << " |\n";
        std::cout << "| num_levels | " << custom->num_levels << " |\n";
        std::cout << "| max_front_size | " << custom->max_front_size << " |\n";
        std::cout << "| total_dense_entries | " << custom->total_dense_entries << " |\n";
        std::cout << "| total_dense_bytes | " << custom->total_dense_bytes << " |\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        const CaseData data = load_case(options);
        const LinearSystem system = build_linear_system(data, options);
        if (!options.dump_reorder_path.empty()) {
            dump_cudss_reorder(data, system, options);
            return 0;
        }
        MetisSymbolicResult symbolic_result;
        MetisSymbolicResult* symbolic_ptr = nullptr;
        std::vector<int32_t> metis_perm;
        if (options.metis_symbolic) {
            symbolic_result = run_metis_symbolic_analysis(system);
            symbolic_ptr = &symbolic_result;
            metis_perm = symbolic_result.perm;
        } else if (options.cudss_perm == "metis") {
            metis_perm = compute_metis_permutation(system);
        }
        const std::vector<int32_t>* user_perm =
            options.cudss_perm == "metis" ? &metis_perm : nullptr;
        const BenchmarkResult result = run_benchmark(system, options, user_perm);
        CustomNumericResult custom_result;
        CustomNumericResult* custom_ptr = nullptr;
        if (options.custom_numeric) {
            custom_result = run_custom_numeric_factorization(system, options);
            custom_ptr = &custom_result;
        }
        if (options.csv) {
            print_csv_header(symbolic_ptr != nullptr,
                             options.custom_numeric,
                             options.split_cudss_phases);
            print_csv_row(data, system, options, result, symbolic_ptr, custom_ptr);
        } else {
            print_markdown(data, system, options, result, symbolic_ptr, custom_ptr);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cudss_pf_benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
