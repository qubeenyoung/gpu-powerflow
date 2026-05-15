#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace linbench {

using Clock = std::chrono::steady_clock;

struct CliOptions {
    std::filesystem::path matrix;
    std::filesystem::path rhs;
    std::filesystem::path xref;
    std::filesystem::path meta;
    std::string dtype = "fp64";
    int repeats = 10;
    int warmup = 3;
    std::filesystem::path out;
    std::filesystem::path config;
};

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct Meta {
    std::string case_name;
    int iteration = -1;
    int matrix_rows = 0;
    int matrix_cols = 0;
    int nnz = 0;
};

struct Stats {
    double mean = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
};

struct Result {
    std::string solver_name;
    std::string solver_version;
    std::string library_path;
    std::string build_status = "ok";
    std::string dtype = "fp64";
    std::string case_name;
    int iteration = -1;
    int matrix_rows = 0;
    int matrix_cols = 0;
    int nnz = 0;
    int repeat_count = 0;
    int warmup_count = 0;
    double load_ms = 0.0;
    double format_convert_ms = 0.0;
    double h2d_ms = 0.0;
    double analysis_ms = 0.0;
    double factorization_ms = 0.0;
    double solve_ms = 0.0;
    double d2h_ms = 0.0;
    double total_solver_ms = 0.0;
    double total_end_to_end_ms = 0.0;
    double peak_gpu_memory_mb = std::numeric_limits<double>::quiet_NaN();
    double relative_residual_2 = std::numeric_limits<double>::quiet_NaN();
    double relative_error_to_x_ref_2 = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
    int num_iterations = -1;
    std::string gpu_resident_after_initial_load = "unknown";
    std::string notes;
    std::map<std::string, Stats> timing_stats;
    std::map<std::string, std::string> extra_strings;
    std::map<std::string, double> extra_numbers;
};

inline double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline CliOptions parse_cli(int argc, char** argv)
{
    CliOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--matrix") {
            opt.matrix = need_value(arg);
        } else if (arg == "--rhs") {
            opt.rhs = need_value(arg);
        } else if (arg == "--xref") {
            opt.xref = need_value(arg);
        } else if (arg == "--meta") {
            opt.meta = need_value(arg);
        } else if (arg == "--dtype") {
            opt.dtype = need_value(arg);
        } else if (arg == "--repeats") {
            opt.repeats = std::stoi(need_value(arg));
        } else if (arg == "--warmup") {
            opt.warmup = std::stoi(need_value(arg));
        } else if (arg == "--out") {
            opt.out = need_value(arg);
        } else if (arg == "--config") {
            opt.config = need_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(
                "usage: --matrix J.mtx --rhs rhs.txt --xref x_ref.txt --meta meta.json "
                "--dtype fp64|fp32 --repeats N --warmup N --out result.json");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.matrix.empty() || opt.rhs.empty() || opt.xref.empty() || opt.meta.empty() || opt.out.empty()) {
        throw std::runtime_error("matrix, rhs, xref, meta, and out are required");
    }
    if (opt.dtype != "fp64" && opt.dtype != "fp32") {
        throw std::runtime_error("dtype must be fp64 or fp32");
    }
    if (opt.repeats < 1 || opt.warmup < 0) {
        throw std::runtime_error("repeats must be positive and warmup must be nonnegative");
    }
    return opt;
}

inline std::string read_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string json_string_value(const std::string& text, const std::string& key)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return m[1].str();
    }
    return "";
}

inline int json_int_value(const std::string& text, const std::string& key)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return std::stoi(m[1].str());
    }
    return 0;
}

inline Meta load_meta(const std::filesystem::path& path)
{
    const std::string text = read_file(path);
    Meta meta;
    meta.case_name = json_string_value(text, "case_name");
    meta.iteration = json_int_value(text, "iteration");
    meta.matrix_rows = json_int_value(text, "matrix_rows");
    meta.matrix_cols = json_int_value(text, "matrix_cols");
    meta.nnz = json_int_value(text, "nnz");
    return meta;
}

inline std::vector<double> load_vector(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open vector " + path.string());
    }
    std::vector<double> values;
    double v = 0.0;
    while (in >> v) {
        values.push_back(v);
    }
    if (values.empty()) {
        throw std::runtime_error("empty vector " + path.string());
    }
    return values;
}

inline CsrMatrix load_matrix_market_csr(const std::filesystem::path& path, double* format_convert_ms)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open Matrix Market file " + path.string());
    }
    std::string banner;
    std::getline(in, banner);
    if (banner.find("MatrixMarket") == std::string::npos || banner.find("coordinate") == std::string::npos) {
        throw std::runtime_error("expected coordinate Matrix Market file " + path.string());
    }
    const bool symmetric = banner.find("symmetric") != std::string::npos;
    std::string line;
    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("missing Matrix Market size line " + path.string());
        }
    } while (!line.empty() && line[0] == '%');
    std::istringstream dims(line);
    int rows = 0;
    int cols = 0;
    int entries = 0;
    dims >> rows >> cols >> entries;
    if (rows <= 0 || cols <= 0 || entries < 0) {
        throw std::runtime_error("invalid Matrix Market dimensions in " + path.string());
    }

    struct Entry {
        int row;
        int col;
        double value;
    };
    std::vector<Entry> coo;
    coo.reserve(static_cast<std::size_t>(entries) * (symmetric ? 2 : 1));
    for (int k = 0; k < entries; ++k) {
        int i = 0;
        int j = 0;
        double value = 0.0;
        in >> i >> j >> value;
        --i;
        --j;
        coo.push_back({i, j, value});
        if (symmetric && i != j) {
            coo.push_back({j, i, value});
        }
    }

    const auto t0 = Clock::now();
    std::sort(coo.begin(), coo.end(), [](const Entry& a, const Entry& b) {
        return a.row == b.row ? a.col < b.col : a.row < b.row;
    });
    std::vector<Entry> summed;
    summed.reserve(coo.size());
    for (const Entry& e : coo) {
        if (!summed.empty() && summed.back().row == e.row && summed.back().col == e.col) {
            summed.back().value += e.value;
        } else {
            summed.push_back(e);
        }
    }
    CsrMatrix csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.nnz = static_cast<int>(summed.size());
    csr.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    csr.col_idx.resize(summed.size());
    csr.values.resize(summed.size());
    for (const Entry& e : summed) {
        if (e.row < 0 || e.row >= rows || e.col < 0 || e.col >= cols) {
            throw std::runtime_error("Matrix Market entry out of bounds in " + path.string());
        }
        ++csr.row_ptr[static_cast<std::size_t>(e.row + 1)];
    }
    for (int r = 0; r < rows; ++r) {
        csr.row_ptr[static_cast<std::size_t>(r + 1)] += csr.row_ptr[static_cast<std::size_t>(r)];
    }
    std::vector<int> cursor = csr.row_ptr;
    for (const Entry& e : summed) {
        const int dst = cursor[static_cast<std::size_t>(e.row)]++;
        csr.col_idx[static_cast<std::size_t>(dst)] = e.col;
        csr.values[static_cast<std::size_t>(dst)] = e.value;
    }
    const auto t1 = Clock::now();
    if (format_convert_ms) {
        *format_convert_ms = elapsed_ms(t0, t1);
    }
    return csr;
}

inline Stats make_stats(std::vector<double> values)
{
    Stats s;
    if (values.empty()) {
        return s;
    }
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();
    s.median = values.size() % 2 == 0
        ? 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2])
        : values[values.size() / 2];
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double accum = 0.0;
    for (double v : values) {
        const double d = v - s.mean;
        accum += d * d;
    }
    s.stddev = std::sqrt(accum / static_cast<double>(values.size()));
    return s;
}

inline double norm2(const std::vector<double>& x)
{
    long double acc = 0.0;
    for (double v : x) {
        acc += static_cast<long double>(v) * static_cast<long double>(v);
    }
    return static_cast<double>(std::sqrt(acc));
}

inline std::vector<double> residual(const CsrMatrix& A, const std::vector<double>& x, const std::vector<double>& rhs)
{
    if (static_cast<int>(x.size()) != A.cols || static_cast<int>(rhs.size()) != A.rows) {
        throw std::runtime_error("dimension mismatch while computing residual");
    }
    std::vector<double> r(rhs.size(), 0.0);
    for (int i = 0; i < A.rows; ++i) {
        double sum = 0.0;
        for (int p = A.row_ptr[static_cast<std::size_t>(i)]; p < A.row_ptr[static_cast<std::size_t>(i + 1)]; ++p) {
            sum += A.values[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(A.col_idx[static_cast<std::size_t>(p)])];
        }
        r[static_cast<std::size_t>(i)] = sum - rhs[static_cast<std::size_t>(i)];
    }
    return r;
}

inline double relative_error(const std::vector<double>& x, const std::vector<double>& ref)
{
    if (x.size() != ref.size()) {
        throw std::runtime_error("dimension mismatch while computing relative error");
    }
    std::vector<double> d(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        d[i] = x[i] - ref[i];
    }
    return norm2(d) / std::max(norm2(ref), std::numeric_limits<double>::min());
}

inline std::string escape_json(const std::string& text)
{
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

inline std::string json_number(double value)
{
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream ss;
    ss << std::setprecision(17) << value;
    return ss.str();
}

inline void write_stats_object(std::ostream& out, const Stats& s)
{
    out << "{"
        << "\"mean\":" << json_number(s.mean)
        << ",\"median\":" << json_number(s.median)
        << ",\"min\":" << json_number(s.min)
        << ",\"max\":" << json_number(s.max)
        << ",\"stddev\":" << json_number(s.stddev)
        << "}";
}

inline void write_result_json(const std::filesystem::path& path, const Result& r)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write result " + path.string());
    }
    out << "{\n";
    auto field_s = [&](const std::string& key, const std::string& value, bool comma = true) {
        out << "  \"" << key << "\": \"" << escape_json(value) << "\"" << (comma ? "," : "") << "\n";
    };
    auto field_i = [&](const std::string& key, int value, bool comma = true) {
        out << "  \"" << key << "\": " << value << (comma ? "," : "") << "\n";
    };
    auto field_d = [&](const std::string& key, double value, bool comma = true) {
        out << "  \"" << key << "\": " << json_number(value) << (comma ? "," : "") << "\n";
    };
    auto field_b = [&](const std::string& key, bool value, bool comma = true) {
        out << "  \"" << key << "\": " << (value ? "true" : "false") << (comma ? "," : "") << "\n";
    };
    field_s("solver_name", r.solver_name);
    field_s("solver_version", r.solver_version);
    field_s("library_path", r.library_path);
    field_s("build_status", r.build_status);
    field_s("dtype", r.dtype);
    field_s("case_name", r.case_name);
    field_i("iteration", r.iteration);
    field_i("matrix_rows", r.matrix_rows);
    field_i("matrix_cols", r.matrix_cols);
    field_i("nnz", r.nnz);
    field_i("repeat_count", r.repeat_count);
    field_i("warmup_count", r.warmup_count);
    field_d("load_ms", r.load_ms);
    field_d("format_convert_ms", r.format_convert_ms);
    field_d("h2d_ms", r.h2d_ms);
    field_d("analysis_ms", r.analysis_ms);
    field_d("factorization_ms", r.factorization_ms);
    field_d("solve_ms", r.solve_ms);
    field_d("d2h_ms", r.d2h_ms);
    field_d("total_solver_ms", r.total_solver_ms);
    field_d("total_end_to_end_ms", r.total_end_to_end_ms);
    field_d("peak_gpu_memory_mb", r.peak_gpu_memory_mb);
    field_d("relative_residual_2", r.relative_residual_2);
    field_d("relative_error_to_x_ref_2", r.relative_error_to_x_ref_2);
    field_b("converged", r.converged);
    field_i("num_iterations", r.num_iterations);
    field_s("gpu_resident_after_initial_load", r.gpu_resident_after_initial_load);
    field_s("notes", r.notes);
    out << "  \"timing_stats\": {";
    bool first = true;
    for (const auto& [key, stats] : r.timing_stats) {
        out << (first ? "\n" : ",\n") << "    \"" << escape_json(key) << "\": ";
        write_stats_object(out, stats);
        first = false;
    }
    out << (first ? "}" : "\n  }");
    for (const auto& [key, value] : r.extra_strings) {
        out << ",\n  \"" << escape_json(key) << "\": \"" << escape_json(value) << "\"";
    }
    for (const auto& [key, value] : r.extra_numbers) {
        out << ",\n  \"" << escape_json(key) << "\": " << json_number(value);
    }
    out << "\n}\n";
}

}  // namespace linbench
