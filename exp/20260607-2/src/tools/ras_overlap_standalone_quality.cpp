#include "cupf_minimal/case_data.hpp"
#include "cupf_minimal/direct_cudss_solver.hpp"

#include "cuiter/common/cuda_utils.hpp"
#include "cuiter/solver/gmres_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path case_root = "/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps";
    std::filesystem::path jf_root = "raw/cupf_jf_dumps";
    std::filesystem::path output = "results/ras_overlap_standalone_quality.csv";
    std::vector<std::string> cases = {
        "case2383wp",
        "case3120sp",
        "case9241pegase",
        "case13659pegase",
        "case6468rte",
    };
    std::vector<int32_t> block_sizes = {8, 16};
    int32_t iteration = 1;
};

struct Row {
    std::string case_name;
    int32_t iteration = 1;
    std::string preconditioner;
    int32_t overlap = 0;
    int32_t block_size = 0;
    int32_t n = 0;
    int32_t nnz = 0;
    double true_linear_rel_res = 0.0;
    double true_linear_abs_res = 0.0;
    double dx_error_ratio = 0.0;
    double dx_cosine = 0.0;
    double theta_error_ratio = 0.0;
    double vmag_error_ratio = 0.0;
    double setup_ms = 0.0;
    double solve_ms = 0.0;
    double preconditioner_apply_ms = 0.0;
    double ras_setup_ms = 0.0;
    double ras_apply_ms = 0.0;
    std::string status = "ok";
    std::string error_message;
};

std::vector<std::string> split_list(const std::string& value)
{
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<int32_t> split_int_list(const std::string& value)
{
    std::vector<int32_t> out;
    for (const std::string& item : split_list(value)) {
        out.push_back(std::stoi(item));
    }
    if (out.empty()) {
        throw std::runtime_error("empty integer list");
    }
    return out;
}

void expect_token(std::istream& in, const char* expected, const std::filesystem::path& path)
{
    std::string token;
    in >> token;
    if (token != expected) {
        throw std::runtime_error("expected token '" + std::string(expected) + "' in " + path.string());
    }
}

cuiter::CsrMatrix load_cupf_csr_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open cuPF CSR dump: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "csr_matrix") {
        throw std::runtime_error("not a CSR dump: " + path.string());
    }

    cuiter::CsrMatrix matrix;
    int32_t nnz = 0;
    expect_token(in, "rows", path);
    in >> matrix.rows;
    expect_token(in, "cols", path);
    in >> matrix.cols;
    expect_token(in, "nnz", path);
    in >> nnz;
    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows + 1));
    matrix.col_idx.resize(static_cast<std::size_t>(nnz));
    matrix.values.resize(static_cast<std::size_t>(nnz));
    expect_token(in, "row_ptr", path);
    for (int32_t i = 0; i <= matrix.rows; ++i) {
        in >> matrix.row_ptr[static_cast<std::size_t>(i)];
    }
    expect_token(in, "col_idx", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.col_idx[static_cast<std::size_t>(i)];
    }
    expect_token(in, "values", path);
    for (int32_t i = 0; i < nnz; ++i) {
        in >> matrix.values[static_cast<std::size_t>(i)];
    }
    if (!in || matrix.rows != matrix.cols || matrix.row_ptr.back() != nnz) {
        throw std::runtime_error("malformed CSR dump: " + path.string());
    }
    return matrix;
}

std::vector<double> load_cupf_vector_dump(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open cuPF vector dump: " + path.string());
    }
    expect_token(in, "type", path);
    std::string type;
    in >> type;
    if (type != "vector") {
        throw std::runtime_error("not a vector dump: " + path.string());
    }
    expect_token(in, "size", path);
    int32_t n = 0;
    in >> n;
    expect_token(in, "values", path);
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        int32_t index = 0;
        double value = 0.0;
        in >> index >> value;
        values[static_cast<std::size_t>(index)] = value;
    }
    if (!in) {
        throw std::runtime_error("malformed vector dump: " + path.string());
    }
    return values;
}

std::filesystem::path jacobian_path(const std::filesystem::path& jf_root,
                                    const std::string& case_name,
                                    int32_t iteration)
{
    const std::filesystem::path case_dir = jf_root / case_name;
    const std::filesystem::path direct = case_dir / ("J" + std::to_string(iteration) + ".txt");
    if (std::filesystem::exists(direct)) {
        return direct;
    }
    return case_dir / "repeat_00" / ("jacobian_iter" + std::to_string(iteration) + ".txt");
}

std::filesystem::path rhs_path(const std::filesystem::path& jf_root,
                               const std::string& case_name,
                               int32_t iteration)
{
    const std::filesystem::path case_dir = jf_root / case_name;
    const std::vector<std::filesystem::path> candidates = {
        case_dir / ("F" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_iter" + std::to_string(iteration) + ".txt"),
        case_dir / "repeat_00" / ("residual_before_update_iter" + std::to_string(iteration) + ".txt"),
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return candidates.front();
}

double norm2(const std::vector<double>& values, int32_t begin = 0, int32_t end = -1)
{
    if (end < 0) {
        end = static_cast<int32_t>(values.size());
    }
    double sum = 0.0;
    for (int32_t i = begin; i < end; ++i) {
        sum += values[static_cast<std::size_t>(i)] * values[static_cast<std::size_t>(i)];
    }
    return std::sqrt(sum);
}

std::vector<double> spmv(const cuiter::CsrMatrix& matrix, const std::vector<double>& x)
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

std::vector<double> residual(const cuiter::CsrMatrix& matrix,
                             const std::vector<double>& rhs,
                             const std::vector<double>& x)
{
    std::vector<double> r = rhs;
    const std::vector<double> ax = spmv(matrix, x);
    for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] -= ax[i];
    }
    return r;
}

double cosine(const std::vector<double>& a, const std::vector<double>& b)
{
    double dot = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    const double denom = norm2(a) * norm2(b);
    return denom > 0.0 ? dot / denom : 0.0;
}

std::vector<double> solve_cudss(const cuiter::CsrMatrix& matrix, const std::vector<double>& rhs)
{
    cuiter::DeviceBuffer<int32_t> d_row_ptr;
    cuiter::DeviceBuffer<int32_t> d_col_idx;
    cuiter::DeviceBuffer<double> d_values;
    cuiter::DeviceBuffer<double> d_rhs;
    cuiter::DeviceBuffer<double> d_x;
    d_row_ptr.assign(matrix.row_ptr.data(), matrix.row_ptr.size());
    d_col_idx.assign(matrix.col_idx.data(), matrix.col_idx.size());
    d_values.assign(matrix.values.data(), matrix.values.size());
    d_rhs.assign(rhs.data(), rhs.size());
    d_x.resize(rhs.size());

    cupf_minimal::DirectCudssSolver solver;
    solver.initialize(matrix, d_row_ptr.data(), d_col_idx.data(), d_values.data(), d_rhs.data(), d_x.data());
    solver.analyze();
    solver.factorize();
    solver.solve();

    std::vector<double> x(rhs.size(), 0.0);
    d_x.copy_to(x.data(), x.size());
    return x;
}

Row solve_iterative(const std::string& case_name,
                    int32_t iteration,
                    const cuiter::CsrMatrix& matrix,
                    const std::vector<double>& rhs,
                    const std::vector<double>& x_cudss,
                    int32_t n_pvpq,
                    int32_t block_size,
                    const std::string& preconditioner)
{
    cuiter::GmresSolverOptions options;
    options.max_iters = 2;
    options.restart = 1;
    options.rel_tolerance = 0.0;
    options.abs_tolerance = 0.0;
    options.preconditioner = preconditioner;
    options.block_size = block_size;
    options.use_fp32_preconditioner = true;
    options.use_bicgstab_fixed_path = true;
    options.use_bicgstab_fused_fixed2 = true;
    options.block_jacobi_apply = cuiter::BlockJacobiApplyMode::InverseGemv;

    cuiter::GmresSolver solver(options);
    solver.analyze(matrix);
    const cuiter::LinearSolveResult result = solver.solve(matrix.values, rhs);
    const std::vector<double>& x = result.solution;
    std::vector<double> err(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        err[i] = x[i] - x_cudss[i];
    }
    const std::vector<double> r = residual(matrix, rhs, x);

    Row row;
    row.case_name = case_name;
    row.iteration = iteration;
    row.preconditioner = preconditioner;
    row.overlap = preconditioner == "ras_overlap1" ? 1 : 0;
    row.block_size = block_size;
    row.n = matrix.rows;
    row.nnz = matrix.nnz();
    row.true_linear_abs_res = norm2(r);
    row.true_linear_rel_res = row.true_linear_abs_res / std::max(norm2(rhs), std::numeric_limits<double>::min());
    row.dx_error_ratio = norm2(err) / std::max(norm2(x_cudss), std::numeric_limits<double>::min());
    row.dx_cosine = cosine(x, x_cudss);
    row.theta_error_ratio =
        norm2(err, 0, n_pvpq) / std::max(norm2(x_cudss, 0, n_pvpq), std::numeric_limits<double>::min());
    row.vmag_error_ratio =
        norm2(err, n_pvpq, matrix.rows) /
        std::max(norm2(x_cudss, n_pvpq, matrix.rows), std::numeric_limits<double>::min());
    row.setup_ms = 1000.0 * result.timings.setup_total_seconds;
    row.solve_ms = 1000.0 * result.timings.solve_total_seconds;
    row.preconditioner_apply_ms = 1000.0 * result.timings.preconditioner_apply_seconds;
    row.ras_setup_ms = 1000.0 * result.timings.ras_setup_seconds;
    row.ras_apply_ms = 1000.0 * result.timings.ras_apply_seconds;
    return row;
}

CliOptions parse_args(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-root" && i + 1 < argc) {
            options.case_root = argv[++i];
        } else if (arg == "--jf-root" && i + 1 < argc) {
            options.jf_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.cases = split_list(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            options.block_sizes = split_int_list(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            options.iteration = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void ensure_parent_dir(const std::filesystem::path& path)
{
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_csv(const std::filesystem::path& path, const std::vector<Row>& rows)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    out << std::setprecision(12);
    out << "case,iteration,preconditioner,overlap,block_size,n,nnz,true_linear_rel_res,"
           "true_linear_abs_res,dx_error_ratio,dx_cosine,theta_error_ratio,vmag_error_ratio,"
           "setup_ms,solve_ms,preconditioner_apply_ms,ras_setup_ms,ras_apply_ms,status,error_message\n";
    for (const Row& row : rows) {
        out << row.case_name << ','
            << row.iteration << ','
            << row.preconditioner << ','
            << row.overlap << ','
            << row.block_size << ','
            << row.n << ','
            << row.nnz << ','
            << row.true_linear_rel_res << ','
            << row.true_linear_abs_res << ','
            << row.dx_error_ratio << ','
            << row.dx_cosine << ','
            << row.theta_error_ratio << ','
            << row.vmag_error_ratio << ','
            << row.setup_ms << ','
            << row.solve_ms << ','
            << row.preconditioner_apply_ms << ','
            << row.ras_setup_ms << ','
            << row.ras_apply_ms << ','
            << row.status << ','
            << row.error_message << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_args(argc, argv);
        std::vector<Row> rows;
        for (const std::string& case_name : options.cases) {
            const cupf_minimal::DumpCaseData case_data =
                cupf_minimal::load_dump_case(options.case_root / case_name);
            const int32_t n_pvpq = static_cast<int32_t>(case_data.pv.size() + case_data.pq.size());
            const cuiter::CsrMatrix matrix =
                load_cupf_csr_dump(jacobian_path(options.jf_root, case_name, options.iteration));
            const std::vector<double> rhs =
                load_cupf_vector_dump(rhs_path(options.jf_root, case_name, options.iteration));
            const std::vector<double> x_cudss = solve_cudss(matrix, rhs);
            for (int32_t block_size : options.block_sizes) {
                for (const std::string& preconditioner : {"metis_block_jacobi", "ras_overlap1"}) {
                    try {
                        rows.push_back(solve_iterative(case_name,
                                                       options.iteration,
                                                       matrix,
                                                       rhs,
                                                       x_cudss,
                                                       n_pvpq,
                                                       block_size,
                                                       preconditioner));
                    } catch (const std::exception& ex) {
                        Row row;
                        row.case_name = case_name;
                        row.iteration = options.iteration;
                        row.preconditioner = preconditioner;
                        row.overlap = preconditioner == "ras_overlap1" ? 1 : 0;
                        row.block_size = block_size;
                        row.n = matrix.rows;
                        row.nnz = matrix.nnz();
                        row.status = "fail";
                        row.error_message = ex.what();
                        rows.push_back(row);
                    }
                    const Row& row = rows.back();
                    std::cout << "[quality] " << case_name
                              << " bs=" << block_size
                              << " precond=" << preconditioner
                              << " rel=" << row.true_linear_rel_res
                              << " dxerr=" << row.dx_error_ratio
                              << " status=" << row.status << '\n';
                }
            }
        }
        write_csv(options.output, rows);
        std::cout << "[done] wrote " << options.output << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "ras_overlap_standalone_quality failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
