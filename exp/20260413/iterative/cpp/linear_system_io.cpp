#include "linear_system_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace exp_20260413::iterative {
namespace {

void ensure_parent_dir(const std::filesystem::path& path)
{
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void require_open(const std::ifstream& in, const std::filesystem::path& path)
{
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
    }
}

void require_open(const std::ofstream& out, const std::filesystem::path& path)
{
    if (!out) {
        throw std::runtime_error("Failed to open " + path.string());
    }
}

void skip_comment_line(std::istream& in, const std::string& token)
{
    if (!token.empty() && token[0] == '#') {
        std::string rest;
        std::getline(in, rest);
    }
}

std::vector<double> read_vector_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    require_open(in, path);

    std::string token;
    int32_t size = -1;
    std::vector<double> values;

    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            skip_comment_line(in, token);
            continue;
        }
        if (token == "type") {
            std::string type;
            in >> type;
            if (type != "vector") {
                throw std::runtime_error("Expected vector file: " + path.string());
            }
        } else if (token == "size") {
            in >> size;
            if (size < 0) {
                throw std::runtime_error("Invalid vector size in " + path.string());
            }
            values.assign(static_cast<std::size_t>(size), 0.0);
        } else if (token == "values") {
            if (size < 0) {
                throw std::runtime_error("Vector size must appear before values in " + path.string());
            }
            for (int32_t i = 0; i < size; ++i) {
                int32_t index = -1;
                double value = 0.0;
                if (!(in >> index >> value)) {
                    throw std::runtime_error("Malformed vector entry in " + path.string());
                }
                if (index < 0 || index >= size) {
                    throw std::runtime_error("Vector index out of bounds in " + path.string());
                }
                values[static_cast<std::size_t>(index)] = value;
            }
        } else {
            throw std::runtime_error("Unknown token in vector file " + path.string() + ": " + token);
        }
    }

    if (size < 0) {
        throw std::runtime_error("Missing vector size in " + path.string());
    }
    return values;
}

}  // namespace

std::string iter_dir_name(int32_t iter)
{
    std::ostringstream out;
    out << "iter_" << std::setw(3) << std::setfill('0') << iter;
    return out.str();
}

void write_csr(const std::filesystem::path& path,
               int32_t rows,
               int32_t cols,
               const std::vector<int32_t>& row_ptr,
               const std::vector<int32_t>& col_idx,
               const std::vector<double>& values)
{
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("CSR dimensions must be non-negative");
    }
    if (row_ptr.size() != static_cast<std::size_t>(rows + 1)) {
        throw std::runtime_error("CSR row_ptr size does not match row count");
    }
    if (col_idx.size() != values.size()) {
        throw std::runtime_error("CSR col_idx and values sizes differ");
    }

    ensure_parent_dir(path);
    std::ofstream out(path);
    require_open(out, path);

    out << std::scientific << std::setprecision(17);
    out << "# cupf iterative linear system csr v1\n";
    out << "type csr\n";
    out << "rows " << rows << "\n";
    out << "cols " << cols << "\n";
    out << "nnz " << values.size() << "\n";
    out << "row_ptr";
    for (int32_t value : row_ptr) {
        out << ' ' << value;
    }
    out << "\ncol_idx";
    for (int32_t value : col_idx) {
        out << ' ' << value;
    }
    out << "\nvalues";
    for (double value : values) {
        out << ' ' << value;
    }
    out << '\n';
}

void write_vector(const std::filesystem::path& path,
                  const std::vector<double>& values)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    require_open(out, path);

    out << std::scientific << std::setprecision(17);
    out << "# cupf iterative linear system vector v1\n";
    out << "type vector\n";
    out << "size " << values.size() << "\n";
    out << "values\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        out << i << ' ' << values[i] << '\n';
    }
}

void write_metadata(const std::filesystem::path& path,
                    const std::vector<std::pair<std::string, std::string>>& entries)
{
    ensure_parent_dir(path);
    std::ofstream out(path);
    require_open(out, path);

    out << "# cupf iterative linear system metadata v1\n";
    for (const auto& [key, value] : entries) {
        out << key << ' ' << value << '\n';
    }
}

LinearSystemSnapshot read_snapshot(const std::filesystem::path& snapshot_dir)
{
    const auto csr_path = snapshot_dir / "J.csr";
    const auto rhs_path = snapshot_dir / "rhs.txt";

    std::ifstream in(csr_path);
    require_open(in, csr_path);

    LinearSystemSnapshot snapshot;
    int64_t nnz = -1;
    std::string token;

    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            skip_comment_line(in, token);
            continue;
        }
        if (token == "type") {
            std::string type;
            in >> type;
            if (type != "csr") {
                throw std::runtime_error("Expected CSR matrix: " + csr_path.string());
            }
        } else if (token == "rows") {
            in >> snapshot.rows;
        } else if (token == "cols") {
            in >> snapshot.cols;
        } else if (token == "nnz") {
            in >> nnz;
        } else if (token == "row_ptr") {
            if (snapshot.rows < 0) {
                throw std::runtime_error("rows must appear before row_ptr in " + csr_path.string());
            }
            snapshot.row_ptr.resize(static_cast<std::size_t>(snapshot.rows + 1));
            for (int32_t& value : snapshot.row_ptr) {
                in >> value;
            }
        } else if (token == "col_idx") {
            if (nnz < 0) {
                throw std::runtime_error("nnz must appear before col_idx in " + csr_path.string());
            }
            snapshot.col_idx.resize(static_cast<std::size_t>(nnz));
            for (int32_t& value : snapshot.col_idx) {
                in >> value;
            }
        } else if (token == "values") {
            if (nnz < 0) {
                throw std::runtime_error("nnz must appear before values in " + csr_path.string());
            }
            snapshot.values.resize(static_cast<std::size_t>(nnz));
            for (double& value : snapshot.values) {
                in >> value;
            }
        } else {
            throw std::runtime_error("Unknown token in CSR file " + csr_path.string() + ": " + token);
        }
    }

    if (snapshot.rows <= 0 || snapshot.cols <= 0 || nnz < 0 ||
        snapshot.row_ptr.size() != static_cast<std::size_t>(snapshot.rows + 1) ||
        snapshot.col_idx.size() != static_cast<std::size_t>(nnz) ||
        snapshot.values.size() != static_cast<std::size_t>(nnz)) {
        throw std::runtime_error("Incomplete CSR snapshot in " + csr_path.string());
    }

    snapshot.rhs = read_vector_file(rhs_path);
    if (snapshot.rhs.size() != static_cast<std::size_t>(snapshot.rows)) {
        throw std::runtime_error("RHS size does not match matrix rows in " + snapshot_dir.string());
    }

    const auto x_path = snapshot_dir / "x_direct.txt";
    if (std::filesystem::exists(x_path)) {
        snapshot.x_direct = read_vector_file(x_path);
    }

    return snapshot;
}

std::vector<std::filesystem::path> find_snapshot_dirs(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> dirs;
    if (!std::filesystem::exists(root)) {
        return dirs;
    }

    auto is_snapshot_dir = [](const std::filesystem::path& dir) {
        return std::filesystem::exists(dir / "J.csr") &&
               std::filesystem::exists(dir / "rhs.txt");
    };

    if (is_snapshot_dir(root)) {
        dirs.push_back(root);
    } else {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_directory() && is_snapshot_dir(entry.path())) {
                dirs.push_back(entry.path());
            }
        }
    }

    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

double inf_norm(const std::vector<double>& values)
{
    double norm = 0.0;
    for (double value : values) {
        norm = std::max(norm, std::abs(value));
    }
    return norm;
}

double residual_inf_norm(const LinearSystemSnapshot& snapshot,
                         const std::vector<double>& x)
{
    if (x.size() != static_cast<std::size_t>(snapshot.cols)) {
        throw std::runtime_error("Solution size does not match matrix cols");
    }
    if (snapshot.rhs.size() != static_cast<std::size_t>(snapshot.rows)) {
        throw std::runtime_error("RHS size does not match matrix rows");
    }

    double norm = 0.0;
    for (int32_t row = 0; row < snapshot.rows; ++row) {
        double sum = 0.0;
        for (int32_t k = snapshot.row_ptr[static_cast<std::size_t>(row)];
             k < snapshot.row_ptr[static_cast<std::size_t>(row + 1)];
             ++k) {
            sum += snapshot.values[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(snapshot.col_idx[static_cast<std::size_t>(k)])];
        }
        norm = std::max(norm, std::abs(sum - snapshot.rhs[static_cast<std::size_t>(row)]));
    }
    return norm;
}

double diff_inf_norm(const std::vector<double>& lhs,
                     const std::vector<double>& rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Vector sizes differ");
    }

    double norm = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        norm = std::max(norm, std::abs(lhs[i] - rhs[i]));
    }
    return norm;
}

}  // namespace exp_20260413::iterative
