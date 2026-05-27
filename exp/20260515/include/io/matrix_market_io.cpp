#include "io/matrix_market_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

namespace benchio {
namespace {

struct MatrixMarketHeader {
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
};

struct Triplet {
    int row = 0;
    int col = 0;
    float value = 0.0f;
};

/// Return a path string suitable for diagnostics.
std::string path_string(const std::filesystem::path& path)
{
    return path.string();
}

/// Convert ASCII text to lower case.
std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

/// Return true when a Matrix Market line is empty or a comment.
bool is_ignored_line(const std::string& line)
{
    const auto first = std::find_if_not(line.begin(), line.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    return first == line.end() || *first == '%';
}

/// Parse the Matrix Market banner line.
MatrixMarketHeader parse_banner(const std::string& line, const std::filesystem::path& path)
{
    std::istringstream stream(line);
    std::string banner;
    MatrixMarketHeader header;
    stream >> banner >> header.object >> header.format >> header.field >> header.symmetry;
    if (!stream || lowercase(banner) != "%%matrixmarket") {
        throw std::runtime_error("invalid Matrix Market banner in file: " + path_string(path));
    }

    header.object = lowercase(header.object);
    header.format = lowercase(header.format);
    header.field = lowercase(header.field);
    header.symmetry = lowercase(header.symmetry);
    if (header.object != "matrix" || header.format != "coordinate") {
        throw std::runtime_error("expected Matrix Market coordinate matrix in file: " +
                                 path_string(path));
    }
    if (header.field != "real" && header.field != "integer" && header.field != "pattern") {
        throw std::runtime_error("expected real/integer/pattern Matrix Market field in file " +
                                 path_string(path) + ", got '" + header.field + "'");
    }
    if (header.symmetry != "general") {
        throw std::runtime_error("Matrix Market symmetry '" + header.symmetry +
                                 "' is rejected for the general nonsymmetric benchmark loader: " +
                                 path_string(path));
    }
    return header;
}

/// Read the next non-comment Matrix Market line while tracking source line numbers.
bool read_data_line(std::ifstream& input,
                    std::string& line,
                    std::size_t& line_number)
{
    while (std::getline(input, line)) {
        ++line_number;
        if (!is_ignored_line(line)) {
            return true;
        }
    }
    return false;
}

/// Convert a signed count to int with diagnostics tied to the input file.
int checked_int(long long value, const std::string& field, const std::filesystem::path& path)
{
    if (value < 0 || value > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Matrix Market " + field + " is outside int range in file: " +
                                 path_string(path));
    }
    return static_cast<int>(value);
}

/// Sort coordinate entries by CSR order and merge duplicate coordinates.
CsrMatrix<float, int> build_csr(int rows,
                                int cols,
                                std::vector<Triplet>& triplets,
                                const std::filesystem::path& path)
{
    std::sort(triplets.begin(), triplets.end(), [](const Triplet& lhs, const Triplet& rhs) {
        if (lhs.row != rhs.row) {
            return lhs.row < rhs.row;
        }
        return lhs.col < rhs.col;
    });

    CsrMatrix<float, int> matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    matrix.col_ind.reserve(triplets.size());
    matrix.values.reserve(triplets.size());

    int current_row = 0;
    std::size_t cursor = 0;
    while (cursor < triplets.size()) {
        const int row = triplets[cursor].row;
        const int col = triplets[cursor].col;
        while (current_row < row) {
            matrix.row_ptr[static_cast<std::size_t>(current_row + 1)] =
                static_cast<int>(matrix.col_ind.size());
            ++current_row;
        }

        double value_sum = 0.0;
        while (cursor < triplets.size() && triplets[cursor].row == row &&
               triplets[cursor].col == col) {
            value_sum += static_cast<double>(triplets[cursor].value);
            ++cursor;
        }
        matrix.col_ind.push_back(col);
        matrix.values.push_back(static_cast<float>(value_sum));
    }

    while (current_row < rows) {
        matrix.row_ptr[static_cast<std::size_t>(current_row + 1)] =
            static_cast<int>(matrix.col_ind.size());
        ++current_row;
    }

    matrix.nnz = checked_int(static_cast<long long>(matrix.col_ind.size()), "merged nnz", path);
    validate_csr_matrix(matrix, "CSR matrix loaded from " + path_string(path));
    return matrix;
}

}  // namespace

CsrMatrix<float, int> load_matrix_market_csr_fp32(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Matrix Market file: " + path_string(path));
    }

    std::string line;
    std::size_t line_number = 0;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty Matrix Market file: " + path_string(path));
    }
    ++line_number;

    const MatrixMarketHeader header = parse_banner(line, path);
    if (!read_data_line(input, line, line_number)) {
        throw std::runtime_error("missing Matrix Market dimension line: " + path_string(path));
    }

    long long rows_ll = 0;
    long long cols_ll = 0;
    long long nnz_ll = 0;
    {
        std::istringstream dims(line);
        dims >> rows_ll >> cols_ll >> nnz_ll;
        if (!dims || rows_ll <= 0 || cols_ll <= 0 || nnz_ll < 0) {
            throw std::runtime_error("invalid Matrix Market dimensions at line " +
                                     std::to_string(line_number) + " in file: " +
                                     path_string(path));
        }
    }

    const int rows = checked_int(rows_ll, "rows", path);
    const int cols = checked_int(cols_ll, "cols", path);
    const int declared_nnz = checked_int(nnz_ll, "nnz", path);

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(declared_nnz));

    int entries_read = 0;
    while (entries_read < declared_nnz && read_data_line(input, line, line_number)) {
        std::istringstream row_stream(line);
        long long row_one_based = 0;
        long long col_one_based = 0;
        double value = 1.0;
        row_stream >> row_one_based >> col_one_based;
        if (header.field != "pattern") {
            row_stream >> value;
        }
        if (!row_stream) {
            throw std::runtime_error("invalid Matrix Market entry at line " +
                                     std::to_string(line_number) + " in file: " +
                                     path_string(path));
        }
        if (row_one_based < 1 || row_one_based > rows || col_one_based < 1 ||
            col_one_based > cols) {
            throw std::runtime_error("Matrix Market coordinate is out of range at line " +
                                     std::to_string(line_number) + " in file: " +
                                     path_string(path));
        }

        Triplet triplet;
        triplet.row = static_cast<int>(row_one_based - 1);
        triplet.col = static_cast<int>(col_one_based - 1);
        triplet.value = static_cast<float>(value);
        triplets.push_back(triplet);
        ++entries_read;
    }

    if (entries_read != declared_nnz) {
        throw std::runtime_error("Matrix Market file ended after " + std::to_string(entries_read) +
                                 " entries; expected " + std::to_string(declared_nnz) +
                                 " in file: " + path_string(path));
    }

    return build_csr(rows, cols, triplets, path);
}

}  // namespace benchio
