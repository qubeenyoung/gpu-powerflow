#include "tools/matrix_io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sparse_direct::io {
namespace {

using matrix::CsrMatrix;
using matrix::Index;
using matrix::Value;

struct CoordinateEntry {
    Index row = 0;
    Index col = 0;
    Value value = 0.0;
};

std::string to_lower(std::string text)
{
    for (char& ch : text) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return text;
}

std::string next_data_line(std::istream& input)
{
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && line[first] != '%') {
            return line;
        }
    }

    throw std::runtime_error("unexpected end of Matrix Market file");
}

MatrixMarketInfo read_matrix_market_header(std::istream& input)
{
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty Matrix Market file");
    }

    std::string banner;
    MatrixMarketInfo info;

    std::istringstream stream(line);
    stream >> banner >> info.object >> info.format >> info.field >> info.symmetry;
    if (!stream) {
        throw std::runtime_error("invalid Matrix Market header");
    }

    banner = to_lower(banner);
    info.object = to_lower(info.object);
    info.format = to_lower(info.format);
    info.field = to_lower(info.field);
    info.symmetry = to_lower(info.symmetry);

    if (banner != "%%matrixmarket" || info.object != "matrix") {
        throw std::runtime_error("unsupported Matrix Market header");
    }

    return info;
}

}  // namespace

matrix::CsrMatrix read_matrix_market_csr(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Matrix Market file: " + path.string());
    }

    const MatrixMarketInfo info = read_matrix_market_header(input);
    const bool pattern_values = info.field == "pattern";
    const bool complex_values = info.field == "complex";
    const bool mirror_triangle = info.symmetry == "symmetric" || info.symmetry == "hermitian";

    if (info.format != "coordinate") {
        throw std::runtime_error("expected Matrix Market coordinate matrix: " + path.string());
    }
    if (info.field != "real" && info.field != "integer" && info.field != "complex" && info.field != "pattern") {
        throw std::runtime_error("unsupported Matrix Market coordinate field: " + info.field);
    }
    if (info.symmetry != "general" && info.symmetry != "symmetric" && info.symmetry != "hermitian") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + info.symmetry);
    }

    // Shape line: rows columns entries. Matrix Market coordinates are 1-based.
    Index rows = 0;
    Index cols = 0;
    Index declared_nnz = 0;

    std::istringstream shape(next_data_line(input));
    shape >> rows >> cols >> declared_nnz;
    if (!shape || rows <= 0 || cols <= 0 || declared_nnz < 0) {
        throw std::runtime_error("invalid Matrix Market coordinate shape: " + path.string());
    }

    // Read all coordinates, expanding symmetric/hermitian storage to the full
    // matrix because the direct solvers benchmarked here consume full CSR/CSC.
    std::vector<CoordinateEntry> entries;
    entries.reserve(static_cast<std::size_t>(declared_nnz) * (mirror_triangle ? 2u : 1u));

    for (Index line_index = 0; line_index < declared_nnz; ++line_index) {
        int one_based_row = 0;
        int one_based_col = 0;
        double value = 1.0;
        double ignored_imaginary = 0.0;

        std::istringstream row_stream(next_data_line(input));
        if (pattern_values) {
            row_stream >> one_based_row >> one_based_col;
        } else if (complex_values) {
            row_stream >> one_based_row >> one_based_col >> value >> ignored_imaginary;
        } else {
            row_stream >> one_based_row >> one_based_col >> value;
        }

        if (!row_stream) {
            throw std::runtime_error("failed to parse Matrix Market coordinate entry: " + path.string());
        }

        CoordinateEntry entry;
        entry.row = one_based_row - 1;
        entry.col = one_based_col - 1;
        entry.value = value;

        if (entry.row < 0 || entry.row >= rows || entry.col < 0 || entry.col >= cols) {
            throw std::runtime_error("Matrix Market coordinate is out of range: " + path.string());
        }

        entries.push_back(entry);
        if (mirror_triangle && entry.row != entry.col) {
            entries.push_back({entry.col, entry.row, entry.value});
        }
    }

    // Sort by CSR order and sum duplicate coordinates. SuiteSparse files can
    // contain duplicates, and summing them gives the mathematical matrix.
    std::sort(entries.begin(), entries.end(), [](const CoordinateEntry& lhs, const CoordinateEntry& rhs) {
        if (lhs.row != rhs.row) {
            return lhs.row < rhs.row;
        }
        return lhs.col < rhs.col;
    });

    std::vector<CoordinateEntry> merged_entries;
    merged_entries.reserve(entries.size());
    for (const CoordinateEntry& entry : entries) {
        if (!merged_entries.empty() &&
            merged_entries.back().row == entry.row &&
            merged_entries.back().col == entry.col) {
            merged_entries.back().value += entry.value;
        } else {
            merged_entries.push_back(entry);
        }
    }

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.assign(static_cast<std::size_t>(rows + 1), 0);
    matrix.col_idx.resize(merged_entries.size());
    matrix.values.resize(merged_entries.size());

    for (const CoordinateEntry& entry : merged_entries) {
        ++matrix.row_ptr[entry.row + 1];
    }
    for (Index row = 0; row < rows; ++row) {
        matrix.row_ptr[row + 1] += matrix.row_ptr[row];
    }
    for (std::size_t index = 0; index < merged_entries.size(); ++index) {
        matrix.col_idx[index] = merged_entries[index].col;
        matrix.values[index] = merged_entries[index].value;
    }

    matrix.validate();
    return matrix;
}

matrix::CscMatrix read_matrix_market_csc(const std::filesystem::path& path)
{
    return matrix::to_csc(read_matrix_market_csr(path));
}

DenseVector read_matrix_market_vector(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Matrix Market vector: " + path.string());
    }

    const MatrixMarketInfo info = read_matrix_market_header(input);
    if (info.format != "array") {
        throw std::runtime_error("expected Matrix Market array vector: " + path.string());
    }
    if (info.field != "real" && info.field != "integer") {
        throw std::runtime_error("unsupported Matrix Market vector field: " + info.field);
    }
    if (info.symmetry != "general") {
        throw std::runtime_error("Matrix Market array vector must be general: " + path.string());
    }

    DenseVector vector;

    std::istringstream shape(next_data_line(input));
    shape >> vector.rows >> vector.cols;
    if (!shape || vector.rows <= 0 || vector.cols <= 0) {
        throw std::runtime_error("invalid Matrix Market array shape: " + path.string());
    }

    const std::size_t expected_values =
        static_cast<std::size_t>(vector.rows) * static_cast<std::size_t>(vector.cols);
    vector.values.reserve(expected_values);

    while (vector.values.size() < expected_values) {
        std::istringstream value_stream(next_data_line(input));

        double value = 0.0;
        while (value_stream >> value) {
            vector.values.push_back(value);
            if (vector.values.size() == expected_values) {
                break;
            }
        }

        if (!value_stream.eof()) {
            throw std::runtime_error("failed to parse Matrix Market vector value: " + path.string());
        }
    }

    return vector;
}

void write_matrix_market_vector(
    const std::filesystem::path& path,
    const std::vector<matrix::Value>& values)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open Matrix Market vector for writing: " + path.string());
    }

    output
        << "%%MatrixMarket matrix array real general\n"
        << "% generated by sparse_direct_solver\n"
        << values.size() << " 1\n"
        << std::setprecision(17);

    for (matrix::Value value : values) {
        output << value << "\n";
    }
}

}  // namespace sparse_direct::io
