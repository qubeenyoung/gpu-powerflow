#include "dump_case_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>


namespace cupf::tests {
namespace {

struct MatrixEntry {
    int32_t row = 0;
    int32_t col = 0;
    std::complex<double> value;
};

bool is_comment_or_empty(const std::string& line)
{
    for (char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch == '#' || ch == '%';
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
    throw std::runtime_error("Unexpected end of file while reading " + path.string());
}

std::vector<std::complex<double>> load_complex_pairs(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    std::vector<std::complex<double>> values;
    std::string line;
    while (std::getline(in, line)) {
        if (is_comment_or_empty(line)) {
            continue;
        }

        std::istringstream iss(line);
        double real = 0.0;
        double imag = 0.0;
        if (!(iss >> real >> imag)) {
            throw std::runtime_error("Malformed complex vector entry in " + path.string());
        }
        values.emplace_back(real, imag);
    }

    return values;
}

std::vector<int32_t> load_int_values(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
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
            throw std::runtime_error("Malformed integer entry in " + path.string());
        }
        values.push_back(value);
    }

    return values;
}

void load_matrix_market_csr(const std::filesystem::path& path,
                            int32_t& rows,
                            int32_t& cols,
                            std::vector<int32_t>& indptr,
                            std::vector<int32_t>& indices,
                            std::vector<std::complex<double>>& data)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Missing MatrixMarket header in " + path.string());
    }

    std::string lowered = header;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    const bool symmetric = lowered.find("symmetric") != std::string::npos;
    if (lowered.find("matrixmarket") == std::string::npos ||
        lowered.find("coordinate") == std::string::npos ||
        lowered.find("complex") == std::string::npos) {
        throw std::runtime_error("Unsupported MatrixMarket format in " + path.string());
    }

    std::istringstream dims(next_payload_line(in, path));
    int32_t nnz = 0;
    if (!(dims >> rows >> cols >> nnz)) {
        throw std::runtime_error("Malformed MatrixMarket dimensions in " + path.string());
    }

    std::vector<MatrixEntry> entries;
    entries.reserve(symmetric ? nnz * 2 : nnz);

    for (int32_t k = 0; k < nnz; ++k) {
        std::string line = next_payload_line(in, path);
        std::istringstream iss(line);

        int32_t row_1based = 0;
        int32_t col_1based = 0;
        double real = 0.0;
        double imag = 0.0;
        if (!(iss >> row_1based >> col_1based >> real >> imag)) {
            throw std::runtime_error("Malformed MatrixMarket entry in " + path.string());
        }

        MatrixEntry entry{
            row_1based - 1,
            col_1based - 1,
            std::complex<double>(real, imag),
        };
        entries.push_back(entry);

        if (symmetric && entry.row != entry.col) {
            entries.push_back(MatrixEntry{entry.col, entry.row, entry.value});
        }
    }

    std::sort(entries.begin(), entries.end(), [](const MatrixEntry& lhs, const MatrixEntry& rhs) {
        return std::tie(lhs.row, lhs.col) < std::tie(rhs.row, rhs.col);
    });

    std::vector<MatrixEntry> merged;
    merged.reserve(entries.size());
    for (const auto& entry : entries) {
        if (!merged.empty() &&
            merged.back().row == entry.row &&
            merged.back().col == entry.col) {
            merged.back().value += entry.value;
            continue;
        }
        merged.push_back(entry);
    }

    indptr.assign(rows + 1, 0);
    for (const auto& entry : merged) {
        if (entry.row < 0 || entry.row >= rows || entry.col < 0 || entry.col >= cols) {
            throw std::runtime_error("Matrix entry out of bounds in " + path.string());
        }
        ++indptr[entry.row + 1];
    }

    for (int32_t row = 0; row < rows; ++row) {
        indptr[row + 1] += indptr[row];
    }

    indices.resize(merged.size());
    data.resize(merged.size());
    std::vector<int32_t> cursor = indptr;

    for (const auto& entry : merged) {
        const int32_t pos = cursor[entry.row]++;
        indices[pos] = entry.col;
        data[pos] = entry.value;
    }
}

void validate_case_data(const DumpCaseData& data)
{
    if (data.rows <= 0 || data.cols <= 0) {
        throw std::runtime_error("Loaded case has invalid Ybus dimensions");
    }
    if (data.rows != data.cols) {
        throw std::runtime_error("Ybus must be square");
    }
    if (static_cast<int32_t>(data.sbus.size()) != data.rows) {
        throw std::runtime_error("Sbus size does not match Ybus rows");
    }
    if (static_cast<int32_t>(data.v0.size()) != data.rows) {
        throw std::runtime_error("V0 size does not match Ybus rows");
    }
}

}  // namespace

DumpCaseData load_dump_case(const std::filesystem::path& case_dir)
{
    if (!std::filesystem::exists(case_dir)) {
        throw std::runtime_error("Case directory does not exist: " + case_dir.string());
    }

    DumpCaseData data;
    data.case_name = case_dir.filename().string();

    load_matrix_market_csr(case_dir / "dump_Ybus.mtx",
                           data.rows, data.cols,
                           data.indptr, data.indices, data.ybus_data);
    data.sbus = load_complex_pairs(case_dir / "dump_Sbus.txt");
    data.v0 = load_complex_pairs(case_dir / "dump_V.txt");
    data.pv = load_int_values(case_dir / "dump_pv.txt");
    data.pq = load_int_values(case_dir / "dump_pq.txt");

    validate_case_data(data);
    return data;
}

}  // namespace cupf::tests
