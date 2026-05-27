#include "io/dense_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>

namespace benchio {
namespace {

struct NpyHeader {
    std::string descr;
    bool fortran_order = false;
    std::vector<std::size_t> shape;
    std::size_t data_offset = 0;
};

/// Return a path string suitable for diagnostics.
std::string path_string(const std::filesystem::path& path)
{
    return path.string();
}

/// Trim ASCII whitespace from both ends of a string.
std::string trim(std::string text)
{
    const auto first = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

/// Extract a string value from the Python-literal NPY header.
std::string parse_header_string(const std::string& header,
                                const std::string& key,
                                const std::filesystem::path& path)
{
    const std::regex pattern("['\"]" + key + "['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
    std::smatch match;
    if (!std::regex_search(header, match, pattern)) {
        throw std::runtime_error("missing '" + key + "' in NPY header: " + path_string(path));
    }
    return match[1].str();
}

/// Extract a boolean value from the Python-literal NPY header.
bool parse_header_bool(const std::string& header,
                       const std::string& key,
                       const std::filesystem::path& path)
{
    const std::regex pattern("['\"]" + key + "['\"]\\s*:\\s*(True|False)");
    std::smatch match;
    if (!std::regex_search(header, match, pattern)) {
        throw std::runtime_error("missing '" + key + "' in NPY header: " + path_string(path));
    }
    return match[1].str() == "True";
}

/// Extract the tuple-valued shape from the Python-literal NPY header.
std::vector<std::size_t> parse_header_shape(const std::string& header,
                                            const std::filesystem::path& path)
{
    const std::regex pattern("['\"]shape['\"]\\s*:\\s*\\(([^\\)]*)\\)");
    std::smatch match;
    if (!std::regex_search(header, match, pattern)) {
        throw std::runtime_error("missing 'shape' in NPY header: " + path_string(path));
    }

    std::vector<std::size_t> shape;
    std::stringstream stream(match[1].str());
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        try {
            shape.push_back(static_cast<std::size_t>(std::stoull(token)));
        } catch (const std::exception&) {
            throw std::runtime_error("invalid NPY shape token '" + token +
                                     "' in file: " + path_string(path));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("empty NPY shape in file: " + path_string(path));
    }
    return shape;
}

/// Read a little-endian unsigned 16-bit value from two bytes.
std::uint16_t read_le_u16(const unsigned char* bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8);
}

/// Read a little-endian unsigned 32-bit value from four bytes.
std::uint32_t read_le_u32(const unsigned char* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

/// Parse the NPY header and leave the stream positioned at the data payload.
NpyHeader read_npy_header(std::ifstream& input, const std::filesystem::path& path)
{
    unsigned char prefix[8] = {};
    input.read(reinterpret_cast<char*>(prefix), 8);
    if (input.gcount() != 8 || std::memcmp(prefix, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("file is not a NumPy .npy file: " + path_string(path));
    }

    const unsigned char major = prefix[6];
    const unsigned char minor = prefix[7];
    std::size_t header_len = 0;
    std::size_t preamble_len = 0;
    if (major == 1) {
        unsigned char len_bytes[2] = {};
        input.read(reinterpret_cast<char*>(len_bytes), 2);
        if (input.gcount() != 2) {
            throw std::runtime_error("truncated NPY v1 header length: " + path_string(path));
        }
        header_len = read_le_u16(len_bytes);
        preamble_len = 10;
    } else if (major == 2 || major == 3) {
        unsigned char len_bytes[4] = {};
        input.read(reinterpret_cast<char*>(len_bytes), 4);
        if (input.gcount() != 4) {
            throw std::runtime_error("truncated NPY v2/v3 header length: " + path_string(path));
        }
        header_len = read_le_u32(len_bytes);
        preamble_len = 12;
    } else {
        throw std::runtime_error("unsupported NPY version " + std::to_string(major) + "." +
                                 std::to_string(minor) + " in file: " + path_string(path));
    }

    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header_len));
    if (input.gcount() != static_cast<std::streamsize>(header_len)) {
        throw std::runtime_error("truncated NPY header payload: " + path_string(path));
    }

    NpyHeader out;
    out.descr = parse_header_string(header, "descr", path);
    out.fortran_order = parse_header_bool(header, "fortran_order", path);
    out.shape = parse_header_shape(header, path);
    out.data_offset = preamble_len + header_len;
    return out;
}

/// Return the product of all shape dimensions with overflow checks by division.
std::size_t element_count(const std::vector<std::size_t>& shape,
                          const std::filesystem::path& path)
{
    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        if (dim == 0) {
            throw std::runtime_error("NPY shape has a zero dimension: " + path_string(path));
        }
        if (count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("NPY shape element count overflows size_t: " +
                                     path_string(path));
        }
        count *= dim;
    }
    return count;
}

/// Load the raw float32 payload from a NPY file after validating dtype and size.
std::vector<float> load_npy_fp32_payload(const std::filesystem::path& path, NpyHeader& header)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open NPY file: " + path_string(path));
    }

    header = read_npy_header(input, path);
    if (header.descr != "<f4" && header.descr != "|f4") {
        throw std::runtime_error("expected little-endian fp32 NPY dtype '<f4' in file " +
                                 path_string(path) + ", got '" + header.descr + "'");
    }

    const std::size_t count = element_count(header.shape, path);
    std::vector<float> values(count);
    const std::size_t bytes = count * sizeof(float);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("truncated fp32 NPY data payload: " + path_string(path));
    }
    return values;
}

}  // namespace

DenseMatrix<float> load_dense_matrix_fp32(const std::filesystem::path& path)
{
    NpyHeader header;
    std::vector<float> values = load_npy_fp32_payload(path, header);
    if (header.shape.size() != 2) {
        throw std::runtime_error("expected a 2-D dense matrix NPY file: " + path_string(path));
    }

    DenseMatrix<float> matrix;
    matrix.rows = header.shape[0];
    matrix.cols = header.shape[1];
    matrix.values = std::move(values);
    matrix.layout = header.fortran_order ? MatrixLayout::ColMajor : MatrixLayout::RowMajor;
    validate_dense_matrix(matrix, "dense matrix loaded from " + path_string(path));
    return matrix;
}

std::vector<float> load_dense_vector_fp32(const std::filesystem::path& path)
{
    NpyHeader header;
    std::vector<float> values = load_npy_fp32_payload(path, header);
    if (header.shape.size() == 1) {
        return values;
    }
    if (header.shape.size() == 2 && (header.shape[0] == 1 || header.shape[1] == 1)) {
        return values;
    }
    throw std::runtime_error("expected a 1-D vector or singleton 2-D NPY file: " +
                             path_string(path));
}

DenseMatrix<float> to_column_major_fp32(const DenseMatrix<float>& matrix)
{
    return to_column_major(matrix);
}

}  // namespace benchio
