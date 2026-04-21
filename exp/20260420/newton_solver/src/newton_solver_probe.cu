#include "newton_solver.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct MatrixEntry {
    int32_t row = 0;
    int32_t col = 0;
    float real = 0.0f;
    float imag = 0.0f;
};

struct CaseData {
    std::string name;
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    std::vector<int32_t> row_ptr;
    std::vector<int32_t> row;
    std::vector<int32_t> col;
    std::vector<float> y_re;
    std::vector<float> y_im;

    std::vector<float> sbus_re;
    std::vector<float> sbus_im;
    std::vector<double> v0_re;
    std::vector<double> v0_im;

    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct Options {
    fs::path data_root = "datasets/texas_univ_cases/cuPF_datasets";
    std::string case_name = "all";
    int32_t max_iter = 20;
    double tolerance = 1e-6;
    int32_t batch_size = 1;
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(const std::vector<T>& values)
    {
        assign(values);
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void assign(const std::vector<T>& values)
    {
        count = values.size();
        if (count == 0) {
            return;
        }

        void* raw = nullptr;
        cudaError_t err = cudaMalloc(&raw, sizeof(T) * count);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(err));
        }
        ptr = static_cast<T*>(raw);

        err = cudaMemcpy(ptr, values.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy: ") + cudaGetErrorString(err));
        }
    }
};

bool isPayloadLine(const std::string& line)
{
    for (char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return ch != '%' && ch != '#';
        }
    }
    return false;
}

std::string nextPayloadLine(std::ifstream& in, const fs::path& path)
{
    std::string line;
    while (std::getline(in, line)) {
        if (isPayloadLine(line)) {
            return line;
        }
    }
    throw std::runtime_error("unexpected end of file while reading " + path.string());
}

std::vector<int32_t> loadIntList(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::vector<int32_t> values;
    std::string line;
    while (std::getline(in, line)) {
        if (!isPayloadLine(line)) {
            continue;
        }
        std::istringstream iss(line);
        int32_t value = 0;
        if (!(iss >> value)) {
            throw std::runtime_error("malformed integer in " + path.string());
        }
        values.push_back(value);
    }
    return values;
}

template <typename T>
void loadComplexVector(const fs::path& path,
                       std::vector<T>& real,
                       std::vector<T>& imag)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::string line;
    while (std::getline(in, line)) {
        if (!isPayloadLine(line)) {
            continue;
        }
        std::istringstream iss(line);
        double re = 0.0;
        double im = 0.0;
        if (!(iss >> re >> im)) {
            throw std::runtime_error("malformed complex vector entry in " + path.string());
        }
        real.push_back(static_cast<T>(re));
        imag.push_back(static_cast<T>(im));
    }
}

template <typename T>
std::vector<T> repeatBatch(const std::vector<T>& values, int32_t batch_size)
{
    if (batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    std::vector<T> repeated(values.size() * static_cast<std::size_t>(batch_size));
    for (int32_t batch = 0; batch < batch_size; ++batch) {
        std::copy(values.begin(),
                  values.end(),
                  repeated.begin() + static_cast<std::size_t>(batch) * values.size());
    }
    return repeated;
}

void loadYbus(const fs::path& path, CaseData& data)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("missing MatrixMarket header in " + path.string());
    }

    std::string lowered = header;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lowered.find("matrixmarket") == std::string::npos ||
        lowered.find("coordinate") == std::string::npos ||
        lowered.find("complex") == std::string::npos) {
        throw std::runtime_error("unsupported MatrixMarket format in " + path.string());
    }
    const bool symmetric = lowered.find("symmetric") != std::string::npos;

    std::istringstream dims(nextPayloadLine(in, path));
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    int32_t nnz = 0;
    if (!(dims >> n_rows >> n_cols >> nnz) || n_rows != n_cols) {
        throw std::runtime_error("malformed MatrixMarket dimensions in " + path.string());
    }

    std::vector<MatrixEntry> entries;
    entries.reserve(symmetric ? static_cast<std::size_t>(nnz) * 2 : static_cast<std::size_t>(nnz));

    for (int32_t k = 0; k < nnz; ++k) {
        std::istringstream iss(nextPayloadLine(in, path));
        int32_t row = 0;
        int32_t col = 0;
        double re = 0.0;
        double im = 0.0;
        if (!(iss >> row >> col >> re >> im)) {
            throw std::runtime_error("malformed MatrixMarket entry in " + path.string());
        }
        --row;
        --col;
        entries.push_back(MatrixEntry{row, col, static_cast<float>(re), static_cast<float>(im)});
        if (symmetric && row != col) {
            entries.push_back(MatrixEntry{col, row, static_cast<float>(re), static_cast<float>(im)});
        }
    }

    std::sort(entries.begin(), entries.end(), [](const MatrixEntry& lhs, const MatrixEntry& rhs) {
        return std::tie(lhs.row, lhs.col) < std::tie(rhs.row, rhs.col);
    });

    std::vector<MatrixEntry> merged;
    merged.reserve(entries.size());
    for (const MatrixEntry& entry : entries) {
        if (entry.row < 0 || entry.row >= n_rows || entry.col < 0 || entry.col >= n_cols) {
            throw std::runtime_error("MatrixMarket entry out of bounds in " + path.string());
        }
        if (!merged.empty() && merged.back().row == entry.row && merged.back().col == entry.col) {
            merged.back().real += entry.real;
            merged.back().imag += entry.imag;
            continue;
        }
        merged.push_back(entry);
    }

    data.n_bus = n_rows;
    data.n_edges = static_cast<int32_t>(merged.size());
    data.row_ptr.assign(static_cast<std::size_t>(n_rows) + 1, 0);
    data.row.resize(merged.size());
    data.col.resize(merged.size());
    data.y_re.resize(merged.size());
    data.y_im.resize(merged.size());

    for (const MatrixEntry& entry : merged) {
        ++data.row_ptr[static_cast<std::size_t>(entry.row) + 1];
    }
    for (int32_t row = 0; row < n_rows; ++row) {
        data.row_ptr[static_cast<std::size_t>(row) + 1] += data.row_ptr[static_cast<std::size_t>(row)];
    }

    for (std::size_t i = 0; i < merged.size(); ++i) {
        data.row[i] = merged[i].row;
        data.col[i] = merged[i].col;
        data.y_re[i] = merged[i].real;
        data.y_im[i] = merged[i].imag;
    }
}

CaseData loadCase(const fs::path& case_dir)
{
    CaseData data;
    data.name = case_dir.filename().string();
    loadYbus(case_dir / "dump_Ybus.mtx", data);
    loadComplexVector(case_dir / "dump_Sbus.txt", data.sbus_re, data.sbus_im);
    loadComplexVector(case_dir / "dump_V.txt", data.v0_re, data.v0_im);
    data.pv = loadIntList(case_dir / "dump_pv.txt");
    data.pq = loadIntList(case_dir / "dump_pq.txt");

    if (static_cast<int32_t>(data.sbus_re.size()) != data.n_bus ||
        static_cast<int32_t>(data.v0_re.size()) != data.n_bus ||
        data.sbus_re.size() != data.sbus_im.size() ||
        data.v0_re.size() != data.v0_im.size()) {
        throw std::runtime_error("case vector sizes do not match Ybus for " + case_dir.string());
    }
    return data;
}

bool isCaseDir(const fs::path& path)
{
    return fs::is_directory(path) &&
           fs::exists(path / "dump_Ybus.mtx") &&
           fs::exists(path / "dump_Sbus.txt") &&
           fs::exists(path / "dump_V.txt") &&
           fs::exists(path / "dump_pv.txt") &&
           fs::exists(path / "dump_pq.txt");
}

std::vector<fs::path> listCases(const Options& options)
{
    if (options.case_name != "all") {
        return {options.data_root / options.case_name};
    }

    std::vector<fs::path> cases;
    for (const fs::directory_entry& entry : fs::directory_iterator(options.data_root)) {
        if (isCaseDir(entry.path())) {
            cases.push_back(entry.path());
        }
    }
    std::sort(cases.begin(), cases.end());
    return cases;
}

YbusGraph makeHostGraph(const CaseData& data)
{
    return YbusGraph{
        data.n_bus,
        data.n_edges,
        data.row.data(),
        data.col.data(),
        data.row_ptr.data(),
        data.y_re.data(),
        data.y_im.data(),
    };
}

YbusGraph makeDeviceGraph(const CaseData& data,
                          const DeviceBuffer<int32_t>& row_ptr,
                          const DeviceBuffer<int32_t>& row,
                          const DeviceBuffer<int32_t>& col,
                          const DeviceBuffer<float>& y_re,
                          const DeviceBuffer<float>& y_im)
{
    return YbusGraph{
        data.n_bus,
        data.n_edges,
        row.ptr,
        col.ptr,
        row_ptr.ptr,
        y_re.ptr,
        y_im.ptr,
    };
}

void runCase(const fs::path& case_dir, const Options& options)
{
    const auto total_begin = std::chrono::steady_clock::now();
    const CaseData data = loadCase(case_dir);

    DeviceBuffer<int32_t> d_row_ptr(data.row_ptr);
    DeviceBuffer<int32_t> d_row(data.row);
    DeviceBuffer<int32_t> d_col(data.col);
    DeviceBuffer<float> d_y_re(data.y_re);
    DeviceBuffer<float> d_y_im(data.y_im);
    const std::vector<float> sbus_re_batch = repeatBatch(data.sbus_re, options.batch_size);
    const std::vector<float> sbus_im_batch = repeatBatch(data.sbus_im, options.batch_size);
    const std::vector<double> v0_re_batch = repeatBatch(data.v0_re, options.batch_size);
    const std::vector<double> v0_im_batch = repeatBatch(data.v0_im, options.batch_size);
    DeviceBuffer<float> d_sbus_re(sbus_re_batch);
    DeviceBuffer<float> d_sbus_im(sbus_im_batch);
    DeviceBuffer<double> d_v0_re(v0_re_batch);
    DeviceBuffer<double> d_v0_im(v0_im_batch);

    const YbusGraph host_ybus = makeHostGraph(data);
    const YbusGraph device_ybus = makeDeviceGraph(data, d_row_ptr, d_row, d_col, d_y_re, d_y_im);

    exp20260420::newton_solver::NewtonWorkspace ws;
    const auto solver_begin = std::chrono::steady_clock::now();
    exp20260420::newton_solver::newtonAnalyze(
        ws,
        host_ybus,
        device_ybus,
        data.pv.data(),
        static_cast<int32_t>(data.pv.size()),
        data.pq.data(),
        static_cast<int32_t>(data.pq.size()),
        options.batch_size);

    exp20260420::newton_solver::NewtonOptions solver_options;
    solver_options.max_iter = options.max_iter;
    solver_options.tolerance = options.tolerance;
    solver_options.batch_size = options.batch_size;

    const exp20260420::newton_solver::NewtonResult result =
        exp20260420::newton_solver::newtonSolve(
            ws,
            d_sbus_re.ptr,
            d_sbus_im.ptr,
            d_v0_re.ptr,
            d_v0_im.ptr,
            solver_options);
    const auto solver_end = std::chrono::steady_clock::now();
    const double solver_elapsed_ms =
        std::chrono::duration<double, std::milli>(solver_end - solver_begin).count();

    exp20260420::newton_solver::newtonDestroy(ws);
    const auto total_end = std::chrono::steady_clock::now();
    const double total_elapsed_ms =
        std::chrono::duration<double, std::milli>(total_end - total_begin).count();

    std::cout << "case=" << data.name
              << " n_bus=" << data.n_bus
              << " n_edges=" << data.n_edges
              << " n_pv=" << data.pv.size()
              << " n_pq=" << data.pq.size()
              << " batch_size=" << options.batch_size
              << " iterations=" << result.iterations
              << " converged=" << (result.converged ? "true" : "false")
              << " final_mismatch=" << std::scientific << std::setprecision(6)
              << result.final_mismatch
              << " solver_elapsed_ms=" << std::fixed << std::setprecision(3)
              << solver_elapsed_ms
              << " total_elapsed_ms=" << total_elapsed_ms << '\n';
}

Options parseArgs(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--data-root") {
            options.data_root = needValue("--data-root");
        } else if (arg == "--case") {
            options.case_name = needValue("--case");
        } else if (arg == "--max-iter") {
            options.max_iter = std::stoi(needValue("--max-iter"));
        } else if (arg == "--tolerance") {
            options.tolerance = std::stod(needValue("--tolerance"));
        } else if (arg == "--batch-size") {
            options.batch_size = std::stoi(needValue("--batch-size"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--data-root PATH] [--case NAME|all]"
                         " [--max-iter N] [--tolerance T] [--batch-size N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.batch_size <= 0) {
        throw std::runtime_error("--batch-size must be positive");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseArgs(argc, argv);
        const std::vector<fs::path> cases = listCases(options);
        if (cases.empty()) {
            throw std::runtime_error("no cases found under " + options.data_root.string());
        }

        for (const fs::path& case_dir : cases) {
            runCase(case_dir, options);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
