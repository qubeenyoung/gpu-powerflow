#include "common/data_types.hpp"
#include "common/jacobian_build.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void fill_jacobian_edge(
    const YbusGraph ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values);

__global__ void fill_jacobian_vertex(
    const YbusGraph ybus,
    const float* __restrict__ v_re,
    const float* __restrict__ v_im,
    const float* __restrict__ v_norm_re,
    const float* __restrict__ v_norm_im,
    int32_t n_rows,
    const int32_t* __restrict__ pvpq,
    const int32_t* __restrict__ offdiagJ11,
    const int32_t* __restrict__ offdiagJ21,
    const int32_t* __restrict__ offdiagJ12,
    const int32_t* __restrict__ offdiagJ22,
    const int32_t* __restrict__ diagJ11,
    const int32_t* __restrict__ diagJ21,
    const int32_t* __restrict__ diagJ12,
    const int32_t* __restrict__ diagJ22,
    float* __restrict__ J_values);

namespace fs = std::filesystem;

namespace {

struct CaseData {
    std::string name;
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    std::vector<int32_t> row_ptr;
    std::vector<int32_t> row;
    std::vector<int32_t> col;
    std::vector<float> y_re;
    std::vector<float> y_im;

    std::vector<float> v_re;
    std::vector<float> v_im;
    std::vector<float> v_norm_re;
    std::vector<float> v_norm_im;

    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct Options {
    fs::path data_root = "datasets/texas_univ_cases/cuPF_datasets";
    std::string case_name = "all";
    std::string mode = "both";
    int32_t warmup = 10;
    int32_t iters = 100;
};

struct Timing {
    std::string case_name;
    int32_t n_bus = 0;
    int32_t n_edges = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t jac_dim = 0;
    int32_t jac_nnz = 0;
    double analyze_ms = 0.0;
    float edge_fill_ms = 0.0f;
    float vertex_fill_ms = 0.0f;
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    int32_t count = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(const std::vector<T>& values)
    {
        assign(values);
    }

    explicit DeviceBuffer(int32_t n)
    {
        allocate(n);
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    void allocate(int32_t n)
    {
        count = n;
        if (count == 0) {
            return;
        }

        void* raw = nullptr;
        cudaMalloc(&raw, sizeof(T) * count);
        ptr = (T*)raw;
    }

    void assign(const std::vector<T>& values)
    {
        allocate(values.size());
        if (count > 0) {
            cudaMemcpy(ptr, values.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
        }
    }
};

void checkCuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

std::vector<int32_t> loadIntList(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::vector<int32_t> values;
    int32_t value = 0;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

void loadVoltage(const fs::path& path, CaseData& data)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    float re = 0.0f;
    float im = 0.0f;
    while (in >> re >> im) {
        data.v_re.push_back(re);
        data.v_im.push_back(im);

        const float mag = std::hypot(re, im);
        if (mag > 1e-6f) {
            data.v_norm_re.push_back(re / mag);
            data.v_norm_im.push_back(im / mag);
        } else {
            data.v_norm_re.push_back(0.0f);
            data.v_norm_im.push_back(0.0f);
        }
    }
}

void loadYbus(const fs::path& path, CaseData& data)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::string line;
    std::getline(in, line);

    do {
        std::getline(in, line);
    } while (!line.empty() && line[0] == '%');

    std::istringstream shape(line);
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    int32_t nnz = 0;
    shape >> n_rows >> n_cols >> nnz;

    data.n_bus = n_rows;
    data.n_edges = nnz;
    data.row.assign(nnz, 0);
    data.col.assign(nnz, 0);
    data.y_re.assign(nnz, 0.0f);
    data.y_im.assign(nnz, 0.0f);
    data.row_ptr.assign(n_rows + 1, 0);

    for (int32_t k = 0; k < nnz; ++k) {
        int32_t row = 0;
        int32_t col = 0;
        float re = 0.0f;
        float im = 0.0f;
        in >> row >> col >> re >> im;

        --row;
        --col;

        data.row[k] = row;
        data.col[k] = col;
        data.y_re[k] = re;
        data.y_im[k] = im;
        ++data.row_ptr[row + 1];
    }

    for (int32_t row = 0; row < n_rows; ++row) {
        data.row_ptr[row + 1] += data.row_ptr[row];
    }
}

CaseData loadCase(const fs::path& case_dir)
{
    CaseData data;
    data.name = case_dir.filename().string();
    loadYbus(case_dir / "dump_Ybus.mtx", data);
    loadVoltage(case_dir / "dump_V.txt", data);
    data.pv = loadIntList(case_dir / "dump_pv.txt");
    data.pq = loadIntList(case_dir / "dump_pq.txt");
    return data;
}

std::vector<fs::path> listCases(const Options& options)
{
    std::vector<fs::path> cases;
    if (options.case_name != "all") {
        cases.push_back(options.data_root / options.case_name);
        return cases;
    }

    for (const auto& item : fs::directory_iterator(options.data_root)) {
        if (fs::exists(item.path() / "dump_Ybus.mtx")) {
            cases.push_back(item.path());
        }
    }

    std::sort(cases.begin(), cases.end());
    return cases;
}

YbusGraph makeHostGraph(const CaseData& data)
{
    YbusGraph ybus;
    ybus.n_bus = data.n_bus;
    ybus.n_edges = data.n_edges;
    ybus.row = data.row.data();
    ybus.col = data.col.data();
    ybus.real = data.y_re.data();
    ybus.imag = data.y_im.data();
    ybus.row_ptr = data.row_ptr.data();
    return ybus;
}

YbusGraph makeDeviceGraph(const CaseData& data,
                          const DeviceBuffer<int32_t>& row_ptr,
                          const DeviceBuffer<int32_t>& row,
                          const DeviceBuffer<int32_t>& col,
                          const DeviceBuffer<float>& y_re,
                          const DeviceBuffer<float>& y_im)
{
    YbusGraph ybus;
    ybus.n_bus = data.n_bus;
    ybus.n_edges = data.n_edges;
    ybus.row = row.ptr;
    ybus.col = col.ptr;
    ybus.real = y_re.ptr;
    ybus.imag = y_im.ptr;
    ybus.row_ptr = row_ptr.ptr;
    return ybus;
}

template <typename Fn>
double measureCpuMs(Fn fn)
{
    const auto begin = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

float elapsedKernelMs(cudaEvent_t begin, cudaEvent_t end, int32_t iters)
{
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
    return ms / iters;
}

float runEdgeFill(const Options& options,
                  const CaseData& data,
                  const JacobianPattern& pattern,
                  const JacobianMap& map)
{
    DeviceBuffer<int32_t> d_row_ptr(data.row_ptr);
    DeviceBuffer<int32_t> d_row(data.row);
    DeviceBuffer<int32_t> d_col(data.col);
    DeviceBuffer<float> d_y_re(data.y_re);
    DeviceBuffer<float> d_y_im(data.y_im);
    DeviceBuffer<float> d_v_re(data.v_re);
    DeviceBuffer<float> d_v_im(data.v_im);
    DeviceBuffer<float> d_v_norm_re(data.v_norm_re);
    DeviceBuffer<float> d_v_norm_im(data.v_norm_im);

    DeviceBuffer<int32_t> d_J11(map.offdiagJ11);
    DeviceBuffer<int32_t> d_J21(map.offdiagJ21);
    DeviceBuffer<int32_t> d_J12(map.offdiagJ12);
    DeviceBuffer<int32_t> d_J22(map.offdiagJ22);
    DeviceBuffer<int32_t> d_diagJ11(map.diagJ11);
    DeviceBuffer<int32_t> d_diagJ21(map.diagJ21);
    DeviceBuffer<int32_t> d_diagJ12(map.diagJ12);
    DeviceBuffer<int32_t> d_diagJ22(map.diagJ22);
    DeviceBuffer<float> d_values(pattern.nnz);

    const YbusGraph ybus = makeDeviceGraph(data, d_row_ptr, d_row, d_col, d_y_re, d_y_im);
    const int32_t block = 256;
    const int32_t grid = (data.n_edges + block - 1) / block;

    checkCuda(cudaMemset(d_values.ptr, 0, sizeof(float) * pattern.nnz), "edge memset");

    for (int32_t iter = 0; iter < options.warmup; ++iter) {
        fill_jacobian_edge<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaDeviceSynchronize(), "edge warmup");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaEventCreate(&begin), "edge event begin");
    checkCuda(cudaEventCreate(&end), "edge event end");

    checkCuda(cudaEventRecord(begin), "edge record begin");
    for (int32_t iter = 0; iter < options.iters; ++iter) {
        fill_jacobian_edge<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaEventRecord(end), "edge record end");
    checkCuda(cudaEventSynchronize(end), "edge synchronize");
    checkCuda(cudaGetLastError(), "edge kernel");

    const float ms = elapsedKernelMs(begin, end, options.iters);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms;
}

float runVertexFill(const Options& options,
                    const CaseData& data,
                    const BusIndexMap& index,
                    const JacobianPattern& pattern,
                    const JacobianMap& map)
{
    DeviceBuffer<int32_t> d_row_ptr(data.row_ptr);
    DeviceBuffer<int32_t> d_row(data.row);
    DeviceBuffer<int32_t> d_col(data.col);
    DeviceBuffer<float> d_y_re(data.y_re);
    DeviceBuffer<float> d_y_im(data.y_im);
    DeviceBuffer<float> d_v_re(data.v_re);
    DeviceBuffer<float> d_v_im(data.v_im);
    DeviceBuffer<float> d_v_norm_re(data.v_norm_re);
    DeviceBuffer<float> d_v_norm_im(data.v_norm_im);

    DeviceBuffer<int32_t> d_pvpq(index.pvpq);
    DeviceBuffer<int32_t> d_J11(map.offdiagJ11);
    DeviceBuffer<int32_t> d_J21(map.offdiagJ21);
    DeviceBuffer<int32_t> d_J12(map.offdiagJ12);
    DeviceBuffer<int32_t> d_J22(map.offdiagJ22);
    DeviceBuffer<int32_t> d_diagJ11(map.diagJ11);
    DeviceBuffer<int32_t> d_diagJ21(map.diagJ21);
    DeviceBuffer<int32_t> d_diagJ12(map.diagJ12);
    DeviceBuffer<int32_t> d_diagJ22(map.diagJ22);
    DeviceBuffer<float> d_values(pattern.nnz);

    const YbusGraph ybus = makeDeviceGraph(data, d_row_ptr, d_row, d_col, d_y_re, d_y_im);
    const int32_t n_rows = index.n_pvpq;
    const int32_t block = 256;
    const int32_t grid = (n_rows + block - 1) / block;

    checkCuda(cudaMemset(d_values.ptr, 0, sizeof(float) * pattern.nnz), "vertex memset");

    for (int32_t iter = 0; iter < options.warmup; ++iter) {
        fill_jacobian_vertex<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            n_rows, d_pvpq.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaDeviceSynchronize(), "vertex warmup");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaEventCreate(&begin), "vertex event begin");
    checkCuda(cudaEventCreate(&end), "vertex event end");

    checkCuda(cudaEventRecord(begin), "vertex record begin");
    for (int32_t iter = 0; iter < options.iters; ++iter) {
        fill_jacobian_vertex<<<grid, block>>>(
            ybus, d_v_re.ptr, d_v_im.ptr, d_v_norm_re.ptr, d_v_norm_im.ptr,
            n_rows, d_pvpq.ptr,
            d_J11.ptr, d_J21.ptr, d_J12.ptr, d_J22.ptr,
            d_diagJ11.ptr, d_diagJ21.ptr, d_diagJ12.ptr, d_diagJ22.ptr,
            d_values.ptr);
    }
    checkCuda(cudaEventRecord(end), "vertex record end");
    checkCuda(cudaEventSynchronize(end), "vertex synchronize");
    checkCuda(cudaGetLastError(), "vertex kernel");

    const float ms = elapsedKernelMs(begin, end, options.iters);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms;
}

Timing runCase(const Options& options, const fs::path& case_dir)
{
    const CaseData data = loadCase(case_dir);
    const YbusGraph host_ybus = makeHostGraph(data);

    Timing timing;
    timing.case_name = data.name;
    timing.n_bus = data.n_bus;
    timing.n_edges = data.n_edges;
    timing.n_pv = data.pv.size();
    timing.n_pq = data.pq.size();

    JacobianBuild build;
    timing.analyze_ms = measureCpuMs([&]() {
        build = buildJacobian(host_ybus, data.pv.data(), data.pv.size(), data.pq.data(), data.pq.size());
    });

    timing.jac_dim = build.pattern.dim;
    timing.jac_nnz = build.pattern.nnz;

    if (options.mode == "edge" || options.mode == "both") {
        timing.edge_fill_ms = runEdgeFill(options, data, build.pattern, build.map);
    }

    if (options.mode == "vertex" || options.mode == "both") {
        timing.vertex_fill_ms = runVertexFill(options, data, build.index, build.pattern, build.map);
    }

    return timing;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int32_t i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            options.data_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            options.case_name = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            options.mode = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iters = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::stoi(argv[++i]);
        }
    }
    return options;
}

void printHeader()
{
    std::cout
        << "case,n_bus,ybus_nnz,n_pv,n_pq,jac_dim,jac_nnz,"
        << "analyze_ms,edge_fill_ms,vertex_fill_ms\n";
}

void printTiming(const Timing& timing)
{
    std::cout
        << timing.case_name << ','
        << timing.n_bus << ','
        << timing.n_edges << ','
        << timing.n_pv << ','
        << timing.n_pq << ','
        << timing.jac_dim << ','
        << timing.jac_nnz << ','
        << timing.analyze_ms << ','
        << timing.edge_fill_ms << ','
        << timing.vertex_fill_ms << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        const std::vector<fs::path> cases = listCases(options);

        printHeader();
        for (const fs::path& case_dir : cases) {
            printTiming(runCase(options, case_dir));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
