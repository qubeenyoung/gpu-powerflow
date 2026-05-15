#include "benchmark_support.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace exp20260426::jac_asm_bench {

std::string normalizeMode(std::string mode)
{
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

bool wantsEdge(const Options& options)
{
    const std::string mode = normalizeMode(options.mode);
    return mode == "edge" || mode == "both" || mode == "all";
}

bool wantsEdgeNoAtomic(const Options& options)
{
    const std::string mode = normalizeMode(options.mode);
    return mode == "edge_no_atomic" || mode == "all";
}

bool wantsEdgeBuild(const Options& options)
{
    const std::string mode = normalizeMode(options.mode);
    return mode == "edge_build" || wantsEdge(options) || wantsEdgeNoAtomic(options);
}

bool wantsVertex(const Options& options)
{
    const std::string mode = normalizeMode(options.mode);
    return mode == "vertex" || mode == "vertex_thread" || mode == "both" || mode == "all";
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

YbusCsr makeHostCsr(const CaseData& data)
{
    YbusCsr ybus;
    ybus.n_bus = data.n_bus;
    ybus.n_edges = data.n_edges;
    ybus.row_ptr = data.row_ptr.data();
    ybus.col = data.col.data();
    ybus.real = data.y_re.data();
    ybus.imag = data.y_im.data();
    return ybus;
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
        } else if (arg == "--cpu-repeats" && i + 1 < argc) {
            options.cpu_repeats = std::stoi(argv[++i]);
        }
    }
    options.mode = normalizeMode(options.mode);
    return options;
}

void printHeader()
{
    std::cout
        << "case,n_bus,ybus_nnz,n_pv,n_pq,jac_dim,jac_nnz,"
        << "analyze_ms,edge_map_ms,analyze_fused_edge_map_ms,"
        << "edge_fill_ms,edge_fill_no_atomic_ms,vertex_fill_ms\n";
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
        << timing.edge_map_ms << ','
        << timing.analyze_fused_edge_map_ms << ','
        << timing.edge_fill_ms << ','
        << timing.edge_fill_no_atomic_ms << ','
        << timing.vertex_fill_ms << '\n';
}

}  // namespace exp20260426::jac_asm_bench
