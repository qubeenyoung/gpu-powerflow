#pragma once

#include "common/data_types.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace exp20260426::jac_asm_bench {

struct CaseData {
    std::string name;
    int32_t n_bus = 0;
    int32_t n_edges = 0;

    std::vector<int32_t> row_ptr;
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
    std::filesystem::path data_root = "datasets/matpower8.1/cupf_all_dumps";
    std::string case_name = "all";
    std::string mode = "both";
    int32_t warmup = 10;
    int32_t iters = 100;
    int32_t cpu_repeats = 10;
};

struct Timing {
    std::string case_name;
    int32_t n_bus = 0;
    int32_t n_edges = 0;
    int32_t n_pv = 0;
    int32_t n_pq = 0;
    int32_t jac_dim = 0;
    int32_t jac_nnz = 0;
    // Common symbolic analysis: Jacobian pattern plus value-slot maps shared by
    // vertex and edge kernels.
    double analyze_ms = 0.0;
    // Edge-only preprocessing: materialize CSR row_ptr into row[k].
    double edge_map_ms = 0.0;
    // Common symbolic analysis with row[k] materialization fused into the
    // Ybus->Jacobian map pass.
    double analyze_fused_edge_map_ms = 0.0;
    float edge_fill_ms = 0.0f;
    float edge_fill_no_atomic_ms = 0.0f;
    float vertex_fill_ms = 0.0f;
};

template <typename Fn>
double measureCpuMs(Fn fn)
{
    const auto begin = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename Fn>
double measureCpuAverageMs(int32_t repeats, Fn fn)
{
    double total_ms = 0.0;
    for (int32_t repeat = 0; repeat < repeats; ++repeat) {
        total_ms += measureCpuMs(fn);
    }
    return total_ms / repeats;
}

std::string normalizeMode(std::string mode);
bool wantsEdge(const Options& options);
bool wantsEdgeNoAtomic(const Options& options);
bool wantsEdgeBuild(const Options& options);
bool wantsVertex(const Options& options);

CaseData loadCase(const std::filesystem::path& case_dir);
std::vector<std::filesystem::path> listCases(const Options& options);
YbusCsr makeHostCsr(const CaseData& data);

Options parseOptions(int argc, char** argv);
void printHeader();
void printTiming(const Timing& timing);

}  // namespace exp20260426::jac_asm_bench
