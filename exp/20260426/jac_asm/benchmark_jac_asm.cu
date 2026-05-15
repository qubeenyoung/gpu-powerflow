#include "benchmark_fill.hpp"
#include "benchmark_support.hpp"
#include "common/jacobian_build.hpp"
#include "edge/edge_build.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace bench = exp20260426::jac_asm_bench;

namespace {

bench::Timing runCase(const bench::Options& options,
                      const std::filesystem::path& case_dir)
{
    const bench::CaseData data = bench::loadCase(case_dir);
    const YbusCsr host_ybus = bench::makeHostCsr(data);

    bench::Timing timing;
    timing.case_name = data.name;
    timing.n_bus = data.n_bus;
    timing.n_edges = data.n_edges;
    timing.n_pv = data.pv.size();
    timing.n_pq = data.pq.size();

    JacobianBuild build;
    timing.analyze_ms = bench::measureCpuAverageMs(options.cpu_repeats, [&]() {
        build = buildJacobian(host_ybus, data.pv.data(), data.pv.size(), data.pq.data(), data.pq.size());
    });

    timing.jac_dim = build.pattern.dim;
    timing.jac_nnz = build.pattern.nnz;

    if (bench::wantsEdgeBuild(options)) {
        EdgeYbusMap edge_map;
        timing.edge_map_ms = bench::measureCpuAverageMs(options.cpu_repeats, [&]() {
            edge_map = buildEdgeYbusMap(host_ybus);
        });

        EdgeJacobianBuild fused_build;
        timing.analyze_fused_edge_map_ms = bench::measureCpuAverageMs(options.cpu_repeats, [&]() {
            fused_build = buildJacobianWithEdgeYbusMap(
                host_ybus, data.pv.data(), data.pv.size(), data.pq.data(), data.pq.size());
        });

        if (bench::wantsEdge(options)) {
            timing.edge_fill_ms = bench::runEdgeFill(
                options, data, fused_build.edge_map, fused_build.jacobian.pattern, fused_build.jacobian.map);
        }

        if (bench::wantsEdgeNoAtomic(options)) {
            timing.edge_fill_no_atomic_ms = bench::runEdgeFillNoAtomic(
                options, data, fused_build.edge_map, fused_build.jacobian.pattern, fused_build.jacobian.map);
        }
    }

    if (bench::wantsVertex(options)) {
        timing.vertex_fill_ms = bench::runVertexFill(options, data, build.index, build.pattern, build.map);
    }

    return timing;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const bench::Options options = bench::parseOptions(argc, argv);
        const std::vector<std::filesystem::path> cases = bench::listCases(options);

        bench::printHeader();
        for (const std::filesystem::path& case_dir : cases) {
            bench::printTiming(runCase(options, case_dir));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
