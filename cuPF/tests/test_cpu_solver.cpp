#include "dump_case_loader.hpp"

#include "newton_solver/core/newton_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>

#ifdef CUPF_WITH_CUDA
  #include <cuda_runtime.h>
#endif


namespace {

constexpr const char* kCase30IeeePath = "/workspace/v1/core/dumps/case30_ieee";

double max_voltage_delta(const std::vector<std::complex<double>>& lhs,
                         const std::vector<std::complex<double>>& rhs)
{
    double max_delta = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(lhs[i] - rhs[i]));
    }
    return max_delta;
}

NRResultF64 run_solver(const NewtonOptions& options)
{
    const auto data = cupf::tests::load_dump_case(kCase30IeeePath);
    const YbusViewF64 ybus = data.ybus();

    NewtonSolver solver(options);
    solver.analyze(ybus,
                   data.pv.data(), static_cast<int32_t>(data.pv.size()),
                   data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig cfg;
    cfg.tolerance = 1e-8;
    cfg.max_iter = 15;

    NRResultF64 result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 cfg,
                 result);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::isfinite(result.final_mismatch));
    EXPECT_LE(result.final_mismatch, cfg.tolerance);
    EXPECT_EQ(static_cast<int32_t>(result.V.size()), data.rows);
    EXPECT_GT(max_voltage_delta(data.v0, result.V), 0.0);

    return result;
}

#ifdef CUPF_WITH_CUDA
bool cuda_device_available()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

}  // namespace

TEST(CpuSolverSmoke, EdgeBasedCase30Converges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }

    NewtonOptions options;
    options.backend = BackendKind::CPU;
    options.compute = ComputePolicy::FP64;

    (void)run_solver(options);
}

#ifdef CUPF_WITH_CUDA
TEST(CudaSolverSmoke, MixedCase30Converges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;

    (void)run_solver(options);
}

TEST(CudaSolverSmoke, Fp64Case30Converges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP64;

    (void)run_solver(options);
}
#endif
