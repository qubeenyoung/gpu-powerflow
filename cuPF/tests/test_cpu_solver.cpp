#include "dump_case_loader.hpp"
#include "small_cases.hpp"

#include "newton_solver/core/newton_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <vector>

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

NRResult run_solver(const NewtonOptions& options)
{
    const auto data = cupf::tests::load_dump_case(kCase30IeeePath);
    const YbusView ybus = data.ybus();

    NewtonSolver solver(options);
    solver.initialize(ybus,
                   data.pv.data(), static_cast<int32_t>(data.pv.size()),
                   data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig cfg;
    cfg.tolerance = 1e-8;
    cfg.max_iter = 15;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 cfg,
                 SolveOptions{},
                 result);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::isfinite(result.final_mismatch));
    EXPECT_LE(result.final_mismatch, cfg.tolerance);
    EXPECT_EQ(static_cast<int32_t>(result.V.size()), data.rows);
    EXPECT_GT(max_voltage_delta(data.v0, result.V), 0.0);

    return result;
}

NRResult run_two_bus_solver(const NewtonOptions& options,
                            double tolerance,
                            double voltage_tolerance)
{
    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig cfg;
    cfg.tolerance = tolerance;
    cfg.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 cfg,
                 SolveOptions{},
                 result);

    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(std::isfinite(result.final_mismatch));
    EXPECT_LE(result.final_mismatch, cfg.tolerance);
    ASSERT_EQ(result.V.size(), data.expected_v.size());
    EXPECT_LE(std::abs(result.V[0] - data.expected_v[0]), voltage_tolerance);
    EXPECT_LE(std::abs(result.V[1] - data.expected_v[1]), voltage_tolerance);
    return result;
}

std::vector<std::complex<double>> repeat_complex_vector(const std::vector<std::complex<double>>& src,
                                                        int32_t batch_size)
{
    std::vector<std::complex<double>> dst(
        static_cast<std::size_t>(batch_size) * src.size());
    for (int32_t b = 0; b < batch_size; ++b) {
        std::copy(src.begin(),
                  src.end(),
                  dst.begin() + static_cast<std::ptrdiff_t>(b) *
                                  static_cast<std::ptrdiff_t>(src.size()));
    }
    return dst;
}

#ifdef CUPF_WITH_CUDA
bool cuda_device_available()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

}  // namespace

TEST(CpuSolverDeterministic, SolveBeforeInitializeThrows)
{
    const auto data = cupf::tests::make_two_bus_case();
    NewtonSolver solver;
    NRConfig cfg;
    NRResult result;

    EXPECT_THROW(
        solver.solve(data.ybus(),
                     data.sbus.data(),
                     data.v0.data(),
                     data.pv.data(), static_cast<int32_t>(data.pv.size()),
                     data.pq.data(), static_cast<int32_t>(data.pq.size()),
                     cfg,
                     SolveOptions{},
                     result),
        std::runtime_error);
}

TEST(CpuSolverDeterministic, TwoBusConvergesWithoutExternalDump)
{
    NewtonOptions options;
    options.backend = BackendKind::CPU;
    options.compute = ComputePolicy::FP64;

    (void)run_two_bus_solver(options, 1e-10, 1e-8);
}

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

TEST(CpuSolverSmoke, PandapowerJacobianWithKluCase30Converges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }

    NewtonOptions options;
    options.backend = BackendKind::CPU;
    options.compute = ComputePolicy::FP64;
    options.cpu_jacobian = CpuJacobianKind::Pandapower;
    options.cpu_linear_solver = CpuLinearSolverKind::KLU;

    (void)run_solver(options);
}

TEST(CpuSolverSmoke, PandapowerJacobianWithUmfpackCase30Converges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }

    NewtonOptions options;
    options.backend = BackendKind::CPU;
    options.compute = ComputePolicy::FP64;
    options.cpu_jacobian = CpuJacobianKind::Pandapower;
    options.cpu_linear_solver = CpuLinearSolverKind::UMFPACK;

    (void)run_solver(options);
}

#ifdef CUPF_WITH_CUDA
TEST(CudaSolverDeterministic, Fp64TwoBusConvergesWithoutExternalDump)
{
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP64;

    (void)run_two_bus_solver(options, 1e-10, 1e-8);
}

#ifdef CUPF_ENABLE_CUSTOM_SOLVER
TEST(CudaSolverDeterministic, CustomFp64TwoBusConvergesWithoutExternalDump)
{
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP64;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;

    (void)run_two_bus_solver(options, 1e-10, 1e-8);
}
#endif

TEST(CudaSolverDeterministic, Fp32TwoBusConvergesWithoutExternalDump)
{
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP32;

    (void)run_two_bus_solver(options, 1e-6, 5e-4);
}

TEST(CudaSolverDeterministic, MixedTwoBusBatchConvergesWithoutExternalDump)
{
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();
    constexpr int32_t batch_size = 2;
    const auto batched_sbus = cupf::tests::repeat_complex_vector(data.sbus, batch_size);
    const auto batched_v0 = cupf::tests::repeat_complex_vector(data.v0, batch_size);

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig cfg;
    cfg.tolerance = 1e-6;
    cfg.max_iter = 20;

    NRBatchResult result;
    solver.solve_batch(ybus,
                       batched_sbus.data(),
                       ybus.rows,
                       batched_v0.data(),
                       ybus.rows,
                       batch_size,
                       data.pv.data(), static_cast<int32_t>(data.pv.size()),
                       data.pq.data(), static_cast<int32_t>(data.pq.size()),
                       cfg,
                       SolveOptions{},
                       result);

    ASSERT_EQ(result.n_bus, data.rows);
    ASSERT_EQ(result.batch_size, batch_size);
    ASSERT_EQ(result.V.size(), static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(data.rows));
    ASSERT_EQ(result.final_mismatch.size(), static_cast<std::size_t>(batch_size));
    ASSERT_EQ(result.converged.size(), static_cast<std::size_t>(batch_size));

    for (int32_t b = 0; b < batch_size; ++b) {
        EXPECT_TRUE(result.converged[static_cast<std::size_t>(b)] != 0);
        EXPECT_TRUE(std::isfinite(result.final_mismatch[static_cast<std::size_t>(b)]));
        EXPECT_LE(result.final_mismatch[static_cast<std::size_t>(b)], cfg.tolerance);
        for (int32_t bus = 0; bus < data.rows; ++bus) {
            const auto actual =
                result.V[static_cast<std::size_t>(b) * static_cast<std::size_t>(data.rows) +
                         static_cast<std::size_t>(bus)];
            EXPECT_LE(std::abs(actual - data.expected_v[static_cast<std::size_t>(bus)]), 5e-4);
        }
    }
}

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

TEST(CudaSolverSmoke, MixedCase30BatchConverges)
{
    if (!std::filesystem::exists(kCase30IeeePath)) {
        GTEST_SKIP() << "dump case not available at " << kCase30IeeePath;
    }
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    const auto data = cupf::tests::load_dump_case(kCase30IeeePath);
    const YbusView ybus = data.ybus();
    constexpr int32_t batch_size = 2;
    const auto batched_sbus = repeat_complex_vector(data.sbus, batch_size);
    const auto batched_v0 = repeat_complex_vector(data.v0, batch_size);

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                   data.pv.data(), static_cast<int32_t>(data.pv.size()),
                   data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig cfg;
    cfg.tolerance = 1e-8;
    cfg.max_iter = 15;

    NRBatchResult result;
    solver.solve_batch(ybus,
                       batched_sbus.data(),
                       ybus.rows,
                       batched_v0.data(),
                       ybus.rows,
                       batch_size,
                       data.pv.data(), static_cast<int32_t>(data.pv.size()),
                       data.pq.data(), static_cast<int32_t>(data.pq.size()),
                       cfg,
                       SolveOptions{},
                       result);

    EXPECT_EQ(result.n_bus, data.rows);
    EXPECT_EQ(result.batch_size, batch_size);
    EXPECT_EQ(result.V.size(), static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(data.rows));
    ASSERT_EQ(result.final_mismatch.size(), static_cast<std::size_t>(batch_size));
    ASSERT_EQ(result.converged.size(), static_cast<std::size_t>(batch_size));

    for (int32_t b = 0; b < batch_size; ++b) {
        EXPECT_TRUE(result.converged[static_cast<std::size_t>(b)] != 0);
        EXPECT_TRUE(std::isfinite(result.final_mismatch[static_cast<std::size_t>(b)]));
        EXPECT_LE(result.final_mismatch[static_cast<std::size_t>(b)], cfg.tolerance);
    }

    for (int32_t bus = 0; bus < data.rows; ++bus) {
        const auto v0 = result.V[static_cast<std::size_t>(bus)];
        const auto v1 = result.V[static_cast<std::size_t>(data.rows + bus)];
        EXPECT_LE(std::abs(v0 - v1), 1e-6);
    }
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

TEST(CudaSolverSmoke, Fp64Case30EdgeAtomicConverges)
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
    options.cuda_jacobian = CudaJacobianKind::EdgeAtomic;

    (void)run_solver(options);
}

TEST(CudaSolverSmoke, Fp64Case30VertexWarpConverges)
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
    options.cuda_jacobian = CudaJacobianKind::VertexWarp;

    (void)run_solver(options);
}
#endif
