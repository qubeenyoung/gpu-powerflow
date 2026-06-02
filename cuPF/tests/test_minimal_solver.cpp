#include "small_cases.hpp"

#include "newton_solver/core/newton_solver.hpp"

#include <cmath>
#include <complex>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(CUPF_WITH_CUDA) && defined(CUPF_ENABLE_CUSTOM_SOLVER)
  #include <cuda_runtime.h>
#endif


namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(std::complex<double> actual,
                  std::complex<double> expected,
                  double tolerance,
                  const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void solve_before_initialize_throws()
{
    const auto data = cupf::tests::make_two_bus_case();
    NewtonSolver solver;
    NRConfig config;
    NRResult result;

    bool threw = false;
    try {
        solver.solve(data.ybus(),
                     data.sbus.data(),
                     data.v0.data(),
                     data.pv.data(), static_cast<int32_t>(data.pv.size()),
                     data.pq.data(), static_cast<int32_t>(data.pq.size()),
                     config,
                     SolveOptions{},
                     result);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    require(threw, "solve() before initialize() must throw");
}

void cpu_two_bus_converges()
{
    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonSolver solver;
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-10;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus CPU solve did not converge");
    require(std::isfinite(result.final_mismatch), "final mismatch must be finite");
    require(result.final_mismatch <= config.tolerance, "final mismatch exceeds tolerance");
    require(result.V.size() == data.expected_v.size(), "unexpected voltage result size");
    require_near(result.V[0], data.expected_v[0], 1e-10, "slack voltage changed");
    require_near(result.V[1], data.expected_v[1], 1e-8, "PQ voltage mismatch");
}

#if defined(CUPF_WITH_CUDA) && defined(CUPF_ENABLE_CUSTOM_SOLVER)
bool cuda_device_available()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void custom_cuda_fp64_two_bus_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP64;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-10;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus custom CUDA FP64 solve did not converge");
    require(std::isfinite(result.final_mismatch), "custom CUDA FP64 final mismatch must be finite");
    require(result.final_mismatch <= config.tolerance,
            "custom CUDA FP64 final mismatch exceeds tolerance");
    require(result.V.size() == data.expected_v.size(),
            "unexpected custom CUDA FP64 voltage result size");
    require_near(result.V[0], data.expected_v[0], 1e-10, "custom CUDA FP64 slack voltage changed");
    require_near(result.V[1], data.expected_v[1], 1e-8, "custom CUDA FP64 PQ voltage mismatch");
}

void custom_cuda_fp32_two_bus_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP32;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus custom CUDA FP32 solve did not converge");
    require(std::isfinite(result.final_mismatch), "custom CUDA FP32 final mismatch must be finite");
    require(result.final_mismatch <= config.tolerance,
            "custom CUDA FP32 final mismatch exceeds tolerance");
}

void custom_cuda_mixed_two_bus_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus custom CUDA Mixed solve did not converge");
    require(std::isfinite(result.final_mismatch), "custom CUDA Mixed final mismatch must be finite");
    require(result.final_mismatch <= config.tolerance,
            "custom CUDA Mixed final mismatch exceeds tolerance");
}

void custom_cuda_mixed_two_bus_batch_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();
    constexpr int32_t batch_size = 2;
    const std::vector<std::complex<double>> sbus = {
        data.sbus[0], data.sbus[1], data.sbus[0], data.sbus[1]};
    const std::vector<std::complex<double>> v0 = {
        data.v0[0], data.v0[1], data.v0[0], data.v0[1]};

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 20;

    NRBatchResult result;
    solver.solve_batch(ybus,
                       sbus.data(), ybus.rows,
                       v0.data(), ybus.rows,
                       batch_size,
                       data.pv.data(), static_cast<int32_t>(data.pv.size()),
                       data.pq.data(), static_cast<int32_t>(data.pq.size()),
                       config,
                       SolveOptions{},
                       result);

    require(result.batch_size == batch_size, "custom CUDA Mixed batch result size mismatch");
    for (int32_t b = 0; b < batch_size; ++b) {
        require(result.converged[b] != 0, "two-bus custom CUDA Mixed batch did not converge");
        require(result.final_mismatch[b] <= config.tolerance,
                "custom CUDA Mixed batch final mismatch exceeds tolerance");
    }
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void custom_cuda_fp32_graph_two_bus_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::FP32;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;
    options.use_cuda_graph = true;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus custom CUDA FP32 graph solve did not converge");
    require(result.final_mismatch <= config.tolerance,
            "custom CUDA FP32 graph final mismatch exceeds tolerance");
}

void custom_cuda_mixed_graph_two_bus_converges()
{
    if (!cuda_device_available()) {
        return;
    }

    const auto data = cupf::tests::make_two_bus_case();
    const YbusView ybus = data.ybus();

    NewtonOptions options;
    options.backend = BackendKind::CUDA;
    options.compute = ComputePolicy::Mixed;
    options.cuda_linear_solver = CudaLinearSolverKind::Custom;
    options.use_cuda_graph = true;

    NewtonSolver solver(options);
    solver.initialize(ybus,
                      data.pv.data(), static_cast<int32_t>(data.pv.size()),
                      data.pq.data(), static_cast<int32_t>(data.pq.size()));

    NRConfig config;
    config.tolerance = 1e-6;
    config.max_iter = 20;

    NRResult result;
    solver.solve(ybus,
                 data.sbus.data(),
                 data.v0.data(),
                 data.pv.data(), static_cast<int32_t>(data.pv.size()),
                 data.pq.data(), static_cast<int32_t>(data.pq.size()),
                 config,
                 SolveOptions{},
                 result);

    require(result.converged, "two-bus custom CUDA Mixed graph solve did not converge");
    require(result.final_mismatch <= config.tolerance,
            "custom CUDA Mixed graph final mismatch exceeds tolerance");
}
#endif
#endif

}  // namespace


int main()
{
    try {
        solve_before_initialize_throws();
        cpu_two_bus_converges();
#if defined(CUPF_WITH_CUDA) && defined(CUPF_ENABLE_CUSTOM_SOLVER)
        custom_cuda_fp64_two_bus_converges();
        custom_cuda_fp32_two_bus_converges();
        custom_cuda_mixed_two_bus_converges();
        custom_cuda_mixed_two_bus_batch_converges();
#ifdef CUPF_ENABLE_CUDA_GRAPH
        custom_cuda_fp32_graph_two_bus_converges();
        custom_cuda_mixed_graph_two_bus_converges();
#endif
#endif
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
