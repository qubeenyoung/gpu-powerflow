#include "small_cases.hpp"

#include "newton_solver/core/newton_solver.hpp"

#include <cmath>
#include <complex>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>


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

}  // namespace


int main()
{
    try {
        solve_before_initialize_throws();
        cpu_two_bus_converges();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
