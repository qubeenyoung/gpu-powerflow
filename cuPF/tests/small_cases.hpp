#pragma once

#include "newton_solver/core/newton_solver_types.hpp"

#include <complex>
#include <cstdint>
#include <vector>


namespace cupf::tests {

struct SolverCaseData {
    int32_t rows = 0;
    int32_t cols = 0;

    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<std::complex<double>> ybus_data;

    std::vector<std::complex<double>> sbus;
    std::vector<std::complex<double>> v0;
    std::vector<std::complex<double>> expected_v;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;

    YbusView ybus() const
    {
        return YbusView{
            indptr.data(),
            indices.data(),
            ybus_data.data(),
            rows,
            cols,
            static_cast<int32_t>(ybus_data.size()),
        };
    }
};

SolverCaseData make_two_bus_case();

std::vector<std::complex<double>> repeat_complex_vector(
    const std::vector<std::complex<double>>& src,
    int32_t batch_size);

}  // namespace cupf::tests
