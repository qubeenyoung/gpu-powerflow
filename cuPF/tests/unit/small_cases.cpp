#include "small_cases.hpp"

#include <algorithm>
#include <cstddef>


namespace cupf::tests {

SolverCaseData make_two_bus_case()
{
    SolverCaseData data;
    data.rows = 2;
    data.cols = 2;
    data.indptr = {0, 2, 4};
    data.indices = {0, 1, 0, 1};

    const std::complex<double> y(1.0, -5.0);
    data.ybus_data = {y, -y, -y, y};

    const std::complex<double> slack_v(1.0, 0.0);
    const std::complex<double> pq_v(0.97, -0.05);
    const std::complex<double> pq_current = y * (pq_v - slack_v);
    const std::complex<double> pq_sbus = pq_v * std::conj(pq_current);

    data.sbus = {std::complex<double>(0.0, 0.0), pq_sbus};
    data.v0 = {slack_v, std::complex<double>(1.0, 0.0)};
    data.expected_v = {slack_v, pq_v};
    data.pv = {};
    data.pq = {1};
    return data;
}

std::vector<std::complex<double>> repeat_complex_vector(
    const std::vector<std::complex<double>>& src,
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

}  // namespace cupf::tests
