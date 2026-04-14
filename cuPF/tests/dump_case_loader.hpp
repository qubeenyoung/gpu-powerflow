#pragma once

#include "newton_solver/core/newton_solver_types.hpp"

#include <complex>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>


namespace cupf::tests {

struct DumpCaseData {
    std::string case_name;

    int32_t rows = 0;
    int32_t cols = 0;

    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<std::complex<double>> ybus_data;

    std::vector<std::complex<double>> sbus;
    std::vector<std::complex<double>> v0;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;

    YbusViewF64 ybus() const
    {
        return YbusViewF64{
            indptr.data(),
            indices.data(),
            ybus_data.data(),
            rows,
            cols,
            static_cast<int32_t>(ybus_data.size()),
        };
    }
};

DumpCaseData load_dump_case(const std::filesystem::path& case_dir);

}  // namespace cupf::tests
