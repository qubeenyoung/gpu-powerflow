#pragma once

// experimental minimal cuPF NR port

#include <complex>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cupf_minimal {

struct YbusView {
    const int32_t* indptr = nullptr;
    const int32_t* indices = nullptr;
    const std::complex<double>* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t nnz = 0;
};

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

    YbusView ybus() const;
};

DumpCaseData load_dump_case(const std::filesystem::path& case_dir);

}  // namespace cupf_minimal

