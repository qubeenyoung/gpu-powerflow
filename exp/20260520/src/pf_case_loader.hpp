#ifndef EXP_20260520_PF_CASE_LOADER_HPP
#define EXP_20260520_PF_CASE_LOADER_HPP

#include <complex>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace exp_20260520 {

using Complex = std::complex<double>;

struct YbusMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<Complex> values;
};

struct CaseData {
    std::string case_name;
    YbusMatrix ybus;
    std::vector<Complex> v0;
    std::vector<Complex> sbus;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
};

struct CsrMatrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> values;

    int32_t nnz() const
    {
        return static_cast<int32_t>(values.size());
    }
};

struct LinearSystem {
    CsrMatrix matrix;
    std::vector<double> rhs;
    std::vector<double> x_ref;
    std::string rhs_mode;
};

CaseData load_case(const std::filesystem::path& case_dir, const std::string& case_name);
LinearSystem build_linear_system(const CaseData& data, const std::string& rhs_mode);

std::vector<double> matvec(const CsrMatrix& matrix, const std::vector<double>& x);
double norm2(const std::vector<double>& values);
double relative_error(const std::vector<double>& x, const std::vector<double>& ref);

}  // namespace exp_20260520

#endif
