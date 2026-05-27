#include "glu_lu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include "nicslu.h"
#include "preprocess.h"
}

#include "numeric.h"
#include "symbolic.h"

namespace exp_20260520 {
namespace {

using Clock = std::chrono::steady_clock;

template <typename Func>
double time_ms(Func&& func)
{
    const auto start = Clock::now();
    func();
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

const char* nicslu_status_name(int status)
{
    switch (status) {
    case NICS_OK:
        return "NICS_OK";
    case NICSLU_GENERAL_FAIL:
        return "NICSLU_GENERAL_FAIL";
    case NICSLU_ARGUMENT_ERROR:
        return "NICSLU_ARGUMENT_ERROR";
    case NICSLU_MEMORY_OVERFLOW:
        return "NICSLU_MEMORY_OVERFLOW";
    case NICSLU_FILE_CANNOT_OPEN:
        return "NICSLU_FILE_CANNOT_OPEN";
    case NICSLU_MATRIX_STRUCTURAL_SINGULAR:
        return "NICSLU_MATRIX_STRUCTURAL_SINGULAR";
    case NICSLU_MATRIX_NUMERIC_SINGULAR:
        return "NICSLU_MATRIX_NUMERIC_SINGULAR";
    case NICSLU_MATRIX_INVALID:
        return "NICSLU_MATRIX_INVALID";
    case NICSLU_MATRIX_ENTRY_DUPLICATED:
        return "NICSLU_MATRIX_ENTRY_DUPLICATED";
    case NICSLU_THREADS_NOT_INITIALIZED:
        return "NICSLU_THREADS_NOT_INITIALIZED";
    case NICSLU_MATRIX_NOT_INITIALIZED:
        return "NICSLU_MATRIX_NOT_INITIALIZED";
    case NICSLU_SCHEDULER_NOT_INITIALIZED:
        return "NICSLU_SCHEDULER_NOT_INITIALIZED";
    case NICSLU_SINGLE_THREAD:
        return "NICSLU_SINGLE_THREAD";
    case NICSLU_THREADS_INIT_FAIL:
        return "NICSLU_THREADS_INIT_FAIL";
    case NICSLU_MATRIX_NOT_ANALYZED:
        return "NICSLU_MATRIX_NOT_ANALYZED";
    case NICSLU_MATRIX_NOT_FACTORIZED:
        return "NICSLU_MATRIX_NOT_FACTORIZED";
    case NICSLU_NUMERIC_OVERFLOW:
        return "NICSLU_NUMERIC_OVERFLOW";
    default:
        return "NICSLU_UNKNOWN";
    }
}

void check_nicslu(int status, const char* expression)
{
    if (status != NICS_OK) {
        std::ostringstream oss;
        oss << expression << " failed: " << nicslu_status_name(status)
            << " (" << status << ')';
        throw std::runtime_error(oss.str());
    }
}

struct CscArrays {
    std::vector<double> values;
    std::vector<unsigned int> row_idx;
    std::vector<unsigned int> col_ptr;
};

CscArrays csr_to_csc(const CsrMatrix& matrix)
{
    if (matrix.rows != matrix.cols) {
        throw std::runtime_error("GLU requires a square matrix");
    }
    if (matrix.rows <= 0 || matrix.nnz() <= 0) {
        throw std::runtime_error("GLU requires a non-empty matrix");
    }

    CscArrays csc;
    csc.values.resize(static_cast<std::size_t>(matrix.nnz()));
    csc.row_idx.resize(static_cast<std::size_t>(matrix.nnz()));
    csc.col_ptr.assign(static_cast<std::size_t>(matrix.cols + 1), 0);

    for (const int32_t col : matrix.col_idx) {
        if (col < 0 || col >= matrix.cols) {
            throw std::runtime_error("CSR column index out of range");
        }
        ++csc.col_ptr[static_cast<std::size_t>(col + 1)];
    }
    for (int32_t col = 0; col < matrix.cols; ++col) {
        csc.col_ptr[static_cast<std::size_t>(col + 1)] +=
            csc.col_ptr[static_cast<std::size_t>(col)];
    }

    std::vector<unsigned int> cursor = csc.col_ptr;
    for (int32_t row = 0; row < matrix.rows; ++row) {
        const int32_t row_begin = matrix.row_ptr[static_cast<std::size_t>(row)];
        const int32_t row_end = matrix.row_ptr[static_cast<std::size_t>(row + 1)];
        if (row_begin < 0 || row_end < row_begin || row_end > matrix.nnz()) {
            throw std::runtime_error("CSR row pointer is invalid");
        }
        for (int32_t p = row_begin; p < row_end; ++p) {
            const int32_t col = matrix.col_idx[static_cast<std::size_t>(p)];
            const unsigned int dst = cursor[static_cast<std::size_t>(col)]++;
            csc.row_idx[static_cast<std::size_t>(dst)] = static_cast<unsigned int>(row);
            csc.values[static_cast<std::size_t>(dst)] = matrix.values[static_cast<std::size_t>(p)];
        }
    }
    return csc;
}

bool contains_cuda_failure_text(const std::string& text)
{
    return text.find("failed") != std::string::npos ||
           text.find("No CUDA-capable GPU") != std::string::npos ||
           text.find("skipping GPU factorization") != std::string::npos;
}

void check_factor_values(const Symbolic_Matrix& symbolic)
{
    const auto bad_it = std::find_if(symbolic.val.begin(), symbolic.val.end(), [](REAL value) {
        return !std::isfinite(static_cast<double>(value));
    });
    if (bad_it != symbolic.val.end()) {
        throw std::runtime_error("GLU numeric factorization produced NaN/Inf values");
    }
}

double parse_total_gpu_time_ms(const std::string& text)
{
    const std::string needle = "Total GPU time:";
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return 0.0;
    }

    std::istringstream iss(text.substr(pos + needle.size()));
    double value = 0.0;
    iss >> value;
    return value;
}

}  // namespace

GluFactorization::GluFactorization(GluOptions options)
    : options_(options)
{
}

GluFactorization::~GluFactorization()
{
    symbolic_.reset();
    if (nicslu_ != nullptr) {
        NicsLU_Destroy(nicslu_);
        std::free(nicslu_);
        nicslu_ = nullptr;
    }
}

void GluFactorization::factorize(const CsrMatrix& matrix)
{
    symbolic_.reset();
    if (nicslu_ != nullptr) {
        NicsLU_Destroy(nicslu_);
        std::free(nicslu_);
        nicslu_ = nullptr;
    }
    timings_ = {};
    stats_ = {};
    glu_stdout_.clear();
    glu_stderr_.clear();

    const CscArrays csc = csr_to_csc(matrix);
    stats_.n = matrix.rows;
    stats_.input_nnz = matrix.nnz();

    nicslu_ = static_cast<SNicsLU*>(std::malloc(sizeof(SNicsLU)));
    if (nicslu_ == nullptr) {
        throw std::bad_alloc();
    }

    double* analyzed_values = nullptr;
    unsigned int* analyzed_row_idx = nullptr;
    unsigned int* analyzed_col_ptr = nullptr;

    try {
        timings_.analyze_ms = time_ms([&]() {
            check_nicslu(NicsLU_Initialize(nicslu_), "NicsLU_Initialize");
            check_nicslu(NicsLU_CreateMatrix(nicslu_,
                                             static_cast<uint__t>(matrix.rows),
                                             static_cast<uint__t>(matrix.nnz()),
                                             const_cast<double*>(csc.values.data()),
                                             reinterpret_cast<uint__t*>(const_cast<unsigned int*>(csc.row_idx.data())),
                                             reinterpret_cast<uint__t*>(const_cast<unsigned int*>(csc.col_ptr.data()))),
                         "NicsLU_CreateMatrix");
            nicslu_->cfgi[0] = 1;
            nicslu_->cfgf[1] = 0;
            check_nicslu(NicsLU_Analyze(nicslu_), "NicsLU_Analyze");
            const int dump_status = my_DumpA(nicslu_, &analyzed_values, &analyzed_row_idx, &analyzed_col_ptr);
            if (dump_status != 0) {
                std::ostringstream oss;
                oss << "my_DumpA failed: " << dump_status;
                throw std::runtime_error(oss.str());
            }
        });

        stats_.analyzed_nnz = static_cast<int32_t>(analyzed_col_ptr[static_cast<std::size_t>(nicslu_->n)]);
        std::ostringstream glu_out;
        std::ostringstream glu_err;

        symbolic_ = std::make_unique<Symbolic_Matrix>(nicslu_->n, glu_out, glu_err);
        timings_.symbolic_fill_ms = time_ms([&]() {
            symbolic_->fill_in(analyzed_row_idx, analyzed_col_ptr);
        });
        stats_.symbolic_nnz = static_cast<int32_t>(symbolic_->nnz);

        timings_.symbolic_csr_ms = time_ms([&]() {
            symbolic_->csr();
        });
        timings_.symbolic_predict_ms = time_ms([&]() {
            symbolic_->predictLU(analyzed_row_idx, analyzed_col_ptr, analyzed_values);
        });
        timings_.symbolic_level_ms = time_ms([&]() {
            symbolic_->leveling();
        });
        stats_.num_levels = symbolic_->num_lev;

        std::free(analyzed_values);
        std::free(analyzed_row_idx);
        std::free(analyzed_col_ptr);
        analyzed_values = nullptr;
        analyzed_row_idx = nullptr;
        analyzed_col_ptr = nullptr;

        timings_.numeric_ms = time_ms([&]() {
            LUonDevice(*symbolic_, glu_out, glu_err, options_.perturb);
        });

        glu_stdout_ = glu_out.str();
        glu_stderr_ = glu_err.str();
        timings_.numeric_gpu_event_ms = parse_total_gpu_time_ms(glu_stdout_);
        if (contains_cuda_failure_text(glu_stderr_)) {
            throw std::runtime_error("GLU numeric factorization reported an error: " + glu_stderr_);
        }
        check_factor_values(*symbolic_);
    } catch (...) {
        std::free(analyzed_values);
        std::free(analyzed_row_idx);
        std::free(analyzed_col_ptr);
        throw;
    }
}

std::vector<double> GluFactorization::solve(const std::vector<double>& rhs)
{
    if (!symbolic_ || nicslu_ == nullptr) {
        throw std::runtime_error("GLU factorization has not been created");
    }
    if (rhs.size() != static_cast<std::size_t>(stats_.n)) {
        throw std::runtime_error("RHS length does not match GLU matrix dimension");
    }

    std::vector<REAL> rhs_real(rhs.size());
    std::transform(rhs.begin(), rhs.end(), rhs_real.begin(), [](double value) {
        return static_cast<REAL>(value);
    });

    std::vector<REAL> x_real;
    timings_.solve_ms = time_ms([&]() {
        x_real = symbolic_->solve(nicslu_, rhs_real);
    });

    std::vector<double> x(x_real.size());
    std::transform(x_real.begin(), x_real.end(), x.begin(), [](REAL value) {
        return static_cast<double>(value);
    });
    return x;
}

GluResult solve_with_glu(const CsrMatrix& matrix,
                         const std::vector<double>& rhs,
                         const GluOptions& options)
{
    GluFactorization factorization(options);
    factorization.factorize(matrix);
    GluResult result;
    result.x = factorization.solve(rhs);
    result.timings = factorization.timings();
    result.stats = factorization.stats();
    result.glu_stdout = factorization.glu_stdout();
    result.glu_stderr = factorization.glu_stderr();
    return result;
}

}  // namespace exp_20260520
