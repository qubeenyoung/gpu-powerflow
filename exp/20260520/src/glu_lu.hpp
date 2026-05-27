#ifndef EXP_20260520_GLU_LU_HPP
#define EXP_20260520_GLU_LU_HPP

#include "pf_case_loader.hpp"

#include <memory>
#include <string>
#include <vector>

struct tagSNicsLU;
typedef struct tagSNicsLU SNicsLU;
class Symbolic_Matrix;

namespace exp_20260520 {

struct GluTimings {
    double analyze_ms = 0.0;
    double symbolic_fill_ms = 0.0;
    double symbolic_csr_ms = 0.0;
    double symbolic_predict_ms = 0.0;
    double symbolic_level_ms = 0.0;
    double numeric_ms = 0.0;
    double numeric_gpu_event_ms = 0.0;
    double solve_ms = 0.0;
};

struct GluStats {
    int32_t n = 0;
    int32_t input_nnz = 0;
    int32_t analyzed_nnz = 0;
    int32_t symbolic_nnz = 0;
    int32_t num_levels = 0;
};

struct GluOptions {
    bool perturb = false;
    bool keep_glu_log = false;
};

struct GluResult {
    std::vector<double> x;
    GluTimings timings;
    GluStats stats;
    std::string glu_stdout;
    std::string glu_stderr;
};

class GluFactorization {
public:
    explicit GluFactorization(GluOptions options);
    ~GluFactorization();

    GluFactorization(const GluFactorization&) = delete;
    GluFactorization& operator=(const GluFactorization&) = delete;

    void factorize(const CsrMatrix& matrix);
    std::vector<double> solve(const std::vector<double>& rhs);

    const GluTimings& timings() const { return timings_; }
    const GluStats& stats() const { return stats_; }
    const std::string& glu_stdout() const { return glu_stdout_; }
    const std::string& glu_stderr() const { return glu_stderr_; }

private:
    GluOptions options_;
    SNicsLU* nicslu_ = nullptr;
    std::unique_ptr<Symbolic_Matrix> symbolic_;
    GluTimings timings_;
    GluStats stats_;
    std::string glu_stdout_;
    std::string glu_stderr_;
};

GluResult solve_with_glu(const CsrMatrix& matrix,
                         const std::vector<double>& rhs,
                         const GluOptions& options);

}  // namespace exp_20260520

#endif
