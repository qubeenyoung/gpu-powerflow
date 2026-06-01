#pragma once

#include <cstdint>

extern "C" {
#include <umfpack.h>
}

struct CpuFp64Storage;
struct InitializeContext;
struct IterationContext;


struct CpuLinearSolveUMFPACK {
    CpuLinearSolveUMFPACK();
    ~CpuLinearSolveUMFPACK();

    CpuLinearSolveUMFPACK(CpuLinearSolveUMFPACK&& other) noexcept;
    CpuLinearSolveUMFPACK& operator=(CpuLinearSolveUMFPACK&& other) noexcept;
    CpuLinearSolveUMFPACK(const CpuLinearSolveUMFPACK&) = delete;
    CpuLinearSolveUMFPACK& operator=(const CpuLinearSolveUMFPACK&) = delete;

    void initialize(CpuFp64Storage& buf, const InitializeContext& ctx);
    void prepare_rhs(CpuFp64Storage& buf, IterationContext& ctx);
    void factorize(CpuFp64Storage& buf, IterationContext& ctx);
    void solve(CpuFp64Storage& buf, IterationContext& ctx);
    void solve_transpose(const double* rhs, double* solution, int32_t dim, int32_t nrhs = 1);

    bool supports_transpose_solve() const { return true; }
    bool factorized() const { return numeric_ != nullptr; }

private:
    void release();
    void check_status(int status, const char* where) const;

    void* symbolic_ = nullptr;
    void* numeric_ = nullptr;
    double control_[UMFPACK_CONTROL]{};
    double info_[UMFPACK_INFO]{};
    const int32_t* ap_ = nullptr;
    const int32_t* ai_ = nullptr;
    const double* ax_ = nullptr;
    int32_t dim_ = 0;
    bool has_symbolic_ = false;
};
