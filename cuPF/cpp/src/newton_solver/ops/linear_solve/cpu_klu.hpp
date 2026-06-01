#pragma once

#include <cstdint>

extern "C" {
#include <klu.h>
}

struct CpuFp64Storage;
struct InitializeContext;
struct IterationContext;


// ---------------------------------------------------------------------------
// CpuLinearSolveKLU: KLU 기반 희소 직접 솔버 (CPU FP64 경로).
//
// KLU symbolic/numeric 상태를 직접 소유한다. 이전에 CpuFp64Storage에
// 있던 lu, has_klu_symbolic 필드가 여기로 이동되었다.
// ---------------------------------------------------------------------------
struct CpuLinearSolveKLU {
    CpuLinearSolveKLU();
    ~CpuLinearSolveKLU();

    CpuLinearSolveKLU(CpuLinearSolveKLU&& other) noexcept;
    CpuLinearSolveKLU& operator=(CpuLinearSolveKLU&& other) noexcept;
    CpuLinearSolveKLU(const CpuLinearSolveKLU&) = delete;
    CpuLinearSolveKLU& operator=(const CpuLinearSolveKLU&) = delete;

    void initialize(CpuFp64Storage& buf, const InitializeContext& ctx);
    void prepare_rhs(CpuFp64Storage& buf, IterationContext& ctx);
    void factorize(CpuFp64Storage& buf, IterationContext& ctx);
    void solve(CpuFp64Storage& buf, IterationContext& ctx);
    void solve_transpose(const double* rhs, double* solution, int32_t dim, int32_t nrhs = 1);

    bool supports_transpose_solve() const { return true; }
    bool factorized() const { return numeric_ != nullptr; }

private:
    void release();

    klu_symbolic* symbolic_ = nullptr;
    klu_numeric* numeric_ = nullptr;
    klu_common common_{};
    bool has_symbolic_ = false;
};
