// ---------------------------------------------------------------------------
// cpu_superlu_solve.cpp — 레퍼런스 선형 솔버 (SuperLU 일회성 분해)
//
// 검증 목적의 선형 솔버 구현. KLU(CpuLinearSolveKLU)와 달리
// symbolic 분석을 재사용하지 않고 매 반복 dgssv()로 일회성 완전 분해를 수행한다.
//
// SuperLU API 호출 순서:
//   1. dCreate_CompCol_Matrix() : J CSC → SuperMatrix A 등록 (값 복사 없음, 포인터 공유)
//   2. rhs 준비: rhs[i] = -F[i]  (J·dx = -F)
//   3. dCreate_Dense_Matrix()   : rhs → SuperMatrix B 등록
//   4. dgssv()                  : DOFACT 모드 — symbolic + numeric + solve 한 번에 수행
//      → perm_c (열 치환), perm_r (행 치환), L, U 는 내부 생성
//   5. B.Store → dx 복사 (SuperLU가 B를 in-place로 갱신하므로 B.Store에 결과 있음)
//   6. 모든 SuperMatrix 자원 해제 (Destroy_*)
//
// analyze() 는 no-op: symbolic 분석이 run() 내부 dgssv()에 포함되어 있음.
// ---------------------------------------------------------------------------

#include "cpu_superlu_solve.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/timer.hpp"

#include <superlu/slu_ddefs.h>

#include <stdexcept>
#include <string>
#include <vector>


CpuLinearSolveSuperLU::CpuLinearSolveSuperLU(IStorage& storage)
    : storage_(static_cast<CpuFp64Storage&>(storage)) {}


void CpuLinearSolveSuperLU::analyze(const AnalyzeContext& ctx)
{
    (void)ctx;
}


void CpuLinearSolveSuperLU::run(IterationContext& ctx)
{
    (void)ctx;

    if (storage_.J.rows() != storage_.dimF || storage_.J.cols() != storage_.dimF) {
        throw std::runtime_error("CpuLinearSolveSuperLU::run: Jacobian shape does not match dimF");
    }
    if (storage_.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuLinearSolveSuperLU::run: Jacobian is empty");
    }

    newton_solver::utils::ScopedTimer timer("CPU.naive.solve.superlu");

    const int n = storage_.J.rows();
    const int nnz = storage_.J.nonZeros();

    // J(CSC) → SuperMatrix A 등록 (포인터 공유, 복사 없음)
    SuperMatrix A;
    dCreate_CompCol_Matrix(
        &A, n, n, nnz,
        const_cast<double*>(storage_.J.valuePtr()),
        const_cast<int*>(storage_.J.innerIndexPtr()),
        const_cast<int*>(storage_.J.outerIndexPtr()),
        SLU_NC, SLU_D, SLU_GE);

    // RHS = -F
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = -storage_.F[static_cast<std::size_t>(i)];
    }

    // dense RHS → SuperMatrix B 등록
    SuperMatrix B;
    dCreate_Dense_Matrix(&B, n, 1, rhs.data(), n, SLU_DN, SLU_D, SLU_GE);

    // DOFACT: symbolic + numeric + solve 를 dgssv() 한 번에 수행
    superlu_options_t options;
    set_default_options(&options);
    options.Fact = DOFACT;

    SuperMatrix L;
    SuperMatrix U;
    std::vector<int> perm_c(static_cast<std::size_t>(n));
    std::vector<int> perm_r(static_cast<std::size_t>(n));

    SuperLUStat_t stat;
    StatInit(&stat);

    int info = 0;
    dgssv(&options, &A, perm_c.data(), perm_r.data(), &L, &U, &B, &stat, &info);

    StatFree(&stat);

    if (info != 0) {
        Destroy_SuperMatrix_Store(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
        throw std::runtime_error(
            "CpuLinearSolveSuperLU::run: SuperLU dgssv failed, info=" +
            std::to_string(info));
    }

    const auto* dn = static_cast<DNformat*>(B.Store);
    const auto* sol = static_cast<double*>(dn->nzval);
    for (int i = 0; i < n; ++i) {
        storage_.dx[static_cast<std::size_t>(i)] = sol[i];
    }

    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
}
