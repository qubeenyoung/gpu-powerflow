#include "mysolver/solver.hpp"

#include <stdexcept>

#include <suitesparse/amd.h>
#include <suitesparse/klu.h>

#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"

// M0/M1 backend: SuiteSparse KLU behind the cuDSS phase contract.
//
// M1 keeps two candidate orderings — KLU's native AMD and mysolver's METIS ND —
// and factorize() picks whichever produces the smaller fill. METIS ND is not
// uniformly better than AMD (e.g. it doubles fill on the onetone2 circuit
// matrix), so selecting the min-fill ordering guarantees mysolver is never worse
// than the AMD baseline while keeping ND's wins (wang3, rajat15). KLU still does
// BTF + numeric; klu_common is needed to free its symbolic/numeric, so each
// ordering owns its own common.
namespace mysolver {
namespace {

struct KluOrdering {
    klu_common common{};
    klu_symbolic* symbolic = nullptr;
    int ordering_code = 0;  // 0 = AMD, 3 = METIS ND (user_order)

    explicit KluOrdering(int code)
    {
        klu_defaults(&common);
        common.ordering = code;
        if (code == 3) {
            common.user_order = &klu_metis_user_order;
        }
        ordering_code = code;
    }
    ~KluOrdering()
    {
        if (symbolic) {
            klu_free_symbolic(&symbolic, &common);
        }
    }
    KluOrdering(const KluOrdering&) = delete;
    KluOrdering& operator=(const KluOrdering&) = delete;
};

struct KluAnalysis {
    int n = 0;
    std::vector<std::shared_ptr<KluOrdering>> candidates;  // AMD, METIS ND
};

struct KluFactor {
    std::shared_ptr<KluOrdering> chosen;  // keeps the winning common + symbolic alive
    klu_numeric* numeric = nullptr;

    KluFactor() = default;
    ~KluFactor()
    {
        if (numeric && chosen) {
            klu_free_numeric(&numeric, &chosen->common);
        }
    }
    KluFactor(const KluFactor&) = delete;
    KluFactor& operator=(const KluFactor&) = delete;
};

}  // namespace

AnalyzeResult analyze(const sparse_direct::matrix::CscMatrix& A)
{
    auto state = std::make_shared<KluAnalysis>();
    const int n = A.cols;
    state->n = n;

    // Predicted fill only decides whether METIS ND is worth *trying*; the final
    // choice is by ACTUAL fill in factorize() (reliable, gate-A-safe). This
    // prunes the second factorization on AMD-favoured matrices (single factor,
    // e.g. the case6468rte KPI) while still trying ND where it may help. The
    // whole-matrix prediction ignores KLU's BTF blocking and can be wrong
    // (e.g. onetone2), so it must not be trusted as the final selector.
    bool try_nd = false;
    if (n > 1) {
        std::vector<int> sym_cp, sym_ri;
        symbolic::symmetric_pattern(n, A.col_ptr.data(), A.row_idx.data(), sym_cp, sym_ri);
        std::vector<int> amd_perm(n);
        double info[AMD_INFO];
        const int amd_status =
            amd_order(n, sym_cp.data(), sym_ri.data(), amd_perm.data(), nullptr, info);
        std::vector<int> nd_perm;
        const bool nd_ok = reordering::metis_nd(n, A.col_ptr.data(), A.row_idx.data(), nd_perm);
        if (amd_status == AMD_OK && nd_ok) {
            const long fill_amd =
                symbolic::predicted_fill_perm(n, A.col_ptr.data(), A.row_idx.data(), amd_perm);
            const long fill_nd =
                symbolic::predicted_fill_perm(n, A.col_ptr.data(), A.row_idx.data(), nd_perm);
            try_nd = (fill_nd < fill_amd);
        }
    }

    // AMD (== KLU's ordering=0) is always a candidate; ND is added only when
    // predicted to help. factorize() keeps whichever yields the smaller actual fill.
    std::vector<int> codes = {0};
    if (try_nd) {
        codes.push_back(3);
    }
    for (int code : codes) {
        auto ordering = std::make_shared<KluOrdering>(code);
        ordering->symbolic = klu_analyze(
            n, const_cast<int*>(A.col_ptr.data()), const_cast<int*>(A.row_idx.data()),
            &ordering->common);
        if (ordering->symbolic) {
            state->candidates.push_back(std::move(ordering));
        }
    }
    if (state->candidates.empty()) {
        throw std::runtime_error("mysolver::analyze: klu_analyze failed");
    }
    return AnalyzeResult{std::move(state)};
}

void factorize(const sparse_direct::matrix::CscMatrix& A,
               const AnalyzeResult& a,
               FactorState* out)
{
    if (!a.valid() || out == nullptr) {
        throw std::runtime_error("mysolver::factorize: invalid arguments");
    }
    auto analysis = std::static_pointer_cast<KluAnalysis>(a.impl);

    // Factor with each candidate ordering; keep the one with the smallest fill.
    std::shared_ptr<KluOrdering> best;
    klu_numeric* best_numeric = nullptr;
    long best_fill = -1;
    for (const std::shared_ptr<KluOrdering>& candidate : analysis->candidates) {
        klu_numeric* numeric = klu_factor(
            const_cast<int*>(A.col_ptr.data()),
            const_cast<int*>(A.row_idx.data()),
            const_cast<double*>(A.values.data()),
            candidate->symbolic,
            &candidate->common);
        if (!numeric) {
            continue;
        }
        const long fill = static_cast<long>(numeric->lnz) + static_cast<long>(numeric->unz);
        if (best_numeric == nullptr || fill < best_fill) {
            if (best_numeric) {
                klu_free_numeric(&best_numeric, &best->common);
            }
            best = candidate;
            best_numeric = numeric;
            best_fill = fill;
        } else {
            klu_free_numeric(&numeric, &candidate->common);
        }
    }
    if (!best_numeric) {
        throw std::runtime_error("mysolver::factorize: klu_factor failed for all orderings");
    }

    auto factor = std::make_shared<KluFactor>();
    factor->chosen = best;
    factor->numeric = best_numeric;
    out->lnz = best_numeric->lnz;
    out->unz = best_numeric->unz;
    out->ordering = best->ordering_code;
    out->impl = std::move(factor);
}

void solve(const FactorState& f,
           const AnalyzeResult& a,
           const std::vector<double>& rhs,
           std::vector<double>& x_out)
{
    if (!f.valid() || !a.valid()) {
        throw std::runtime_error("mysolver::solve: invalid state");
    }
    auto analysis = std::static_pointer_cast<KluAnalysis>(a.impl);
    auto factor = std::static_pointer_cast<KluFactor>(f.impl);

    x_out = rhs;  // KLU solves in place: rhs is overwritten with the solution.
    const int ok = klu_solve(
        factor->chosen->symbolic,
        factor->numeric,
        analysis->n,
        1,
        x_out.data(),
        &factor->chosen->common);
    if (!ok) {
        throw std::runtime_error("mysolver::solve: klu_solve failed");
    }
}

}  // namespace mysolver
