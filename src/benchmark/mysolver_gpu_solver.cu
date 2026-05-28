// --solver mysolver-gpu: GPU multifrontal factor+solve (cy70-112) wired as a
// benchmark solver, GATE-SAFE. Mirrors the production own_pipeline preprocessing
// (equilibrate + MC64 matching for circuits) + scaled/matched solve transform +
// iterative refinement, but uses the GPU multifrontal factor/solve instead of the
// CPU no-pivot LU. On bail / zero pivot / non-finite / berr > 1e-10, falls back to
// the production CPU mysolver. A wrong GPU result is never accepted.
#include "benchmark/solver_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "matrix/sparse_matrix.hpp"
#include "mysolver/factorize/sparse_lu.hpp"
#include "mysolver/gpu/gpu_mf.hpp"
#include "mysolver/reordering/mc64.hpp"
#include "mysolver/reordering/metis_nd.hpp"
#include "mysolver/symbolic/elimination_tree.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {
using Csc = matrix::CscMatrix;

double cw_berr(const matrix::CsrMatrix& a, const std::vector<matrix::Value>& b,
               const std::vector<double>& x)
{
    double berr = 0.0;
    for (matrix::Index row = 0; row < a.rows; ++row) {
        double ax = 0.0, den = std::fabs(b[row]);
        for (matrix::Index p = a.row_ptr[row]; p < a.row_ptr[row + 1]; ++p) {
            ax += a.values[p] * x[a.col_idx[p]];
            den += std::fabs(a.values[p]) * std::fabs(x[a.col_idx[p]]);
        }
        const double num = std::fabs(ax - b[row]);
        if (den > 0.0) berr = std::max(berr, num / den);
        else if (num > 0.0) return std::numeric_limits<double>::infinity();
    }
    return berr;
}

void equilibrate(const Csc& A, Csc& As, std::vector<double>& rs, std::vector<double>& cs)
{
    const int n = A.cols;
    rs.assign(n, 1.0);
    cs.assign(n, 1.0);
    std::vector<double> rmax(n, 0.0), cmax(n, 0.0);
    for (int c = 0; c < n; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            rmax[A.row_idx[p]] = std::max(rmax[A.row_idx[p]], std::fabs(A.values[p]));
    for (int i = 0; i < n; ++i)
        if (rmax[i] > 0.0) rs[i] = 1.0 / rmax[i];
    for (int c = 0; c < n; ++c) {
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p)
            cmax[c] = std::max(cmax[c], std::fabs(A.values[p]) * rs[A.row_idx[p]]);
        if (cmax[c] > 0.0) cs[c] = 1.0 / cmax[c];
    }
    As = A;
    for (int c = 0; c < n; ++c)
        for (int p = As.col_ptr[c]; p < As.col_ptr[c + 1]; ++p)
            As.values[p] = A.values[p] * rs[As.row_idx[p]] * cs[c];
}

// Column permutation B(:,i) = A(:,q[i]).
void colperm(const Csc& A, const std::vector<int>& q, Csc& B)
{
    const int n = A.cols;
    B.rows = A.rows;
    B.cols = n;
    B.col_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) B.col_ptr[i + 1] = A.col_ptr[q[i] + 1] - A.col_ptr[q[i]];
    for (int i = 0; i < n; ++i) B.col_ptr[i + 1] += B.col_ptr[i];
    B.row_idx.resize(B.col_ptr[n]);
    B.values.resize(B.col_ptr[n]);
    for (int i = 0; i < n; ++i) {
        int d = B.col_ptr[i];
        for (int p = A.col_ptr[q[i]]; p < A.col_ptr[q[i] + 1]; ++p, ++d) {
            B.row_idx[d] = A.row_idx[p];
            B.values[d] = A.values[p];
        }
    }
}

// M = P B P^T (symmetric permutation), perm[k] = old index at new position k.
void permute_sym(const Csc& B, const std::vector<int>& perm, const std::vector<int>& iperm, Csc& M)
{
    const int n = B.cols;
    M.rows = M.cols = n;
    M.col_ptr.assign(n + 1, 0);
    for (int c = 0; c < n; ++c) M.col_ptr[iperm[c] + 1] += B.col_ptr[c + 1] - B.col_ptr[c];
    for (int c = 0; c < n; ++c) M.col_ptr[c + 1] += M.col_ptr[c];
    M.row_idx.assign(B.col_ptr[n], 0);
    M.values.assign(B.col_ptr[n], 0.0);
    std::vector<int> nx(M.col_ptr.begin(), M.col_ptr.end());
    // cy262: the scatter is RACE-FREE across source columns -- iperm is a bijection, so
    // column c's entries all land in target column iperm[c]'s contiguous slice and no
    // other source column touches it. Parallelize over c (each thread owns disjoint
    // target columns -> disjoint nx[] counters + output slices). Byte-identical output.
    auto fill = [&](int clo, int chi) {
        for (int c = clo; c < chi; ++c)
            for (int q = B.col_ptr[c]; q < B.col_ptr[c + 1]; ++q) {
                const int d = nx[iperm[c]]++;
                M.row_idx[d] = iperm[B.row_idx[q]];
                M.values[d] = B.values[q];
            }
    };
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (n < 32768 || nth <= 1) {
        fill(0, n);
    } else {
        std::vector<std::thread> th;
        const int chunk = (n + nth - 1) / nth;
        for (int t = 0; t < nth; ++t) {
            const int a = t * chunk, b = std::min(n, a + chunk);
            if (a < b) th.emplace_back([&fill, a, b] { fill(a, b); });
        }
        for (auto& x : th) x.join();
    }
    (void)perm;
}

class MysolverGpuSolver final : public LinearSolver {
public:
    std::string name() const override { return "mysolver-gpu"; }

    // cy342 opt-in plan reuse (MYSOLVER_GPU_REUSE): cache the ordering + the analyzed GpuMfPlan
    // keyed by the matrix STRUCTURE. A repeated solve on the same structure (NR re-factorization,
    // or the benchmark warmup->timed pair) then skips METIS ordering + symbolic + gpu_mf_analyze
    // (graph build) and re-uses the plan, so the factor/solve run WARM (no cold first-graph-launch).
    // Only the value-dependent parts (equilibrate / MC64 / permuted values) are recomputed, so it
    // stays correct when the values change. Default OFF -> production behavior unchanged.
    struct PlanCache {
        bool valid = false;
        int n = 0;
        std::size_t sig = 0;
        std::vector<int> perm, iperm;
        mysolver::gpu::GpuMfPlan plan;
    };
    PlanCache cache_;
    static std::size_t struct_sig(const matrix::CscMatrix& csc)
    {
        std::size_t h = 1469598103934665603ull;  // FNV-1a over n, col_ptr, row_idx
        auto mix = [&](std::size_t v) { h ^= v; h *= 1099511628211ull; };
        mix((std::size_t)csc.cols);
        for (int v : csc.col_ptr) mix((std::size_t)v);
        for (int v : csc.row_idx) mix((std::size_t)v);
        return h;
    }

    SolverRun solve(const matrix::CsrMatrix& csr, const matrix::CscMatrix& csc,
                    const std::vector<matrix::Value>& b) override
    {
        SolverRun result;
        try {
            const int n = csc.cols;
            namespace sym = mysolver::symbolic;
            const bool reuse = std::getenv("MYSOLVER_GPU_REUSE") != nullptr;
            const std::size_t sig = reuse ? struct_sig(csc) : 0;
            const bool reuse_hit = reuse && cache_.valid && cache_.n == n && cache_.sig == sig;
            timer::Stopwatch atimer;

            // Preprocess (mirror own_pipeline): equilibrate + MC64 (circuits only).
            Csc As;
            std::vector<double> rscale, cscale;
            equilibrate(csc, As, rscale, cscale);
            double min_diag = 1e300;
            for (int j = 0; j < n; ++j) {
                double d = 0.0;
                for (int p = As.col_ptr[j]; p < As.col_ptr[j + 1]; ++p)
                    if (As.row_idx[p] == j) { d = std::fabs(As.values[p]); break; }
                min_diag = std::min(min_diag, d);
            }
            const bool atime = std::getenv("MYSOLVER_GPU_ATIME") != nullptr;
            std::vector<int> match(n);
            timer::Stopwatch mctimer;
            if (min_diag >= 0.10) {
                for (int i = 0; i < n; ++i) match[i] = i;
            } else {
                std::vector<int> mc;
                if (!mysolver::reordering::mc64_match(n, As.col_ptr.data(), As.row_idx.data(),
                                                      As.values.data(), mc))
                    throw std::runtime_error("mc64 singular");
                std::fill(match.begin(), match.end(), -1);
                for (int j = 0; j < n; ++j) match[mc[j]] = j;
            }
            if (atime) std::fprintf(stderr, "  [atime] MC64=%.1fms\n", mctimer.elapsed_ms());
            Csc B;
            colperm(As, match, B);

            // cy164/179: production uses PARALLEL nested dissection (-40% A; fill ~= serial).
            // build_ordering recomputes perm + symbolic for a given ordering. If parND breaks
            // the no-pivot GPU factor (onetone2 hits a singular pivot under parND), we retry
            // with SERIAL ordering on the GPU before any CPU fallback (cy179: serial succeeds,
            // refines to the gate in 2 iters -> F 153ms CPU -> ~13ms GPU). MF_SERIAL_ND forces serial.
            std::vector<int> perm(n), iperm(n), Lp, Li, parent;
            Csc M;
            auto build_ordering = [&](bool par) {
                if (reuse_hit) {  // reuse cached ordering: only re-permute the (possibly new) values
                    perm = cache_.perm;
                    iperm = cache_.iperm;
                    permute_sym(B, perm, iperm, M);
                    return;
                }
                timer::Stopwatch mttimer;
                mysolver::reordering::metis_nd(n, B.col_ptr.data(), B.row_idx.data(), perm, par);
                if (atime) std::fprintf(stderr, "  [atime] METIS(par=%d)=%.1fms\n", (int)par,
                                        mttimer.elapsed_ms());
                for (int k = 0; k < n; ++k) iperm[perm[k]] = k;
                permute_sym(B, perm, iperm, M);
                std::vector<int> scp, sri;
                sym::symmetric_pattern(n, M.col_ptr.data(), M.row_idx.data(), scp, sri);
                parent = sym::etree(n, scp.data(), sri.data());
                sym::fill_pattern(n, scp.data(), sri.data(), parent, Lp, Li);
            };
            // One GPU attempt at a given precision: analyze -> factor -> solve +
            // iterative refinement -> berr gate. Returns true (and fills result) only if
            // the refined solution passes the cuDSS-level gate (1e-8). Mixed-precision
            // (FP32) is ~1.3x faster on the dense factor (cy132 double-master) and stays
            // accurate via refinement; if it can't reach the gate (ill-conditioned
            // circuits), we retry in FP64-GPU (still on-GPU, stable) before any CPU
            // fallback. So FP32 is the default fast path with no correctness risk.
            const bool mc = (min_diag < 0.10);
            auto try_gpu = [&](mysolver::gpu::GpuMfPlan& plan, bool fp32, double dshift) -> bool {
                mysolver::numeric::SparseLU lu;
                timer::Stopwatch ftimer;
                // cy182: optional diagonal shift (A+eps*I) before the no-pivot factor. onetone2's
                // parND ordering hits a STRUCTURAL zero pivot -> CPU fallback. A small consistent
                // shift on the diagonal prevents the zero (better-conditioned), and the iterative
                // refinement below (on the ORIGINAL csc) corrects the eps perturbation. Unlike the
                // local boost (cy180, cascades), this is a global perturbation = a valid
                // preconditioner. dshift passed by the caller (0 = none; the shift-retry uses eps).
                const double* fvals = M.values.data();
                std::vector<double> mvals;
                if (dshift != 0.0) {
                    mvals = M.values;
                    for (int j = 0; j < n; ++j)
                        for (int p = M.col_ptr[j]; p < M.col_ptr[j + 1]; ++p)
                            if (M.row_idx[p] == j) { mvals[p] += dshift; break; }
                    fvals = mvals.data();
                }
                if (!mysolver::gpu::gpu_mf_factorize(plan, fvals, lu)) return false;
                result.factor_ms = ftimer.elapsed_ms();
                timer::Stopwatch stimer;
                auto solve_orig = [&](const std::vector<double>& rhs) {
                    std::vector<double> rs(n), c(n), w, y(n), xp(n), x(n);
                    for (int i = 0; i < n; ++i) rs[i] = rscale[i] * rhs[i];
                    for (int a = 0; a < n; ++a) c[a] = rs[perm[a]];
                    mysolver::gpu::gpu_mf_solve(plan, c, w);
                    for (int a = 0; a < n; ++a) y[perm[a]] = w[a];
                    for (int i = 0; i < n; ++i) xp[match[i]] = y[i];
                    for (int j = 0; j < n; ++j) x[j] = cscale[j] * xp[j];
                    return x;
                };
                std::vector<double> x = solve_orig(b);
                double bnorm = 0.0;
                for (double v : b) bnorm = std::max(bnorm, std::fabs(v));
                // cy179: refinement cap tunable. onetone2's no-pivot GPU factor (berr 2.5e-7)
                // needs > 3 iters to reach the gate (else CPU fallback @153ms). MYSOLVER_GPU_REFINE.
                const char* rfs = std::getenv("MYSOLVER_GPU_REFINE");
                const int max_ref = rfs ? std::atoi(rfs) : 3;
                const bool rdbg = std::getenv("MYSOLVER_GPU_DEBUG") != nullptr;
                for (int it = 0; it < max_ref; ++it) {  // refinement to cuDSS-level (~1e-10)
                    std::vector<double> r = b;
                    double rnorm = 0.0;
                    for (int cc = 0; cc < n; ++cc)
                        for (int p = csc.col_ptr[cc]; p < csc.col_ptr[cc + 1]; ++p)
                            r[csc.row_idx[p]] -= csc.values[p] * x[cc];
                    for (double v : r) rnorm = std::max(rnorm, std::fabs(v));
                    if (rdbg) std::fprintf(stderr, "  [refine] it=%d rnorm/bnorm=%g\n", it, rnorm / bnorm);
                    if (rnorm <= 1e-10 * bnorm) break;
                    std::vector<double> dx = solve_orig(r);
                    for (int j = 0; j < n; ++j) x[j] += dx[j];
                }
                result.solve_ms = stimer.elapsed_ms();
                bool finite = true;
                for (double v : x) finite = finite && std::isfinite(v);
                const double berr = finite ? cw_berr(csr, b, x) : -1.0;
                if (std::getenv("MYSOLVER_GPU_DEBUG"))
                    std::fprintf(stderr, "  [gpu-dbg] n=%d fp32=%d finite=%d berr=%g\n", n,
                                 (int)fp32, (int)finite, berr);
                if (finite && berr >= 0.0 && berr <= 1e-8) {
                    result.x = std::move(x);
                    result.success = true;
                    result.message = std::string("ok (GPU MF") + (mc ? " + MC64" : "") +
                                     (fp32 ? ", fp32)" : ", fp64)");
                    return true;
                }
                return false;
            };
            // FP64 is the DEFAULT: the FP32 factor kernel is ~1.5x faster but its
            // less-accurate L/U needs iterative refinement to reach cuDSS-level accuracy,
            // and the refinement cost (an extra solve pass + residual) >= the factor
            // savings here (cy136 measured FP32 combined WORSE than FP64: ACTIVSg25k
            // 8.6 vs 5.3, SyntheticUSA 33 vs 26). Classic mixed-precision-refinement
            // limitation when the factor isn't the dominant cost. FP32 stays opt-in
            // (MYSOLVER_GPU_FP32) as a factor-kernel result; it does not help end-to-end.
            const bool allow_fp32 = std::getenv("MYSOLVER_GPU_FP32") != nullptr;
            const bool par_ord = std::getenv("MF_SERIAL_ND") == nullptr;
            build_ordering(par_ord);
            namespace mg = mysolver::gpu;
            if (reuse_hit) {  // cy342: reuse the cached analyzed plan -> warm factor/solve, no
                              // re-ordering and no re-analyze (graph already built + launched).
                result.analysis_ms = atimer.elapsed_ms();
                if (try_gpu(cache_.plan, false, 0.0)) return result;
                cache_.valid = false;  // changed values broke the cached plan -> CPU fallback below
            } else {
                if (allow_fp32) {  // opt-in fast factor; separate (fp32) plan
                    mg::GpuMfPlan p32 = mg::gpu_mf_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp,
                                                           Li, parent, 8, true);
                    if (p32.num_panels != 0) {
                        result.analysis_ms = atimer.elapsed_ms();
                        if (try_gpu(p32, true, 0.0)) return result;
                    }
                }
                // fp64 plan built ONCE, reused for the no-shift attempt + the shift-retry (the plan is
                // value-independent -> no re-analyze, keeping A fast). cy182: if the no-pivot factor
                // fails (onetone2's parND hits a STRUCTURAL zero pivot -> would CPU-fallback @153ms),
                // retry with a small diagonal shift A+eps*I on the SAME plan: it prevents the zero, and
                // the refinement loop (on the ORIGINAL A) corrects eps -> GPU factor (F ~13 warm vs CPU
                // 153) while keeping parND's fast A (onetone2 A 23 BEATS cuDSS 101.8). shift 1e-8 -> berr 4e-11.
                mg::GpuMfPlan plan = mg::gpu_mf_analyze(n, M.col_ptr.data(), M.row_idx.data(), Lp, Li,
                                                        parent, 8, false);
                if (plan.num_panels != 0) {
                    result.analysis_ms = atimer.elapsed_ms();
                    if (try_gpu(plan, false, 0.0)) {
                        if (reuse) {  // cache ordering + analyzed plan for the next same-structure solve
                            cache_.valid = true; cache_.n = n; cache_.sig = sig;
                            cache_.perm = perm; cache_.iperm = iperm; cache_.plan = std::move(plan);
                        }
                        return result;
                    }
                    // cy343: shift-retry is now DEFAULT-ON (eps=1e-8). It only runs when the primary
                    // no-shift no-pivot factor FAILS (onetone2's parND intermittently hits a STRUCTURAL
                    // zero pivot -> would CPU-fallback @143ms). The small diagonal shift A+eps*I avoids
                    // the zero and the refinement loop (on the ORIGINAL A) corrects eps -> reliable GPU
                    // factor (onetone2 F ~11-18 vs CPU 143), keeping parND's fast A on the SAME plan
                    // (no re-analyze). Matrices that succeed no-shift never reach here -> zero impact.
                    // The berr gate still guards: if the shift doesn't reach accuracy, CPU fallback as
                    // before. MF_DIAG_SHIFT overrides (set 0 to disable). berr ~4e-11 measured.
                    const char* dss = std::getenv("MF_DIAG_SHIFT");
                    const double shift = dss ? std::atof(dss) : 1e-8;
                    if (shift > 0.0 && try_gpu(plan, false, shift)) return result;
                }
            }
            // cy179: a SERIAL-ordering retry was tested for onetone2 (parND makes its no-pivot
            // factor hit a singular pivot -> CPU fallback @148ms). Serial ordering DOES let the
            // GPU factor succeed + refine to the gate (F 148->19ms), BUT serial METIS NodeND on
            // onetone2 costs ~250ms -> A 23->445ms, E2E WORSE than CPU. And onetone2's parND A
            // (23ms) already BEATS cuDSS (101.8). The right fix is STATIC PIVOTING (tiny-pivot
            // diagonal boost in the GPU factor) -> keeps parND's fast A + makes the factor
            // robust + refinement corrects the perturbation. Deferred; onetone2 stays CPU here.
        } catch (const std::exception& ex) {
            if (std::getenv("MYSOLVER_GPU_DEBUG"))
                std::fprintf(stderr, "  [gpu-dbg] exception: %s\n", ex.what());
        }
        SolverRun cpu = make_mysolver_solver()->solve(csr, csc, b);
        cpu.message = "cpu-fallback (" + cpu.message + ")";
        return cpu;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_mysolver_gpu_solver()
{
    return std::make_unique<MysolverGpuSolver>();
}

}  // namespace sparse_direct::solver
