#include "jfnk_bicgstab.hpp"

#include "newton_solver/core/contexts.hpp"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#ifdef NEWTON_KRYLOV_WITH_HYPRE
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <mpi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace exp_20260414::newton_krylov {
namespace {

using Clock = std::chrono::steady_clock;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;

double elapsed_sec(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

double inf_norm(const Vector& x)
{
    double norm = 0.0;
    for (int64_t i = 0; i < x.size(); ++i) {
        norm = std::max(norm, std::abs(x[i]));
    }
    return norm;
}

double state_inf_norm(const CpuFp64Storage& storage,
                      const int32_t* pv,
                      int32_t n_pv,
                      const int32_t* pq,
                      int32_t n_pq)
{
    double norm = 0.0;
    for (int32_t i = 0; i < n_pv; ++i) {
        norm = std::max(norm, std::abs(std::arg(storage.V[static_cast<std::size_t>(pv[i])])));
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const auto& v = storage.V[static_cast<std::size_t>(pq[i])];
        norm = std::max(norm, std::abs(std::arg(v)));
        norm = std::max(norm, std::abs(std::abs(v)));
    }
    return norm;
}

void copy_static_state(const CpuFp64Storage& src, CpuFp64Storage& dst)
{
    dst.Ybus = src.Ybus;
    dst.Ybus_indptr = src.Ybus_indptr;
    dst.Ybus_indices = src.Ybus_indices;
    dst.Ybus_data = src.Ybus_data;
    dst.Sbus = src.Sbus;
}

void copy_vector_to_storage(const Vector& x, std::vector<double>& out)
{
    out.resize(static_cast<std::size_t>(x.size()));
    for (int64_t i = 0; i < x.size(); ++i) {
        out[static_cast<std::size_t>(i)] = x[i];
    }
}

bool all_finite(const Vector& x)
{
    for (int64_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) {
            return false;
        }
    }
    return true;
}

double safe_pivot(double value, double pivot_tol)
{
    if (std::abs(value) >= pivot_tol) {
        return value;
    }
    return value < 0.0 ? -pivot_tol : pivot_tol;
}

struct BusLocalBlockJacobi {
    std::vector<double> pv_inv;
    std::vector<double> pq_inv00;
    std::vector<double> pq_inv01;
    std::vector<double> pq_inv10;
    std::vector<double> pq_inv11;

    Vector apply(const Vector& rhs, int32_t n_pv, int32_t n_pq) const
    {
        Vector out = rhs;
        for (int32_t i = 0; i < n_pv; ++i) {
            out[i] = pv_inv[static_cast<std::size_t>(i)] * rhs[i];
        }
        for (int32_t i = 0; i < n_pq; ++i) {
            const int32_t p = n_pv + i;
            const int32_t q = n_pv + n_pq + i;
            const std::size_t k = static_cast<std::size_t>(i);
            out[p] = pq_inv00[k] * rhs[p] + pq_inv01[k] * rhs[q];
            out[q] = pq_inv10[k] * rhs[p] + pq_inv11[k] * rhs[q];
        }
        return out;
    }
};

double safe_inverse(double value)
{
    constexpr double kPivotTol = 1e-12;
    if (!std::isfinite(value) || std::abs(value) <= kPivotTol) {
        return 1.0;
    }
    return 1.0 / value;
}

class Ilu0Preconditioner {
public:
    bool compute(const SparseMatrix& matrix, double pivot_tol)
    {
        pivot_tol_ = pivot_tol;
        n_ = static_cast<int32_t>(matrix.rows());
        if (matrix.rows() != matrix.cols()) {
            return false;
        }

        const int32_t nnz = static_cast<int32_t>(matrix.nonZeros());
        row_ptr_.assign(matrix.outerIndexPtr(), matrix.outerIndexPtr() + n_ + 1);
        col_idx_.assign(matrix.innerIndexPtr(), matrix.innerIndexPtr() + nnz);
        lu_.assign(matrix.valuePtr(), matrix.valuePtr() + nnz);
        diag_pos_.assign(static_cast<std::size_t>(n_), -1);
        row_pos_.clear();
        row_pos_.resize(static_cast<std::size_t>(n_));

        for (int32_t row = 0; row < n_; ++row) {
            auto& positions = row_pos_[static_cast<std::size_t>(row)];
            positions.reserve(static_cast<std::size_t>(row_ptr_[row + 1] - row_ptr_[row]));
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                positions[col] = p;
                if (col == row) {
                    diag_pos_[static_cast<std::size_t>(row)] = p;
                }
            }
            if (diag_pos_[static_cast<std::size_t>(row)] < 0) {
                return false;
            }
        }

        for (int32_t row = 0; row < n_; ++row) {
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                double value = lu_[static_cast<std::size_t>(p)];
                const int32_t max_elim_col = std::min(row, col);

                for (int32_t q = row_ptr_[row]; q < row_ptr_[row + 1]; ++q) {
                    const int32_t elim_col = col_idx_[static_cast<std::size_t>(q)];
                    if (elim_col >= max_elim_col) {
                        break;
                    }

                    const auto& elim_row = row_pos_[static_cast<std::size_t>(elim_col)];
                    const auto found = elim_row.find(col);
                    if (found != elim_row.end()) {
                        value -= lu_[static_cast<std::size_t>(q)] *
                                 lu_[static_cast<std::size_t>(found->second)];
                    }
                }

                if (col < row) {
                    const int32_t diag_pos = diag_pos_[static_cast<std::size_t>(col)];
                    double diag = lu_[static_cast<std::size_t>(diag_pos)];
                    if (std::abs(diag) < pivot_tol_) {
                        diag = safe_pivot(diag, pivot_tol_);
                        lu_[static_cast<std::size_t>(diag_pos)] = diag;
                    }
                    lu_[static_cast<std::size_t>(p)] = value / diag;
                } else {
                    if (col == row && std::abs(value) < pivot_tol_) {
                        value = safe_pivot(value, pivot_tol_);
                    }
                    lu_[static_cast<std::size_t>(p)] = value;
                }
            }
        }

        return true;
    }

    Vector solve(const Vector& rhs) const
    {
        Vector y(rhs.size());
        Vector x(rhs.size());
        y.setZero();
        x.setZero();

        for (int32_t row = 0; row < n_; ++row) {
            double sum = rhs[row];
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                if (col >= row) {
                    break;
                }
                sum -= lu_[static_cast<std::size_t>(p)] * y[col];
            }
            y[row] = sum;
        }

        for (int32_t row = n_ - 1; row >= 0; --row) {
            double sum = y[row];
            double diag = 0.0;
            for (int32_t p = row_ptr_[row]; p < row_ptr_[row + 1]; ++p) {
                const int32_t col = col_idx_[static_cast<std::size_t>(p)];
                const double value = lu_[static_cast<std::size_t>(p)];
                if (col == row) {
                    diag = value;
                } else if (col > row) {
                    sum -= value * x[col];
                }
            }
            x[row] = sum / safe_pivot(diag, pivot_tol_);
        }

        return x;
    }

private:
    int32_t n_ = 0;
    double pivot_tol_ = 1e-12;
    std::vector<int32_t> row_ptr_;
    std::vector<int32_t> col_idx_;
    std::vector<double> lu_;
    std::vector<int32_t> diag_pos_;
    std::vector<std::unordered_map<int32_t, int32_t>> row_pos_;
};

std::vector<std::vector<int32_t>> build_jacobian_pattern_by_column(const CpuFp64Storage& storage,
                                                                   const IterationContext& ctx)
{
    const int32_t n_pvpq = ctx.n_pv + ctx.n_pq;
    const int32_t dim = n_pvpq + ctx.n_pq;

    std::vector<int32_t> pvpq_index(static_cast<std::size_t>(storage.n_bus), -1);
    std::vector<int32_t> pq_index(static_cast<std::size_t>(storage.n_bus), -1);
    for (int32_t i = 0; i < ctx.n_pv; ++i) {
        pvpq_index[static_cast<std::size_t>(ctx.pv[i])] = i;
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        pvpq_index[static_cast<std::size_t>(ctx.pq[i])] = ctx.n_pv + i;
        pq_index[static_cast<std::size_t>(ctx.pq[i])] = i;
    }

    std::vector<std::vector<int32_t>> col_rows(static_cast<std::size_t>(dim));
    auto add = [&](int32_t row, int32_t col) {
        if (row >= 0 && col >= 0) {
            col_rows[static_cast<std::size_t>(col)].push_back(row);
        }
    };

    auto add_bus_pair = [&](int32_t row_bus, int32_t col_bus) {
        const int32_t row_p = pvpq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q_local = pq_index[static_cast<std::size_t>(row_bus)];
        const int32_t row_q = row_q_local >= 0 ? n_pvpq + row_q_local : -1;
        const int32_t col_theta = pvpq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm_local = pq_index[static_cast<std::size_t>(col_bus)];
        const int32_t col_vm = col_vm_local >= 0 ? n_pvpq + col_vm_local : -1;

        add(row_p, col_theta);
        add(row_p, col_vm);
        add(row_q, col_theta);
        add(row_q, col_vm);
    };

    for (int32_t row_bus = 0; row_bus < storage.n_bus; ++row_bus) {
        for (int32_t k = storage.Ybus_indptr[static_cast<std::size_t>(row_bus)];
             k < storage.Ybus_indptr[static_cast<std::size_t>(row_bus + 1)];
             ++k) {
            add_bus_pair(row_bus, storage.Ybus_indices[static_cast<std::size_t>(k)]);
        }
    }
    for (int32_t bus = 0; bus < storage.n_bus; ++bus) {
        add_bus_pair(bus, bus);
    }

    for (auto& rows : col_rows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    return col_rows;
}

template <typename ApplyJv>
SparseMatrix assemble_fd_jacobian_matrix(int32_t dim,
                                         const std::vector<std::vector<int32_t>>& col_rows,
                                         ApplyJv apply_jv)
{
    using Triplet = Eigen::Triplet<double, int32_t>;
    std::size_t nnz = 0;
    for (const auto& rows : col_rows) {
        nnz += rows.size();
    }

    std::vector<Triplet> trips;
    trips.reserve(nnz);

    Vector basis = Vector::Zero(dim);
    for (int32_t col = 0; col < dim; ++col) {
        basis[col] = 1.0;
        const Vector j_col = apply_jv(basis);
        basis[col] = 0.0;

        if (!all_finite(j_col)) {
            throw std::runtime_error("nonfinite_fd_preconditioner_column");
        }

        for (int32_t row : col_rows[static_cast<std::size_t>(col)]) {
            trips.emplace_back(row, col, j_col[row]);
        }
    }

    SparseMatrix matrix(dim, dim);
    matrix.setFromTriplets(trips.begin(), trips.end());
    matrix.makeCompressed();
    return matrix;
}

#ifdef NEWTON_KRYLOV_WITH_HYPRE
void check_hypre(HYPRE_Int code, const char* call)
{
    if (code != 0) {
        throw std::runtime_error(std::string("HYPRE call failed: ") + call +
                                 " code=" + std::to_string(code));
    }
}

class HypreRuntime {
public:
    HypreRuntime()
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            int provided = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
            owns_mpi_ = true;
        }
        check_hypre(HYPRE_Init(), "HYPRE_Init");
    }

    ~HypreRuntime()
    {
        HYPRE_Finalize();
        if (owns_mpi_) {
            int mpi_finalized = 0;
            MPI_Finalized(&mpi_finalized);
            if (!mpi_finalized) {
                MPI_Finalize();
            }
        }
    }

private:
    bool owns_mpi_ = false;
};

void initialize_hypre_runtime()
{
    static HypreRuntime runtime;
    (void)runtime;
}

void clear_hypre_errors()
{
    check_hypre(HYPRE_ClearAllErrors(), "HYPRE_ClearAllErrors");
}

class HypreIjMatrix {
public:
    explicit HypreIjMatrix(const SparseMatrix& matrix)
    {
        const int32_t n = static_cast<int32_t>(matrix.rows());
        check_hypre(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &matrix_),
                    "HYPRE_IJMatrixCreate");
        check_hypre(HYPRE_IJMatrixSetObjectType(matrix_, HYPRE_PARCSR),
                    "HYPRE_IJMatrixSetObjectType");

        std::vector<HYPRE_Int> row_sizes(static_cast<std::size_t>(n));
        std::vector<std::vector<HYPRE_BigInt>> row_cols(static_cast<std::size_t>(n));
        std::vector<std::vector<HYPRE_Complex>> row_values(static_cast<std::size_t>(n));

        for (int32_t row = 0; row < n; ++row) {
            for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
                row_cols[static_cast<std::size_t>(row)].push_back(static_cast<HYPRE_BigInt>(it.col()));
                row_values[static_cast<std::size_t>(row)].push_back(it.value());
            }
        }
        for (int32_t row = 0; row < n; ++row) {
            row_sizes[static_cast<std::size_t>(row)] =
                static_cast<HYPRE_Int>(row_cols[static_cast<std::size_t>(row)].size());
        }

        check_hypre(HYPRE_IJMatrixSetRowSizes(matrix_, row_sizes.data()),
                    "HYPRE_IJMatrixSetRowSizes");
        check_hypre(HYPRE_IJMatrixInitialize(matrix_), "HYPRE_IJMatrixInitialize");

        for (int32_t row = 0; row < n; ++row) {
            HYPRE_Int ncols = row_sizes[static_cast<std::size_t>(row)];
            HYPRE_BigInt hypre_row = row;
            check_hypre(HYPRE_IJMatrixSetValues(
                            matrix_,
                            1,
                            &ncols,
                            &hypre_row,
                            row_cols[static_cast<std::size_t>(row)].data(),
                            row_values[static_cast<std::size_t>(row)].data()),
                        "HYPRE_IJMatrixSetValues");
        }

        check_hypre(HYPRE_IJMatrixAssemble(matrix_), "HYPRE_IJMatrixAssemble");

        void* object = nullptr;
        check_hypre(HYPRE_IJMatrixGetObject(matrix_, &object), "HYPRE_IJMatrixGetObject");
        parcsr_ = static_cast<HYPRE_ParCSRMatrix>(object);
    }

    ~HypreIjMatrix()
    {
        if (matrix_ != nullptr) {
            HYPRE_IJMatrixDestroy(matrix_);
        }
    }

    HypreIjMatrix(const HypreIjMatrix&) = delete;
    HypreIjMatrix& operator=(const HypreIjMatrix&) = delete;

    HYPRE_ParCSRMatrix parcsr() const { return parcsr_; }

private:
    HYPRE_IJMatrix matrix_ = nullptr;
    HYPRE_ParCSRMatrix parcsr_ = nullptr;
};

class HypreIjVector {
public:
    explicit HypreIjVector(const Vector& values)
    {
        create(static_cast<int32_t>(values.size()));
        std::vector<HYPRE_BigInt> indices(static_cast<std::size_t>(values.size()));
        std::vector<HYPRE_Complex> hypre_values(static_cast<std::size_t>(values.size()));
        for (int32_t i = 0; i < values.size(); ++i) {
            indices[static_cast<std::size_t>(i)] = i;
            hypre_values[static_cast<std::size_t>(i)] = values[i];
        }
        check_hypre(HYPRE_IJVectorSetValues(
                        vector_,
                        static_cast<HYPRE_Int>(values.size()),
                        indices.data(),
                        hypre_values.data()),
                    "HYPRE_IJVectorSetValues");
        assemble();
    }

    HypreIjVector(int32_t n, double initial_value)
    {
        create(n);
        std::vector<HYPRE_BigInt> indices(static_cast<std::size_t>(n));
        std::vector<HYPRE_Complex> hypre_values(static_cast<std::size_t>(n), initial_value);
        for (int32_t i = 0; i < n; ++i) {
            indices[static_cast<std::size_t>(i)] = i;
        }
        check_hypre(HYPRE_IJVectorSetValues(vector_, n, indices.data(), hypre_values.data()),
                    "HYPRE_IJVectorSetValues");
        assemble();
    }

    ~HypreIjVector()
    {
        if (vector_ != nullptr) {
            HYPRE_IJVectorDestroy(vector_);
        }
    }

    HypreIjVector(const HypreIjVector&) = delete;
    HypreIjVector& operator=(const HypreIjVector&) = delete;

    HYPRE_ParVector parvector() const { return parvector_; }

    Vector values() const
    {
        HYPRE_BigInt lower = 0;
        HYPRE_BigInt upper = -1;
        check_hypre(HYPRE_IJVectorGetLocalRange(vector_, &lower, &upper),
                    "HYPRE_IJVectorGetLocalRange");

        const int32_t n = static_cast<int32_t>(upper - lower + 1);
        std::vector<HYPRE_BigInt> indices(static_cast<std::size_t>(n));
        std::vector<HYPRE_Complex> hypre_values(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            indices[static_cast<std::size_t>(i)] = lower + i;
        }
        check_hypre(HYPRE_IJVectorGetValues(vector_, n, indices.data(), hypre_values.data()),
                    "HYPRE_IJVectorGetValues");

        Vector out(n);
        for (int32_t i = 0; i < n; ++i) {
            out[i] = hypre_values[static_cast<std::size_t>(i)];
        }
        return out;
    }

private:
    void create(int32_t n)
    {
        check_hypre(HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &vector_),
                    "HYPRE_IJVectorCreate");
        check_hypre(HYPRE_IJVectorSetObjectType(vector_, HYPRE_PARCSR),
                    "HYPRE_IJVectorSetObjectType");
        check_hypre(HYPRE_IJVectorInitialize(vector_), "HYPRE_IJVectorInitialize");
    }

    void assemble()
    {
        check_hypre(HYPRE_IJVectorAssemble(vector_), "HYPRE_IJVectorAssemble");
        void* object = nullptr;
        check_hypre(HYPRE_IJVectorGetObject(vector_, &object), "HYPRE_IJVectorGetObject");
        parvector_ = static_cast<HYPRE_ParVector>(object);
    }

    HYPRE_IJVector vector_ = nullptr;
    HYPRE_ParVector parvector_ = nullptr;
};

class HypreAmgPreconditioner {
public:
    ~HypreAmgPreconditioner()
    {
        if (solver_ != nullptr) {
            HYPRE_BoomerAMGDestroy(solver_);
        }
    }

    bool compute(const SparseMatrix& matrix)
    {
        initialize_hypre_runtime();
        matrix_ = std::make_unique<HypreIjMatrix>(matrix);
        check_hypre(HYPRE_BoomerAMGCreate(&solver_), "HYPRE_BoomerAMGCreate");
        check_hypre(HYPRE_BoomerAMGSetPrintLevel(solver_, 0), "HYPRE_BoomerAMGSetPrintLevel");
        check_hypre(HYPRE_BoomerAMGSetLogging(solver_, 0), "HYPRE_BoomerAMGSetLogging");
        check_hypre(HYPRE_BoomerAMGSetMaxIter(solver_, 1), "HYPRE_BoomerAMGSetMaxIter");
        check_hypre(HYPRE_BoomerAMGSetTol(solver_, 0.0), "HYPRE_BoomerAMGSetTol");

        HypreIjVector rhs0(static_cast<int32_t>(matrix.rows()), 0.0);
        HypreIjVector x0(static_cast<int32_t>(matrix.rows()), 0.0);
        const HYPRE_Int setup_code =
            HYPRE_BoomerAMGSetup(solver_, matrix_->parcsr(), rhs0.parvector(), x0.parvector());
        if (setup_code != 0) {
            clear_hypre_errors();
            return false;
        }
        return true;
    }

    Vector solve(const Vector& rhs) const
    {
        HypreIjVector hypre_rhs(rhs);
        HypreIjVector hypre_x(static_cast<int32_t>(rhs.size()), 0.0);
        const HYPRE_Int solve_code =
            HYPRE_BoomerAMGSolve(solver_, matrix_->parcsr(), hypre_rhs.parvector(), hypre_x.parvector());
        if (solve_code != 0) {
            clear_hypre_errors();
        }
        return hypre_x.values();
    }

private:
    std::unique_ptr<HypreIjMatrix> matrix_;
    HYPRE_Solver solver_ = nullptr;
};
#endif

}  // namespace

JfnkLinearSolveBiCGSTAB::JfnkLinearSolveBiCGSTAB(IStorage& storage, JfnkOptions options)
    : storage_(static_cast<CpuFp64Storage&>(storage))
    , scratch_mismatch_(scratch_)
    , scratch_voltage_update_(scratch_)
    , options_(options)
{}

void JfnkLinearSolveBiCGSTAB::analyze(const AnalyzeContext& ctx)
{
    scratch_.prepare(ctx);
    scratch_prepared_ = true;
}

void JfnkLinearSolveBiCGSTAB::run(IterationContext& ctx)
{
    if (!scratch_prepared_) {
        throw std::runtime_error("JfnkLinearSolveBiCGSTAB::run: analyze() must be called first");
    }
    if (storage_.dimF <= 0 || static_cast<int32_t>(storage_.F.size()) != storage_.dimF) {
        throw std::runtime_error("JfnkLinearSolveBiCGSTAB::run: storage is not prepared");
    }
    if (options_.linear_tolerance <= 0.0 || options_.max_inner_iterations <= 0) {
        throw std::runtime_error("JfnkLinearSolveBiCGSTAB::run: invalid linear solver options");
    }
    const bool use_gmres = options_.solver == "gmres_none" || options_.solver == "fgmres";
    const bool use_bicgstab = options_.solver == "bicgstab_none" || options_.solver == "bicgstab";

    if (!use_bicgstab && !use_gmres) {
        throw std::runtime_error("JfnkLinearSolveBiCGSTAB::run: unknown solver " + options_.solver);
    }
    if (options_.preconditioner != "none" &&
        options_.preconditioner != "bus_block_jacobi_fd" &&
        options_.preconditioner != "ilut_fd" &&
        options_.preconditioner != "ilu0_fd" &&
        options_.preconditioner != "amg_fd") {
        throw std::runtime_error(
            "JfnkLinearSolveBiCGSTAB::run: unknown preconditioner " + options_.preconditioner);
    }
#ifndef NEWTON_KRYLOV_WITH_HYPRE
    if (options_.preconditioner == "amg_fd") {
        throw std::runtime_error(
            "JfnkLinearSolveBiCGSTAB::run: amg_fd requires NEWTON_KRYLOV_WITH_HYPRE=ON");
    }
#endif

    const auto solve_start = Clock::now();

    stats_.last_success = false;
    stats_.last_iterations = 0;
    stats_.last_jv_calls = 0;
    stats_.last_estimated_error = std::numeric_limits<double>::quiet_NaN();
    stats_.last_epsilon = 0.0;
    stats_.last_failure_reason.clear();

    copy_static_state(storage_, scratch_);

    Vector base_f(storage_.dimF);
    Vector rhs(storage_.dimF);
    for (int32_t i = 0; i < storage_.dimF; ++i) {
        base_f[i] = storage_.F[static_cast<std::size_t>(i)];
        rhs[i] = -base_f[i];
    }

    const double rhs_norm = std::max(inf_norm(rhs), std::numeric_limits<double>::min());
    const double atol = options_.linear_tolerance * rhs_norm;

    const double state_norm =
        state_inf_norm(storage_, ctx.pv, ctx.n_pv, ctx.pq, ctx.n_pq);

    auto choose_epsilon = [&](const Vector& v) {
        if (!options_.auto_epsilon) {
            return options_.fixed_epsilon;
        }
        const double v_norm = std::max(inf_norm(v), std::numeric_limits<double>::min());
        const double eps = std::sqrt(std::numeric_limits<double>::epsilon()) *
                           (1.0 + state_norm) / v_norm;
        return std::max(eps, 1e-12);
    };

    auto apply_jv = [&](const Vector& v) {
        const auto jv_start = Clock::now();

        const double eps = choose_epsilon(v);
        stats_.last_epsilon = eps;

        scratch_.V = storage_.V;
        std::fill(scratch_.F.begin(), scratch_.F.end(), 0.0);
        std::fill(scratch_.dx.begin(), scratch_.dx.end(), 0.0);
        for (int32_t i = 0; i < storage_.dimF; ++i) {
            scratch_.dx[static_cast<std::size_t>(i)] = eps * v[i];
        }

        IterationContext scratch_ctx{
            .storage = scratch_,
            .config = ctx.config,
            .pv = ctx.pv,
            .n_pv = ctx.n_pv,
            .pq = ctx.pq,
            .n_pq = ctx.n_pq,
            .iter = ctx.iter,
        };

        const auto update_start = Clock::now();
        scratch_voltage_update_.run(scratch_ctx);
        const auto update_end = Clock::now();

        const auto mismatch_start = Clock::now();
        scratch_mismatch_.run(scratch_ctx);
        const auto mismatch_end = Clock::now();

        Vector out(storage_.dimF);
        for (int32_t i = 0; i < storage_.dimF; ++i) {
            out[i] = (scratch_.F[static_cast<std::size_t>(i)] - base_f[i]) / eps;
        }

        const auto jv_end = Clock::now();
        ++stats_.last_jv_calls;
        ++stats_.total_jv_calls;
        stats_.total_jv_update_sec += elapsed_sec(update_start, update_end);
        stats_.total_jv_mismatch_sec += elapsed_sec(mismatch_start, mismatch_end);
        stats_.total_jv_sec += elapsed_sec(jv_start, jv_end);

        return out;
    };

    BusLocalBlockJacobi bus_block_jacobi;
    Eigen::IncompleteLUT<double, int32_t> ilut_preconditioner;
    Ilu0Preconditioner ilu0_preconditioner;
#ifdef NEWTON_KRYLOV_WITH_HYPRE
    HypreAmgPreconditioner amg_preconditioner;
#endif

    if (options_.preconditioner == "bus_block_jacobi_fd") {
        const auto setup_start = Clock::now();

        bus_block_jacobi.pv_inv.assign(static_cast<std::size_t>(ctx.n_pv), 1.0);
        bus_block_jacobi.pq_inv00.assign(static_cast<std::size_t>(ctx.n_pq), 1.0);
        bus_block_jacobi.pq_inv01.assign(static_cast<std::size_t>(ctx.n_pq), 0.0);
        bus_block_jacobi.pq_inv10.assign(static_cast<std::size_t>(ctx.n_pq), 0.0);
        bus_block_jacobi.pq_inv11.assign(static_cast<std::size_t>(ctx.n_pq), 1.0);

        Vector basis = Vector::Zero(storage_.dimF);
        for (int32_t i = 0; i < ctx.n_pv; ++i) {
            basis.setZero();
            basis[i] = 1.0;
            const Vector col = apply_jv(basis);
            bus_block_jacobi.pv_inv[static_cast<std::size_t>(i)] =
                safe_inverse(col[i]);
        }

        for (int32_t i = 0; i < ctx.n_pq; ++i) {
            const int32_t p = ctx.n_pv + i;
            const int32_t q = ctx.n_pv + ctx.n_pq + i;

            basis.setZero();
            basis[p] = 1.0;
            const Vector theta_col = apply_jv(basis);

            basis.setZero();
            basis[q] = 1.0;
            const Vector vm_col = apply_jv(basis);

            const double a = theta_col[p];
            const double c = theta_col[q];
            const double b = vm_col[p];
            const double d = vm_col[q];
            const double det = a * d - b * c;
            const std::size_t k = static_cast<std::size_t>(i);
            if (std::isfinite(det) && std::abs(det) > 1e-12) {
                bus_block_jacobi.pq_inv00[k] = d / det;
                bus_block_jacobi.pq_inv01[k] = -b / det;
                bus_block_jacobi.pq_inv10[k] = -c / det;
                bus_block_jacobi.pq_inv11[k] = a / det;
            }
        }

        const auto setup_end = Clock::now();
        stats_.total_preconditioner_setup_sec += elapsed_sec(setup_start, setup_end);
    }

    if (options_.preconditioner == "ilut_fd" ||
        options_.preconditioner == "ilu0_fd" ||
        options_.preconditioner == "amg_fd") {
        const auto setup_start = Clock::now();
        bool setup_ok = false;

        try {
            const auto col_rows = build_jacobian_pattern_by_column(storage_, ctx);
            const SparseMatrix fd_matrix =
                assemble_fd_jacobian_matrix(storage_.dimF, col_rows, apply_jv);

            if (options_.preconditioner == "ilut_fd") {
                ilut_preconditioner.setDroptol(options_.ilut_drop_tol);
                ilut_preconditioner.setFillfactor(options_.ilut_fill_factor);
                ilut_preconditioner.compute(fd_matrix);
                setup_ok = ilut_preconditioner.info() == Eigen::Success;
            } else if (options_.preconditioner == "ilu0_fd") {
                setup_ok = ilu0_preconditioner.compute(fd_matrix, options_.ilu_pivot_tol);
            } else {
#ifdef NEWTON_KRYLOV_WITH_HYPRE
                setup_ok = amg_preconditioner.compute(fd_matrix);
#else
                setup_ok = false;
#endif
            }
        } catch (const std::exception& ex) {
            stats_.last_failure_reason = std::string("preconditioner_setup_") + ex.what();
            setup_ok = false;
        }

        const auto setup_end = Clock::now();
        stats_.total_preconditioner_setup_sec += elapsed_sec(setup_start, setup_end);

        if (!setup_ok) {
            if (stats_.last_failure_reason.empty()) {
                stats_.last_failure_reason = "preconditioner_setup";
            }
            ++stats_.linear_failures;
            stats_.total_inner_iterations += stats_.last_iterations;
            stats_.max_inner_iterations =
                std::max(stats_.max_inner_iterations, stats_.last_iterations);
            copy_vector_to_storage(Vector::Zero(storage_.dimF), storage_.dx);
            const auto solve_end = Clock::now();
            stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
            return;
        }
    }

    auto apply_preconditioner = [&](const Vector& input) {
        if (options_.preconditioner == "bus_block_jacobi_fd") {
            return bus_block_jacobi.apply(input, ctx.n_pv, ctx.n_pq);
        }
        if (options_.preconditioner == "ilut_fd") {
            return Vector(ilut_preconditioner.solve(input));
        }
        if (options_.preconditioner == "ilu0_fd") {
            return ilu0_preconditioner.solve(input);
        }
#ifdef NEWTON_KRYLOV_WITH_HYPRE
        if (options_.preconditioner == "amg_fd") {
            return amg_preconditioner.solve(input);
        }
#endif
        return Vector(input);
    };

    if (use_gmres) {
        if (options_.gmres_restart <= 0) {
            throw std::runtime_error("JfnkLinearSolveBiCGSTAB::run: --gmres-restart must be positive");
        }

        Vector x = Vector::Zero(storage_.dimF);
        Vector r = rhs;
        const double rhs_norm2 = std::max(rhs.norm(), std::numeric_limits<double>::min());
        const double atol2 = options_.linear_tolerance * rhs_norm2;

        double residual_norm2 = r.norm();
        stats_.last_estimated_error = residual_norm2 / rhs_norm2;
        if (residual_norm2 <= atol2) {
            stats_.last_success = true;
            copy_vector_to_storage(x, storage_.dx);
            const auto solve_end = Clock::now();
            stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
            return;
        }

        while (stats_.last_iterations < options_.max_inner_iterations) {
            const double beta = r.norm();
            if (!std::isfinite(beta)) {
                stats_.last_failure_reason = "nonfinite_residual";
                break;
            }
            if (beta <= atol2) {
                stats_.last_success = true;
                stats_.last_estimated_error = beta / rhs_norm2;
                break;
            }

            const int32_t basis_dim = std::min(
                options_.gmres_restart,
                options_.max_inner_iterations - stats_.last_iterations);

            Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(storage_.dimF, basis_dim + 1);
            Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(basis_dim + 1, basis_dim);
            Eigen::VectorXd g = Eigen::VectorXd::Zero(basis_dim + 1);
            g[0] = beta;
            basis.col(0) = r / beta;

            Vector x_candidate = x;
            bool restart_done = true;

            for (int32_t j = 0; j < basis_dim; ++j) {
                Vector w = apply_jv(apply_preconditioner(basis.col(j)));
                if (!all_finite(w)) {
                    stats_.last_failure_reason = "nonfinite_jv";
                    restart_done = false;
                    break;
                }

                for (int32_t i = 0; i <= j; ++i) {
                    hessenberg(i, j) = basis.col(i).dot(w);
                    w -= hessenberg(i, j) * basis.col(i);
                }

                hessenberg(j + 1, j) = w.norm();
                const bool happy_breakdown =
                    hessenberg(j + 1, j) <= std::numeric_limits<double>::epsilon();
                if (!happy_breakdown) {
                    basis.col(j + 1) = w / hessenberg(j + 1, j);
                }

                ++stats_.last_iterations;

                const auto h = hessenberg.block(0, 0, j + 2, j + 1);
                const auto g_head = g.head(j + 2);
                const Eigen::VectorXd y = h.colPivHouseholderQr().solve(g_head);
                const double residual_estimate = (g_head - h * y).norm();

                x_candidate = x + apply_preconditioner(basis.leftCols(j + 1) * y);
                stats_.last_estimated_error = residual_estimate / rhs_norm2;

                if (stats_.last_estimated_error <= options_.linear_tolerance || happy_breakdown) {
                    x = x_candidate;
                    stats_.last_success = true;
                    restart_done = false;
                    break;
                }
            }

            if (!restart_done) {
                break;
            }

            x = x_candidate;
            const Vector ax = apply_jv(x);
            if (!all_finite(ax)) {
                stats_.last_failure_reason = "nonfinite_restart_residual";
                break;
            }
            r = rhs - ax;
            stats_.last_estimated_error = r.norm() / rhs_norm2;
        }

        if (!stats_.last_success && stats_.last_failure_reason.empty()) {
            stats_.last_failure_reason = "max_inner_iterations";
        }
        if (!stats_.last_success) {
            ++stats_.linear_failures;
        }

        stats_.total_inner_iterations += stats_.last_iterations;
        stats_.max_inner_iterations = std::max(stats_.max_inner_iterations, stats_.last_iterations);

        copy_vector_to_storage(x, storage_.dx);

        const auto solve_end = Clock::now();
        stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
        return;
    }

    Vector x = Vector::Zero(storage_.dimF);
    Vector r = rhs;
    double residual_norm = inf_norm(r);
    stats_.last_estimated_error = residual_norm / rhs_norm;

    if (residual_norm <= atol) {
        stats_.last_success = true;
        copy_vector_to_storage(x, storage_.dx);
        const auto solve_end = Clock::now();
        stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
        return;
    }

    const Vector r_hat = r;
    Vector p = Vector::Zero(storage_.dimF);
    Vector v_vec = Vector::Zero(storage_.dimF);

    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (int32_t iter = 0; iter < options_.max_inner_iterations; ++iter) {
        const double rho = r_hat.dot(r);
        if (!std::isfinite(rho) || std::abs(rho) <= std::numeric_limits<double>::min()) {
            stats_.last_failure_reason = "rho_breakdown";
            break;
        }

        const double beta = (rho / rho_prev) * (alpha / omega);
        p = r + beta * (p - omega * v_vec);

        const Vector p_hat = apply_preconditioner(p);
        v_vec = apply_jv(p_hat);
        if (!all_finite(v_vec)) {
            stats_.last_failure_reason = "nonfinite_jv";
            break;
        }

        const double denom_alpha = r_hat.dot(v_vec);
        if (!std::isfinite(denom_alpha) ||
            std::abs(denom_alpha) <= std::numeric_limits<double>::min()) {
            stats_.last_failure_reason = "alpha_breakdown";
            break;
        }
        alpha = rho / denom_alpha;

        const Vector s = r - alpha * v_vec;
        const double s_norm = inf_norm(s);
        if (s_norm <= atol) {
            x += alpha * p_hat;
            stats_.last_iterations = iter + 1;
            stats_.last_estimated_error = s_norm / rhs_norm;
            stats_.last_success = true;
            break;
        }

        const Vector s_hat = apply_preconditioner(s);
        const Vector t = apply_jv(s_hat);
        if (!all_finite(t)) {
            stats_.last_failure_reason = "nonfinite_jv";
            break;
        }

        const double tt = t.dot(t);
        if (!std::isfinite(tt) || tt <= std::numeric_limits<double>::min()) {
            stats_.last_failure_reason = "omega_breakdown";
            break;
        }
        omega = t.dot(s) / tt;
        if (!std::isfinite(omega) || std::abs(omega) <= std::numeric_limits<double>::min()) {
            stats_.last_failure_reason = "omega_breakdown";
            break;
        }

        x += alpha * p_hat + omega * s_hat;
        r = s - omega * t;

        residual_norm = inf_norm(r);
        stats_.last_iterations = iter + 1;
        stats_.last_estimated_error = residual_norm / rhs_norm;
        if (!all_finite(x) || !std::isfinite(residual_norm)) {
            stats_.last_failure_reason = "nonfinite_iterate";
            break;
        }
        if (residual_norm <= atol) {
            stats_.last_success = true;
            break;
        }

        rho_prev = rho;
    }

    if (!stats_.last_success && stats_.last_failure_reason.empty()) {
        stats_.last_failure_reason = "max_inner_iterations";
    }
    if (!stats_.last_success) {
        ++stats_.linear_failures;
    }

    stats_.total_inner_iterations += stats_.last_iterations;
    stats_.max_inner_iterations = std::max(stats_.max_inner_iterations, stats_.last_iterations);

    copy_vector_to_storage(x, storage_.dx);

    const auto solve_end = Clock::now();
    stats_.total_solve_sec += elapsed_sec(solve_start, solve_end);
}

}  // namespace exp_20260414::newton_krylov
