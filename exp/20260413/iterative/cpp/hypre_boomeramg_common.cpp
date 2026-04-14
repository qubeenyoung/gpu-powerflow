#include "hypre_boomeramg_common.hpp"

#include <mpi.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp_20260413::iterative::probe {
namespace {

// Keep MPI/HYPRE process lifetime outside individual solves. iterative_probe may
// solve many snapshots in one process, so finalizing after one case would break
// the next case.
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

}  // namespace

void check_hypre(HYPRE_Int code, const char* call)
{
    if (code != 0) {
        throw std::runtime_error(std::string("HYPRE call failed: ") + call +
                                 " code=" + std::to_string(code));
    }
}

void initialize_hypre_runtime()
{
    static HypreRuntime runtime;
    (void)runtime;
}

void clear_hypre_errors()
{
    check_hypre(HYPRE_ClearAllErrors(), "HYPRE_ClearAllErrors");
}

HypreIjMatrix::HypreIjMatrix(const SparseMatrix& matrix)
{
    const int32_t n = static_cast<int32_t>(matrix.rows());
    check_hypre(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &matrix_),
                "HYPRE_IJMatrixCreate");
    check_hypre(HYPRE_IJMatrixSetObjectType(matrix_, HYPRE_PARCSR),
                "HYPRE_IJMatrixSetObjectType");

    std::vector<HYPRE_Int> row_sizes(static_cast<std::size_t>(n));
    for (int32_t row = 0; row < n; ++row) {
        row_sizes[static_cast<std::size_t>(row)] =
            matrix.outerIndexPtr()[row + 1] - matrix.outerIndexPtr()[row];
    }
    check_hypre(HYPRE_IJMatrixSetRowSizes(matrix_, row_sizes.data()),
                "HYPRE_IJMatrixSetRowSizes");
    check_hypre(HYPRE_IJMatrixInitialize(matrix_), "HYPRE_IJMatrixInitialize");

    for (int32_t row = 0; row < n; ++row) {
        const int32_t begin = matrix.outerIndexPtr()[row];
        const int32_t end = matrix.outerIndexPtr()[row + 1];
        HYPRE_Int ncols = end - begin;
        HYPRE_BigInt hypre_row = row;
        std::vector<HYPRE_BigInt> cols(static_cast<std::size_t>(ncols));
        std::vector<HYPRE_Complex> values(static_cast<std::size_t>(ncols));
        for (int32_t k = begin; k < end; ++k) {
            const std::size_t local = static_cast<std::size_t>(k - begin);
            cols[local] = matrix.innerIndexPtr()[k];
            values[local] = matrix.valuePtr()[k];
        }
        check_hypre(HYPRE_IJMatrixSetValues(
                        matrix_, 1, &ncols, &hypre_row, cols.data(), values.data()),
                    "HYPRE_IJMatrixSetValues");
    }

    check_hypre(HYPRE_IJMatrixAssemble(matrix_), "HYPRE_IJMatrixAssemble");

    void* object = nullptr;
    check_hypre(HYPRE_IJMatrixGetObject(matrix_, &object), "HYPRE_IJMatrixGetObject");
    parcsr_ = static_cast<HYPRE_ParCSRMatrix>(object);
}

HypreIjMatrix::~HypreIjMatrix()
{
    if (matrix_ != nullptr) {
        HYPRE_IJMatrixDestroy(matrix_);
    }
}

HypreIjVector::HypreIjVector(const Vector& values)
{
    const int32_t n = static_cast<int32_t>(values.size());
    check_hypre(HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &vector_),
                "HYPRE_IJVectorCreate");
    check_hypre(HYPRE_IJVectorSetObjectType(vector_, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType");
    check_hypre(HYPRE_IJVectorInitialize(vector_), "HYPRE_IJVectorInitialize");

    std::vector<HYPRE_BigInt> indices(static_cast<std::size_t>(n));
    std::vector<HYPRE_Complex> hypre_values(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        indices[static_cast<std::size_t>(i)] = i;
        hypre_values[static_cast<std::size_t>(i)] = values[i];
    }

    check_hypre(HYPRE_IJVectorSetValues(vector_, n, indices.data(), hypre_values.data()),
                "HYPRE_IJVectorSetValues");
    check_hypre(HYPRE_IJVectorAssemble(vector_), "HYPRE_IJVectorAssemble");

    void* object = nullptr;
    check_hypre(HYPRE_IJVectorGetObject(vector_, &object), "HYPRE_IJVectorGetObject");
    parvector_ = static_cast<HYPRE_ParVector>(object);
}

HypreIjVector::HypreIjVector(int32_t n, double initial_value)
{
    check_hypre(HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, n - 1, &vector_),
                "HYPRE_IJVectorCreate");
    check_hypre(HYPRE_IJVectorSetObjectType(vector_, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType");
    check_hypre(HYPRE_IJVectorInitialize(vector_), "HYPRE_IJVectorInitialize");

    std::vector<HYPRE_BigInt> indices(static_cast<std::size_t>(n));
    std::vector<HYPRE_Complex> hypre_values(static_cast<std::size_t>(n), initial_value);
    for (int32_t i = 0; i < n; ++i) {
        indices[static_cast<std::size_t>(i)] = i;
    }

    check_hypre(HYPRE_IJVectorSetValues(vector_, n, indices.data(), hypre_values.data()),
                "HYPRE_IJVectorSetValues");
    check_hypre(HYPRE_IJVectorAssemble(vector_), "HYPRE_IJVectorAssemble");

    void* object = nullptr;
    check_hypre(HYPRE_IJVectorGetObject(vector_, &object), "HYPRE_IJVectorGetObject");
    parvector_ = static_cast<HYPRE_ParVector>(object);
}

HypreIjVector::~HypreIjVector()
{
    if (vector_ != nullptr) {
        HYPRE_IJVectorDestroy(vector_);
    }
}

Vector HypreIjVector::values() const
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

HYPRE_Solver create_boomeramg_preconditioner()
{
    HYPRE_Solver preconditioner = nullptr;
    check_hypre(HYPRE_BoomerAMGCreate(&preconditioner), "HYPRE_BoomerAMGCreate");
    check_hypre(HYPRE_BoomerAMGSetPrintLevel(preconditioner, 0), "HYPRE_BoomerAMGSetPrintLevel");
    check_hypre(HYPRE_BoomerAMGSetLogging(preconditioner, 0), "HYPRE_BoomerAMGSetLogging");
    check_hypre(HYPRE_BoomerAMGSetMaxIter(preconditioner, 1), "HYPRE_BoomerAMGSetMaxIter");
    check_hypre(HYPRE_BoomerAMGSetTol(preconditioner, 0.0), "HYPRE_BoomerAMGSetTol");
    return preconditioner;
}

}  // namespace exp_20260413::iterative::probe
