#include "third_party_solvers/pangulu_solver.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

typedef unsigned long long int sparse_pointer_t;
typedef unsigned int sparse_index_t;
typedef double sparse_value_t;

#include <pangulu.h>

#include "third_party_solvers/utils.hpp"
#include "tools/timer.hpp"

namespace sparse_direct::solver {
namespace {

std::vector<sparse_pointer_t> to_pangulu_col_ptr(const std::vector<matrix::Index>& values)
{
    std::vector<sparse_pointer_t> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0) {
            throw std::runtime_error("PanguLU does not accept negative sparse pointers");
        }
        converted[i] = static_cast<sparse_pointer_t>(values[i]);
    }
    return converted;
}

std::vector<sparse_index_t> to_pangulu_row_idx(const std::vector<matrix::Index>& values)
{
    std::vector<sparse_index_t> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0) {
            throw std::runtime_error("PanguLU does not accept negative sparse indices");
        }
        converted[i] = static_cast<sparse_index_t>(values[i]);
    }
    return converted;
}

class PanguLuSolver final : public LinearSolver {
public:
    std::string name() const override
    {
        return "pangulu-gpu";
    }

    SolverRun solve(
        const matrix::CsrMatrix&,
        const matrix::CscMatrix& csc,
        const std::vector<matrix::Value>& b) override
    {
        SolverRun result;
        void* pangulu_handle = nullptr;

        try {
            utils::require_square(csc, "PanguLU");
            utils::require_rhs_size(csc.rows, b.size());
            utils::require_cuda_device();
            utils::ensure_mpi_initialized();

            const int reordering_threads = utils::positive_env_or("PANGULU_REORDERING_NTHREAD", 4);
            const int worker_threads = utils::positive_env_or("PANGULU_NTHREAD", 1);
            const int supernode_block_size = utils::positive_env_or("PANGULU_NB", csc.cols);
            const int kernel_warps = utils::positive_env_or("PANGULU_GPU_KERNEL_WARPS", 4);
            const int data_move_warps = utils::positive_env_or("PANGULU_GPU_DATA_MOVE_WARPS", 4);

            std::vector<sparse_pointer_t> col_ptr = to_pangulu_col_ptr(csc.col_ptr);
            std::vector<sparse_index_t> row_idx = to_pangulu_row_idx(csc.row_idx);
            std::vector<sparse_value_t> values = csc.values;
            result.x = b;

            pangulu_init_options init_options{};
            init_options.nb = supernode_block_size;
            init_options.gpu_kernel_warp_per_block = kernel_warps;
            init_options.gpu_data_move_warp_per_block = data_move_warps;
            init_options.nthread = worker_threads;
            init_options.reordering_nthread = reordering_threads;
            init_options.sizeof_value = sizeof(sparse_value_t);
            init_options.is_complex_matrix = 0;
            init_options.mpi_recv_buffer_level = 0.5F;

            MPI_Barrier(MPI_COMM_WORLD);
            timer::Stopwatch phase_timer;
            pangulu_init(
                static_cast<sparse_index_t>(csc.cols),
                static_cast<sparse_pointer_t>(csc.nnz()),
                col_ptr.data(),
                row_idx.data(),
                values.data(),
                &init_options,
                &pangulu_handle);
            MPI_Barrier(MPI_COMM_WORLD);
            result.analysis_ms = phase_timer.elapsed_ms();

            pangulu_gstrf_options factor_options{};
            MPI_Barrier(MPI_COMM_WORLD);
            phase_timer.reset();
            pangulu_gstrf(&factor_options, &pangulu_handle);
            utils::synchronize_cuda("cudaDeviceSynchronize PanguLU factorization");
            MPI_Barrier(MPI_COMM_WORLD);
            result.factor_ms = phase_timer.elapsed_ms();

            pangulu_gstrs_options solve_options{};
            MPI_Barrier(MPI_COMM_WORLD);
            phase_timer.reset();
            pangulu_gstrs(result.x.data(), &solve_options, &pangulu_handle);
            utils::synchronize_cuda("cudaDeviceSynchronize PanguLU solve");
            MPI_Barrier(MPI_COMM_WORLD);
            result.solve_ms = phase_timer.elapsed_ms();

            result.success = true;
            result.message = "ok";
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
            result.x.clear();
        }

        if (pangulu_handle != nullptr) {
            pangulu_finalize(&pangulu_handle);
        }

        return result;
    }
};

}  // namespace

std::unique_ptr<LinearSolver> make_pangulu_gpu_solver()
{
    return std::make_unique<PanguLuSolver>();
}

}  // namespace sparse_direct::solver
