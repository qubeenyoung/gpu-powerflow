#include "benchmark_common.hpp"

#include <mpi.h>
#include <superlu_ddefs.h>

#include <algorithm>
#include <iostream>

extern "C" void pzconvertUROWDATA2skyline() {}

namespace {

struct CscMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int_t> col_ptr;
    std::vector<int_t> row_idx;
    std::vector<double> values;
};

CscMatrix csr_to_csc(const linbench::CsrMatrix& csr)
{
    CscMatrix csc;
    csc.rows = csr.rows;
    csc.cols = csr.cols;
    csc.nnz = csr.nnz;
    csc.col_ptr.assign(static_cast<std::size_t>(csr.cols + 1), 0);
    csc.row_idx.resize(static_cast<std::size_t>(csr.nnz));
    csc.values.resize(static_cast<std::size_t>(csr.nnz));
    for (int row = 0; row < csr.rows; ++row) {
        for (int p = csr.row_ptr[static_cast<std::size_t>(row)]; p < csr.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            ++csc.col_ptr[static_cast<std::size_t>(csr.col_idx[static_cast<std::size_t>(p)] + 1)];
        }
    }
    for (int col = 0; col < csr.cols; ++col) {
        csc.col_ptr[static_cast<std::size_t>(col + 1)] += csc.col_ptr[static_cast<std::size_t>(col)];
    }
    std::vector<int_t> cursor = csc.col_ptr;
    for (int row = 0; row < csr.rows; ++row) {
        for (int p = csr.row_ptr[static_cast<std::size_t>(row)]; p < csr.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            const int col = csr.col_idx[static_cast<std::size_t>(p)];
            const int_t dst = cursor[static_cast<std::size_t>(col)]++;
            csc.row_idx[static_cast<std::size_t>(dst)] = row;
            csc.values[static_cast<std::size_t>(dst)] = csr.values[static_cast<std::size_t>(p)];
        }
    }
    return csc;
}

linbench::Result unsupported_fp32(const linbench::CliOptions& opt)
{
    linbench::Meta meta = linbench::load_meta(opt.meta);
    linbench::Result result;
    result.solver_name = "SuperLU_DIST";
    result.solver_version = "9.2.1";
    result.library_path = SUPERLU_DIST_LIBRARY_PATH_STR;
    result.build_status = "unsupported";
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = meta.matrix_rows;
    result.matrix_cols = meta.matrix_cols;
    result.nnz = meta.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.gpu_resident_after_initial_load = "unsupported";
    result.notes = "Second-pass SuperLU_DIST wrapper implements the FP64 ABglobal path only; FP32 remains a candidate for a follow-up wrapper using psgssvx_ABglobal.";
    return result;
}

linbench::Result run_solver(const linbench::CliOptions& opt, int rank, int nprocs)
{
    const auto total_start = linbench::Clock::now();
    double format_convert_ms = 0.0;
    const auto load_start = linbench::Clock::now();
    linbench::CsrMatrix A_csr = linbench::load_matrix_market_csr(opt.matrix, &format_convert_ms);
    std::vector<double> rhs = linbench::load_vector(opt.rhs);
    std::vector<double> x_ref = linbench::load_vector(opt.xref);
    linbench::Meta meta = linbench::load_meta(opt.meta);
    const auto load_stop = linbench::Clock::now();
    if (A_csr.rows != A_csr.cols || A_csr.rows != static_cast<int>(rhs.size()) ||
        A_csr.cols != static_cast<int>(x_ref.size())) {
        throw std::runtime_error("matrix/vector dimensions are inconsistent");
    }
    const auto convert_start = linbench::Clock::now();
    CscMatrix A_csc = csr_to_csc(A_csr);
    const auto convert_stop = linbench::Clock::now();

    double* nzval = doubleMalloc_dist(A_csc.nnz);
    int_t* rowind = intMalloc_dist(A_csc.nnz);
    int_t* colptr = intMalloc_dist(A_csc.cols + 1);
    if (!nzval || !rowind || !colptr) {
        throw std::runtime_error("SuperLU_DIST allocation failed for CSC matrix");
    }
    std::copy(A_csc.values.begin(), A_csc.values.end(), nzval);
    std::copy(A_csc.row_idx.begin(), A_csc.row_idx.end(), rowind);
    std::copy(A_csc.col_ptr.begin(), A_csc.col_ptr.end(), colptr);

    SuperMatrix A;
    dCreate_CompCol_Matrix_dist(&A, A_csc.rows, A_csc.cols, A_csc.nnz, nzval, rowind, colptr, SLU_NC, SLU_D, SLU_GE);
    gridinfo_t grid;
    superlu_gridinit(MPI_COMM_WORLD, 1, nprocs, &grid);
    if (grid.iam == -1) {
        throw std::runtime_error("MPI rank is outside the SuperLU_DIST process grid");
    }

    std::vector<double> x_solution(static_cast<std::size_t>(A_csc.cols), 0.0);
    auto solve_once = [&]() -> std::pair<double, int> {
        superlu_dist_options_t options;
        set_default_options_dist(&options);
        options.PrintStat = NO;
        options.ColPerm = METIS_AT_PLUS_A;
        options.RowPerm = LargeDiag_MC64;
        options.IterRefine = NOREFINE;
        dScalePermstruct_t scale_perm;
        dLUstruct_t lu;
        SuperLUStat_t stat;
        dScalePermstructInit(A_csc.rows, A_csc.cols, &scale_perm);
        dLUstructInit(A_csc.cols, &lu);
        PStatInit(&stat);
        std::vector<double> b = rhs;
        std::vector<double> berr(1, 0.0);
        int info = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        const auto start = linbench::Clock::now();
        pdgssvx_ABglobal(&options, &A, &scale_perm, b.data(), A_csc.rows, 1, &grid, &lu, berr.data(), &stat, &info);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto stop = linbench::Clock::now();
        PStatFree(&stat);
        dDestroy_LU(A_csc.cols, &grid, &lu);
        dScalePermstructFree(&scale_perm);
        dLUstructFree(&lu);
        if (rank == 0) {
            x_solution = b;
        }
        return {linbench::elapsed_ms(start, stop), info};
    };

    for (int i = 0; i < opt.warmup; ++i) {
        solve_once();
    }
    std::vector<double> solve_ms;
    solve_ms.reserve(static_cast<std::size_t>(opt.repeats));
    int last_info = 0;
    for (int i = 0; i < opt.repeats; ++i) {
        auto [ms, info] = solve_once();
        solve_ms.push_back(ms);
        last_info = info;
    }

    superlu_gridexit(&grid);
    Destroy_CompCol_Matrix_dist(&A);

    linbench::Stats solve_stats = linbench::make_stats(solve_ms);
    std::vector<double> residual = linbench::residual(A_csr, x_solution, rhs);
    const double rel_res = linbench::norm2(residual) /
        std::max(linbench::norm2(rhs), std::numeric_limits<double>::min());
    const double rel_err = linbench::relative_error(x_solution, x_ref);

    const auto total_stop = linbench::Clock::now();
    linbench::Result result;
    result.solver_name = "SuperLU_DIST";
    result.solver_version = std::to_string(SUPERLU_DIST_MAJOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_MINOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_PATCH_VERSION);
    result.library_path = SUPERLU_DIST_LIBRARY_PATH_STR;
    result.dtype = opt.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = A_csr.rows;
    result.matrix_cols = A_csr.cols;
    result.nnz = A_csr.nnz;
    result.repeat_count = opt.repeats;
    result.warmup_count = opt.warmup;
    result.load_ms = linbench::elapsed_ms(load_start, load_stop);
    result.format_convert_ms = format_convert_ms + linbench::elapsed_ms(convert_start, convert_stop);
    result.h2d_ms = 0.0;
    result.analysis_ms = 0.0;
    result.factorization_ms = 0.0;
    result.solve_ms = solve_stats.mean;
    result.d2h_ms = 0.0;
    result.total_solver_ms = solve_stats.mean;
    result.total_end_to_end_ms = linbench::elapsed_ms(total_start, total_stop);
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = last_info == 0 && rel_res < 1e-8;
    result.num_iterations = 1;
    result.gpu_resident_after_initial_load = "cpu_gpu_hybrid_abglobal";
    result.notes = "CUDA-enabled SuperLU_DIST 2D ABglobal direct solve launched under MPI. The public ABglobal driver path is monolithic here; timings include ordering, factorization, solve, and MPI synchronization and are reported as solve_ms.";
    result.timing_stats["solve_ms"] = solve_stats;
    result.timing_stats["total_solver_ms"] = solve_stats;
    result.extra_numbers["mpi_ranks"] = static_cast<double>(nprocs);
    result.extra_numbers["superlu_info"] = static_cast<double>(last_info);
    result.extra_strings["solver_type"] = "distributed sparse direct";
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "yes";
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rc = 0;
    try {
        linbench::CliOptions opt = linbench::parse_cli(argc, argv);
        linbench::Result result = opt.dtype == "fp32" ? unsupported_fp32(opt) : run_solver(opt, rank, nprocs);
        if (rank == 0) {
            linbench::write_result_json(opt.out, result);
        }
    } catch (const std::exception& exc) {
        if (rank == 0) {
            std::cerr << "SuperLU_DIST benchmark failed: " << exc.what() << "\n";
        }
        rc = 2;
    }
    MPI_Finalize();
    return rc;
}
