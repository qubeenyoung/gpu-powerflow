#include "benchmark_common.hpp"

#include <mpi.h>
#include <superlu_ddefs.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

extern "C" void pzconvertUROWDATA2skyline() {}

namespace {

struct PhaseOptions {
    linbench::CliOptions cli;
    std::string colperm_name = "NATURAL";
    std::string rowperm_name = "LargeDiag_MC64";
    int nprow = 0;
    int npcol = 0;
    int acc_offload = -1;
    int iter_refine = -1;
    int equil = -1;
    int replace_tiny_pivot = -1;
    int par_symb_fact = -1;
};

struct CscMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int_t> col_ptr;
    std::vector<int_t> row_idx;
    std::vector<double> values;
};

std::string upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

PhaseOptions parse_phase_cli(int argc, char** argv)
{
    PhaseOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--matrix") {
            opt.cli.matrix = need_value(arg);
        } else if (arg == "--rhs") {
            opt.cli.rhs = need_value(arg);
        } else if (arg == "--xref") {
            opt.cli.xref = need_value(arg);
        } else if (arg == "--meta") {
            opt.cli.meta = need_value(arg);
        } else if (arg == "--dtype") {
            opt.cli.dtype = need_value(arg);
        } else if (arg == "--repeats") {
            opt.cli.repeats = std::stoi(need_value(arg));
        } else if (arg == "--warmup") {
            opt.cli.warmup = std::stoi(need_value(arg));
        } else if (arg == "--out") {
            opt.cli.out = need_value(arg);
        } else if (arg == "--config") {
            opt.cli.config = need_value(arg);
        } else if (arg == "--colperm") {
            opt.colperm_name = need_value(arg);
        } else if (arg == "--rowperm") {
            opt.rowperm_name = need_value(arg);
        } else if (arg == "--nprow") {
            opt.nprow = std::stoi(need_value(arg));
        } else if (arg == "--npcol") {
            opt.npcol = std::stoi(need_value(arg));
        } else if (arg == "--acc-offload") {
            opt.acc_offload = std::stoi(need_value(arg));
        } else if (arg == "--iter-refine") {
            const std::string v = upper(need_value(arg));
            if (v == "NO" || v == "NOREFINE") {
                opt.iter_refine = NOREFINE;
            } else if (v == "SINGLE") {
                opt.iter_refine = SLU_SINGLE;
            } else if (v == "DOUBLE") {
                opt.iter_refine = SLU_DOUBLE;
            } else if (v == "EXTRA") {
                opt.iter_refine = SLU_EXTRA;
            } else {
                throw std::runtime_error("unsupported iter-refine value: " + v);
            }
        } else if (arg == "--equil") {
            opt.equil = std::stoi(need_value(arg));
        } else if (arg == "--replace-tiny-pivot") {
            opt.replace_tiny_pivot = std::stoi(need_value(arg));
        } else if (arg == "--par-symb-fact") {
            opt.par_symb_fact = std::stoi(need_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.cli.matrix.empty() || opt.cli.rhs.empty() || opt.cli.xref.empty() ||
        opt.cli.meta.empty() || opt.cli.out.empty()) {
        throw std::runtime_error("matrix, rhs, xref, meta, and out are required");
    }
    if (opt.cli.dtype != "fp64" && opt.cli.dtype != "fp32") {
        throw std::runtime_error("dtype must be fp64 or fp32");
    }
    return opt;
}

colperm_t parse_colperm(const std::string& name)
{
    const std::string u = upper(name);
    if (u == "NATURAL") {
        return NATURAL;
    }
    if (u == "MMD_AT_PLUS_A") {
        return MMD_AT_PLUS_A;
    }
    if (u == "MMD_ATA") {
        return MMD_ATA;
    }
    if (u == "COLAMD") {
        return COLAMD;
    }
    throw std::runtime_error("unsupported SuperLU_DIST ColPerm for this audit: " + name);
}

rowperm_t parse_rowperm(const std::string& name)
{
    const std::string u = upper(name);
    if (u == "LARGEDIAG_MC64") {
        return LargeDiag_MC64;
    }
    if (u == "NOROWPERM") {
        return NOROWPERM;
    }
    if (u == "LARGEDIAG_HWPM") {
        return LargeDiag_HWPM;
    }
    throw std::runtime_error("unsupported SuperLU_DIST RowPerm for this audit: " + name);
}

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
        for (int p = csr.row_ptr[static_cast<std::size_t>(row)];
             p < csr.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            ++csc.col_ptr[static_cast<std::size_t>(csr.col_idx[static_cast<std::size_t>(p)] + 1)];
        }
    }
    for (int col = 0; col < csr.cols; ++col) {
        csc.col_ptr[static_cast<std::size_t>(col + 1)] += csc.col_ptr[static_cast<std::size_t>(col)];
    }
    std::vector<int_t> cursor = csc.col_ptr;
    for (int row = 0; row < csr.rows; ++row) {
        for (int p = csr.row_ptr[static_cast<std::size_t>(row)];
             p < csr.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
            const int col = csr.col_idx[static_cast<std::size_t>(p)];
            const int_t dst = cursor[static_cast<std::size_t>(col)]++;
            csc.row_idx[static_cast<std::size_t>(dst)] = row;
            csc.values[static_cast<std::size_t>(dst)] = csr.values[static_cast<std::size_t>(p)];
        }
    }
    return csc;
}

double mpi_max(double value)
{
    double global = 0.0;
    MPI_Reduce(&value, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global;
}

linbench::Result unsupported_fp32(const PhaseOptions& opt, int nprocs)
{
    const linbench::Meta meta = linbench::load_meta(opt.cli.meta);
    linbench::Result result;
    result.solver_name = "SuperLU_DIST";
    result.solver_version = std::to_string(SUPERLU_DIST_MAJOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_MINOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_PATCH_VERSION);
    result.library_path = SUPERLU_DIST_LIBRARY_PATH_STR;
    result.build_status = "unsupported";
    result.dtype = opt.cli.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = meta.matrix_rows;
    result.matrix_cols = meta.matrix_cols;
    result.nnz = meta.nnz;
    result.repeat_count = 1;
    result.warmup_count = 0;
    result.gpu_resident_after_initial_load = "unsupported";
    result.notes = "SuperLU_DIST phase diagnostic wrapper implements only the FP64 ABglobal path.";
    result.extra_numbers["mpi_ranks"] = static_cast<double>(nprocs);
    result.extra_strings["colperm"] = opt.colperm_name;
    result.extra_strings["rowperm"] = opt.rowperm_name;
    return result;
}

linbench::Result run_superlu_once(const PhaseOptions& opt, int rank, int nprocs)
{
    const auto process_start = linbench::Clock::now();

    double mm_to_csr_ms = 0.0;
    const auto load_start = linbench::Clock::now();
    linbench::CsrMatrix A_csr = linbench::load_matrix_market_csr(opt.cli.matrix, &mm_to_csr_ms);
    std::vector<double> rhs = linbench::load_vector(opt.cli.rhs);
    std::vector<double> x_ref = linbench::load_vector(opt.cli.xref);
    const linbench::Meta meta = linbench::load_meta(opt.cli.meta);
    const auto load_stop = linbench::Clock::now();

    if (A_csr.rows != A_csr.cols || A_csr.rows != static_cast<int>(rhs.size()) ||
        A_csr.cols != static_cast<int>(x_ref.size())) {
        throw std::runtime_error("matrix/vector dimensions are inconsistent");
    }

    const auto convert_start = linbench::Clock::now();
    CscMatrix A_csc = csr_to_csc(A_csr);
    const auto convert_stop = linbench::Clock::now();

    const auto construct_start = linbench::Clock::now();
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
    const auto construct_stop = linbench::Clock::now();

    const int grid_rows = opt.nprow > 0 ? opt.nprow : 1;
    const int grid_cols = opt.npcol > 0 ? opt.npcol : nprocs;
    if (grid_rows * grid_cols != nprocs) {
        throw std::runtime_error("nprow*npcol must match MPI rank count");
    }

    const auto grid_start = linbench::Clock::now();
    gridinfo_t grid;
    superlu_gridinit(MPI_COMM_WORLD, grid_rows, grid_cols, &grid);
    const auto grid_stop = linbench::Clock::now();
    if (grid.iam == -1) {
        throw std::runtime_error("MPI rank is outside the SuperLU_DIST process grid");
    }

    superlu_dist_options_t options;
    set_default_options_dist(&options);
    options.PrintStat = NO;
    options.ColPerm = parse_colperm(opt.colperm_name);
    options.RowPerm = parse_rowperm(opt.rowperm_name);
    options.IterRefine = opt.iter_refine >= 0 ? static_cast<IterRefine_t>(opt.iter_refine) : NOREFINE;
    if (opt.acc_offload >= 0) {
        options.superlu_acc_offload = opt.acc_offload;
    }
    if (opt.equil >= 0) {
        options.Equil = opt.equil ? YES : NO;
    }
    if (opt.replace_tiny_pivot >= 0) {
        options.ReplaceTinyPivot = opt.replace_tiny_pivot ? YES : NO;
    }
    if (opt.par_symb_fact >= 0) {
        options.ParSymbFact = opt.par_symb_fact ? YES : NO;
    }

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
    const auto solve_start = linbench::Clock::now();
    pdgssvx_ABglobal(&options, &A, &scale_perm, b.data(), A_csc.rows, 1, &grid, &lu, berr.data(), &stat, &info);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto solve_stop = linbench::Clock::now();

    const double superlu_call_wall_ms = linbench::elapsed_ms(solve_start, solve_stop);
    const double equil_ms = stat.utime ? stat.utime[EQUIL] * 1000.0 : 0.0;
    const double rowperm_ms = stat.utime ? stat.utime[ROWPERM] * 1000.0 : 0.0;
    const double colperm_ms = stat.utime ? stat.utime[COLPERM] * 1000.0 : 0.0;
    const double symbfac_ms = stat.utime ? stat.utime[SYMBFAC] * 1000.0 : 0.0;
    const double dist_ms = stat.utime ? stat.utime[DIST] * 1000.0 : 0.0;
    const double fact_ms = stat.utime ? stat.utime[FACT] * 1000.0 : 0.0;
    const double solve_ms = stat.utime ? stat.utime[SOLVE] * 1000.0 : 0.0;
    const double sol_tot_ms = stat.utime ? stat.utime[SOL_TOT] * 1000.0 : 0.0;
    const double refine_ms = stat.utime ? stat.utime[REFINE] * 1000.0 : 0.0;
    const double gpu_buffer_mb = static_cast<double>(stat.gpu_buffer) / (1024.0 * 1024.0);
    const double peak_buffer_mb = static_cast<double>(stat.peak_buffer) / (1024.0 * 1024.0);
    const int refine_steps = stat.RefineSteps;

    const auto cleanup_start = linbench::Clock::now();
    PStatFree(&stat);
    dDestroy_LU(A_csc.cols, &grid, &lu);
    dScalePermstructFree(&scale_perm);
    dLUstructFree(&lu);
    superlu_gridexit(&grid);
    Destroy_CompCol_Matrix_dist(&A);
    const auto cleanup_stop = linbench::Clock::now();

    std::vector<double> residual = linbench::residual(A_csr, b, rhs);
    const double abs_res = linbench::norm2(residual);
    const double rhs_norm = linbench::norm2(rhs);
    const double rel_res = abs_res / std::max(rhs_norm, std::numeric_limits<double>::min());
    const double scaled_res = abs_res / std::max(1.0, rhs_norm);
    const double rel_err = linbench::relative_error(b, x_ref);

    const auto process_stop = linbench::Clock::now();

    const double load_ms = mpi_max(linbench::elapsed_ms(load_start, load_stop));
    const double convert_ms = mpi_max(mm_to_csr_ms + linbench::elapsed_ms(convert_start, convert_stop));
    const double construct_ms = mpi_max(linbench::elapsed_ms(construct_start, construct_stop));
    const double grid_ms = mpi_max(linbench::elapsed_ms(grid_start, grid_stop));
    const double call_ms = mpi_max(superlu_call_wall_ms);
    const double cleanup_ms = mpi_max(linbench::elapsed_ms(cleanup_start, cleanup_stop));
    const double total_ms = mpi_max(linbench::elapsed_ms(process_start, process_stop));
    const double global_equil_ms = mpi_max(equil_ms);
    const double global_rowperm_ms = mpi_max(rowperm_ms);
    const double global_colperm_ms = mpi_max(colperm_ms);
    const double global_symbfac_ms = mpi_max(symbfac_ms);
    const double global_dist_ms = mpi_max(dist_ms);
    const double global_fact_ms = mpi_max(fact_ms);
    const double global_solve_ms = mpi_max(solve_ms);
    const double global_sol_tot_ms = mpi_max(sol_tot_ms);
    const double global_refine_ms = mpi_max(refine_ms);
    const double global_gpu_buffer_mb = mpi_max(gpu_buffer_mb);
    const double global_peak_buffer_mb = mpi_max(peak_buffer_mb);

    linbench::Result result;
    result.solver_name = "SuperLU_DIST";
    result.solver_version = std::to_string(SUPERLU_DIST_MAJOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_MINOR_VERSION) + "." +
        std::to_string(SUPERLU_DIST_PATCH_VERSION);
    result.library_path = SUPERLU_DIST_LIBRARY_PATH_STR;
    result.dtype = opt.cli.dtype;
    result.case_name = meta.case_name;
    result.iteration = meta.iteration;
    result.matrix_rows = A_csr.rows;
    result.matrix_cols = A_csr.cols;
    result.nnz = A_csr.nnz;
    result.repeat_count = 1;
    result.warmup_count = 0;
    result.load_ms = load_ms;
    result.format_convert_ms = convert_ms;
    result.h2d_ms = 0.0;
    result.analysis_ms = global_equil_ms + global_rowperm_ms + global_colperm_ms + global_symbfac_ms + global_dist_ms;
    result.factorization_ms = global_fact_ms;
    result.solve_ms = global_solve_ms > 0.0 ? global_solve_ms : global_sol_tot_ms;
    result.d2h_ms = 0.0;
    result.total_solver_ms = call_ms;
    result.total_end_to_end_ms = total_ms;
    result.peak_gpu_memory_mb = global_gpu_buffer_mb;
    result.relative_residual_2 = rel_res;
    result.relative_error_to_x_ref_2 = rel_err;
    result.converged = info == 0 && scaled_res < 1e-8;
    result.num_iterations = 1;
    result.gpu_resident_after_initial_load = "cpu_gpu_hybrid_abglobal";
    result.notes = "One-shot in-process SuperLU_DIST ABglobal diagnostic. Internal phase times come from SuperLUStat_t when populated; total_solver_ms is the measured pdgssvx_ABglobal wall time.";
    result.timing_stats["total_solver_ms"] = linbench::Stats{call_ms, call_ms, call_ms, call_ms, 0.0};
    result.timing_stats["factorization_ms"] = linbench::Stats{global_fact_ms, global_fact_ms, global_fact_ms, global_fact_ms, 0.0};
    result.timing_stats["solve_ms"] = linbench::Stats{result.solve_ms, result.solve_ms, result.solve_ms, result.solve_ms, 0.0};
    result.extra_strings["CUDA_enabled"] = "yes";
    result.extra_strings["MPI_required"] = "yes";
    result.extra_strings["colperm"] = opt.colperm_name;
    result.extra_strings["rowperm"] = opt.rowperm_name;
    result.extra_strings["phase_visibility"] = "SuperLUStat_t_internal_plus_wrapper_wall";
    result.extra_strings["solver_type"] = "distributed sparse direct";
    result.extra_numbers["absolute_residual_2"] = abs_res;
    result.extra_numbers["scaled_residual_2"] = scaled_res;
    result.extra_numbers["rhs_norm_2"] = rhs_norm;
    result.extra_numbers["mpi_ranks"] = static_cast<double>(nprocs);
    result.extra_numbers["nprow"] = static_cast<double>(grid_rows);
    result.extra_numbers["npcol"] = static_cast<double>(grid_cols);
    result.extra_numbers["superlu_info"] = static_cast<double>(info);
    result.extra_numbers["berr"] = berr[0];
    result.extra_numbers["refine_steps"] = static_cast<double>(refine_steps);
    result.extra_numbers["grid_init_ms"] = grid_ms;
    result.extra_numbers["matrix_construction_ms"] = construct_ms;
    result.extra_numbers["cleanup_ms"] = cleanup_ms;
    result.extra_numbers["superlu_call_wall_ms"] = call_ms;
    result.extra_numbers["superlu_equil_ms"] = global_equil_ms;
    result.extra_numbers["superlu_rowperm_ms"] = global_rowperm_ms;
    result.extra_numbers["superlu_colperm_ms"] = global_colperm_ms;
    result.extra_numbers["superlu_symbolic_ms"] = global_symbfac_ms;
    result.extra_numbers["superlu_distribute_ms"] = global_dist_ms;
    result.extra_numbers["superlu_factor_ms"] = global_fact_ms;
    result.extra_numbers["superlu_solve_ms"] = global_solve_ms;
    result.extra_numbers["superlu_sol_tot_ms"] = global_sol_tot_ms;
    result.extra_numbers["superlu_refine_ms"] = global_refine_ms;
    result.extra_numbers["superlu_peak_buffer_mb"] = global_peak_buffer_mb;
    result.extra_numbers["superlu_acc_offload"] = static_cast<double>(options.superlu_acc_offload);
    result.extra_numbers["superlu_equil_enabled"] = options.Equil == YES ? 1.0 : 0.0;
    result.extra_numbers["superlu_replace_tiny_pivot"] = options.ReplaceTinyPivot == YES ? 1.0 : 0.0;
    result.extra_numbers["superlu_par_symb_fact"] = options.ParSymbFact == YES ? 1.0 : 0.0;
    result.extra_numbers["superlu_iter_refine"] = static_cast<double>(options.IterRefine);
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    const auto main_start = linbench::Clock::now();
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int rc = 0;
    PhaseOptions opt;
    linbench::Result result;
    bool have_result = false;
    try {
        opt = parse_phase_cli(argc, argv);
        result = opt.cli.dtype == "fp32" ? unsupported_fp32(opt, nprocs) : run_superlu_once(opt, rank, nprocs);
        have_result = true;
    } catch (const std::exception& exc) {
        if (rank == 0) {
            std::cerr << "SuperLU_DIST phase benchmark failed: " << exc.what() << "\n";
        }
        rc = 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto finalize_start = linbench::Clock::now();
    MPI_Finalize();
    const auto finalize_stop = linbench::Clock::now();

    if (rank == 0 && have_result) {
        result.extra_numbers["mpi_finalize_ms"] = linbench::elapsed_ms(finalize_start, finalize_stop);
        result.extra_numbers["main_wall_ms_including_finalize"] = linbench::elapsed_ms(main_start, finalize_stop);
        linbench::write_result_json(opt.cli.out, result);
    }
    return rc;
}
