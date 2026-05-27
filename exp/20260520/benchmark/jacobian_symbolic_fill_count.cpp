#include "pf_case_loader.hpp"

extern "C" {
#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"
#include "reordering/metis_nd.h"
#include "symbolic/symbolic_factorize.h"
#include "symbolic/symbolic_supernode.h"
}

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path case_dir;
    std::string case_name;
    std::string rhs_mode = "synthetic";
    bool csv = false;
    bool validate_symmetry = true;
    bool apply_inv_perm = true;
};

struct ScopedCsr {
    CSRMatrix matrix{};
    ~ScopedCsr() { csr_destroy(&matrix); }
    ScopedCsr() = default;
    ScopedCsr(const ScopedCsr&) = delete;
    ScopedCsr& operator=(const ScopedCsr&) = delete;
    ScopedCsr(ScopedCsr&& other) noexcept : matrix(other.matrix)
    {
        std::memset(&other.matrix, 0, sizeof(other.matrix));
    }
    ScopedCsr& operator=(ScopedCsr&& other) noexcept
    {
        if (this != &other) {
            csr_destroy(&matrix);
            matrix = other.matrix;
            std::memset(&other.matrix, 0, sizeof(other.matrix));
        }
        return *this;
    }
};

struct ScopedCsc {
    CSCMatrix matrix{};
    ~ScopedCsc() { csc_destroy(&matrix); }
    ScopedCsc() = default;
    ScopedCsc(const ScopedCsc&) = delete;
    ScopedCsc& operator=(const ScopedCsc&) = delete;
};

struct ScopedSupernodes {
    SymbolicSupernodeSet set{};
    ~ScopedSupernodes() { symbolic_supernodes_destroy(&set); }
    ScopedSupernodes() = default;
    ScopedSupernodes(const ScopedSupernodes&) = delete;
    ScopedSupernodes& operator=(const ScopedSupernodes&) = delete;
};

template <typename Func>
double time_ms(Func&& func)
{
    const auto start = Clock::now();
    func();
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

[[noreturn]] void die_sds(int status, const char* expr)
{
    throw std::runtime_error(std::string(expr) + " failed with status " +
                             std::to_string(status));
}

void check_sds(int status, const char* expr)
{
    if (status != SDS_OK) {
        die_sds(status, expr);
    }
}

void print_usage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " --case-dir PATH [options]\n\n"
        << "Options:\n"
        << "  --case NAME\n"
        << "  --rhs-mode synthetic|mismatch\n"
        << "  --csv\n"
        << "  --apply-inv-perm\n"
        << "  --apply-metis-perm\n"
        << "  --no-validate-symmetry\n";
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--case-dir") {
            options.case_dir = need_value("--case-dir");
        } else if (arg == "--case") {
            options.case_name = need_value("--case");
        } else if (arg == "--rhs-mode") {
            options.rhs_mode = need_value("--rhs-mode");
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--apply-inv-perm") {
            options.apply_inv_perm = true;
        } else if (arg == "--apply-metis-perm") {
            options.apply_inv_perm = false;
        } else if (arg == "--no-validate-symmetry") {
            options.validate_symmetry = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.case_dir.empty()) {
        throw std::runtime_error("--case-dir is required");
    }
    if (options.case_name.empty()) {
        options.case_name = options.case_dir.filename().string();
    }
    if (options.rhs_mode != "synthetic" && options.rhs_mode != "mismatch") {
        throw std::runtime_error("--rhs-mode must be synthetic or mismatch");
    }
    return options;
}

ScopedCsr make_project_csr(const exp_20260520::CsrMatrix& matrix)
{
    ScopedCsr out;
    check_sds(csr_create(&out.matrix, matrix.rows, matrix.cols, matrix.nnz()),
              "csr_create");
    std::memcpy(out.matrix.rowptr,
                matrix.row_ptr.data(),
                static_cast<std::size_t>(matrix.rows + 1) * sizeof(int));
    std::memcpy(out.matrix.colind,
                matrix.col_idx.data(),
                static_cast<std::size_t>(matrix.nnz()) * sizeof(int));
    std::memcpy(out.matrix.values,
                matrix.values.data(),
                static_cast<std::size_t>(matrix.nnz()) * sizeof(double));
    return out;
}

void print_csv_header()
{
    std::cout
        << "case_name,n_bus,n_pv,n_pq,linear_dim,linear_nnz,reordered_nnz,"
        << "L_nnz,U_nnz,LU_unique_nnz,fill_in_nnz,fill_ratio,extra_fill_ratio,"
        << "L_strict_tail_supernodes,L_strict_tail_max_width,"
        << "ordering_ms,symbolic_ms,supernode_ms\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_args(argc, argv);
        const exp_20260520::CaseData data =
            exp_20260520::load_case(options.case_dir, options.case_name);
        const exp_20260520::LinearSystem system =
            exp_20260520::build_linear_system(data, options.rhs_mode);

        ScopedCsr matrix = make_project_csr(system.matrix);
        ScopedCsr graph;
        ScopedCsr matrix_perm;
        ScopedCsc L;
        ScopedCsc U;
        ScopedSupernodes l_supernodes;
        SymbolicFactorizeStats stats{};
        std::vector<int> perm(static_cast<std::size_t>(matrix.matrix.nrows), 0);
        std::vector<int> inv_perm(static_cast<std::size_t>(matrix.matrix.nrows), 0);

        const double ordering_ms = time_ms([&]() {
            check_sds(build_metis_graph(&matrix.matrix,
                                        &graph.matrix,
                                        options.validate_symmetry ? 1 : 0),
                      "build_metis_graph");
            check_sds(compute_metis_nd_ordering(&graph.matrix,
                                                perm.data(),
                                                inv_perm.data()),
                      "compute_metis_nd_ordering");
            const int* reorder_perm =
                options.apply_inv_perm ? inv_perm.data() : perm.data();
            check_sds(apply_symmetric_permutation(&matrix.matrix,
                                                  reorder_perm,
                                                  &matrix_perm.matrix),
                      "apply_symmetric_permutation");
        });

        const double symbolic_ms = time_ms([&]() {
            check_sds(symbolic_factorize_reordered_csr_with_stats(&matrix_perm.matrix,
                                                                  &L.matrix,
                                                                  &U.matrix,
                                                                  &stats),
                      "symbolic_factorize_reordered_csr_with_stats");
        });

        const double supernode_ms = time_ms([&]() {
            check_sds(symbolic_supernodes_find_strict(&L.matrix,
                                                      &l_supernodes.set),
                      "symbolic_supernodes_find_strict");
        });

        const int lu_unique_nnz = L.matrix.nnz + U.matrix.nnz - matrix_perm.matrix.nrows;
        const int fill_in = lu_unique_nnz - matrix_perm.matrix.nnz;
        const double fill_ratio =
            matrix_perm.matrix.nnz > 0
                ? static_cast<double>(lu_unique_nnz) /
                      static_cast<double>(matrix_perm.matrix.nnz)
                : 0.0;
        const double extra_ratio =
            matrix_perm.matrix.nnz > 0
                ? static_cast<double>(fill_in) /
                      static_cast<double>(matrix_perm.matrix.nnz)
                : 0.0;

        if (options.csv) {
            print_csv_header();
            std::cout << std::setprecision(12)
                      << data.case_name << ','
                      << data.ybus.rows << ','
                      << data.pv.size() << ','
                      << data.pq.size() << ','
                      << system.matrix.rows << ','
                      << system.matrix.nnz() << ','
                      << matrix_perm.matrix.nnz << ','
                      << L.matrix.nnz << ','
                      << U.matrix.nnz << ','
                      << lu_unique_nnz << ','
                      << fill_in << ','
                      << fill_ratio << ','
                      << extra_ratio << ','
                      << l_supernodes.set.num_supernodes << ','
                      << l_supernodes.set.max_width << ','
                      << ordering_ms << ','
                      << symbolic_ms << ','
                      << supernode_ms << '\n';
        } else {
            std::cout << std::setprecision(12)
                      << "case: " << data.case_name << "\n"
                      << "n_bus: " << data.ybus.rows << "\n"
                      << "linear_dim: " << system.matrix.rows << "\n"
                      << "linear_nnz: " << system.matrix.nnz() << "\n"
                      << "reordered_nnz: " << matrix_perm.matrix.nnz << "\n"
                      << "L_nnz: " << L.matrix.nnz << "\n"
                      << "U_nnz: " << U.matrix.nnz << "\n"
                      << "LU_unique_nnz: " << lu_unique_nnz << "\n"
                      << "fill_in_nnz: " << fill_in << "\n"
                      << "fill_ratio: " << fill_ratio << "\n"
                      << "extra_fill_ratio: " << extra_ratio << "\n"
                      << "L_strict_tail_supernodes: "
                      << l_supernodes.set.num_supernodes << "\n"
                      << "L_strict_tail_max_width: "
                      << l_supernodes.set.max_width << "\n"
                      << "ordering_ms: " << ordering_ms << "\n"
                      << "symbolic_ms: " << symbolic_ms << "\n"
                      << "supernode_ms: " << supernode_ms << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "jacobian_symbolic_fill_count failed: " << ex.what() << '\n';
        return 1;
    }
}
