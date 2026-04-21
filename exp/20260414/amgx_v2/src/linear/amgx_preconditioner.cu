#include "amgx_preconditioner.hpp"

#include <amgx_c.h>

#include <stdexcept>
#include <string>

namespace exp_20260414::amgx_v2 {
namespace {

void check_amgx(AMGX_RC code, const char* call)
{
    if (code != AMGX_RC_OK) {
        char message[4096] = {};
        AMGX_get_error_string(code, message, sizeof(message));
        throw std::runtime_error(std::string("AMGX call failed: ") + call + ": " + message);
    }
}

const char* amgx_config(int32_t block_dim, AmgxBlockSmoother block_smoother)
{
    if (block_dim > 1) {
        if (block_smoother == AmgxBlockSmoother::BlockJacobi) {
            return R"json(
{
  "config_version": 2,
  "solver": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": "BLOCK_JACOBI",
    "presweeps": 0,
    "postsweeps": 3,
    "cycle": "V",
    "coarse_solver": "NOSOLVER",
    "max_iters": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "relaxation_factor": 0.75,
    "tolerance": 0.0,
    "monitor_residual": 0,
    "print_grid_stats": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0
  }
}
)json";
        }

        return R"json(
{
  "config_version": 2,
  "solver": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": "MULTICOLOR_DILU",
    "matrix_coloring_scheme": "PARALLEL_GREEDY",
    "max_uncolored_percentage": 0.05,
    "presweeps": 0,
    "postsweeps": 3,
    "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "relaxation_factor": 0.75,
    "tolerance": 0.0,
    "monitor_residual": 0,
    "print_grid_stats": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0
  }
}
)json";
    }

    return R"json(
{
  "config_version": 2,
  "solver": {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": "BLOCK_JACOBI",
    "presweeps": 0,
    "postsweeps": 3,
    "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "relaxation_factor": 0.75,
    "tolerance": 0.0,
    "monitor_residual": 0,
    "print_grid_stats": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0
  }
}
)json";
}

class AmgxRuntime {
public:
    AmgxRuntime()
    {
        check_amgx(AMGX_initialize(), "AMGX_initialize");
        AMGX_register_print_callback([](const char*, int) {});
    }

    ~AmgxRuntime()
    {
        AMGX_finalize();
    }
};

void initialize_amgx_runtime()
{
    static AmgxRuntime runtime;
    (void)runtime;
}

void destroy_if_present(AMGX_solver_handle& handle)
{
    if (handle != nullptr) {
        AMGX_solver_destroy(handle);
        handle = nullptr;
    }
}

void destroy_if_present(AMGX_vector_handle& handle)
{
    if (handle != nullptr) {
        AMGX_vector_destroy(handle);
        handle = nullptr;
    }
}

void destroy_if_present(AMGX_matrix_handle& handle)
{
    if (handle != nullptr) {
        AMGX_matrix_destroy(handle);
        handle = nullptr;
    }
}

void destroy_if_present(AMGX_resources_handle& handle)
{
    if (handle != nullptr) {
        AMGX_resources_destroy(handle);
        handle = nullptr;
    }
}

void destroy_if_present(AMGX_config_handle& handle)
{
    if (handle != nullptr) {
        AMGX_config_destroy(handle);
        handle = nullptr;
    }
}

}  // namespace

struct AmgxPreconditioner::Impl {
    int32_t n = 0;
    int32_t nnz = 0;
    int32_t block_dim = 1;
    AMGX_config_handle config = nullptr;
    AMGX_resources_handle resources = nullptr;
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle rhs = nullptr;
    AMGX_vector_handle x = nullptr;
    AMGX_solver_handle solver = nullptr;

    void destroy()
    {
        destroy_if_present(solver);
        destroy_if_present(x);
        destroy_if_present(rhs);
        destroy_if_present(matrix);
        destroy_if_present(resources);
        destroy_if_present(config);
        n = 0;
        nnz = 0;
        block_dim = 1;
    }
};

AmgxPreconditioner::AmgxPreconditioner()
    : impl_(new Impl{})
{}

AmgxPreconditioner::~AmgxPreconditioner()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
    delete impl_;
}

void AmgxPreconditioner::setup(const CsrMatrixView& matrix)
{
    if (matrix.rows <= 0 || matrix.nnz <= 0 || matrix.row_ptr == nullptr ||
        matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error("AmgxPreconditioner::setup received an invalid matrix");
    }
    setup_impl(matrix.rows,
               matrix.nnz,
               1,
               AmgxBlockSmoother::MulticolorDilu,
               matrix.row_ptr,
               matrix.col_idx,
               matrix.values);
}

void AmgxPreconditioner::setup(const BlockCsrMatrixView& matrix)
{
    setup(matrix, AmgxBlockSmoother::MulticolorDilu);
}

void AmgxPreconditioner::setup(const BlockCsrMatrixView& matrix, AmgxBlockSmoother smoother)
{
    if (matrix.rows <= 0 || matrix.nnz <= 0 || matrix.block_dim <= 1 ||
        matrix.row_ptr == nullptr || matrix.col_idx == nullptr || matrix.values == nullptr) {
        throw std::runtime_error("AmgxPreconditioner::setup received an invalid block matrix");
    }
    setup_impl(matrix.rows,
               matrix.nnz,
               matrix.block_dim,
               smoother,
               matrix.row_ptr,
               matrix.col_idx,
               matrix.values);
}

void AmgxPreconditioner::setup_impl(int32_t rows,
                                    int32_t nnz,
                                    int32_t block_dim,
                                    AmgxBlockSmoother smoother,
                                    const int32_t* row_ptr,
                                    const int32_t* col_idx,
                                    const double* values)
{
    impl_->destroy();
    initialize_amgx_runtime();

    impl_->n = rows;
    impl_->nnz = nnz;
    impl_->block_dim = block_dim;
    check_amgx(AMGX_config_create(&impl_->config, amgx_config(block_dim, smoother)),
               "AMGX_config_create");
    check_amgx(AMGX_resources_create_simple(&impl_->resources, impl_->config),
               "AMGX_resources_create_simple");
    check_amgx(AMGX_matrix_create(&impl_->matrix, impl_->resources, AMGX_mode_dDDI),
               "AMGX_matrix_create");
    check_amgx(AMGX_vector_create(&impl_->rhs, impl_->resources, AMGX_mode_dDDI),
               "AMGX_vector_create(rhs)");
    check_amgx(AMGX_vector_create(&impl_->x, impl_->resources, AMGX_mode_dDDI),
               "AMGX_vector_create(x)");
    check_amgx(AMGX_solver_create(&impl_->solver, impl_->resources, AMGX_mode_dDDI, impl_->config),
               "AMGX_solver_create");

    // AMGX_mode_dDDI tells AMGX that matrix and vector values live on device.
    check_amgx(AMGX_matrix_upload_all(impl_->matrix,
                                      impl_->n,
                                      impl_->nnz,
                                      impl_->block_dim,
                                      impl_->block_dim,
                                      row_ptr,
                                      col_idx,
                                      values,
                                      nullptr),
               "AMGX_matrix_upload_all");
    check_amgx(AMGX_vector_set_zero(impl_->rhs, impl_->n, impl_->block_dim),
               "AMGX_vector_set_zero(rhs)");
    check_amgx(AMGX_vector_set_zero(impl_->x, impl_->n, impl_->block_dim),
               "AMGX_vector_set_zero(x)");
    check_amgx(AMGX_solver_setup(impl_->solver, impl_->matrix), "AMGX_solver_setup");
}

void AmgxPreconditioner::apply(const double* rhs_device, double* x_device, int32_t n) const
{
    const int32_t scalar_n = ready() ? impl_->n * impl_->block_dim : 0;
    if (!ready() || n != scalar_n || rhs_device == nullptr || x_device == nullptr) {
        throw std::runtime_error("AmgxPreconditioner::apply called before setup");
    }
    // Despite the API names, these use device pointers in AMGX_mode_dDDI.
    check_amgx(AMGX_vector_upload(impl_->rhs, impl_->n, impl_->block_dim, rhs_device),
               "AMGX_vector_upload(rhs)");
    check_amgx(AMGX_vector_set_zero(impl_->x, impl_->n, impl_->block_dim),
               "AMGX_vector_set_zero(x)");
    check_amgx(AMGX_solver_solve_with_0_initial_guess(impl_->solver, impl_->rhs, impl_->x),
               "AMGX_solver_solve_with_0_initial_guess");
    check_amgx(AMGX_vector_download(impl_->x, x_device), "AMGX_vector_download(x)");
}

bool AmgxPreconditioner::ready() const
{
    return impl_ != nullptr && impl_->solver != nullptr;
}

}  // namespace exp_20260414::amgx_v2
