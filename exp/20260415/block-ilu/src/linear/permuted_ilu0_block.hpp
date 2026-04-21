#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "linear/cusparse_ilu0_block.hpp"
#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

enum class J11ReorderMode {
    None,
    Amd,
    Colamd,
    Rcm,
};

enum class J11SolverKind {
    Ilu0,
    PartitionDenseLu,
    ExactKlu,
};

enum class J11DenseBackend {
    CublasGetrf,
    TcNoPivot,
    CusolverGetrf,
};

enum class J11PartitionMode {
    Bfs,
    Metis,
};

J11ReorderMode parse_j11_reorder_mode(const std::string& name);
const char* j11_reorder_mode_name(J11ReorderMode mode);
J11SolverKind parse_j11_solver_kind(const std::string& name);
const char* j11_solver_kind_name(J11SolverKind kind);
J11DenseBackend parse_j11_dense_backend(const std::string& name);
const char* j11_dense_backend_name(J11DenseBackend backend);
J11PartitionMode parse_j11_partition_mode(const std::string& name);
const char* j11_partition_mode_name(J11PartitionMode mode);

class PermutedIlu0Block {
public:
    explicit PermutedIlu0Block(std::string name);
    ~PermutedIlu0Block() = default;

    PermutedIlu0Block(const PermutedIlu0Block&) = delete;
    PermutedIlu0Block& operator=(const PermutedIlu0Block&) = delete;

    void analyze(DeviceCsrMatrixView matrix,
                 const HostCsrPattern& host_pattern,
                 J11ReorderMode mode);
    void factorize();
    void solve(const double* rhs_device, double* out_device);
    void set_stream(cudaStream_t stream);
    void clear_stream_noexcept();

    int32_t rows() const { return rows_; }
    int32_t nnz() const { return nnz_; }
    int32_t last_zero_pivot() const { return ilu_.last_zero_pivot(); }
    J11ReorderMode mode() const { return mode_; }
    bool permuted() const { return mode_ != J11ReorderMode::None; }

private:
    void build_permuted_pattern(const HostCsrPattern& host_pattern);
    DeviceCsrMatrixView active_matrix_view() const;

    std::string name_;
    J11ReorderMode mode_ = J11ReorderMode::None;
    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    bool analyzed_ = false;
    cudaStream_t stream_ = nullptr;

    DeviceCsrMatrixView original_matrix_;
    CusparseIlu0Block ilu_;

    std::vector<int32_t> new_to_old_;
    std::vector<int32_t> old_to_new_;
    std::vector<int32_t> source_pos_for_permuted_nnz_;

    DeviceBuffer<int32_t> d_perm_row_ptr_;
    DeviceBuffer<int32_t> d_perm_col_idx_;
    DeviceBuffer<int32_t> d_new_to_old_;
    DeviceBuffer<int32_t> d_old_to_new_;
    DeviceBuffer<int32_t> d_source_pos_for_permuted_nnz_;
    DeviceBuffer<double> d_perm_values_;
    DeviceBuffer<double> d_perm_rhs_;
    DeviceBuffer<double> d_perm_solution_;
};

}  // namespace exp_20260415::block_ilu
