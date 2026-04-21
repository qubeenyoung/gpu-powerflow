#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "linear/permuted_ilu0_block.hpp"
#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

class PartitionedDenseLuJ11Block {
public:
    explicit PartitionedDenseLuJ11Block(std::string name);
    ~PartitionedDenseLuJ11Block();

    PartitionedDenseLuJ11Block(const PartitionedDenseLuJ11Block&) = delete;
    PartitionedDenseLuJ11Block& operator=(const PartitionedDenseLuJ11Block&) = delete;

    void analyze(DeviceCsrMatrixView matrix,
                 const HostCsrPattern& host_pattern,
                 J11ReorderMode reorder_mode,
                 int32_t max_block_size,
                 J11PartitionMode partition_mode);
    void factorize();
    void solve(const double* rhs_device, double* out_device);
    void set_stream(cudaStream_t stream);
    void clear_stream_noexcept();

    int32_t rows() const { return rows_; }
    int32_t nnz() const { return nnz_; }
    int32_t block_size() const { return max_block_size_; }
    int32_t block_count() const { return block_count_; }
    int32_t in_block_nnz() const { return in_block_nnz_; }
    int32_t last_zero_pivot() const { return last_zero_pivot_; }

private:
    void build_permutation(const HostCsrPattern& host_pattern, J11ReorderMode reorder_mode);
    void build_graph_partitions(const HostCsrPattern& host_pattern,
                                J11PartitionMode partition_mode);
    void build_dense_maps(const HostCsrPattern& host_pattern);
    void build_pointer_arrays();
    void check_info_array(const char* stage);

    std::string name_;
    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    int32_t max_block_size_ = 0;
    int32_t block_count_ = 0;
    int32_t in_block_nnz_ = 0;
    int32_t last_zero_pivot_ = -1;
    bool analyzed_ = false;
    bool factorized_ = false;

    DeviceCsrMatrixView original_matrix_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    std::vector<int32_t> new_to_old_;
    std::vector<int32_t> old_to_new_;
    std::vector<int32_t> block_of_new_;
    std::vector<int32_t> local_of_new_;
    std::vector<int32_t> block_sizes_;
    std::vector<int32_t> slot_new_index_;
    std::vector<int32_t> dense_source_pos_;
    std::vector<int32_t> dense_dest_pos_;

    DeviceBuffer<int32_t> d_new_to_old_;
    DeviceBuffer<int32_t> d_block_sizes_;
    DeviceBuffer<int32_t> d_slot_new_index_;
    DeviceBuffer<int32_t> d_dense_source_pos_;
    DeviceBuffer<int32_t> d_dense_dest_pos_;

    DeviceBuffer<double> d_dense_values_;
    DeviceBuffer<double> d_dense_rhs_;
    DeviceBuffer<double*> d_matrix_ptrs_;
    DeviceBuffer<double*> d_rhs_ptrs_;
    DeviceBuffer<int32_t> d_pivots_;
    DeviceBuffer<int32_t> d_info_;
    std::vector<int32_t> h_info_;
};

}  // namespace exp_20260415::block_ilu
