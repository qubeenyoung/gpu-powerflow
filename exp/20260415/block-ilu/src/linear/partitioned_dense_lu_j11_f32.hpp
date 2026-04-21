#pragma once

#include "assembly/reduced_jacobian_assembler.hpp"
#include "linear/permuted_ilu0_block.hpp"
#include "model/reduced_jacobian.hpp"
#include "utils/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdint>
#include <string>
#include <vector>

namespace exp_20260415::block_ilu {

class PartitionedDenseLuJ11BlockF32 {
public:
    explicit PartitionedDenseLuJ11BlockF32(std::string name);
    ~PartitionedDenseLuJ11BlockF32();

    PartitionedDenseLuJ11BlockF32(const PartitionedDenseLuJ11BlockF32&) = delete;
    PartitionedDenseLuJ11BlockF32& operator=(const PartitionedDenseLuJ11BlockF32&) = delete;

    void analyze(DeviceCsrMatrixViewF32 matrix,
                 const HostCsrPattern& host_pattern,
                 J11ReorderMode reorder_mode,
                 int32_t max_block_size,
                 J11DenseBackend dense_backend,
                 J11PartitionMode partition_mode);
    void factorize();
    void solve(const float* rhs_device, float* out_device);
    void set_stream(cudaStream_t stream);
    void clear_stream_noexcept();

    int32_t last_zero_pivot() const { return last_zero_pivot_; }

private:
    void build_permutation(const HostCsrPattern& host_pattern, J11ReorderMode reorder_mode);
    void build_graph_partitions(const HostCsrPattern& host_pattern,
                                J11PartitionMode partition_mode);
    void build_dense_maps(const HostCsrPattern& host_pattern);
    void build_pointer_arrays();
    void check_factor_info();
    void factorize_cublas_getrf();
    void factorize_cusolver_getrf();
    void factorize_tc_no_pivot();
    void solve_cublas_getrf(const float* rhs_device, float* out_device);
    void solve_tc_no_pivot(const float* rhs_device, float* out_device);

    std::string name_;
    int32_t rows_ = 0;
    int32_t nnz_ = 0;
    int32_t max_block_size_ = 0;
    int32_t block_count_ = 0;
    int32_t in_block_nnz_ = 0;
    int32_t last_zero_pivot_ = -1;
    J11DenseBackend dense_backend_ = J11DenseBackend::CublasGetrf;
    bool analyzed_ = false;
    bool factorized_ = false;

    DeviceCsrMatrixViewF32 original_matrix_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;

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

    DeviceBuffer<float> d_dense_values_;
    DeviceBuffer<float> d_dense_rhs_;
    DeviceBuffer<float*> d_matrix_ptrs_;
    DeviceBuffer<float*> d_rhs_ptrs_;
    DeviceBuffer<int32_t> d_pivots_;
    DeviceBuffer<int32_t> d_info_;
    DeviceBuffer<float> d_cusolver_workspace_;
    std::vector<int32_t> h_info_;
};

}  // namespace exp_20260415::block_ilu
