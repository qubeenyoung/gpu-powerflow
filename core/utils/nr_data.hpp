#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

// 소영수정: CUDA 사용 시에만 CUDA 헤더 포함
#ifdef USE_CUDA
#include <cuComplex.h>
#endif

#ifndef DATASET_ROOT
#define DATASET_ROOT "/workspace/datasets/nr_dataset"
#endif

#ifndef SAVE_ROOT
#define SAVE_ROOT "/workspace/exp"
#endif

namespace nr_data
{

using YbusType = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor, int32_t>;
using JacobianType = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
using VectorXcd = Eigen::VectorX<std::complex<double>>;
using VectorXi32 = Eigen::VectorX<int32_t>;
using VectorXd = Eigen::VectorX<double>;

// 소영수정: CUDA 사용 시에만 NRDeviceData 선언
#ifdef USE_CUDA
struct NRDeviceData;
#endif

// CPU data (double) **
struct NRData {
    YbusType Ybus;
    VectorXcd Sbus;
    VectorXcd V0;
    VectorXi32 pv;
    VectorXi32 pq;

    void load_data(const std::string& case_name);

#ifdef USE_CUDA
    NRDeviceData to_device();  // 소영: CUDA 사용 시에만 gpu로 data 보냄
#endif
};

// GPU data (float) **
#ifdef USE_CUDA
struct NRDeviceData {
    // Ybus - csr
    cuFloatComplex* Ybus_val;
    int32_t* Ybus_col_ind;
    int32_t* Ybus_row_ptr;

    cuFloatComplex* Sbus;
    cuFloatComplex* V0;
    int32_t* pv;
    int32_t* pq;

    NRDeviceData();

    ~NRDeviceData();
};
#endif  // 소영수정: USE_CUDA

struct NRResult {
    VectorXcd V; // 최종 계산된 전압 (double)
    double normF = 0.0; // 최종 오차
    int iter = -1; // 반복 횟수
    bool converged = false; // 성공 여부

    void save_result(const std::string& case_name);
};

}
