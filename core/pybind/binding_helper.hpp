#pragma once
/**
 * @file binding_helper.hpp
 * @brief Python(NumPy/SciPy) ↔ C++(Eigen) 변환 헬퍼 선언
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>

namespace pf_binding {

namespace py = pybind11;
using cd = std::complex<double>;

/**
 * @brief scipy.sparse.csc_matrix 또는 (data, indices, indptr, shape) 튜플을
 *        Eigen::SparseMatrix<std::complex<double>, ColMajor, int32_t> 로 변환
 * @param obj Python 객체 (csc_matrix 또는 (data, indices, indptr, shape))
 * @return Eigen CSC 희소 행렬
 * @throws std::runtime_error 타입/차원 불일치 시
 */
Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>
to_eigen_csc(py::handle obj);

/**
 * @brief 1D numpy.ndarray → Eigen 벡터 (깊은 복사)
 * @tparam T 요소 타입 (double, std::complex<double> 등)
 * @param arr_like 파이썬 배열(강제 캐스팅됨)
 * @param name 오류 메시지용 이름
 * @return Eigen::Matrix<T, -1, 1>
 * @throws std::runtime_error 1D가 아니면
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
to_eigen_vec(py::handle arr_like, const char* name);

/**
 * @brief 1D numpy.ndarray(int32) → Eigen int32 벡터 (깊은 복사)
 * @param arr_like 파이썬 배열(강제 캐스팅됨)
 * @param name 오류 메시지용 이름
 * @return Eigen::Matrix<int32_t, -1, 1>
 * @throws std::runtime_error 1D가 아니면
 */
Eigen::Matrix<int32_t, Eigen::Dynamic, 1>
to_eigen_vec_i32(py::handle arr_like, const char* name);

} // namespace pf_binding
