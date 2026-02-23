/**
 * @file binding_helper.cpp
 * @brief Python(NumPy/SciPy) ↔ C++(Eigen) 변환 헬퍼 정의
 */
#include "binding_helper.hpp"

#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <cstring>   // std::memcpy

namespace pf_binding {

Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>
to_eigen_csc(py::handle obj)
{
    // 1) scipy.sparse.csc_matrix 처리
    try {
        py::module sp = py::module::import("scipy.sparse");
        py::object csc_type = sp.attr("csc_matrix");
        if (py::isinstance(obj, csc_type)) {
            // 안전하게 CSC 보장
            py::object csc = obj.attr("tocsc")();

            // shape
            py::tuple shape = csc.attr("shape").cast<py::tuple>();
            const int32_t nrows = shape[0].cast<int32_t>();
            const int32_t ncols = shape[1].cast<int32_t>();

            // 배열 뽑기 (강제 캐스팅: 복소/정수 dtype 자동 맞춤)
            auto data    = csc.attr("data").cast<py::array_t<cd, py::array::c_style | py::array::forcecast>>();
            auto indices = csc.attr("indices").cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            auto indptr  = csc.attr("indptr").cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();

            const int64_t nnz = data.size();

            Eigen::Map<const Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>> A(
                nrows, ncols, static_cast<int32_t>(nnz),
                indptr.data(), indices.data(), reinterpret_cast<const cd*>(data.data())
            );
            return Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>(A); // 복사 생성으로 소유권 확보
        }
    } catch (...) {
        // scipy 모듈 부재/예외 시 아래 튜플 경로로 시도
    }

    // 2) (data, indices, indptr, shape) 튜플
    if (py::isinstance<py::tuple>(obj)) {
        py::tuple t = obj.cast<py::tuple>();
        if (t.size() != 4) {
            throw std::runtime_error("Expected tuple (data, indices, indptr, shape).");
        }

        auto data    = t[0].cast<py::array_t<cd, py::array::c_style | py::array::forcecast>>();
        auto indices = t[1].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        auto indptr  = t[2].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        py::tuple shape = t[3].cast<py::tuple>();

        const int32_t nrows = shape[0].cast<int32_t>();
        const int32_t ncols = shape[1].cast<int32_t>();
        const int64_t nnz   = data.size();

        Eigen::Map<const Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>> A(
            nrows, ncols, static_cast<int32_t>(nnz),
            indptr.data(), indices.data(), reinterpret_cast<const cd*>(data.data())
        );
        return Eigen::SparseMatrix<cd, Eigen::ColMajor, int32_t>(A);
    }

    throw std::runtime_error(
        "ybus must be scipy.sparse.csc_matrix or a tuple (data, indices, indptr, shape).");
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
to_eigen_vec(py::handle arr_like, const char* name)
{
    auto arr = py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(arr_like);
    if (arr.ndim() != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1-D numpy array.");
    }
    Eigen::Matrix<T, Eigen::Dynamic, 1> v(arr.size());
    std::memcpy(v.data(), arr.data(), sizeof(T) * static_cast<size_t>(arr.size()));
    return v;
}

// 템플릿 명시적 인스턴스화(필요 타입만)
template Eigen::Matrix<cd, Eigen::Dynamic, 1> to_eigen_vec(py::handle, const char*);
template Eigen::Matrix<double, Eigen::Dynamic, 1> to_eigen_vec(py::handle, const char*);

Eigen::Matrix<int32_t, Eigen::Dynamic, 1>
to_eigen_vec_i32(py::handle arr_like, const char* name)
{
    auto arr = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(arr_like);
    if (arr.ndim() != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1-D numpy array of int32.");
    }
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> v(arr.size());
    std::memcpy(v.data(), arr.data(), sizeof(int32_t) * static_cast<size_t>(arr.size()));
    return v;
}

} // namespace pf_binding
