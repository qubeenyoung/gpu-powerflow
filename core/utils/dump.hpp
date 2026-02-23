#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include "nr_data.hpp" 
#include "io.hpp"
#include "spdlog/spdlog.h" 

#define DUMP_DATA

inline std::string& dump_path() {
    static std::string path = "./dump";
    return path;
}

inline void init_dump_path(const std::string& base_path) {
    dump_path() = base_path;
    std::filesystem::create_directories(base_path);
    spdlog::info("[DUMP] Base path initialized: {}", base_path);
}

/**
 * @brief 벡터(Eigen Dense)를 .npy로 덤프합니다. (DUMP_DATA가 정의된 경우에만)
 * @tparam EigenVec  Eigen 벡터 타입 (예: Eigen::VectorXd, Eigen::VectorXcd 등)
 * @param base_path  저장 디렉토리 (예: "results/case9/dumps")
 * @param var_name   변수명(파일명 접두, 예: "V")
 * @param iteration  반복 횟수(파일명 접미)
 * @param data       저장할 Eigen 벡터
 *
 * @note 저장 파일명: {base_path}/{var_name}_iter{iteration}.npy
 */
template <typename EigenVec>
inline void dump_vector(
    const std::string& var_name,
    int iteration,
    const EigenVec& data)
{
#if defined(DUMP_DATA)
  try {
    using T = typename EigenVec::Scalar;
    const std::string filename = dump_path() + "/" + var_name + "_iter" + std::to_string(iteration) + ".npy";

    std::vector<T> std_vec(data.data(), data.data() + static_cast<std::size_t>(data.size()));
    pf_io::save_vector(filename, std_vec, var_name.c_str());  // 소영수정: std::string -> const char*

  } catch (const std::exception& e) {
    spdlog::warn("[DUMP] Failed to dump vector {}: {}", var_name, e.what());
  }
#endif
}

/**
 * @brief 행렬(CSR/CSC 무관)을 CSC로 변환하여 .npz로 덤프합니다. (DUMP_DATA가 정의된 경우에만)
 * @tparam Scalar        double 또는 std::complex<double>
 * @tparam StorageOrder  Eigen::RowMajor 또는 Eigen::ColMajor
 * @param base_path      저장 디렉토리
 * @param var_name       변수명(파일명 접두)
 * @param iteration      반복 횟수(파일명 접미)
 * @param data           저장할 Eigen 희소행렬
 *
 * @note 저장 파일명: {base_path}/{var_name}_iter{iteration}.npz
 * @note 변환: (Row/Col) → CSC(ColMajor), 압축 보장 후 저장
 */
template <typename Scalar, int StorageOrder>
inline void dump_matrix(
    const std::string& var_name,
    int iteration,
    const Eigen::SparseMatrix<Scalar, StorageOrder, int32_t>& data)
{
#if defined(DUMP_DATA)
  try {
    const std::string filename = dump_path() + "/" + var_name + "_iter" + std::to_string(iteration) + ".npz";

    // 스칼라에 맞춘 CSC로 변환 (double/complex 모두 안전)
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int32_t> csc = data;
    if (!csc.isCompressed()) csc.makeCompressed();

    // 기존 저장기(템플릿) 호출: save_as_csc는 Scalar 템플릿이어야 함
    pf_io::save_as_csc(filename, csc);

  } catch (const std::exception& e) {
    spdlog::warn("[DUMP] Failed to dump matrix {}: {}", var_name, e.what());
  }
#endif
}