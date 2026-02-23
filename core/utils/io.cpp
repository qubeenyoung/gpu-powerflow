#include "io.hpp"

#include "cnpy.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <stdexcept>   

/**
 * @brief .npy 파일에서 complex128 벡터를 로드하여 std::vector로 반환합니다.
 */
std::vector<std::complex<double>> 
pf_io::load_complex_vector(const std::string& filename) 
{
    cnpy::NpyArray arr;
    try {
        arr = cnpy::npy_load(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading file " + filename + ": " + e.what());
    }

    std::complex<double>* data_ptr = arr.data<std::complex<double>>();
    size_t len = arr.shape[0]; 

    return std::vector<std::complex<double>>(data_ptr, data_ptr + len);
}

/**
 * @brief .npy 파일에서 int32 벡터를 로드하여 std::vector<int32_t>로 반환합니다.
 */
std::vector<int32_t> 
pf_io::load_int32_vector(const std::string& filename) {
    cnpy::NpyArray arr;
    try {
        arr = cnpy::npy_load(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading file " + filename + ": " + e.what());
    }
    
    if (arr.shape.empty()) {
        return std::vector<int32_t>();
    }

    int32_t* data_ptr = arr.data<int32_t>();
    size_t len = arr.shape[0];

    return std::vector<int32_t>(data_ptr, data_ptr + len);
}
