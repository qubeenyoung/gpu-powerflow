cmake_minimum_required(VERSION 3.22)

project(exp20260420_mismatch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(exp20260420_mismatch STATIC
  src/mismatch.cu
)

target_include_directories(exp20260420_mismatch PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../jac_asm/src/common
)

target_link_libraries(exp20260420_mismatch PUBLIC
  CUDA::cudart
  CUDA::cusparse
)

set_target_properties(exp20260420_mismatch PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
