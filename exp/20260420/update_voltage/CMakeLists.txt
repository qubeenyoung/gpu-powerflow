cmake_minimum_required(VERSION 3.22)

project(exp20260420_update_voltage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(exp20260420_update_voltage STATIC
  src/update_voltage.cu
)

target_include_directories(exp20260420_update_voltage PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(exp20260420_update_voltage PUBLIC
  CUDA::cudart
)

set_target_properties(exp20260420_update_voltage PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
