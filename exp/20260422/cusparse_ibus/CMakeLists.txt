cmake_minimum_required(VERSION 3.22)

project(exp_20260422_cusparse_ibus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)

add_executable(cusparse_ibus_bench
  src/cusparse_ibus_bench.cu
  ../../../cuPF/tests/dump_case_loader.cpp
)

target_include_directories(cusparse_ibus_bench
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuPF/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuPF/cpp/inc
)

target_link_libraries(cusparse_ibus_bench
  PRIVATE
    CUDA::cudart
    CUDA::cusparse
)

set_target_properties(cusparse_ibus_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
