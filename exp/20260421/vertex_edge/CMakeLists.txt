cmake_minimum_required(VERSION 3.22)

project(exp_20260421_vertex_edge LANGUAGES CXX CUDA)

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

add_executable(vertex_edge_bench
  benchmark_vertex_edge.cu
  src/common/jacobian_build.cpp
  src/edge/edge_fill_jac.cu
  src/vertex/vertex_fill_jac.cu
  src/update/update_voltage.cu
)

target_include_directories(vertex_edge_bench
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/edge
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vertex
    ${CMAKE_CURRENT_SOURCE_DIR}/src/update
)

target_link_libraries(vertex_edge_bench
  PRIVATE
    CUDA::cudart
)

set_target_properties(vertex_edge_bench PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
)
