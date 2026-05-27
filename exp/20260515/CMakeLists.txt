cmake_minimum_required(VERSION 3.22)

project(exp_20260515_general_linear_system_io LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_library(LIB_METIS NAMES metis libmetis.so.5
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT LIB_METIS)
  message(FATAL_ERROR "METIS library is required.")
endif()

add_library(general_linear_system_io STATIC
  include/io/dense_io.cpp
  include/io/matrix_market_io.cpp
)

target_compile_features(general_linear_system_io PUBLIC cxx_std_17)

target_include_directories(general_linear_system_io
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(check_dense_io
  tools/check_dense_io.cpp
)

target_link_libraries(check_dense_io
  PRIVATE
    general_linear_system_io
)

set_target_properties(check_dense_io PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(check_sparse_io
  tools/check_sparse_io.cpp
)

target_link_libraries(check_sparse_io
  PRIVATE
    general_linear_system_io
)

set_target_properties(check_sparse_io PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(cpu_dense_v0
  src/cpu_dense_v0.cpp
)

target_link_libraries(cpu_dense_v0
  PRIVATE
    general_linear_system_io
)

set_target_properties(cpu_dense_v0 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(cpu_sparse_v0
  src/cpu_sparse_v0.cpp
)

target_link_libraries(cpu_sparse_v0
  PRIVATE
    general_linear_system_io
    ${LIB_METIS}
)

set_target_properties(cpu_sparse_v0 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
