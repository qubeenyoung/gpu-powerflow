cmake_minimum_required(VERSION 3.22)

project(exp_20260520_glu LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

find_library(METIS_LIBRARY NAMES metis libmetis.so.5
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT METIS_LIBRARY)
  message(FATAL_ERROR "METIS library is required")
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(GLU_ROOT "${REPO_ROOT}/third_party/glu" CACHE PATH "Path to the upstream GLU source tree")
set(GLU_SRC "${GLU_ROOT}/src")
set(GLU_INCLUDE_DIR "${GLU_ROOT}/include")

if(NOT EXISTS "${GLU_SRC}/numeric.cu")
  message(FATAL_ERROR "GLU source tree not found at ${GLU_SRC}")
endif()
if(NOT EXISTS "${GLU_INCLUDE_DIR}/symbolic.h")
  message(FATAL_ERROR "GLU include tree not found at ${GLU_INCLUDE_DIR}")
endif()

file(GLOB GLU_NICSLU_SOURCES CONFIGURE_DEPENDS
  "${GLU_SRC}/nicslu/source/*.c"
)

add_library(exp20260520_nicslu STATIC
  ${GLU_NICSLU_SOURCES}
  "${GLU_SRC}/nicslu/util/nicslu_readfile.c"
  "${GLU_SRC}/preprocess/preprocess.c"
)

target_include_directories(exp20260520_nicslu
  PUBLIC
    "${GLU_SRC}/nicslu/include"
    "${GLU_SRC}/nicslu/util"
    "${GLU_SRC}/preprocess"
)

target_compile_definitions(exp20260520_nicslu
  PRIVATE
    NO_ATOMIC
    SSE2
)

target_compile_options(exp20260520_nicslu
  PRIVATE
    -msse2
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-sign-compare
    -Wno-implicit-function-declaration
    -Wno-builtin-declaration-mismatch
    -Wno-unused-result
)

add_library(exp20260520_core STATIC
  src/io/matrix_market_io.c
  src/matrix/csc_matrix.c
  src/matrix/csr_matrix.c
  src/pf_case_loader.cpp
  src/glu_lu.cpp
  src/reordering/metis_nd.c
  src/symbolic/symbolic_factorize.c
  src/symbolic/symbolic_supernode.c
  "${GLU_SRC}/symbolic.cc"
  "${GLU_SRC}/numeric.cu"
  "${GLU_SRC}/Timer.cpp"
)

target_include_directories(exp20260520_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reordering"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/symbolic"
    "${GLU_INCLUDE_DIR}"
    "${GLU_SRC}/nicslu/include"
    "${GLU_SRC}/nicslu/util"
    "${GLU_SRC}/preprocess"
)

target_compile_definitions(exp20260520_core
  PUBLIC
    GLU_DEBUG=0
)

target_link_libraries(exp20260520_core
  PUBLIC
    exp20260520_nicslu
    "${METIS_LIBRARY}"
    CUDA::cudart
    m
    pthread
    rt
)

add_executable(glu_pf_benchmark
  benchmark/glu_pf_benchmark.cpp
)

target_link_libraries(glu_pf_benchmark
  PRIVATE
    exp20260520_core
)

add_executable(symbolic_fill_count
  benchmark/symbolic_fill_count.c
)

target_link_libraries(symbolic_fill_count
  PRIVATE
    exp20260520_core
)

add_executable(jacobian_symbolic_fill_count
  benchmark/jacobian_symbolic_fill_count.cpp
)

target_link_libraries(jacobian_symbolic_fill_count
  PRIVATE
    exp20260520_core
)
