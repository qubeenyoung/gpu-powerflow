cmake_minimum_required(VERSION 3.22)

option(WITH_CUDA "Build the CUDA backend" OFF)
option(BUILD_PYTHON_BINDINGS "Build the pybind11 module" OFF)
option(BUILD_NAIVE_CPU "Build PyPower-equivalent naive CPU backend (CpuAlgorithm::PyPowerLike)" OFF)
option(BUILD_BENCHMARKS "Build benchmark helper executables" OFF)

project(cuPF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 REQUIRED)
find_package(spdlog QUIET)

find_path(SUITESPARSE_INCLUDE_DIR
  NAMES klu.h btf.h amd.h colamd.h SuiteSparse_config.h suitesparse_config.h
  PATHS
    /usr/include
    /usr/include/suitesparse
    /usr/local/include
    /usr/local/include/suitesparse
)
find_library(LIB_KLU
  NAMES klu
  PATHS
    /lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)
find_library(LIB_BTF
  NAMES btf
  PATHS
    /lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)
find_library(LIB_AMD
  NAMES amd
  PATHS
    /lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)
find_library(LIB_COLAMD
  NAMES colamd
  PATHS
    /lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)
find_library(LIB_SSCONFIG
  NAMES suitesparseconfig SuiteSparse_config
  PATHS
    /lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)

if(NOT SUITESPARSE_INCLUDE_DIR OR NOT LIB_KLU OR NOT LIB_BTF OR NOT LIB_AMD OR NOT LIB_COLAMD OR NOT LIB_SSCONFIG)
  message(FATAL_ERROR
    "cuPF CPU backend requires SuiteSparse KLU (klu, btf, amd, colamd, suitesparseconfig).")
endif()

set(CUPF_CORE_SOURCES
  cpp/src/newton_solver/core/jacobian_builder.cpp
  cpp/src/newton_solver/core/newton_solver.cpp
  cpp/src/newton_solver/backend/cpp/compute_mismatch.cpp
  cpp/src/newton_solver/backend/cpp/initialize.cpp
  cpp/src/newton_solver/backend/cpp/klu_solver.cpp
  cpp/src/newton_solver/backend/cpp/update_jacobian.cpp
)

if(BUILD_NAIVE_CPU)
  list(APPEND CUPF_CORE_SOURCES
    cpp/src/newton_solver/backend/cpp_naive/naive_initialize.cpp
    cpp/src/newton_solver/backend/cpp_naive/naive_jacobian.cpp
    cpp/src/newton_solver/backend/cpp_naive/naive_solver.cpp
  )
endif()

if(WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  find_package(CUDAToolkit REQUIRED)
  find_path(CUDSS_INCLUDE_DIR
    NAMES cuDSS.h cudss.h
    HINTS
      /usr/include
      /usr/local/include
      /usr/local/cuda/include
      $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/include
  )
  find_library(CUDSS_LIBRARY
    NAMES cudss
    HINTS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/lib
  )

  if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
    message(FATAL_ERROR "WITH_CUDA=ON requires cuDSS headers and library. Set CUDSS_INCLUDE_DIR/CUDSS_LIBRARY if needed.")
  endif()

  list(APPEND CUPF_CORE_SOURCES
    cpp/src/newton_solver/backend/cuda/compute_mismatch.cu
    cpp/src/newton_solver/backend/cuda/cudss_solver.cpp
    cpp/src/newton_solver/backend/cuda/initialize.cpp
    cpp/src/newton_solver/backend/cuda/update_jacobian.cu
  )
endif()

add_library(cupf_core STATIC ${CUPF_CORE_SOURCES})

target_include_directories(cupf_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc
)

target_link_libraries(cupf_core
  PUBLIC
    Eigen3::Eigen
    ${LIB_KLU}
    ${LIB_BTF}
    ${LIB_AMD}
    ${LIB_COLAMD}
    ${LIB_SSCONFIG}
)
target_include_directories(cupf_core PUBLIC ${SUITESPARSE_INCLUDE_DIR})

if(spdlog_FOUND)
  target_link_libraries(cupf_core PUBLIC spdlog::spdlog)
endif()

if(BUILD_NAIVE_CPU)
  target_compile_definitions(cupf_core PUBLIC WITH_NAIVE_CPU)
  # The naive backend impl header lives next to its .cpp files.
  target_include_directories(cupf_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/newton_solver/backend/cpp_naive
  )
  # SuperLU: header at /usr/include/superlu, library libsuperlu.so
  find_path(SUPERLU_INCLUDE_DIR slu_ddefs.h
    PATHS /usr/include/superlu /usr/local/include/superlu)
  find_library(SUPERLU_LIBRARY superlu
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
  if(NOT SUPERLU_INCLUDE_DIR OR NOT SUPERLU_LIBRARY)
    message(FATAL_ERROR
      "BUILD_NAIVE_CPU=ON requires SuperLU. Install with: apt install libsuperlu-dev")
  endif()
  target_include_directories(cupf_core PRIVATE ${SUPERLU_INCLUDE_DIR})
  target_link_libraries(cupf_core PUBLIC ${SUPERLU_LIBRARY})
  message(STATUS "SuperLU found: ${SUPERLU_LIBRARY} (headers: ${SUPERLU_INCLUDE_DIR})")
endif()

if(WITH_CUDA)
  target_compile_definitions(cupf_core PUBLIC WITH_CUDA)
  target_include_directories(cupf_core PUBLIC ${CUDSS_INCLUDE_DIR})
  target_link_libraries(cupf_core
    PUBLIC
      CUDA::cudart
      CUDA::cusparse
      ${CUDSS_LIBRARY}
  )
endif()

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PYBIND11_CMAKE_DIR)
    list(PREPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
  endif()

  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(cupf bindings/pybind_module.cpp)
  target_link_libraries(cupf PRIVATE cupf_core pybind11::module)
  target_include_directories(cupf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc)
endif()

if(BUILD_TESTING)
  add_executable(cupf_case_smoke
    tests/cpp/dump_case_loader.cpp
    tests/cpp/test_solve_dump_case.cpp
  )
  target_link_libraries(cupf_case_smoke PRIVATE cupf_core)
  target_include_directories(cupf_case_smoke
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
  )

  add_test(
    NAME cupf_case30_cpu_smoke
    COMMAND cupf_case_smoke
      --case-dir ${CMAKE_CURRENT_SOURCE_DIR}/../v1/core/dumps/case30_ieee
      --backend cpu
  )

  if(BUILD_NAIVE_CPU)
    add_test(
      NAME cupf_case30_naive_smoke
      COMMAND cupf_case_smoke
        --case-dir ${CMAKE_CURRENT_SOURCE_DIR}/../v1/core/dumps/case30_ieee
        --backend cpu --algorithm pypower_like
    )
    add_test(
      NAME cupf_case30_naive_vs_opt
      COMMAND cupf_case_smoke
        --case-dir ${CMAKE_CURRENT_SOURCE_DIR}/../v1/core/dumps/case30_ieee
        --backend cpu --compare --compare-tol 1e-10
    )
  endif()

  if(WITH_CUDA)
    add_test(
      NAME cupf_case30_cuda_smoke
      COMMAND cupf_case_smoke
        --case-dir ${CMAKE_CURRENT_SOURCE_DIR}/../v1/core/dumps/case30_ieee
        --backend cuda
    )

    add_test(
      NAME cupf_case30_cuda_vertex_smoke
      COMMAND cupf_case_smoke
        --case-dir ${CMAKE_CURRENT_SOURCE_DIR}/../v1/core/dumps/case30_ieee
        --backend cuda
        --jacobian vertex_based
    )
  endif()
endif()

if(BUILD_BENCHMARKS)
  add_executable(cupf_case_benchmark
    benchmarks/cpp/cupf_case_benchmark.cpp
    tests/cpp/dump_case_loader.cpp
  )
  target_link_libraries(cupf_case_benchmark PRIVATE cupf_core)
  target_include_directories(cupf_case_benchmark
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
  )
endif()
