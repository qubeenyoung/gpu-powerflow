# =========================
# Root CMakeLists.txt
# =========================
cmake_minimum_required(VERSION 3.22)

# C++/CUDA 활성화
project(NewtonPF LANGUAGES CXX CUDA)

# ---- C++ 표준 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 기본 빌드 타입(미지정 시 Release) ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================
# 옵션
# =========================
option(ENABLE_BLOCK_TIMER "USE_BLOCK_TIMER 매크로 활성화" OFF)
option(ENABLE_DUMP_DATA   "DUMP_DATA 매크로 활성화" OFF)

option(BUILD_CPP   "CPP Backend" ON)
option(BUILD_CUDA  "CUDA Backend" OFF)
option(BUILD_CUDSS "cuDSS GPU Sparse Solver" OFF)
option(BUILD_AMGX  "AMGX Backend" OFF)

option(BUILD_PYBIND "Python 바인딩(PyBind11/LibTorch)" ON)
option(BUILD_TEST   "테스트 타깃 빌드" ON)

# 선택적 매크로
if(ENABLE_BLOCK_TIMER)
  add_compile_definitions(USE_BLOCK_TIMER)
  message(STATUS "USE_BLOCK_TIMER enabled")
endif()

if(ENABLE_DUMP_DATA)
  add_compile_definitions(DUMP_DATA)
  message(STATUS "DUMP_DATA enabled")
endif()

# =========================
# 필수/공용 라이브러리
# =========================
# 헤더 전용/임포트 타깃 위주로 사용
find_package(Eigen3 REQUIRED)               # Eigen3::Eigen
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

# ---- SuiteSparse (KLU) manual find ----
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES klu.h btf.h amd.h colamd.h SuiteSparse_config.h suitesparse_config.h
  PATHS /usr/include /usr/include/suitesparse
)

find_library(LIB_KLU      NAMES klu      PATHS /usr/lib/x86_64-linux-gnu)
find_library(LIB_BTF      NAMES btf      PATHS /usr/lib/x86_64-linux-gnu)
find_library(LIB_AMD      NAMES amd      PATHS /usr/lib/x86_64-linux-gnu)
find_library(LIB_COLAMD   NAMES colamd   PATHS /usr/lib/x86_64-linux-gnu)
find_library(LIB_SSCONFIG NAMES suitesparseconfig SuiteSparse_config
                           PATHS /usr/lib/x86_64-linux-gnu)

if(NOT SUITESPARSE_INCLUDE_DIR)
  message(FATAL_ERROR "SuiteSparse headers not found (klu.h, ...).")
endif()
if(NOT LIB_KLU)
  message(FATAL_ERROR "libklu not found.")
endif()

set(SUITESPARSE_KLU_LIBS
  ${LIB_KLU}
  ${LIB_BTF}
  ${LIB_AMD}
  ${LIB_COLAMD}
  ${LIB_SSCONFIG}
)
message(STATUS "SuiteSparse include: ${SUITESPARSE_INCLUDE_DIR}")
message(STATUS "SuiteSparse KLU libs: ${SUITESPARSE_KLU_LIBS}")
# ---- SuiteSparse (KLU) manual find end ----

find_package(nlohmann_json 3.12.0 REQUIRED) # nlohmann_json::nlohmann_json
find_package(spdlog REQUIRED)               # spdlog::spdlog
find_package(ZLIB REQUIRED)                 # ZLIB::ZLIB
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "OpenBLAS: ${OpenBLAS_LIBRARIES}")
message(STATUS "SuiteSparse: ${SuiteSparse_LIBRARIES}")

# cnpy: 패키지가 없으면 /usr/local 기준 IMPORTED 타깃 생성
find_package(cnpy QUIET)
if(NOT TARGET cnpy::cnpy)
  add_library(cnpy::cnpy SHARED IMPORTED)
  set_target_properties(cnpy::cnpy PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libcnpy.so"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
  )
  target_link_libraries(cnpy::cnpy INTERFACE ZLIB::ZLIB)
  message(STATUS "cnpy not found by config; using IMPORTED /usr/local/lib/libcnpy.so")
else()
  message(STATUS "Found cnpy via package config")
endif()

# =========================
# CUDA
# =========================
if(BUILD_CUDA)
  # 언어 활성화로 충분. 별도 SDK 구성 필요 시:
  # find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA build enabled")
endif()

# =========================
# Python/pybind11/LibTorch (선택)
# =========================
if(BUILD_PYBIND)

  find_package(pybind11 REQUIRED CONFIG)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")

  # LibTorch 경로를 Python에서 직접 조회해 CMAKE_PREFIX_PATH에 주입 (환경변수 의존 제거)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(TORCH_CMAKE_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    message(STATUS "Torch CMake prefix: ${TORCH_CMAKE_PREFIX}")
  else()
    message(WARNING "Could not query torch.utils.cmake_prefix_path; ensure LibTorch is installed in this env.")
  endif()

  find_package(Torch REQUIRED)      # Torch::Torch
  message(STATUS "LibTorch found: ${TORCH_LIBRARIES}")
  message(STATUS "pybind11 found")
endif()

# =========================
# AMGX (선택)
# =========================
if(BUILD_AMGX)
  # 배포에 CMake config가 없을 수 있으므로 수동 힌트
  # 필요 시 -DAMGX_LIBRARY=... -DAMGX_INCLUDE_DIR=... 로 오버라이드
  find_library(AMGX_LIBRARY NAMES amgxsh PATHS /usr/local/lib)
  find_path(AMGX_INCLUDE_DIR NAMES amgx_c.h PATHS /usr/local/include)
  if(NOT AMGX_LIBRARY OR NOT AMGX_INCLUDE_DIR)
    message(FATAL_ERROR "AMGX not found. Set AMGX_LIBRARY and AMGX_INCLUDE_DIR.")
  endif()
  message(STATUS "Found AMGX: ${AMGX_LIBRARY} @ ${AMGX_INCLUDE_DIR}")
  # 실제 링크는 core/ 타깃에서 처리
endif()

# =========================
# 하위 디렉터리
# =========================
add_subdirectory(core)

# =========================
# 상태 출력
# =========================
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Options -> CPP:${BUILD_CPP} CUDA:${BUILD_CUDA} AMGX:${BUILD_AMGX} PYBIND:${BUILD_PYBIND} TEST:${BUILD_TEST}")
