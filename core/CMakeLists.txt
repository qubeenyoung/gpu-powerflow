# =========================
# core/CMakeLists.txt
# =========================

# 소스 트리
set(CORE_CPP_SRCS
  cpp/jacobian.cpp
  cpp/mismatch.cpp
  cpp/newtonpf.cpp

  utils/io.cpp
  utils/nr_data.cpp
)

# cuDSS 소스 (선택)
if(BUILD_CUDSS)
  list(APPEND CORE_CPP_SRCS cpp/cudss_solver.cpp)
endif()

set(CORE_PUBLIC_HEADERS
  cpp/newtonpf.hpp
  utils/io.hpp
  utils/nr_data.hpp
  utils/dump.hpp
  utils/timer.hpp
  utils/cuda_utils.cuh   # CUDA OFF여도 헤더 참조 가능 (guard 필요)
)

# CUDA 소스(선택)
set(CORE_CUDA_SRCS)
if(BUILD_CUDA)
  # CUDA accelerator for hybrid Newton-Raphson
  list(APPEND CORE_CUDA_SRCS
    cuda/cuda_accel.cu
  )
endif()

# cuDSS permutation kernel (requires BUILD_CUDSS)
if(BUILD_CUDSS)
  list(APPEND CORE_CUDA_SRCS
    cuda/permutation_kernel.cu
  )
endif()

# -------------------------
# 코어 라이브러리
# -------------------------
add_library(newtonpf_core STATIC
  ${CORE_CPP_SRCS}
  ${CORE_CUDA_SRCS}
)

# 포함 경로
target_include_directories(newtonpf_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

# 컴파일 옵션/정의
target_compile_features(newtonpf_core PUBLIC cxx_std_17)
set_target_properties(newtonpf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -------------------------
# 링크 라이브러리
# -------------------------
# Eigen / BLAS / SuiteSparse / json / cnpy / spdlog
target_link_libraries(newtonpf_core
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    cnpy::cnpy
)

# OpenBLAS (배포에 따라 임포트 타깃이 있을 수도/없을 수도)
target_link_libraries(newtonpf_core
  PUBLIC
    BLAS::BLAS
    LAPACK::LAPACK
)

# SuiteSparse (배포별 변수명이 다를 수 있어 포괄적으로 처리)
target_include_directories(newtonpf_core PUBLIC ${SUITESPARSE_INCLUDE_DIR})
target_link_libraries(newtonpf_core PUBLIC ${SUITESPARSE_KLU_LIBS})

# AMGX (선택)
if(BUILD_AMGX)
  if(DEFINED AMGX_INCLUDE_DIR)
    target_include_directories(newtonpf_core PUBLIC ${AMGX_INCLUDE_DIR})
  endif()
  if(DEFINED AMGX_LIBRARY)
    target_link_libraries(newtonpf_core PUBLIC ${AMGX_LIBRARY})
  endif()
  # 소영CUDA: BUILD_AMGX 매크로 정의
  target_compile_definitions(newtonpf_core PUBLIC BUILD_AMGX)
  # 소영CUDA: AmgX needs OpenMP
  find_package(OpenMP REQUIRED)
  target_link_libraries(newtonpf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA (선택)
if(BUILD_CUDA)
  # 소영CUDA: CUDA 특화 옵션
  target_compile_options(newtonpf_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_definitions(newtonpf_core PUBLIC USE_CUDA)  # 소영수정: BUILD_CUDA 매크로 정의

  # 소영CUDA: CUDA 라이브러리 링크 (cuBLAS, cuSPARSE, cuSOLVER)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(newtonpf_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
  )
endif()

# cuDSS (선택)
if(BUILD_CUDSS)
  target_compile_definitions(newtonpf_core PUBLIC USE_CUDSS)

  # cuDSS library path (pip install nvidia-cudss-cu12)
  if(DEFINED CUDSS_INCLUDE_DIR)
    target_include_directories(newtonpf_core PUBLIC ${CUDSS_INCLUDE_DIR})
  else()
    # Default to pip-installed location
    set(CUDSS_INCLUDE_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/include")
    if(EXISTS ${CUDSS_INCLUDE_DIR})
      target_include_directories(newtonpf_core PUBLIC ${CUDSS_INCLUDE_DIR})
      message(STATUS "Found cuDSS include: ${CUDSS_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "cuDSS include directory not found: ${CUDSS_INCLUDE_DIR}")
    endif()
  endif()

  if(DEFINED CUDSS_LIBRARY)
    target_link_libraries(newtonpf_core PUBLIC ${CUDSS_LIBRARY})
  else()
    # Default to pip-installed location
    set(CUDSS_LIB_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/lib")
    find_library(CUDSS_LIB cudss HINTS ${CUDSS_LIB_DIR} NO_DEFAULT_PATH)
    if(CUDSS_LIB)
      target_link_libraries(newtonpf_core PUBLIC ${CUDSS_LIB})
      message(STATUS "Found cuDSS library: ${CUDSS_LIB}")
    else()
      message(FATAL_ERROR "cuDSS library not found in ${CUDSS_LIB_DIR}")
    endif()
  endif()

  # cuDSS requires CUDA runtime
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(newtonpf_core PUBLIC CUDA::cudart)
endif()

# =========================
# PyBind11 모듈 (선택)
# =========================
if(BUILD_PYBIND)
  # 파이썬 모듈명: newtonpf_cpp (import newtonpf_cpp)
  pybind11_add_module(newtonpf_cpp
    pybind/binding_helper.cpp
    pybind/cpp_binding.cpp
  )
  target_link_libraries(newtonpf_cpp
    PRIVATE
      newtonpf_core
      pybind11::module
      pybind11::headers
  )
  # LibTorch 연동이 필요한 경우에만 링크 (존재할 때만)
  if(TARGET Torch::Torch)
    target_link_libraries(newtonpf_cpp PRIVATE Torch::Torch)
    set_target_properties(newtonpf_cpp PROPERTIES
    BUILD_RPATH "\$ORIGIN;\$ORIGIN/.."
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/.."
    )
  endif()
  target_compile_features(newtonpf_cpp PRIVATE cxx_std_17)
endif()

# =========================
# 테스트 (선택)
# =========================
if(BUILD_TEST)
  add_executable(test_newtonpf test/test_newtonpf.cpp)
  target_link_libraries(test_newtonpf PRIVATE newtonpf_core)
  target_compile_features(test_newtonpf PRIVATE cxx_std_17)

  # CPU-GPU Integration Test
  add_executable(test_integration test/test_integration.cpp)
  target_link_libraries(test_integration PRIVATE newtonpf_core)
  target_compile_features(test_integration PRIVATE cxx_std_17)

  # Detailed Benchmark
  add_executable(benchmark_detailed test/benchmark_detailed.cpp)
  target_link_libraries(benchmark_detailed PRIVATE newtonpf_core)
  target_compile_features(benchmark_detailed PRIVATE cxx_std_17)

  # Verify All Converging Cases (Mixed Precision FP32)
  add_executable(verify_all_converging test/verify_all_converging.cpp)
  target_link_libraries(verify_all_converging PRIVATE newtonpf_core)
  target_compile_features(verify_all_converging PRIVATE cxx_std_17)

  # Multi-batch Jacobian Performance Test
  add_executable(benchmark_multibatch test/benchmark_multibatch.cpp)
  target_link_libraries(benchmark_multibatch PRIVATE newtonpf_core)
  target_compile_features(benchmark_multibatch PRIVATE cxx_std_17)

  # atomicAdd 병목 정량화 벤치마크
  add_executable(benchmark_atomic test/benchmark_atomic.cpp)
  target_link_libraries(benchmark_atomic PRIVATE newtonpf_core)
  target_compile_features(benchmark_atomic PRIVATE cxx_std_17)

  # memcpy vs Kernel 시간 측정
  add_executable(test_memcpy_timing test/test_memcpy_timing.cpp)
  target_link_libraries(test_memcpy_timing PRIVATE newtonpf_core)
  target_compile_features(test_memcpy_timing PRIVATE cxx_std_17)

  # No-atomic Kernel 프로파일링용
  add_executable(profile_no_atomic test/profile_no_atomic.cpp)
  target_link_libraries(profile_no_atomic PRIVATE newtonpf_core)
  target_compile_features(profile_no_atomic PRIVATE cxx_std_17)
endif()
