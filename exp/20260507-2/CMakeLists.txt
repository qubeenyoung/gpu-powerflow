cmake_minimum_required(VERSION 3.22)

project(exp_20260607_2_gmres_block_jacobi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

set(GINKGO_INSTALL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../20260510/lin_sol/third_party/ginkgo/install"
    CACHE PATH "Path to the local Ginkgo installation")
if(EXISTS "${GINKGO_INSTALL_DIR}/lib/cmake/Ginkgo/GinkgoConfig.cmake")
  find_package(Ginkgo CONFIG QUIET PATHS "${GINKGO_INSTALL_DIR}")
else()
  find_package(Ginkgo CONFIG QUIET)
endif()

find_library(LIB_METIS NAMES metis libmetis.so.5
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT LIB_METIS)
  message(FATAL_ERROR "METIS library is required.")
endif()

find_path(CUDSS_INCLUDE_DIR cudss.h
          PATHS
            /root/.local/lib/python3.10/site-packages/nvidia/cu12/include
            /usr/local/cuda/include
            /usr/include)
find_library(CUDSS_LIBRARY NAMES cudss libcudss.so
             PATHS
               /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib
               /usr/local/cuda/lib64
               /usr/lib
               /usr/lib/x86_64-linux-gnu)
find_path(AMGX_INCLUDE_DIR amgx_c.h
          PATHS
            /usr/local/include
            /usr/include)
find_library(AMGX_LIBRARY NAMES amgxsh amgx libamgxsh.so libamgx.a
             PATHS
               /usr/local/lib
               /usr/lib
               /usr/lib/x86_64-linux-gnu)

add_library(cuiter_gmres_block_jacobi STATIC
  src/kernels/block_jacobi_kernels.cu
  src/kernels/gmres_kernels.cu
  src/reorder/metis_partition.cpp
  src/preconditioner/metis_block_jacobi_preconditioner.cu
  src/solver/gmres_solver.cu
)

target_compile_features(cuiter_gmres_block_jacobi PUBLIC cxx_std_17)

target_include_directories(cuiter_gmres_block_jacobi
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(cuiter_gmres_block_jacobi
  PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    ${LIB_METIS}
)

set_target_properties(cuiter_gmres_block_jacobi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(gmres_block_jacobi_bench
  src/tools/gmres_block_jacobi_bench.cpp
)

target_compile_features(gmres_block_jacobi_bench PRIVATE cxx_std_17)

target_link_libraries(gmres_block_jacobi_bench
  PRIVATE
    cuiter_gmres_block_jacobi
)

set_target_properties(gmres_block_jacobi_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(metis_block_composition
  src/tools/metis_block_composition.cpp
  src/cupf_minimal/case_data.cpp
  src/cupf_minimal/jacobian_analysis.cpp
)

target_compile_features(metis_block_composition PRIVATE cxx_std_17)

target_include_directories(metis_block_composition
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(metis_block_composition
  PRIVATE
    cuiter_gmres_block_jacobi
)

set_target_properties(metis_block_composition PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(block_ilu_symbolic_analyzer
  src/tools/block_ilu_symbolic_analyzer.cpp
)

target_compile_features(block_ilu_symbolic_analyzer PRIVATE cxx_std_17)

target_include_directories(block_ilu_symbolic_analyzer
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(block_ilu_symbolic_analyzer
  PRIVATE
    cuiter_gmres_block_jacobi
)

set_target_properties(block_ilu_symbolic_analyzer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(diagonal_block_sparsity
  src/tools/diagonal_block_sparsity.cpp
)

target_compile_features(diagonal_block_sparsity PRIVATE cxx_std_17)

target_include_directories(diagonal_block_sparsity
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(diagonal_block_sparsity
  PRIVATE
    cuiter_gmres_block_jacobi
)

set_target_properties(diagonal_block_sparsity PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(Ginkgo_FOUND)
  add_executable(ginkgo_bicgstab_preconditioner_bench
    src/tools/ginkgo_bicgstab_preconditioner_bench.cpp
  )

  target_compile_features(ginkgo_bicgstab_preconditioner_bench PRIVATE cxx_std_17)

  target_include_directories(ginkgo_bicgstab_preconditioner_bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  target_link_libraries(ginkgo_bicgstab_preconditioner_bench
    PRIVATE
      Ginkgo::ginkgo
  )

  set_target_properties(ginkgo_bicgstab_preconditioner_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${GINKGO_INSTALL_DIR}/lib"
  )
else()
  message(WARNING "Ginkgo not found; ginkgo_bicgstab_preconditioner_bench will not be built.")
endif()

if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY)
  add_library(cupf_minimal_hybrid_nr STATIC
    src/cupf_minimal/case_data.cpp
    src/cupf_minimal/jacobian_analysis.cpp
    src/cupf_minimal/nr_kernels.cu
    src/cupf_minimal/direct_cudss_solver.cpp
    src/cupf_minimal/hybrid_nr.cpp
  )

  target_compile_features(cupf_minimal_hybrid_nr PUBLIC cxx_std_17)

  target_include_directories(cupf_minimal_hybrid_nr
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CUDSS_INCLUDE_DIR}"
  )

  target_link_libraries(cupf_minimal_hybrid_nr
    PUBLIC
      cuiter_gmres_block_jacobi
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      "${CUDSS_LIBRARY}"
  )

  if(Ginkgo_FOUND)
    target_compile_definitions(cupf_minimal_hybrid_nr PUBLIC CUITER_WITH_GINKGO=1)
    target_link_libraries(cupf_minimal_hybrid_nr PUBLIC Ginkgo::ginkgo)
  endif()

  set_target_properties(cupf_minimal_hybrid_nr PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(cudss_jf_bench
    src/tools/cudss_jf_bench.cpp
  )

  target_compile_features(cudss_jf_bench PRIVATE cxx_std_17)

  target_include_directories(cudss_jf_bench
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CUDSS_INCLUDE_DIR}"
  )

  target_link_libraries(cudss_jf_bench
    PRIVATE
      CUDA::cudart
      "${CUDSS_LIBRARY}"
  )

  get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)
  set_target_properties(cudss_jf_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(cudss_j11_bench
    src/tools/cudss_j11_bench.cpp
  )

  target_compile_features(cudss_j11_bench PRIVATE cxx_std_17)

  target_link_libraries(cudss_j11_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(cudss_j11_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(j11_correction_bench
    src/tools/j11_correction_bench.cpp
  )

  target_compile_features(j11_correction_bench PRIVATE cxx_std_17)

  target_link_libraries(j11_correction_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(j11_correction_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(j11_j22_correction_bench
    src/tools/j11_j22_correction_bench.cpp
  )

  target_compile_features(j11_j22_correction_bench PRIVATE cxx_std_17)

  target_link_libraries(j11_j22_correction_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(j11_j22_correction_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(hybrid_nr_bench
    src/tools/hybrid_nr_bench.cpp
  )

  target_compile_features(hybrid_nr_bench PRIVATE cxx_std_17)

  target_link_libraries(hybrid_nr_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(hybrid_nr_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  if(Ginkgo_FOUND)
    set_target_properties(hybrid_nr_bench PROPERTIES
      BUILD_RPATH "${CUDSS_LIBRARY_DIR};${GINKGO_INSTALL_DIR}/lib"
    )
  endif()

  add_executable(ras_overlap_symbolic_analyzer
    src/tools/ras_overlap_symbolic_analyzer.cpp
  )

  target_compile_features(ras_overlap_symbolic_analyzer PRIVATE cxx_std_17)

  target_link_libraries(ras_overlap_symbolic_analyzer
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(ras_overlap_symbolic_analyzer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(ras_overlap_standalone_quality
    src/tools/ras_overlap_standalone_quality.cpp
  )

  target_compile_features(ras_overlap_standalone_quality PRIVATE cxx_std_17)

  target_link_libraries(ras_overlap_standalone_quality
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(ras_overlap_standalone_quality PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(metis_coupling_drift_diagnostic
    src/tools/metis_coupling_drift_diagnostic.cpp
  )

  target_compile_features(metis_coupling_drift_diagnostic PRIVATE cxx_std_17)

  target_link_libraries(metis_coupling_drift_diagnostic
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(metis_coupling_drift_diagnostic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(block_ilu0_pilot_bench
    src/tools/block_ilu0_pilot_bench.cpp
  )

  target_compile_features(block_ilu0_pilot_bench PRIVATE cxx_std_17)

  target_link_libraries(block_ilu0_pilot_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(block_ilu0_pilot_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(gpu_block_ilu0_phase_bench
    src/tools/gpu_block_ilu0_phase_bench.cu
  )

  target_compile_features(gpu_block_ilu0_phase_bench PRIVATE cxx_std_17)

  target_link_libraries(gpu_block_ilu0_phase_bench
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(gpu_block_ilu0_phase_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  add_executable(dx_diff_localization
    src/tools/dx_diff_localization.cpp
  )

  target_compile_features(dx_diff_localization PRIVATE cxx_std_17)

  target_link_libraries(dx_diff_localization
    PRIVATE
      cupf_minimal_hybrid_nr
  )

  set_target_properties(dx_diff_localization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
  )

  if(AMGX_INCLUDE_DIR AND AMGX_LIBRARY)
    add_executable(amgx_jf_oracle_bench
      src/tools/amgx_jf_oracle_bench.cpp
    )

    target_compile_features(amgx_jf_oracle_bench PRIVATE cxx_std_17)

    target_include_directories(amgx_jf_oracle_bench
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CUDSS_INCLUDE_DIR}"
        "${AMGX_INCLUDE_DIR}"
    )

    target_link_libraries(amgx_jf_oracle_bench
      PRIVATE
        cupf_minimal_hybrid_nr
        "${AMGX_LIBRARY}"
    )

    if(OpenMP_CXX_FOUND)
      target_link_libraries(amgx_jf_oracle_bench PRIVATE OpenMP::OpenMP_CXX)
    endif()

    get_filename_component(AMGX_LIBRARY_DIR "${AMGX_LIBRARY}" DIRECTORY)
    set_target_properties(amgx_jf_oracle_bench PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      BUILD_RPATH "${CUDSS_LIBRARY_DIR};${AMGX_LIBRARY_DIR}"
    )
  else()
    message(WARNING "AMGX headers/library not found; amgx_jf_oracle_bench will not be built.")
  endif()
else()
  message(WARNING "cuDSS headers/library not found; cudss_jf_bench and hybrid_nr_bench will not be built.")
endif()
