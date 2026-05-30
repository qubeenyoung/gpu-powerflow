cmake_minimum_required(VERSION 3.22)

project(custom_linear_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLS_BUILD_CUDA_OPS "Build copied CUDA multifrontal operation kernels" ON)
option(CLS_BUILD_SCRIPTS "Build custom_linear_solver command-line scripts" ON)
option(CLS_BUILD_CUDSS_SCRIPT "Build cuDSS command-line comparison script" OFF)
option(CLS_ENABLE_NVTX "Enable NVTX ranges in command-line profiling scripts" OFF)
set(CLS_CUDA_ARCHITECTURES "86" CACHE STRING
  "CUDA architectures for custom_linear_solver_ops; must be >= 60 for atomicAdd(double)")

add_library(custom_linear_solver_api INTERFACE)
target_include_directories(custom_linear_solver_api INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

if(CLS_BUILD_CUDA_OPS)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    PATHS /usr/include /usr/local/include /opt/third_party/install/common/include)
  find_library(METIS_LIBRARY
    NAMES metis
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/third_party/install/common/lib)

  if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "custom_linear_solver requires METIS for nested-dissection ordering.")
  endif()

  set(CLS_OP_SOURCES
    src/solver.cpp
    src/matrix/pattern_kernels.cu
    src/plan/multifrontal_plan.cu
    src/factorize/multifrontal.cu
    src/solve/multifrontal.cu
    src/symbolic/elimination_tree.cpp
    src/symbolic/supernode.cpp
    src/symbolic/multifrontal.cpp
    src/reordering/metis_nd.cpp
  )

  add_library(custom_linear_solver_ops STATIC ${CLS_OP_SOURCES})
  target_include_directories(custom_linear_solver_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${METIS_INCLUDE_DIR}
  )
  target_link_libraries(custom_linear_solver_ops PUBLIC
    CUDA::cudart
    ${METIS_LIBRARY}
  )
  set_target_properties(custom_linear_solver_ops PROPERTIES
    CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
  )

  target_link_libraries(custom_linear_solver_api INTERFACE custom_linear_solver_ops)

  if(CLS_BUILD_SCRIPTS)
    add_executable(custom_linear_solver_run
      scripts/io.cpp
      scripts/run_custom_solver.cu
    )
    target_include_directories(custom_linear_solver_run PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts
    )
    target_link_libraries(custom_linear_solver_run PRIVATE
      custom_linear_solver_api
    )
    set_target_properties(custom_linear_solver_run PROPERTIES
      CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF
    )

    if(CLS_BUILD_CUDSS_SCRIPT)
      find_path(CUDSS_INCLUDE_DIR
        NAMES cuDSS.h cudss.h
        HINTS
          /usr/include
          /usr/local/include
          /usr/local/cuda/include
          $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/include)
      find_library(CUDSS_LIBRARY
        NAMES cudss
        HINTS
          /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/lib)

      if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
        message(FATAL_ERROR "CLS_BUILD_CUDSS_SCRIPT=ON requires cuDSS headers/library.")
      endif()
      get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)

      add_executable(cudss_run
        scripts/io.cpp
        scripts/run_cudss_solver.cpp
      )
      target_include_directories(cudss_run PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDSS_INCLUDE_DIR}
      )
      target_link_libraries(cudss_run PRIVATE
        CUDA::cudart
        ${CUDSS_LIBRARY}
      )
      set(CUDSS_RUN_RPATH "${CUDSS_LIBRARY_DIR}")
      if(CLS_ENABLE_NVTX)
        find_path(NVTX_INCLUDE_DIR
          NAMES nvtx3/nvToolsExt.h nvToolsExt.h
          HINTS
            /usr/local/cuda/include
            /usr/local/cuda-12.8/targets/x86_64-linux/include
            $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/nvtx/include
            /usr/local/lib/python3.10/dist-packages/nvidia/nvtx/include)
        find_library(NVTOOLSEXT_LIBRARY
          NAMES nvToolsExt
          HINTS
            /usr/local/cuda/lib64
            /usr/local/cuda-12.8/targets/x86_64-linux/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu)
        if(NOT NVTX_INCLUDE_DIR OR NOT NVTOOLSEXT_LIBRARY)
          message(FATAL_ERROR "CLS_ENABLE_NVTX=ON requires NVTX headers and libnvToolsExt.")
        endif()
        get_filename_component(NVTOOLSEXT_LIBRARY_DIR "${NVTOOLSEXT_LIBRARY}" DIRECTORY)
        target_compile_definitions(cudss_run PRIVATE CLS_ENABLE_NVTX=1)
        target_include_directories(cudss_run PRIVATE ${NVTX_INCLUDE_DIR})
        target_link_libraries(cudss_run PRIVATE ${NVTOOLSEXT_LIBRARY})
        set(CUDSS_RUN_RPATH "${CUDSS_RUN_RPATH};${NVTOOLSEXT_LIBRARY_DIR}")
      endif()
      set_target_properties(cudss_run PROPERTIES
        BUILD_RPATH "${CUDSS_RUN_RPATH}"
      )
    endif()
  endif()
endif()
