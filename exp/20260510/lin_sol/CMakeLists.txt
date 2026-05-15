cmake_minimum_required(VERSION 3.22)

project(lin_sol_benchmark LANGUAGES CXX CUDA)

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

set(COMMON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/common/cpp")

find_path(CUDSS_INCLUDE_DIR cudss.h
    PATHS
        "$ENV{CUDSS_ROOT}/include"
        "/root/.local/lib/python3.10/site-packages/nvidia/cu12/include"
        "/usr/local/cuda/include"
        "/usr/local/cuda/targets/x86_64-linux/include"
        "/usr/local/include"
)
find_library(CUDSS_LIBRARY
    NAMES cudss libcudss.so
    PATHS
        "$ENV{CUDSS_ROOT}/lib"
        "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib"
        "/usr/local/cuda/lib64"
        "/usr/local/cuda/targets/x86_64-linux/lib"
        "/usr/local/lib"
)

if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY)
    add_executable(cudss_benchmark solvers/cudss/cudss_benchmark.cu)
    target_include_directories(cudss_benchmark PRIVATE "${COMMON_INCLUDE}" "${CUDSS_INCLUDE_DIR}")
    target_link_libraries(cudss_benchmark PRIVATE CUDA::cudart "${CUDSS_LIBRARY}")
    target_compile_definitions(cudss_benchmark PRIVATE CUDSS_LIBRARY_PATH_STR="${CUDSS_LIBRARY}")
    get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)
    set_target_properties(cudss_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
    )
else()
    message(WARNING "cuDSS headers/libraries not found; cudss_benchmark will not be built.")
endif()

add_executable(cusolver_benchmark solvers/cusolver/cusolver_benchmark.cu)
target_include_directories(cusolver_benchmark PRIVATE "${COMMON_INCLUDE}")
target_link_libraries(cusolver_benchmark PRIVATE CUDA::cudart CUDA::cusolver CUDA::cusparse)
target_compile_definitions(cusolver_benchmark PRIVATE CUSOLVER_LIBRARY_PATH_STR="$<TARGET_FILE:CUDA::cusolver>")
set_target_properties(cusolver_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

find_path(AMGX_INCLUDE_DIR amgx_c.h
    PATHS
        "$ENV{AMGX_ROOT}/include"
        "/usr/local/include"
        "/usr/include"
)
find_library(AMGX_LIBRARY
    NAMES amgxsh amgx libamgxsh.so libamgx.a
    PATHS
        "$ENV{AMGX_ROOT}/lib"
        "/usr/local/lib"
        "/usr/lib"
        "/usr/lib/x86_64-linux-gnu"
)

if(AMGX_INCLUDE_DIR AND AMGX_LIBRARY)
    add_executable(amgx_benchmark solvers/amgx/amgx_benchmark.cu)
    target_include_directories(amgx_benchmark PRIVATE "${COMMON_INCLUDE}" "${AMGX_INCLUDE_DIR}")
    target_link_libraries(amgx_benchmark PRIVATE CUDA::cudart "${AMGX_LIBRARY}" dl)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(amgx_benchmark PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(amgx_benchmark PRIVATE AMGX_LIBRARY_PATH_STR="${AMGX_LIBRARY}")
    get_filename_component(AMGX_LIBRARY_DIR "${AMGX_LIBRARY}" DIRECTORY)
    set_target_properties(amgx_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${AMGX_LIBRARY_DIR}"
    )
else()
    message(WARNING "AMGx headers/libraries not found; amgx_benchmark will not be built.")
endif()

find_package(Ginkgo CONFIG
    PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ginkgo/install/lib/cmake/Ginkgo"
        "$ENV{GINKGO_ROOT}/lib/cmake/Ginkgo"
        "$ENV{GINKGO_ROOT}/lib64/cmake/Ginkgo"
)

if(Ginkgo_FOUND)
    add_executable(ginkgo_benchmark solvers/ginkgo/ginkgo_benchmark.cpp)
    target_include_directories(ginkgo_benchmark PRIVATE "${COMMON_INCLUDE}")
    target_link_libraries(ginkgo_benchmark PRIVATE Ginkgo::ginkgo CUDA::cudart)
    target_compile_definitions(ginkgo_benchmark PRIVATE
        GINKGO_LIBRARY_PATH_STR="$<TARGET_FILE:Ginkgo::ginkgo>"
    )
    set_target_properties(ginkgo_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ginkgo/install/lib"
    )
else()
    message(WARNING "Ginkgo was not found; ginkgo_benchmark will not be built.")
endif()

find_package(MPI)
find_library(SUPERLU_DIST_LIBRARY
    NAMES superlu_dist libsuperlu_dist.a
    PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/superlu_dist/install/lib"
        "$ENV{SUPERLU_DIST_ROOT}/lib"
)
find_path(SUPERLU_DIST_INCLUDE_DIR superlu_ddefs.h
    PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/superlu_dist/install/include"
        "$ENV{SUPERLU_DIST_ROOT}/include"
)

if(MPI_CXX_FOUND AND SUPERLU_DIST_LIBRARY AND SUPERLU_DIST_INCLUDE_DIR)
    add_executable(superlu_dist_benchmark solvers/superlu_dist/superlu_dist_benchmark.cpp)
    target_include_directories(superlu_dist_benchmark PRIVATE "${COMMON_INCLUDE}" "${SUPERLU_DIST_INCLUDE_DIR}")
    target_link_libraries(superlu_dist_benchmark PRIVATE
        MPI::MPI_CXX
        "${SUPERLU_DIST_LIBRARY}"
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        OpenMP::OpenMP_CXX
        /usr/lib/x86_64-linux-gnu/libblas.so
        /usr/lib/x86_64-linux-gnu/liblapack.so
        m
    )
    target_compile_definitions(superlu_dist_benchmark PRIVATE
        SUPERLU_DIST_LIBRARY_PATH_STR="${SUPERLU_DIST_LIBRARY}"
    )
    set_target_properties(superlu_dist_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
else()
    message(WARNING "SuperLU_DIST headers/libraries or MPI CXX were not found; superlu_dist_benchmark will not be built.")
endif()
