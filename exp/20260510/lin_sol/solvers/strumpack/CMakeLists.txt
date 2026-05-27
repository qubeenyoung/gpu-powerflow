cmake_minimum_required(VERSION 3.22)

project(strumpack_lin_sol_benchmark LANGUAGES C CXX CUDA Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(STRUMPACK CONFIG REQUIRED
    PATHS
        "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/lin_sol/strumpack/install/lib/cmake/STRUMPACK"
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install_accel_on/lib/cmake/STRUMPACK"
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install_nolto/lib/cmake/STRUMPACK"
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install/lib/cmake/STRUMPACK"
)

set(STRUMPACK_BENCH_LIBRARY_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/lin_sol/strumpack/install/lib/libstrumpack.a")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/lin_sol/strumpack/install/lib/libstrumpack.a")
    set(STRUMPACK_BENCH_LIBRARY_PATH
        "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/lin_sol/strumpack/install/lib/libstrumpack.a")
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install_accel_on/lib/libstrumpack.a")
    set(STRUMPACK_BENCH_LIBRARY_PATH
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install_accel_on/lib/libstrumpack.a")
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install/lib/libstrumpack.a")
    set(STRUMPACK_BENCH_LIBRARY_PATH
        "${CMAKE_CURRENT_LIST_DIR}/../../third_party/strumpack/install/lib/libstrumpack.a")
endif()

add_executable(strumpack_benchmark strumpack_benchmark.cpp strumpack_cuda_link.cu)
target_include_directories(strumpack_benchmark PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../common/cpp"
)
target_link_libraries(strumpack_benchmark PRIVATE STRUMPACK::strumpack)
target_compile_definitions(strumpack_benchmark PRIVATE
    STRUMPACK_LIBRARY_PATH_STR="${STRUMPACK_BENCH_LIBRARY_PATH}"
)
set_target_properties(strumpack_benchmark PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(strumpack_seq_benchmark strumpack_seq_benchmark.cpp strumpack_cuda_link.cu)
target_include_directories(strumpack_seq_benchmark PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../common/cpp"
)
target_link_libraries(strumpack_seq_benchmark PRIVATE STRUMPACK::strumpack)
target_compile_definitions(strumpack_seq_benchmark PRIVATE
    STRUMPACK_LIBRARY_PATH_STR="${STRUMPACK_BENCH_LIBRARY_PATH}"
)
set_target_properties(strumpack_seq_benchmark PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(strumpack_ybus_benchmark strumpack_ybus_benchmark.cpp strumpack_cuda_link.cu)
target_link_libraries(strumpack_ybus_benchmark PRIVATE STRUMPACK::strumpack CUDA::cudart)
target_compile_definitions(strumpack_ybus_benchmark PRIVATE
    STRUMPACK_LIBRARY_PATH_STR="${STRUMPACK_BENCH_LIBRARY_PATH}"
)
set_target_properties(strumpack_ybus_benchmark PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
