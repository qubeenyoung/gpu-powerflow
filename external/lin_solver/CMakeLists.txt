cmake_minimum_required(VERSION 3.22)

project(sparse_direct_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggressive host optimization for the numeric kernels (auto-vectorize the scalar
# factor/solve loops). Release only.
add_compile_options($<$<CONFIG:Release>:-march=native>
                    $<$<CONFIG:Release>:-funroll-loops>)

find_package(MPI REQUIRED)

# Optional CUDA (GPU factorization research track). Guarded so CPU-only builds
# still work if nvcc is absent.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 (sm_86); needed for atomicAdd(double)
endif()

option(ENABLE_TIMER "Enable PROFILE_SCOPE wall-time logging" OFF)
option(ENABLE_CUDA_TIMER "Enable PROFILE_SCOPE CUDA event timing" OFF)
option(ENABLE_NVTX "Enable PROFILE_SCOPE NVTX ranges for Nsight tools" OFF)

set(SUITESPARSE_ROOT "/opt/third_party/install/common" CACHE PATH "SuiteSparse install prefix")
set(MKL_ROOT "/opt/intel/oneapi/mkl/2026.0" CACHE PATH "oneMKL install prefix")
set(MUMPS_ROOT "/opt/third_party/install/mumps" CACHE PATH "MUMPS install prefix")
set(PASTIX_ROOT "/opt/third_party/install/pastix-cuda" CACHE PATH "PaStiX install prefix")
set(STRUMPACK_ROOT "/opt/third_party/install/strumpack-cuda" CACHE PATH "STRUMPACK install prefix")
set(GLU3_ROOT "/opt/third_party/install/glu3" CACHE PATH "GLU 3.0 install prefix")
set(GLU3_SOURCE_ROOT "/opt/third_party/src/glu3" CACHE PATH "GLU 3.0 source prefix")
set(PANGULU_ROOT "/opt/third_party/install/pangulu" CACHE PATH "PanguLU install prefix")
set(CUDSS_ROOT "/opt/nvidia/cudss" CACHE PATH "cuDSS install prefix")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA install prefix")

add_library(matrix_io
    src/tools/matrix_io.cpp
)

target_include_directories(matrix_io PUBLIC
    src
)

add_library(compute_error
    src/tools/compute_error.cpp
)

target_include_directories(compute_error PUBLIC
    src
)

target_link_libraries(compute_error PUBLIC
    matrix_io
)

add_library(timer
    src/tools/timer.cpp
)

target_include_directories(timer PUBLIC
    src
    "${CUDA_ROOT}/include"
)

if(ENABLE_TIMER)
    target_compile_definitions(timer PUBLIC PROFILE_ENABLE_TIMER)
endif()

if(ENABLE_CUDA_TIMER)
    target_compile_definitions(timer PUBLIC PROFILE_ENABLE_CUDA_TIMER)
    target_link_directories(timer PUBLIC
        "${CUDA_ROOT}/lib64"
    )
    target_link_libraries(timer PUBLIC
        cudart
    )
endif()

add_library(nvtx_profiler
    src/tools/nvtx_profiler.cpp
)

target_include_directories(nvtx_profiler PUBLIC
    src
    "${CUDA_ROOT}/include"
)

if(ENABLE_NVTX)
    target_compile_definitions(nvtx_profiler PUBLIC PROFILE_ENABLE_NVTX)
endif()

add_library(profiling
    src/tools/profiling.cpp
)

target_include_directories(profiling PUBLIC
    src
)

target_link_libraries(profiling PUBLIC
    timer
    nvtx_profiler
)

# cy168: LD_PRELOAD shared lib for DETERMINISTIC parallel nested dissection.
# Run with LD_PRELOAD=$<TARGET_FILE:det_rand> to make the parND ordering reproducible
# (thread-local rand_r interposes METIS's racy global glibc rand()). See thread_safe_rand.c.
add_library(det_rand SHARED src/mysolver/reordering/thread_safe_rand.cpp)

add_executable(benchmark
    src/benchmark/third_party_solvers.cpp
    src/benchmark/solver_registry.cpp
    src/benchmark/mysolver_gpu_solver.cu
    src/mysolver/gpu/gpu_mf.cu
    src/mysolver/solver.cpp
    src/mysolver/reordering/metis_nd.cpp
    src/mysolver/reordering/gpu_nd.cu
    src/mysolver/reordering/mc64.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/factorize/own_pipeline.cpp
    src/third_party_solvers/utils.cpp
    src/third_party_solvers/superlu_solver.cpp
    src/third_party_solvers/suitesparse_solvers.cpp
    src/third_party_solvers/pardiso_solver.cpp
    src/third_party_solvers/mumps_solver.cpp
    src/third_party_solvers/pastix_solver.cpp
    src/third_party_solvers/strumpack_solver.cpp
    src/third_party_solvers/cudss_solver.cpp
    src/third_party_solvers/glu_solver.cpp
    src/third_party_solvers/pangulu_solver.cpp
)

target_include_directories(benchmark PRIVATE
    src
    "${SUITESPARSE_ROOT}/include"
    "${MKL_ROOT}/include"
    "${MUMPS_ROOT}/include"
    "${PASTIX_ROOT}/include"
    "${STRUMPACK_ROOT}/include"
    "${GLU3_ROOT}/include"
    "${GLU3_SOURCE_ROOT}/src/preprocess"
    "${GLU3_SOURCE_ROOT}/src/nicslu/include"
    "${PANGULU_ROOT}/include"
    "${CUDSS_ROOT}/include"
    "${CUDA_ROOT}/include"
)

target_link_directories(benchmark PRIVATE
    "${SUITESPARSE_ROOT}/lib"
    "${MKL_ROOT}/lib"
    "${MUMPS_ROOT}/lib"
    "${PASTIX_ROOT}/lib"
    "${STRUMPACK_ROOT}/lib"
    "${GLU3_ROOT}/lib"
    "${PANGULU_ROOT}/lib"
    "${CUDSS_ROOT}/lib"
    "${CUDA_ROOT}/lib64"
)

target_link_libraries(benchmark PRIVATE
    matrix_io
    compute_error
    profiling
    klu
    umfpack
    superlu
    mkl_rt
    dmumps
    mumps_common
    pord
    pastix
    spm
    pastix_kernels
    pastix_kernels_cuda
    pastix_starpu
    strumpack
    "${GLU3_SOURCE_ROOT}/src/symbolic.o"
    "${GLU3_SOURCE_ROOT}/src/numeric.o"
    "${GLU3_SOURCE_ROOT}/src/Timer.o"
    "${GLU3_SOURCE_ROOT}/src/preprocess/preprocess.o"
    "${GLU3_ROOT}/lib/nicslu.a"
    "${GLU3_ROOT}/lib/nicslu_util.a"
    pangulu
    mynd
    cudss
    cudart
    cublas
    cusparse
    metis
    amd
    colamd
    btf
    suitesparseconfig
    m
    MPI::MPI_CXX
)

set_target_properties(benchmark PROPERTIES
    BUILD_RPATH "${SUITESPARSE_ROOT}/lib;${MKL_ROOT}/lib;${MUMPS_ROOT}/lib;${PASTIX_ROOT}/lib;${STRUMPACK_ROOT}/lib;${GLU3_ROOT}/lib;${PANGULU_ROOT}/lib;${CUDSS_ROOT}/lib;${CUDA_ROOT}/lib64"
)
target_compile_definitions(benchmark PRIVATE MYSOLVER_HAVE_GPU_ND)

add_executable(mysolver_test
    tests/mysolver/test_solver.cpp
    src/mysolver/solver.cpp
    src/mysolver/reordering/metis_nd.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/schedule.cpp
    src/mysolver/factorize/dense_lu.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/factorize/supernodal.cpp
    src/mysolver/symbolic/multifrontal.cpp
)

target_include_directories(mysolver_test PRIVATE
    src
    "${SUITESPARSE_ROOT}/include"
    "${MKL_ROOT}/include"
)

target_link_directories(mysolver_test PRIVATE
    "${SUITESPARSE_ROOT}/lib"
    "${MKL_ROOT}/lib"
)

target_link_libraries(mysolver_test PRIVATE
    klu
    metis
    cxsparse
    amd
    btf
    colamd
    suitesparseconfig
    mkl_rt
    m
)

set_target_properties(mysolver_test PROPERTIES
    BUILD_RPATH "${SUITESPARSE_ROOT}/lib;${MKL_ROOT}/lib"
)

add_executable(own_numeric_check
    src/tools/own_numeric_check.cpp
    src/mysolver/factorize/own_pipeline.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
    src/mysolver/reordering/mc64.cpp
)

target_include_directories(own_numeric_check PRIVATE
    src
    "${SUITESPARSE_ROOT}/include"
)

target_link_directories(own_numeric_check PRIVATE
    "${SUITESPARSE_ROOT}/lib"
)

target_link_libraries(own_numeric_check PRIVATE
    matrix_io
    compute_error
    amd
    btf
    suitesparseconfig
    m
)

set_target_properties(own_numeric_check PROPERTIES
    BUILD_RPATH "${SUITESPARSE_ROOT}/lib"
)

add_executable(mc64_check
    src/tools/mc64_check.cpp
    src/mysolver/reordering/mc64.cpp
)
target_include_directories(mc64_check PRIVATE src)
target_link_libraries(mc64_check PRIVATE matrix_io m)

add_executable(refactor_bench
    src/tools/refactor_bench.cpp
    src/mysolver/factorize/own_pipeline.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
    src/mysolver/reordering/mc64.cpp
)
target_include_directories(refactor_bench PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(refactor_bench PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(refactor_bench PRIVATE matrix_io compute_error amd btf suitesparseconfig m)
set_target_properties(refactor_bench PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(amalg_stats
    src/tools/amalg_stats.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
)
target_include_directories(amalg_stats PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(amalg_stats PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(amalg_stats PRIVATE matrix_io amd suitesparseconfig m)
set_target_properties(amalg_stats PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(multifrontal_check
    src/tools/multifrontal_check.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
)
target_include_directories(multifrontal_check PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(multifrontal_check PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(multifrontal_check PRIVATE matrix_io amd suitesparseconfig m)
set_target_properties(multifrontal_check PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(mf_factor_check
    src/tools/mf_factor_check.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
)
target_include_directories(mf_factor_check PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(mf_factor_check PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(mf_factor_check PRIVATE matrix_io amd suitesparseconfig m)
set_target_properties(mf_factor_check PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(btf_stats src/tools/btf_stats.cpp)
target_include_directories(btf_stats PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(btf_stats PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(btf_stats PRIVATE matrix_io btf suitesparseconfig m)
set_target_properties(btf_stats PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(level_stats
    src/tools/level_stats.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/multifrontal.cpp
    src/mysolver/factorize/sparse_lu.cpp
    src/mysolver/reordering/metis_nd.cpp
)
target_include_directories(level_stats PRIVATE src "${SUITESPARSE_ROOT}/include")
target_link_directories(level_stats PRIVATE "${SUITESPARSE_ROOT}/lib")
target_link_libraries(level_stats PRIVATE matrix_io metis amd suitesparseconfig m)
set_target_properties(level_stats PROPERTIES BUILD_RPATH "${SUITESPARSE_ROOT}/lib")

add_executable(symbolic_stats
    src/tools/symbolic_stats.cpp
    src/mysolver/symbolic/elimination_tree.cpp
    src/mysolver/symbolic/supernode.cpp
    src/mysolver/symbolic/schedule.cpp
)

target_include_directories(symbolic_stats PRIVATE
    src
    "${SUITESPARSE_ROOT}/include"
)

target_link_directories(symbolic_stats PRIVATE
    "${SUITESPARSE_ROOT}/lib"
)

target_link_libraries(symbolic_stats PRIVATE
    matrix_io
    amd
    cxsparse
    suitesparseconfig
    m
)

set_target_properties(symbolic_stats PROPERTIES
    BUILD_RPATH "${SUITESPARSE_ROOT}/lib"
)

if(CMAKE_CUDA_COMPILER)
    add_executable(gpu_test
        tests/mysolver/test_gpu.cpp
        src/mysolver/gpu/gpu_spmv.cu
        src/mysolver/gpu/gpu_factor.cu
        src/mysolver/gpu/gpu_solve.cu
        src/mysolver/factorize/sparse_lu.cpp
        src/mysolver/symbolic/elimination_tree.cpp
        src/mysolver/symbolic/supernode.cpp
        src/mysolver/symbolic/multifrontal.cpp
    )
    target_include_directories(gpu_test PRIVATE src "${CUDA_ROOT}/include")
    set_target_properties(gpu_test PROPERTIES BUILD_RPATH "${CUDA_ROOT}/lib64")

    add_executable(gpu_factor_bench
        src/tools/gpu_factor_bench.cu
        src/mysolver/gpu/gpu_factor.cu
        src/mysolver/factorize/sparse_lu.cpp
        src/mysolver/symbolic/elimination_tree.cpp
        src/mysolver/symbolic/supernode.cpp
        src/mysolver/symbolic/multifrontal.cpp
        src/mysolver/reordering/metis_nd.cpp
        src/tools/matrix_io.cpp
    )
    target_include_directories(gpu_factor_bench PRIVATE src
        "${CUDA_ROOT}/include" "${SUITESPARSE_ROOT}/include")
    target_link_directories(gpu_factor_bench PRIVATE "${SUITESPARSE_ROOT}/lib")
    target_link_libraries(gpu_factor_bench PRIVATE metis amd suitesparseconfig m)
    set_target_properties(gpu_factor_bench PROPERTIES
        BUILD_RPATH "${CUDA_ROOT}/lib64;${SUITESPARSE_ROOT}/lib")

    add_executable(gpu_mf_bench
        src/tools/gpu_mf_bench.cu
        src/mysolver/gpu/gpu_factor.cu
        src/mysolver/gpu/gpu_mf.cu
        src/mysolver/factorize/sparse_lu.cpp
        src/mysolver/symbolic/elimination_tree.cpp
        src/mysolver/symbolic/supernode.cpp
        src/mysolver/symbolic/multifrontal.cpp
        src/mysolver/reordering/metis_nd.cpp
        src/mysolver/reordering/gpu_nd.cu
        src/tools/matrix_io.cpp
    )
    target_include_directories(gpu_mf_bench PRIVATE src
        "${CUDA_ROOT}/include" "${SUITESPARSE_ROOT}/include")
    target_link_directories(gpu_mf_bench PRIVATE "${SUITESPARSE_ROOT}/lib")
    target_link_libraries(gpu_mf_bench PRIVATE metis amd suitesparseconfig m)
    set_target_properties(gpu_mf_bench PROPERTIES
        BUILD_RPATH "${CUDA_ROOT}/lib64;${SUITESPARSE_ROOT}/lib")
    target_compile_definitions(gpu_mf_bench PRIVATE MYSOLVER_HAVE_GPU_ND)

    add_executable(circuit_mf_test
        src/tools/circuit_mf_test.cu
        src/mysolver/gpu/gpu_mf.cu
        src/mysolver/factorize/sparse_lu.cpp
        src/mysolver/symbolic/elimination_tree.cpp
        src/mysolver/symbolic/supernode.cpp
        src/mysolver/symbolic/multifrontal.cpp
        src/mysolver/reordering/mc64.cpp
        src/mysolver/reordering/metis_nd.cpp
        src/mysolver/reordering/gpu_nd.cu
        src/tools/matrix_io.cpp
    )
    target_include_directories(circuit_mf_test PRIVATE src
        "${CUDA_ROOT}/include" "${SUITESPARSE_ROOT}/include")
    target_link_directories(circuit_mf_test PRIVATE "${SUITESPARSE_ROOT}/lib")
    target_link_libraries(circuit_mf_test PRIVATE metis amd suitesparseconfig m)
    set_target_properties(circuit_mf_test PROPERTIES
        BUILD_RPATH "${CUDA_ROOT}/lib64;${SUITESPARSE_ROOT}/lib")
    target_compile_definitions(circuit_mf_test PRIVATE MYSOLVER_HAVE_GPU_ND)

    add_executable(batched_gemm_probe src/tools/batched_gemm_probe.cu)
    target_include_directories(batched_gemm_probe PRIVATE src "${CUDA_ROOT}/include")
    target_link_libraries(batched_gemm_probe PRIVATE cublas m)
    set_target_properties(batched_gemm_probe PROPERTIES BUILD_RPATH "${CUDA_ROOT}/lib64")

    add_executable(fp_probe src/tools/fp_probe.cu)
    target_include_directories(fp_probe PRIVATE src "${CUDA_ROOT}/include")
    target_link_libraries(fp_probe PRIVATE m)
    set_target_properties(fp_probe PROPERTIES BUILD_RPATH "${CUDA_ROOT}/lib64")

    add_executable(gpu_solve_bench
        src/tools/gpu_solve_bench.cu
        src/mysolver/gpu/gpu_solve.cu
        src/mysolver/factorize/sparse_lu.cpp
        src/mysolver/symbolic/elimination_tree.cpp
        src/mysolver/symbolic/supernode.cpp
        src/mysolver/symbolic/multifrontal.cpp
        src/mysolver/reordering/metis_nd.cpp
        src/tools/matrix_io.cpp
    )
    target_include_directories(gpu_solve_bench PRIVATE src
        "${CUDA_ROOT}/include" "${SUITESPARSE_ROOT}/include")
    target_link_directories(gpu_solve_bench PRIVATE "${SUITESPARSE_ROOT}/lib")
    target_link_libraries(gpu_solve_bench PRIVATE metis amd suitesparseconfig m)
    set_target_properties(gpu_solve_bench PROPERTIES
        BUILD_RPATH "${CUDA_ROOT}/lib64;${SUITESPARSE_ROOT}/lib")
endif()

add_executable(prepare_dataset_vectors
    prepare_datasets/cpp/prepare_dataset_vectors.cpp
)

target_include_directories(prepare_dataset_vectors PRIVATE
    src
    "${SUITESPARSE_ROOT}/include"
    "${SUITESPARSE_ROOT}/include/suitesparse"
)

target_link_directories(prepare_dataset_vectors PRIVATE
    "${SUITESPARSE_ROOT}/lib"
)

target_link_libraries(prepare_dataset_vectors PRIVATE
    matrix_io
    klu
    amd
    btf
    colamd
    suitesparseconfig
    m
)

set_target_properties(prepare_dataset_vectors PROPERTIES
    BUILD_RPATH "${SUITESPARSE_ROOT}/lib"
)
