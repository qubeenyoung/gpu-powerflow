# ---------------------------------------------------------------------------
# Benchmark executables
# ---------------------------------------------------------------------------

set(CUPF_CASE_BENCHMARK_SOURCES
  cpp/cupf_case_benchmark.cpp
  ${CMAKE_SOURCE_DIR}/tests/dump_case_loader.cpp
)

if(CUPF_SUPERLU_ENABLED)
  list(APPEND CUPF_CASE_BENCHMARK_SOURCES
    ${CMAKE_SOURCE_DIR}/cpp/src/newton_solver/reference/cpu_naive_jacobian_f64.cpp
    ${CMAKE_SOURCE_DIR}/cpp/src/newton_solver/reference/cpu_superlu_solve.cpp
  )
endif()

add_executable(cupf_case_benchmark ${CUPF_CASE_BENCHMARK_SOURCES})

target_link_libraries(cupf_case_benchmark PRIVATE cupf)
target_include_directories(cupf_case_benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/cpp/inc
  ${CMAKE_SOURCE_DIR}/cpp/src
  ${CMAKE_SOURCE_DIR}/tests
)
