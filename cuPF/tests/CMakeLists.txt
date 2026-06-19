# ===========================================================================
# cuPF/tests layout
#   run_cupf.py            the runner (measurement) — the one you run
#   cupf_bench.cpp         C++ benchmark that run_cupf.py drives (cpp path)
#   dump_case_loader.*     shared dump-case loader (bench + gtest)
#   unit/                  C++ correctness tests (fixtures + minimal smoke + gtest)
#   smoke/                 Python smoke tests (ctest)
#   scripts/build_cupf.sh  thin cmake build wrapper
# ===========================================================================

# ---------------------------------------------------------------------------
# cupf_bench: the single cuPF C++ benchmark (solve / operators / debug modes).
# ---------------------------------------------------------------------------
if(BUILD_EVALUATORS)
  add_executable(cupf_bench
    cupf_bench.cpp
    dump_case_loader.cpp
  )
  target_link_libraries(cupf_bench PRIVATE cupf)
  if(WITH_CUDA)
    target_link_libraries(cupf_bench PRIVATE CUDA::cudart)  # cudaDeviceSynchronize for clean timing
  endif()
  target_include_directories(cupf_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp/inc
    ${CMAKE_CURRENT_SOURCE_DIR})   # so #include "dump_case_loader.hpp" resolves
endif()

# Evaluator-only configure (BUILD_EVALUATORS=ON, BUILD_TESTING=OFF): stop here.
if(NOT BUILD_TESTING)
  return()
endif()

# ---------------------------------------------------------------------------
# unit/ — dependency-free C++ smoke test (no GoogleTest)
# ---------------------------------------------------------------------------
add_executable(cupf_minimal_tests
  unit/small_cases.cpp
  unit/test_minimal_solver.cpp
)
target_link_libraries(cupf_minimal_tests PRIVATE cupf)
target_include_directories(cupf_minimal_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/cpp/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/unit)   # small_cases.hpp
add_test(NAME cupf_minimal_tests COMMAND cupf_minimal_tests)

# ---------------------------------------------------------------------------
# smoke/ — Python binding + torch autograd smoke tests
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    add_test(NAME cupf_python_bindings_smoke
      COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_bindings.py")
    set_tests_properties(cupf_python_bindings_smoke PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_cupf>:${CMAKE_SOURCE_DIR}/python")

    if(CUPF_WITH_TORCH)
      add_test(NAME cupf_python_torch_autograd_smoke
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_torch_autograd.py")
      set_tests_properties(cupf_python_torch_autograd_smoke PROPERTIES
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_cupf>:${CMAKE_SOURCE_DIR}/python")
    endif()
  else()
    message(STATUS "cuPF: Python interpreter not found, skipping Python binding smoke tests")
  endif()
endif()

# ---------------------------------------------------------------------------
# unit/ — GoogleTest C++ unit tests (built only when GoogleTest is available)
# ---------------------------------------------------------------------------
option(CUPF_REQUIRE_GTEST
  "Fail configure when BUILD_TESTING=ON but GoogleTest is unavailable" OFF)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  if(CUPF_REQUIRE_GTEST)
    message(FATAL_ERROR "cuPF: GoogleTest not found and CUPF_REQUIRE_GTEST=ON")
  endif()
  message(STATUS "cuPF: GoogleTest not found, skipping GTest-based C++ tests")
  return()
endif()

add_executable(cupf_tests
  dump_case_loader.cpp
  unit/small_cases.cpp
  unit/test_cpu_solver.cpp
)
target_link_libraries(cupf_tests PRIVATE cupf GTest::gtest_main)
target_include_directories(cupf_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/cpp/inc
  ${CMAKE_CURRENT_SOURCE_DIR}        # dump_case_loader.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unit)  # small_cases.hpp

include(GoogleTest)
gtest_discover_tests(cupf_tests)
