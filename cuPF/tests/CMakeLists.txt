# ---------------------------------------------------------------------------
# C++ evaluator executable (shares dump_case_loader with the gtest target)
# ---------------------------------------------------------------------------
if(BUILD_EVALUATORS)
  add_executable(cupf_cpp_evaluate
    cpp/cupf_cpp_evaluate.cpp
    dump_case_loader.cpp
  )
  target_link_libraries(cupf_cpp_evaluate PRIVATE cupf)
  target_include_directories(cupf_cpp_evaluate PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp/inc
    ${CMAKE_CURRENT_SOURCE_DIR})   # so #include "dump_case_loader.hpp" resolves
endif()

# Evaluator-only configure (BUILD_EVALUATORS=ON, BUILD_TESTING=OFF): stop here.
if(NOT BUILD_TESTING)
  return()
endif()

# ---------------------------------------------------------------------------
# Dependency-free smoke test
# ---------------------------------------------------------------------------
add_executable(cupf_minimal_tests
  small_cases.cpp
  test_minimal_solver.cpp
)

target_link_libraries(cupf_minimal_tests PRIVATE cupf)
target_include_directories(cupf_minimal_tests PRIVATE ${PROJECT_SOURCE_DIR}/cpp/inc)
add_test(NAME cupf_minimal_tests COMMAND cupf_minimal_tests)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    add_test(NAME cupf_python_bindings_smoke
      COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_bindings.py")
    set_tests_properties(cupf_python_bindings_smoke PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_cupf>:${CMAKE_SOURCE_DIR}/python")

    if(CUPF_WITH_TORCH)
      add_test(NAME cupf_python_torch_autograd_smoke
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_torch_autograd.py")
      set_tests_properties(cupf_python_torch_autograd_smoke PROPERTIES
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:_cupf>:${CMAKE_SOURCE_DIR}/python")
    endif()
  else()
    message(STATUS "cuPF: Python interpreter not found, skipping Python binding smoke tests")
  endif()
endif()

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

set(CUPF_TEST_SOURCES
  dump_case_loader.cpp
  small_cases.cpp
  test_cpu_solver.cpp
)

if(NOT CUPF_TEST_SOURCES)
  message(STATUS "cuPF: test skeleton present but no C++ test sources are registered yet")
  return()
endif()

# ---------------------------------------------------------------------------
# C++ unit tests
# ---------------------------------------------------------------------------
add_executable(cupf_tests ${CUPF_TEST_SOURCES})

target_link_libraries(cupf_tests PRIVATE cupf GTest::gtest_main)
target_include_directories(cupf_tests PRIVATE ${PROJECT_SOURCE_DIR}/cpp/inc)

include(GoogleTest)
gtest_discover_tests(cupf_tests)
