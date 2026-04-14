find_package(GTest QUIET)

if(NOT GTest_FOUND)
  message(STATUS "cuPF: GoogleTest not found, skipping C++ tests")
  return()
endif()

set(CUPF_TEST_SOURCES
  dump_case_loader.cpp
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
target_include_directories(cupf_tests PRIVATE ${CMAKE_SOURCE_DIR}/cpp/inc)

include(GoogleTest)
gtest_discover_tests(cupf_tests)
