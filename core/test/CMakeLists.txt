# --- 1. 테스트 실행 파일 정의 ---
add_executable(test_newtonpf test_newtonpf.cpp)

# --- 2. 코어 라이브러리 링크 ---
target_link_libraries(test_newtonpf PRIVATE newtonpf_core)

target_compile_features(test_newtonpf PRIVATE cxx_std_17)

# --- 3. Multi-batch 벤치마크 실행 파일 ---
add_executable(benchmark_multibatch benchmark_multibatch.cpp)
target_link_libraries(benchmark_multibatch PRIVATE newtonpf_core)
target_compile_features(benchmark_multibatch PRIVATE cxx_std_17)

# --- 4. atomicAdd 벤치마크 실행 파일 ---
add_executable(benchmark_atomic benchmark_atomic.cpp)
target_link_libraries(benchmark_atomic PRIVATE newtonpf_core)
target_compile_features(benchmark_atomic PRIVATE cxx_std_17)

# --- 5. 세부 타이밍 벤치마크 실행 파일 ---
add_executable(benchmark_detailed benchmark_detailed.cpp)
target_link_libraries(benchmark_detailed PRIVATE newtonpf_core)
target_compile_features(benchmark_detailed PRIVATE cxx_std_17)


