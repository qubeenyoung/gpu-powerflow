# --- 1. 파이썬 모듈(.so) 정의 ---
# (모듈 이름은 'newtonpf_cpp'가 됩니다)
pybind11_add_module(newtonpf_cpp SHARED
  binding_helper.cpp
  cpp_binding.cpp
)

# --- 2. 코어 라이브러리 및 LibTorch 링크 ---
target_link_libraries(newtonpf_cpp
  PRIVATE
    newtonpf_core     # C++/CUDA 핵심 로직
    pybind11::module  # Pybind11 자체
)

# Pybind 모듈은 LibTorch가 추가로 필요함
if(TARGET Torch::Torch)
  target_link_libraries(newtonpf_cpp PRIVATE Torch::Torch)
endif()

target_compile_features(newtonpf_cpp PRIVATE cxx_std_17)