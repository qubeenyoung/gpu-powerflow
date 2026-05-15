// Forces a CUDA device-link step when linking the STRUMPACK static CUDA library.
extern "C" __global__ void strumpack_benchmark_cuda_link_anchor() {}
