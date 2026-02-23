/**
 * @brief CUDA API 호출을 확인하고 에러 시 예외를 던지는 매크로
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            spdlog::error("CUDA Error at {}:{}: {} ({})", \
                __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)