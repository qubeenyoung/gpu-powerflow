#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#if __has_include(<cudss.h>)
#include <cudss.h>
#elif __has_include("cudss.h")
#include "cudss.h"
#endif

#include <stdexcept>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = (call); \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuSPARSE error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - status=" + std::to_string(static_cast<int>(status))); \
        } \
    } while (0)
#endif

#ifndef CUDSS_CHECK
#define CUDSS_CHECK(call) \
    do { \
        cudssStatus_t status = (call); \
        if (status != CUDSS_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuDSS error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - status=" + std::to_string(static_cast<int>(status))); \
        } \
    } while (0)
#endif
