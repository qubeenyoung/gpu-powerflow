#pragma once

// ---------------------------------------------------------------------------
// cudss_config.hpp — cuDSS 핸들·솔버 설정 헬퍼
//
// cuDSS 관련 CMake 옵션을 cudssHandle/cudssConfig API 에 연결한다.
// cuda_cudss32.cpp / cuda_cudss64.cpp 에서만 include 한다.
//
// CMake 옵션 설명:
//
//   CUPF_CUDSS_REORDERING_ALG (기본값: CUDSS_ALG_DEFAULT)
//     희소 행렬 재순서화(reordering) 알고리즘 선택.
//     fill-in 을 줄여 인수분해 속도를 향상시킨다.
//     선택 가능 값: CUDSS_ALG_DEFAULT, CUDSS_ALG_1, CUDSS_ALG_2, CUDSS_ALG_3
//
//   CUPF_CUDSS_ENABLE_MT (정의 시 활성화, 기본 비활성)
//     cuDSS 멀티스레드 레이어 활성화.
//     cudssSetThreadingLayer() 로 host-side 병렬 인수분해를 활성화한다.
//     CUPF_CUDSS_HOST_NTHREADS 와 함께 사용한다.
//
//   CUPF_CUDSS_HOST_NTHREADS (정의 시 숫자값)
//     host-side 스레드 수 지정. 미정의 시 cuDSS 기본값(일반적으로 1) 사용.
//
//   CUPF_CUDSS_ND_NLEVELS (정의 시 숫자값)
//     Nested Dissection 재귀 레벨 수.
//     대형 계통(수천 버스 이상)에서 fill-in 감소 효과가 크다.
//     CUPF_CUDSS_REORDERING_ALG=CUDSS_ALG_2 와 함께 사용하면 효과적.
//
//   CUPF_CUDSS_USE_MATCHING + CUPF_CUDSS_MATCHING_ALG
//     cuDSS matching을 활성화하고 matching 알고리즘을 지정한다.
//
//   CUPF_CUDSS_ENABLE_MG + CUPF_CUDSS_MG_DEVICE_INDICES_STR
//     cuDSS multi-GPU handle(cudssCreateMg)과 device config를 사용한다.
//     예: CUPF_CUDSS_MG_DEVICE_INDICES_STR="0,1"
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "utils/cuda_utils.hpp"

#ifdef CUPF_ENABLE_CUDSS

#include <cctype>
#include <stdexcept>
#include <string_view>
#include <vector>

// 기본값: cuDSS 내부 알고리즘 선택에 맡김
#ifndef CUPF_CUDSS_REORDERING_ALG
#define CUPF_CUDSS_REORDERING_ALG CUDSS_ALG_DEFAULT
#endif

#ifndef CUPF_CUDSS_MATCHING_ALG
#define CUPF_CUDSS_MATCHING_ALG CUDSS_ALG_DEFAULT
#endif

#ifndef CUPF_CUDSS_MG_DEVICE_INDICES_STR
#define CUPF_CUDSS_MG_DEVICE_INDICES_STR ""
#endif

namespace cupf_cudss_detail {

#ifdef CUPF_CUDSS_ENABLE_MG
inline std::vector<int> parse_mg_device_indices()
{
    constexpr std::string_view indices_text(CUPF_CUDSS_MG_DEVICE_INDICES_STR);
    std::vector<int> indices;

    std::size_t start = 0;
    while (start <= indices_text.size()) {
        const std::size_t comma = indices_text.find(',', start);
        const std::size_t end = (comma == std::string_view::npos) ? indices_text.size() : comma;
        if (end == start) {
            throw std::runtime_error("CUPF_CUDSS_MG_DEVICE_INDICES_STR contains an empty device index");
        }

        int value = 0;
        for (std::size_t i = start; i < end; ++i) {
            const unsigned char ch = static_cast<unsigned char>(indices_text[i]);
            if (!std::isdigit(ch)) {
                throw std::runtime_error("CUPF_CUDSS_MG_DEVICE_INDICES_STR must contain only comma-separated non-negative integers");
            }
            value = value * 10 + static_cast<int>(ch - static_cast<unsigned char>('0'));
        }
        indices.push_back(value);

        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }

    if (indices.empty()) {
        throw std::runtime_error("CUPF_CUDSS_ENABLE_MG requires at least one cuDSS MG device index");
    }
    return indices;
}

inline std::vector<int>& mg_device_indices()
{
    static std::vector<int> indices = parse_mg_device_indices();
    return indices;
}
#endif

// cuDSS handle 생성. MG 빌드에서는 cudssCreateMg로 device set을 고정한다.
inline void create_handle(cudssHandle_t* handle)
{
#ifdef CUPF_CUDSS_ENABLE_MG
    std::vector<int>& indices = mg_device_indices();
    CUDSS_CHECK(cudssCreateMg(handle, static_cast<int>(indices.size()), indices.data()));
#else
    CUDSS_CHECK(cudssCreate(handle));
#endif
}

// 멀티스레드 레이어 설정 (CUPF_CUDSS_ENABLE_MT 정의 시 활성화).
inline void configure_handle(cudssHandle_t handle)
{
#ifdef CUPF_CUDSS_ENABLE_MT
    CUDSS_CHECK(cudssSetThreadingLayer(handle, nullptr));
#else
    (void)handle;
#endif
}

// 재순서화 알고리즘, 호스트 스레드 수, Nested Dissection 레벨 적용.
inline void configure_solver(cudssConfig_t config)
{
    cudssAlgType_t reordering_alg = CUPF_CUDSS_REORDERING_ALG;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_REORDERING_ALG,
        &reordering_alg,
        sizeof(reordering_alg)));

#ifdef CUPF_CUDSS_USE_MATCHING
    int use_matching = 1;
    cudssAlgType_t matching_alg = CUPF_CUDSS_MATCHING_ALG;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_USE_MATCHING,
        &use_matching,
        sizeof(use_matching)));
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_MATCHING_ALG,
        &matching_alg,
        sizeof(matching_alg)));
#endif

#ifdef CUPF_CUDSS_ENABLE_MG
    std::vector<int>& device_indices = mg_device_indices();
    int device_count = static_cast<int>(device_indices.size());
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_DEVICE_COUNT,
        &device_count,
        sizeof(device_count)));
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_DEVICE_INDICES,
        device_indices.data(),
        sizeof(int) * device_indices.size()));
#endif

#ifdef CUPF_CUDSS_HOST_NTHREADS
    int host_nthreads = CUPF_CUDSS_HOST_NTHREADS;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_HOST_NTHREADS,
        &host_nthreads,
        sizeof(host_nthreads)));
#endif

#ifdef CUPF_CUDSS_ND_NLEVELS
    int nd_nlevels = CUPF_CUDSS_ND_NLEVELS;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_ND_NLEVELS,
        &nd_nlevels,
        sizeof(nd_nlevels)));
#endif
}

}  // namespace cupf_cudss_detail

#endif  // CUPF_ENABLE_CUDSS

#endif  // CUPF_WITH_CUDA
