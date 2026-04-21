#pragma once

// ---------------------------------------------------------------------------
// cudss_config.hpp — cuDSS 핸들·솔버 설정 헬퍼
//
// cuDSS 관련 CMake 옵션과 NewtonOptions.cudss 런타임 설정을
// cudssHandle/cudssConfig API 에 연결한다.
// cuda_cudss32.cpp / cuda_cudss64.cpp 에서만 include 한다.
//
// CMake 옵션 설명:
//
//   CUPF_CUDSS_REORDERING_ALG (기본값: CUDSS_ALG_DEFAULT)
//     희소 행렬 재순서화(reordering) 알고리즘 선택.
//     fill-in 을 줄여 인수분해 속도를 향상시킨다.
//     선택 가능 값: CUDSS_ALG_DEFAULT, CUDSS_ALG_1 (AMD), CUDSS_ALG_2 (ND)
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
//     현재 빌드 옵션에서는 CUPF_CUDSS_REORDERING_ALG=CUDSS_ALG_DEFAULT 와 함께 사용한다.
//
// 런타임 옵션:
//
//   NewtonOptions.cudss.use_matching
//     factorization 전 matching 전처리를 활성화한다.
//
//   NewtonOptions.cudss.matching_alg
//     matching 활성화 시 사용할 알고리즘.
//
//   NewtonOptions.cudss.auto_pivot_epsilon / pivot_epsilon
//     작은 pivot을 대체할 pivot epsilon. auto_pivot_epsilon=true면 cuDSS 기본값 사용.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"
#include "utils/cuda_utils.hpp"

#include <cmath>
#include <stdexcept>

#ifdef CUPF_ENABLE_CUDSS

// 기본값: cuDSS 내부 알고리즘 선택에 맡김
#ifndef CUPF_CUDSS_REORDERING_ALG
#define CUPF_CUDSS_REORDERING_ALG CUDSS_ALG_DEFAULT
#endif

namespace cupf_cudss_detail {

// 멀티스레드 레이어 설정 (CUPF_CUDSS_ENABLE_MT 정의 시 활성화).
inline void configure_handle(cudssHandle_t handle)
{
#ifdef CUPF_CUDSS_ENABLE_MT
    CUDSS_CHECK(cudssSetThreadingLayer(handle, nullptr));
#else
    (void)handle;
#endif
}

inline cudssAlgType_t to_cudss_alg(CuDSSAlgorithm alg)
{
    switch (alg) {
    case CuDSSAlgorithm::Default:
        return CUDSS_ALG_DEFAULT;
    case CuDSSAlgorithm::Alg1:
        return CUDSS_ALG_1;
    case CuDSSAlgorithm::Alg2:
        return CUDSS_ALG_2;
    case CuDSSAlgorithm::Alg3:
        return CUDSS_ALG_3;
    case CuDSSAlgorithm::Alg4:
        return CUDSS_ALG_4;
    case CuDSSAlgorithm::Alg5:
        return CUDSS_ALG_5;
    }
    return CUDSS_ALG_DEFAULT;
}

inline bool analysis_requires_matrix_values(const CuDSSOptions& options)
{
    return options.use_matching;
}

// 재순서화 알고리즘, matching, pivot epsilon 등 solver config 적용.
inline void configure_solver(cudssConfig_t config, const CuDSSOptions& options)
{
    cudssAlgType_t reordering_alg = CUPF_CUDSS_REORDERING_ALG;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_REORDERING_ALG,
        &reordering_alg,
        sizeof(reordering_alg)));

    if (options.use_matching) {
        if (reordering_alg == CUDSS_ALG_1 || reordering_alg == CUDSS_ALG_2) {
            throw std::invalid_argument(
                "cuDSS matching is not supported with CUPF_CUDSS_REORDERING_ALG=ALG_1 or ALG_2");
        }

        int use_matching = 1;
        CUDSS_CHECK(cudssConfigSet(
            config,
            CUDSS_CONFIG_USE_MATCHING,
            &use_matching,
            sizeof(use_matching)));

        cudssAlgType_t matching_alg = to_cudss_alg(options.matching_alg);
        CUDSS_CHECK(cudssConfigSet(
            config,
            CUDSS_CONFIG_MATCHING_ALG,
            &matching_alg,
            sizeof(matching_alg)));
    }

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

    if (!options.auto_pivot_epsilon) {
        if (!std::isfinite(options.pivot_epsilon) || options.pivot_epsilon < 0.0) {
            throw std::invalid_argument("cuDSS pivot_epsilon must be finite and non-negative");
        }

        double pivot_epsilon = options.pivot_epsilon;
        CUDSS_CHECK(cudssConfigSet(
            config,
            CUDSS_CONFIG_PIVOT_EPSILON,
            &pivot_epsilon,
            sizeof(pivot_epsilon)));
    }
}

}  // namespace cupf_cudss_detail

#endif  // CUPF_ENABLE_CUDSS

#endif  // CUPF_WITH_CUDA
