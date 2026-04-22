#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"

#include <cstdint>


// ---------------------------------------------------------------------------
// InitializeContext: Jacobian analysis 후 Buffers::prepare()에 전달하는 입력.
//
// solve() 전 한 번만 전달된다. Ybus 희소 구조, 버스 인덱스,
// Jacobian 맵 등 device-side 디스크립터를 초기화하는 데 필요한 정보를 담는다.
// ---------------------------------------------------------------------------
struct InitializeContext {
    const YbusView&           ybus;
    const JacobianScatterMap& maps;
    const JacobianPattern&    J;
    int32_t                   n_bus  = 0;
    const int32_t*            pv     = nullptr;
    int32_t                   n_pv   = 0;
    const int32_t*            pq     = nullptr;
    int32_t                   n_pq   = 0;
};


// ---------------------------------------------------------------------------
// SolveContext: solve() 시작 시 Buffers::upload()에 전달하는 입력.
// ---------------------------------------------------------------------------
struct SolveContext {
    const YbusView*             ybus;
    const std::complex<double>* sbus;
    const std::complex<double>* V0;
    const NRConfig*             config;

    int32_t batch_size   = 1;
    int64_t sbus_stride  = 0;
    int64_t V0_stride    = 0;

    bool    ybus_values_batched = false;
    int64_t ybus_value_stride   = 0;
};


// ---------------------------------------------------------------------------
// IterationContext: NR 반복 루프 내 모든 Op stage가 공유하는 상태.
//
// Op는 이 컨텍스트를 읽고 쓴다. IStorage 참조를 제거하여 Pipeline과
// 독립적으로 유지된다.
// ---------------------------------------------------------------------------
struct IterationContext {
    const NRConfig& config;

    const int32_t* pv   = nullptr;
    int32_t        n_pv = 0;
    const int32_t* pq   = nullptr;
    int32_t        n_pq = 0;

    int32_t iter      = 0;
    double  normF     = 0.0;
    bool    converged = false;

    bool    jacobian_updated_this_iter = false;
    int32_t jacobian_age               = 0;
};
