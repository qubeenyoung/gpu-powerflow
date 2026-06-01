#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"

#include <cstdint>
#include <string>


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
// AdjointCache: per-pipeline record of the factorization cached at the final
// converged state for a later solve_adjoint() (J^T solve). Holds both the
// "is the cache usable" flags the adjoint path checks and provenance/diagnostic
// flags surfaced in AdjointResult (how J^T was obtained for this backend).
// Reset at the start of every forward solve and (re)populated by
// prepare_adjoint_cache(). See newton_solver.cpp / newton_solver_adjoint.cpp.
// ---------------------------------------------------------------------------
struct AdjointCache {
    // --- usability: is there an exact cached factorization to reuse? ---
    bool has_adjoint_cache = false;                    // a factorization is cached
    bool adjoint_cache_matches_final_state = false;    // ...and it is the converged-state J
    bool factorization_supports_transpose_solve = false;  // cache can solve J^T directly (CPU/KLU)
    bool refactorized_for_adjoint_cache = false;       // J was refactorized to build the cache
    bool reused_forward_factorization = false;         // reused the last NR-iteration factorization

    // --- provenance flags (diagnostics; copied into AdjointResult) ---
    bool used_explicit_transpose = false;              // explicit J^T was materialized (cuDSS)
    bool includes_host_device_transfer = false;
    bool jt_symbolic_analyzed_at_initialize = false;   // J^T symbolic analysis done in initialize()
    bool jt_values_transposed_on_device = false;       // J values scattered to J^T on device
    bool jt_factorized_during_forward_cache = false;   // J^T factorized while caching (not in backward)
    bool host_roundtrip_for_jt_transpose = false;

    // --- dimensions / metadata ---
    int64_t final_state_generation = 0;
    double final_mismatch_norm = 0.0;
    int32_t batch_size = 0;     // batch the cache was built for (must match the adjoint call)
    int32_t dimF = 0;
    std::string backend_name;                  // e.g. "cuda_cudss_fp64", "cpu_klu"
    std::string transpose_solve_backend_name;
    double factorization_time_ms = 0.0;        // wall-clock of the cache factorization
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
