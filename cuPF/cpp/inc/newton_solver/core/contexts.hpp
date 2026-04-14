#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/jacobian_types.hpp"
#include "newton_solver/ops/op_interfaces.hpp"

#include <cstdint>


// ---------------------------------------------------------------------------
// AnalyzeContext: JacobianBuilder 분석 후 IStorage::prepare()에 전달하는 입력.
//
// solve() 전 한 번만 전달된다. Ybus 희소 구조, 버스 인덱스,
// Jacobian 맵 등 device-side 디스크립터를 초기화하는 데 필요한 정보를 담는다.
// ---------------------------------------------------------------------------
struct AnalyzeContext {
    const YbusView&          ybus;    // Ybus 희소 구조 (값은 무시될 수 있음)
    const JacobianMaps&      maps;    // Jacobian 산포 맵
    const JacobianStructure& J;       // Jacobian CSR 희소 패턴
    int32_t                  n_bus  = 0;
    const int32_t*           pv     = nullptr;
    int32_t                  n_pv   = 0;
    const int32_t*           pq     = nullptr;
    int32_t                  n_pq   = 0;
};


// ---------------------------------------------------------------------------
// SolveContext: solve() 시작 시 IStorage::upload()에 전달하는 입력.
//
// 현재 Ybus 값, Sbus, 초기 전압 벡터를 담는다.
// public I/O는 항상 FP64다.
// ---------------------------------------------------------------------------
struct SolveContext {
    const YbusViewF64*          ybus;    // Ybus 값 (FP64)
    const std::complex<double>* sbus;    // 복소 전력 주입 (FP64)
    const std::complex<double>* V0;      // 초기 전압 벡터 (FP64)
    const NRConfig*             config;
};


// ---------------------------------------------------------------------------
// IterationContext: NR 반복 루프 내 모든 Op stage가 공유하는 상태.
//
// 각 Op는 이 컨텍스트를 읽고 쓴다.
// MismatchOp 실행 후 converged가 true이면 루프를 종료한다.
// ---------------------------------------------------------------------------
struct IterationContext {
    IStorage&       storage;
    const NRConfig& config;

    const int32_t* pv   = nullptr;
    int32_t        n_pv = 0;
    const int32_t* pq   = nullptr;
    int32_t        n_pq = 0;

    int32_t iter      = 0;     // 현재 반복 횟수 (0-based)
    double  normF     = 0.0;   // 최신 mismatch 노름
    bool    converged = false;
};
