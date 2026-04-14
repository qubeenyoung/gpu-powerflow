#pragma once

#include "jacobian_types.hpp"

#include <complex>
#include <cstdint>
#include <vector>


// ---------------------------------------------------------------------------
// CSRView: 희소 행렬에 대한 non-owning view.
//
// 메모리를 소유하지 않으며 할당도 하지 않는다.
// 호출자가 배열 생명주기를 관리한다.
// Python 바인딩 레이어에서 numpy 배열의 raw pointer를 이 구조체로 감싼다.
// ---------------------------------------------------------------------------
template<typename T, typename IndexType = int32_t>
struct CSRView {
    const IndexType* indptr;   // 행 포인터,     크기 = rows + 1
    const IndexType* indices;  // 열 인덱스,     크기 = nnz
    const T*         data;     // 비영 원소 값,  크기 = nnz

    IndexType rows, cols, nnz;
};


// ---------------------------------------------------------------------------
// Ybus view 타입 별칭.
// ---------------------------------------------------------------------------
using YbusViewF64 = CSRView<std::complex<double>>;
using YbusView    = YbusViewF64;


// ---------------------------------------------------------------------------
// NRConfig: Newton-Raphson 수렴 조건 설정.
// ---------------------------------------------------------------------------
struct NRConfig {
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
};


// ---------------------------------------------------------------------------
// BackendKind: 연산 백엔드 선택.
// ---------------------------------------------------------------------------
enum class BackendKind {
    CPU,
    CUDA,
};


// ---------------------------------------------------------------------------
// ComputePolicy: 내부 계산 정밀도 정책.
//
//   FP64  — 내부 전 단계 FP64 (CPU 또는 CUDA).
//   Mixed — public I/O는 FP64, 내부 Jacobian/solve는 FP32 (CUDA 전용).
//           고정 프로파일이며, stage별 자유 조합이 아니다.
//
// Mixed 프로파일의 내부 구성:
//   mismatch       — FP64
//   jacobian       — FP32
//   linear solve   — FP32 (cuDSS FP32)
//   voltage update — FP64 상태 기준
// ---------------------------------------------------------------------------
enum class ComputePolicy {
    FP64,
    Mixed,
};


// ---------------------------------------------------------------------------
// NewtonOptions: 생성자에 전달하는 solver 설정.
//
// 사용자는 backend, compute policy, Jacobian 빌드 알고리즘만 선택한다.
// kernel 세부 변형은 core 내부에서 결정한다.
// ---------------------------------------------------------------------------
struct NewtonOptions {
    BackendKind         backend = BackendKind::CPU;
    ComputePolicy       compute = ComputePolicy::FP64;
    JacobianBuilderType jacobian_builder = JacobianBuilderType::EdgeBased;
};


// ---------------------------------------------------------------------------
// NRResultF64: solve() 결과. public I/O는 항상 FP64.
// ---------------------------------------------------------------------------
struct NRResultF64 {
    std::vector<std::complex<double>> V;

    int32_t iterations     = 0;
    double  final_mismatch = 0.0;
    bool    converged      = false;
};

using NRResult = NRResultF64;
