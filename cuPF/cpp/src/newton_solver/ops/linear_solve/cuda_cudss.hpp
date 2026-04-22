#pragma once

#ifdef CUPF_WITH_CUDA

#include "newton_solver/core/newton_solver_types.hpp"

#include <utility>


struct InitializeContext;
struct IterationContext;
struct CudaFp64Buffers;
struct CudaMixedBuffers;


// ---------------------------------------------------------------------------
// CudaLinearSolveCuDSS<T, Buffers>: cuDSS 기반 희소 직접 솔버.
//
// T = double : CUDA FP64 프로파일 (Buffers = CudaFp64Buffers)
// T = float  : CUDA Mixed 프로파일 (Buffers = CudaMixedBuffers)
//
// cuDSS 핸들·디스크립터 등 solver 상태를 소유한다.
// 버퍼는 각 메서드 호출 시 직접 전달받는다.
// ---------------------------------------------------------------------------
template <typename T, typename Buffers>
struct CudaLinearSolveCuDSS {
    explicit CudaLinearSolveCuDSS(CuDSSOptions cudss_options = {});
    ~CudaLinearSolveCuDSS();

    CudaLinearSolveCuDSS(CudaLinearSolveCuDSS&& other) noexcept
        : cudss_options_(other.cudss_options_)
        , state_(std::exchange(other.state_, nullptr))
    {}

    CudaLinearSolveCuDSS(const CudaLinearSolveCuDSS&)            = delete;
    CudaLinearSolveCuDSS& operator=(const CudaLinearSolveCuDSS&) = delete;
    CudaLinearSolveCuDSS& operator=(CudaLinearSolveCuDSS&&)      = delete;

    void initialize(Buffers& buf, const InitializeContext& ctx);
    void prepare_rhs(Buffers& buf, IterationContext& ctx);
    void factorize(Buffers& buf, IterationContext& ctx);
    void solve(Buffers& buf, IterationContext& ctx);

private:
    struct State;

    void ensure_descriptors(Buffers& buf);
    T*   rhs_data(Buffers& buf);

    CuDSSOptions cudss_options_;
    State*       state_ = nullptr;
};


extern template struct CudaLinearSolveCuDSS<double, CudaFp64Buffers>;
extern template struct CudaLinearSolveCuDSS<float,  CudaMixedBuffers>;

#endif  // CUPF_WITH_CUDA
