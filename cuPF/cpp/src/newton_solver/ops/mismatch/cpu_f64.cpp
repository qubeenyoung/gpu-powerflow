// ---------------------------------------------------------------------------
// cpu_f64.cpp (Mismatch)
//
// CPU FP64 전력 미스매치(mismatch) 계산.
//
// ■ 미스매치 공식
//   S_calc_i = V_i · conj(I_i)  where I_i = Σ_j Y_ij · V_j
//   mis_i = S_calc_i - S_spec_i = V_i · conj(Ybus · V)_i - Sbus_i
//
// ■ F 벡터 패킹 순서 (dimF = n_pvpq + n_pq = n_pv + 2*n_pq)
//   [0, n_pv)         → ΔP_pv  (pv 버스 유효전력 미스매치)
//   [n_pv, n_pvpq)    → ΔP_pq  (pq 버스 유효전력 미스매치)
//   [n_pvpq, dimF)    → ΔQ_pq  (pq 버스 무효전력 미스매치)
//
//   pv 버스는 전압 크기가 제어되므로 ΔQ는 포함하지 않는다.
//   slack 버스는 θ·|V| 모두 제어되므로 ΔP·ΔQ 모두 포함하지 않는다.
//
// ■ 수렴 판정
//   normF = max|F_i| (L∞ 노름). tolerance 이하이면 수렴.
// ---------------------------------------------------------------------------

#include "cpu_f64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>


namespace {

using CpuComplexVectorF64 = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

}  // namespace


CpuMismatchOpF64::CpuMismatchOpF64(IStorage& storage)
    : storage_(static_cast<CpuFp64Storage&>(storage)) {}


void CpuMismatchOpF64::run(IterationContext& ctx)
{
    if (storage_.n_bus <= 0 || storage_.dimF <= 0) {
        throw std::runtime_error("CpuMismatchOpF64::run: storage is not prepared");
    }
    if ((ctx.n_pv > 0 && ctx.pv == nullptr) || (ctx.n_pq > 0 && ctx.pq == nullptr)) {
        throw std::invalid_argument("CpuMismatchOpF64::run: pv/pq pointers must not be null");
    }

    Eigen::Map<const CpuComplexVectorF64> V(storage_.V.data(), storage_.n_bus);
    Eigen::Map<const CpuComplexVectorF64> Sbus(storage_.Sbus.data(), storage_.n_bus);
    Eigen::Map<CpuComplexVectorF64> Ibus(storage_.Ibus.data(), storage_.n_bus);

    // Ibus = Ybus · V (Eigen CSC SpMV)
    // Jacobian fill에서 재사용할 수 있도록 캐시 플래그를 세운다.
    Ibus = storage_.Ybus * V;
    storage_.has_cached_Ibus = true;

    // mis_i = V_i · conj(I_i) - Sbus_i = S_calc_i - S_spec_i
    const CpuComplexVectorF64 mis =
        V.array() * Ibus.array().conjugate() - Sbus.array();

    // F 벡터에 ΔP_pv, ΔP_pq, ΔQ_pq 순으로 패킹
    int32_t k = 0;
    for (int32_t i = 0; i < ctx.n_pv; ++i) {
        storage_.F[static_cast<std::size_t>(k++)] = mis[ctx.pv[i]].real();  // ΔP_pv
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        storage_.F[static_cast<std::size_t>(k++)] = mis[ctx.pq[i]].real();  // ΔP_pq
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        storage_.F[static_cast<std::size_t>(k++)] = mis[ctx.pq[i]].imag();  // ΔQ_pq
    }

    // normF = max|F_i| (L∞ 노름)
    ctx.normF = 0.0;
    for (double value : storage_.F) {
        ctx.normF = std::max(ctx.normF, std::abs(value));
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CpuMismatchOpF64::run: mismatch norm is not finite");
    }

    newton_solver::utils::dumpArray("residual",
                                    ctx.iter,
                                    storage_.F.data(),
                                    static_cast<int32_t>(storage_.F.size()));
    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}
