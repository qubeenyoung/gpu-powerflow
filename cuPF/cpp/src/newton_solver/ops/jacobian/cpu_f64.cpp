// ---------------------------------------------------------------------------
// cpu_f64.cpp (Jacobian)
//
// CPU FP64 Jacobian fill. 전력조류 Jacobian의 수치값을 계산해 J에 채운다.
//
// ■ Jacobian 편미분 공식 (극좌표계, polar form)
//
//   복소 전력 주입: S_i = V_i · conj(I_i)  where I_i = Σ_j Y_ij · V_j
//
//   오프 대각 (i ≠ j):
//     term_va = ∂S_i/∂θ_j = -j · V_i · conj(Y_ij · V_j)
//     term_vm = ∂S_i/∂|V_j| = V_i · conj(Y_ij · V̂_j)   (V̂ = V/|V|)
//
//   대각 (i = i):
//     term_va = ∂S_i/∂θ_i = +j · V_i · conj(I_i)
//     term_vm = ∂S_i/∂|V_i| = conj(I_i) · V̂_i
//
//   Jacobian 블록 매핑:
//     J11[i,j] = Re(term_va) = ∂P_i/∂θ_j
//     J21[i,j] = Im(term_va) = ∂Q_i/∂θ_j
//     J12[i,j] = Re(term_vm) = ∂P_i/∂|V_j|
//     J22[i,j] = Im(term_vm) = ∂Q_i/∂|V_j|
//
// ■ 구현 전략
//   analyze()에서 미리 계산한 mapJ**/diagJ** 맵을 사용해 J.values에 직접 scatter.
//   오프 대각: = (write). 대각: += (누산, Ibus에 모든 이웃 기여가 합산되어 있음).
//   Ibus가 이미 계산되어 있으면 (has_cached_Ibus) SpMV를 생략한다.
// ---------------------------------------------------------------------------

#include "cpu_f64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <complex>
#include <stdexcept>


namespace {

using CpuComplexVectorF64 = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
static constexpr std::complex<double> kImaginaryUnit(0.0, 1.0);

}  // namespace


CpuJacobianOpF64::CpuJacobianOpF64(IStorage& storage)
    : storage_(static_cast<CpuFp64Storage&>(storage)) {}


void CpuJacobianOpF64::run(IterationContext& ctx)
{
    (void)ctx;

    if (storage_.n_bus <= 0 || storage_.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuJacobianOpF64::run: storage is not prepared");
    }

    // J.values를 0으로 초기화 (매 반복마다 새로 계산)
    double* J_values = storage_.J.valuePtr();
    std::fill(J_values, J_values + storage_.J.nonZeros(), 0.0);

    Eigen::Map<const CpuComplexVectorF64> V(storage_.V.data(), storage_.n_bus);
    Eigen::Map<CpuComplexVectorF64> Ibus(storage_.Ibus.data(), storage_.n_bus);

    // Ibus = Ybus · V. MismatchOp이 먼저 실행했으면 캐시를 재사용한다.
    if (!storage_.has_cached_Ibus) {
        Ibus = storage_.Ybus * V;
        storage_.has_cached_Ibus = true;
    }

    // V̂ = V / |V| (단위 벡터). |V| < 1e-8이면 수치 안정성을 위해 클램프.
    const Eigen::Matrix<double, Eigen::Dynamic, 1> Vm_clamped =
        V.cwiseAbs().cwiseMax(1e-8);
    const CpuComplexVectorF64 Vnorm =
        V.array() / Vm_clamped.cast<std::complex<double>>().array();

    // -----------------------------------------------------------------------
    // 오프 대각 기여: Ybus (i, j) 원소마다 term_va / term_vm을 계산해 scatter.
    //
    //   term_va = -j · V_i · conj(Y_ij · V_j)
    //   term_vm =      V_i · conj(Y_ij · V̂_j)
    //
    // mapJ** 값이 -1이면 해당 블록에 기여하지 않으므로 write 생략.
    // -----------------------------------------------------------------------
    int32_t t = 0;
    for (int32_t row = 0; row < storage_.n_bus; ++row) {
        for (int32_t k = storage_.Ybus_indptr[static_cast<std::size_t>(row)];
             k < storage_.Ybus_indptr[static_cast<std::size_t>(row + 1)];
             ++k, ++t) {
            const int32_t y_i = row;
            const int32_t y_j = storage_.Ybus_indices[static_cast<std::size_t>(k)];
            const std::complex<double> y = storage_.Ybus_data[static_cast<std::size_t>(k)];

            // term_va = -j · V_i · conj(Y_ij · V_j)
            const std::complex<double> va =
                -kImaginaryUnit * storage_.V[static_cast<std::size_t>(y_i)] *
                std::conj(y * storage_.V[static_cast<std::size_t>(y_j)]);

            // term_vm = V_i · conj(Y_ij · V̂_j)
            const std::complex<double> vm =
                storage_.V[static_cast<std::size_t>(y_i)] *
                std::conj(y * Vnorm[y_j]);

            const int32_t p11 = storage_.maps.mapJ11[static_cast<std::size_t>(t)];
            const int32_t p21 = storage_.maps.mapJ21[static_cast<std::size_t>(t)];
            const int32_t p12 = storage_.maps.mapJ12[static_cast<std::size_t>(t)];
            const int32_t p22 = storage_.maps.mapJ22[static_cast<std::size_t>(t)];

            if (p11 >= 0) J_values[p11] = va.real();  // ∂P_i/∂θ_j
            if (p21 >= 0) J_values[p21] = va.imag();  // ∂Q_i/∂θ_j
            if (p12 >= 0) J_values[p12] = vm.real();  // ∂P_i/∂|V_j|
            if (p22 >= 0) J_values[p22] = vm.imag();  // ∂Q_i/∂|V_j|
        }
    }

    // -----------------------------------------------------------------------
    // 대각 기여: 버스 i마다 Ibus를 이용해 term_va / term_vm을 누산(+=).
    //
    //   term_va = +j · V_i · conj(I_i)
    //   term_vm =      conj(I_i) · V̂_i
    //
    // Ibus[i] = Σ_j Y_ij · V_j 이므로 이웃 기여가 모두 포함되어 있다.
    // 오프 대각 fill에서 이미 = (write)로 채웠으므로 여기서는 += (누산).
    // -----------------------------------------------------------------------
    for (int32_t bus = 0; bus < storage_.n_bus; ++bus) {
        // term_va_diag = +j · V_i · conj(I_i)
        const std::complex<double> va =
            kImaginaryUnit *
            (storage_.V[static_cast<std::size_t>(bus)] *
             std::conj(storage_.Ibus[static_cast<std::size_t>(bus)]));

        // term_vm_diag = conj(I_i) · V̂_i
        const std::complex<double> vm =
            std::conj(storage_.Ibus[static_cast<std::size_t>(bus)]) * Vnorm[bus];

        const int32_t q11 = storage_.maps.diagJ11[static_cast<std::size_t>(bus)];
        const int32_t q21 = storage_.maps.diagJ21[static_cast<std::size_t>(bus)];
        const int32_t q12 = storage_.maps.diagJ12[static_cast<std::size_t>(bus)];
        const int32_t q22 = storage_.maps.diagJ22[static_cast<std::size_t>(bus)];

        if (q11 >= 0) J_values[q11] += va.real();  // ∂P_i/∂θ_i
        if (q21 >= 0) J_values[q21] += va.imag();  // ∂Q_i/∂θ_i
        if (q12 >= 0) J_values[q12] += vm.real();  // ∂P_i/∂|V_i|
        if (q22 >= 0) J_values[q22] += vm.imag();  // ∂Q_i/∂|V_i|
    }
}
