// ---------------------------------------------------------------------------
// cpu_naive_jacobian_f64.cpp — 레퍼런스 Jacobian 구현 (PyPower/MATPOWER 방식)
//
// 검증 목적의 Jacobian fill 구현. edge-based 방식과는 달리
// PyPower/MATPOWER의 dSbus_dV 알고리즘을 그대로 따른다.
//
// 3단계 알고리즘:
//   1. dS_dVm 계산 (전 버스 × 전 버스 희소 복소 행렬):
//        dS_dVm[i,j] = V[i] · conj(Ybus[i,j] · V̂[j])   (j≠i 오프 대각)
//        dS_dVm[i,i] +=  conj(Ibus[i]) · V̂[i]            (대각 추가)
//      Ibus = Ybus·V 캐시 (has_cached_Ibus 이면 재계산 생략).
//
//   2. dS_dVa 계산 (전 버스 × 전 버스 희소 복소 행렬):
//        dS_dVa = j · diag(V) · conj(Ybus · diag(V) - diag(Ibus))
//        구현: 먼저 tmp[i,j] = -Ybus[i,j]·V[j] 를 구성하고 대각 += Ibus[i],
//              이후 scaled[i,j] = j·V[i]·conj(tmp[i,j]) 로 변환.
//
//   3. J 조립 (pvpq/pq 인덱스로 슬라이싱):
//        J11 = Re(dS_dVa)[pvpq, pvpq]  (∂P/∂θ)
//        J21 = Im(dS_dVa)[pq,   pvpq]  (∂Q/∂θ)
//        J12 = Re(dS_dVm)[pvpq, pq]    (∂P/∂|V|)
//        J22 = Im(dS_dVm)[pq,   pq]    (∂Q/∂|V|)
//
//   이 구현은 매 반복 dS_dVm, dS_dVa 전 행렬을 재구성하므로
//   edge-based 방식보다 느리지만, 결과가 엄밀히 동일함을 검증하는 데 사용한다.
// ---------------------------------------------------------------------------

#include "cpu_naive_jacobian_f64.hpp"

#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <vector>


namespace {

using CpuComplexMatrixF64 = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor, int32_t>;
using CpuComplexVectorF64 = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
using ComplexTriplet = Eigen::Triplet<std::complex<double>, int32_t>;
using RealTriplet = Eigen::Triplet<double, int32_t>;
constexpr std::complex<double> kImaginaryUnit(0.0, 1.0);

}  // namespace


void CpuNaiveJacobianOpF64::run(CpuFp64Buffers& storage_, IterationContext& ctx)
{
    if (storage_.n_bus <= 0 || storage_.dimF <= 0) {
        throw std::runtime_error("CpuNaiveJacobianOpF64::run: storage is not prepared");
    }
    if ((ctx.n_pv > 0 && ctx.pv == nullptr) || (ctx.n_pq > 0 && ctx.pq == nullptr)) {
        throw std::invalid_argument("CpuNaiveJacobianOpF64::run: pv/pq pointers must not be null");
    }

    const int32_t n = storage_.n_bus;
    const int32_t n_pvpq = ctx.n_pv + ctx.n_pq;
    const int32_t n_pq = ctx.n_pq;
    const int32_t dimF = n_pvpq + n_pq;

    std::vector<int32_t> pvpq(static_cast<std::size_t>(n_pvpq));
    if (ctx.n_pv > 0) {
        std::copy(ctx.pv, ctx.pv + ctx.n_pv, pvpq.begin());
    }
    if (ctx.n_pq > 0) {
        std::copy(ctx.pq, ctx.pq + ctx.n_pq, pvpq.begin() + ctx.n_pv);
    }

    std::vector<int32_t> bus_to_pvpq(static_cast<std::size_t>(n), -1);
    std::vector<int32_t> bus_to_pq(static_cast<std::size_t>(n), -1);
    for (int32_t i = 0; i < n_pvpq; ++i) {
        bus_to_pvpq[static_cast<std::size_t>(pvpq[static_cast<std::size_t>(i)])] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        bus_to_pq[static_cast<std::size_t>(ctx.pq[i])] = i;
    }

    Eigen::Map<const CpuComplexVectorF64> V(storage_.V.data(), n);

    // Ibus = Ybus·V  캐시 재사용 (MismatchOp이 이미 계산했으면 생략)
    if (!storage_.has_cached_Ibus) {
        compute_ibus(storage_);
    }

    // V̂ = V / |V|  (단위 벡터, 영전압 방어를 위해 1e-8 clamp)
    const Eigen::Matrix<double, Eigen::Dynamic, 1> Vm_safe =
        V.cwiseAbs().cwiseMax(1e-8);
    const CpuComplexVectorF64 Vnorm =
        V.array() / Vm_safe.cast<std::complex<double>>().array();

    // 단계 1: dS_dVm = diag(V)·conj(Ybus·diag(V̂)) + diag(conj(Ibus)·V̂)
    CpuComplexMatrixF64 dS_dVm(n, n);
    {
        std::vector<ComplexTriplet> trips;
        trips.reserve(static_cast<std::size_t>(storage_.Ybus.nonZeros() + n));

        for (int32_t col = 0; col < n; ++col) {
            const std::complex<double> vn_col = Vnorm[col];
            for (CpuComplexMatrixF64::InnerIterator it(storage_.Ybus, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                trips.emplace_back(
                    row,
                    col,
                    storage_.V[static_cast<std::size_t>(row)] * std::conj(it.value() * vn_col));
            }
        }

        for (int32_t i = 0; i < n; ++i) {
            trips.emplace_back(i, i, std::conj(storage_.Ibus[static_cast<std::size_t>(i)]) * Vnorm[i]);
        }

        dS_dVm.setFromTriplets(trips.begin(), trips.end());
        dS_dVm.makeCompressed();
    }

    // 단계 2: dS_dVa = j·diag(V)·conj(Ybus·diag(V) - diag(Ibus))
    //   tmp[i,j] = -Ybus[i,j]·V[j]   +  Ibus[i] (대각)
    //   dS_dVa[i,j] = j·V[i]·conj(tmp[i,j])
    CpuComplexMatrixF64 dS_dVa(n, n);
    {
        std::vector<ComplexTriplet> trips;
        trips.reserve(static_cast<std::size_t>(storage_.Ybus.nonZeros() + n));

        for (int32_t col = 0; col < n; ++col) {
            const std::complex<double> v_col = storage_.V[static_cast<std::size_t>(col)];
            for (CpuComplexMatrixF64::InnerIterator it(storage_.Ybus, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                trips.emplace_back(row, col, -it.value() * v_col);
            }
        }

        for (int32_t i = 0; i < n; ++i) {
            trips.emplace_back(i, i, storage_.Ibus[static_cast<std::size_t>(i)]);
        }

        CpuComplexMatrixF64 tmp(n, n);
        tmp.setFromTriplets(trips.begin(), trips.end());
        tmp.makeCompressed();

        std::vector<ComplexTriplet> scaled_trips;
        scaled_trips.reserve(static_cast<std::size_t>(tmp.nonZeros()));
        for (int32_t col = 0; col < n; ++col) {
            for (CpuComplexMatrixF64::InnerIterator it(tmp, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                scaled_trips.emplace_back(
                    row,
                    col,
                    kImaginaryUnit * storage_.V[static_cast<std::size_t>(row)] * std::conj(it.value()));
            }
        }

        dS_dVa.setFromTriplets(scaled_trips.begin(), scaled_trips.end());
        dS_dVa.makeCompressed();
    }

    // 단계 3: J 조립 — pvpq/pq 인덱스로 슬라이싱
    //   J[:n_pvpq, :n_pvpq] = Re(dS_dVa)[pvpq, pvpq]   (J11 = ∂P/∂θ)
    //   J[n_pvpq:, :n_pvpq] = Im(dS_dVa)[pq,   pvpq]   (J21 = ∂Q/∂θ)
    //   J[:n_pvpq, n_pvpq:] = Re(dS_dVm)[pvpq, pq]     (J12 = ∂P/∂|V|)
    //   J[n_pvpq:, n_pvpq:] = Im(dS_dVm)[pq,   pq]     (J22 = ∂Q/∂|V|)
    {
        std::vector<RealTriplet> trips;
        trips.reserve(static_cast<std::size_t>(dS_dVa.nonZeros() + dS_dVm.nonZeros()));

        for (int32_t col = 0; col < n; ++col) {
            const int32_t jcol = bus_to_pvpq[static_cast<std::size_t>(col)];
            if (jcol < 0) {
                continue;
            }

            for (CpuComplexMatrixF64::InnerIterator it(dS_dVa, col); it; ++it) {
                const int32_t bus_i = static_cast<int32_t>(it.row());

                const int32_t irow_pvpq = bus_to_pvpq[static_cast<std::size_t>(bus_i)];
                if (irow_pvpq >= 0) {
                    trips.emplace_back(irow_pvpq, jcol, it.value().real());
                }

                const int32_t irow_pq = bus_to_pq[static_cast<std::size_t>(bus_i)];
                if (irow_pq >= 0) {
                    trips.emplace_back(n_pvpq + irow_pq, jcol, it.value().imag());
                }
            }
        }

        for (int32_t col = 0; col < n; ++col) {
            const int32_t jcol_pq = bus_to_pq[static_cast<std::size_t>(col)];
            if (jcol_pq < 0) {
                continue;
            }
            const int32_t jcol = n_pvpq + jcol_pq;

            for (CpuComplexMatrixF64::InnerIterator it(dS_dVm, col); it; ++it) {
                const int32_t bus_i = static_cast<int32_t>(it.row());

                const int32_t irow_pvpq = bus_to_pvpq[static_cast<std::size_t>(bus_i)];
                if (irow_pvpq >= 0) {
                    trips.emplace_back(irow_pvpq, jcol, it.value().real());
                }

                const int32_t irow_pq = bus_to_pq[static_cast<std::size_t>(bus_i)];
                if (irow_pq >= 0) {
                    trips.emplace_back(n_pvpq + irow_pq, jcol, it.value().imag());
                }
            }
        }

        storage_.J.resize(dimF, dimF);
        storage_.J.setFromTriplets(trips.begin(), trips.end());
        storage_.J.makeCompressed();
    }
}
