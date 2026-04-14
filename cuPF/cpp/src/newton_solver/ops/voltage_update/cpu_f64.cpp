// ---------------------------------------------------------------------------
// cpu_f64.cpp (VoltageUpdate)
//
// CPU FP64 전압 갱신 (Newton-Raphson 보정 적용).
//
// ■ 갱신 절차
//   선형 풀이 결과 dx는 F 벡터와 같은 순서로 패킹되어 있다.
//     dx[0, n_pv)          → Δθ_pv  (pv 버스 각도 보정)
//     dx[n_pv, n_pvpq)     → Δθ_pq  (pq 버스 각도 보정)
//     dx[n_pvpq, dimF)     → Δ|V|_pq (pq 버스 전압 크기 보정)
//
//   1단계: V → Va, Vm 분해  (arg, abs)
//   2단계: dx를 Va, Vm에 적용
//            θ_pv[i]  += dx[i]
//            θ_pq[i]  += dx[n_pv + i]
//            |V|_pq[i] += dx[n_pvpq + i]
//   3단계: Va, Vm → V 재구성  (V = |V|·e^{jθ} = Vm·(cosθ + j·sinθ))
//
// ■ 참고
//   pv 버스의 |V|는 제어값이므로 dx에 포함되지 않는다 (Vm은 변하지 않음).
//   slack 버스는 θ, |V| 모두 고정이므로 dx에 포함되지 않는다.
//   V가 변경되었으므로 Ibus 캐시를 무효화한다.
// ---------------------------------------------------------------------------

#include "cpu_f64.hpp"

#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>


CpuVoltageUpdateF64::CpuVoltageUpdateF64(IStorage& storage)
    : storage_(static_cast<CpuFp64Storage&>(storage)) {}


void CpuVoltageUpdateF64::run(IterationContext& ctx)
{
    if ((ctx.n_pv > 0 && ctx.pv == nullptr) || (ctx.n_pq > 0 && ctx.pq == nullptr)) {
        throw std::invalid_argument("CpuVoltageUpdateF64::run: pv/pq pointers must not be null");
    }

    // 1단계: 현재 복소 전압 V를 극좌표 (Va, Vm)으로 분해
    for (int32_t bus = 0; bus < storage_.n_bus; ++bus) {
        storage_.Va[static_cast<std::size_t>(bus)] = std::arg(storage_.V[static_cast<std::size_t>(bus)]);
        storage_.Vm[static_cast<std::size_t>(bus)] = std::abs(storage_.V[static_cast<std::size_t>(bus)]);
    }

    // 2단계: dx 보정 적용
    //   Δθ_pv: dx[0 .. n_pv)
    for (int32_t i = 0; i < ctx.n_pv; ++i) {
        storage_.Va[static_cast<std::size_t>(ctx.pv[i])] += storage_.dx[static_cast<std::size_t>(i)];
    }
    //   Δθ_pq: dx[n_pv .. n_pvpq)
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        storage_.Va[static_cast<std::size_t>(ctx.pq[i])] +=
            storage_.dx[static_cast<std::size_t>(ctx.n_pv + i)];
    }
    //   Δ|V|_pq: dx[n_pvpq .. dimF)
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        storage_.Vm[static_cast<std::size_t>(ctx.pq[i])] +=
            storage_.dx[static_cast<std::size_t>(ctx.n_pv + ctx.n_pq + i)];
    }

    // 3단계: V = Vm · (cos(Va) + j·sin(Va)) 재구성
    for (int32_t bus = 0; bus < storage_.n_bus; ++bus) {
        const double vm = storage_.Vm[static_cast<std::size_t>(bus)];
        const double va = storage_.Va[static_cast<std::size_t>(bus)];
        storage_.V[static_cast<std::size_t>(bus)] =
            std::complex<double>(vm * std::cos(va), vm * std::sin(va));
    }

    // V가 변경되었으므로 다음 반복의 SpMV에서 Ibus를 재계산해야 한다.
    storage_.has_cached_Ibus = false;
}
