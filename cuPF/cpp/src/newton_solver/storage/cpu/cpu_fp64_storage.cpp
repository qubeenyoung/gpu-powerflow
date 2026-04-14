// ---------------------------------------------------------------------------
// cpu_fp64_storage.cpp — CPU FP64 Storage 구현
//
// CPU FP64 경로의 host-side 버퍼와 Eigen/KLU 상태를 관리한다.
//
// prepare() — analyze() 단계에서 한 번만 호출:
//   1. 위상 메타데이터(n_bus, n_pvpq, n_pq, dimF) 설정
//   2. Ybus CSR 구조 복사 (Ybus_indptr/indices)
//   3. 가변 버퍼 (F, dx, Va, Vm, V, Ibus, Sbus) 초기화
//   4. Jacobian CSC 조립 (JacobianBuilder가 생성한 CSR → Eigen triplet → CSC):
//      Eigen은 내부적으로 CSC를 사용하나 JacobianMaps는 CSR 기반 인덱스.
//      → csr_to_csc 리맵 테이블을 생성해 mapJ**/diagJ** 를 CSC 위치로 변환.
//   5. J.values 를 0으로 초기화 (구조는 유지, 값만 0)
//
// upload() — solve() 마다 호출:
//   1. Ybus 희소 구조 변경 여부 검증 (indptr/indices 일치 확인)
//   2. Ybus_data 갱신 → Eigen::SparseMatrix<complex<double>> Ybus 재조립
//      (setFromTriplets: CSR → CSC, Eigen 내부 정렬 포함)
//   3. V0, Sbus 복사; Va, Vm 초기화 (std::arg, std::abs)
//   4. F, dx 초기화; Ibus 캐시 무효화
//
// 핵심 특이사항:
//   - has_cached_Ibus: MismatchOp이 Ibus 계산 후 true → JacobianOp 재사용.
//     VoltageUpdateOp이 V 변경 후 false (upload()도 false로 리셋).
//   - CSR→CSC 리맵: JacobianMaps의 위치가 CSR 기준이므로,
//     CpuJacobianOp가 J.valuePtr()[pos] 를 직접 scatter하려면 CSC 위치가 필요.
// ---------------------------------------------------------------------------

#include "cpu_fp64_storage.hpp"

#include "utils/timer.hpp"

#include <stdexcept>
#include <string>
#include <vector>


namespace {

template <typename T>
void require_pointer(const T* ptr, const char* name, int32_t count)
{
    if (count > 0 && ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
}

}  // namespace


void CpuFp64Storage::prepare(const AnalyzeContext& ctx)
{
    require_pointer(ctx.ybus.indptr, "AnalyzeContext.ybus.indptr", ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "AnalyzeContext.ybus.indices", ctx.ybus.nnz);

    n_bus = ctx.n_bus;
    n_pvpq = ctx.n_pv + ctx.n_pq;
    n_pq = ctx.n_pq;
    dimF = n_pvpq + n_pq;

    if (!ctx.maps.pvpq.empty() && ctx.maps.n_pvpq != n_pvpq) {
        throw std::runtime_error("CpuFp64Storage::prepare: JacobianMaps n_pvpq does not match pv/pq dimensions");
    }

    const bool has_static_jacobian =
        ctx.J.dim > 0 || ctx.J.nnz > 0 || !ctx.J.row_ptr.empty() || !ctx.J.col_idx.empty();

    if (has_static_jacobian && ctx.J.dim != dimF) {
        throw std::runtime_error("CpuFp64Storage::prepare: Jacobian dimension does not match dimF");
    }

    maps = ctx.maps;
    J_pattern = ctx.J;
    has_klu_symbolic = false;

    F.assign(dimF, 0.0);
    dx.assign(dimF, 0.0);

    Va.assign(n_bus, 0.0);
    Vm.assign(n_bus, 1.0);
    V.assign(n_bus, std::complex<double>(0.0, 0.0));

    Ybus_indptr.assign(ctx.ybus.indptr, ctx.ybus.indptr + ctx.ybus.rows + 1);
    Ybus_indices.assign(ctx.ybus.indices, ctx.ybus.indices + ctx.ybus.nnz);
    Ybus_data.assign(ctx.ybus.nnz, std::complex<double>(0.0, 0.0));
    if (ctx.ybus.data != nullptr) {
        for (int32_t k = 0; k < ctx.ybus.nnz; ++k) {
            Ybus_data[static_cast<std::size_t>(k)] = ctx.ybus.data[k];
        }
    }

    Ibus.assign(n_bus, std::complex<double>(0.0, 0.0));
    Sbus.assign(n_bus, std::complex<double>(0.0, 0.0));
    has_cached_Ibus = false;

    if (has_static_jacobian) {
        {
            // JacobianStructure(CSR) → Eigen::SparseMatrix(CSC) 조립.
            // 값은 더미(1.0); 구조(패턴)만 확정하면 된다.
            newton_solver::utils::ScopedTimer timer("CPU.analyze.buildJacobianCSC");

            using Triplet = Eigen::Triplet<double, int32_t>;
            std::vector<Triplet> trips;
            trips.reserve(static_cast<std::size_t>(ctx.J.nnz));

            for (int32_t row = 0; row < ctx.J.dim; ++row) {
                for (int32_t k = ctx.J.row_ptr[static_cast<std::size_t>(row)];
                     k < ctx.J.row_ptr[static_cast<std::size_t>(row + 1)];
                     ++k) {
                    trips.emplace_back(row, ctx.J.col_idx[static_cast<std::size_t>(k)], 1.0);
                }
            }

            J.resize(ctx.J.dim, ctx.J.dim);
            J.setFromTriplets(trips.begin(), trips.end());
            J.makeCompressed();
        }

        {
            // JacobianMaps 를 CSR 위치 → CSC 위치로 리맵.
            //
            // JacobianBuilder는 CSR 기반 인덱스로 mapJ**/diagJ** 를 생성한다.
            // 그런데 CpuJacobianOp는 J.valuePtr()[pos] 에 직접 scatter하므로
            // CSC 위치(pos)로 변환해야 한다.
            //
            // 알고리즘:
            //   (a) CSC(col, row) → CSR(row, col) 역변환으로 CSR rowptr 재구성
            //   (b) CSC 순서로 순회하며 csr_pos ↔ csc_pos 매핑 테이블(csr_to_csc) 생성
            //   (c) mapJ**/diagJ** 의 각 CSR 위치를 csr_to_csc[csr_pos] 로 치환
            newton_solver::utils::ScopedTimer timer("CPU.analyze.remapJacobianMaps");

            const int32_t* csc_col_ptr = J.outerIndexPtr();
            const int32_t* csc_row_idx = J.innerIndexPtr();
            const int32_t j_nnz = J.nonZeros();

            // (a) CSC 행 인덱스에서 CSR row_ptr 재구성 (행 별 원소 수 카운트)
            std::vector<int32_t> csr_row_ptr(static_cast<std::size_t>(ctx.J.dim + 1), 0);
            for (int32_t k = 0; k < j_nnz; ++k) {
                ++csr_row_ptr[static_cast<std::size_t>(csc_row_idx[k] + 1)];
            }
            for (int32_t row = 0; row < ctx.J.dim; ++row) {
                csr_row_ptr[static_cast<std::size_t>(row + 1)] += csr_row_ptr[static_cast<std::size_t>(row)];
            }

            // (b) CSC 순서로 순회 → csr_to_csc[csr_pos] = csc_pos
            std::vector<int32_t> csr_to_csc(static_cast<std::size_t>(j_nnz), -1);
            std::vector<int32_t> row_cursor = csr_row_ptr;

            for (int32_t col = 0; col < ctx.J.dim; ++col) {
                for (int32_t k = csc_col_ptr[col]; k < csc_col_ptr[col + 1]; ++k) {
                    const int32_t row = csc_row_idx[k];
                    const int32_t csr_pos = row_cursor[static_cast<std::size_t>(row)]++;
                    csr_to_csc[static_cast<std::size_t>(csr_pos)] = k;  // csc_pos = k
                }
            }

            auto remap_positions = [&csr_to_csc](std::vector<int32_t>& positions) {
                for (int32_t& pos : positions) {
                    pos = (pos >= 0) ? csr_to_csc[static_cast<std::size_t>(pos)] : -1;
                }
            };

            remap_positions(maps.mapJ11);
            remap_positions(maps.mapJ12);
            remap_positions(maps.mapJ21);
            remap_positions(maps.mapJ22);
            remap_positions(maps.diagJ11);
            remap_positions(maps.diagJ12);
            remap_positions(maps.diagJ21);
            remap_positions(maps.diagJ22);

            std::fill(J.valuePtr(), J.valuePtr() + J.nonZeros(), 0.0);
        }

    } else {
        J.resize(dimF, dimF);
        J.makeCompressed();
    }
}


void CpuFp64Storage::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CpuFp64Storage::upload: solve context is incomplete");
    }

    const YbusViewF64& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus || ybus.nnz != static_cast<int32_t>(Ybus_data.size())) {
        throw std::runtime_error("CpuFp64Storage::upload: Ybus dimensions do not match analyze()");
    }

    require_pointer(ybus.indptr,  "SolveContext.ybus->indptr",  ybus.rows + 1);
    require_pointer(ybus.indices, "SolveContext.ybus->indices", ybus.nnz);
    require_pointer(ybus.data,    "SolveContext.ybus->data",    ybus.nnz);
    require_pointer(ctx.sbus, "SolveContext.sbus", n_bus);
    require_pointer(ctx.V0,   "SolveContext.V0",   n_bus);

    for (int32_t row = 0; row <= n_bus; ++row) {
        if (ybus.indptr[row] != Ybus_indptr[static_cast<std::size_t>(row)]) {
            throw std::runtime_error("CpuFp64Storage::upload: Ybus 희소 구조가 analyze() 이후 변경되었습니다.");
        }
    }
    for (int32_t k = 0; k < ybus.nnz; ++k) {
        if (ybus.indices[k] != Ybus_indices[static_cast<std::size_t>(k)]) {
            throw std::runtime_error("CpuFp64Storage::upload: Ybus 희소 구조가 analyze() 이후 변경되었습니다.");
        }
        Ybus_data[static_cast<std::size_t>(k)] = ybus.data[k];
    }

    {
        newton_solver::utils::ScopedTimer timer("CPU.solve.buildYbusCSC");

        using Triplet = Eigen::Triplet<std::complex<double>, int32_t>;
        std::vector<Triplet> trips;
        trips.reserve(static_cast<std::size_t>(ybus.nnz));

        for (int32_t row = 0; row < n_bus; ++row) {
            for (int32_t k = Ybus_indptr[static_cast<std::size_t>(row)];
                 k < Ybus_indptr[static_cast<std::size_t>(row + 1)];
                 ++k) {
                trips.emplace_back(row,
                                   Ybus_indices[static_cast<std::size_t>(k)],
                                   Ybus_data[static_cast<std::size_t>(k)]);
            }
        }

        Ybus.resize(n_bus, n_bus);
        Ybus.setFromTriplets(trips.begin(), trips.end());
        Ybus.makeCompressed();
    }

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        V[static_cast<std::size_t>(bus)]    = ctx.V0[bus];
        Va[static_cast<std::size_t>(bus)]   = std::arg(V[static_cast<std::size_t>(bus)]);
        Vm[static_cast<std::size_t>(bus)]   = std::abs(V[static_cast<std::size_t>(bus)]);
        Sbus[static_cast<std::size_t>(bus)] = ctx.sbus[bus];
        Ibus[static_cast<std::size_t>(bus)] = std::complex<double>(0.0, 0.0);
    }

    std::fill(F.begin(), F.end(), 0.0);
    std::fill(dx.begin(), dx.end(), 0.0);
    has_cached_Ibus = false;
}


void CpuFp64Storage::download_result(NRResultF64& result) const
{
    result.V = V;
}
