// ---------------------------------------------------------------------------
// cpu_fp64_storage.cpp — CpuFp64Buffers 구현
// ---------------------------------------------------------------------------

#include "cpu_fp64_storage.hpp"

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


void CpuFp64Buffers::prepare(const InitializeContext& ctx)
{
    require_pointer(ctx.ybus.indptr, "InitializeContext.ybus.indptr", ctx.ybus.rows + 1);
    require_pointer(ctx.ybus.indices, "InitializeContext.ybus.indices", ctx.ybus.nnz);

    n_bus  = ctx.n_bus;
    n_pvpq = ctx.n_pv + ctx.n_pq;
    n_pq   = ctx.n_pq;
    dimF   = n_pvpq + n_pq;

    if (!ctx.maps.pvpq.empty() && ctx.maps.n_pvpq != n_pvpq) {
        throw std::runtime_error("CpuFp64Buffers::prepare: JacobianScatterMap n_pvpq does not match pv/pq dimensions");
    }

    const bool has_static_jacobian =
        ctx.J.dim > 0 || ctx.J.nnz > 0 || !ctx.J.row_ptr.empty() || !ctx.J.col_idx.empty();

    if (has_static_jacobian && ctx.J.dim != dimF) {
        throw std::runtime_error("CpuFp64Buffers::prepare: Jacobian dimension does not match dimF");
    }

    maps      = ctx.maps;
    J_pattern = ctx.J;

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
            using Triplet = Eigen::Triplet<double, int32_t>;
            std::vector<Triplet> trips;
            trips.reserve(static_cast<std::size_t>(ctx.J.nnz));

            for (int32_t row = 0; row < ctx.J.dim; ++row) {
                for (int32_t k = ctx.J.row_ptr[static_cast<std::size_t>(row)];
                     k < ctx.J.row_ptr[static_cast<std::size_t>(row + 1)]; ++k) {
                    trips.emplace_back(row, ctx.J.col_idx[static_cast<std::size_t>(k)], 1.0);
                }
            }

            J.resize(ctx.J.dim, ctx.J.dim);
            J.setFromTriplets(trips.begin(), trips.end());
            J.makeCompressed();
        }

        {
            // JacobianScatterMap(CSR 기반) → CSC 위치로 리맵.
            // CpuJacobianOpF64가 J.valuePtr()[pos]에 직접 scatter하므로 필요.
            const int32_t* csc_col_ptr = J.outerIndexPtr();
            const int32_t* csc_row_idx = J.innerIndexPtr();
            const int32_t  j_nnz       = J.nonZeros();

            std::vector<int32_t> csr_row_ptr(static_cast<std::size_t>(ctx.J.dim + 1), 0);
            for (int32_t k = 0; k < j_nnz; ++k) {
                ++csr_row_ptr[static_cast<std::size_t>(csc_row_idx[k] + 1)];
            }
            for (int32_t row = 0; row < ctx.J.dim; ++row) {
                csr_row_ptr[static_cast<std::size_t>(row + 1)] += csr_row_ptr[static_cast<std::size_t>(row)];
            }

            std::vector<int32_t> csr_to_csc(static_cast<std::size_t>(j_nnz), -1);
            std::vector<int32_t> row_cursor = csr_row_ptr;

            for (int32_t col = 0; col < ctx.J.dim; ++col) {
                for (int32_t k = csc_col_ptr[col]; k < csc_col_ptr[col + 1]; ++k) {
                    const int32_t row     = csc_row_idx[k];
                    const int32_t csr_pos = row_cursor[static_cast<std::size_t>(row)]++;
                    csr_to_csc[static_cast<std::size_t>(csr_pos)] = k;
                }
            }

            auto remap = [&csr_to_csc](std::vector<int32_t>& positions) {
                for (int32_t& pos : positions) {
                    pos = (pos >= 0) ? csr_to_csc[static_cast<std::size_t>(pos)] : -1;
                }
            };

            remap(maps.mapJ11);  remap(maps.mapJ12);
            remap(maps.mapJ21);  remap(maps.mapJ22);
            remap(maps.diagJ11); remap(maps.diagJ12);
            remap(maps.diagJ21); remap(maps.diagJ22);

            std::fill(J.valuePtr(), J.valuePtr() + J.nonZeros(), 0.0);
        }
    } else {
        J.resize(dimF, dimF);
        J.makeCompressed();
    }
}


void CpuFp64Buffers::upload(const SolveContext& ctx)
{
    if (ctx.ybus == nullptr || ctx.sbus == nullptr || ctx.V0 == nullptr) {
        throw std::invalid_argument("CpuFp64Buffers::upload: solve context is incomplete");
    }

    const YbusView& ybus = *ctx.ybus;
    if (ybus.rows != n_bus || ybus.cols != n_bus ||
        ybus.nnz != static_cast<int32_t>(Ybus_data.size())) {
        throw std::runtime_error("CpuFp64Buffers::upload: Ybus dimensions do not match initialize()");
    }

    require_pointer(ybus.indptr,  "SolveContext.ybus->indptr",  ybus.rows + 1);
    require_pointer(ybus.indices, "SolveContext.ybus->indices", ybus.nnz);
    require_pointer(ybus.data,    "SolveContext.ybus->data",    ybus.nnz);
    require_pointer(ctx.sbus,     "SolveContext.sbus",          n_bus);
    require_pointer(ctx.V0,       "SolveContext.V0",            n_bus);

    for (int32_t row = 0; row <= n_bus; ++row) {
        if (ybus.indptr[row] != Ybus_indptr[static_cast<std::size_t>(row)]) {
            throw std::runtime_error("CpuFp64Buffers::upload: Ybus 희소 구조가 initialize() 이후 변경되었습니다.");
        }
    }
    for (int32_t k = 0; k < ybus.nnz; ++k) {
        if (ybus.indices[k] != Ybus_indices[static_cast<std::size_t>(k)]) {
            throw std::runtime_error("CpuFp64Buffers::upload: Ybus 희소 구조가 initialize() 이후 변경되었습니다.");
        }
        Ybus_data[static_cast<std::size_t>(k)] = ybus.data[k];
    }

    {
        using Triplet = Eigen::Triplet<std::complex<double>, int32_t>;
        std::vector<Triplet> trips;
        trips.reserve(static_cast<std::size_t>(ybus.nnz));

        for (int32_t row = 0; row < n_bus; ++row) {
            for (int32_t k = Ybus_indptr[static_cast<std::size_t>(row)];
                 k < Ybus_indptr[static_cast<std::size_t>(row + 1)]; ++k) {
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


void CpuFp64Buffers::download(NRResult& result) const
{
    result.V = V;
}
