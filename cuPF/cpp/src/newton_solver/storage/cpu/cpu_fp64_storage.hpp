#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/jacobian/jacobian_analysis.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>


// ---------------------------------------------------------------------------
// CpuTriplet / CpuCscMatrix: Eigen-free CSC sparse matrix.
//
// `CpuCscMatrix<T>` replaces `Eigen::SparseMatrix<T, ColMajor, int32_t>`
// for the CPU FP64 backend. The layout is column-major and "always
// compressed":
//
//   indptr  size = cols + 1   (column start offsets)
//   indices size = nnz        (row indices, sorted ascending per column)
//   values  size = nnz        (entry values, aligned with `indices`)
//
// The exposed accessors mirror the subset of the Eigen API the existing CPU
// pipeline uses: `outerIndexPtr() / innerIndexPtr() / valuePtr() / nonZeros()
// / rows() / cols() / isCompressed() / resize() / makeCompressed() /
// setFromTriplets()`. Duplicate (row, col) entries are summed, matching
// `Eigen::SparseMatrix::setFromTriplets()` default behaviour.
// ---------------------------------------------------------------------------
template <typename T>
struct CpuTriplet {
    int32_t row;
    int32_t col;
    T       value;

    CpuTriplet() = default;
    CpuTriplet(int32_t r, int32_t c, const T& v) : row(r), col(c), value(v) {}
};


template <typename T>
struct CpuCscMatrix {
    int32_t              rows_ = 0;
    int32_t              cols_ = 0;
    std::vector<int32_t> indptr;   // size = cols_ + 1
    std::vector<int32_t> indices;  // size = nnz
    std::vector<T>       values;   // size = nnz

    int32_t rows()     const noexcept { return rows_; }
    int32_t cols()     const noexcept { return cols_; }
    int32_t nonZeros() const noexcept { return static_cast<int32_t>(values.size()); }
    bool    isCompressed() const noexcept { return true; }

    const int32_t* outerIndexPtr() const noexcept { return indptr.data(); }
    int32_t*       outerIndexPtr()       noexcept { return indptr.data(); }
    const int32_t* innerIndexPtr() const noexcept { return indices.data(); }
    int32_t*       innerIndexPtr()       noexcept { return indices.data(); }
    const T*       valuePtr()      const noexcept { return values.data(); }
    T*             valuePtr()            noexcept { return values.data(); }

    void resize(int32_t r, int32_t c) {
        rows_ = r;
        cols_ = c;
        indptr.assign(static_cast<std::size_t>(c + 1), 0);
        indices.clear();
        values.clear();
    }

    void makeCompressed() noexcept { /* always compressed */ }

    template <class Iter>
    void setFromTriplets(Iter first, Iter last);
};


template <typename T>
template <class Iter>
void CpuCscMatrix<T>::setFromTriplets(Iter first, Iter last)
{
    indptr.assign(static_cast<std::size_t>(cols_ + 1), 0);

    // Pass 1: count entries per column.
    for (Iter it = first; it != last; ++it) {
        ++indptr[static_cast<std::size_t>(it->col + 1)];
    }
    for (int32_t c = 0; c < cols_; ++c) {
        indptr[static_cast<std::size_t>(c + 1)] +=
            indptr[static_cast<std::size_t>(c)];
    }
    const int32_t total = indptr[static_cast<std::size_t>(cols_)];

    // Pass 2: bucket triplets into per-column slabs.
    std::vector<int32_t> rows_raw(static_cast<std::size_t>(total), 0);
    std::vector<T>       vals_raw(static_cast<std::size_t>(total), T(0));
    std::vector<int32_t> cursor = indptr;
    for (Iter it = first; it != last; ++it) {
        const int32_t pos = cursor[static_cast<std::size_t>(it->col)]++;
        rows_raw[static_cast<std::size_t>(pos)] = it->row;
        vals_raw[static_cast<std::size_t>(pos)] = it->value;
    }

    // Pass 3: sort each column by row and accumulate duplicates.
    indices.clear();
    values.clear();
    indices.reserve(static_cast<std::size_t>(total));
    values.reserve(static_cast<std::size_t>(total));
    std::vector<int32_t> new_indptr(static_cast<std::size_t>(cols_ + 1), 0);
    std::vector<int32_t> perm;

    for (int32_t c = 0; c < cols_; ++c) {
        new_indptr[static_cast<std::size_t>(c)] =
            static_cast<int32_t>(indices.size());
        const int32_t lo = indptr[static_cast<std::size_t>(c)];
        const int32_t hi = indptr[static_cast<std::size_t>(c + 1)];
        const int32_t n  = hi - lo;
        perm.assign(static_cast<std::size_t>(n), 0);
        for (int32_t i = 0; i < n; ++i) {
            perm[static_cast<std::size_t>(i)] = lo + i;
        }
        std::sort(perm.begin(), perm.end(),
                  [&](int32_t a, int32_t b) {
                      return rows_raw[static_cast<std::size_t>(a)] <
                             rows_raw[static_cast<std::size_t>(b)];
                  });

        int32_t i = 0;
        while (i < n) {
            const int32_t pi  = perm[static_cast<std::size_t>(i)];
            const int32_t row = rows_raw[static_cast<std::size_t>(pi)];
            T             acc = vals_raw[static_cast<std::size_t>(pi)];
            int32_t       j   = i + 1;
            while (j < n) {
                const int32_t pj = perm[static_cast<std::size_t>(j)];
                if (rows_raw[static_cast<std::size_t>(pj)] != row) break;
                acc += vals_raw[static_cast<std::size_t>(pj)];
                ++j;
            }
            indices.push_back(row);
            values.push_back(acc);
            i = j;
        }
    }
    new_indptr[static_cast<std::size_t>(cols_)] =
        static_cast<int32_t>(indices.size());
    indptr = std::move(new_indptr);
}


using CpuYbusMatrixF64     = CpuCscMatrix<std::complex<double>>;
using CpuJacobianMatrixF64 = CpuCscMatrix<double>;


// ---------------------------------------------------------------------------
// CpuFp64Storage: CPU FP64 경로의 host-side 버퍼.
//
// 메모리와 레이아웃만 소유한다. KLU solver 상태는 CpuLinearSolveKLU가 갖는다.
// ---------------------------------------------------------------------------
struct CpuFp64Storage {
    void prepare(const InitializeContext& ctx);
    void upload(const SolveContext& ctx);
    void download(NRResult& result) const;

    CpuYbusMatrixF64     Ybus;
    CpuJacobianMatrixF64 J;

    JacobianScatterMap   maps;
    JacobianPattern      J_pattern;

    std::vector<double>               F;
    std::vector<double>               dx;

    std::vector<double>               Va;
    std::vector<double>               Vm;
    std::vector<std::complex<double>> V;

    std::vector<int32_t>              Ybus_indptr;
    std::vector<int32_t>              Ybus_indices;
    std::vector<std::complex<double>> Ybus_data;
    std::vector<std::complex<double>> Ibus;
    bool                              has_cached_Ibus = false;

    std::vector<std::complex<double>> Sbus;

    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;
};
