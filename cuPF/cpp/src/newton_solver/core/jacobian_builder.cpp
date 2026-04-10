#include "newton_solver/core/jacobian_builder.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <vector>
#include <complex>


JacobianBuilder::JacobianBuilder(JacobianBuilderType type)
    : type_(type) {}


JacobianBuilder::Result JacobianBuilder::analyze(
    const YbusView& ybus,
    const int32_t*  pv, int32_t n_pv,
    const int32_t*  pq, int32_t n_pq)
{
    switch (type_) {
        case JacobianBuilderType::EdgeBased:
            return analyzeEdgeBased(ybus, pv, n_pv, pq, n_pq);

        case JacobianBuilderType::VertexBased:
            return analyzeVertexBased(ybus, pv, n_pv, pq, n_pq);
    }
    return {};
}


// ---------------------------------------------------------------------------
// analyzeEdgeBased
//
// Builds the Jacobian sparsity pattern and mapping tables by iterating over
// every non-zero entry in Ybus. Each (i,j) edge contributes to up to four
// quadrants of the Jacobian depending on which bus sets (pv/pq) i and j belong to.
//
// The resulting JacobianMaps encode, for every Ybus entry k (CSR order), the
// index into J.values where that entry contributes — so the update step can
// fill J values in O(nnz) without any searching.
//
// Internally uses Eigen triplets for deduplication (setFromTriplets handles
// duplicate (row,col) pairs). The final output is a backend-agnostic
// JacobianStructure (CSR), not an Eigen type.
// ---------------------------------------------------------------------------
JacobianBuilder::Result JacobianBuilder::analyzeEdgeBased(
    const YbusView& ybus,
    const int32_t*  pv, int32_t n_pv,
    const int32_t*  pq, int32_t n_pq)
{
    const int32_t n_bus  = ybus.rows;
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim_J  = n_pvpq + n_pq;   // total Jacobian dimension

    // ------------------------------------------------------------------
    // 1. Build pvpq index list: pvpq = [pv[0..n_pv), pq[0..n_pq)]
    // ------------------------------------------------------------------
    JacobianMaps maps;
    maps.pvpq.resize(n_pvpq);
    for (int32_t i = 0; i < n_pv; ++i) maps.pvpq[i]        = pv[i];
    for (int32_t i = 0; i < n_pq; ++i) maps.pvpq[n_pv + i] = pq[i];
    maps.n_pvpq = n_pvpq;
    maps.n_pq   = n_pq;

    // ------------------------------------------------------------------
    // 2. Build global bus → Jacobian row/col index lookup tables.
    //
    //   rmap_pvpq[bus] = row in J11/J12 block  (-1 if bus not in pvpq)
    //   rmap_pq  [bus] = row in J21/J22 block  (-1 if bus not in pq)
    //                    (offset by n_pvpq so J21/J22 rows follow J11/J12)
    //   cmap_pvpq/cmap_pq mirror the above for columns.
    // ------------------------------------------------------------------
    std::vector<int32_t> rmap_pvpq(n_bus, -1);
    std::vector<int32_t> rmap_pq  (n_bus, -1);
    std::vector<int32_t> cmap_pvpq(n_bus, -1);
    std::vector<int32_t> cmap_pq  (n_bus, -1);

    for (int32_t i = 0; i < n_pvpq; ++i) rmap_pvpq[maps.pvpq[i]] = i;
    for (int32_t i = 0; i < n_pq;   ++i) rmap_pq  [pq[i]]        = i + n_pvpq;
    for (int32_t j = 0; j < n_pvpq; ++j) cmap_pvpq[maps.pvpq[j]] = j;
    for (int32_t j = 0; j < n_pq;   ++j) cmap_pq  [pq[j]]        = j + n_pvpq;

    // ------------------------------------------------------------------
    // 3. Collect triplets for Jacobian sparsity pattern.
    //
    //    Off-diagonals: every non-zero Ybus(i,j) with i≠j contributes
    //    to whichever J quadrants are valid (both i,j must be in pvpq/pq).
    //
    //    Diagonals: every bus contributes one entry per valid quadrant.
    //
    //    setFromTriplets sums duplicates — dummy=1.0 is used throughout
    //    since only the sparsity pattern matters here.
    // ------------------------------------------------------------------
    using SpD     = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Triplet = Eigen::Triplet<double>;

    std::vector<Triplet> trips;
    trips.reserve(4 * ybus.nnz + 4 * n_bus);

    constexpr double dummy = 1.0;

    // Off-diagonal Ybus entries
    for (int32_t row = 0; row < n_bus; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            const int32_t Y_i = row;
            const int32_t Y_j = ybus.indices[k];
            if (Y_i == Y_j) continue;

            const int32_t Ji_pvpq = rmap_pvpq[Y_i];
            const int32_t Ji_pq   = rmap_pq  [Y_i];
            const int32_t Jj_pvpq = cmap_pvpq[Y_j];
            const int32_t Jj_pq   = cmap_pq  [Y_j];

            if (Ji_pvpq >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pvpq, Jj_pvpq, dummy);
            if (Ji_pq   >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pq,   Jj_pvpq, dummy);
            if (Ji_pvpq >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pvpq, Jj_pq,   dummy);
            if (Ji_pq   >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pq,   Jj_pq,   dummy);
        }
    }

    // Diagonal entries (one per bus)
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const int32_t Ji_pvpq = rmap_pvpq[bus];
        const int32_t Ji_pq   = rmap_pq  [bus];
        const int32_t Jj_pvpq = cmap_pvpq[bus];
        const int32_t Jj_pq   = cmap_pq  [bus];

        if (Ji_pvpq >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pvpq, Jj_pvpq, dummy);
        if (Ji_pq   >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pq,   Jj_pvpq, dummy);
        if (Ji_pvpq >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pvpq, Jj_pq,   dummy);
        if (Ji_pq   >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pq,   Jj_pq,   dummy);
    }

    // ------------------------------------------------------------------
    // 4. Assemble Eigen CSC from triplets (for deduplication only).
    //
    //    setFromTriplets merges duplicate (row,col) pairs.
    //    makeCompressed produces sorted col/row indices — needed so that
    //    when we convert CSC→CSR below, the resulting col_idx within each
    //    row is sorted (required for binary search in find_coeff_index).
    // ------------------------------------------------------------------
    SpD J_csc(dim_J, dim_J);
    J_csc.setFromTriplets(trips.begin(), trips.end());
    J_csc.makeCompressed();

    // ------------------------------------------------------------------
    // 5. Convert Eigen CSC → JacobianStructure (CSR).
    //
    //    Iterating columns in order 0..dim_J-1 and inserting each entry
    //    into its row produces col_idx sorted within each row (because
    //    entries appear in column order as we go col 0, 1, 2, ...).
    // ------------------------------------------------------------------
    const int32_t  j_nnz  = J_csc.nonZeros();
    const int32_t* csc_cp = J_csc.outerIndexPtr();   // column pointers
    const int32_t* csc_ri = J_csc.innerIndexPtr();   // row indices

    JacobianStructure J_csr;
    J_csr.dim = dim_J;
    J_csr.nnz = j_nnz;

    J_csr.row_ptr.assign(dim_J + 1, 0);
    for (int32_t k = 0; k < j_nnz; ++k) J_csr.row_ptr[csc_ri[k] + 1]++;
    for (int32_t r = 0; r < dim_J; ++r) J_csr.row_ptr[r + 1] += J_csr.row_ptr[r];

    J_csr.col_idx.resize(j_nnz);
    std::vector<int32_t> row_cursor(J_csr.row_ptr.begin(), J_csr.row_ptr.end());

    for (int32_t col = 0; col < dim_J; ++col) {
        for (int32_t k = csc_cp[col]; k < csc_cp[col + 1]; ++k) {
            int32_t row     = csc_ri[k];
            int32_t csr_pos = row_cursor[row]++;
            J_csr.col_idx[csr_pos] = col;
        }
    }

    // ------------------------------------------------------------------
    // 6. Build mapping tables: Ybus entry k (CSR order) → J value index (CSR).
    //
    //    Binary search within each row since col_idx is sorted.
    // ------------------------------------------------------------------
    auto find_coeff_index = [&](int32_t row, int32_t col) -> int32_t {
        const int32_t* begin = J_csr.col_idx.data() + J_csr.row_ptr[row];
        const int32_t* end   = J_csr.col_idx.data() + J_csr.row_ptr[row + 1];
        const int32_t* it    = std::lower_bound(begin, end, col);
        if (it != end && *it == col) return static_cast<int32_t>(it - J_csr.col_idx.data());
        return -1;
    };

    maps.mapJ11.assign(ybus.nnz, -1);
    maps.mapJ12.assign(ybus.nnz, -1);
    maps.mapJ21.assign(ybus.nnz, -1);
    maps.mapJ22.assign(ybus.nnz, -1);

    // Iterate Ybus in CSR order — t matches the CSR indexing of the maps
    int32_t t = 0;
    for (int32_t row = 0; row < n_bus; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k, ++t) {
            const int32_t Y_i = row;
            const int32_t Y_j = ybus.indices[k];

            const int32_t Ji_pvpq = rmap_pvpq[Y_i];
            const int32_t Ji_pq   = rmap_pq  [Y_i];
            const int32_t Jj_pvpq = cmap_pvpq[Y_j];
            const int32_t Jj_pq   = cmap_pq  [Y_j];

            if (Ji_pvpq >= 0 && Jj_pvpq >= 0) maps.mapJ11[t] = find_coeff_index(Ji_pvpq, Jj_pvpq);
            if (Ji_pq   >= 0 && Jj_pvpq >= 0) maps.mapJ21[t] = find_coeff_index(Ji_pq,   Jj_pvpq);
            if (Ji_pvpq >= 0 && Jj_pq   >= 0) maps.mapJ12[t] = find_coeff_index(Ji_pvpq, Jj_pq);
            if (Ji_pq   >= 0 && Jj_pq   >= 0) maps.mapJ22[t] = find_coeff_index(Ji_pq,   Jj_pq);
        }
    }

    // ------------------------------------------------------------------
    // 7. Build diagonal mapping tables: bus → J diagonal index per quadrant.
    // ------------------------------------------------------------------
    maps.diagJ11.assign(n_bus, -1);
    maps.diagJ12.assign(n_bus, -1);
    maps.diagJ21.assign(n_bus, -1);
    maps.diagJ22.assign(n_bus, -1);

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const int32_t Ji_pvpq = rmap_pvpq[bus];
        const int32_t Ji_pq   = rmap_pq  [bus];
        const int32_t Jj_pvpq = cmap_pvpq[bus];
        const int32_t Jj_pq   = cmap_pq  [bus];

        if (Ji_pvpq >= 0 && Jj_pvpq >= 0) maps.diagJ11[bus] = find_coeff_index(Ji_pvpq, Jj_pvpq);
        if (Ji_pq   >= 0 && Jj_pvpq >= 0) maps.diagJ21[bus] = find_coeff_index(Ji_pq,   Jj_pvpq);
        if (Ji_pvpq >= 0 && Jj_pq   >= 0) maps.diagJ12[bus] = find_coeff_index(Ji_pvpq, Jj_pq);
        if (Ji_pq   >= 0 && Jj_pq   >= 0) maps.diagJ22[bus] = find_coeff_index(Ji_pq,   Jj_pq);
    }

    return { std::move(maps), std::move(J_csr) };
}


// ---------------------------------------------------------------------------
// analyzeVertexBased
//
// The vertex-based CUDA fill kernel still consumes the same CSR-indexed
// Jacobian maps as the edge-based path:
//   - mapJ**[k] stays indexed by Ybus CSR entry k
//   - diagJ**[bus] stays indexed by bus
//
// What changes is the ownership model in the update kernel: one warp owns one
// active bus row and fills all Jacobian entries for that row without atomics.
//
// Because the sparsity pattern is unchanged, we can reuse the edge-based
// analysis result and only stamp the builder type for the backend.
// ---------------------------------------------------------------------------
JacobianBuilder::Result JacobianBuilder::analyzeVertexBased(
    const YbusView& ybus,
    const int32_t*  pv, int32_t n_pv,
    const int32_t*  pq, int32_t n_pq)
{
    Result result = analyzeEdgeBased(ybus, pv, n_pv, pq, n_pq);
    result.maps.builder_type = JacobianBuilderType::VertexBased;
    return result;
}
