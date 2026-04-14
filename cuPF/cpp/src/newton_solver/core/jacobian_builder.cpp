// ---------------------------------------------------------------------------
// jacobian_builder.cpp
//
// NR 전력조류 Jacobian의 희소 구조를 한 번만 분석해 맵 테이블을 생성한다.
//
// ■ 수학적 배경
//   전력조류 Jacobian은 2×2 블록 구조를 갖는다.
//
//     J = [ J11  J12 ] = [ ∂P/∂θ     ∂P/∂|V| ]
//         [ J21  J22 ]   [ ∂Q/∂θ     ∂Q/∂|V| ]
//
//   행: J11/J12는 pvpq 버스, J21/J22는 pq 버스 (dimF = n_pvpq + n_pq)
//   열: J11/J21는 pvpq 버스, J12/J22는 pq 버스
//
// ■ 분석 목표
//   Ybus의 k번째 비영 원소 (Y_i, Y_j)가 J의 어느 위치(CSR 인덱스)에 기여하는지
//   미리 계산해 mapJ11/12/21/22에 저장한다. NR 반복 중에는 이 맵을 이용해
//   Ybus 값에서 J 값을 직접 scatter하므로, 위상 분기 없이 O(nnz) fill이 된다.
// ---------------------------------------------------------------------------

#include "newton_solver/core/jacobian_builder.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <stdexcept>
#include <vector>


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
    throw std::invalid_argument("JacobianBuilder: unknown builder type");
}


// ---------------------------------------------------------------------------
// analyzeEdgeBased: Ybus 비영 원소 순회로 Jacobian 희소 구조와 산포 맵을 생성.
//
// ■ 행·열 인덱스 맵 (rmap / cmap)
//   Jacobian의 행·열은 버스 번호가 아니라 pvpq/pq 내 순서로 매겨진다.
//   버스 번호 → J 인덱스 변환을 O(1)로 하기 위해 역방향 맵을 미리 만든다.
//
//   rmap_pvpq[bus] = bus가 pvpq의 몇 번째인지 (행, J11/J21에 사용)
//   rmap_pq[bus]   = bus가 pq의 몇 번째인지 + n_pvpq (행, J21/J22에 사용)
//   cmap_pvpq[bus] = bus가 pvpq의 몇 번째인지 (열, J11/J12에 사용)
//   cmap_pq[bus]   = bus가 pq의 몇 번째인지 + n_pvpq (열, J12/J22에 사용)
//
//   -1이면 해당 버스가 그 집합에 속하지 않아 J에 기여하지 않음을 의미한다.
//
// ■ 희소 구조 생성 절차
//   1단계: Ybus 비영 원소마다 최대 4개의 J triplet을 등록한다.
//          오프 대각 원소(i≠j): (i→j 방향 전류) → J11/J12/J21/J22에 기여.
//          대각 원소(i=j):     (자기 어드미턴스) → 같은 방식으로 처리.
//          모든 버스의 대각 기여도 별도 triplet 루프로 등록한다.
//   2단계: Eigen으로 CSC 행렬을 조립해 중복 좌표를 병합하고 정렬한다.
//   3단계: CSC → CSR 변환으로 J_csr을 생성한다.
//   4단계: Ybus 원소 k마다 find_coeff_index()로 J의 CSR 위치를 찾아
//          mapJ11/12/21/22에 기록한다.
// ---------------------------------------------------------------------------
JacobianBuilder::Result JacobianBuilder::analyzeEdgeBased(
    const YbusView& ybus,
    const int32_t*  pv, int32_t n_pv,
    const int32_t*  pq, int32_t n_pq)
{
    const int32_t n_bus  = ybus.rows;
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim_J  = n_pvpq + n_pq;  // = n_pv + 2*n_pq

    // pvpq = pv ∥ pq (연접)
    JacobianMaps maps;
    maps.builder_type = JacobianBuilderType::EdgeBased;
    maps.pvpq.resize(n_pvpq);
    for (int32_t i = 0; i < n_pv; ++i) maps.pvpq[i]        = pv[i];
    for (int32_t i = 0; i < n_pq; ++i) maps.pvpq[n_pv + i] = pq[i];
    maps.n_pvpq = n_pvpq;
    maps.n_pq   = n_pq;

    // -----------------------------------------------------------------------
    // 역방향 인덱스 맵: 버스 번호 → J 행·열 인덱스
    //   rmap_pvpq[bus] = J에서 해당 버스의 행 인덱스 (J11/J21 행)
    //   rmap_pq[bus]   = J에서 해당 버스의 행 인덱스 (J21/J22 행, n_pvpq 오프셋)
    //   cmap_*         = 열 버전 (대칭 구조이므로 현재는 rmap_*와 동일)
    // -----------------------------------------------------------------------
    std::vector<int32_t> rmap_pvpq(n_bus, -1);
    std::vector<int32_t> rmap_pq  (n_bus, -1);
    std::vector<int32_t> cmap_pvpq(n_bus, -1);
    std::vector<int32_t> cmap_pq  (n_bus, -1);

    for (int32_t i = 0; i < n_pvpq; ++i) rmap_pvpq[maps.pvpq[i]] = i;
    for (int32_t i = 0; i < n_pq;   ++i) rmap_pq  [pq[i]]        = i + n_pvpq;
    for (int32_t i = 0; i < n_pvpq; ++i) cmap_pvpq[maps.pvpq[i]] = i;
    for (int32_t i = 0; i < n_pq;   ++i) cmap_pq  [pq[i]]        = i + n_pvpq;

    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Triplet      = Eigen::Triplet<double>;

    // -----------------------------------------------------------------------
    // 1단계: J 희소 패턴 수집 (triplet 방식)
    //
    // Ybus 원소 (Y_i, Y_j)는 네 개의 J 블록에 기여할 수 있다.
    //   (Ji_pvpq, Jj_pvpq) → J11   (∂P_i/∂θ_j)
    //   (Ji_pq,   Jj_pvpq) → J21   (∂Q_i/∂θ_j)
    //   (Ji_pvpq, Jj_pq)   → J12   (∂P_i/∂|V_j|)
    //   (Ji_pq,   Jj_pq)   → J22   (∂Q_i/∂|V_j|)
    // 각 맵이 -1이면 해당 버스가 그 집합에 없으므로 기여 없음.
    //
    // 오프 대각(i≠j)과 대각(i=j)을 분리하는 이유:
    //   대각 원소는 모든 이웃의 기여를 합산(+=)해야 하므로 scatter 시
    //   atomic add가 필요하다. 오프 대각은 단순 write로 충분하다.
    // -----------------------------------------------------------------------
    std::vector<Triplet> trips;
    trips.reserve(4 * ybus.nnz + 4 * n_bus);

    constexpr double dummy = 1.0;  // 희소 패턴 등록용 더미 값

    // 오프 대각 원소 (i ≠ j)
    for (int32_t row = 0; row < n_bus; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            const int32_t Y_i = row;
            const int32_t Y_j = ybus.indices[k];
            if (Y_i == Y_j) continue;  // 대각은 아래 루프에서 별도 처리

            const int32_t Ji_pvpq = rmap_pvpq[Y_i];
            const int32_t Ji_pq   = rmap_pq[Y_i];
            const int32_t Jj_pvpq = cmap_pvpq[Y_j];
            const int32_t Jj_pq   = cmap_pq[Y_j];

            if (Ji_pvpq >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pvpq, Jj_pvpq, dummy);
            if (Ji_pq   >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pq,   Jj_pvpq, dummy);
            if (Ji_pvpq >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pvpq, Jj_pq,   dummy);
            if (Ji_pq   >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pq,   Jj_pq,   dummy);
        }
    }

    // 대각 원소 (i = i): 각 버스의 자기 자리 패턴 등록
    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const int32_t Ji_pvpq = rmap_pvpq[bus];
        const int32_t Ji_pq   = rmap_pq[bus];
        const int32_t Jj_pvpq = cmap_pvpq[bus];
        const int32_t Jj_pq   = cmap_pq[bus];

        if (Ji_pvpq >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pvpq, Jj_pvpq, dummy);
        if (Ji_pq   >= 0 && Jj_pvpq >= 0) trips.emplace_back(Ji_pq,   Jj_pvpq, dummy);
        if (Ji_pvpq >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pvpq, Jj_pq,   dummy);
        if (Ji_pq   >= 0 && Jj_pq   >= 0) trips.emplace_back(Ji_pq,   Jj_pq,   dummy);
    }

    // -----------------------------------------------------------------------
    // 2단계: Eigen으로 CSC 행렬 조립 (중복 triplet 병합 + 열 내 정렬)
    // -----------------------------------------------------------------------
    SparseMatrix J_csc(dim_J, dim_J);
    J_csc.setFromTriplets(trips.begin(), trips.end());
    J_csc.makeCompressed();

    // -----------------------------------------------------------------------
    // 3단계: CSC → CSR 변환
    //
    // KLU(CPU)와 cuDSS(CUDA)는 CSR을 요구한다.
    // Eigen은 CSC만 제공하므로 행 포인터와 열 인덱스를 직접 재배치한다.
    //
    // 변환 알고리즘:
    //   (a) CSC 원소를 순회하며 각 행의 원소 개수를 row_ptr에 누적한다.
    //   (b) prefix-sum으로 row_ptr을 확정한다.
    //   (c) CSC를 다시 순회하며 col_idx를 row별로 채운다.
    // -----------------------------------------------------------------------
    JacobianStructure J_csr;
    J_csr.dim = dim_J;
    J_csr.nnz = J_csc.nonZeros();
    J_csr.row_ptr.assign(dim_J + 1, 0);

    const int32_t* csc_col_ptr = J_csc.outerIndexPtr();
    const int32_t* csc_row_idx = J_csc.innerIndexPtr();

    // (a) 각 행의 원소 수 집계
    for (int32_t k = 0; k < J_csr.nnz; ++k) J_csr.row_ptr[csc_row_idx[k] + 1]++;
    // (b) prefix-sum
    for (int32_t row = 0; row < dim_J; ++row) J_csr.row_ptr[row + 1] += J_csr.row_ptr[row];

    // (c) 열 인덱스 채우기
    J_csr.col_idx.resize(J_csr.nnz);
    std::vector<int32_t> row_cursor(J_csr.row_ptr.begin(), J_csr.row_ptr.end());

    for (int32_t col = 0; col < dim_J; ++col) {
        for (int32_t k = csc_col_ptr[col]; k < csc_col_ptr[col + 1]; ++k) {
            const int32_t row     = csc_row_idx[k];
            const int32_t csr_pos = row_cursor[row]++;
            J_csr.col_idx[csr_pos] = col;
        }
    }

    // -----------------------------------------------------------------------
    // 4단계: Ybus 원소 → J CSR 위치 맵 생성 (binary search)
    //
    // Ybus의 k번째 원소(CSR 순서)가 J.values의 어느 위치에 scatter되는지를
    // mapJ** 배열에 기록한다. -1은 해당 블록에 기여하지 않음을 의미한다.
    //
    // find_coeff_index: J CSR에서 (row, col)의 절댓값 인덱스를 이진 탐색으로 반환.
    // -----------------------------------------------------------------------
    auto find_coeff_index = [&](int32_t row, int32_t col) -> int32_t {
        const int32_t* begin = J_csr.col_idx.data() + J_csr.row_ptr[row];
        const int32_t* end   = J_csr.col_idx.data() + J_csr.row_ptr[row + 1];
        const int32_t* it    = std::lower_bound(begin, end, col);
        if (it != end && *it == col)
            return static_cast<int32_t>(it - J_csr.col_idx.data());
        return -1;
    };

    // 오프 대각 맵: Ybus 원소 순서(CSR)에 따라 J 위치를 기록
    maps.mapJ11.assign(ybus.nnz, -1);
    maps.mapJ12.assign(ybus.nnz, -1);
    maps.mapJ21.assign(ybus.nnz, -1);
    maps.mapJ22.assign(ybus.nnz, -1);

    int32_t t = 0;
    for (int32_t row = 0; row < n_bus; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k, ++t) {
            const int32_t Y_i = row;
            const int32_t Y_j = ybus.indices[k];

            const int32_t Ji_pvpq = rmap_pvpq[Y_i];
            const int32_t Ji_pq   = rmap_pq[Y_i];
            const int32_t Jj_pvpq = cmap_pvpq[Y_j];
            const int32_t Jj_pq   = cmap_pq[Y_j];

            if (Ji_pvpq >= 0 && Jj_pvpq >= 0) maps.mapJ11[t] = find_coeff_index(Ji_pvpq, Jj_pvpq);
            if (Ji_pq   >= 0 && Jj_pvpq >= 0) maps.mapJ21[t] = find_coeff_index(Ji_pq,   Jj_pvpq);
            if (Ji_pvpq >= 0 && Jj_pq   >= 0) maps.mapJ12[t] = find_coeff_index(Ji_pvpq, Jj_pq);
            if (Ji_pq   >= 0 && Jj_pq   >= 0) maps.mapJ22[t] = find_coeff_index(Ji_pq,   Jj_pq);
        }
    }

    // 대각 맵: 버스 번호 → J 대각 위치 (NR fill 시 += 누산에 사용)
    maps.diagJ11.assign(n_bus, -1);
    maps.diagJ12.assign(n_bus, -1);
    maps.diagJ21.assign(n_bus, -1);
    maps.diagJ22.assign(n_bus, -1);

    for (int32_t bus = 0; bus < n_bus; ++bus) {
        const int32_t Ji_pvpq = rmap_pvpq[bus];
        const int32_t Ji_pq   = rmap_pq[bus];
        const int32_t Jj_pvpq = cmap_pvpq[bus];
        const int32_t Jj_pq   = cmap_pq[bus];

        if (Ji_pvpq >= 0 && Jj_pvpq >= 0) maps.diagJ11[bus] = find_coeff_index(Ji_pvpq, Jj_pvpq);
        if (Ji_pq   >= 0 && Jj_pvpq >= 0) maps.diagJ21[bus] = find_coeff_index(Ji_pq,   Jj_pvpq);
        if (Ji_pvpq >= 0 && Jj_pq   >= 0) maps.diagJ12[bus] = find_coeff_index(Ji_pvpq, Jj_pq);
        if (Ji_pq   >= 0 && Jj_pq   >= 0) maps.diagJ22[bus] = find_coeff_index(Ji_pq,   Jj_pq);
    }

    return {std::move(maps), std::move(J_csr)};
}


// ---------------------------------------------------------------------------
// analyzeVertexBased: 희소 구조 분석은 EdgeBased와 동일하며,
// builder_type만 VertexBased로 바꾸어 CUDA 커널 선택에 활용한다.
//
// EdgeBased와 VertexBased의 차이는 NR fill 단계에 있다.
//   EdgeBased   — 스레드가 Ybus 원소(엣지) 하나를 담당. atomic add 필요.
//   VertexBased — warp가 버스(정점) 하나를 담당. warp-level reduction으로
//                 대각 누산 후 lane 0이 한 번에 write. atomic 불필요.
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
