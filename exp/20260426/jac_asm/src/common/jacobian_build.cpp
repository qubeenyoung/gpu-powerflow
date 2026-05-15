#include "jacobian_build.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace exp20260426::newton_solver {
namespace {

// ----------------------------------------------------------------------------
// Key packing utilities.
//
// (row, col) 좌표를 단일 uint64로 패킹한다. 상위 32비트=row, 하위 32비트=col.
// 이렇게 하면 단순 정수 정렬만으로 row-major 정렬이 자동으로 성립하므로,
// 패턴 키들을 std::sort 한 번으로 CSR 친화적인 순서로 만들 수 있다.
// int32 → uint32 캐스팅을 거치는 이유는 음수 키(논리적으로 발생하면 안 되지만)
// 가 비트 표현에서 양수보다 크게 정렬되는 사고를 막기 위함.
// ----------------------------------------------------------------------------
uint64_t make_key(int32_t row, int32_t col)
{
    return (uint64_t(uint32_t(row)) << 32) | uint32_t(col);
}

int32_t key_row(uint64_t key)
{
    return int32_t(key >> 32);
}

int32_t key_col(uint64_t key)
{
    return int32_t(key & 0xffffffffu);
}

// row나 col이 음수면 "해당 블록 entry가 존재하지 않음"을 의미한다.
// (예: slack 버스에 대응하는 행, PV 버스에 대응하는 V열 등)
// 이 함수가 필터 역할을 하므로 호출부는 분기 없이 4개 블록 키를 그냥 push 할 수 있다.
void add_pattern(std::vector<uint64_t>& keys, int32_t row, int32_t col)
{
    if (row >= 0 && col >= 0) {
        keys.push_back(make_key(row, col));
    }
}

// 완성된 Jacobian CSR 패턴에서 (row, col)에 해당하는 절대 슬롯 인덱스를 반환.
// pattern.col_idx는 각 행 내부에서 오름차순으로 정렬되어 있으므로 이진 탐색이 가능하다.
// 패턴에 그 entry가 존재하지 않으면(또는 row/col 자체가 -1이면) -1을 반환하여,
// 호출부의 build.map[*]에 그대로 저장되도록 한다.
// 반환값이 -1이면 채움(fill) 단계에서 그 슬롯에 쓰지 않는다.
int32_t find_coeff_index(const JacobianPattern& pattern, int32_t row, int32_t col)
{
    if (row < 0 || col < 0) {
        return -1;
    }

    const int32_t begin = pattern.row_ptr[row];
    const int32_t end = pattern.row_ptr[row + 1];
    const auto first = pattern.col_idx.begin() + begin;
    const auto last = pattern.col_idx.begin() + end;
    const auto it = std::lower_bound(first, last, col);
    if (it == last || *it != col) {
        return -1;
    }
    return int32_t(it - pattern.col_idx.begin());
}

}  // namespace

// ============================================================================
// buildJacobian
//
// Newton-Raphson 조류계산용 Jacobian의 *심볼릭(희소) 구조*를 한 번 빌드해 둔다.
// 수치값을 채우는 fill 커널은 별도로 존재하며, 여기서 만든 build.map의 슬롯
// 인덱스를 보고 탐색 없이 O(1)로 직접 대입한다.
//
// 결과 JacobianBuild의 세 구성요소:
//   - index   : pvpq 순서, bus → J 행/열 매핑 (논리 좌표계)
//   - pattern : Jacobian의 CSR sparsity (row_ptr, col_idx, nnz, dim)
//   - map     : Ybus의 각 nonzero / 각 bus → coeff 배열 슬롯 인덱스
//
// Jacobian 블록 구조 (극좌표):
//        |  J11 (∂P/∂θ)   J12 (∂P/∂V) |
//   J =  |                              |
//        |  J21 (∂Q/∂θ)   J22 (∂Q/∂V) |
//   - 행: P 방정식은 PV+PQ 버스(n_pvpq), Q 방정식은 PQ 버스만(n_pq)
//   - 열: θ는 PV+PQ, V는 PQ만
//   ⇒ dim = n_pvpq + n_pq, slack 버스는 모든 블록에서 제외된다.
// ============================================================================
JacobianBuild buildJacobian(const YbusCsr& ybus,
                            const int32_t* pv,
                            int32_t n_pv,
                            const int32_t* pq,
                            int32_t n_pq,
                            std::vector<int32_t>* edge_row)
{
    JacobianBuild build;
    // The reduced system keeps angle variables for PV+PQ buses and voltage
    // magnitude variables for PQ buses only.
    const int32_t n_pvpq = n_pv + n_pq;
    const int32_t dim = n_pvpq + n_pq;

    // ---- (1) 인덱스 메타데이터: 외부에 공개되는 매핑 ----------------------
    // pvpq 배열은 [PV..., PQ...] 순서로 배치된다. 즉 pvpq의 첫 n_pv개는 PV 버스,
    // 그 뒤 n_pq개는 PQ 버스. 이 배치가 J의 행/열 인덱스 의미와 일치한다.
    // bus_to_pvpq[bus]: 그 버스의 θ-블록 행/열 인덱스 (slack은 -1로 남는다)
    // bus_to_pq[bus]:   그 버스의 V-블록 행/열 인덱스 (V는 PQ만, 그 외 -1)
    build.index.n_pvpq = n_pvpq;
    build.index.n_pq = n_pq;
    build.index.dim = dim;
    build.index.pvpq.resize(n_pvpq);
    build.index.bus_to_pvpq.assign(ybus.n_bus, -1);
    build.index.bus_to_pq.assign(ybus.n_bus, -1);
    for (int32_t i = 0; i < n_pv; ++i) {
        const int32_t bus = pv[i];
        build.index.pvpq[i] = bus;
        build.index.bus_to_pvpq[bus] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[i];
        build.index.pvpq[n_pv + i] = bus;
        build.index.bus_to_pvpq[bus] = n_pv + i;
        // V-블록은 θ-블록 뒤에 위치하므로 n_pvpq만큼 오프셋된다.
        build.index.bus_to_pq[bus] = n_pvpq + i;
    }

    // ---- (2) 로컬 라우팅 테이블: bus → 4개 블록의 행/열 인덱스 ----------
    // 같은 정보를 풀어둔 형태이지만, 이후 inner loop에서 bus 한 번 lookup으로
    // 4개 블록의 (row, col)을 즉시 얻기 위한 캐시 친화 테이블.
    // 해당 블록에 등장하지 않는 버스는 -1을 유지하여 add_pattern/find_coeff_index의
    // 분기 없는 필터링에 활용된다.
    std::vector<int32_t> rmap_pvpq(ybus.n_bus, -1);  // 버스 → P-방정식 행
    std::vector<int32_t> rmap_pq(ybus.n_bus, -1);    // 버스 → Q-방정식 행
    std::vector<int32_t> cmap_pvpq(ybus.n_bus, -1);  // 버스 → θ-변수 열
    std::vector<int32_t> cmap_pq(ybus.n_bus, -1);    // 버스 → V-변수 열

    for (int32_t i = 0; i < n_pvpq; ++i) {
        const int32_t bus = build.index.pvpq[i];
        rmap_pvpq[bus] = i;
        cmap_pvpq[bus] = i;
    }
    for (int32_t i = 0; i < n_pq; ++i) {
        const int32_t bus = pq[i];
        rmap_pq[bus] = i + n_pvpq;
        cmap_pq[bus] = i + n_pvpq;
    }

    // ---- (3) Symbolic pass: Ybus 희소성 → Jacobian 패턴 키 수집 ---------
    // 핵심 아이디어: Ybus의 각 nonzero (row, col)은 J11/J12/J21/J22 네 블록
    // 각각에서 최대 한 개씩의 entry를 만들 수 있다. 슬랙/PV로 인해 일부는
    // -1 인덱스가 되어 자동 탈락(add_pattern에서 필터).
    // 상한 4*n_edges + 4*n_bus를 미리 reserve하여 재할당을 피한다.
    std::vector<uint64_t> keys;
    keys.reserve(4 * ybus.n_edges + 4 * ybus.n_bus);

    // Common symbolic pass: convert the Ybus sparsity into the four Jacobian
    // blocks. This pass is required before either fill kernel can run.
    //
    // 오프대각 부분: row != col인 Ybus nonzero에서만 4개 블록 entry를 추가.
    // (대각은 다음 루프에서 따로 처리되며, 여기서 중복 추가하지 않는다.)
    for (int32_t row = 0; row < ybus.n_bus; ++row) {
        for (int32_t k = ybus.row_ptr[row]; k < ybus.row_ptr[row + 1]; ++k) {
            const int32_t col = ybus.col[k];
            if (row == col) {
                continue;
            }

            const int32_t ri_pvpq = rmap_pvpq[row];
            const int32_t ri_pq = rmap_pq[row];
            const int32_t cj_pvpq = cmap_pvpq[col];
            const int32_t cj_pq = cmap_pq[col];

            add_pattern(keys, ri_pvpq, cj_pvpq);  // J11
            add_pattern(keys, ri_pq, cj_pvpq);    // J21
            add_pattern(keys, ri_pvpq, cj_pq);    // J12
            add_pattern(keys, ri_pq, cj_pq);      // J22
        }
    }

    // 대각 부분: 모든 버스에 대해 자기 자신 위치의 4개 블록 entry를 추가.
    // 대각 항은 Ybus 대각의 값뿐 아니라 인접 항들의 합산으로 계산되므로,
    // Ybus에 명시적인 대각 nonzero가 없더라도 Jacobian에는 항상 등장해야 한다.
    // 그래서 위 루프와 분리되어 무조건 추가된다.
    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t ri_pvpq = rmap_pvpq[bus];
        const int32_t ri_pq = rmap_pq[bus];
        const int32_t cj_pvpq = cmap_pvpq[bus];
        const int32_t cj_pq = cmap_pq[bus];

        add_pattern(keys, ri_pvpq, cj_pvpq);
        add_pattern(keys, ri_pq, cj_pvpq);
        add_pattern(keys, ri_pvpq, cj_pq);
        add_pattern(keys, ri_pq, cj_pq);
    }

    // ---- (4) 정렬 + 중복제거 → CSR (row_ptr, col_idx) 구성 --------------
    // uint64 키의 단순 정렬이 곧 row-major 정렬이므로 별도 비교자가 필요 없다.
    // 같은 (row, col)이 여러 경로로 추가될 수 있으므로 unique로 중복 제거.
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    build.pattern.dim = dim;
    build.pattern.nnz = int32_t(keys.size());
    build.pattern.row_ptr.assign(dim + 1, 0);
    build.pattern.col_idx.resize(keys.size());

    // row_ptr 누적합: 1) 행별 카운트를 row_ptr[row+1]에 기록, 2) prefix-sum으로
    //                   누적해 일반적인 CSR row_ptr 형태로 변환.
    for (uint64_t key : keys) {
        ++build.pattern.row_ptr[key_row(key) + 1];
    }
    for (int32_t row = 0; row < dim; ++row) {
        build.pattern.row_ptr[row + 1] += build.pattern.row_ptr[row];
    }
    // 키가 이미 row-major + col 오름차순으로 정렬되어 있으므로,
    // col_idx는 그냥 순서대로 col 부분만 꺼내 채우면 된다.
    for (std::size_t pos = 0; pos < keys.size(); ++pos) {
        build.pattern.col_idx[pos] = key_col(keys[pos]);
    }

    // ---- (5) 값-슬롯 맵: edge/bus → coeff[] 절대 인덱스 ----------------
    // 패턴이 확정된 후에야 find_coeff_index가 의미 있는 값을 돌려줄 수 있으므로
    // Ybus를 한 번 더 순회한다(첫 번째 순회와 작업이 다름: 그땐 패턴 키 수집,
    // 지금은 슬롯 인덱스 lookup).
    // 모든 슬롯 배열은 -1로 초기화되어 "이 edge/bus는 이 블록에 entry 없음"을
    // 표현할 수 있게 한다.
    build.map.offdiagJ11.assign(ybus.n_edges, -1);
    build.map.offdiagJ12.assign(ybus.n_edges, -1);
    build.map.offdiagJ21.assign(ybus.n_edges, -1);
    build.map.offdiagJ22.assign(ybus.n_edges, -1);
    if (edge_row != nullptr) {
        edge_row->assign(ybus.n_edges, 0);
    }

    // Shared value-slot map: each Ybus nonzero k stores the destination slots
    // for off-diagonal J blocks. If requested, row[k] is filled here too, so
    // CSR->COO materialization is fused into this pass.
    //
    // edge는 Ybus 전체 nonzero를 0..n_edges-1로 평탄화한 인덱스(=k와 동일하지만,
    // 행 경계를 가로질러 단조 증가). edge_row를 요청한 경우 같은 루프에서
    // CSR(row_ptr) → COO(row[]) 변환을 끼워서 한다.
    int32_t edge = 0;
    for (int32_t row = 0; row < ybus.n_bus; ++row) {
        for (int32_t k = ybus.row_ptr[row]; k < ybus.row_ptr[row + 1]; ++k, ++edge) {
            const int32_t col = ybus.col[k];
            if (edge_row != nullptr) {
                (*edge_row)[edge] = row;
            }
            const int32_t ri_pvpq = rmap_pvpq[row];
            const int32_t ri_pq = rmap_pq[row];
            const int32_t cj_pvpq = cmap_pvpq[col];
            const int32_t cj_pq = cmap_pq[col];

            // 주의: 대각(row==col)도 여기서 함께 처리된다. 대각 entry는 별도의
            // map.diagJ*에 기록되므로 여기서 굳이 분기로 거를 필요는 없다.
            // 다만 그 슬롯은 fill 단계의 오프대각 커널에서 사용되지 않거나,
            // 같은 위치를 대각/오프대각 양쪽에서 일관되게 가리키게 된다.
            build.map.offdiagJ11[edge] =
                find_coeff_index(build.pattern, ri_pvpq, cj_pvpq);
            build.map.offdiagJ21[edge] =
                find_coeff_index(build.pattern, ri_pq, cj_pvpq);
            build.map.offdiagJ12[edge] =
                find_coeff_index(build.pattern, ri_pvpq, cj_pq);
            build.map.offdiagJ22[edge] =
                find_coeff_index(build.pattern, ri_pq, cj_pq);
        }
    }
    (void)edge;  // 루프 후 사용처 없음 — unused 경고 억제용.

    // ---- (6) 대각 슬롯 맵: 버스 b → 대각 4개 블록의 coeff 인덱스 -------
    // 위 오프대각 맵과 별도 배열로 두는 이유는 fill 커널에서 대각 항의 수식이
    // 오프대각과 다르기 때문 (대각은 자기 자신 + 모든 인접 항의 합산을 포함).
    // 따라서 채움 단계에서 두 커널이 각각 자기 맵만 보고 독립적으로 동작한다.
    build.map.diagJ11.assign(ybus.n_bus, -1);
    build.map.diagJ12.assign(ybus.n_bus, -1);
    build.map.diagJ21.assign(ybus.n_bus, -1);
    build.map.diagJ22.assign(ybus.n_bus, -1);

    for (int32_t bus = 0; bus < ybus.n_bus; ++bus) {
        const int32_t ri_pvpq = rmap_pvpq[bus];
        const int32_t ri_pq = rmap_pq[bus];
        const int32_t cj_pvpq = cmap_pvpq[bus];
        const int32_t cj_pq = cmap_pq[bus];

        build.map.diagJ11[bus] =
            find_coeff_index(build.pattern, ri_pvpq, cj_pvpq);
        build.map.diagJ21[bus] =
            find_coeff_index(build.pattern, ri_pq, cj_pvpq);
        build.map.diagJ12[bus] =
            find_coeff_index(build.pattern, ri_pvpq, cj_pq);
        build.map.diagJ22[bus] =
            find_coeff_index(build.pattern, ri_pq, cj_pq);
    }

    // 반환 직전까지 build는 완전한 symbolic pattern과 fill-time lookup table을 가진다.
    return build;
}

}  // namespace exp20260426::newton_solver
