#include "io/linear_system.hpp"
#include "io/matrix_market_io.hpp"

#include <metis.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>


// ============================================================================
// 자료구조
// ============================================================================

// Symbolic factorization 결과: L과 U의 nonzero 위치만 저장 (값 없음).
// L_cols[i] = i번째 행의 L 부분에 nonzero가 있는 열들 (오름차순).
// U_cols[i] = i번째 행의 U 부분에 nonzero가 있는 열들 (오름차순, 대각 포함).
// L은 단위 하삼각이라 대각(=1)은 명시적으로 저장 안 함.
struct SymbolicLU {
    int n = 0;
    std::vector<std::vector<int>> L_cols; // 엄격한 하삼각 (대각 제외)
    std::vector<std::vector<int>> U_cols; // 상삼각 + 대각
};


// Numerical factorization 결과: 실제 값이 든 CSR 형식의 L, U.
// L_row_ptr/L_col_ind/L_values : L의 CSR 표현 (행 단위 저장)
// U_diag_pos[i] = i번째 행의 U_values 배열에서 대각 원소가 위치한 인덱스.
// 분해 도중 피봇을 빨리 찾기 위해 미리 기록해둠.
struct NumericLU {
    int n = 0;
    std::vector<int> L_row_ptr;
    std::vector<int> L_col_ind;
    std::vector<float> L_values;
    std::vector<int> U_row_ptr;
    std::vector<int> U_col_ind;
    std::vector<float> U_values;
    std::vector<int> U_diag_pos;
};


// ============================================================================
// [Step 2] Symbolic factorization
// ----------------------------------------------------------------------------
// 실제 부동소수점 연산 없이 "LU 분해 후 어디에 nonzero가 생길지"를 예측한다.
// 한 행씩 위에서 아래로 진행하며 fill-in 위치를 누적해서 채워 넣는다.
//
// 핵심 규칙 (i번째 행 처리):
//   - A의 i행 자체에 있는 nonzero 열들이 일단 row_pattern에 들어간다.
//   - i보다 작은 인덱스 k가 row_pattern에 있으면, U[k] 행의 nonzero 패턴이
//     i행에도 전파된다(= fill-in). 그 전파된 위치를 row_pattern에 추가한다.
//   - 이걸 반복하면 i번째 행이 최종적으로 갖게 될 nonzero 패턴이 나온다.
//
// 결과는 값이 아닌 인덱스만이라 정수 연산이고, CPU에서 수행한다.
// ============================================================================
SymbolicLU symbolic_lu(
    const benchio::CsrMatrix<float, int>& A
) {
    SymbolicLU S;
    S.n = A.rows;
    S.L_cols.resize(A.rows);
    S.U_cols.resize(A.rows);

    for (int i = 0; i < A.rows; i++) {
        // 1) A의 i행 원본 패턴을 set에 담는다. set이라 자동 정렬 + 중복 제거.
        std::set<int> row_pattern;
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            row_pattern.insert(A.col_ind[p]);
        }
        row_pattern.insert(i); // 대각 원소도 항상 포함시킴(분해 시 피봇 자리)

        // 2) Fill-in 전파:
        //    row_pattern 안에서 k < i인 원소를 순회하면서,
        //    "i행이 k행에 의해 소거될 때 어떤 새 nonzero가 생기는지"를 누적.
        //    U_cols[k] (k행의 U 부분)의 모든 열 col > k 에 대해
        //    row_pattern에 col을 추가한다.
        //    set이라 같은 col이 또 들어와도 한 번만 유지됨.
        //
        //    주의: row_pattern을 순회하면서 동시에 insert 하는 건 std::set의
        //    경우 이터레이터가 무효화되지 않으므로 안전하다. 단 새로 들어온
        //    원소들 중 *it < i 조건을 만족하는 것들도 순회 대상이 된다.
        for (auto it = row_pattern.begin(); it != row_pattern.end() && *it < i; ++it) {
            const int k = *it;
            for (const int col : S.U_cols[k]) {
                if (col > k) {
                    row_pattern.insert(col);
                }
            }
        }

        // 3) 최종 패턴을 L (대각 미만) / U (대각 이상)로 분배해서 저장.
        for (const int col : row_pattern) {
            if (col < i) {
                S.L_cols[i].push_back(col);
            } else {
                S.U_cols[i].push_back(col);
            }
        }
    }

    return S;
}


// ============================================================================
// [Step 5] Numerical factorization (left-looking)
// ----------------------------------------------------------------------------
// Symbolic 단계에서 결정된 패턴을 따라 실제 값을 채워 넣는다.
//
// "Left-looking" 방식: i행을 처리할 때, 이미 분해가 끝난 위쪽 행들(0..i-1)을
// 돌아보며 i행을 갱신한다. (Right-looking은 반대로 i행을 분해한 직후 아래
// 행들에 즉시 영향을 퍼뜨림.)
//
// 한 행 i 처리 흐름:
//   a) row[]에 A의 i행 원본 값을 먼저 깐다.
//   b) S.L_cols[i]에 적힌 k들에 대해 (k가 이 행이 의존하는 위쪽 행)
//      - L_ik = row[k] / U_kk
//      - row[col] -= L_ik * U_kj  (k행의 U 부분을 빼서 i행을 갱신)
//   c) 갱신이 끝난 row[]에서 col<i 자리는 L, col>=i 자리는 U로 떨궈 저장.
//
// row[]는 i행을 임시로 dense하게 표현한 것 — 희소 행렬을 한 행씩 일시적으로
// dense로 펼쳤다가 다시 압축해 저장하는 패턴이다. unordered_map을 쓰는 이유는
// 미리 알려진 열 인덱스에 빠르게 접근하기 위해서.
//
// 누적은 float가 아닌 double로 함 — 같은 값을 여러 번 더하는 동안의 반올림
// 오차를 줄이기 위한 표준적인 기법. 저장은 다시 float로 캐스팅.
// ============================================================================
NumericLU numeric_left_looking_lu(
    const benchio::CsrMatrix<float, int>& A,
    const SymbolicLU& S
) {
    NumericLU LU;
    LU.n = A.rows;
    LU.L_row_ptr.assign(A.rows + 1, 0);
    LU.U_row_ptr.assign(A.rows + 1, 0);
    LU.U_diag_pos.assign(A.rows, -1);

    for (int i = 0; i < A.rows; i++) {
        // i행을 일시적으로 dense하게 펼친 작업공간.
        // key = 열 인덱스, value = 그 자리의 누적 값.
        std::unordered_map<int, double> row;
        row.reserve(S.L_cols[i].size() + S.U_cols[i].size());

        // a) A의 i행 원본 값을 깐다.
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; p++) {
            row[A.col_ind[p]] += A.values[p];
        }
        // 대각 자리가 A에 없었더라도 map에 키를 만들어둠 (값은 0).
        // 나중에 U 저장 단계에서 대각이 누락되지 않도록 보장.
        row[i] += 0.0;

        // b) 의존하는 위쪽 행들(k)을 차례로 적용해 i행을 갱신.
        //    S.L_cols[i]는 i행의 L 부분 패턴 = i행이 의존하는 k들의 목록.
        for (const int k : S.L_cols[i]) {
            // k행은 이미 처리되었으므로 U_kk가 존재해야 한다.
            const int diag_pos = LU.U_diag_pos[k];
            if (diag_pos < 0) {
                throw std::runtime_error("missing U diagonal during sparse LU");
            }

            // 피봇 = U_kk. 0이면 (피보팅 없으니) 분해 실패.
            const double pivot = LU.U_values[diag_pos];
            if (std::abs(pivot) <= std::numeric_limits<float>::epsilon()) {
                throw std::runtime_error("zero pivot during sparse LU");
            }

            // L_ik 계산 후 즉시 L에 기록.
            // (Symbolic 단계에서 패턴을 오름차순으로 만들었으므로 L_col_ind도
            //  자연스럽게 오름차순으로 쌓인다.)
            const double l_ik = row[k] / pivot;
            LU.L_col_ind.push_back(k);
            LU.L_values.push_back(l_ik);

            // i행에서 k행의 U 부분 × L_ik를 빼서 갱신.
            // U_kj (j > k)만 빼면 됨 — j <= k 부분은 이미 처리됨.
            for (int p = LU.U_row_ptr[k];
                 p < LU.U_row_ptr[k + 1];
                 p++) {
                const int col = LU.U_col_ind[p];
                if (col > k) {
                    row[col] -= l_ik * LU.U_values[p];
                }
            }
        }

        // L의 i행이 다 채워졌으므로 row_ptr 마감.
        LU.L_row_ptr[i + 1] = LU.L_col_ind.size();

        // c) row[] 중 col >= i인 부분을 U로 떨군다.
        //    U_cols[i]는 오름차순이라 col == i가 가장 먼저 등장 → 대각 위치 기록.
        for (const int col : S.U_cols[i]) {
            if (col == i) {
                LU.U_diag_pos[i] = LU.U_col_ind.size();
            }
            LU.U_col_ind.push_back(col);
            LU.U_values.push_back(row[col]);
        }

        LU.U_row_ptr[i + 1] = LU.U_col_ind.size();
    }

    return LU;
}


// ============================================================================
// [Step 6] Triangular solve: Ax = b 를 LU로 푼다.
// ----------------------------------------------------------------------------
//   1) Forward substitution:  Ly = b  (L의 대각이 1이라 나눗셈 불필요)
//   2) Backward substitution: Ux = y  (U의 대각으로 나눗셈)
//
// CSR이 행 단위 저장이라 자연스럽게 행 순회 가능 — for i, then for column in row.
// 누적은 여기서도 double로 수행.
// ============================================================================
void triangular_solve(
    const NumericLU& LU,
    const std::vector<float>& b,
    std::vector<float>& x
) {
    std::vector<float> y(LU.n, 0.0f);

    // 1) Forward: y_i = b_i - sum_{j<i} L_ij * y_j
    //    (L의 대각이 1이라 / L_ii 없음)
    for (int i = 0; i < LU.n; i++) {
        double sum = b[i];
        for (int p = LU.L_row_ptr[i];
             p < LU.L_row_ptr[i + 1];
             p++) {
            const int col = LU.L_col_ind[p];
            sum -= LU.L_values[p] * y[col];
        }
        y[i] = sum;
    }

    // 2) Backward: x_i = (y_i - sum_{j>i} U_ij * x_j) / U_ii
    //    아래에서 위로 진행 (i = n-1 ... 0)
    x.assign(LU.n, 0.0f);
    for (int i = LU.n - 1; i >= 0; i--) {
        double sum = y[i];
        double diag = 0.0;
        for (int p = LU.U_row_ptr[i];
             p < LU.U_row_ptr[i + 1];
             p++) {
            const int col = LU.U_col_ind[p];
            const double value = LU.U_values[p];
            if (col == i) {
                diag = value;          // 대각: 마지막에 나눗셈에 사용
            } else if (col > i) {
                sum -= value * x[col];
            }
            // col < i 인 경우는 U에 존재할 수 없지만(이론상), 들어와도 무시.
        }

        if (std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("zero diagonal during sparse triangular solve");
        }
        x[i] = sum / diag;
    }
}


// ============================================================================
// 전체 파이프라인의 baseline 버전.
// Symbolic → Numerical → Solve를 차례대로 호출한다.
// 진행 상황을 로그로 찍어 fill-in 규모와 메모리 사용량을 가늠할 수 있게 한다.
// ============================================================================
void solve_sparse_v0(
    const benchio::CsrMatrix<float, int>& A,
    const std::vector<float>& b,
    std::vector<float>& x
) {
    printf("A rows: %d, A cols: %d, nnz: %d, b size: %zu\n",
           A.rows,
           A.cols,
           A.nnz,
           b.size()
    );

    // [Step 2] 패턴만 계산.
    const SymbolicLU S = symbolic_lu(A);

    // L, U의 예상 nnz 출력 (원본 nnz와 비교해 fill-in 규모 확인 가능).
    std::size_t L_nnz = 0;
    std::size_t U_nnz = 0;
    for (int i = 0; i < S.n; i++) {
        L_nnz += S.L_cols[i].size();
        U_nnz += S.U_cols[i].size();
    }
    printf("symbolic LU: L nnz=%zu, U nnz=%zu\n", L_nnz, U_nnz);

    // [Step 5] 실제 값 계산.
    const NumericLU LU = numeric_left_looking_lu(A, S);
    printf("numeric LU: L nnz=%zu, U nnz=%zu\n",
           LU.L_values.size(),
           LU.U_values.size());

    // [Step 6] Ly=b → Ux=y.
    triangular_solve(LU, b, x);
}


// ============================================================================
// [Step 1] METIS Nested Dissection으로 fill-reducing 순열 만들기
// ----------------------------------------------------------------------------
// METIS는 행렬을 *대칭 그래프*로 본다 (노드=행/열, 엣지=nonzero).
// 따라서 비대칭 행렬이라도 패턴을 대칭화해서 넣어야 한다:
//   A[i][j]가 nonzero이면 그래프에 i—j 엣지 추가, 그리고 j—i도 추가.
// 그러면 METIS가 그래프를 재귀적으로 분할해서(separator 찾기) fill-in을
// 최소화하는 순열을 반환한다.
//
// METIS 그래프 포맷:
//   xadj   : CSR의 row_ptr 같은 역할 (노드별 인접 리스트 시작 인덱스)
//   adjncy : 인접 노드 인덱스를 죽 늘어놓은 배열
//
// 반환되는 두 배열:
//   perm[old]  = new   ("이 노드는 새 순서에서 몇 번째인가")
//   iperm[new] = old   ("새 순서의 i번째 자리는 원래 누구였나")
// 이 코드는 iperm을 반환 — "new_row → old_row" 매핑이라 reorder에 쓰기 편하다.
// ============================================================================
std::vector<idx_t> make_metis_nd_permutation(
    const benchio::CsrMatrix<float, int>& A
) {
    // 1) CSR을 대칭 인접 리스트로 변환.
    //    원본이 대칭이 아닐 수도 있으니 (i,j)와 (j,i) 모두 양방향으로 추가.
    std::vector<std::vector<idx_t>> adjacency(A.rows);
    for (int row = 0; row < A.rows; row++) {
        for (int p = A.row_ptr[row]; p < A.row_ptr[row + 1]; p++) {
            const int col = A.col_ind[p];
            if (col == row) {
                continue; // 자기 자신은 엣지 아님
            }
            adjacency[row].push_back(col);
            adjacency[col].push_back(row);
        }
    }

    // 2) 각 행의 이웃 목록을 정렬·중복 제거한 뒤 METIS CSR(xadj/adjncy)로 변환.
    std::vector<idx_t> xadj(A.rows + 1, 0);
    std::vector<idx_t> adjncy;
    for (int row = 0; row < A.rows; row++) {
        std::vector<idx_t>& neighbors = adjacency[row];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        xadj[row] = adjncy.size();
        adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
    }
    xadj[A.rows] = adjncy.size();

    // 3) METIS Nested Dissection 호출.
    idx_t n = A.rows;
    std::vector<idx_t> perm(A.rows, 0);
    std::vector<idx_t> iperm(A.rows, 0);

    const int status =
        METIS_NodeND(&n, xadj.data(), adjncy.data(), nullptr, nullptr, perm.data(), iperm.data());
    if (status != METIS_OK) {
        printf("METIS_NodeND failed: status=%d\n", status);
        // TODO: 실패 시 항등 순열로 폴백하거나 예외를 던지는 게 안전.
    }
    return iperm; // new_row → old_row
}


// ============================================================================
// 순열을 실제 CSR 행렬에 적용 — PAQ를 만든다.
// 대칭 순열을 쓰므로 행 순서와 열 순서가 동일하게 바뀜.
//
// new_to_old[new_row] = old_row 매핑을 기반으로:
//   - 새 행 i의 내용 = 원래 행 new_to_old[i]
//   - 그 행 안의 열 번호도 old_to_new로 다시 매핑
//   - 새 열 인덱스 기준으로 정렬해서 CSR로 저장
// ============================================================================
benchio::CsrMatrix<float, int> reorder_csr(
    const benchio::CsrMatrix<float, int>& A,
    const std::vector<idx_t>& new_to_old
) {
    // 역방향 매핑 만들기: old_row → new_row
    std::vector<int> old_to_new(A.rows, 0);
    for (int new_row = 0; new_row < A.rows; new_row++) {
        old_to_new[new_to_old[new_row]] = new_row;
    }

    benchio::CsrMatrix<float, int> R;
    R.rows = A.rows;
    R.cols = A.cols;
    R.nnz = A.nnz;
    R.row_ptr.assign(A.rows + 1, 0);
    R.col_ind.reserve(A.nnz);
    R.values.reserve(A.nnz);

    // 새 행 순서대로 채워 넣는다.
    for (int new_row = 0; new_row < A.rows; new_row++) {
        const int old_row = new_to_old[new_row];

        // 원래 행의 (열, 값) 쌍을 가져오면서 열도 새 인덱스로 변환.
        std::vector<std::pair<int, float>> row_entries;
        for (int p = A.row_ptr[old_row]; p < A.row_ptr[old_row + 1]; p++) {
            row_entries.push_back({old_to_new[A.col_ind[p]], A.values[p]});
        }

        // CSR은 행 안에서 열 인덱스가 정렬돼 있어야 깔끔하므로 정렬.
        std::sort(row_entries.begin(), row_entries.end());
        for (const auto& entry : row_entries) {
            R.col_ind.push_back(entry.first);
            R.values.push_back(entry.second);
        }
        R.row_ptr[new_row + 1] = R.col_ind.size();
    }

    return R;
}


// ============================================================================
// 벡터에 행 순열 적용 (b, x_true 같은 우변/참값을 재배열).
// y[new_i] = x[old_i]  where old_i = new_to_old[new_i]
// ============================================================================
std::vector<float> reorder_vector(
    const std::vector<float>& x,
    const std::vector<idx_t>& new_to_old
) {
    std::vector<float> y(new_to_old.size());
    for (std::size_t new_i = 0; new_i < new_to_old.size(); new_i++) {
        y[new_i] = x[new_to_old[new_i]];
    }
    return y;
}


// ============================================================================
// 엔트리 포인트: 한 행렬을 골라서 전체 파이프라인을 끝까지 돌린다.
// ----------------------------------------------------------------------------
//   1) Matrix Market 파일에서 희소 행렬 A 로드
//   2) 임의의 참 해 x_true 생성, b = A·x_true 로 우변 만들기 (정답 알려진 문제)
//   3) METIS로 fill-reducing 순열 만들기 → A', b', x_true' 재배열
//   4) 재배열된 시스템에 대해 sparse LU + solve 수행
//
// x_true와 비교 가능한 정답이 있으니 residual로 정확도 검증이 가능.
// 단 코드 끝에서 풀린 x를 다시 원래 순서로 되돌리는 부분이 빠져 있어
// 검증을 추가하려면 inverse permutation을 한 번 더 적용해야 한다.
// ============================================================================
int main()
{
    const std::filesystem::path sparse_mtx =
        "exp/20260515/data/sparse/scircuit/scircuit.mtx";

    // SuiteSparse Matrix Collection의 회로 시뮬레이션 행렬 (비대칭).
    const benchio::CsrMatrix<float, int> A =
        benchio::load_matrix_market_csr_fp32(sparse_mtx);

    // 참 해 x_true를 [-1, 1]에서 균등 난수로 생성.
    // seed를 고정해서 매 실행 동일한 입력 (재현성 확보).
    const int x_seed = 20260516;
    std::mt19937 rng(x_seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> x_true(A.cols);
    for (float& value : x_true) {
        value = uniform(rng);
    }

    // b = A · x_true. 즉 "정답을 알고 있는 시스템"을 만들어둔다.
    const std::vector<float> b =
        benchio::make_rhs_from_x_true(A, x_true);

    // Sanity check: x_true가 정말 정답이면 residual ≈ 0이어야 함.
    const double input_residual =
        benchio::relative_residual(A, x_true, b);

    printf("sparse loaded: rows=%d cols=%d nnz=%d residual=%.6e\n",
           A.rows,
           A.cols,
           A.nnz,
           input_residual);

    // [Step 1] Fill-reducing ordering 생성 및 적용.
    const std::vector<idx_t> new_to_old = make_metis_nd_permutation(A);
    const benchio::CsrMatrix<float, int> A_reordered = reorder_csr(A, new_to_old);
    const std::vector<float> b_reordered = reorder_vector(b, new_to_old);
    const std::vector<float> x_true_reordered = reorder_vector(x_true, new_to_old);

    // 재배열 후에도 동일한 시스템이라 residual이 같아야 한다 (검증).
    const double reordered_residual =
        benchio::relative_residual(A_reordered, x_true_reordered, b_reordered);

    printf("metis nd reordered: rows=%d cols=%d nnz=%d residual=%.6e\n",
           A_reordered.rows,
           A_reordered.cols,
           A_reordered.nnz,
           reordered_residual);

    // 실제 분해 + 해 구하기.
    // (현재 x는 재배열된 좌표계의 해. 원래 좌표로 복원하려면 inverse permutation
    //  필요 — perm[old]=new를 사용해 x_original[old] = x[perm[old]] 식으로.)
    std::vector<float> x;
    solve_sparse_v0(A_reordered, b_reordered, x);

    return 0;
}
