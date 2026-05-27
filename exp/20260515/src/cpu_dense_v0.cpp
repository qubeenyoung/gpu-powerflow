#include "io/linear_system.hpp"
#include "io/matrix_market_io.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>


void solve_dense_lu_v0(
    const benchio::DenseMatrix<float>& A_rd,
    const std::vector<float>& b,
    std::vector<float>& x
) {
    int n = A_rd.rows; // square matrix

    std::vector<float> A(n * n);
    for (int i = 0; i < n * n; i++) {
        A[i] = A_rd.values[i];
    }


    // ===== 1. LU 분해 (in-place, no pivoting) =====
    for (int k = 0; k < n-1; k++) {
        float pivot = A[k * n + k];
        // L 열 채우기
        for (int i = k+1; i < n; i++) {
            A[i * n + k] /= pivot;
        }
        // Trailing submatrix 업데이트
        for (int i = k+1; i < n; i++) {
            for (int j = k+1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }

    // ===== 2. Forward substitution: Ly = b =====
    std::vector<float> y(n);
    for (int i = 0; i < n; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= A[i * n + j] * y[j];  // L_ij
        }
        y[i] = sum;  // L_ii = 1
    }

    // ===== 3. Backward substitution: Ux = y =====
    x.resize(n);
    for (int i = n-1; i >= 0; i--) {
        float sum = y[i];
        for (int j = i+1; j < n; j++) {
            sum -= A[i * n + j] * x[j];  // U_ij
        }
        x[i] = sum / A[i * n + i];  // / U_ii
    }

}


int main()
{
    const int dense_n = 4096;
    const int dense_seed = 20260515;
    const std::filesystem::path sparse_mtx =
        "exp/20260515/data/sparse/scircuit/scircuit.mtx";

    // Dense input: generate A = randn(n,n) / sqrt(n), random x_true, b=A*x_true.
    benchio::DenseMatrix<float> dense_A;
    dense_A.rows = dense_n;
    dense_A.cols = dense_n;
    dense_A.layout = benchio::MatrixLayout::RowMajor;
    dense_A.values.resize(dense_n * dense_n);

    std::mt19937 rng(dense_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float scale = 1.0f / std::sqrt(dense_n);
    for (float& value : dense_A.values) {
        value = normal(rng) * scale;
    }

    std::mt19937 x_rng(dense_seed + 1);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> x_true(dense_n);
    for (float& value : x_true) {
        value = uniform(x_rng);
    }

    const std::vector<float> b =
        benchio::make_rhs_from_x_true(dense_A, x_true);

    std::vector<float> x(dense_n, 0.0f);

    solve_dense_lu_v0(dense_A, b, x);

    const double dense_input_residual = benchio::relative_residual(dense_A, x, b);
    printf("dense generated: n=%d seed=%d residual=%.6e\n",
           dense_n,
           dense_seed,
           dense_input_residual
    );



    return 0;
}
