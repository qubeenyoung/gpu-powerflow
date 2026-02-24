#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include "nr_data.hpp"

nr_data::NRResult
newtonPF(const nr_data::YbusType& ybus,  // 소영수정: const reference로 변경 (복사 방지)
        const nr_data::VectorXcd& sbus,   // 소영수정: const reference로 변경
        const nr_data::VectorXcd& V0,     // 소영수정: const reference로 변경
        const nr_data::VectorXi32& pv,    // 소영수정: const reference로 변경
        const nr_data::VectorXi32& pq,    // 소영수정: const reference로 변경
        const double tolerance,
        const int32_t max_iter
      );

/**
 * @brief Batch Newton-Raphson 전력조류 계산 (N개 케이스 GPU 동시 처리)
 *
 * 동일한 Ybus/pv/pq를 공유하면서 N개의 서로 다른 Sbus/V0 케이스를 GPU에서
 * 동시에 처리합니다. cuDSS Uniform Batch API를 활용하여 N개의 LU 분해 및
 * Solve를 병렬로 실행합니다.
 *
 * GPU 파이프라인 (배치):
 *   0. 초기화: N개 V0/Sbus GPU 업로드 (1회)
 *   1. Batch Mismatch: N개 cuSPARSE SpMV + Mismatch 커널
 *   2. Batch Jacobian: N개 Jacobian 병렬 계산 (FP32)
 *   3. Batch Permutation: N개 Eigen→CSR 재배열
 *   4. Batch LU 분해: cuDSS Uniform Batch Factorization
 *   5. Batch Solve: cuDSS Uniform Batch Solve
 *   6. Batch UpdateV: N개 FP64 Va/Vm 업데이트
 *
 * @param ybus      Ybus 행렬 (공통)
 * @param sbus_vec  N개 Sbus 벡터
 * @param V0_vec    N개 초기 전압 벡터
 * @param pv        PV 버스 인덱스 (공통)
 * @param pq        PQ 버스 인덱스 (공통)
 * @param tolerance 수렴 허용 오차
 * @param max_iter  최대 반복 횟수
 * @return          N개 NRResult
 */
std::vector<nr_data::NRResult>
newtonPF_batch(
    const nr_data::YbusType& ybus,
    const std::vector<nr_data::VectorXcd>& sbus_vec,
    const std::vector<nr_data::VectorXcd>& V0_vec,
    const nr_data::VectorXi32& pv,
    const nr_data::VectorXi32& pq,
    double tolerance = 1e-8,
    int max_iter = 50
);

void
mismatch(double &normF,
        nr_data::VectorXd &F,
        const nr_data::VectorXcd &V,
        const nr_data::VectorXcd &Ibus,
        const nr_data::VectorXcd &Sbus,
        const nr_data::VectorXi32 &pv,
        const nr_data::VectorXi32 &pq
      );

/**
 * @brief Struct to store the Jacobian matrix and related data for Newton-Raphson iterations.
 * 
 * This struct pre-analyzes the sparsity pattern of the Jacobian matrix (J)
 * and stores index mappings between the non-zero values of Ybus and J.
 * This avoids rebuilding the matrix repeatedly and allows for fast value-only updates.
 *
 * Overall structure of the Jacobian matrix J:
 *
 * c1 (n_pvpq)      c2 (n_pq)
 * <-------------><------------>
 * +-------------+-------------+ ^
 * |             |             | | r1
 * |     J11     |     J12     | | (n_pvpq)
 * |             |             | |
 * +-------------+-------------+ v
 * |             |             | ^
 * |     J21     |     J22     | | r2
 * |             |             | | (n_pq)
 * +-------------+-------------+ v
 */
struct Jacobian {
    /// @brief The final assembled sparse Jacobian matrix of type double.
    nr_data::JacobianType J;
    nr_data::VectorXi32 pvpq, pq;

    // --- Size information for the full Jacobian and its sub-matrices ---
    int32_t R = 0, C = 0;   ///< Total number of rows (R) and columns (C) of the J matrix.
    int32_t r1 = 0, c1 = 0; ///< Number of rows for J11, J12 (r1) and columns for J11, J21 (c1).
    int32_t r2 = 0, c2 = 0; ///< Number of rows for J21, J22 (r2) and columns for J12, J22 (c2).

    /*
      * @brief Maps for linking non-zero elements of Ybus to their corresponding value index in J.
      *
      * These maps store the location in J's value buffer where the k-th non-zero of
      * Ybus contributes. The size of these vectors is equal to Ybus.nonZeros().
      *
      * Ybus.nonZeros() = nz
      * <------------------------------------->
      * +---+---+---+-----+------+
      * | 0 | 1 | 2 | ... | nz-1 |  (k-th non-zero in Ybus)
      * +---+---+---+-----+------+
      * |       |
      * | mapJ11[k] -> maps to an index p in J.valuePtr() for sub-matrix J11
      * v
      * +---+---+---+-----+------+
      * | p | p'| p"| ... | p_n  |  (index in jacobian.J.valuePtr())
      * +---+---+---+-----+------+
      */
    std::vector<int32_t> mapJ11; ///< Map for J11 (pvpq, pvpq). Re(dS/dVa).
    std::vector<int32_t> mapJ21; ///< Map for J21 (pq,   pvpq). Im(dS/dVa).
    std::vector<int32_t> mapJ12; ///< Map for J12 (pvpq, pq).   Re(dS/dVm).
    std::vector<int32_t> mapJ22; ///< Map for J22 (pq,   pq).   Im(dS/dVm).

    /*
    * @brief Maps for linking the diagonal contributions of each bus to their value index in J.
    *
    * These maps store the location in J's value buffer for the diagonal contribution
    * of the i-th bus. The size of these vectors is equal to the total number of buses.
    *
    * Number of buses = nb
    * <------------------------------------->
    * +---+---+---+-----+------+
    * | 0 | 1 | 2 | ... | nb-1 |  (i-th bus index)
    * +---+---+---+-----+------+
    * |         |
    * | diagMapJ11[i] -> maps to an index q in J.valuePtr() for sub-matrix J11
    * v
    * +---+---+---+-----+------+
    * | q | q'| q"| ... | q_n  |  (index in jacobian.J.valuePtr())
    * +---+---+---+-----+------+
    */
    std::vector<int32_t> diagMapJ11;
    std::vector<int32_t> diagMapJ12;
    std::vector<int32_t> diagMapJ21;
    std::vector<int32_t> diagMapJ22;

    // Function Declarations
    void analyze(
                const nr_data::YbusType& Ybus,
                const nr_data::VectorXi32& pv,
                const nr_data::VectorXi32& pq);

    void update(
                const nr_data::YbusType& Ybus,
                const nr_data::VectorXcd& V,
                const nr_data::VectorXcd& Ibus);
};

