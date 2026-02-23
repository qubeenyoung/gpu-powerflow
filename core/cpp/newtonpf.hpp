#include <Eigen/Sparse>
#include <Eigen/Dense>
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

