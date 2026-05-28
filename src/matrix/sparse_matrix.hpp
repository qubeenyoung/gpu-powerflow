#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sparse_direct::matrix {

using Index = int;
using Value = double;

/*
 * Standard sparse matrix containers used by the benchmark code.
 *
 * The containers own their storage and use zero-based indices.  They do not
 * hide the sparse layout behind accessors because solver adapters usually need
 * direct access to the pointer/index/value arrays.
 */
struct CsrMatrix {
    Index rows = 0;
    Index cols = 0;
    std::vector<Index> row_ptr;
    std::vector<Index> col_idx;
    std::vector<Value> values;

    Index nnz() const
    {
        return static_cast<Index>(values.size());
    }

    bool empty() const
    {
        return rows == 0 || cols == 0;
    }

    void validate() const
    {
        if (rows < 0 || cols < 0) {
            throw std::runtime_error("CSR matrix has a negative dimension");
        }
        if (row_ptr.size() != static_cast<std::size_t>(rows + 1)) {
            throw std::runtime_error("CSR row_ptr size does not match row count");
        }
        if (col_idx.size() != values.size()) {
            throw std::runtime_error("CSR col_idx and values sizes differ");
        }
        if (row_ptr.empty() || row_ptr.front() != 0 || row_ptr.back() != nnz()) {
            throw std::runtime_error("CSR row_ptr boundaries are invalid");
        }

        for (Index row = 0; row < rows; ++row) {
            if (row_ptr[row] > row_ptr[row + 1]) {
                throw std::runtime_error("CSR row_ptr is not monotonic");
            }
        }

        for (Index col : col_idx) {
            if (col < 0 || col >= cols) {
                throw std::runtime_error("CSR column index is out of range");
            }
        }
    }
};

struct CscMatrix {
    Index rows = 0;
    Index cols = 0;
    std::vector<Index> col_ptr;
    std::vector<Index> row_idx;
    std::vector<Value> values;

    Index nnz() const
    {
        return static_cast<Index>(values.size());
    }

    bool empty() const
    {
        return rows == 0 || cols == 0;
    }

    void validate() const
    {
        if (rows < 0 || cols < 0) {
            throw std::runtime_error("CSC matrix has a negative dimension");
        }
        if (col_ptr.size() != static_cast<std::size_t>(cols + 1)) {
            throw std::runtime_error("CSC col_ptr size does not match column count");
        }
        if (row_idx.size() != values.size()) {
            throw std::runtime_error("CSC row_idx and values sizes differ");
        }
        if (col_ptr.empty() || col_ptr.front() != 0 || col_ptr.back() != nnz()) {
            throw std::runtime_error("CSC col_ptr boundaries are invalid");
        }

        for (Index col = 0; col < cols; ++col) {
            if (col_ptr[col] > col_ptr[col + 1]) {
                throw std::runtime_error("CSC col_ptr is not monotonic");
            }
        }

        for (Index row : row_idx) {
            if (row < 0 || row >= rows) {
                throw std::runtime_error("CSC row index is out of range");
            }
        }
    }
};

inline CscMatrix to_csc(const CsrMatrix& csr)
{
    csr.validate();

    CscMatrix csc;
    csc.rows = csr.rows;
    csc.cols = csr.cols;
    csc.col_ptr.assign(static_cast<std::size_t>(csr.cols + 1), 0);
    csc.row_idx.resize(csr.values.size());
    csc.values.resize(csr.values.size());

    // Count entries per column, then prefix-sum the counts into CSC pointers.
    for (Index col : csr.col_idx) {
        ++csc.col_ptr[col + 1];
    }
    for (Index col = 0; col < csr.cols; ++col) {
        csc.col_ptr[col + 1] += csc.col_ptr[col];
    }

    // Fill the CSC arrays by walking CSR rows in order.
    std::vector<Index> next_slot = csc.col_ptr;
    for (Index row = 0; row < csr.rows; ++row) {
        for (Index csr_pos = csr.row_ptr[row]; csr_pos < csr.row_ptr[row + 1]; ++csr_pos) {
            const Index col = csr.col_idx[csr_pos];
            const Index csc_pos = next_slot[col]++;

            csc.row_idx[csc_pos] = row;
            csc.values[csc_pos] = csr.values[csr_pos];
        }
    }

    return csc;
}

}  // namespace sparse_direct::matrix
