#pragma once

#include "iterative_probe_common.hpp"

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>

#include <cstdint>

namespace exp_20260413::iterative::probe {

void check_hypre(HYPRE_Int code, const char* call);
void initialize_hypre_runtime();
void clear_hypre_errors();

class HypreIjMatrix {
public:
    explicit HypreIjMatrix(const SparseMatrix& matrix);
    ~HypreIjMatrix();

    HypreIjMatrix(const HypreIjMatrix&) = delete;
    HypreIjMatrix& operator=(const HypreIjMatrix&) = delete;

    HYPRE_ParCSRMatrix parcsr() const { return parcsr_; }

private:
    HYPRE_IJMatrix matrix_ = nullptr;
    HYPRE_ParCSRMatrix parcsr_ = nullptr;
};

class HypreIjVector {
public:
    explicit HypreIjVector(const Vector& values);
    HypreIjVector(int32_t n, double initial_value);
    ~HypreIjVector();

    HypreIjVector(const HypreIjVector&) = delete;
    HypreIjVector& operator=(const HypreIjVector&) = delete;

    HYPRE_ParVector parvector() const { return parvector_; }
    Vector values() const;

private:
    HYPRE_IJVector vector_ = nullptr;
    HYPRE_ParVector parvector_ = nullptr;
};

HYPRE_Solver create_boomeramg_preconditioner();

}  // namespace exp_20260413::iterative::probe
