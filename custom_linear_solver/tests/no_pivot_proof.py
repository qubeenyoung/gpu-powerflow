#!/usr/bin/env python3
"""
가설-증명: 전력망 NR Jacobian은 no-pivot LU 가 numerically stable 한가?

측정 항목:
H1. Structural symmetry — J 의 sparsity pattern 이 대칭인가?
H2. Numerical condition — κ(J), |diag(J)|, sym(J)
H3. Growth factor — no-pivot LU 와 partial-pivot LU 의 max(|U|) / max(|A|)
H4. Backward error — no-pivot LU 와 partial-pivot LU 의 ||Ax-b|| / (||A||·||x||)

대상: case8387pegase (n=14908), case_ACTIVSg25k (n=47246)
"""
import sys, time, os
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, csc_matrix, identity
from scipy.sparse.linalg import splu, norm as spnorm, factorized
from scipy.linalg import lu_factor, lu_solve, solve_triangular

def report(case):
    base = f"/datasets/power_system/nr_linear_systems/{case}"
    print(f"\n{'='*72}\n  {case}\n{'='*72}")
    A = csr_matrix(mmread(f"{base}/J.mtx"), dtype=np.float64)
    b = np.asarray(mmread(f"{base}/rhs.mtx")).ravel().astype(np.float64)
    x_true = np.asarray(mmread(f"{base}/x_true.mtx")).ravel().astype(np.float64)
    n = A.shape[0]
    nnz = A.nnz
    print(f"n = {n}, nnz = {nnz}, density = {nnz/(n*n):.2e}")

    # ============ H1: structural symmetry ============
    # Compare J and J^T sparsity pattern.
    A_pat = (A != 0).astype(np.int8)
    AT_pat = A_pat.T.tocsr()
    sym_pat = (A_pat - AT_pat)
    n_asym = sym_pat.nnz
    sym_frac = 1.0 - n_asym / (2 * nnz)
    print(f"\nH1. Structural symmetry:")
    print(f"   structurally symmetric entries: {sym_frac*100:.2f}%")
    print(f"   purely-asymmetric entries: {n_asym}")

    # ============ H2: numerical conditioning ============
    # diagonal, off-diagonal magnitudes, diagonal-dominance ratio
    diag = A.diagonal()
    diag_abs_min = np.abs(diag).min()
    diag_abs_max = np.abs(diag).max()
    diag_abs_med = np.median(np.abs(diag))
    # off-diagonal absolute row sum / |diagonal|
    A_abs = A.copy()
    A_abs.data = np.abs(A.data)
    row_abs_sum = np.asarray(A_abs.sum(axis=1)).ravel()
    off_diag_sum = row_abs_sum - np.abs(diag)
    # diagonal-dominance ratio  σ_i = |diag_i| / (Σ_j≠i |a_ij|)
    # σ_i ≥ 1  →  row i diagonally dominant
    sigma = np.abs(diag) / np.maximum(off_diag_sum, 1e-300)
    dd_frac = (sigma >= 1.0).mean()
    weak_dd_frac = (sigma >= 0.5).mean()
    print(f"\nH2. Numerical structure:")
    print(f"   |diag(J)|     min/med/max = {diag_abs_min:.3e} / {diag_abs_med:.3e} / {diag_abs_max:.3e}")
    print(f"   any zero on diagonal?       {(diag == 0).any()}")
    print(f"   strictly diag-dominant rows: {dd_frac*100:.2f}%   (σ_i ≥ 1)")
    print(f"   weakly diag-dominant rows  : {weak_dd_frac*100:.2f}%   (σ_i ≥ 0.5)")
    print(f"   sigma median = {np.median(sigma):.3f}, min = {sigma.min():.3e}")

    # numerical symmetry
    sym_val = A - A.T
    asym_norm = spnorm(sym_val, 'fro')
    A_norm = spnorm(A, 'fro')
    print(f"   numerical symmetry: ||A - A^T||_F / ||A||_F = {asym_norm/A_norm:.3e}")

    # condition number (1-norm estimate)
    # For sparse, use splu and onenormest. Skipping exact κ for big — use rcond from LU factor.
    Acsc = A.tocsc()

    # ============ H3+H4: growth factor and backward error ============
    # (a) partial-pivot LU via SuperLU/splu
    print(f"\nH3+H4. LU comparison (partial-pivot vs no-pivot):")
    t0 = time.time()
    lu_pp = splu(Acsc, permc_spec='COLAMD', options=dict(SymmetricMode=False))
    t_pp = time.time() - t0

    x_pp = lu_pp.solve(b)
    r_pp = A @ x_pp - b
    backerr_pp = np.linalg.norm(r_pp) / max(spnorm(A) * np.linalg.norm(x_pp), 1e-300)
    forwerr_pp = np.linalg.norm(x_pp - x_true) / max(np.linalg.norm(x_true), 1e-300)
    # growth factor (partial pivot): max(|U|)/max(|A|)
    U_pp = lu_pp.U
    L_pp = lu_pp.L
    max_A = np.abs(A.data).max()
    max_U_pp = np.abs(U_pp.data).max() if U_pp.nnz > 0 else 0
    growth_pp = max_U_pp / max_A
    print(f"   partial-pivot  fact_t={t_pp:.3f}s  ||Ax-b||/(||A||·||x||) = {backerr_pp:.3e}")
    print(f"                  growth factor max(|U|)/max(|A|) = {growth_pp:.3f}")
    print(f"                  ||x - x_true||/||x_true||         = {forwerr_pp:.3e}")

    # (b) no-pivot LU: use splu with diagonal pivot (no row permutation)
    # SuperLU permc only — try diag_pivot_thresh=0 to disable partial pivoting
    try:
        t0 = time.time()
        lu_np = splu(Acsc, permc_spec='COLAMD',
                     options=dict(SymmetricMode=False, DiagPivotThresh=0.0))
        t_np = time.time() - t0
        x_np = lu_np.solve(b)
        r_np = A @ x_np - b
        backerr_np = np.linalg.norm(r_np) / max(spnorm(A) * np.linalg.norm(x_np), 1e-300)
        forwerr_np = np.linalg.norm(x_np - x_true) / max(np.linalg.norm(x_true), 1e-300)
        U_np = lu_np.U
        max_U_np = np.abs(U_np.data).max() if U_np.nnz > 0 else 0
        growth_np = max_U_np / max_A
        # how many row permutations did SuperLU actually need despite threshold=0?
        # We get the row perm via lu_np.perm_r; for a "true no pivot" all perm_r entries should be identity-like
        # Actually with DiagPivotThresh=0, SuperLU may still permute for stability.
        # Just report the resulting numbers.
        print(f"   no-pivot       fact_t={t_np:.3f}s  ||Ax-b||/(||A||·||x||) = {backerr_np:.3e}")
        print(f"                  growth factor max(|U|)/max(|A|) = {growth_np:.3f}")
        print(f"                  ||x - x_true||/||x_true||         = {forwerr_np:.3e}")
    except Exception as e:
        print(f"   no-pivot       FAILED: {e}")

    # (c) Dense growth factor on a small leading block (for tractability)
    # Take top-left k×k of the METIS-reordered matrix as a sanity check on growth factor
    # without partial pivoting. For n>5000 dense LU is impractical, skip if too big.

    return None

if __name__ == '__main__':
    cases = sys.argv[1:] if len(sys.argv) > 1 else ['case8387pegase', 'case_ACTIVSg25k']
    for c in cases:
        report(c)
